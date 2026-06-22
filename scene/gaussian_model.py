#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
# import tinycudann as tcnn
from icecream import ic
from scene import sphere_init

MAX_PRIMITIVES = 8_000_000

@torch.jit.script
def inv_opacity(y):
    x = (-(1 - y).clip(min=1e-10).log()).clip(min=0)
    return x

@torch.jit.script
def get_major_axis(scales): 
    # scales = scales.detach()
    # convert scales to max length. Each element corresponds to the length of each segment in xyz

    max_integration_length = scales.max(dim=-1, keepdim=True).values * 2

    return max_integration_length

@torch.jit.script
def get_major_axis_density(opacity, scales):
    scales = scales.detach()

    max_integration_length = get_major_axis(scales)

    densities = inv_opacity(opacity) / max_integration_length

    return densities.reshape(-1)
    

@torch.jit.script
def get_minor_axis_density(opacity, scales):
    scales = scales.detach()
    minor_axis = scales.min(dim=-1, keepdim=True).values * 2
    densities = inv_opacity(opacity).reshape(minor_axis.shape) / minor_axis
    return densities
    
def divide_opacity(opacity, scales):
    density = get_minor_axis_density(opacity, scales).reshape(opacity.shape) / 2
    minor_axis = scales.min(dim=-1).values.reshape(opacity.shape)
    minor_opacity = (1 - (-density * minor_axis).exp()).clip(min=0, max=0.99).reshape(opacity.shape)
    return minor_opacity

def f1(x):
    return torch.expm1(x).log()

def f2(x):
    return x + (1 - x.neg().exp()).log()

def inverse_softplus(x):
    big = x > torch.tensor(torch.finfo(x.dtype).max).log()
    return torch.where(
        big,
        f2(x.masked_fill(~big, 1.)),
        f1(x.masked_fill(big, 1.)),
    )

class GaussianModel:

    def setup_functions(self, max_opacity):        
        self.max_prim_size = 250.0
        self.min_prim_size = 3e-4

        self.max_opacity = max_opacity
        self.max_planned_opacity = max_opacity
        self.opacity_activation = lambda x: self.max_opacity*torch.sigmoid(x)
        self.inverse_opacity_activation = lambda y: inverse_sigmoid((y/self.max_opacity).clip(min=1e-3, max=0.999))
        self.scaling_activation = lambda x: (torch.nn.functional.softplus(x) + self.min_prim_size).clip(max=self.max_prim_size)
        self.scaling_inverse_activation = lambda y: inverse_softplus((y-self.min_prim_size).clip(min=1e-4, max=self.max_prim_size))


        self.density_activation = lambda x: torch.exp(x).clip(max=1000)
        self.inverse_density_activation = lambda y: torch.log(y.clip(min=1e-10))


        self.feature_activation = lambda x:x
        self.inverse_feature_activation = lambda x:x

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, max_opacity=0.99, tmin=0.2):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree          
        assert sh_degree <= 3, "Check slang implementation, does currently support max sh degree 3"
        self._xyz = torch.empty(0)
        #self._features_dc = torch.empty(0) old spherical harmonics
        #self._features_rest = torch.empty(0)
        self._albedo = torch.empty(0) #lpc
        self._sh_normals = torch.empty(0) #lpc    
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(max_opacity)
        self.tmin = tmin

    def capture(self):
        return (
            self.active_sh_degree,
            self.max_sh_degree,            
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,            
            self.tmin
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self.max_sh_degree,            
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,            
            self.tmin
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    def get_scale_and_density_for_rendering(self) -> tuple[torch.Tensor, torch.Tensor]:
        opacity = self.opacity_activation(self._opacity)
        scaling = self.scaling_activation(self._scaling)
        density = get_minor_axis_density(opacity, scaling)
        return (scaling, density)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self.feature_activation(self._features_dc)
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_albedo(self):
        return self._albedo #lpc

    @property
    def get_sh_normals(self):
        return self._sh_normals #lpc

    def get_densify_gradient(self):
        grads = (self.xyz_gradient_accum / self.denom)
        grads[grads.isnan()] = 0.0
        return grads.reshape(-1)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_additional_pts: int, additional_size_multi: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = self.inverse_feature_activation(RGB2SH(fused_color))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001)
        scales = 1*torch.sqrt(dist2)[...,None].repeat(1, 3)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots = torch.nn.functional.normalize(rots)


        # add points using sphere init
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        if num_additional_pts > 0:
            center = torch.mean(fused_point_cloud, dim=0)
            sph_means, sph_scales, sph_quats, sph_densities, sph_feats = \
                sphere_init.sphere_init(center, num_additional_pts, torch.device("cuda"), a=20, radius=2*fused_point_cloud.std(dim=0).max(), scale_multi=additional_size_multi)

            fused_point_cloud = torch.cat([fused_point_cloud, sph_means], dim=0)
            
            sph_colors = torch.ones((sph_means.shape[0], 3), device='cuda') * 0.5 #lpc
            fused_color = torch.cat([fused_color, sph_colors], dim=0) #lpc
            
            
            sph_scales = sph_scales.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = torch.cat([scales, sph_scales], dim=0).clip(min=self.min_prim_size, max=self.max_prim_size)
            rots = torch.cat([rots, sph_quats], dim=0)
            sph_features = self.inverse_feature_activation(RGB2SH(0.5*torch.ones((sph_means.shape[0], 3, 1), device='cuda')))
            features = torch.cat([features, torch.cat([
                sph_features,
                torch.zeros((sph_means.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1), device='cuda')
            ], dim=2)], dim=0)
            dist = torch.linalg.norm(sph_means, dim=-1, keepdim=True)
            scaling = 1-1/(dist+1)
            opacities = torch.cat([opacities, inverse_sigmoid(0.1 * scaling)], dim=0)

        raw_scales = self.scaling_inverse_activation(scales)

        num_points = fused_point_cloud.shape[0] #lpc

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
       
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(raw_scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        num_final_points = self._xyz.shape[0] #lpc
        self._albedo = nn.Parameter(fused_color.requires_grad_(True)) #lpc
        self._sh_normals = nn.Parameter(torch.zeros((num_points, 9), device="cuda", dtype=torch.float32).requires_grad_(True)) #lpc
        
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.per_point_3d_filter_scale = torch.zeros(
            [scales.shape[0], 1], dtype=torch.float32, device='cuda'
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        ic(self.spatial_lr_scale)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            #{'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            #{'params': [self._features_rest], 'lr': training_args.feature_rest_lr, "name": "f_rest"},
            {'params': [self._albedo], 'lr': training_args.feature_lr, "name": "albedo"}, #lpc
            {'params': [self._sh_normals], 'lr': training_args.normal_lr, "name": "sh_normals"}, #lpc
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=[0.9, 0.999])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):  #lpc
            l.append('albedo_{}'.format(i))
        for i in range(9): #lpc
            l.append('sh_normal_{}'.format(i))
        #for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #    l.append('f_dc_{}'.format(i))
        #for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #    l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        albedo = self._albedo.detach().cpu().numpy() #lpc
        sh_normals = self._sh_normals.detach().cpu().numpy() #lpc
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty((xyz.shape[0]), dtype=dtype_full)
        #attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, axis=1) 
        attributes = np.concatenate((xyz, normals, albedo, sh_normals, opacities, scale, rotation), axis=1) #lpc
        for i, (attribute, _) in enumerate(dtype_full):
            elements[attribute] = attributes[:, i]
        # elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, value=0.01):
        # set opacity such that the density accumulated along the major axis produces the target opacity
        target_opacity = torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*value)
        scaling = self.get_scaling
        target_density = get_major_axis_density(target_opacity, scaling)
        # convert target density to the minor axis opacity
        minor_opacity = (1 - (-target_density * scaling.min(dim=-1).values).exp()).clip(min=1e-3, max=0.99)
        opacities_new = self.inverse_opacity_activation(minor_opacity.reshape(target_opacity.shape))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        #features_dc = np.zeros((xyz.shape[0], 3, 1))
        #features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        #features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        #features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        albedo = np.zeros((xyz.shape[0], 3)) #lpc
        albedo[:, 0] = np.asarray(plydata.elements[0]["albedo_0"]) #lpc
        albedo[:, 1] = np.asarray(plydata.elements[0]["albedo_1"]) #lpc
        albedo[:, 2] = np.asarray(plydata.elements[0]["albedo_2"]) #lpc

        sh_normals = np.zeros((xyz.shape[0], 9)) #lpc
        for idx in range(9): 
            sh_normals[:, idx] = np.asarray(plydata.elements[0]["sh_normal_{}".format(idx)]) #lpc

        #extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        #features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        #for idx, attr_name in enumerate(extra_f_names):
        #    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        #features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        #self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        #self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))

        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)) #lpc
        self._sh_normals = nn.Parameter(torch.tensor(sh_normals, dtype=torch.float, device="cuda").requires_grad_(True)) #lpc
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        self.max_opacity = self.max_planned_opacity

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        #self._features_dc = optimizable_tensors["f_dc"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._albedo = optimizable_tensors["albedo"]       #lpc
        self._sh_normals = optimizable_tensors["sh_normals"] #lpc
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_albedo, new_sh_normals, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        #"f_dc": new_features_dc,
        #"f_rest": new_features_rest,
        "albedo": new_albedo, #lpc
        "sh_normals": new_sh_normals, #lpc
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        #self._features_dc = optimizable_tensors["f_dc"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._albedo = optimizable_tensors["albedo"] #lpc
        self._sh_normals = optimizable_tensors["sh_normals"] #lpc  
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        scaling, density = self.get_scale_and_density_for_rendering()
        # Cloning may have added points so pad gradients
        padded_grad = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        major_axis = (torch.max(scaling, dim=-1).values * 2).reshape(density.shape)
        major_opacity = (1 - (-density * major_axis).exp()).clip(min=0).squeeze()

        selected_pts_mask = (padded_grad >= grad_threshold) 
        size_mask = torch.min(scaling, dim=1).values > self.percent_dense*scene_extent
        selected_pts_mask = (selected_pts_mask & size_mask) | (major_opacity > 0.99)

        print(
           f"Split {selected_pts_mask.sum()}/{selected_pts_mask.shape[0]} primitives. Size mask: {size_mask.sum()} Grad mask: {torch.sum(padded_grad >= grad_threshold)} Opacity mask: {torch.sum(major_opacity > 0.99)} Grad mean: {padded_grad.mean()}"
        )

        stds = scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds/3) # full scaling seems to add too much variance in placement
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        div_scaling = self.scaling_inverse_activation((scaling[selected_pts_mask] / (0.8 * N)).clip(min=self.min_prim_size))
        new_scaling = div_scaling.repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        #new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        #new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_sh_normals = self._sh_normals[selected_pts_mask].repeat(N, 1)

        # halve opacity given new scaling
        minor_axis = div_scaling.min(dim=-1, keepdim=True).values * 2
        density = inv_opacity(self.opacity_activation(self._opacity[selected_pts_mask])).reshape(minor_axis.shape) / minor_axis
        minor_opacity = (1 - (-density * minor_axis).exp()).clip(min=0)
        new_opacity = self.inverse_opacity_activation(minor_opacity / 2).repeat(N,1)

        self.densification_postfix(new_xyz, new_albedo, new_sh_normals, new_opacity, new_scaling, new_rotation) #lpc

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = (grads >= grad_threshold).reshape(-1)
        scaling = self.get_scaling
        size_mask = torch.min(scaling, dim=1).values <= self.percent_dense*scene_extent
        selected_pts_mask = selected_pts_mask & size_mask

        new_opacity = self.inverse_opacity_activation(self.opacity_activation(self._opacity[selected_pts_mask]) / 2)
        new_xyz = self._xyz[selected_pts_mask]
        #new_features_dc = self._features_dc[selected_pts_mask]
        #new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_albedo = self._albedo[selected_pts_mask] #lpc
        new_sh_normals = self._sh_normals[selected_pts_mask] #lpc

        self.densification_postfix(new_xyz, new_albedo, new_sh_normals, new_opacity, new_scaling, new_rotation)
        print(
            f"Cloned {selected_pts_mask.sum()}/{selected_pts_mask.shape[0]} primitives. Size mask: {size_mask.sum()}. Max gradient: {grads.max()}"
        )


    def decrease_opacity(self, amount):
        opacities_new = self.inverse_opacity_activation((self.get_opacity - amount))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad
    def densify_and_prune(self, max_grad, extent):
        prune_mask = (self.get_opacity < 0.005).reshape(-1)        
        print(f"Pruned {prune_mask.sum()} primitives. Mean Prune Opacity: {self.get_opacity[prune_mask].mean()}")
        self.prune_points(prune_mask)        

        grads = self.get_densify_gradient()
        if self.get_xyz.shape[0] < MAX_PRIMITIVES:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)            

        torch.cuda.empty_cache()

    def add_densification_stats(self, densification_metric, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.abs(densification_metric.grad[update_filter])
        self.denom[update_filter] += 1
