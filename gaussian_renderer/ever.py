# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math

from scene.gaussian_model import GaussianModel
from ever.splinetracers.fast_ellipsoid_splinetracer import trace_rays
from utils.sh_utils import RGB2SH
MAX_ITERS = 400
from kornia import create_meshgrid
import numpy as np
from scene.dataset_readers import ProjectionType

def get_ray_directions(H, W, focal, center=None, random=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]# + 0.5
    if random:
        grid = grid + torch.rand_like(grid)
    else:
        grid = grid + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def camera2rays_full(view, **kwargs):
    w = view.image_width  # // 4
    h = view.image_height  # // 4
    # y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    device = torch.device('cuda')

    x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')

    fx = 0.5 * w / np.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)  # original focal length
    pixtocams = torch.eye(3, device=device)
    pixtocams[0, 0] = 1/fx
    pixtocams[1, 1] = 1/fy
    pixtocams[0, 2] = -w/2/fx
    pixtocams[1, 2] = -h/2/fy

    T = torch.linalg.inv(view.world_view_transform.T).to(device)
    origins, _, directions, _, _ = camera_utils_zipnerf.pixels_to_rays(
        x.reshape(-1), y.reshape(-1),
        pixtocams.reshape(1, 3, 3),
        T[:3].reshape(1, 3, 4),
        camtype=view.model,
        distortion_params=view.distortion_params,
        xnp=torch
    )
    origins = origins.float().cuda().contiguous()
    directions = directions.float().cuda().contiguous()
    # ic(camera2rays(view)[1])
    # ic(directions)
    return origins, directions

def camera2rays(view, **kwargs):
    w = view.image_width
    h = view.image_height

    fx = 0.5 * w / math.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / math.tan(0.5 * view.FoVy)  # original focal length

    directions = get_ray_directions(h, w, [fx, fy], **kwargs).cuda()  # (h, w, 3)
    directions = (directions / torch.norm(directions, dim=-1, keepdim=True))

    T = torch.linalg.inv(view.world_view_transform.T.cuda())
    rays_o, rays_d = get_rays(
        directions,
        T,
    )  # both (h*w, 3)
    rays_o = (rays_o).contiguous()
    return rays_o, rays_d

def _eval_sh2_scalar(coeffs, dirs):
    """
    Antons function
    Evaluate degree-2 SH at unit directions dirs [N,3].

    coeffs: [N, 9] SH coefficients (bands 0,1,2)
    dirs:   [N, 3] unit direction vectors
    returns: [N] float values in (0, 1) after sigmoid
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    # Standard real SH basis Y_l^m, order: Y00, Y1-1,Y10,Y11, Y2-2,Y2-1,Y20,Y21,Y22
    basis = torch.stack([
        torch.ones_like(x) * 0.2820948,                   # Y00
        0.4886025 * y,                                     # Y1-1
        0.4886025 * z,                                     # Y10
        0.4886025 * x,                                     # Y11
        1.0925484 * x * y,                                 # Y2-2
        1.0925484 * y * z,                                 # Y2-1
        0.3153916 * (3*z*z - 1),                          # Y20
        1.0925484 * x * z,                                 # Y21
        0.5462742 * (x*x - y*y),                          # Y22
    ], dim=-1)                                             # [N, 9]
    return (coeffs * basis).sum(dim=-1)                   # [N]


def compute_comoving_light_color(pc, view, light_offsets):
    """
    Berechnet das Co-Moving Light mit gelernten Normalen (Ohne Wasser-Effekte).
    
    pc:            parameters of gaussian model
    view:          camera object
    light_offsets: list of light position relative to the camera e.g. [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    """
    sh_normals = pc.get_sh_normals 
    gxyz       = pc.get_xyz    
    albedo     = pc.get_albedo

    device = gxyz.device

    c2w         = torch.linalg.inv(view.world_view_transform.T.cuda())

    #light intemsity per point
    total_irradiance = torch.zeros_like(gxyz)

    if not isinstance(light_offsets, torch.Tensor):
        offsets_tensor = torch.tensor(light_offsets, dtype=torch.float32, device=device)
    else:
        offsets_tensor = light_offsets.to(device=device, dtype=torch.float32)
    
    if offsets_tensor.ndim == 1:
        offsets_tensor = offsets_tensor.unsqueeze(0)

    for offset in offsets_tensor:
        #light position
        hom_light_pos = torch.cat([offset, torch.tensor([1.0], device=device)])
        
        # transform relative position into world coordinates
        light_pos = (c2w @ hom_light_pos)[:3]
        # ──────────────────────────────────────────────────────────────────────────────────

        # calculate vectors from lights to gaussians
        to_gaussian   = gxyz - light_pos.unsqueeze(0)                    # [N, 3]
        dist          = to_gaussian.norm(dim=-1).clamp(min=1e-8)         # [N]
        to_gaussian_n = to_gaussian / dist.unsqueeze(-1)

        #inv_sq = 1.0 / (4 * np.pi * dist.pow(2)) # Inverse-Square-Law
        inv_sq = 1.0
        raw_sh = _eval_sh2_scalar(sh_normals, to_gaussian_n)              
        lambert = torch.sigmoid(raw_sh)

        light_power = 12.5 # test value

        contrib = (light_power * inv_sq * lambert).unsqueeze(-1)          # [N, 1]
        total_irradiance += contrib
        
    print(f"Total irradiance: {total_irradiance}")
    print(f"Albedo: {albedo}")
    net_color = (albedo / math.pi) * total_irradiance    
    return net_color

def splinerender(
    view,
    pc: GaussianModel,
    pipe,
    light_tensor, #lpc
    scaling_modifier=1.0,
    random=False,
    tmin=None,
    tmax=1e7,
    mode="lighted", #lpc
):
    device = pc.get_xyz.device
    if view.model == ProjectionType.PERSPECTIVE:
        rays_o, rays_d = camera2rays(view, random=random)
    else:
        rays_o, rays_d = camera2rays_full(view, random=False)

    densification_metric = torch.zeros((pc.get_xyz.shape[0], 1), device=device)
    densification_metric.requires_grad = True
    
    scales, density = pc.get_scale_and_density_for_rendering()    
    scales *= scaling_modifier
    
    if(mode=="lighted"):
        
        net_color = compute_comoving_light_color(pc, view, light_tensor) #lpc
    elif(mode=="no_lighting"):
        ambient_intensity = 1 #lpc
        net_color = pc.get_albedo * ambient_intensity #lpc
    elif(mode=="normals"):
        
        raw_normals = pc.get_sh_normals[:, :3] #lpc
        norm = torch.norm(raw_normals, dim=-1, keepdim=True).clamp(min=1e-8) #lpc
        normalized_normals = raw_normals / norm #lpc
        net_color = normalized_normals * 0.5 + 0.5 #lpc

    rendered_features = RGB2SH(net_color).reshape(-1, 1, 3) #lpc

    tmin = pc.tmin if tmin is None else tmin
    out, extras = trace_rays(
        pc.get_xyz,
        scales,
        pc.get_rotation,
        density,
        rendered_features,
        0,
        rays_o,
        rays_d,
        tmin,
        tmax,
        densification_metric=densification_metric,
        max_iters=MAX_ITERS
    )

    torch.cuda.synchronize()
    radii = torch.ones_like(densification_metric[..., 0])

    rendered_image = out[:, :3].T.reshape(3, view.image_height, view.image_width)
    num_pixels = (extras['touch_count'] // 2)

    # aspect_ratio = scales.max(dim=-1).values / scales.min(dim=-1).values
    side_length = (num_pixels).float().sqrt() #/ aspect_ratio # mul by 2 to get to rect, then sqrt
    radii = side_length / 2 * np.sqrt(2) * 2.5 * 5

    return {
        "render": rendered_image,
        "densification_metric": densification_metric,
        "visibility_filter": num_pixels >= 4,
        "touch_count": extras['touch_count'],
        "radii": radii, # match gaussian radius
        "iters": extras["iters"].reshape(view.image_height, view.image_width),
        "opacity": out[:, 3].reshape(-1, 1),
        "distortion_loss": out[:, 4].reshape(-1, 1),
    }

