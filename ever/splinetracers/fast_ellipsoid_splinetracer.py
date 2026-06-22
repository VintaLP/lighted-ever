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

from pathlib import Path
from typing import *

import slangtorch
import torch
from torch.autograd import Function
from ever.eval_sh import eval_sh

import sys

from utils.sh_utils import RGB2SH
sys.path.append(str(Path(__file__).parent))

from build.splinetracer.extension import fast_ellipsoid_splinetracer_cpp_extension as sp
kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"),
    # includePaths=[str(Path(__file__).parent / 'slang')]
)

otx = sp.OptixContext(torch.device("cuda:0"))


class SplineTracer(Function):
    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        features: torch.Tensor,
        sh_degree: int,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: float,
        densification_metric: torch.Tensor, # this is required for backward pass gradients
        max_iters: int
    ):
        device = rayo.device
        assert mean.device == device
        prims = sp.Primitives(device)
        prims.add_primitives(mean, scale, quat, density, features)
        gas = sp.GAS(otx, device, prims, True, False, True)        
        forward = sp.Forward(otx, device, prims, True)
        out = forward.trace_rays(gas, sh_degree, rayo, rayd, tmin, tmax, max_iters)        

        ctx.device = device
        ctx.max_iters = max_iters
        ctx.saved = out["saved"]
        ctx.tmin = tmin
        ctx.tmax = tmax
        ctx.sh_degree = sh_degree
        tri_collection = out["tri_collection"]
        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = (distortion_pt1 - distortion_pt2)
        color_and_loss = torch.cat([out["image"], distortion_loss.reshape(-1, 1)], dim=1)
        initial_inds = out['initial_touch_inds'][:out['initial_touch_count'][0]]

        ctx.save_for_backward(
            mean, scale, quat, density, features, rayo, rayd, tri_collection, out['initial_drgb'], initial_inds
        )

        return color_and_loss, dict(
                iters=ctx.saved.iters,
                touch_count=ctx.saved.touch_count,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, extras_dict=None):
        (
            mean,
            scale,
            quat,
            density,
            features,
            rayo,
            rayd,
            tri_collection,
            initial_drgb,
            initial_inds
        ) = ctx.saved_tensors
        device = ctx.device

        num_prims = mean.shape[0]
        num_rays = rayo.shape[0]
        dL_dmeans = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dscales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dquats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        dL_dfeatures = torch.zeros_like(features)
        dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)

        dL_densification_metric = torch.zeros((num_prims, 1), dtype=torch.float32, device=device)

        touch_count = torch.zeros((num_prims), dtype=torch.int32, device=device)

        dL_dinital_drgb = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        block_size = 16
        if ctx.saved.iters.sum() > 0:
            dual_model = (
                mean,
                scale,
                quat,
                density,
                features,
                dL_dmeans,
                dL_dscales,
                dL_dquats,
                dL_ddensities,
                dL_dfeatures,
                dL_drayo,
                dL_drayd,
                dL_densification_metric,
            )

            kernels.backwards_kernel(
                last_states=ctx.saved.states,
                iters=ctx.saved.iters,
                tri_collection=tri_collection,
                ray_origins=rayo,
                ray_directions=rayd,
                model=dual_model,
                dL_dinital_drgb=dL_dinital_drgb,
                touch_count=touch_count,
                dL_doutputs=grad_output.contiguous(),
                tmin=ctx.tmin,
                tmax=ctx.tmax,
                max_iters=ctx.max_iters,
                sh_degree=ctx.sh_degree
            ).launchRaw(
                blockSize=(block_size, 1, 1),
                gridSize=(num_rays // block_size + 1, 1, 1),
            )
            if initial_inds.shape[0] > 0:
                ray_block_size = 64
                second_block_size = 16
                kernels.backwards_initial_drgb_kernel(
                    ray_origins=rayo,
                    ray_directions=rayd,
                    model=dual_model,
                    initial_drgb=initial_drgb,
                    initial_inds=initial_inds,
                    dL_dinital_drgb=dL_dinital_drgb,
                    touch_count=touch_count,
                    tmin=ctx.tmin,
                ).launchRaw(
                    blockSize=(ray_block_size, second_block_size, 1),
                    gridSize=(
                        rayo.shape[0] // ray_block_size + 1,
                        initial_inds.shape[0] // second_block_size + 1,
                        1),
                )
        v = 1e+3
        mean_v = 1e+3
        return (
            dL_dmeans.clip(min=-mean_v, max=mean_v),
            dL_dscales.clip(min=-v, max=v),
            dL_dquats.clip(min=-v, max=v),
            dL_ddensities.clip(min=-50, max=50).reshape(density.shape),
            dL_dfeatures.clip(min=-v, max=v),
            None, # sh_degree
            dL_drayo.clip(min=-v, max=v),
            dL_drayd.clip(min=-v, max=v),
            None, # tmin
            None, # tmax
            dL_densification_metric, # densification_metric
            None, # max_iters
        )


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    sh_degree: int,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000,
    densification_metric=None,
    max_iters: int = 500,
    per_ray_sh: bool = False,
):    
    if not per_ray_sh:
        # instead of passing the flag down and branching in slang 'encode' the color as sh0
        colors = eval_sh(mean, features, rayo[0], sh_degree)
        features = RGB2SH(colors)[:,torch.newaxis,:]
        sh_degree = 0

    out = SplineTracer.apply(
        mean.contiguous(),
        scale.contiguous(),
        quat.contiguous(),
        density.contiguous(),
        features.contiguous(),
        sh_degree,
        rayo,
        rayd,
        tmin,
        tmax,
        densification_metric,
        max_iters
    )
    return out

