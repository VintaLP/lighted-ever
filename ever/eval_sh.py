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

kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "splinetracers/fast_ellipsoid_splinetracer/slang/sh_kernel.slang")
)

class EvalSH(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor,
        sh_degree: int
    ):
        block_size = 64
        rayo = rayo.reshape(3).contiguous()
        means = means.contiguous()
        features = features.contiguous()
        color = torch.zeros_like(means)
        ctx.sh_degree = sh_degree
        num_prim = means.shape[0]
        kernels.sh_kernel(
            means=means, features=features, ray_origin=rayo, colors=color, sh_degree=sh_degree
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(num_prim // block_size + 1, 1, 1),
        )

        ctx.save_for_backward(
            means, features, rayo, color
        )
        return color

    @staticmethod
    def backward(ctx, dL_dcolor: torch.Tensor):
        block_size = 64
        means, features, rayo, color = ctx.saved_tensors
        num_prim = means.shape[0]
        dL_dfeat = torch.zeros_like(features)
        dL_dcolor = dL_dcolor.contiguous()
        kernels.bw_sh_kernel(means=means, features=features, dL_dfeatures=dL_dfeat, ray_origin=rayo, dL_dcolors=dL_dcolor, sh_degree=ctx.sh_degree).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=((num_prim + block_size - 1) // block_size, 1, 1),
        )
        return None, dL_dfeat, None, None, None


def eval_sh(
        means,
        features,
        rayo,
        sh_degree):
    out = EvalSH.apply(
        means,
        features,
        rayo,
        sh_degree
    )
    return out

