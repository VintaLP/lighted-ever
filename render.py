#
# Copyright (C) 2023, Inria, Google
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import math
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.ever import splinerender
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from scene import GaussianModel
from scene.dataset_readers import ProjectionType
from datetime import datetime

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,folder_suffix,light_tensor,mode="lighted", ):
    
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders/{folder_suffix}") #fynn
    no_lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"rendersNoLight/{folder_suffix}") #fynn
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"render_normals/{folder_suffix}") #fynn
    #render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if mode=="lighted":
        makedirs(render_path, exist_ok=True) #fynn
    elif mode=="no_lighting":
        makedirs(no_lighting_path, exist_ok=True) #fynn
    elif mode=="normals":
        makedirs(normal_path, exist_ok=True) #fynn
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx != 424:
        #     continue
        N = 1
        frendering = None
        for i in range(N):
            rendering = splinerender(view, gaussians,pipeline,light_tensor,mode=mode, random=False)["render"]
            if frendering is None:
                frendering = rendering / N
            else:
                frendering += rendering / N
        gt = view.original_image[0:3, :, :]
        if(mode=="lighted"):
            torchvision.utils.save_image(frendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        elif(mode=="no_lighting"):
            torchvision.utils.save_image(frendering, os.path.join(no_lighting_path, '{0:05d}'.format(idx) + ".png"))  
        elif(mode=="normals"):
            torchvision.utils.save_image(frendering, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))  
        else:
            print("wrong mode!")
            break        
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, checkpoint, opt,mode="lighted"):
    with torch.no_grad():
        #gaussians = GaussianModel(dataset.sh_degree, dataset.use_neural_network, dataset.max_opacity, dataset.tmin)
        gaussians = GaussianModel(sh_degree=dataset.sh_degree, max_opacity=dataset.max_opacity, tmin=dataset.tmin)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        light_tensor = torch.tensor(scene.light_offset, dtype=torch.float32, device="cuda") #lpc

        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        folder_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") #fynn
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, folder_suffix,light_tensor, mode)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, folder_suffix, light_tensor, mode)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="lighted") #lcp
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.checkpoint = args.checkpoint if hasattr(args, "checkpoint") else None

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.checkpoint, op.extract(args), args.mode)
