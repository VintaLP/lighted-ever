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
import subprocess

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, folder_suffix, mode="lighted"):
    
    # Setup dynamic paths based on render mode (split by train/test via 'name')
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    
    if mode == "lighted":
        render_path = os.path.join(base_path, f"renders_lighted/{folder_suffix}")
        gts_path = os.path.join(base_path, "gt_lighted")
    elif mode == "unlit":
        render_path = os.path.join(base_path, f"renders_unlit/{folder_suffix}")
        gts_path = os.path.join(base_path, "gt_unlit")
    elif mode == "normals":
        render_path = os.path.join(base_path, f"render_normals/{folder_suffix}")
        gts_path = os.path.join(base_path, "gt_normals")
    elif mode == "brightness":
        render_path = os.path.join(base_path, f"render_brightness/{folder_suffix}")
        gts_path = os.path.join(base_path, "gt_brightness")
    else:
        print(f"[ERROR] Wrong mode: {mode}")
        return
        
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} ({mode})")):
        
        # Fetch light directly from the current camera view
        light_tensor = getattr(view, "light", None)
        
        rendering = splinerender(view, gaussians, pipeline, light_tensor, mode=mode, random=False)["render"]
                
        if mode == "lighted": gt = view.original_image[0:3, :, :]
        elif mode == "unlit": gt = view.original_unlit[0:3, :, :]
        elif mode == "normals": gt = view.original_normals[0:3, :, :]
        elif mode == "brightness": gt = view.original_brightness[0:3, :, :]
        
        # Use original camera name from dataset (e.g. cam1/0001.png)
        image_name = view.image_name
        if not image_name.endswith(".png"):
            image_name += ".png"
            
        out_render = os.path.join(render_path, image_name)
        out_gt = os.path.join(gts_path, image_name)
        
        # Automatically create subdirectories if they do not exist
        os.makedirs(os.path.dirname(out_render), exist_ok=True)
        os.makedirs(os.path.dirname(out_gt), exist_ok=True)
        
        # Save the generated render and ground truth
        torchvision.utils.save_image(rendering, out_render)
        torchvision.utils.save_image(gt, out_gt)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, checkpoint, opt, mode="lighted"):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree=dataset.sh_degree, max_opacity=dataset.max_opacity, tmin=dataset.tmin, light_strength=dataset.light_strength)
        
        # Load scene only ONCE per script execution to save memory
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        folder_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        modes_to_render = ["lighted", "unlit", "normals", "brightness"] if mode == "all" else [mode]
        
        for current_mode in modes_to_render:
            if not skip_train:
                 render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, folder_suffix, current_mode)
            if not skip_test:
                 render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, folder_suffix, current_mode)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="lighted") 
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    args.checkpoint = args.checkpoint if hasattr(args, "checkpoint") else None

    # Render process
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.checkpoint, op.extract(args), args.mode)
    
    # Change permissions
    
            
    # Auto-execute metrics.py
    try:
        print("\n[INFO] Starting automated metrics evaluation...")
        # Startet automatisch metrics.py und übergibt den aktuellen Modellpfad
        subprocess.run(["python", "refined_metrics.py", "-m", args.model_path], check=True)
        print("[INFO] Evaluation finished successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to run refined_metrics.py: {e}")

    if os.path.exists(args.model_path):
            try:
                print(f"\n[INFO] Fixing file permissions for: {args.model_path}")
                subprocess.run(["chmod", "-R", "777", args.model_path], check=True)
                subprocess.run(["chown", "-R", "1000:1000", args.model_path], check=True)
                print("[INFO] Permissions fixed successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to change permissions: {e}")    