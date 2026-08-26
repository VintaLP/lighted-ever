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

import os
import torch
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import traceback
from utils.system_utils import searchForMaxIteration
import time
from gaussian_renderer.ever import splinerender
from gaussian_renderer import network_gui

from scene.dataset_readers import ProjectionType

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

PREVIEW_RES_FACTOR = 1

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, light_strength=dataset.light_strength)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.training_setup(opt)
    
    if checkpoint and not checkpoint.endswith("None"):
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        load_iteration = -1
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))
        gaussians.load_ply(os.path.join(dataset.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(loaded_iter),
                                                       "point_cloud.ply"))  

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    


    torch.cuda.empty_cache()
    st = time.time()

    initial_model_state = None
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    #initial_ply_path = os.path.join(current_script_dir, "normal_visualize_point_cloud.ply")
    initial_ply_path = os.path.join(os.path.join(dataset.model_path,
                                                       "point_cloud",
                                                       "iteration_1",
                                                       "point_cloud.ply"))
    if os.path.exists(initial_ply_path):
        temp_g = GaussianModel(dataset.sh_degree, light_strength=dataset.light_strength)
        temp_g.load_ply(initial_ply_path)
        initial_model_state = {
            "xyz": temp_g._xyz.detach().clone(),
            "albedo": temp_g._albedo.detach().clone(),
            "normals": temp_g._normals.detach().clone(),
            "scaling": temp_g._scaling.detach().clone(),
            "rotation": temp_g._rotation.detach().clone(),
            "opacity": temp_g._opacity.detach().clone()
        }
        del temp_g
    
    saved_current_state = None
    while True:
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                custom_cam, do_training, show_unlit, switch_camera, keep_alive, scaling_modifer, light_strength, ambient_light, light_switch, normals, only_brightness, normal_cloud = network_gui.receive()
                if custom_cam != None:    

                    if normal_cloud:
                        if saved_current_state is None:
                            saved_current_state = {
                                "xyz": gaussians._xyz.detach().clone(),
                                "albedo": gaussians._albedo.detach().clone(),
                                "normals": gaussians._normals.detach().clone(),
                                "scaling": gaussians._scaling.detach().clone(),
                                "rotation": gaussians._rotation.detach().clone(),
                                "opacity": gaussians._opacity.detach().clone()
                            }
                        if initial_model_state is not None:
                            with torch.no_grad():
                                gaussians._xyz = initial_model_state["xyz"].clone()
                                gaussians._albedo = initial_model_state["albedo"].clone()
                                gaussians._normals = initial_model_state["normals"].clone()
                                gaussians._scaling = initial_model_state["scaling"].clone()
                                gaussians._rotation = initial_model_state["rotation"].clone()
                                gaussians._opacity = initial_model_state["opacity"].clone()
                    else:
                        if saved_current_state is not None:
                            with torch.no_grad():
                                gaussians._xyz = saved_current_state["xyz"].clone()
                                gaussians._albedo = saved_current_state["albedo"].clone()
                                gaussians._normals = saved_current_state["normals"].clone()
                                gaussians._scaling = saved_current_state["scaling"].clone()
                                gaussians._rotation = saved_current_state["rotation"].clone()
                                gaussians._opacity = saved_current_state["opacity"].clone()
                            saved_current_state = None
                                          
                    # custom_cam.model = viewpoint_cam.model
                    # custom_cam.distortion_params = viewpoint_cam.distortion_params
                    # custom_cam.model=ProjectionType.FISHEYE
                    custom_cam.model=ProjectionType.PERSPECTIVE
                    image_width = custom_cam.image_width
                    image_height = custom_cam.image_height
                    custom_cam.image_width = image_width // PREVIEW_RES_FACTOR
                    custom_cam.image_height = image_height // PREVIEW_RES_FACTOR

                    st = time.time()   
                    light = scene.light
                    custom_cam.colmap_id = 2 if switch_camera else 1 
                    mode = ""
                    if normals:
                        mode = "normals"
                    elif only_brightness:
                        mode = "only_brightness"
                    else:
                        mode=("no_lighting" if light_switch else "lighted")

                    
                    gaussians.light_strength =  int(light_strength)



                    net_image = splinerender(custom_cam, gaussians, pipe, light,scaling_modifier=scaling_modifer, random=False,debug_iteration=30000, tmin=0, mode=mode, ambient_intensity=ambient_light)["render"]
                    # net_image = renderFunc(custom_cam, gaussians, pipe, background, scaling_modifer, random=False, tmin=0)["render"]
                    print(f"{1/(time.time()-st)}", end='\r')
                    net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    net_image = cv2.resize(net_image, (image_width, image_height))
                    # ic(net_image.shape, net_image.dtype)
                    net_image_bytes = memoryview(net_image)
                network_gui.send(net_image_bytes, dataset.source_path)
                torch.cuda.empty_cache()
            except Exception as e:
                print(traceback.format_exc())
                network_gui.conn = None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mode", default="lighted") #lcp

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    args.mode = "no_lighting"

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done

