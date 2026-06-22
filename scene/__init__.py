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
import os
import random
import copy
import pickle
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from icecream import ic
from utils import cam_util

def transform_cameras_pca(cameras):
    if len(cameras) == 0:
        return cameras, np.eye(4)
    poses = np.stack([
        np.linalg.inv(view.world_view_transform.T.cpu().numpy())[:3]
        for view in cameras], axis=0)
    new_poses, transform = cam_util.transform_poses_pca(poses)
    for i, cam in enumerate(cameras):
        T = np.eye(4)
        T[:3] = new_poses[i][:3]
        T = torch.linalg.inv(torch.tensor(T).float()).to(cam.world_view_transform.device)
        T[:3, 0] = T[:3, 0]*torch.linalg.det(T[:3, :3])
        cameras[i] = set_pose(cam, T)
    return cameras, transform

def set_pose(camera, T):
    # camera.world_view_transform = T.T
    # camera.full_proj_transform = (
    #     camera.world_view_transform.unsqueeze(0).bmm(
    #         camera.projection_matrix.unsqueeze(0))).squeeze(0)
    # camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    camera.R = T[:3, :3].T.numpy()
    camera.T = T[:3, 3].numpy()
    camera.update()
    return camera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.light_offset = getattr(scene_info, "light_offset", np.array([0.0, 0.0, 0.0])) #lpc
        print(f"Loaded light offset: {self.light_offset}")
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loaded Train Cameras: {len(self.train_cameras[resolution_scale])}")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loaded Test Cameras: {len(self.test_cameras[resolution_scale])}")
    
        if self.loaded_iter:
            # self.gaussians.load_th(os.path.join(self.model_path,
            #                                                "point_cloud",
            #                                                "iteration_" + str(self.loaded_iter),
            #                                                "point_cloud.th"))
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.num_additional_pts, args.additional_size_multi)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
