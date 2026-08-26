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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def normal_consistency_loss(xyz, xyz_normals,k=8, max_samples=1000):
    """
    xyz: gaussian positions
    xyz_normal: xyz normals of gaussians
    k: nearest neighbour count

    punishes dissimilar normals close to each other
    """

    N = xyz.shape[0]

    if N <= k:
        return torch.tensor(0.0, device=xyz.device)
    
    if N<=max_samples:
        sampled_xyz = xyz
        sampled_normals = xyz_normals
    else:
        indices = torch.randperm(N, device=xyz.device)[:max_samples]
        sampled_xyz = xyz[indices]
        sampled_normals = xyz_normals[indices]


    nx = sampled_normals[:, 0]
    ny = sampled_normals[:, 1]
    nz = sampled_normals[:, 2]

    normals = torch.stack([nx, ny, nz], dim=1)

    #calculate distance between gaussians
    dist_matrix = torch.cdist(sampled_xyz, xyz)

    #calculate k-nearest neighbors
    _, indices = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
    neighbors_idx = indices[:, 1:]

    #get normals of nearest neighbors
    neighbor_theta = xyz_normals[neighbors_idx, 0]
    neighbor_phi = xyz_normals[neighbors_idx, 1]

    neighbor_normals_x = torch.sin(neighbor_theta) * torch.cos(neighbor_phi)
    neighbor_normals_y = torch.sin(neighbor_theta) * torch.sin(neighbor_phi)
    neighbor_normals_z = torch.cos(neighbor_theta)

    neighbor_normals = torch.stack([neighbor_normals_x, neighbor_normals_y, neighbor_normals_z], dim=2)

    #calculate cosinus simularity between normals
    central_normals_expanded = normals.unsqueeze(1).expand(-1, k, -1) 
    cos_sim = torch.sum(central_normals_expanded * neighbor_normals, dim=2)

    loss = (1.0 - cos_sim).mean()
    return loss