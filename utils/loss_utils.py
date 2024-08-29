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

def l1_loss(network_output, gt, average=True):
    diff = torch.abs((network_output - gt))
    if average:
        return diff.mean()
    else:
        return diff

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
        return ssim_map

def total_variation_loss(img, mask=None):
    d_w = torch.pow(img[:, :-1] - img[:, 1:], 2)
    d_h = torch.pow(img[:-1, :] - img[1:, :], 2)

    if mask is not None:
        d_w *= mask[:, :-1]
        d_h *= mask[:-1, :]

    w_variance = torch.mean(d_w)
    h_variance = torch.mean(d_h)

    return h_variance + w_variance

def robust_mask(error_per_pixel: torch.Tensor, loss_threshold: float) -> torch.Tensor:
    epsilon = 1e-3
    error_per_pixel = error_per_pixel.mean(axis=-1, keepdims=True)
    error_per_pixel = error_per_pixel.squeeze(-1).unsqueeze(0)
    is_inlier_pixel = (error_per_pixel < loss_threshold).float()
    window_size = 3
    channel = 1
    window = torch.ones((1, 1, window_size, window_size),
            dtype=torch.float) / (window_size * window_size)
    if error_per_pixel.is_cuda:
        window = window.cuda(error_per_pixel.get_device())
    window = window.type_as(error_per_pixel)
    has_inlier_neighbors = F.conv2d(
            is_inlier_pixel, window, padding=window_size // 2, groups=channel)
    has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()
    is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > epsilon).float()
    pred_mask = is_inlier_pixel.squeeze(0).unsqueeze(-1)
    return pred_mask

def update_running_stats(running_stats, err, cfg):
    running_stats["hist_err"] = 0.95 * running_stats["hist_err"] + err
    mid_err = torch.sum(running_stats["hist_err"]) * cfg.robust_percentile 
    running_stats["avg_err"] = torch.linspace(0, 1, cfg.bin_size+1)[
            torch.where(torch.cumsum(running_stats["hist_err"], 0) >= mid_err)[0][0]]

    lower_err = torch.sum(running_stats["hist_err"]) * cfg.lower_bound
    upper_err = torch.sum(running_stats["hist_err"]) * cfg.upper_bound

    running_stats["lower_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(running_stats["hist_err"], 0) >= lower_err)[0][0]]
    running_stats["upper_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(running_stats["hist_err"], 0) >= upper_err)[0][0]]

