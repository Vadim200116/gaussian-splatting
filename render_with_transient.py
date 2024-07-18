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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import segmentation_models_pytorch as smp
from pathlib import Path

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    pred_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred")
    overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "overlay")
    overlay_diff_path = os.path.join(model_path, name, "ours_{}".format(iteration), "overlay_diff")

    transient_path = os.path.join(model_path, "transient.pth")
    transient_model = smp.UnetPlusPlus('timm-mobilenetv3_small_100', in_channels=3, encoder_weights='imagenet',
                                       classes=1,
                                       activation="sigmoid", encoder_depth=5,
                                       decoder_channels=[224, 128, 64, 32, 16]).to("cuda")
    transient_model.load_state_dict(torch.load(transient_path))
    transient_model.eval()

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(pred_path, exist_ok=True)
    makedirs(overlay_path, exist_ok=True)
    makedirs(overlay_diff_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path,  f"{view.image_name}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}.png"))

        gt_image_resized = torch.nn.functional.interpolate(gt.unsqueeze(0), size=(224, 224), mode='bilinear',
                                                           align_corners=True)
        # predict weights for each pixel with transient model
        with torch.no_grad():
            weights = transient_model(gt_image_resized)
            weights = torch.nn.functional.interpolate(weights, size=(gt.shape[1], gt.shape[2]), mode='bilinear',
                                                      align_corners=True).squeeze(0)

        # save as grayscale image
        torchvision.utils.save_image(weights[0], os.path.join(pred_path, f"{view.image_name}.png"))

        overlay = gt.clone()  # [3, h, w]
        overlay[1] = overlay[1] + weights[0]
        overlay[1] = torch.clamp(overlay[1], 0, 1)
        # save overlay image
        torchvision.utils.save_image(overlay, os.path.join(overlay_path, f"{view.image_name}.png"))

        overlay_diff = gt.clone()
        weights = weights.squeeze()
        m1 = (weights > 0.9)  # 1 is dynamic
        if view.mask is not None:
            m2 = view.mask.cuda().bool()  # 0 is dynamic
        else:
            m2 = torch.ones_like(weights).bool()
        overlay_diff[1, torch.logical_and(m1, m2)] = 1
        overlay_diff[0, torch.logical_and(~m1, ~m2)] = 1
        # save overlay imag
        torchvision.utils.save_image(overlay_diff, os.path.join(overlay_diff_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)