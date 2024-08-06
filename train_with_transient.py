# - make new branch in my repo and cherry-pick commits
# - build new docker with depth rasterizer
# - launch on full resolution
# - launch with lowest resolution to then upsample mask

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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, total_variation_loss, image2canny
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, normalize_depth
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
# import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
import torch.nn.functional as F
import scipy
from  utils.general_utils import pred_weights, mask_image, make_gif, prep_img, join_mask_semantics
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, stop_train_transient, eval_path):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    transient_model = smp.UnetPlusPlus('timm-mobilenetv3_small_100', in_channels=8 if dataset.flow else 3, encoder_weights='imagenet', classes=1,
                             activation="sigmoid", encoder_depth=5, decoder_channels=[224, 128, 64, 32, 16]).to("cuda")
    # transient_model = smp.UnetPlusPlus('timm-efficientnet-b0', in_channels=3, encoder_weights='imagenet',
    #                                    classes=1,
    #                                    activation="sigmoid", encoder_depth=5,
    #                                    decoder_channels=[224, 128, 64, 32, 16]).to("cuda")
    transient_optimizer = torch.optim.Adam(transient_model.parameters(), lr=1e-5)
    transient_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(transient_optimizer, opt.iterations, 5e-6)
    overlays_path = Path(dataset.model_path) / "overlays"
    overlays_path.mkdir(exist_ok=True)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_mapping = {int(view.image_name): view for view in scene.getTrainCameras()}
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if dataset.flow:
            if viewpoint_cam.flow is not None:
                flow = viewpoint_cam.flow.cuda()
                next_image_idx = int(viewpoint_cam.image_name) + 1
                next_image = viewpoint_mapping[next_image_idx].original_image.cuda()
                transient_input = torch.cat((gt_image, next_image, flow), dim=0)
            else:
                continue
        else:
            transient_input = gt_image
        weights = pred_weights(transient_input, transient_model)

        Ll1 = l1_loss(image, gt_image)
        mask = (weights.clone().detach() > 0.5).float()

        if viewpoint_cam.semantics is not None:
            mask = join_mask_semantics(mask, viewpoint_cam.semantics)

        alpha = np.exp(opt.schedule_beta * np.floor((1 + iteration) / 1.5))
        sampled_mask = 1 - torch.bernoulli(
            torch.clip(alpha + (1 - alpha) * (1-mask.clone()),
            min=0.0, max=1.0)
        )

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, size_average=False))
        loss =  ((1-sampled_mask) * loss).mean()

        if opt.lambda_tv and iteration > opt.tv_from_iter and iteration < opt.tv_until_iter:
            depth = normalize_depth(render_pkg["depth"])

            tv_mask = None
            if iteration > opt.canny_start: 
                canny_mask = image2canny(image.permute(1,2,0), 50, 150, isEdge1=False)
                canny_mask = scipy.ndimage.binary_erosion(canny_mask, iterations=2, border_value=1)
                tv_mask = torch.tensor(canny_mask).cuda()
            tv = total_variation_loss(depth, tv_mask)
            loss += opt.lambda_tv * tv

        loss.backward()

        transient_loss = ((1-weights) * l1_loss(image.detach(), gt_image)).mean() + 0.1 * torch.abs(weights).mean()
        transient_loss.backward()

        iter_end.record()

        transient_optimizer.step()
        transient_optimizer.zero_grad()
        transient_scheduler.step()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(transient_model.state_dict(), scene.model_path + f"/transient_{iteration}.pth")
                if iteration == saving_iterations[-1]:
                    transient_model.eval()
                    rendered_images = []
                    masked_images = []
                    masked_images_semantics = []
                    cams = sorted(scene.getTrainCameras(), key=lambda x: int(x.image_name))
    
                    for viewpoint_cam in tqdm(cams):
                        rendered_images.append(prep_img(render(viewpoint_cam, gaussians, pipe, bg)["render"]))
                        gt_image = viewpoint_cam.original_image.cuda()
                        if dataset.flow:
                            if viewpoint_cam.flow is not None:
                                flow = viewpoint_cam.flow.cuda()
                                next_image_idx = int(viewpoint_cam.image_name) + 1
                                next_image = viewpoint_mapping[next_image_idx].original_image.cuda()
                                transient_input = torch.cat((gt_image, next_image, flow), dim=0).cuda()
                            else:
                                continue
                        else:
                            transient_input = gt_image
                        masked_images.append(mask_image(transient_input, transient_model))
                        masked_images_semantics.append(mask_image(transient_input, transient_model, semantics=viewpoint_cam.semantics))

                    make_gif(masked_images, os.path.join(scene.model_path, "masked.gif"), framerate=8)
                    make_gif(masked_images_semantics, os.path.join(scene.model_path, "masked+sam.gif"), framerate=8)
                    make_gif(rendered_images, os.path.join(scene.model_path, "rendered.gif"), framerate=8)
                    if eval_path:
                        dataset.source_path = eval_path

                        stub_gaussians = GaussianModel(dataset.sh_degree)
                        eval_scene = Scene(dataset, stub_gaussians)    
                        ref_cams = []
                        for cam in eval_scene.getTrainCameras():
                            try:
                                int(cam.image_name)
                            except:
                                cam.image_name = cam.image_name[4:]
                                ref_cams.append(cam)

                        ref_cams = sorted(ref_cams, key= lambda x: int(x.image_name))
                        rendered_images = []
                        for cam in tqdm(ref_cams):
                            rendered_images.append(prep_img(render(cam, gaussians, pipe, bg)["render"]))

                        make_gif(rendered_images, os.path.join(args.model_path, f"similar_traj.gif"), framerate=8, rate=10)


                        ssims = []
                        psnrs = []
                        for idx, view in enumerate(tqdm(ref_cams)):
                            with torch.no_grad():
                                rendered_img  = torch.clamp(render(view, gaussians, pp, bg)["render"], 0.0, 1.0)
                                gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
                                ssims.append(ssim(rendered_img, gt_image).mean())
                                psnrs.append(psnr(rendered_img, gt_image).mean())
                                if (idx + 1) % 50 == 0:
                                    torch.cuda.empty_cache()

                        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))

                        with open(os.path.join(args.model_path, "report.txt"), "w") as f:
                            f.write("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                            f.write("\n")
                            f.write("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if viewpoint.mask is not None:
                        mask = viewpoint.mask.cuda()
                        gt_image[:, mask == 0] = 0
                        image[:, mask == 0] = 0
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6003)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--stop_train_transient', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default=None)

    # parser.add_argument("--masked", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.stop_train_transient, args.eval_path)

    # All done
    print("\nTraining complete.")
