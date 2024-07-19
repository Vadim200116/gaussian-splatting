import torch
from pathlib import Path
import argparse
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import ImageReadMode


# def split_segmentation_tensor(image_tensor):
#     # Ensure the input tensor is in the shape [3, h, w]
#     assert image_tensor.shape[0] == 3, "Input tensor should have 3 color channels"
#     # Reshape and permute the tensor for easier processing
#     h, w = image_tensor.shape[1], image_tensor.shape[2]
#     image_tensor = image_tensor.view(3, -1) # Shape: [3, h * w]
#     # Identify unique colors
#     unique_colors, inverse_indices = torch.unique(image_tensor, return_inverse=True, dim=1)
#     # Create binary masks
#     n = unique_colors.shape[1]
#     masks = torch.zeros(n, h * w, dtype=torch.bool, device=image_tensor.device).scatter(0, inverse_indices.unsqueeze(0), 1).reshape(n, h, w)
#     return masks, unique_colors.transpose(0, 1)


# def compute_mask_ioa(mask, dynamic_score, dynamic_threshold=0.9, cover_threshold=0.75):
#     masks, colors = split_segmentation_tensor(mask)  # [n, h, w] (bool), [n, 3]
#     dynamic_score = dynamic_score > dynamic_threshold  # [1, h, w] (bool)
#     i = (masks * dynamic_score).sum(dim=(1, 2)).float()
#     a = masks.sum(dim=(1, 2)).float()
#     indices = i / (a + 1e-6) > cover_threshold
#     masks = masks[indices]
#     colors = colors[indices]
#     dynamic_mask = masks.sum(dim=0)  # [h, w], bool
#     return dynamic_mask, masks, colors


# def compute_mask_ioa(mask, dynamic_score, dynamic_threshold=0.9, cover_threshold=0.75, prev_masks=None, prev_colors=None):
#     masks, colors = split_segmentation_tensor(mask)  # [n, h, w] (bool), [n, 3]
#     dynamic_score = dynamic_score > dynamic_threshold  # [1, h, w] (bool)
#     i = (masks * dynamic_score).sum(dim=(1, 2)).float()
#     a = masks.sum(dim=(1, 2)).float()
#     indices = i / (a + 1e-6) > cover_threshold
#     if prev_masks is not None and prev_colors is not None:
#         # downscale 4 times to avoid out of memory
#         resized_masks = torch.nn.functional.interpolate(masks.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).bool()
#         resized_prev_masks = torch.nn.functional.interpolate(prev_masks.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).bool()
#         i = (resized_masks.unsqueeze(1) * resized_prev_masks.unsqueeze(0)).sum(dim=(-1, -2)).float()  # [n, m]
#         col = (colors.unsqueeze(1) == prev_colors.unsqueeze(0)).all(dim=2)  # [n, m]
#         new_indices = torch.any((i / (a.view(-1, 1) + 1e-6) > cover_threshold) * col, dim=1)  # [n]
#         indices = indices | new_indices
#     masks = masks[indices]
#     colors = colors[indices]
#     dynamic_mask = masks.sum(dim=0)  # [h, w], bool
#     return dynamic_mask, masks, colors


def compute_mask_ioa(mask_tensor, dynamic_score, dynamic_threshold=0.9, cover_threshold=0.85):
    unique_ids, inverse_indices = torch.unique(mask_tensor.view(1, -1), return_inverse=True)
    # Create binary masks
    n = unique_ids.shape[0]
    h, w = mask_tensor.shape[-2:]
    masks = torch.zeros(n, h * w, dtype=torch.bool, device=mask_tensor.device).scatter(0, inverse_indices,
                                                                                       1).reshape(n, h, w)
    dynamic_score = dynamic_score > dynamic_threshold  # [1, h, w] (bool)
    i = (masks * dynamic_score).sum(dim=(-1, -2)).float()
    a = masks.sum(dim=(-1, -2)).float()
    indices = i / (a + 1e-6) > cover_threshold
    masks = masks[indices]
    unique_ids = unique_ids[indices]
    dynamic_mask = masks.sum(dim=0)  # [h, w], bool
    return dynamic_mask, masks, unique_ids


def select_mask_by_ids(mask_tensor, ids):
    unique_ids, inverse_indices = torch.unique(mask_tensor.view(1, -1), return_inverse=True)
    # Create binary masks
    n = unique_ids.shape[0]
    h, w = mask_tensor.shape[-2:]
    masks = torch.zeros(n, h * w, dtype=torch.bool, device=mask_tensor.device).scatter(0, inverse_indices,
                                                                                       1).reshape(n, h, w)
    # get indices where unique colors are equal to any of the colors
    indices = torch.any((unique_ids.unsqueeze(1) == ids.unsqueeze(0)), dim=1)  # [n]
    masks = masks[indices]
    dynamic_mask = masks.sum(dim=0)  # [h, w], bool
    return dynamic_mask


# def select_mask_by_colors(mask, colors):
#     masks, unique_colors = split_segmentation_tensor(mask)  # [n, h, w], [n, 3]
#     # get indices where unique colors are equal to any of the colors
#     indices = torch.any((unique_colors.unsqueeze(1) == colors.unsqueeze(0)).all(dim=2), dim=1)  # [n]
#     masks = masks[indices]
#     colors = unique_colors[indices]
#
#     dynamic_mask = masks.sum(dim=0)  # [h, w], bool
#     return dynamic_mask, masks, colors


def run(mask_path, dynamic_score_path, output_mask_path):
    mask_path = Path(mask_path)
    dynamic_score_path = Path(dynamic_score_path)
    output_mask_color_path = Path(output_mask_path) / 'color_masks'
    output_mask_path = Path(output_mask_path) / 'masks'

    output_mask_path.mkdir(parents=True, exist_ok=True)
    output_mask_color_path.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_path.glob('*.png'))


    # for mask_file in mask_paths:
    #     mask = torchvision.io.read_image(str(mask_file), mode=ImageReadMode.RGB).cuda()
    #     dynamic_score = torchvision.io.read_image(str(dynamic_score_path / mask_file.name), mode=ImageReadMode.GRAY).cuda()
    #     mask = torchvision.transforms.functional.resize(mask, dynamic_score.shape[-2:], interpolation=InterpolationMode.NEAREST)
    #
    #     dynamic_mask, masks, colors = compute_mask_ioa(mask, dynamic_score)
    #     torchvision.utils.save_image(dynamic_mask.float(), output_mask_path / mask_file.name)
    #     # save combined mask with different colors
    #     color_mask = (masks.unsqueeze(1).float() * colors.view(-1, 3, 1, 1) / 255.).sum(dim=0)
    #     torchvision.utils.save_image(color_mask, output_mask_color_path / mask_file.name)

    ids_count = {}
    for mask_file in mask_paths:
        mask_tensor = torch.load(mask_file.parent.parent / "Masks" / (mask_file.stem + '.pt')).cuda().unsqueeze(0)
        dynamic_score = torchvision.io.read_image(str(dynamic_score_path / mask_file.name), mode=ImageReadMode.GRAY).cuda().float() / 255.
        mask_tensor = torchvision.transforms.functional.resize(mask_tensor, dynamic_score.shape[-2:], interpolation=InterpolationMode.NEAREST)

        dynamic_mask, masks, unique_ids = compute_mask_ioa(mask_tensor, dynamic_score)
        for i in unique_ids:
            i = i.item()
            if i not in ids_count:
                ids_count[i] = 0
            ids_count[i] += 1

    print(ids_count)

    selected_ids = torch.tensor([c for c, count in ids_count.items() if count > 30]).cuda()

    for mask_file in mask_paths:
        color_mask = torchvision.io.read_image(str(mask_file), mode=ImageReadMode.RGB).cuda()
        mask_tensor = torch.load(mask_file.parent.parent / "Masks" / (mask_file.stem + '.pt')).cuda().unsqueeze(0)
        # dynamic_score = torchvision.io.read_image(str(dynamic_score_path / mask_file.name), mode=ImageReadMode.GRAY).cuda()
        # mask = torchvision.transforms.functional.resize(mask, dynamic_score.shape[-2:], interpolation=InterpolationMode.NEAREST)

        dynamic_mask = select_mask_by_ids(mask_tensor, selected_ids)
        torchvision.utils.save_image(dynamic_mask.float(), output_mask_path / mask_file.name)
        # save combined mask with different colors
        color_mask[:, dynamic_mask.logical_not()] = 0
        color_mask = color_mask.float() / 255.
        torchvision.utils.save_image(color_mask, output_mask_color_path / mask_file.name)

    # prev_masks, prev_colors = None, None
    # for mask_file in mask_paths:
    #     mask = torchvision.io.read_image(str(mask_file), mode=ImageReadMode.RGB).cuda()
    #     dynamic_score = torchvision.io.read_image(str(dynamic_score_path / mask_file.name), mode=ImageReadMode.GRAY).cuda()
    #     mask = torchvision.transforms.functional.resize(mask, dynamic_score.shape[-2:], interpolation=InterpolationMode.NEAREST)
    #
    #     dynamic_mask, masks, colors = compute_mask_ioa(mask, dynamic_score, prev_masks=prev_masks, prev_colors=prev_colors)
    #     prev_masks, prev_colors = masks, colors
    #     torchvision.utils.save_image(dynamic_mask.float(), output_mask_path / mask_file.name)
    #     # save combined mask with different colors
    #     color_mask = (masks.unsqueeze(1).float() * colors.view(-1, 3, 1, 1) / 255.).sum(dim=0)
    #     torchvision.utils.save_image(color_mask, output_mask_color_path / mask_file.name)



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Compute mask IoU')
    parser.add_argument('--mask_path', type=str, help='Path to the mask images')
    parser.add_argument('--dynamic_score_path', type=str, help='Path to the dynamic score images')
    parser.add_argument('--output_mask_path', type=str, help='Path to save the output masks')
    args = parser.parse_args()
    run(args.mask_path, args.dynamic_score_path, args.output_mask_path)