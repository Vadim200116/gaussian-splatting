import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F

def pad_and_unpad(func):
    def wrapper(self, x):
        width, height = x.shape[2:]
        target_width = (width + 31) // 32 * 32
        target_height = (height + 31) // 32 * 32

        pad_width_left = (target_width - width) // 2
        pad_width_right = target_width - width - pad_width_left
        pad_height_top = (target_height - height) // 2
        pad_height_bottom = target_height - height - pad_height_top

        padded_input = F.pad(x, 
                            (pad_height_top, pad_height_bottom, pad_width_left, pad_width_right), 
                            mode='replicate')

        weights = func(self, padded_input).squeeze()
        rec_weights = weights[pad_width_left: pad_width_left + width, pad_height_top: pad_height_top + height]

        return rec_weights

    return wrapper

class UnetModel(torch.nn.Module):
    def __init__(self, in_channels: int=3):
        super().__init__()

        self.unet = smp.UnetPlusPlus('timm-mobilenetv3_small_100', in_channels=in_channels, encoder_weights='imagenet', classes=1,
                             activation="sigmoid", encoder_depth=5, decoder_channels=[224, 128, 64, 32, 16])

    @pad_and_unpad
    def forward(self, x):
        return self.unet(x)