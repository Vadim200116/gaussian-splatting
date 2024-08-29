import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch import nn

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

class MLPModel(torch.nn.Module):

    def __init__(self, num_classes: int, num_features: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid(),
        )

    def get_regularizer(self):
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(
            abs(self.mlp[2].weight.data)
        )

    def _prep_semantics(self, semantics,  height: int, width: int, num_frequencies: int, device: str = "cuda"):
        sf = nn.Upsample(
                size=(height, width),
                mode="bilinear",
        )(semantics.unsqueeze(0)).squeeze(0)
        pos_enc = self._get_positional_encodings(
            height, width, num_frequencies, device
        ).permute((2, 0, 1))
        sf = torch.cat([sf, pos_enc], dim=0)
        sf_flat = sf.reshape(sf.shape[0], -1).permute((1, 0))
        return sf_flat
    
    def _get_positional_encodings(
        self, height: int, width: int, num_frequencies: int, device: str = "cuda"
    ) -> torch.Tensor:
        """Generates positional encodings for a given image size and frequency range.

        Args:
        height: height of the image
        width: width of the image
        num_frequencies: number of frequencies
        device: device to use

        Returns:

        """
        # Generate grid of (x, y) coordinates
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        # Normalize coordinates to the range [0, 1]
        y = y / (height - 1)
        x = x / (width - 1)

        # Create frequency range [1, 2, 4, ..., 2^(num_frequencies-1)]
        frequencies = (
            torch.pow(2, torch.arange(num_frequencies, device=device)).float() * torch.pi
        )

        # Compute sine and cosine of the frequencies multiplied by the coordinates
        y_encodings = torch.cat(
            [torch.sin(frequencies * y[..., None]), torch.cos(frequencies * y[..., None])],
            dim=-1,
        )
        x_encodings = torch.cat(
            [torch.sin(frequencies * x[..., None]), torch.cos(frequencies * x[..., None])],
            dim=-1,
        )

        # Combine the encodings
        pos_encodings = torch.cat([y_encodings, x_encodings], dim=-1)

        return pos_encodings        

    def forward(self, semantics,  height: int, width: int, num_frequencies: int, device: str = "cuda") -> torch.Tensor:
        return self.mlp(self._prep_semantics(semantics, height, width, num_frequencies, device)).reshape(height, width)
