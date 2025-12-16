import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import configuration and external modules
from config import (
    SCALE, IMAGE_SIZE, EMBED_DIM, SWIN_DEPTHS,
    NUM_HEADS, WINDOW_SIZE, MLP_RATIO
)
from .reconstruction import DifferentiableFBP
from .swin_blocks import SwinWrapper


# --------------------------------------------------------------------------
#  Enhanced SinoUpsampler with Residual Skip Connection
# --------------------------------------------------------------------------
class ResidualSinoUpsampler(nn.Module):
    """
    Upsamples sinogram along angular (height) dimension using PixelShuffle,
    with a bicubic-interpolated skip connection for residual learning.
    Inspired by EDSR, SwinIR, and dual-domain CT SR models.
    """
    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__()
        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * self.scale, kernel_size=3, padding=1)
        self.refine_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        print(f"ResidualSinoUpsampler initialized: {self.scale}x angular upsampling with bicubic skip.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic skip connection (preserves low-frequency structure)
        skip = F.interpolate(x, scale_factor=(self.scale, 1), mode='bicubic', align_corners=False)
        
        # PixelShuffle path
        up = self.conv(x)
        b, c, h, w = up.shape
        up = up.view(b, c // self.scale, self.scale, h, w)
        up = up.permute(0, 1, 3, 2, 4).contiguous()
        up = up.view(b, c // self.scale, h * self.scale, w)
        
        # Residual addition + refinement
        out = self.act(up + skip)
        out = self.refine_conv(out)
        return out


# --------------------------------------------------------------------------
#  Main Dual-Domain Swin Model
# --------------------------------------------------------------------------
class DualDomainSwin(nn.Module):
    """
    Dual-Domain CT Swin Model with:
    - Residual upsampling in sinogram domain
    - Proper min-max normalization & denormalization
    - Global residual from initial FBP reconstruction
    - Full HU-scale output
    """

    @staticmethod
    def _minmax_norm(x, eps: float = 1e-6):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        scale = (x_max - x_min).clamp_min(eps)
        x_n = (x - x_min) / scale
        return x_n, x_min, scale

    def __init__(self, lr_sino_shape: Tuple[int, int], scale: int | None = None):
        super().__init__()
        self.lr_sino_h, self.lr_sino_w = lr_sino_shape
        self.window_size = WINDOW_SIZE
        self.scale = int(scale) if scale is not None else int(SCALE)

        # Head: LR sinogram to feature space
        self.conv_in = nn.Conv2d(1, EMBED_DIM, kernel_size=3, padding=1)

        # Padding for window divisibility
        h_sino, w_sino = self.lr_sino_h, self.lr_sino_w
        pad_h = (self.window_size - h_sino % self.window_size) % self.window_size
        pad_w = (self.window_size - w_sino % self.window_size) % self.window_size
        sino_padded_res = (h_sino + pad_h, w_sino + pad_w)

        # Sinogram domain processing
        self.sinogram_processor = SwinWrapper(
            dim=EMBED_DIM,
            input_resolution=sino_padded_res,
            depth=SWIN_DEPTHS[0],
            num_heads=NUM_HEADS[0],
            window_size=WINDOW_SIZE,
            mlp_ratio=MLP_RATIO
        )
        print(f"Sinogram processor: {self.lr_sino_h}x{self.lr_sino_w} -> padded {sino_padded_res}")

        # Upsampling with residual skip
        self.upsampler = nn.Sequential(
            ResidualSinoUpsampler(EMBED_DIM, self.scale),
            nn.Conv2d(EMBED_DIM, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Differentiable FBP bridge
        self.diff_fbp = DifferentiableFBP(self.scale)

        # Image domain processing
        self.image_processor = SwinWrapper(
            dim=EMBED_DIM,
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            depth=SWIN_DEPTHS[1],
            num_heads=NUM_HEADS[1],
            window_size=WINDOW_SIZE,
            mlp_ratio=MLP_RATIO
        )
        print(f"Image processor: {IMAGE_SIZE}x{IMAGE_SIZE}")

        # Tail: Feature to image
        self.conv_reco_in = nn.Conv2d(1, EMBED_DIM, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(EMBED_DIM, 1, kernel_size=3, padding=1)

        # Global residual (from initial reco to final output)
        self.global_residual_conv = nn.Conv2d(1, 1, kernel_size=1)

        print("--- SOTA Model Initialization Complete ---")

    def forward(self, x_s: torch.Tensor, metadata: dict) -> torch.Tensor:
        # 1. Normalize input sinogram
        x_s_norm, x_min, x_scale = self._minmax_norm(x_s)

        # 2. Sinogram domain: enhance LR features
        sino_feat = self.conv_in(x_s_norm)
        _, _, h, w = sino_feat.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        padded = F.pad(sino_feat, (0, pad_w, 0, pad_h), 'reflect')
        sino_feat = self.sinogram_processor(padded)[:, :, :h, :w]

        # 3. Upsample with residual skip
        sino_sr_norm = torch.sigmoid(self.upsampler(sino_feat))
        sino_sr = sino_sr_norm * x_scale + x_min  # Denormalize to physical units

        # 4. Reconstruct initial image via diff FBP
        img_reco = self.diff_fbp(sino_sr, metadata)
        global_residual = self.global_residual_conv(img_reco)  # Preserve initial reco

        # 5. Image domain: refine reconstruction
        img_feat = self.conv_reco_in(img_reco)
        img_feat = self.image_processor(img_feat)
        y_logits = self.conv_out(img_feat)

        # 6. Add global residual for consistency
        y_final = y_logits + global_residual

        return y_final