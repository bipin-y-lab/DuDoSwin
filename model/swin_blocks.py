# /cluster/du78sywa/swin_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .swin_transformer import BasicLayer

class SwinModule(nn.Module):
    """
    A learnable branch using a Swin BasicLayer
    """
    def __init__(self, channels, input_resolution, swin_depth, num_heads, window_size, mlp_ratio):
        super().__init__()
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.filter_layer = BasicLayer(
            dim=channels, 
            input_resolution=input_resolution, 
            depth=swin_depth,
            num_heads=num_heads, 
            window_size=window_size, 
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm, 
            use_checkpoint=True
        )
        self.skip_conv = nn.Conv2d(channels, channels, kernel_size=1)
        print("SwinModule: Module ready.")

    def forward(self, x):
        guidance = self.guidance_conv(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        filtered_flat = self.filter_layer(x_flat)
        filtered = filtered_flat.transpose(1, 2).view(B, C, H, W)
        skip_out = self.skip_conv(x)
        return guidance * filtered + (1 - guidance) * skip_out

# --------------------------------------------------------------------------
#  3. Main "Swin Wrapper" Composite Block
# --------------------------------------------------------------------------
class SwinWrapper(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio):
        super().__init__()
        # Only the Swin Module branch
        self.swin_md = SwinModule(dim, input_resolution, depth, num_heads, window_size, mlp_ratio)
        
    def forward(self, x):
        # Direct pass through SwinModule
        return self.swin_md(x)