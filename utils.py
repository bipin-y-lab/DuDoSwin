# /cluster/du78sywa/utils.py

import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# --- NEW METRIC IMPORTS ---
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# --------------------------------------------------------------------------
#  Custom Loss Function
# --------------------------------------------------------------------------

class L1L2Loss(nn.Module):
    """
    Loss for CT images in HU units.
    Using L1 + L2 (MSE) combination with appropriate scaling for HU values.
    """
    def __init__(self, l1_weight=1.0, l2_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

    def forward(self, prediction, target):
        # Scale down the loss for HU values (which are in hundreds/thousands)
        # These scaling factors are heuristics; you may need to tune them.
        l1 = self.l1_loss(prediction, target) / 100.0  # Scale for HU range
        l2 = self.l2_loss(prediction, target) / 10000.0  # Scale for HU^2 range
        total_loss = self.l1_weight * l1 + self.l2_weight * l2
        return total_loss

# --------------------------------------------------------------------------
#  Data I/O Functions
# --------------------------------------------------------------------------

def load_tiff_stack_with_metadata(file_path: Path) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Loads a TIFF stack and its metadata from the 'ImageDescription' tag.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if file_path.suffix.lower() not in ('.tif', '.tiff'):
        raise ValueError('File must be a .tif or .tiff')

    with tiff.TiffFile(str(file_path)) as tif:
        data = tif.asarray()
        page = tif.pages[0]
        description_tag = page.tags.get("ImageDescription")
        metadata_str = description_tag.value if description_tag else None

    if not metadata_str:
        return data, None

    try:
        # The metadata string might use single quotes, which is not valid JSON.
        metadata = json.loads(metadata_str.replace("'", "\""))
    except json.JSONDecodeError:
        print(f"[Warning] Could not parse metadata in {file_path.name}. Returning None for metadata.")
        metadata = None

    return data, metadata


def save_to_tiff_stack(array: np.ndarray, out_file: Path):
    """
    Saves a NumPy array to a TIFF file, creating parent directories if needed.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_file), array.astype(np.float32))
    

# --------------------------------------------------------------------------
#  Model Utility
# --------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --------------------------------------------------------------------------
#  Evaluation Metrics (PyTorch-based)
# --------------------------------------------------------------------------

def calculate_psnr(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   data_range: float = 4000.0, 
                   eps: float = 1e-8) -> torch.Tensor:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) on tensors
    for CT images in HU units.
    
    Args:
        pred: Predicted image tensor
        target: Target image tensor
        data_range: The typical dynamic range of CT data (-1000 to 3000 = 4000)
        eps: Small value to prevent division by zero
    
    Returns:
        PSNR value as a tensor.
    """
    mse = F.mse_loss(pred, target)
    if mse < eps:
        return torch.tensor(100.0, device=pred.device, dtype=pred.dtype)
    
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr

def calculate_ssim(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   data_range: float = 4000.0) -> torch.Tensor:
    """
    Calculates the Structural Similarity Index (SSIM) on tensors
    for CT images in HU units.
    
    Args:
        pred: Predicted image tensor
        target: Target image tensor
        data_range: The typical dynamic range of CT data (-1000 to 3000 = 4000)
    
    Returns:
        SSIM value as a tensor.
    """
    # Ensure inputs are float32 for SSIM calculation
    pred_float = pred.to(torch.float32)
    target_float = target.to(torch.float32)
    
    # Initialize metric on the correct device
    ssim_metric = SSIM(data_range=data_range).to(pred.device)
    
    return ssim_metric(pred_float, target_float)

# --------------------------------------------------------------------------
#  Debugging Utilities
# --------------------------------------------------------------------------

def debug_tensor_stats(tensor: torch.Tensor, name: str = "tensor"):
    """
    Print debugging statistics for a tensor
    """
    if tensor.numel() == 0:
        print(f"{name}: Empty tensor")
        return
    
    print(f"{name} stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")
    print()