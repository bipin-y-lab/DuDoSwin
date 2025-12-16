import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import tifffile as tiff
from typing import List, Tuple

from config import SCALE, BASE_DIR

def normalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a tensor to the [0, 1] range."""
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max - tensor_min < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - tensor_min) / (tensor_max - tensor_min)

class DualDomainDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-processed data for efficient training.
    Each sample corresponds to one sinogram and its target reconstruction.
    """
    def __init__(self, file_ids: List[str], scale: int = None):
        super().__init__()
        self.file_ids = file_ids
        self.scale = scale if scale is not None else SCALE

        # Define data paths based on config
        self.lr_sino_dir = BASE_DIR / "data_preprocessed" / f"scale_{self.scale}x" / "lr_sinograms"
        self.target_reco_dir = BASE_DIR / "data_preprocessed" / f"scale_{self.scale}x" / "target_reconstructions"
        self.metadata_dir = BASE_DIR / "data_preprocessed" / f"scale_{self.scale}x" / "metadata"

        # Verify directories exist
        if not self.lr_sino_dir.exists():
            raise FileNotFoundError(f"LR sinogram directory not found: {self.lr_sino_dir}")
        if not self.target_reco_dir.exists():
            raise FileNotFoundError(f"Target reconstruction directory not found: {self.target_reco_dir}")
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")

        print(f"Dataset initialized with {len(self.file_ids)} samples")
        print(f"Loading data from: {self.lr_sino_dir.parent}")

    def __len__(self) -> int:
        return len(self.file_ids)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Loads a single pre-processed (input, target, metadata) triplet from disk.
        
        Returns:
            x_s: Low-resolution sinogram (1, H_lr, W) where H_lr = H_full // SCALE
            y_target: High-resolution target reconstruction (1, H_img, W_img)
            metadata: Dictionary containing reconstruction parameters
        """
        scan_id = self.file_ids[idx]

        # 1. Load low-resolution sinogram
        sino_lr_path = self.lr_sino_dir / f"{scan_id}_sino_lr.tif"
        if not sino_lr_path.exists():
            raise FileNotFoundError(f"LR sinogram not found: {sino_lr_path}")
        
        sino_lr = tiff.imread(sino_lr_path).astype(np.float32)
        x_s = torch.from_numpy(sino_lr)

        # 2. Load high-resolution target reconstruction
        reco_hr_path = self.target_reco_dir / f"{scan_id}_reco_hr.tif"
        if not reco_hr_path.exists():
            raise FileNotFoundError(f"Target reconstruction not found: {reco_hr_path}")
        
        reco_hr = tiff.imread(reco_hr_path).astype(np.float32)
        y_target = torch.from_numpy(reco_hr)

        # 3. Load metadata
        meta_path = self.metadata_dir / f"{scan_id}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        metadata['scan_id'] = scan_id

        # 4. Prevent DataLoader from converting angles to tensor
        # Store angles as a tuple instead of list - DataLoader won't convert tuples to tensors
        if 'angles' in metadata and isinstance(metadata['angles'], list):
            metadata['angles'] = tuple(metadata['angles'])
        
        # Also ensure other metadata values are proper Python types
        for key in ['dso', 'ddo', 'du', 'hu_factor']:
            if key in metadata:
                if hasattr(metadata[key], 'item'):  # numpy scalar
                    metadata[key] = float(metadata[key].item())
                else:
                    metadata[key] = float(metadata[key])
        
        for key in ['lr_angles_count', 'hr_angles_count', 'scale_factor']:
            if key in metadata:
                if hasattr(metadata[key], 'item'):  # numpy scalar
                    metadata[key] = int(metadata[key].item())
                else:
                    metadata[key] = int(metadata[key])

        # 5. NO NORMALIZATION - Keep both in their natural units
        x_s = x_s.unsqueeze(0)  # Shape: (1, H_lr, W) - Raw sinogram values
        y_target = y_target.unsqueeze(0)  # Shape: (1, H_img, W_img) - Raw HU values

        return x_s, y_target, metadata

    def get_sample_info(self, idx: int) -> dict:
        """
        Returns information about a specific sample without loading the data.
        """
        scan_id = self.file_ids[idx]
        meta_path = self.metadata_dir / f"{scan_id}_meta.json"
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return {
                'scan_id': scan_id,
                'projection_id': metadata.get('projection_id', 'unknown'),
                'sinogram_index': metadata.get('sinogram_index', -1),
                'total_sinograms': metadata.get('total_sinograms', -1)
            }
        else:
            return {'scan_id': scan_id, 'projection_id': 'unknown', 'sinogram_index': -1, 'total_sinograms': -1}

    @classmethod
    def get_all_scan_ids(cls, scale: int, base_dir: Path) -> List[str]:
        """
        Utility method to get all available scan IDs for a given scale.
        """
        metadata_dir = base_dir / "data_preprocessed" / f"scale_{scale}x" / "metadata"
        if not metadata_dir.exists():
            return []
        
        scan_ids = []
        for meta_file in metadata_dir.glob("*_meta.json"):
            scan_id = meta_file.stem.replace('_meta', '')
            scan_ids.append(scan_id)
        
        return sorted(scan_ids)