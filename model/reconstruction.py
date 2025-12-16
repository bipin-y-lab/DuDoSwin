import numpy as np
import torch
import torch.nn as nn
from torch_radon import RadonFanbeam
from torch.amp import autocast

from config import FBP_FILTER, IMAGE_SIZE, VOXEL_SIZE, DEVICE, SCALE


class DifferentiableFBP(nn.Module):
    """
    A differentiable Filtered Back-Projection (FBP) layer that properly
    handles the super-resolved sinograms from the dual-domain model.

    This module dynamically configures the Radon transform operator for each
    input sinogram using its corresponding metadata, ensuring that the
    geometry matches the upsampled sinogram dimensions.
    """

    def __init__(self, scale: int | None = None):
        super().__init__()
        self.image_size = IMAGE_SIZE
        self.fbp_filter = FBP_FILTER
        self.voxel_size = VOXEL_SIZE
        self.scale = int(scale) if scale is not None else int(SCALE)
        print(f"DifferentiableFBP initialized for {self.image_size}x{self.image_size} images using '{self.fbp_filter}' filter.")

    def forward(self, sinogram: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Performs the differentiable FBP reconstruction with proper handling
        of super-resolved sinograms.
        
        Args:
            sinogram: Super-resolved sinogram with shape (B, 1, angles*SCALE, detectors)
            metadata: Dictionary containing reconstruction parameters
        
        Returns:
            Reconstructed images with shape (B, 1, IMAGE_SIZE, IMAGE_SIZE)
        """
        batch_size = sinogram.shape[0]

        # Handle batched processing
        if batch_size == 1:
            # Extract single metadata - simplified since dataset prevents tensor conversion
            single_meta = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    # DataLoader wraps single values in lists
                    single_meta[key] = value[0]
                else:
                    # Use value directly
                    single_meta[key] = value
            
            return self._reconstruct_single(sinogram, single_meta)
        else:
            # Process batch sample by sample
            print(f"[Info] Processing batch of {batch_size} samples")
            reconstructions = []
            for i in range(batch_size):
                sino_i = sinogram[i].unsqueeze(0)
                
                # Extract metadata for sample i
                meta_i = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)):
                        meta_i[key] = value[i] if i < len(value) else value[0]
                    elif isinstance(value, torch.Tensor):
                        if value.numel() > i:
                            meta_i[key] = value[i].tolist() if value.dim() > 1 else value[i].item()
                        else:
                            meta_i[key] = value[0].item() if value.numel() > 0 else value.item()
                    else:
                        meta_i[key] = value
                
                reco_i = self._reconstruct_single(sino_i, meta_i)
                reconstructions.append(reco_i)
            return torch.cat(reconstructions, dim=0)

    def _reconstruct_single(self, sinogram_single: torch.Tensor, 
                           metadata_single: dict) -> torch.Tensor:
        """
        Reconstruct one super-resolved sinogram → image with differentiable FBP.
        
        The key insight: The model outputs a super-resolved sinogram with angles*SCALE
        angles, so we need to generate the corresponding angle array for reconstruction.
        """
        # --- 1. Input Validation ---
        if sinogram_single.dim() != 4:
            raise ValueError(f"Expected 4D sinogram (B,C,H,W), got {sinogram_single.shape}")
        
        if sinogram_single.shape[0] != 1:
            raise ValueError(f"This function handles single samples, got batch size {sinogram_single.shape[0]}")

        # --- 2. Extract Sinogram Properties ---
        _, _, sr_angles, detectors = sinogram_single.shape

        # --- 3. Handle all complex metadata types ---
        if 'angles' in metadata_single:
            angles_data = metadata_single['angles']
            
            # --- Convert to NumPy array, handling all cases ---

            # Case 1: The data is a single PyTorch Tensor. Move to CPU and convert.
            if isinstance(angles_data, torch.Tensor):
                hr_angles_rad = angles_data.detach().cpu().numpy()

            # Case 2: The data is a list or tuple.
            elif isinstance(angles_data, (list, tuple)):
                # Sub-case 2a: Check if it's a LIST OF TENSORS.
                if len(angles_data) > 0 and isinstance(angles_data[0], torch.Tensor):
                    # Use a list comprehension to move each tensor to the CPU and get its value.
                    processed_list = [t.cpu().item() for t in angles_data]
                    hr_angles_rad = np.array(processed_list)
                # Sub-case 2b: It's a normal list of numbers (int/float).
                else:
                    hr_angles_rad = np.array(angles_data)

            # Case 3: It's already a NumPy array.
            elif isinstance(angles_data, np.ndarray):
                hr_angles_rad = angles_data

            else:
                raise ValueError(f"Unknown angles data type: {type(angles_data)}")
            
        else:
            raise ValueError("Metadata must contain 'angles' for reconstruction")

        # --- 4. Generate Angles for SR Sinogram ---
        # The SR sinogram should have SCALE times more angles than the low-res input
        if 'lr_angles_count' in metadata_single:
            lr_count = metadata_single['lr_angles_count']
            # Ensure it's a scalar integer
            if isinstance(lr_count, torch.Tensor):
                lr_count = int(lr_count.item())
            else:
                lr_count = int(lr_count)
            expected_sr_angles = lr_count * self.scale
        else:
            # Use HR angles count from original data
            expected_sr_angles = (len(hr_angles_rad) // self.scale) * self.scale
          
        # Validate that our model output matches expectations
        if sr_angles != expected_sr_angles:
            print(f"Warning: Model output has {sr_angles} angles, expected {expected_sr_angles}")
            # Continue with actual model output dimensions
        
        # --- 5. Generate uniform angle distribution for the super-resolved sinogram ---
        # Generate angles based on the original HR angle range
        if len(hr_angles_rad) < 2:
            raise ValueError(f"Need at least 2 angles for interpolation, got {len(hr_angles_rad)}")
            
        start_angle = float(hr_angles_rad[0])  # Ensure scalar
        
        # Calculate angular increment from original data
        angular_increment = float(hr_angles_rad[1] - hr_angles_rad[0])  # Ensure scalar
        
        # For uniform distribution, the end angle should cover the same range
        # as the original data but with higher sampling
        end_angle = start_angle + len(hr_angles_rad) * angular_increment

        # Generate the new angles for the super-resolved sinogram
        angles_rad_sr = np.linspace(start_angle, end_angle, sr_angles, endpoint=False)

        angles_tensor = torch.from_numpy(angles_rad_sr).to(
            device=sinogram_single.device,
            dtype=torch.float32
        )

        # --- 6. Geometry Setup ---
        vox_scaling = 1.0 / self.voxel_size
        
        # Create Radon operator with the super-resolved geometry
        radon_op = RadonFanbeam(
            self.image_size,
            angles_tensor,
            source_distance=vox_scaling * metadata_single['dso'],
            det_distance=vox_scaling * metadata_single['ddo'],
            det_count=detectors,
            det_spacing=vox_scaling * metadata_single['du'],
            clip_to_circle=False,
        )

        # --- 7. Sinogram Preparation ---
        sino_no_channel = sinogram_single.squeeze(1)  # Remove channel dimension: (1, angles, detectors)
        sino_scaled = sino_no_channel * vox_scaling    # Apply physical scaling

        # --- 8. FBP Reconstruction ---
        with autocast('cuda', enabled=False):
            # Ensure float32 for numerical stability
            sino_float32 = sino_scaled.to(torch.float32)
            filtered = radon_op.filter_sinogram(sino_float32, self.fbp_filter)
            reco = radon_op.backprojection(filtered)

        # --- 9. Convert to Hounsfield Units ---
        hu0 = float(metadata_single.get('hu_factor', 0.02))
        # Add epsilon to avoid division by zero
        eps = 1e-6
        hu0_tensor = torch.tensor(max(abs(hu0), eps), device=reco.device, dtype=reco.dtype)
        reco_hu = 1000.0 * ((reco - hu0_tensor) / (hu0_tensor + eps))


        # --- 10. Return with Channel Dimension ---
        result = reco_hu.unsqueeze(1).to(torch.float32)  # Add channel dim: (1, 1, H, W)
        
        return result