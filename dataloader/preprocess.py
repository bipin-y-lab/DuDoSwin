import json
from pathlib import Path
import numpy as np
import torch
from torch_radon import RadonFanbeam
from tqdm import tqdm

import sys
from pathlib import Path

# Add the parent directory to sys.path for relative imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils import load_tiff_stack_with_metadata, save_to_tiff_stack

# Define output directories for the pre-processed data
LR_SINO_DIR = config.BASE_DIR / "data_preprocessed" / f"scale_{config.SCALE}x" / "lr_sinograms"
TARGET_RECO_DIR = config.BASE_DIR / "data_preprocessed" / f"scale_{config.SCALE}x" / "target_reconstructions"
METADATA_DIR = config.BASE_DIR / "data_preprocessed" / f"scale_{config.SCALE}x" / "metadata"

# Create directories
LR_SINO_DIR.mkdir(parents=True, exist_ok=True)
TARGET_RECO_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """
    Pre-processes raw projection data into low-res sinograms and high-res
    target reconstructions. This version correctly handles the angle dimension
    and ensures metadata matches the actual sinogram dimensions.
    """
    all_files = sorted(list(config.PROJ_DIR.glob("*_hd_proj_fan_geometry.tif")))
    print(f"Found {len(all_files)} projection files to process.")

    total_sinograms_processed = 0

    for file_path in tqdm(all_files, desc="Processing Projection Files"):
        projections, metadata = load_tiff_stack_with_metadata(file_path)
        if metadata is None:
            print(f"Skipping {file_path.name} due to missing metadata.")
            continue

        # Get projection identifier (e.g., L014)
        projection_id = file_path.stem.replace('_hd_proj_fan_geometry', '')
        
        print(f"Processing projection {projection_id}")
        print(f"Projection shape: {projections.shape}")  # Debug info
        
        # Handle different projection orientations
        # Check if projections are (angles, detectors, slices) or (slices, angles, detectors)
        if projections.ndim == 3:
            # Determine the correct orientation based on typical CT scan dimensions
            if projections.shape[0] > projections.shape[2]:
                # Likely (angles, detectors, slices)
                num_angles, num_detectors, num_slices = projections.shape
                angle_axis, detector_axis, slice_axis = 0, 1, 2
            else:
                # Likely (slices, angles, detectors) - need to transpose
                print(f"Transposing projection data from shape {projections.shape}")
                projections = np.transpose(projections, (1, 2, 0))  # (slices, angles, detectors) -> (angles, detectors, slices)
                num_angles, num_detectors, num_slices = projections.shape
                angle_axis, detector_axis, slice_axis = 0, 1, 2
        else:
            raise ValueError(f"Unexpected projection shape: {projections.shape}")
        
        print(f"Final projection shape: {projections.shape} (angles={num_angles}, detectors={num_detectors}, slices={num_slices})")
        
        # Process each sinogram (slice) in the projection stack
        for slice_idx in tqdm(range(num_slices), desc=f"Processing {projection_id}", leave=False):
            # Create unique scan ID for each sinogram
            scan_id = f"{projection_id}_sino_{slice_idx:03d}"
            
            # Extract the sinogram slice: (angles, detectors)
            prj_slice = np.flip(projections[:, :, slice_idx], axis=1).copy()
            
            print(f"Processing slice {slice_idx}: sinogram shape = {prj_slice.shape}")

            # --- Create and save Low-Resolution Sinogram (X_s) ---
            sino_hr = torch.tensor(prj_slice, dtype=torch.float32)  # Shape: (angles, detectors)
            sino_lr = sino_hr[::config.SCALE, :]  # Downsample angles by SCALE factor
            
            print(f"HR sinogram shape: {sino_hr.shape}, LR sinogram shape: {sino_lr.shape}")
            
            save_to_tiff_stack(sino_lr.numpy(), LR_SINO_DIR / f"{scan_id}_sino_lr.tif")

            # --- Create High-Resolution Target Image (Y) with CORRECTED Angles ---
            vox_scaling = 1 / config.VOXEL_SIZE
            
            # Generate correct angles for the HIGH-RESOLUTION sinogram
            hr_angle_count = sino_hr.shape[0]  # Use actual HR sinogram angle count
            
            # Generate uniform angle distribution
            # Use the actual angles from the metadata, not a generic linspace
            if len(metadata['angles']) < hr_angle_count:
                raise ValueError(f"Metadata for {projection_id} has fewer angles ({len(metadata['angles'])}) than the projection data ({hr_angle_count}).")
                
            angles_rad = np.array(metadata['angles'])[:hr_angle_count] + (np.pi / 2)
            print(f"Generated {len(angles_rad)} angles for HR reconstruction")
            
            # Create Radon operator with HR angles
            radon_op = RadonFanbeam(
                config.IMAGE_SIZE, 
                angles_rad,
                source_distance=vox_scaling * metadata['dso'],
                det_distance=vox_scaling * metadata['ddo'],
                det_count=sino_hr.shape[1],
                det_spacing=vox_scaling * metadata['du'],
                clip_to_circle=False
            )
            
            sino_hr_scaled = sino_hr * vox_scaling
            
            with torch.no_grad():
                filtered_sino = radon_op.filter_sinogram(sino_hr_scaled.cuda(), config.FBP_FILTER) 
                y_target = radon_op.backprojection(filtered_sino).cpu().numpy()
            
            hu0 = metadata['hu_factor']
            y_target_hu = 1000 * ((y_target - hu0) / hu0)
            
            save_to_tiff_stack(y_target_hu, TARGET_RECO_DIR / f"{scan_id}_reco_hr.tif")

            # --- Save CORRECTED Metadata ---
            sinogram_metadata = metadata.copy()
            
            # Store the complete HR angle array and correct counts
            sinogram_metadata['angles'] = angles_rad.tolist()  # Complete HR angle array
            sinogram_metadata['rotview'] = hr_angle_count      # Actual HR angle count
            sinogram_metadata['lr_angles_count'] = sino_lr.shape[0]  # LR angle count
            sinogram_metadata['hr_angles_count'] = hr_angle_count     # HR angle count
            
            # Additional metadata
            sinogram_metadata['projection_id'] = projection_id
            sinogram_metadata['sinogram_index'] = slice_idx
            sinogram_metadata['total_sinograms'] = num_slices
            sinogram_metadata['scale_factor'] = config.SCALE
            sinogram_metadata['sinogram_shape_hr'] = list(sino_hr.shape)  # (angles, detectors)
            sinogram_metadata['sinogram_shape_lr'] = list(sino_lr.shape)  # (angles//scale, detectors)
            
            with open(METADATA_DIR / f"{scan_id}_meta.json", 'w') as f:
                json.dump(sinogram_metadata, f, indent=4)
            
            total_sinograms_processed += 1
            
            # Break after first slice for debugging
            if slice_idx == 0:
                print(f"Debug: First slice processed successfully")
                print(f"  HR sinogram: {sino_hr.shape}")
                print(f"  LR sinogram: {sino_lr.shape}")
                print(f"  Angles generated: {len(angles_rad)}")
                print(f"  Target image: {y_target_hu.shape}")

    print(f"\nPre-processing complete!")
    print(f"Total sinograms processed: {total_sinograms_processed}")
    print(f"LR Sinograms saved to: {LR_SINO_DIR}")
    print(f"HR Reconstructions saved to: {TARGET_RECO_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")


if __name__ == "__main__":
    main()