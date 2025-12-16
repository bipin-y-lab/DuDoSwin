# /cluster/du78sywa/config.py

from pathlib import Path
import torch

# --------------------------------------------------------------------------
#  I. I/O and General Setup
# --------------------------------------------------------------------------

# Base directory
BASE_DIR = Path("/cluster/du78sywa/")


# Base directory for the project.
BASE_DIR_PROJECT = Path("/cluster/du78sywa/model_dual_domain/dudo_simple")

# Directory where your raw projection TIFFs are stored
# e.g., /cluster/du78sywa/data/projections/L014_hd_proj_fan_geometry.tif
PROJ_DIR = BASE_DIR / "data" / "projections"

# Device to run the training on. Use "cuda" for GPU or "cpu" for CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
#  II. Super-Resolution & Reconstruction Parameters
# --------------------------------------------------------------------------

# The super-resolution factor. A scale of 4 means we use 1/4 of the projection angles.
# This is the primary setting for your experiments.
SCALE = 4

# The final size of the reconstructed image (Height x Width in pixels).
IMAGE_SIZE = 256

# The physical size of a voxel, used for the Radon transform operator.
ORIGINAL_IMAGE_SIZE = 512  # default working size
ORIGINAL_VOXEL_SIZE = 0.7  # default value
VOXEL_SIZE = ORIGINAL_VOXEL_SIZE * (ORIGINAL_IMAGE_SIZE / IMAGE_SIZE)

# The filter to be used for the Filtered Back-Projection (FBP).
# 'hann' is a common choice.
FBP_FILTER = "hann"


# --------------------------------------------------------------------------
#  III. Dual-Domain Model Hyperparameters
# --------------------------------------------------------------------------

# The embedding dimension used throughout the Swin Transformer blocks.
# This is the number of channels after the initial convolution.
EMBED_DIM = 64

# Depth of the Swin Transformer blocks. This is a tuple defining the number of
# Swin layers in the (1) sinogram SR block and (2) image denoising block.
SWIN_DEPTHS = (6, 12)

# Number of attention heads in the Swin Transformer blocks.
NUM_HEADS = (4, 8)

# Window size for the Swin Transformer. Must be chosen carefully based on input size.
WINDOW_SIZE = 16

# Ratio for the MLP (feed-forward) layer expansion in Swin blocks.
MLP_RATIO = 4.0


# --------------------------------------------------------------------------
#  IV. Training Hyperparameters
# --------------------------------------------------------------------------

# Number of samples to process in one batch.
BATCH_SIZE = 1

# The learning rate for the AdamW optimizer.
LEARNING_RATE = 5e-5

# Total number of training epochs.
EPOCHS = 3

# Learning rate scheduling
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# Gradient clipping for training stability
GRAD_CLIP_NORM = 1.0

# --------------------------------------------------------------------------
#  V. Output Directories
# --------------------------------------------------------------------------
# These directories will be created automatically if they don't exist.

# Directory to save the trained model checkpoints.
MODEL_SAVE_DIR = BASE_DIR_PROJECT / f"models_dual_domain/scale_{SCALE}x"

# Directory to save training logs and validation results (e.g., metrics).
RESULTS_DIR = BASE_DIR_PROJECT / f"results_dual_domain/scale_{SCALE}x"

# Directory to save sample reconstructed images during validation/testing.
INFERENCE_OUTPUT_DIR = BASE_DIR_PROJECT / f"inference_dual_domain/scale_{SCALE}x"


print("--- Configuration Loaded ---")
print(f"Project Base Directory: {BASE_DIR}")
print(f"Super-Resolution Scale: {SCALE}x")
print(f"Device: {DEVICE}")
print(f"Model checkpoints will be saved to: {MODEL_SAVE_DIR}")
print("--------------------------")