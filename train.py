# /cluster/du78sywa/model_dual_domain/dudo_simple/train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.empty_cache()

import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Project-specific imports
import config
from dataloader.dataset import DualDomainDataset
from model.dual_domain_model import DualDomainSwin
from utils import L1L2Loss, count_parameters, save_to_tiff_stack, calculate_psnr
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Runs a single training epoch with mixed precision."""
    model.train()
    running_loss, running_psnr = 0.0, 0.0

    pbar = tqdm(loader, desc="[Train]", leave=False)
    for x_s, y_target, metadata in pbar:
        x_s, y_target = x_s.to(device), y_target.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            y_pred = model(x_s, metadata)
            loss = criterion(y_pred, y_target)

        scaler.scale(loss).backward()

        # Add Gradient Clipping for stability ---
        # Unscales the gradients before clipping to see their true norm
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
       
        scaler.step(optimizer)
        scaler.update()

        # Calculate metrics (in no_grad to save computation)
        with torch.no_grad():
            psnr = calculate_psnr(y_pred.detach(), y_target)

        running_loss += loss.item() * x_s.size(0)
        running_psnr += psnr * x_s.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f} dB")

    avg_loss = running_loss / len(loader.dataset)
    avg_psnr = running_psnr / len(loader.dataset)
    return avg_loss, avg_psnr


def validate(model, loader, criterion, device, epoch):
    """Runs a single validation epoch with mixed precision."""
    model.eval()
    running_loss, running_psnr = 0.0, 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="[Valid]", leave=False)
        for i, (x_s, y_target, metadata) in enumerate(pbar):
            x_s, y_target = x_s.to(device), y_target.to(device)

            with autocast():
                y_pred = model(x_s, metadata)
                loss = criterion(y_pred, y_target)

            psnr = calculate_psnr(y_pred, y_target)

            running_loss += loss.item() * x_s.size(0)
            running_psnr += psnr * x_s.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f} dB")

            # Save a sample image from the first batch of the first validation epoch
            if i == 0 and epoch == 1:
                if y_pred.shape[0] > 0:
                    sample_output = y_pred[0].squeeze().cpu().numpy()
                    sample_target = y_target[0].squeeze().cpu().numpy()
                    save_to_tiff_stack(sample_output, config.INFERENCE_OUTPUT_DIR / "sample_epoch1_pred.tif")
                    save_to_tiff_stack(sample_target, config.INFERENCE_OUTPUT_DIR / "sample_epoch1_target.tif")
                    print("\nSaved initial validation sample to", config.INFERENCE_OUTPUT_DIR)

    avg_loss = running_loss / len(loader.dataset)
    avg_psnr = running_psnr / len(loader.dataset)
    return avg_loss, avg_psnr


def test_model(model, loader, device, model_path):
    """Runs final evaluation on the test set with the best model."""
    print("\n--- Running Final Test Evaluation ---")
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Could not find best model at {model_path}. Skipping test.")
        return

    model.eval()
    total_psnr = 0.0
    total_mse = 0.0

    with torch.no_grad():
        for x_s, y_target, metadata in tqdm(loader, desc="[Test]"):
            x_s, y_target = x_s.to(device), y_target.to(device)

            with autocast():
                y_pred = model(x_s, metadata)

            total_psnr += calculate_psnr(y_pred, y_target).item()
            total_mse += torch.mean((y_pred - y_target) ** 2).item()

    avg_psnr = total_psnr / len(loader)
    avg_mse = total_mse / len(loader)
    print(f"Test Results -> Average PSNR: {avg_psnr:.2f} dB | Average MSE: {avg_mse:.6f}")


def main():
    """Main function to run the entire training pipeline."""
    print("--- Starting Dual-Domain Super-Resolution Training ---")
    start_time = time.time()

    # 1. Create Output Directories
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config.INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Find Pre-processed Data and Split BY PATIENT
    meta_dir = config.BASE_DIR / "data_preprocessed" / f"scale_{config.SCALE}x" / "metadata"
    all_file_ids = sorted([p.stem.replace('_meta', '') for p in meta_dir.glob("*.json")])

    if not all_file_ids:
        print(f"Error: No pre-processed data found in {meta_dir}")
        print("Please run preprocess.py first.")
        return

    # Extract unique patient IDs (e.g., L014, L056, etc.)
    patient_ids = sorted(list(set([fid.split('_sino_')[0] for fid in all_file_ids])))
    print(f"Found {len(patient_ids)} patients: {patient_ids}")
    print(f"Total slices across all patients: {len(all_file_ids)}")

    # Split patients: 7 train, 1 validation, 2 additional test (total 3 test including val patient)
    if len(patient_ids) != 10:
        print(f"Warning: Expected 10 patients but found {len(patient_ids)}")

    train_patients = patient_ids[:7]  # First 7 patients for training
    val_patient = [patient_ids[7]]     # 8th patient for validation
    test_patients = patient_ids[7:]    # 8th, 9th, 10th patients for testing (includes val patient)

    print(f"\nPatient split:")
    print(f"  Training patients (7): {train_patients}")
    print(f"  Validation patient (1): {val_patient}")
    print(f"  Test patients (3, includes val): {test_patients}")

    # Get all slice IDs for each patient group
    train_ids = [fid for fid in all_file_ids if fid.split('_sino_')[0] in train_patients]
    val_ids = [fid for fid in all_file_ids if fid.split('_sino_')[0] in val_patient]
    test_ids = [fid for fid in all_file_ids if fid.split('_sino_')[0] in test_patients]

    print(f"\nSlice counts:")
    print(f"  Training slices: {len(train_ids)}")
    print(f"  Validation slices: {len(val_ids)}")
    print(f"  Test slices: {len(test_ids)}")
    
    # Save the patient IDs for each split to a text file
    split_info_path = config.BASE_DIR_PROJECT / "dataloader" / "data_split.txt"
    with open(split_info_path, 'w') as f:
        f.write("=== PATIENT-LEVEL SPLIT ===\n\n")
        f.write("--- Training Patients (7) ---\n")
        f.write("\n".join(train_patients))
        f.write(f"\n(Total training slices: {len(train_ids)})\n")
        
        f.write("\n\n--- Validation Patient (1) ---\n")
        f.write("\n".join(val_patient))
        f.write(f"\n(Total validation slices: {len(val_ids)})\n")
        
        f.write("\n\n--- Test Patients (3, includes validation patient) ---\n")
        f.write("\n".join(test_patients))
        f.write(f"\n(Total test slices: {len(test_ids)})\n")
        
        f.write("\n\n=== SLICE-LEVEL DETAILS ===\n\n")
        f.write("--- Training Set Slice IDs ---\n")
        f.write("\n".join(train_ids))
        f.write("\n\n--- Validation Set Slice IDs ---\n")
        f.write("\n".join(val_ids))
        f.write("\n\n--- Test Set Slice IDs ---\n")
        f.write("\n".join(test_ids))
    print(f"Data split information saved to {split_info_path}")

    # 3. Create Datasets
    train_ds = DualDomainDataset(train_ids)
    val_ds = DualDomainDataset(val_ids)

    # 4. Get LR Sinogram Shape and Initialize Model
    sample_lr_sino, _, _ = train_ds[0]
    lr_sino_shape = sample_lr_sino.shape[1:]
    model = DualDomainSwin(lr_sino_shape=lr_sino_shape).to(config.DEVICE)
    print(f"Model initialized. Trainable parameters: {count_parameters(model):,}")

    # 5. Create DataLoaders with num_workers=0
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Initialize Loss, Optimizer, and Scaler
    criterion = L1L2Loss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    ) if config.USE_SCHEDULER else None
    scaler = GradScaler()

    # 7. Training Loop
    best_val_loss = float('inf')
    best_model_path = config.MODEL_SAVE_DIR / "best_model.pth"

    print("\n--- Beginning Training Loop ---")
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, scaler)
        val_loss, val_psnr = validate(model, val_loader, criterion, config.DEVICE, epoch)
        
        if scheduler:
            scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} | "
              f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} | "
              f"Duration: {epoch_duration:.2f}s")

        # Checkpoint the best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Loss: {best_val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB)")

    total_duration = time.time() - start_time
    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_duration / 3600:.2f} hours")
    print(f"The best model is saved at: {best_model_path}")


if __name__ == "__main__":
    main()