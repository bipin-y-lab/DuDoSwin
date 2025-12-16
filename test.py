# /cluster/du78sywa/model_dual_domain/dudo_simple/test_model.py (Enhanced with Std Dev & Inference Time and LR-FBP baseline)

import argparse
import time
import importlib
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np

# --- New imports for additional metrics ---
try:
    import lpips
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
except ImportError:
    print("Please install required metrics libraries: pip install torchmetrics lpips")
    exit()

# --- Project-specific imports ---
import config
from dataloader.dataset import DualDomainDataset
from model.dual_domain_model import DualDomainSwin
from utils import calculate_psnr, count_parameters, save_to_tiff_stack
from model.reconstruction import DifferentiableFBP


def normalize_for_lpips(tensor: torch.Tensor, hu_window=(-1000, 3000)):
    """
    Clips a tensor to a Hounsfield Unit (HU) window and normalizes it to the [-1, 1] range
    required by the LPIPS metric.
    """
    lo, hi = hu_window
    tensor = torch.clamp(tensor, lo, hi)
    tensor = (tensor - lo) / (hi - lo)  # Normalize to [0, 1]
    return tensor * 2.0 - 1.0           # Scale to [-1, 1]


def calculate_metrics(pred, target, device):
    """Calculate all metrics for a prediction-target pair"""
    # Initialize metrics
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    ssim_metric = SSIM(data_range=4000.0, kernel_size=11, sigma=1.5, reduction='elementwise_mean').to(device)
    
    psnr_val = calculate_psnr(pred, target)
    # Ensure PSNR is a Python float
    if isinstance(psnr_val, torch.Tensor):
        psnr_val = psnr_val.item()
    
    # Calculate SSIM with error handling
    try:
        if torch.isnan(pred).any() or torch.isnan(target).any():
            ssim_val = 0.0
        elif torch.isinf(pred).any() or torch.isinf(target).any():
            ssim_val = 0.0
        else:
            pred_float = pred.float()
            target_float = target.float()
            ssim_tensor = ssim_metric(pred_float, target_float)
            ssim_val = ssim_tensor.item() if hasattr(ssim_tensor, 'item') else float(ssim_tensor)
            if np.isnan(ssim_val) or np.isinf(ssim_val):
                ssim_val = 0.0
    except Exception as e:
        ssim_val = 0.0
        print(f"Warning: SSIM exception: {str(e)}")
    
    # Calculate LPIPS
    pred_lpips = normalize_for_lpips(pred)
    target_lpips = normalize_for_lpips(target)
    lpips_val = lpips_metric(pred_lpips, target_lpips).item()
    
    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "lpips": float(lpips_val)
    }


def evaluate_dudo_swin(model, loader, device, output_dir):
    """Evaluate the DUDO-SWIN model"""
    model.eval()
    all_metrics = []
    inference_times = []
    
    # Create output directories
    sino_dir = output_dir / "sinograms"
    pred_dir = output_dir / "predicted_reconstructions"
    target_dir = output_dir / "target_reconstructions"
    
    for d in [sino_dir, pred_dir, target_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating DUDO-SWIN model...")
    
    # Warmup runs (exclude from timing)
    print("Performing warmup runs...")
    with torch.no_grad():
        for i, (x_s, y_target, metadata) in enumerate(loader):
            if i >= 3:  # 3 warmup iterations
                break
            x_s = x_s.to(device)
            with autocast():
                _ = model(x_s, metadata)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="[DUDO-SWIN Testing]")
        for x_s, y_target, metadata in pbar:
            x_s, y_target = x_s.to(device), y_target.to(device)
            scan_id = metadata['scan_id'][0]

            # Measure inference time
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            start_time = time.time()
            
            with autocast():
                y_pred = model(x_s, metadata)
            
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)

            # Calculate metrics
            metrics = calculate_metrics(y_pred, y_target, device)
            all_metrics.append(metrics)
            
            pbar.set_postfix(
                psnr=f"{metrics['psnr']:.2f}", 
                ssim=f"{metrics['ssim']:.4f}",
                time=f"{inference_time:.2f}ms"
            )
            
            # Save outputs
            save_to_tiff_stack(x_s.squeeze().cpu().numpy(), sino_dir / f"{scan_id}_sino_lr.tif")
            save_to_tiff_stack(y_pred.squeeze().cpu().numpy(), pred_dir / f"{scan_id}_pred.tif")
            save_to_tiff_stack(y_target.squeeze().cpu().numpy(), target_dir / f"{scan_id}_target.tif")

    return all_metrics, inference_times


def evaluate_baseline_method(method_name, loader, device, output_dir):
    """Evaluate bicubic, bilinear interpolation, or direct LR FBP methods"""
    all_metrics = []
    inference_times = []
    fbp = DifferentiableFBP()
    
    # Create output directories
    pred_dir = output_dir / "predicted_reconstructions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    is_interpolation = method_name in ['bicubic', 'bilinear']
    interpolation_mode = method_name if is_interpolation else None
    
    print(f"Evaluating {method_name.upper()} baseline method...")
    
    # Warmup runs
    print("Performing warmup runs...")
    with torch.no_grad():
        for i, (x_s, y_target, metadata) in enumerate(loader):
            if i >= 3:
                break
            x_s = x_s.to(device)
            
            with autocast():
                if is_interpolation:
                    target_sino_height = x_s.shape[2] * config.SCALE
                    x_s_upsampled = F.interpolate(
                        x_s, size=(target_sino_height, x_s.shape[3]),
                        mode=interpolation_mode, align_corners=True
                    )
                    _ = fbp(x_s_upsampled, metadata)
                else: # 'lr_fbp'
                    # FBP reconstruction on the low-resolution sinogram
                    _ = fbp(x_s, metadata)

            if 'cuda' in str(device):
                torch.cuda.synchronize()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{method_name.upper()} Testing]")
        for x_s, y_target, metadata in pbar:
            x_s, y_target = x_s.to(device), y_target.to(device)
            scan_id = metadata['scan_id'][0]

            # Measure inference time
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            start_time = time.time()
            
            with autocast():
                if is_interpolation:
                    # Upsample sinogram using interpolation
                    target_sino_height = x_s.shape[2] * config.SCALE
                    
                    x_s_upsampled = F.interpolate(
                        x_s, 
                        size=(target_sino_height, x_s.shape[3]),
                        mode=interpolation_mode, 
                        align_corners=True
                    )
                    
                    # FBP reconstruction
                    y_pred = fbp(x_s_upsampled, metadata)
                else: # 'lr_fbp'
                    # FBP reconstruction on the low-resolution sinogram
                    y_pred = fbp(x_s, metadata)
            
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            inference_times.append(inference_time)

            # Calculate metrics
            metrics = calculate_metrics(y_pred, y_target, device)
            all_metrics.append(metrics)
            
            pbar.set_postfix(
                psnr=f"{metrics['psnr']:.2f}", 
                ssim=f"{metrics['ssim']:.4f}",
                time=f"{inference_time:.2f}ms"
            )
            
            # Save prediction
            save_to_tiff_stack(y_pred.squeeze().cpu().numpy(), pred_dir / f"{scan_id}_pred.tif")

    return all_metrics, inference_times


def save_method_results(method_name, metrics_list, inference_times, output_dir, scale):
    """Save results for a specific method"""
    # Calculate average and std metrics
    avg_metrics = {
        "psnr_mean": np.mean([m['psnr'] for m in metrics_list]),
        "psnr_std": np.std([m['psnr'] for m in metrics_list]),
        "ssim_mean": np.mean([m['ssim'] for m in metrics_list]),
        "ssim_std": np.std([m['ssim'] for m in metrics_list]),
        "lpips_mean": np.mean([m['lpips'] for m in metrics_list]),
        "lpips_std": np.std([m['lpips'] for m in metrics_list]),
        "inference_time_mean": np.mean(inference_times),
        "inference_time_std": np.std(inference_times)
    }
    
    # Print results
    print(f"\n--- {method_name.upper()} Results ---")
    print(f"  Average PSNR:         {avg_metrics['psnr_mean']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
    print(f"  Average SSIM:         {avg_metrics['ssim_mean']:.4f} ± {avg_metrics['ssim_std']:.4f}")
    print(f"  Average LPIPS:        {avg_metrics['lpips_mean']:.4f} ± {avg_metrics['lpips_std']:.4f}")
    print(f"  Inference Time:       {avg_metrics['inference_time_mean']:.2f} ± {avg_metrics['inference_time_std']:.2f} ms")
    
    # Save detailed CSV with per-sample results
    detailed_csv_path = output_dir / f"detailed_results_scale_{scale}.csv"
    with open(detailed_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["sample_idx", "psnr", "ssim", "lpips", "inference_time_ms"]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        for idx, (metrics, inf_time) in enumerate(zip(metrics_list, inference_times)):
            csv_writer.writerow({
                "sample_idx": idx,
                "psnr": metrics['psnr'],
                "ssim": metrics['ssim'],
                "lpips": metrics['lpips'],
                "inference_time_ms": inf_time
            })
    
    # Save summary CSV
    csv_path = output_dir / f"summary_results_scale_{scale}.csv"
    summary_data = {
        "method": method_name,
        "scale": scale,
        "psnr_mean": avg_metrics['psnr_mean'],
        "psnr_std": avg_metrics['psnr_std'],
        "ssim_mean": avg_metrics['ssim_mean'],
        "ssim_std": avg_metrics['ssim_std'],
        "lpips_mean": avg_metrics['lpips_mean'],
        "lpips_std": avg_metrics['lpips_std'],
        "inference_time_mean_ms": avg_metrics['inference_time_mean'],
        "inference_time_std_ms": avg_metrics['inference_time_std'],
        "num_samples": len(metrics_list)
    }
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=summary_data.keys())
        csv_writer.writeheader()
        csv_writer.writerow(summary_data)
    
    # Save summary text
    summary_path = output_dir / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Method: {method_name.upper()}\n")
        f.write(f"Scale Factor: {scale}x\n")
        f.write(f"Test Samples: {len(metrics_list)}\n\n")
        f.write(f"--- Average Metrics (Mean ± Std) ---\n")
        f.write(f"PSNR:            {avg_metrics['psnr_mean']:.2f} ± {avg_metrics['psnr_std']:.2f} dB\n")
        f.write(f"SSIM:            {avg_metrics['ssim_mean']:.4f} ± {avg_metrics['ssim_std']:.4f}\n")
        f.write(f"LPIPS:           {avg_metrics['lpips_mean']:.4f} ± {avg_metrics['lpips_std']:.4f}\n")
        f.write(f"Inference Time:  {avg_metrics['inference_time_mean']:.2f} ± {avg_metrics['inference_time_std']:.2f} ms\n")
    
    return avg_metrics


def create_comparison_csv(all_results, scale_output_dir, scale):
    """Create a comparison CSV file with all methods"""
    comparison_path = scale_output_dir / f"comparison_scale_{scale}x.csv"
    
    # Prepare data for CSV
    comparison_data = []
    for method_name, metrics in all_results.items():
        comparison_data.append({
            "method": method_name,
            "scale": scale,
            "psnr_mean": metrics['psnr_mean'],
            "psnr_std": metrics['psnr_std'],
            "ssim_mean": metrics['ssim_mean'],
            "ssim_std": metrics['ssim_std'],
            "lpips_mean": metrics['lpips_mean'],
            "lpips_std": metrics['lpips_std'],
            "inference_time_mean_ms": metrics['inference_time_mean'],
            "inference_time_std_ms": metrics['inference_time_std']
        })
    
    # Sort by PSNR (descending)
    comparison_data.sort(key=lambda x: x['psnr_mean'], reverse=True)
    
    with open(comparison_path, 'w', newline='') as csvfile:
        fieldnames = ["method", "scale", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", 
                     "lpips_mean", "lpips_std", "inference_time_mean_ms", "inference_time_std_ms"]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(comparison_data)
    
    # Create LaTeX-friendly table
    latex_path = scale_output_dir / f"latex_table_scale_{scale}x.txt"
    with open(latex_path, 'w') as f:
        f.write("% LaTeX table for research paper\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison of CT sinogram super-resolution methods}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Method & PSNR (dB) & SSIM & LPIPS & Time (ms) \\\\\n")
        f.write("\\hline\n")
        
        for data in comparison_data:
            f.write(f"{data['method'].upper().replace('_', '-')} & "
                   f"{data['psnr_mean']:.2f}$\\pm${data['psnr_std']:.2f} & "
                   f"{data['ssim_mean']:.3f}$\\pm${data['ssim_std']:.3f} & "
                   f"{data['lpips_mean']:.3f}$\\pm${data['lpips_std']:.3f} & "
                   f"{data['inference_time_mean_ms']:.1f}$\\pm${data['inference_time_std_ms']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Comparison CSV saved to: {comparison_path}")
    print(f"LaTeX table saved to: {latex_path}")
    print("\nRanking by PSNR (higher is better):")
    for i, data in enumerate(comparison_data, 1):
        print(f"{i}. {data['method'].upper().replace('_', '-'):12} - "
              f"PSNR: {data['psnr_mean']:.2f}±{data['psnr_std']:.2f} dB, "
              f"SSIM: {data['ssim_mean']:.4f}±{data['ssim_std']:.4f}, "
              f"LPIPS: {data['lpips_mean']:.4f}±{data['lpips_std']:.4f}, "
              f"Time: {data['inference_time_mean_ms']:.2f}±{data['inference_time_std_ms']:.2f} ms")


def main():
    print(f"Device: {config.DEVICE}")
    print(f"Model directory: {config.MODEL_SAVE_DIR}")

    # Setup data
    # Setup data - Use PATIENT-LEVEL split (must match train.py)
    meta_dir = config.BASE_DIR / "data_preprocessed" / f"scale_{config.SCALE}x" / "metadata"
    all_file_ids = sorted([p.stem.replace('_meta', '') for p in meta_dir.glob("*.json")])

    if not all_file_ids:
        print(f"ERROR: No pre-processed data found in {meta_dir}")
        return

    # Extract unique patient IDs (e.g., L014, L056, etc.)
    patient_ids = sorted(list(set([fid.split('_sino_')[0] for fid in all_file_ids])))
    print(f"Found {len(patient_ids)} patients: {patient_ids}")
    print(f"Total slices across all patients: {len(all_file_ids)}")

    # Use the SAME patient split as in train.py
    # Testing uses patients 8, 9, 10 (indices 7, 8, 9) - includes validation patient
    if len(patient_ids) != 10:
        print(f"Warning: Expected 10 patients but found {len(patient_ids)}")

    test_patients = patient_ids[7:]  # 8th, 9th, 10th patients (includes validation patient)

    print(f"\nTest patients (3): {test_patients}")

    # Get all slice IDs for test patients
    test_ids = [fid for fid in all_file_ids if fid.split('_sino_')[0] in test_patients]

    print(f"Using {len(test_ids)} slices from {len(test_patients)} patients for testing.")
    print(f"Slices per patient: {len(test_ids) // len(test_patients)} (average)")

    test_ds = DualDomainDataset(test_ids)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Create main output directory structure
    timestamp = int(time.time())
    scale_output_dir = config.BASE_DIR_PROJECT / "test_results" / f"scale_{config.SCALE}x_{timestamp}"
    scale_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create method-specific directories
    dudo_swin_dir = scale_output_dir / "dudo_swin"
    bicubic_dir = scale_output_dir / "bicubic"
    bilinear_dir = scale_output_dir / "bilinear"
    lr_fbp_dir = scale_output_dir / "lr_fbp" # NEW FBP baseline directory
    
    all_results = {}

    # 1. Evaluate DUDO-SWIN model
    sample_lr_sino, _, _ = test_ds[0]
    lr_sino_shape = sample_lr_sino.shape[1:]
    model = DualDomainSwin(lr_sino_shape=lr_sino_shape).to(config.DEVICE)
    
    model_dir = config.MODEL_SAVE_DIR
    if not model_dir.exists():
        print(f"ERROR: Model directory does not exist: {model_dir}")
        print(f"Please ensure you have trained a model for scale={config.SCALE}x.")
        return

    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    if not model_files:
        print(f"ERROR: No model file (.pth or .pt) found in the directory: {model_dir}")
        return
    
    # Select best model
    best_model_path = None
    if len(model_files) > 1:
        best_files = [f for f in model_files if 'best' in f.name.lower()]
        best_model_path = best_files[0] if best_files else model_files[0]
    else:
        best_model_path = model_files[0]
    
    print(f"Loading model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    print(f"Model loaded successfully. Trainable parameters: {count_parameters(model):,}")
    
    # Evaluate DUDO-SWIN
    dudo_swin_dir.mkdir(parents=True, exist_ok=True)
    dudo_metrics, dudo_times = evaluate_dudo_swin(model, test_loader, config.DEVICE, dudo_swin_dir)
    all_results["dudo_swin"] = save_method_results("dudo_swin", dudo_metrics, dudo_times, dudo_swin_dir, config.SCALE)
    
    # 2. Evaluate Bicubic interpolation (Baseline: Bicubic-SR + FBP)
    bicubic_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Changed function call from evaluate_interpolation_method to evaluate_baseline_method
    bicubic_metrics, bicubic_times = evaluate_baseline_method("bicubic", test_loader, config.DEVICE, bicubic_dir)
    all_results["bicubic"] = save_method_results("bicubic", bicubic_metrics, bicubic_times, bicubic_dir, config.SCALE)
    
    # 3. Evaluate Bilinear interpolation (Baseline: Bilinear-SR + FBP)
    bilinear_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Changed function call from evaluate_interpolation_method to evaluate_baseline_method
    bilinear_metrics, bilinear_times = evaluate_baseline_method("bilinear", test_loader, config.DEVICE, bilinear_dir)
    all_results["bilinear"] = save_method_results("bilinear", bilinear_metrics, bilinear_times, bilinear_dir, config.SCALE)

    # 4. Evaluate Low-Resolution FBP (Baseline: LR-FBP)
    lr_fbp_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Added new evaluation for the requested FBP comparison
    lr_fbp_metrics, lr_fbp_times = evaluate_baseline_method("lr_fbp", test_loader, config.DEVICE, lr_fbp_dir)
    all_results["lr_fbp"] = save_method_results("lr_fbp", lr_fbp_metrics, lr_fbp_times, lr_fbp_dir, config.SCALE)
    
    # 5. Create comparison CSV and LaTeX table
    create_comparison_csv(all_results, scale_output_dir, config.SCALE)
    
    print(f"\n=== ALL EVALUATIONS COMPLETE ===")
    print(f"All results saved under: {scale_output_dir}")
    print("Directory structure:")
    print(f"├── dudo_swin/")
    print(f"│   ├── summary_results_scale_{config.SCALE}.csv")
    print(f"│   ├── detailed_results_scale_{config.SCALE}.csv")
    print(f"│   └── summary_results.txt")
    print(f"├── bicubic/")
    print(f"├── bilinear/")
    print(f"├── lr_fbp/") # Added LR-FBP directory
    print(f"├── comparison_scale_{config.SCALE}x.csv")
    print(f"└── latex_table_scale_{config.SCALE}x.txt")


if __name__ == "__main__":
    main()