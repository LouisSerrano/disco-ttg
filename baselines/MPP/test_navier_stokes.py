import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys

sys.path.append("/mnt/home/lserrano/disco-ttg/")
sys.path.append("/mnt/home/lserrano/disco-ttg/test_time_compute")

from ttc_utils import (
    save_results,
    DEVICE
)

from baselines.MPP.train_2d import MPPLightning
from src.utils.euler_ns_dataset import NavierStokesDatasetWrapper
from einops import rearrange
from src.utils.database import RelativeL2


def load_model_from_checkpoint(checkpoint_path):
    """Load MPP model from Lightning checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        lit_model = MPPLightning.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        lit_model.eval()

        model = lit_model.model.to(DEVICE)
        model.eval()

        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, lit_model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def test_direct_prediction_mpp(model, test_loader, n_output_frames=16):
    """Test direct prediction for MPP model (2D Navier-Stokes - single channel vorticity)"""
    model.eval()
    rel_loss = RelativeL2()

    all_errors = []
    total_samples = 0
    total_error = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_data = batch['input'].to(DEVICE)
            target = batch['output'].to(DEVICE)

            # Rearrange input for AViT model (2D)
            # input shape: (B, T, C, H, W) -> (T, B, C, H, W)
            input_data = rearrange(input_data, "b t c h w -> t b c h w")

            # Create labels and boundary conditions tensors
            # For Navier-Stokes, we have 1 channel (vorticity)
            batch_size = input_data.size(1)
            labels = torch.tensor([[0]], device=DEVICE)  # Single channel
            bcs = torch.tensor([[1, 1]], device=DEVICE)

            predictions = []
            # Direct prediction (autoregressive rollout)
            for _ in range(n_output_frames):
                pred = model(input_data, labels, bcs)
                predictions.append(pred)
                input_data = torch.cat([input_data[1:], pred.unsqueeze(0)], axis=0)

            predictions = torch.stack(predictions, axis=1)

            error = rel_loss(predictions, target)
            all_errors.append(error)
            total_error += error * batch_size
            total_samples += batch_size

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Error = {error.item():.6f}")

    avg_error = total_error / total_samples
    return avg_error, None


def main():
    parser = argparse.ArgumentParser(description='Test MPP baseline for Navier-Stokes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--file_dir', type=str, default='/mnt/home/lserrano/ceph/data/euler_ns_short/',
                        help='Path to data directory')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPU files')
    parser.add_argument('--n_input_frames', type=int, default=16, help='Number of input frames')
    parser.add_argument('--n_output_frames', type=int, default=16, help='Number of output frames')
    parser.add_argument('--vorticity_scale', type=float, default=10.0, help='Vorticity scale factor')
    parser.add_argument('--N_ns_ics', type=int, default=512, help='Number of ICs per viscosity')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading MPP model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return

    # Load test dataset (Navier-Stokes)
    print("\nLoading test dataset (Navier-Stokes)...")
    test_ds = NavierStokesDatasetWrapper(
        file_dir=args.file_dir,
        num_gpus=args.num_gpus,
        input_frames=args.n_input_frames,
        output_frames=args.n_output_frames,
        sub_x=1,
        sub_t=1,
        N_ns_ics=args.N_ns_ics,
        vorticity_scale=args.vorticity_scale
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    print(f"Test dataset (Navier-Stokes): {len(test_ds)} samples")

    # Test direct prediction
    print("\nTesting MPP direct prediction on Navier-Stokes...")
    start_time = time.time()

    direct_error, _ = test_direct_prediction_mpp(model, test_loader, n_output_frames=args.n_output_frames)

    direct_time = time.time() - start_time

    results = {
        'equation_type': 'navier_stokes',
        'model_type': 'MPP',
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_ds),
        'n_input_frames': args.n_input_frames,
        'n_output_frames': args.n_output_frames,
        'vorticity_scale': args.vorticity_scale,
        'direct_prediction': {
            'error': float(direct_error.item()) if hasattr(direct_error, 'item') else float(direct_error),
            'time': direct_time
        }
    }

    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR NAVIER-STOKES MPP BASELINE:")
    error_val = direct_error.item() if hasattr(direct_error, 'item') else direct_error
    print(f"Direct prediction: {error_val:.6f} (time: {direct_time:.2f}s)")
    print(f"Total samples evaluated: {len(test_ds)}")
    print("="*50)

    # Save results
    output_file = os.path.join(args.output_dir, f"navier_stokes_mpp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()
