import torch
import argparse
import os
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import numpy as np
from einops import rearrange
import copy
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

sys.path.append("/mnt/home/lserrano/disco-ball/")
sys.path.append("/mnt/home/lserrano/disco-ball/test-time-compute")

from ttc_utils import (
    save_results,
    DEVICE
)

from GEPS.train_2d import GEPSLightning
from src.utils.euler_ns_dataset import NavierStokesDatasetWrapper
from src.utils.database import RelativeL2


def load_geps_model_from_checkpoint(checkpoint_path):
    """Load GEPS model from Lightning checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        lit_model = GEPSLightning.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        lit_model.eval()
        model = lit_model.model.to(DEVICE)
        model.eval()

        logger.info(f"GEPS model loaded successfully from {checkpoint_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, lit_model

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None, None


def optimize_codes_for_trajectory(model, input_sequence, test_target, cfg, n_output_frames,
                                  n_optimization_steps=50, lr=0.001, weight_decay=0.0, n_pred=1):
    """
    Create and optimize fresh environment codes for a specific trajectory.

    Args:
        model: GEPS forecaster model
        input_sequence: Input trajectory sequence [B, T, C, H, W]
        test_target: Target sequence for testing [B, T, C, H, W]
        cfg: Model configuration
        n_output_frames: Number of output frames to predict during testing
        n_optimization_steps: Number of optimization steps
        lr: Learning rate for code optimization
        weight_decay: Weight decay for code optimization
        n_pred: Number of prediction steps to use during optimization (default: 1)

    Returns:
        optimized_codes: The optimized environment codes tensor
        test_error: Final test error
    """
    if input_sequence.shape[1] < 2:
        logger.warning("Not enough frames for code optimization")
        code_dim = cfg.model.code_c
        return torch.randn(1, code_dim, device=DEVICE, requires_grad=False), float('inf')

    n_samples = input_sequence.shape[0]
    code_dim = cfg.model.code_c
    env_codes = (torch.randn(n_samples, code_dim, device=DEVICE) * 0.01).requires_grad_(True)

    optimizer = torch.optim.Adam([env_codes], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_optimization_steps)

    rel_loss = RelativeL2()
    input_length = input_sequence.shape[1]

    logger.info(f"Optimizing codes for trajectory with {input_length} frames...")

    for step in range(n_optimization_steps):
        optimizer.zero_grad()

        max_start = input_length - n_pred - 1
        if max_start < 0:
            logger.warning(f"Not enough frames for n_pred={n_pred} predictions. Using available frames.")
            n_pred_actual = min(n_pred, input_length - 1)
            i = 0
        else:
            n_pred_actual = n_pred
            i = random.randint(0, max_start)

        current_input = input_sequence[:, i:i+1]
        target = input_sequence[:, i+1:i+1+n_pred_actual]

        # 2D case: Navier-Stokes
        current_input = rearrange(current_input, "b t c h w -> b c h w t")
        target = rearrange(target, "b t c h w -> b c h w t")

        dt = cfg.model.default_integration_time
        time_grid = torch.tensor([j * dt for j in range(n_pred_actual + 1)], device=DEVICE)

        pred = model.forward_with_codes(current_input, time_grid, env_codes)
        pred = pred[..., 1:]

        loss = rel_loss(pred, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == n_optimization_steps - 1:
            logger.debug(f"Step {step}: Loss = {loss.item():.6f}")

            with torch.no_grad():
                test_input = input_sequence[:, -1:].clone()
                model_input = rearrange(test_input, "b t c h w -> b c h w t")

                dt = cfg.model.default_integration_time
                time_grid = torch.tensor([i * dt for i in range(n_output_frames + 1)], device=DEVICE)

                pred = model.forward_with_codes(model_input, time_grid, env_codes)
                pred = pred[..., 1:]

                test_predictions = rearrange(pred, "b c h w t -> b t c h w")

                test_error = rel_loss(test_predictions, test_target)
                logger.debug(f"  Test error: {test_error.item():.6f}")

    logger.debug(f"Code optimization completed. Final training loss: {loss.item():.6f}, Final test error: {test_error.item():.6f}")
    return env_codes.detach(), test_error.item()


def test_geps_inference(model, test_loader, cfg, n_output_frames, n_optimization_steps=50,
                        lr=0.001, weight_decay=0.0, n_pred=1):
    """Test GEPS inference with code optimization for Navier-Stokes"""
    model.eval()
    rel_loss = RelativeL2()

    all_errors = []
    all_times = []
    total_error = 0
    total_samples = 0

    for batch_idx, batch in enumerate(test_loader):
        start_time = time.time()

        input_seq = batch['input'].to(DEVICE)  # [B, T, C, H, W]
        target_seq = batch['output'].to(DEVICE)  # [B, T, C, H, W]

        # Optimize codes for this batch
        optimized_codes, test_error = optimize_codes_for_trajectory(
            model, input_seq, target_seq, cfg, n_output_frames,
            n_optimization_steps, lr, weight_decay, n_pred
        )

        n_samples = input_seq.shape[0]
        total_samples += n_samples
        total_error += test_error * n_samples
        all_errors.append(test_error)

        batch_time = time.time() - start_time
        all_times.append(batch_time)

        running_avg_error = total_error / total_samples
        logger.info(f"Batch {batch_idx}: Error = {test_error:.6f}, Running Avg = {running_avg_error:.6f}, Time = {batch_time:.2f}s")

    avg_error = total_error / total_samples
    avg_time = np.mean(all_times) if all_times else 0

    return avg_error, avg_time


def test_direct_prediction(model, test_loader, cfg, n_output_frames):
    """Test direct prediction without code optimization (using environment index 0)"""
    model.eval()
    rel_loss = RelativeL2()

    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_seq = batch['input'].to(DEVICE)
            target_seq = batch['output'].to(DEVICE)

            batch_size = input_seq.shape[0]

            # Use last frame as starting point
            test_input = input_seq[:, -1:].clone()
            model_input = rearrange(test_input, "b t c h w -> b c h w t")

            # Create time grid
            dt = cfg.model.default_integration_time
            time_grid = torch.tensor([i * dt for i in range(n_output_frames + 1)], device=DEVICE)

            # Use environment index 0 (Euler) for all samples
            env_idx = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

            pred = model(model_input, time_grid, env_idx)
            pred = pred[..., 1:]  # Remove initial timepoint

            predictions = rearrange(pred, "b c h w t -> b t c h w")

            error = rel_loss(predictions, target_seq)
            total_error += error.item() * batch_size
            total_samples += batch_size

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: Error = {error.item():.6f}")

    avg_error = total_error / total_samples
    return avg_error


def main():
    parser = argparse.ArgumentParser(description='Test GEPS baseline for Navier-Stokes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to GEPS model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./GEPS/results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--n_optimization_steps', type=int, default=500, help='Steps for code optimization')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for code optimization')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for code optimization')
    parser.add_argument('--n_pred', type=int, default=1, help='Number of predictions to use during optimization')
    parser.add_argument('--file_dir', type=str, default='/mnt/home/lserrano/ceph/data/euler_ns_short/',
                        help='Path to data directory')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPU files')
    parser.add_argument('--n_input_frames', type=int, default=16, help='Number of input frames')
    parser.add_argument('--n_output_frames', type=int, default=16, help='Number of output frames')
    parser.add_argument('--vorticity_scale', type=float, default=10.0, help='Vorticity scale factor')
    parser.add_argument('--N_ns_ics', type=int, default=512, help='Number of ICs per viscosity')
    parser.add_argument('--mode', type=str, default='optimize', choices=['optimize', 'direct'],
                        help='Testing mode: optimize (code optimization) or direct (env index 0)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading GEPS model...")
    model, lit_model = load_geps_model_from_checkpoint(args.model_path)
    if model is None:
        logger.error("Failed to load model")
        return

    cfg = lit_model.cfg

    # Load test dataset (Navier-Stokes)
    logger.info("Loading test dataset (Navier-Stokes)...")
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

    logger.info(f"Test dataset (Navier-Stokes): {len(test_ds)} samples")

    # Run testing
    logger.info(f"Testing GEPS on Navier-Stokes (mode: {args.mode})...")
    start_time = time.time()

    if args.mode == 'optimize':
        geps_error, geps_time = test_geps_inference(
            model, test_loader, cfg, args.n_output_frames,
            args.n_optimization_steps, args.lr, args.weight_decay, args.n_pred
        )
    else:
        geps_error = test_direct_prediction(model, test_loader, cfg, args.n_output_frames)
        geps_time = 0

    total_time = time.time() - start_time

    results = {
        'equation_type': 'navier_stokes',
        'model_type': 'GEPS',
        'mode': args.mode,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_ds),
        'n_input_frames': args.n_input_frames,
        'n_output_frames': args.n_output_frames,
        'vorticity_scale': args.vorticity_scale,
        'n_optimization_steps': args.n_optimization_steps if args.mode == 'optimize' else None,
        'lr': args.lr if args.mode == 'optimize' else None,
        'geps_inference': {
            'error': float(geps_error),
            'avg_time_per_batch': float(geps_time) if args.mode == 'optimize' else None,
            'total_time': total_time
        }
    }

    # Summary
    logger.info("=" * 50)
    logger.info(f"GEPS NAVIER-STOKES RESULTS (mode: {args.mode}):")
    logger.info(f"Average Error: {geps_error:.6f}")
    if args.mode == 'optimize':
        logger.info(f"Average Time per Batch: {geps_time:.2f}s")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Total samples evaluated: {len(test_ds)}")
    logger.info("=" * 50)

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"navier_stokes_geps_{args.mode}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_results(results, output_file)


if __name__ == "__main__":
    main()
