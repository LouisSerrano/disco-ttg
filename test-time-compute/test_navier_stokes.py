import torch
import argparse
import os
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import numpy as np

# Configure logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ttc_utils import (
    save_results,
    DEVICE
)
from ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    random_operator_selection_batch,
    gradient_selection_multi_operator,
    beam_search_operator_selection,
    beam_search_operator_selection_batch
)

from train.train_euler_diffusion_aggregate import DISCOLitModule
from src.utils.euler_ns_dataset import EulerDiffusionDatasetWrapper, NavierStokesDatasetWrapper


def plot_vorticity_predictions(input_seq, target_seq, pred_seq, save_path, sample_idx=0):
    """Plot 2D vorticity predictions comparing ground truth and prediction.

    Args:
        input_seq: Input sequence (T_in, C, H, W) or (B, T_in, C, H, W)
        target_seq: Target sequence (T_out, C, H, W) or (B, T_out, C, H, W)
        pred_seq: Prediction sequence (T_out, C, H, W) or (B, T_out, C, H, W)
        save_path: Path to save the figure
        sample_idx: Which sample to plot if batched
    """
    # Handle batched input
    if len(input_seq.shape) == 5:
        input_seq = input_seq[sample_idx]
        target_seq = target_seq[sample_idx]
        pred_seq = pred_seq[sample_idx]

    # Convert to numpy
    if torch.is_tensor(input_seq):
        input_seq = input_seq.cpu().numpy()
    if torch.is_tensor(target_seq):
        target_seq = target_seq.cpu().numpy()
    if torch.is_tensor(pred_seq):
        pred_seq = pred_seq.cpu().numpy()

    # Get dimensions
    n_output_frames = target_seq.shape[0]
    n_cols = min(4, n_output_frames)  # Show up to 4 time steps
    time_indices = np.linspace(0, n_output_frames - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10), constrained_layout=True)

    # Find global vmin/vmax for consistent colorbar across ground truth and prediction
    vmin = min(target_seq.min(), pred_seq.min())
    vmax = max(target_seq.max(), pred_seq.max())

    for col, t_idx in enumerate(time_indices):
        # Ground truth
        ax = axes[0, col]
        im = ax.imshow(target_seq[t_idx, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'GT t={t_idx}', fontsize=10)
        ax.axis('off')

        # Prediction
        ax = axes[1, col]
        ax.imshow(pred_seq[t_idx, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'Pred t={t_idx}', fontsize=10)
        ax.axis('off')

        # Error
        ax = axes[2, col]
        error = np.abs(target_seq[t_idx, 0] - pred_seq[t_idx, 0])
        im_err = ax.imshow(error, cmap='hot')
        ax.set_title(f'|Err| t={t_idx}', fontsize=10)
        ax.axis('off')

    # Add colorbars with better placement
    cbar = fig.colorbar(im, ax=axes[:2, :], location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Vorticity', fontsize=10)
    
    # Error colorbar
    cbar_err = fig.colorbar(im_err, ax=axes[2, :], location='right', shrink=0.8, pad=0.02)
    cbar_err.set_label('Error', fontsize=10)

    plt.suptitle(f'Navier-Stokes Prediction (Sample {sample_idx})', fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction plot to {save_path}")


def load_model_from_checkpoint(checkpoint_path):
    """Load DISCO model from Lightning checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        lit_model = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        lit_model.eval()

        model = lit_model.model.to(DEVICE)
        model.eval()

        logger.info(f"Model loaded successfully from {checkpoint_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, lit_model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Test time compute for Navier-Stokes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_operators', type=int, default=2, help='Number of operators for gradient method')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        default=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        help='Methods to test (default: all methods)')
    parser.add_argument('--random_trials', type=int, default=100,
                        help='Number of random compositions to try per sample (default: 100)')
    parser.add_argument('--random_batch_size', type=int, default=16,
                        help='Batch size for random operator selection (default: 16)')
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--beam_batch_size', type=int, default=32,
                        help='Batch size for beam search operator selection (default: 32)')
    parser.add_argument('--min_improvement', type=float, default=1.0,
                        help='Minimum improvement threshold for greedy/beam search (default: 1.0%%)')
    parser.add_argument('--splitting_method', type=str, default='strang',
                        choices=['strang', 'lie'],
                        help='Operator splitting method: strang (2nd order) or lie (1st order) (default: strang)')
    parser.add_argument('--refinement_factor', type=int, default=1,
                        help='Number of sub-steps per dt for finer integration (default: 1)')
    parser.add_argument('--plot', action='store_true',
                        help='Save prediction plots')
    parser.add_argument('--num_plots', type=int, default=4,
                        help='Number of samples to plot (default: 4)')
    # Dataset parameters
    parser.add_argument('--file_dir', type=str, default='/mnt/home/lserrano/ceph/data/euler_ns_short/',
                        help='Path to data directory')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPU files (default: 8)')
    parser.add_argument('--n_input_frames', type=int, default=16,
                        help='Number of input frames (default: 16)')
    parser.add_argument('--n_output_frames', type=int, default=16,
                        help='Number of output frames (default: 16)')
    parser.add_argument('--vorticity_scale', type=float, default=10.0,
                        help='Scale factor for vorticity (default: 10.0)')
    parser.add_argument('--N_ns_ics', type=int, default=512,
                        help='Number of ICs per viscosity for NS dataset (default: 512)')
    parser.add_argument('--max_test_samples', type=int, default=8192,
                        help='Maximum number of test samples to evaluate (default: 8192)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        logger.error("Failed to load model")
        return

    # Get dt from model config (default integration time)
    dt = getattr(model, 'default_integration_time', 0.08)
    logger.info(f"Using dt (integration time): {dt}")

    # Create training dataset (Euler + Diffusion) for operator encoding
    logger.info("Loading training dataset (Euler + Diffusion) for operator encoding...")
    train_ds = EulerDiffusionDatasetWrapper(
        file_dir=args.file_dir,
        num_gpus=args.num_gpus,
        split='train',
        input_frames=args.n_input_frames,
        output_frames=args.n_output_frames,
        sub_x=1,
        sub_t=1,
        val_fraction=0.1,
        seed=42,
        vorticity_scale=args.vorticity_scale,
    )

    # Create test dataset (Navier-Stokes)
    logger.info("Loading test dataset (Navier-Stokes)...")
    test_ds = NavierStokesDatasetWrapper(
        file_dir=args.file_dir,
        num_gpus=args.num_gpus,
        input_frames=args.n_input_frames,
        output_frames=args.n_output_frames,
        sub_x=1,
        sub_t=1,
        N_ns_ics=args.N_ns_ics,
        vorticity_scale=args.vorticity_scale,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    logger.info(f"Train dataset: {len(train_ds)} samples")
    logger.info(f"Test dataset: {len(test_ds)} samples")

    # Encode operators using model codebook
    logger.info("Using model codebook for operators...")
    theta_latent_operators = lit_model.codebook
    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=2)

    logger.info(f"Codebook shape: {theta_latent_operators.shape}")
    logger.info(f"Decoded operators shape: {theta_operators.shape}")

    # Build operator metadata from training data
    operator_metadata = []
    num_environments = train_ds.num_environments
    for env_idx in range(num_environments):
        viscosity = train_ds.dataset.get_viscosity(env_idx)
        operator_metadata.append({
            'operator_id': env_idx,
            'equation_type': "EulerDiffusion",
            'trajectory_indices': [],
            'viscosity': viscosity,
        })
    logger.info(f"Created metadata for {len(operator_metadata)} operators")

    # Get test samples
    test_batch = next(iter(test_loader))
    test_input = test_batch['input'][:args.num_samples].to(DEVICE)
    test_target = test_batch['output'][:args.num_samples].to(DEVICE)

    # Define method registry
    def run_direct_method():
        logger.info("Testing direct prediction...")
        start_time = time.time()
        direct_error, direct_pred = test_direct_prediction(model, test_loader, dt=dt)
        direct_time = time.time() - start_time

        # Plot predictions if requested
        if args.plot:
            plot_dir = os.path.join(args.output_dir, 'plots_direct')
            os.makedirs(plot_dir, exist_ok=True)
            # Get a batch for plotting
            plot_batch = next(iter(test_loader))
            plot_input = plot_batch['input'][:args.num_plots].to(DEVICE)
            plot_target = plot_batch['output'][:args.num_plots].to(DEVICE)
            with torch.no_grad():
                state_labels = torch.tensor(list(range(plot_input.shape[2])), device=DEVICE)
                plot_pred, _ = model(plot_input, state_labels, n_future_steps=plot_target.shape[1])
            for i in range(min(args.num_plots, plot_input.shape[0])):
                plot_vorticity_predictions(
                    plot_input[i].cpu(), plot_target[i].cpu(), plot_pred[i].cpu(),
                    os.path.join(plot_dir, f'direct_sample_{i}.png'), sample_idx=0
                )

        return {
            'error': direct_error,
            'time': direct_time
        }

    def run_greedy_method():
        logger.info(f"Testing greedy operator selection (max {args.max_test_samples} samples)...")
        greedy_results = []
        start_time = time.time()

        # Evaluate over test set
        sample_idx = 0
        total_greedy_error = 0
        done = False
        for batch_idx, batch in enumerate(test_loader):
            if done:
                break
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            for i in range(batch_input.size(0)):
                if sample_idx >= args.max_test_samples:
                    done = True
                    break

                composition, error, pred = greedy_operator_selection(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    max_operators=5,
                    min_improvement_threshold=args.min_improvement,
                    dt=dt,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                
                total_greedy_error += error
                sample_idx += 1
                running_avg = total_greedy_error / sample_idx

                # Get composed viscosities
                composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

                greedy_results.append({
                    'sample_idx': sample_idx - 1,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'viscosity': composed_viscosity}
                })

                if sample_idx <= 10 or (sample_idx - 1) % 5 == 0:
                    logger.info(f"Sample {sample_idx - 1}/{args.max_test_samples}: composition {composition}, error {error:.6f}, running avg {running_avg:.6f}")

        greedy_time = time.time() - start_time
        avg_greedy_error = sum(r['error'] for r in greedy_results) / len(greedy_results) if greedy_results else 0
        logger.info(f"Greedy method evaluated on {len(greedy_results)} samples")

        # Plot predictions if requested
        if args.plot:
            plot_dir = os.path.join(args.output_dir, 'plots_greedy')
            os.makedirs(plot_dir, exist_ok=True)
            # Get a batch for plotting
            plot_batch = next(iter(test_loader))
            plot_input = plot_batch['input'][:args.num_plots].to(DEVICE)
            plot_target = plot_batch['output'][:args.num_plots].to(DEVICE)
            for i in range(min(args.num_plots, plot_input.shape[0])):
                composition, _, plot_pred = greedy_operator_selection(
                    model, theta_latent_operators,
                    plot_input[i:i+1], plot_target[i:i+1],
                    max_operators=5,
                    min_improvement_threshold=args.min_improvement,
                    dt=dt,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                plot_vorticity_predictions(
                    plot_input[i].cpu(), plot_target[i].cpu(), plot_pred[0].cpu(),
                    os.path.join(plot_dir, f'greedy_sample_{i}.png'), sample_idx=0
                )
        return {
            'avg_error': avg_greedy_error,
            'time': greedy_time,
            'details': greedy_results
        }

    def run_random_method():
        logger.info(f"Testing random operator selection (max {args.max_test_samples} samples)...")
        random_results = []
        start_time = time.time()

        # Evaluate over test set
        sample_idx = 0
        total_random_error = 0
        done = False
        logger.info("Starting sample evaluation...")
        for batch_idx, batch in enumerate(test_loader):
            if done:
                break
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            for i in range(batch_input.size(0)):
                if sample_idx >= args.max_test_samples:
                    done = True
                    break

                composition, error, pred = random_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    num_compositions=args.random_trials,
                    composition_lengths=[2, 3],
                    random_batch_size=args.random_batch_size,
                    dt=dt,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                
                total_random_error += error
                sample_idx += 1
                running_avg = total_random_error / sample_idx

                # Get composed viscosities
                composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

                random_results.append({
                    'sample_idx': sample_idx - 1,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'viscosity': composed_viscosity}
                })

                if sample_idx <= 10 or (sample_idx - 1) % 5 == 0:
                    logger.info(f"Sample {sample_idx - 1}/{args.max_test_samples}: composition {composition}, error {error:.6f}, running avg {running_avg:.6f}")

        random_time = time.time() - start_time
        avg_random_error = sum(r['error'] for r in random_results) / len(random_results) if random_results else 0
        logger.info(f"Random method evaluated on {len(random_results)} samples")

        # Plot predictions if requested
        if args.plot:
            plot_dir = os.path.join(args.output_dir, 'plots_random')
            os.makedirs(plot_dir, exist_ok=True)
            # Get a batch for plotting
            plot_batch = next(iter(test_loader))
            plot_input = plot_batch['input'][:args.num_plots].to(DEVICE)
            plot_target = plot_batch['output'][:args.num_plots].to(DEVICE)
            for i in range(min(args.num_plots, plot_input.shape[0])):
                composition, _, plot_pred = random_operator_selection_batch(
                    model, theta_latent_operators,
                    plot_input[i:i+1], plot_target[i:i+1],
                    num_compositions=args.random_trials,
                    composition_lengths=[2, 3, 4],
                    random_batch_size=args.random_batch_size,
                    dt=dt,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                plot_vorticity_predictions(
                    plot_input[i].cpu(), plot_target[i].cpu(), plot_pred[0].cpu(),
                    os.path.join(plot_dir, f'random_sample_{i}.png'), sample_idx=0
                )
        return {
            'avg_error': avg_random_error,
            'time': random_time,
            'details': random_results
        }

    def run_gradient_method():
        logger.info("Testing gradient-based operator selection...")
        start_time = time.time()

        # Process batches separately to avoid OOM
        all_errors = []
        total_samples = 0

        for batch_idx, batch in enumerate(test_loader):
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            # Run gradient optimization on this batch
            theta_latents, grad_pred, batch_avg_error = gradient_selection_multi_operator(
                model, theta_operators,
                batch_input, batch_target,
                num_operators=args.num_operators,
                epochs=200,
                lr=0.01,
                refinement_factor=1,
                splitting_method="strang",
                aux_loss_weight=0,
                dt=dt,
                theta_dim=theta_latent_operators.shape[1]  # latent dimension from codebook
            )

            # Plot first batch if requested
            if args.plot and batch_idx == 0:
                plot_dir = os.path.join(args.output_dir, 'plots_gradient')
                os.makedirs(plot_dir, exist_ok=True)
                for i in range(min(args.num_plots, batch_input.shape[0])):
                    plot_vorticity_predictions(
                        batch_input[i].cpu(), batch_target[i].cpu(), grad_pred[i].cpu(),
                        os.path.join(plot_dir, f'grad_sample_{i}.png'), sample_idx=0
                    )

            # Weight by batch size for proper averaging
            batch_size = batch_input.size(0)
            all_errors.append(batch_avg_error * batch_size)
            total_samples += batch_size

            # Calculate running average
            running_avg_error = sum(all_errors) / total_samples
            logger.info(f"Batch {batch_idx + 1}: {batch_size} samples, batch error {batch_avg_error:.6f}, running avg {running_avg_error:.6f}")

        # Calculate weighted average across all batches
        avg_grad_error = sum(all_errors) / total_samples if total_samples > 0 else 0
        grad_time = time.time() - start_time
        logger.info(f"Gradient method evaluated on {total_samples} samples with average error: {avg_grad_error:.6f}")

        return {
            'avg_error': avg_grad_error,
            'time': grad_time,
        }

    def run_beam_method():
        logger.info(f"Testing beam search operator selection (max {args.max_test_samples} samples)...")
        beam_results = []
        start_time = time.time()

        # Evaluate over test set
        sample_idx = 0
        total_beam_error = 0
        done = False
        logger.info("Starting beam search sample evaluation...")
        for batch_idx, batch in enumerate(test_loader):
            if done:
                break
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            for i in range(batch_input.size(0)):
                if sample_idx >= args.max_test_samples:
                    done = True
                    break

                composition, error, pred = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=3,
                    min_improvement_threshold=args.min_improvement,
                    dt=dt,
                    batch_size=args.beam_batch_size,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                
                total_beam_error += error
                sample_idx += 1
                running_avg = total_beam_error / sample_idx

                # Get composed viscosities
                composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

                beam_results.append({
                    'sample_idx': sample_idx - 1,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'viscosity': composed_viscosity}
                })

                if sample_idx <= 10 or (sample_idx - 1) % 5 == 0:
                    logger.info(f"Sample {sample_idx - 1}/{args.max_test_samples}: composition {composition}, error {error:.6f}, running avg {running_avg:.6f}")

        beam_time = time.time() - start_time
        avg_beam_error = sum(r['error'] for r in beam_results) / len(beam_results) if beam_results else 0
        logger.info(f"Beam search evaluated on {len(beam_results)} samples")

        # Plot predictions if requested
        if args.plot:
            plot_dir = os.path.join(args.output_dir, 'plots_beam')
            os.makedirs(plot_dir, exist_ok=True)
            # Get a batch for plotting
            plot_batch = next(iter(test_loader))
            plot_input = plot_batch['input'][:args.num_plots].to(DEVICE)
            plot_target = plot_batch['output'][:args.num_plots].to(DEVICE)
            for i in range(min(args.num_plots, plot_input.shape[0])):
                composition, _, plot_pred = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    plot_input[i:i+1], plot_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=5,
                    min_improvement_threshold=args.min_improvement,
                    dt=dt,
                    batch_size=args.beam_batch_size,
                    splitting_method=args.splitting_method,
                    refinement_factor=args.refinement_factor
                )
                
                # Enhanced diagnostics
                composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)
                logger.info(f"Beam Plot Sample {i}: composition {composition}, viscosity {composed_viscosity:.6f}")
                
                plot_vorticity_predictions(
                    plot_input[i].cpu(), plot_target[i].cpu(), plot_pred[0].cpu(),
                    os.path.join(plot_dir, f'beam_sample_{i}.png'), sample_idx=0
                )
        return {
            'avg_error': avg_beam_error,
            'time': beam_time,
            'details': beam_results
        }

    # Method registry
    method_registry = {
        'direct': run_direct_method,
        'greedy': run_greedy_method,
        'random': run_random_method,
        'gradient': run_gradient_method,
        'beam': run_beam_method
    }

    results = {
        'equation_type': 'navier_stokes',
        'timestamp': datetime.now().isoformat(),
        'num_operators': args.num_operators,
        'num_samples': args.num_samples,
        'dt': dt,
        'file_dir': args.file_dir,
        'vorticity_scale': args.vorticity_scale,
        'methods': {}
    }

    # Execute selected methods
    logger.info(f"Running methods: {', '.join(args.methods)}")
    for method_name in args.methods:
        if method_name in method_registry:
            results['methods'][method_name] = method_registry[method_name]()
        else:
            logger.warning(f"Unknown method '{method_name}', skipping...")

    # Summary
    logger.info("="*50)
    logger.info("SUMMARY FOR NAVIER-STOKES:")
    for method_name, method_result in results['methods'].items():
        if 'error' in method_result:
            # Direct method format
            logger.info(f"{method_name.title()} prediction: {method_result['error']:.6f} (time: {method_result['time']:.2f}s)")
        elif 'avg_error' in method_result:
            # Other methods format
            logger.info(f"{method_name.title()} selection: {method_result['avg_error']:.6f} (time: {method_result['time']:.2f}s)")
    logger.info("="*50)

    # Save results
    output_file = os.path.join(args.output_dir, f"navier_stokes_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()
