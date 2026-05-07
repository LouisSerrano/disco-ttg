"""
Targeted Navier-Stokes evaluation script.

Performs targeted evaluation on a fixed set of trajectories per viscosity,
saving all plots, inputs, and predictions for detailed analysis and comparison.

Uses deterministic sample selection so different method runs can be compared
on identical trajectories.

Usage:
    # Run with beam search
    python test_navier_stokes_targeted.py \
        --model_path /path/to/checkpoint.ckpt \
        --method beam \
        --output_dir ./results_targeted \
        --beam_width 3

    # Run with direct prediction (same samples due to same seed)
    python test_navier_stokes_targeted.py \
        --model_path /path/to/checkpoint.ckpt \
        --method direct \
        --output_dir ./results_targeted
"""

import torch
import argparse
import os
import time
import logging
import json
from datetime import datetime
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


from test_time_compute.ttc_utils import (
    save_results,
    DEVICE,
    get_relative_l2_error
)
from test_time_compute.ttc_methods import (
    greedy_operator_selection,
    random_operator_selection_batch,
    gradient_selection_multi_operator,
    beam_search_operator_selection_batch,
    get_state_labels
)

from train.train_euler_diffusion_aggregate import DISCOLitModule
from src.utils.euler_ns_dataset import NavierStokesDataset


def get_samples_by_viscosity(dataset, num_per_visc, seed=42):
    """
    Deterministically select samples grouped by viscosity.

    Args:
        dataset: NavierStokesDataset instance
        num_per_visc: Number of samples to select per viscosity
        seed: Random seed for reproducible selection

    Returns:
        dict: {visc_idx: [global_sample_indices]}
    """
    # Group indices by viscosity label (3rd element in tuple)
    visc_groups = {}
    for global_idx, (file_path, local_idx, visc_label) in enumerate(dataset.indices):
        if visc_label not in visc_groups:
            visc_groups[visc_label] = []
        visc_groups[visc_label].append(global_idx)

    # Deterministically select samples for each viscosity
    rng = np.random.RandomState(seed)
    selected = {}

    for visc_idx in sorted(visc_groups.keys()):
        available = visc_groups[visc_idx]
        # Shuffle with fixed seed for this viscosity
        shuffled = rng.permutation(available)
        # Select requested number of samples
        selected[visc_idx] = shuffled[:min(num_per_visc, len(shuffled))].tolist()

        logger.info(f"Viscosity {visc_idx} (nu={dataset.get_viscosity(visc_idx):.6f}): "
                   f"selected {len(selected[visc_idx])} samples from {len(available)} available")

    return selected


def plot_prediction(input_seq, target_seq, pred_seq, save_path, viscosity=None, sample_idx=0,
                   method=None, composition=None, error=None):
    """
    Plot 2D vorticity predictions comparing ground truth and prediction.

    Args:
        input_seq: Input sequence (T_in, C, H, W)
        target_seq: Target sequence (T_out, C, H, W)
        pred_seq: Prediction sequence (T_out, C, H, W)
        save_path: Path to save the figure
        viscosity: Viscosity value for title
        sample_idx: Sample index for title
        method: Method name for title
        composition: Operator composition for title
        error: Error value for title
    """
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
        error_map = np.abs(target_seq[t_idx, 0] - pred_seq[t_idx, 0])
        im_err = ax.imshow(error_map, cmap='hot')
        ax.set_title(f'|Err| t={t_idx}', fontsize=10)
        ax.axis('off')

    # Add colorbars with better placement
    cbar = fig.colorbar(im, ax=axes[:2, :], location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Vorticity', fontsize=10)

    # Error colorbar
    cbar_err = fig.colorbar(im_err, ax=axes[2, :], location='right', shrink=0.8, pad=0.02)
    cbar_err.set_label('Error', fontsize=10)

    # Build title
    title_parts = [f'Sample {sample_idx}']
    if viscosity is not None:
        title_parts.append(f'nu={viscosity:.6f}')
    if method is not None:
        title_parts.append(f'Method: {method}')
    if composition is not None:
        title_parts.append(f'Comp: {composition}')
    if error is not None:
        title_parts.append(f'Error: {error:.6f}')

    plt.suptitle(' | '.join(title_parts), fontsize=12, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_sample_outputs(base_dir, method, visc_value, sample_idx, input_seq, target_seq,
                       pred_seq, metadata):
    """
    Save all outputs for a single sample.

    Args:
        base_dir: Base output directory
        method: Method name (e.g., 'direct', 'beam')
        visc_value: Viscosity value
        sample_idx: Sample index within this viscosity
        input_seq: Input tensor (T_in, C, H, W)
        target_seq: Target tensor (T_out, C, H, W)
        pred_seq: Prediction tensor (T_out, C, H, W)
        metadata: Dict with composition, error, etc.
    """
    # Create directory structure
    visc_dir = f"visc_{visc_value:.6f}".replace('.', '_')
    sample_dir = os.path.join(base_dir, method, visc_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save tensors
    torch.save(input_seq.cpu(), os.path.join(sample_dir, 'input.pt'))
    torch.save(target_seq.cpu(), os.path.join(sample_dir, 'target.pt'))
    torch.save(pred_seq.cpu(), os.path.join(sample_dir, 'prediction.pt'))

    # Save metadata
    with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save plot
    plot_path = os.path.join(sample_dir, 'plot.png')
    plot_prediction(
        input_seq, target_seq, pred_seq, plot_path,
        viscosity=visc_value,
        sample_idx=sample_idx,
        method=method,
        composition=metadata.get('composition'),
        error=metadata.get('error')
    )

    return sample_dir


def run_direct_method(model, input_seq, target_seq, dt):
    """Run direct prediction method."""
    state_labels = get_state_labels(input_seq)

    with torch.no_grad():
        pred, _ = model(input_seq, state_labels, n_future_steps=target_seq.shape[1])

    relative_l2_error = get_relative_l2_error()
    error = relative_l2_error(pred, target_seq).item()

    return pred, error, {'composition': None, 'method': 'direct'}


def run_greedy_method(model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata):
    """Run greedy operator selection method."""
    from test_time_compute.ttc_methods import greedy_operator_selection

    composition, _, pred = greedy_operator_selection(
        model, theta_latent_operators,
        input_seq, target_seq,
        max_operators=5,
        min_improvement_threshold=args.min_improvement,
        dt=dt,
        splitting_method=args.splitting_method,
        refinement_factor=args.refinement_factor
    )

    relative_l2_error = get_relative_l2_error()
    error = relative_l2_error(pred, target_seq).item()

    # Get composed viscosity
    composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

    return pred, error, {
        'composition': composition,
        'composed_viscosity': composed_viscosity,
        'method': 'greedy'
    }


def run_random_method(model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata):
    """Run random operator selection method."""
    composition, error, pred = random_operator_selection_batch(
        model, theta_latent_operators,
        input_seq, target_seq,
        num_compositions=args.random_trials,
        composition_lengths=[2, 3],
        random_batch_size=args.random_batch_size,
        dt=dt,
        splitting_method=args.splitting_method,
        refinement_factor=args.refinement_factor
    )

    # Get composed viscosity
    composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

    return pred, error, {
        'composition': composition,
        'composed_viscosity': composed_viscosity,
        'method': 'random'
    }


def run_beam_method(model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata):
    """Run beam search operator selection method."""
    composition, error, pred = beam_search_operator_selection_batch(
        model, theta_latent_operators,
        input_seq, target_seq,
        beam_width=args.beam_width,
        max_operators=args.max_operators,
        min_improvement_threshold=args.min_improvement,
        dt=dt,
        batch_size=args.beam_batch_size,
        splitting_method=args.splitting_method,
        refinement_factor=args.refinement_factor
    )

    # Get composed viscosity
    composed_viscosity = sum(operator_metadata[op_id]['viscosity'] for op_id in composition)

    return pred, error, {
        'composition': composition,
        'composed_viscosity': composed_viscosity,
        'method': 'beam'
    }


def run_gradient_method(model, theta_operators, theta_latent_operators, input_seq, target_seq, args, dt):
    """Run gradient-based operator selection method."""
    theta_latents, pred, error = gradient_selection_multi_operator(
        model, theta_operators,
        input_seq, target_seq,
        num_operators=args.num_operators,
        epochs=200,
        lr=0.01,
        refinement_factor=args.refinement_factor,
        splitting_method=args.splitting_method,
        aux_loss_weight=0,
        dt=dt,
        theta_dim=theta_latent_operators.shape[1]
    )

    return pred, error, {
        'composition': 'gradient_optimized',
        'method': 'gradient',
        'num_operators': args.num_operators
    }


def load_model_from_checkpoint(checkpoint_path):
    """Load DISCO model from Lightning checkpoint."""
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
    parser = argparse.ArgumentParser(description='Targeted Navier-Stokes evaluation')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                        choices=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        help='Method to use for evaluation')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results_targeted',
                        help='Output directory')

    # Sample selection arguments
    parser.add_argument('--num_samples_per_visc', type=int, default=8,
                        help='Number of samples per viscosity (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for deterministic sample selection (default: 42)')

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

    # Method-specific arguments
    parser.add_argument('--num_operators', type=int, default=2,
                        help='Number of operators for gradient method (default: 2)')
    parser.add_argument('--max_operators', type=int, default=3,
                        help='Maximum operators for beam/greedy search (default: 3)')
    parser.add_argument('--beam_width', type=int, default=4,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--beam_batch_size', type=int, default=32,
                        help='Batch size for beam search (default: 32)')
    parser.add_argument('--random_trials', type=int, default=100,
                        help='Number of random compositions to try (default: 100)')
    parser.add_argument('--random_batch_size', type=int, default=16,
                        help='Batch size for random search (default: 16)')
    parser.add_argument('--min_improvement', type=float, default=1.0,
                        help='Minimum improvement threshold %% (default: 1.0)')
    parser.add_argument('--splitting_method', type=str, default='strang',
                        choices=['strang', 'lie'],
                        help='Operator splitting method (default: strang)')
    parser.add_argument('--refinement_factor', type=int, default=4,
                        help='Number of sub-steps per dt (default: 1)')
    parser.add_argument('--use_encoder', action='store_true',
                        help='Build dictionary from encoder outputs on training data '
                             'instead of using codebook entries (required when codebook_prob=0)')
    parser.add_argument('--num_dict_samples', type=int, default=256,
                        help='Number of training samples to encode for dictionary (default: 256)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        logger.error("Failed to load model")
        return

    # Get dt from model config
    dt = getattr(model, 'default_integration_time', 0.08)
    logger.info(f"Using dt (integration time): {dt}")

    # Create Navier-Stokes dataset (raw, not wrapped - for deterministic access)
    logger.info("Loading Navier-Stokes dataset...")
    ns_dataset = NavierStokesDataset(
        file_dir=args.file_dir,
        num_gpus=args.num_gpus,
        N_ns_ics=args.N_ns_ics
    )

    logger.info(f"Dataset has {len(ns_dataset)} total samples across {ns_dataset.m_visc} viscosities")
    logger.info(f"Viscosities: {ns_dataset.viscosities}")

    # Build operator dictionary
    from src.utils.euler_ns_dataset import EulerDiffusionDatasetWrapper
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

    if args.use_encoder:
        # Encode operators from training data (for models trained without codebook)
        # Ensure balanced coverage: sample equally from each environment
        logger.info(f"Encoding dictionary from ~{args.num_dict_samples} training samples (balanced across environments)...")
        from torch.utils.data import DataLoader, Subset
        import numpy as np

        num_environments = train_ds.num_environments
        samples_per_env = max(1, args.num_dict_samples // num_environments)
        logger.info(f"  {num_environments} environments, {samples_per_env} samples each")

        # Group wrapper dataset indices by environment label using the underlying dataset index
        env_indices = {}
        for i in range(len(train_ds)):
            # Access the label from the underlying dataset's index structure
            if hasattr(train_ds, '_subset_indices') and train_ds._subset_indices is not None:
                real_idx = train_ds._subset_indices[i]
            else:
                real_idx = i
            label = train_ds.dataset.indices[real_idx][3]  # (file_path, dataset_name, local_idx, label)
            if label not in env_indices:
                env_indices[label] = []
            env_indices[label].append(i)

        # Sample balanced indices
        rng = np.random.default_rng(42)
        selected_indices = []
        for env_idx in sorted(env_indices.keys()):
            pool = env_indices[env_idx]
            n_pick = min(samples_per_env, len(pool))
            picked = rng.choice(pool, size=n_pick, replace=False).tolist()
            selected_indices.extend(picked)
            logger.info(f"  Env {env_idx}: picked {n_pick} from {len(pool)} available")

        subset = Subset(train_ds, selected_indices)
        dict_loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)

        theta_latent_operators = []
        operator_metadata = []
        cpt = 0
        with torch.no_grad():
            for batch in dict_loader:
                inp = batch['input'].to(DEVICE)
                state_labels = torch.tensor([0], device=DEVICE)
                theta_latent, _ = model.encode_theta_latent(inp, state_labels)
                theta_latent_operators.append(theta_latent)
                for idx in range(len(inp)):
                    env_idx = batch['environment_idx'][idx].item()
                    viscosity = train_ds.dataset.get_viscosity(env_idx)
                    operator_metadata.append({
                        'operator_id': cpt,
                        'equation_type': "EulerDiffusion",
                        'viscosity': viscosity,
                    })
                    cpt += 1
        theta_latent_operators = torch.cat(theta_latent_operators)
        with torch.no_grad():
            theta_operators = model.decode_theta(theta_latent_operators, dim=2)
        logger.info(f"Encoded {theta_latent_operators.shape[0]} operators from training data")
    else:
        # Use codebook entries (default for models trained with codebook)
        logger.info("Using model codebook for operators...")
        theta_latent_operators = lit_model.codebook
        with torch.no_grad():
            theta_operators = model.decode_theta(theta_latent_operators, dim=2)

        operator_metadata = []
        num_environments = train_ds.num_environments
        for env_idx in range(num_environments):
            viscosity = train_ds.dataset.get_viscosity(env_idx)
            operator_metadata.append({
                'operator_id': env_idx,
                'equation_type': "EulerDiffusion",
                'viscosity': viscosity,
            })

    logger.info(f"Dictionary size: {theta_latent_operators.shape}")
    logger.info(f"Created metadata for {len(operator_metadata)} operators")

    # Deterministically select samples
    logger.info(f"Selecting {args.num_samples_per_visc} samples per viscosity with seed {args.seed}...")
    selected_samples = get_samples_by_viscosity(ns_dataset, args.num_samples_per_visc, args.seed)

    # Results tracking
    all_results = []
    errors_by_viscosity = {visc_idx: [] for visc_idx in selected_samples.keys()}

    total_samples = sum(len(indices) for indices in selected_samples.values())
    processed = 0

    start_time = time.time()

    # Process each viscosity
    for visc_idx in sorted(selected_samples.keys()):
        viscosity_value = ns_dataset.get_viscosity(visc_idx)
        sample_indices = selected_samples[visc_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing viscosity {visc_idx}: nu = {viscosity_value:.6f}")
        logger.info(f"Samples: {sample_indices}")
        logger.info(f"{'='*60}")

        for local_idx, global_idx in enumerate(sample_indices):
            processed += 1
            logger.info(f"\nSample {local_idx} (global {global_idx}) [{processed}/{total_samples}]")

            # Load sample directly from dataset
            trajectory, visc_label = ns_dataset[global_idx]

            # Preprocess trajectory (same as NavierStokesDatasetWrapper)
            trajectory = trajectory.float() / args.vorticity_scale
            trajectory = trajectory.unsqueeze(1)  # Add channel dim: (T, 1, H, W)

            # Extract input/output sequences
            input_seq = trajectory[:args.n_input_frames].unsqueeze(0).to(DEVICE)  # (1, T_in, 1, H, W)
            target_seq = trajectory[args.n_input_frames:args.n_input_frames + args.n_output_frames].unsqueeze(0).to(DEVICE)

            # Run the specified method
            try:
                if args.method == 'direct':
                    pred, error, meta = run_direct_method(model, input_seq, target_seq, dt)
                elif args.method == 'greedy':
                    pred, error, meta = run_greedy_method(
                        model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata
                    )
                elif args.method == 'random':
                    pred, error, meta = run_random_method(
                        model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata
                    )
                elif args.method == 'beam':
                    pred, error, meta = run_beam_method(
                        model, theta_latent_operators, input_seq, target_seq, args, dt, operator_metadata
                    )
                elif args.method == 'gradient':
                    pred, error, meta = run_gradient_method(
                        model, theta_operators, theta_latent_operators, input_seq, target_seq, args, dt
                    )
                else:
                    raise ValueError(f"Unknown method: {args.method}")

            except Exception as e:
                logger.error(f"Error processing sample {global_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Add metadata (convert to native Python types for JSON serialization)
            meta['viscosity'] = float(viscosity_value)
            meta['visc_idx'] = int(visc_idx)
            meta['global_sample_idx'] = int(global_idx)
            meta['local_sample_idx'] = int(local_idx)
            meta['error'] = float(error)

            # Save outputs
            save_sample_outputs(
                args.output_dir, args.method, viscosity_value, local_idx,
                input_seq[0], target_seq[0], pred[0], meta
            )

            # Track results
            all_results.append(meta)
            errors_by_viscosity[visc_idx].append(error)

            logger.info(f"  Error: {error:.6f}")
            if meta.get('composition') is not None:
                logger.info(f"  Composition: {meta['composition']}")

    elapsed_time = time.time() - start_time

    # Calculate summary statistics
    all_errors = [r['error'] for r in all_results]

    summary = {
        'method': args.method,
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'seed': args.seed,
        'num_samples_per_visc': args.num_samples_per_visc,
        'total_samples': len(all_results),
        'elapsed_time_seconds': elapsed_time,
        'overall_mean_error': float(np.mean(all_errors)) if all_errors else None,
        'overall_std_error': float(np.std(all_errors)) if all_errors else None,
        'per_viscosity': {}
    }

    # Per-viscosity statistics
    for visc_idx, errors in errors_by_viscosity.items():
        if errors:
            summary['per_viscosity'][str(visc_idx)] = {
                'viscosity': float(ns_dataset.get_viscosity(visc_idx)),
                'num_samples': len(errors),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors))
            }

    # Save summary
    summary_path = os.path.join(args.output_dir, args.method, 'summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Total samples: {len(all_results)}")
    logger.info(f"Elapsed time: {elapsed_time:.1f}s")
    logger.info(f"Overall mean error: {summary['overall_mean_error']:.6f}")
    logger.info(f"Overall std error: {summary['overall_std_error']:.6f}")

    logger.info("\nPer-viscosity results:")
    for visc_idx in sorted(errors_by_viscosity.keys()):
        if errors_by_viscosity[visc_idx]:
            visc_summary = summary['per_viscosity'][str(visc_idx)]
            logger.info(f"  nu={visc_summary['viscosity']:.6f}: "
                       f"mean={visc_summary['mean_error']:.6f} +/- {visc_summary['std_error']:.6f}")

    logger.info(f"\nResults saved to: {args.output_dir}/{args.method}/")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
