import torch
import argparse
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import random
from einops import rearrange


from test_time_compute.ttc_utils import (
    save_results,
    DEVICE,
    get_relative_l2_error,
)
from test_time_compute.ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    random_operator_selection_batch,
    gradient_selection_multi_operator,
    beam_search_operator_selection,
    beam_search_operator_selection_batch,
    multi_operator_splitting,
    get_state_labels
)

from train.train import DISCOLitModule, TemporalBatchDatasetFly

def load_model_from_checkpoint(checkpoint_path):
    """Load DISCO model from Lightning checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        lit_model = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        lit_model.eval()

        model = lit_model.model.to(DEVICE)
        model.eval()

        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, lit_model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def random_operator_selection_batch_with_tracking(model, theta_latent_operators, inp, target, 
                                                 num_compositions=1000, composition_lengths=[2, 3, 4], 
                                                 dt=4/250, random_batch_size=32, 
                                                 checkpoints=[10, 100, 200, 500, 1000],
                                                 true_v=None, true_D=None, operator_metadata=None):
    """
    Modified random operator selection that tracks all compositions tested
    and analyzes error evolution at specified checkpoints.
    """
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")

    original_shape = x_val.shape
    repeat_pattern = (random_batch_size,) + (1,) * len(original_shape)
    x_val_batch = x_val.unsqueeze(0).repeat(repeat_pattern)
    x_val_batch = rearrange(x_val_batch, 'b t ... -> (b t) ...')

    model.eval()

    # Track all tested compositions
    all_compositions = []
    all_errors = []
    all_param_errors = []
    
    # Track best at each checkpoint
    checkpoint_results = {cp: {
        'best_composition': None,
        'best_fit_error': float('inf'),
        'best_prediction_error': float('inf'),
        'best_v_error': float('inf'),
        'best_D_error': float('inf'),
        'avg_fit_error': 0,
        'num_tested': 0
    } for cp in checkpoints}
    
    best_overall_composition = []
    best_overall_error = float('inf')
    best_overall_pred = None
    best_overall_latent_operators = []

    dim = 1 if len(inp.shape) == 4 else 2
    
    num_tested = 0
    
    for batch_start in range(0, num_compositions, random_batch_size):
        batch_end = min(batch_start + random_batch_size, num_compositions)
        actual_batch_size = batch_end - batch_start
        
        # Random composition length
        length = random.choice(composition_lengths)

        # Random operator indices
        compositions = [[random.randint(0, theta_latent_operators.shape[0]-1) 
                        for _ in range(length)] for _ in range(actual_batch_size)]

        # Create test latent operators for all compositions
        test_latent_operators_batch = []
        for i in range(length):
            comp_operators = torch.stack([
                theta_latent_operators[compositions[k][i]].unsqueeze(0).repeat(original_shape[0], 1) 
                for k in range(actual_batch_size)
            ])
            comp_operators = rearrange(comp_operators, 'b t c -> (b t) c')
            test_latent_operators_batch.append(comp_operators)

        # Adjust x_val_batch for actual batch size if needed
        if actual_batch_size < random_batch_size:
            x_val_batch_current = x_val.unsqueeze(0).repeat((actual_batch_size,) + (1,) * len(original_shape))
            x_val_batch_current = rearrange(x_val_batch_current, 'b t ... -> (b t) ...')
        else:
            x_val_batch_current = x_val_batch
            
        # Predict
        if length == 1:
            with torch.no_grad():
                test_operators = model.decode_theta(test_latent_operators_batch[0], dim)
                pred, _ = model.solve_ode(x_val_batch_current, test_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=1, dt=dt)
                pred = rearrange(pred, '(b t) ... -> b t ...', b=actual_batch_size)
        else:
            with torch.no_grad():
                pred = multi_operator_splitting(model, test_latent_operators_batch, x_val_batch_current,
                                                     nt=1, dt=dt)
                pred = rearrange(pred, '(b t) ... -> b t ...', b=actual_batch_size)

        errors = [relative_l2_error(pred[k], y_val).item() for k in range(actual_batch_size)]

        # Track all compositions and their errors
        for k, (composition, error) in enumerate(zip(compositions, errors)):
            num_tested += 1
            all_compositions.append(composition)
            all_errors.append(error)
            
            # Calculate parameter errors if metadata available
            if operator_metadata and true_v is not None and true_D is not None:
                composed_v = sum(operator_metadata[op_id]['advection_speed'].item() 
                               if hasattr(operator_metadata[op_id]['advection_speed'], 'item') 
                               else operator_metadata[op_id]['advection_speed'] for op_id in composition)
                composed_D = sum(operator_metadata[op_id]['diffusion'].item() 
                               if hasattr(operator_metadata[op_id]['diffusion'], 'item')
                               else operator_metadata[op_id]['diffusion'] for op_id in composition)
                v_error = abs(composed_v - true_v)
                D_error = abs(composed_D - true_D)
                all_param_errors.append({'v_error': v_error, 'D_error': D_error})
            
            # Update best overall
            if error < best_overall_error:
                best_overall_error = error
                best_overall_composition = composition
                best_overall_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in composition]
            
            # Update checkpoint results
            for cp in checkpoints:
                if num_tested <= cp:
                    if error < checkpoint_results[cp]['best_fit_error']:
                        checkpoint_results[cp]['best_composition'] = composition
                        checkpoint_results[cp]['best_fit_error'] = error
                        if all_param_errors:
                            checkpoint_results[cp]['best_v_error'] = all_param_errors[-1]['v_error']
                            checkpoint_results[cp]['best_D_error'] = all_param_errors[-1]['D_error']
                    
                    # Update running average
                    checkpoint_results[cp]['avg_fit_error'] = (
                        (checkpoint_results[cp]['avg_fit_error'] * checkpoint_results[cp]['num_tested'] + error) /
                        (checkpoint_results[cp]['num_tested'] + 1)
                    )
                    checkpoint_results[cp]['num_tested'] += 1

    # Evaluate prediction errors for best compositions at each checkpoint
    for cp in checkpoints:
        if checkpoint_results[cp]['best_composition'] is not None:
            comp = checkpoint_results[cp]['best_composition']
            comp_latent = [theta_latent_operators[idx].unsqueeze(0) for idx in comp]
            
            with torch.no_grad():
                if len(comp) > 1:
                    pred_cp = multi_operator_splitting(model, comp_latent, 
                                                     inp[:, -1], nt=target.shape[1], dt=dt)
                else:
                    operators_cp = model.decode_theta(comp_latent[0], dim)
                    pred_cp, _ = model.solve_ode(inp[:, -1], operators_cp, state_labels,
                                               dim=dim, integration_time=dt, 
                                               n_future_steps=target.shape[1], dt=dt)
                checkpoint_results[cp]['best_prediction_error'] = relative_l2_error(pred_cp, target).item()
    
    # Final prediction on full sequence
    if len(best_overall_composition) > 0:
        with torch.no_grad():
            if len(best_overall_composition) > 1:
                pred = multi_operator_splitting(model, best_overall_latent_operators, 
                                              inp[:, -1], nt=target.shape[1], dt=dt)
            else:
                best_operators = model.decode_theta(best_overall_latent_operators[0], dim)
                pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                        dim=dim, integration_time=dt, 
                                        n_future_steps=target.shape[1], dt=dt)
    else:
        pred = torch.zeros_like(target)

    test_error = relative_l2_error(pred, target).item()
    
    # Prepare evolution results
    evolution_results = {
        'checkpoints': checkpoints,
        'best_fit_errors': [checkpoint_results[cp]['best_fit_error'] for cp in checkpoints],
        'best_prediction_errors': [checkpoint_results[cp]['best_prediction_error'] for cp in checkpoints],
        'avg_fit_errors': [checkpoint_results[cp]['avg_fit_error'] for cp in checkpoints],
        'best_v_errors': [checkpoint_results[cp]['best_v_error'] for cp in checkpoints] if all_param_errors else [],
        'best_D_errors': [checkpoint_results[cp]['best_D_error'] for cp in checkpoints] if all_param_errors else [],
        'all_errors': all_errors,
        'all_compositions': all_compositions,
        'all_param_errors': all_param_errors
    }

    return best_overall_composition, test_error, pred, evolution_results


def analyze_and_plot_evolution(evolution_results, sample_idx, output_dir, true_v=None, true_D=None):
    """
    Analyze and plot the evolution of errors through the random search
    """
    checkpoints = evolution_results['checkpoints']
    best_fit_errors = evolution_results['best_fit_errors']
    best_prediction_errors = evolution_results['best_prediction_errors']
    avg_fit_errors = evolution_results['avg_fit_errors']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Best fit and prediction error evolution
    ax = axes[0, 0]
    ax.plot(checkpoints, best_fit_errors, 'b-o', label='Fit Error', linewidth=2, markersize=8)
    ax.plot(checkpoints, best_prediction_errors, 'r-s', label='Prediction Error', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Compositions Tested')
    ax.set_ylabel('Best Error Found')
    ax.set_title(f'Best Error Evolution - Sample {sample_idx}')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average fit error evolution
    ax = axes[0, 1]
    ax.plot(checkpoints, avg_fit_errors, 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Compositions Tested')
    ax.set_ylabel('Average Fit Error')
    ax.set_title(f'Average Fit Error Evolution - Sample {sample_idx}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter estimation errors (if available)
    if evolution_results.get('best_v_errors') and evolution_results.get('best_D_errors'):
        ax = axes[1, 0]
        ax.plot(checkpoints, evolution_results['best_v_errors'], 'g-^', label='v error', linewidth=2, markersize=8)
        ax.plot(checkpoints, evolution_results['best_D_errors'], 'm-v', label='D error', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Compositions Tested')
        ax.set_ylabel('Relative Parameter Error')
        ax.set_title(f'Parameter Estimation Error - Sample {sample_idx}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if true_v is not None and true_D is not None:
            ax.text(0.05, 0.95, f'True v={true_v:.3f}, D={true_D:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Error distribution at different checkpoints
    ax = axes[1, 1]
    all_errors = evolution_results['all_errors']
    for i, cp in enumerate(checkpoints[::2]):  # Show every other checkpoint to avoid clutter
        errors_up_to_cp = all_errors[:cp]
        ax.hist(errors_up_to_cp, bins=30, alpha=0.5, label=f'n={cp}', density=True)
    ax.set_xlabel('Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'evolution_analysis_sample_{sample_idx}.png'), dpi=150)
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test time compute for advection-diffusion with evolution analysis')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results_evolution', help='Output directory')
    parser.add_argument('--num_operators_to_test', type=int, default=1000, 
                        help='Total number of operator compositions to test in random search')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[10, 100, 200, 500, 1000],
                        help='Checkpoints to analyze error evolution')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples to evaluate')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['E_AD_ALL', 'E_AD_v', 'E_AD_D'],
                        help='Experiment type: E_AD_ALL (v,D in [0,1]), E_AD_v (v in [1,3], D=0), E_AD_D (D in [1,3], v=0)')
    parser.add_argument('--random_batch_size', type=int, default=32,
                        help='Batch size for random operator selection')
    parser.add_argument('--composition_lengths', type=int, nargs='+', default=[2, 3, 4],
                        help='Possible composition lengths to test')
    args = parser.parse_args()

    # Adjust checkpoints if needed
    args.checkpoints = [cp for cp in args.checkpoints if cp <= args.num_operators_to_test]
    if args.num_operators_to_test not in args.checkpoints:
        args.checkpoints.append(args.num_operators_to_test)
    args.checkpoints.sort()

    # Define parameter ranges for each experiment
    EXPERIMENT_CONFIGS = {
        'E_AD_ALL': {
            'v_range': (0.01, 1.0),
            'D_range': (0.01, 1.0),
            'description': 'Both advection and diffusion in [0,1] range'
        },
        'E_AD_v': {
            'v_range': (1.0, 3.0),
            'D_range': (0.0, 0.0),
            'description': 'High advection speed [1,3], no diffusion'
        },
        'E_AD_D': {
            'v_range': (0.0, 0.0),
            'D_range': (1.0, 3.0),
            'description': 'High diffusion [1,3], no advection'
        }
    }

    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 34
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return

    # Load datasets based on experiment type
    experiment_config = EXPERIMENT_CONFIGS[args.experiment]
    print(f"\nLoading datasets for experiment: {args.experiment}")
    print(f"Description: {experiment_config['description']}")
    print(f"v_range: {experiment_config['v_range']}, D_range: {experiment_config['D_range']}")

    # Create train dataset for operator encoding
    train_dataset = TemporalBatchDatasetFly(
        n_batches=4,
        batch_size=64,
        sub_x=1,
        sub_t=1,
        split='train',
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        L=16.0,
        nx=256,
        nt=100,
        T=10.0,
        v_range=(0.01, 1.0),
        D_range=(0.001, 1.0),
        fractal_degree=256,
        fractal_power_range=3,
        seed=42
    )

    # Create test dataset - adjust batches for requested number of samples
    n_test_batches = max(1, args.num_samples // 64)
    test_dataset = TemporalBatchDatasetFly(
        n_batches=n_test_batches,
        batch_size=64,
        sub_x=1,
        sub_t=1,
        split='test',
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        L=16.0,
        nx=256,
        nt=100,
        T=10.0,
        v_range=experiment_config['v_range'],
        D_range=experiment_config['D_range'],
        fractal_degree=256,
        fractal_power_range=3,
        seed=124
    )

    train_loader = train_dataset
    test_loader = test_dataset

    print(f"Train dataset: {100 * 64} samples (approx)")
    print(f"Test dataset: {n_test_batches * 64} samples (will use first {args.num_samples})")

    # Encode operators from training data
    print("\nEncoding operators from training data...")
    theta_latent_operators = []
    operator_metadata = []
    cpt = 0
    
    for batch in train_loader:
        inp = batch['input'].to(DEVICE)
        state_labels = torch.tensor([0], device=DEVICE)
        theta_latent, metadata = model.encode_theta_latent(inp, state_labels)
        theta_latent_operators.append(theta_latent)

        for index in range(len(inp)):
            operator_metadata.append({
                'operator_id': cpt,
                'equation_type': "AdvectionDiffusion",
                'trajectory_indices': [],
                'advection_speed': batch['advection_speed'][index],
                'diffusion': batch['diffusion'][index],
            })
            cpt += 1
            
    theta_latent_operators = torch.cat(theta_latent_operators)
    
    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=1)

    print(f"Encoded {len(operator_metadata)} operators")

    # Run evolution analysis
    print(f"\nRunning random search evolution analysis...")
    print(f"Will test {args.num_operators_to_test} compositions per sample")
    print(f"Checkpoints: {args.checkpoints}")

    all_evolution_results = []
    all_sample_results = []
    
    sample_idx = 0
    for batch_idx, batch in enumerate(test_loader):
        if sample_idx >= args.num_samples:
            break
            
        batch_input = batch['input'].to(DEVICE)
        batch_target = batch['target'].to(DEVICE)
        batch_v = batch['advection_speed']
        batch_D = batch['diffusion']

        for i in range(min(batch_input.size(0), args.num_samples - sample_idx)):
            print(f"\nProcessing sample {sample_idx + 1}/{args.num_samples}")
            # Handle both tensor and float types
            true_v = batch_v[i].item() if hasattr(batch_v[i], 'item') else batch_v[i]
            true_D = batch_D[i].item() if hasattr(batch_D[i], 'item') else batch_D[i]
            print(f"True parameters: v={true_v:.3f}, D={true_D:.3f}")
            
            start_time = time.time()
            
            # Run random search with tracking
            composition, error, pred, evolution_results = random_operator_selection_batch_with_tracking(
                model, theta_latent_operators,
                batch_input[i:i+1], batch_target[i:i+1],
                num_compositions=args.num_operators_to_test,
                composition_lengths=args.composition_lengths,
                random_batch_size=args.random_batch_size,
                dt=10.0/100,
                checkpoints=args.checkpoints,
                true_v=true_v,
                true_D=true_D,
                operator_metadata=operator_metadata
            )
            
            elapsed_time = time.time() - start_time
            
            # Calculate composed parameters
            composed_v = sum(operator_metadata[op_id]['advection_speed'].item() if hasattr(operator_metadata[op_id]['advection_speed'], 'item') 
                           else operator_metadata[op_id]['advection_speed'] for op_id in composition)
            composed_D = sum(operator_metadata[op_id]['diffusion'].item() if hasattr(operator_metadata[op_id]['diffusion'], 'item')
                           else operator_metadata[op_id]['diffusion'] for op_id in composition)
            
            sample_result = {
                'sample_idx': sample_idx,
                'true_v': true_v,
                'true_D': true_D,
                'best_composition': composition,
                'best_error': error,
                'composed_v': composed_v,
                'composed_D': composed_D,
                'v_error': abs(composed_v - true_v),
                'D_error': abs(composed_D - true_D),
                'time': elapsed_time,
                'evolution': evolution_results
            }
            
            all_sample_results.append(sample_result)
            all_evolution_results.append(evolution_results)
            
            print(f"Best composition: {composition}")
            print(f"Best error: {error:.6f}")
            print(f"Composed params: v={composed_v:.3f}, D={composed_D:.3f}")
            print(f"Parameter errors: v_error={sample_result['v_error']:.3f}, D_error={sample_result['D_error']:.3f}")
            print(f"Time: {elapsed_time:.2f}s")
            
            # Plot evolution for this sample
            #analyze_and_plot_evolution(evolution_results, sample_idx, args.output_dir, 
            #                         true_v, true_D)
            
            sample_idx += 1

    # Aggregate results across all samples
    print("\n" + "="*50)
    print("AGGREGATE EVOLUTION ANALYSIS")
    print("="*50)
    
    # Calculate average evolution across all samples
    avg_evolution = {
        'checkpoints': args.checkpoints,
        'avg_best_fit_errors': [],
        'avg_best_prediction_errors': [],
        'avg_v_errors': [],
        'avg_D_errors': [],
        'std_best_fit_errors': [],
        'std_best_prediction_errors': [],
        'std_v_errors': [],
        'std_D_errors': []
    }
    
    for i, cp in enumerate(args.checkpoints):
        best_fit_errors_at_cp = [s['evolution']['best_fit_errors'][i] for s in all_sample_results]
        best_prediction_errors_at_cp = [s['evolution']['best_prediction_errors'][i] for s in all_sample_results]
        avg_evolution['avg_best_fit_errors'].append(np.mean(best_fit_errors_at_cp))
        avg_evolution['avg_best_prediction_errors'].append(np.mean(best_prediction_errors_at_cp))
        avg_evolution['std_best_fit_errors'].append(np.std(best_fit_errors_at_cp))
        avg_evolution['std_best_prediction_errors'].append(np.std(best_prediction_errors_at_cp))
        
        if all_sample_results[0]['evolution'].get('best_v_errors'):
            v_errors_at_cp = [s['evolution']['best_v_errors'][i] for s in all_sample_results]
            D_errors_at_cp = [s['evolution']['best_D_errors'][i] for s in all_sample_results]
            avg_evolution['avg_v_errors'].append(np.mean(v_errors_at_cp))
            avg_evolution['avg_D_errors'].append(np.mean(D_errors_at_cp))
            avg_evolution['std_v_errors'].append(np.std(v_errors_at_cp))
            avg_evolution['std_D_errors'].append(np.std(D_errors_at_cp))
    
    # Print summary
    print("\nAverage Best Error Evolution:")
    for i, cp in enumerate(args.checkpoints):
        print(f"  After {cp:4d} tests:")
        print(f"    Fit Error:        {avg_evolution['avg_best_fit_errors'][i]:.6f} ± {avg_evolution['std_best_fit_errors'][i]:.6f}")
        print(f"    Prediction Error: {avg_evolution['avg_best_prediction_errors'][i]:.6f} ± {avg_evolution['std_best_prediction_errors'][i]:.6f}")
    
    if avg_evolution.get('avg_v_errors'):
        print("\nAverage Parameter Estimation Error Evolution:")
        for i, cp in enumerate(args.checkpoints):
            print(f"  After {cp:4d} tests: v_error={avg_evolution['avg_v_errors'][i]:.3f} ± {avg_evolution['std_v_errors'][i]:.3f}, "
                  f"D_error={avg_evolution['avg_D_errors'][i]:.3f} ± {avg_evolution['std_D_errors'][i]:.3f}")
    
    # Plot aggregate results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Average best error evolution
    ax = axes[0]
    ax.errorbar(args.checkpoints, avg_evolution['avg_best_fit_errors'], 
                yerr=avg_evolution['std_best_fit_errors'], 
                fmt='b-o', label='Fit Error', linewidth=2, markersize=8, capsize=5)
    ax.errorbar(args.checkpoints, avg_evolution['avg_best_prediction_errors'], 
                yerr=avg_evolution['std_best_prediction_errors'], 
                fmt='r-s', label='Prediction Error', linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Number of Compositions Tested')
    ax.set_ylabel('Average Best Error')
    ax.set_title('Average Best Error Evolution Across All Samples')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average parameter estimation errors
    if avg_evolution.get('avg_v_errors'):
        ax = axes[1]
        ax.errorbar(args.checkpoints, avg_evolution['avg_v_errors'], 
                    yerr=avg_evolution['std_v_errors'],
                    fmt='g-^', label='v error', linewidth=2, markersize=8, capsize=5)
        ax.errorbar(args.checkpoints, avg_evolution['avg_D_errors'], 
                    yerr=avg_evolution['std_D_errors'],
                    fmt='m-v', label='D error', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Number of Compositions Tested')
        ax.set_ylabel('Average Relative Parameter Error')
        ax.set_title('Average Parameter Estimation Error Evolution')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'aggregate_evolution_analysis.png'), dpi=150)
    plt.close()
    
    # Save all results
    results = {
        'experiment': args.experiment,
        'num_operators_to_test': args.num_operators_to_test,
        'checkpoints': args.checkpoints,
        'num_samples': args.num_samples,
        'timestamp': datetime.now().isoformat(),
        'aggregate_evolution': avg_evolution,
        'sample_results': all_sample_results
    }
    
    output_file = os.path.join(args.output_dir, 
                               f"evolution_analysis_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Save without the raw error lists to keep file size manageable
    results_to_save = results.copy()
    for sample in results_to_save['sample_results']:
        sample['evolution'] = {k: v for k, v in sample['evolution'].items() 
                               if k not in ['all_errors', 'all_compositions', 'all_param_errors']}
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()