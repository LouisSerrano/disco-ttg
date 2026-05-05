import torch
import argparse
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from ttc_utils import (
    save_results,
    DEVICE,
    get_relative_l2_error,
)
from ttc_methods_flops import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    random_operator_selection_batch,
    gradient_selection_multi_operator,
    beam_search_operator_selection,
    beam_search_operator_selection_batch
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


def main():
    parser = argparse.ArgumentParser(description='Test time compute for advection-diffusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--codebook_size', type=int, default=100, help='Number of operators to encode')
    parser.add_argument('--num_operators', type=int, default=20, help='Number of operators to use for grad')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of test samples to evaluate')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['E_AD_ALL', 'E_AD_v', 'E_AD_D'],
                        help='Experiment type: E_AD_ALL (v,D in [0,1]), E_AD_v (v in [1,3], D=0), E_AD_D (D in [1,3], v=0)')
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
    args = parser.parse_args()

    # Define parameter ranges for each experiment
    EXPERIMENT_CONFIGS = {
        'E_AD_ALL': {
            'v_range': (0.01, 1.0),  # Grid of advection and diffusion in [0,1]
            'D_range': (0.01, 1.0),
            'description': 'Both advection and diffusion in [0,1] range'
        },
        'E_AD_v': {
            'v_range': (1.0, 3.0),   # Advection speed in [1,3]
            'D_range': (0.0, 0.0),   # No diffusion (pure advection)
            'description': 'High advection speed [1,3], no diffusion'
        },
        'E_AD_D': {
            'v_range': (0.0, 0.0),   # No advection (pure diffusion)
            'D_range': (1.0, 3.0),   # Diffusion in [1,3]
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

    # For advection-diffusion, we'll use the TemporalBatchDatasetFly

    # Create train dataset for operator encoding (using experiment-specific parameter ranges)
    train_dataset = TemporalBatchDatasetFly(
        n_batches=4,  # Adjust as needed
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
        v_range=(0.01, 1.0),#experiment_config['v_range'],
        D_range=(0.001, 1.0),#experiment_config['D_range'],
        fractal_degree=256,
        fractal_power_range=3,
        seed=42
    )

    # Create test dataset (using experiment-specific parameter ranges)
    # Calculate batches needed for 512 samples with batch_size=64: 512/64 = 8 batches
    test_dataset = TemporalBatchDatasetFly(
        n_batches=8,  # 8 batches * 64 samples = 512 total samples
        batch_size=8, # 64 
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
    print(f"Test dataset: {8 * 64} = 512 samples")

    # Encode operators from training data
    print("\nEncoding operators from training data...")


    theta_latent_operators = []
    operator_metadata = []
    cpt=0
    for batch in train_loader:
        inp = batch['input'].to(DEVICE)
        state_labels = torch.tensor([0], device=DEVICE)
        theta_latent, metadata= model.encode_theta_latent(inp, state_labels)
        theta_latent_operators.append(theta_latent)

        for index in range(len(inp)):
            operator_metadata.append({
                            'operator_id': cpt,
                            'equation_type': "AdvectionDiffusion",
                            'trajectory_indices': [],
                            'advection_speed':batch['advection_speed'][index],
                            'diffusion':batch['diffusion'][index],
                        })
            cpt+=1
    theta_latent_operators = torch.cat(theta_latent_operators)
    #print('len',len(operator_metadata))
    #print('theta',theta_latent_operators.shape)

    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=1)

    # Note: test samples will be processed batch by batch in each method
    
    # Define method registry
    def run_direct_method():
        print("\nTesting direct prediction...")
        start_time = time.time()
        direct_error, direct_pred, direct_flops, direct_time, avg_prediction_loss = test_direct_prediction(model, test_loader)
        # Keep the wall-clock timer around to sanity check the internal timing
        direct_wall = time.time() - start_time
        return {
            'error': direct_error,
            'time': direct_time,
            'wall_time': direct_wall,
            'flops': direct_flops,
            'avg_prediction_loss': avg_prediction_loss
        }

    def run_greedy_method():
        print("\nTesting greedy operator selection...")
        greedy_results = []
        start_time = time.time()
        total_flops = 0

        # Evaluate over test set (all 512 samples)
        sample_idx = 0
        for batch_idx, batch in enumerate(test_loader):
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['target'].to(DEVICE)

            for i in range(batch_input.size(0)):  # Process all samples in batch
                composition, error, pred, flops, op_time = greedy_operator_selection(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    max_operators=5,
                    dt=10.0/100
                )
                total_flops += flops

                # Calculate composed parameters
                composed_v = sum(operator_metadata[op_id]['advection_speed'] for op_id in composition)
                composed_D = sum(operator_metadata[op_id]['diffusion'] for op_id in composition)

                greedy_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'v': composed_v, 'D': composed_D}
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                    print(f"Composed params: v={composed_v:.3f}, D={composed_D:.3f}")
                sample_idx += 1

        greedy_time = time.time() - start_time
        avg_greedy_error = sum(r['error'] for r in greedy_results) / len(greedy_results) if greedy_results else 0
        print(f"Greedy method evaluated on {len(greedy_results)} samples")
        return {
            'avg_error': avg_greedy_error,
            'time': greedy_time,
            'flops': total_flops,
            'details': greedy_results
        }

    def run_random_method():
        print("\nTesting random operator selection...")
        random_results = []
        start_time = time.time()
        total_flops = 0
        flops_per_iteration = []
        flops_per_trial = np.array([0.0]*args.random_trials)
        best_errors_per_trial = np.array([0.0]*args.random_trials)
        number_of_samples = 0
        global_trials = 0
        global_flops_sum = 0.0
        global_error_sum = 0.0

        # Evaluate over test set (all 512 samples)
        sample_idx = 0
        for batch_idx, batch in enumerate(test_loader):
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['target'].to(DEVICE)

            for i in range(batch_input.size(0)):  # Process all samples in batch
                (
                    composition,
                    error,
                    pred,
                    flops,
                    op_time,
                    flops_trajectory_local,
                    error_trajectory_local,
                    trials_this_call,
                    flops_sum_this_call,
                    error_sum_this_call,
                ) = random_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    num_compositions=args.random_trials,
                    composition_lengths=[2, 3, 4], # [2,3] before
                    random_batch_size=args.random_batch_size,
                    dt=10.0/100
                )
                number_of_samples += 1
                total_flops += flops
                flops_per_iteration.append(flops)
                # Update global running stats - only use up to the expected number of trials
                n_trials = min(len(flops_trajectory_local), args.random_trials)
                flops_per_trial[:n_trials] += flops_trajectory_local[:n_trials]
                best_errors_per_trial[:n_trials] += error_trajectory_local[:n_trials]
                global_trials += trials_this_call
                global_flops_sum += flops_sum_this_call
                global_error_sum += error_sum_this_call

                # Calculate composed parameters
                composed_v = sum(operator_metadata[op_id]['advection_speed'] for op_id in composition)
                composed_D = sum(operator_metadata[op_id]['diffusion'] for op_id in composition)

                random_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'v': composed_v, 'D': composed_D},
                    'flops': flops
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                    print(f"Composed params: v={composed_v:.3f}, D={composed_D:.3f}")
                sample_idx += 1

        random_time = time.time() - start_time
        avg_random_error = sum(r['error'] for r in random_results) / len(random_results) if random_results else 0
        print(f"Random method evaluated on {len(random_results)} samples")
        return {
            'avg_error': avg_random_error,
            'time': random_time,
            'flops': total_flops,
            'flops_per_iteration': flops_per_iteration,
            'avg_flops_per_iteration': (sum(flops_per_iteration) / len(flops_per_iteration)) if flops_per_iteration else 0,
            'flops_per_trial': (flops_per_trial/number_of_samples).tolist(),
            'errors_per_trial': (best_errors_per_trial/number_of_samples).tolist(),
            'details': random_results
        }

    def run_gradient_method():
        print("\nTesting gradient-based operator selection...")
        start_time = time.time()

        # Process batches separately to avoid OOM
        all_errors = []
        total_samples = 0

        for batch_idx, batch in enumerate(test_loader):

            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['target'].to(DEVICE)

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
                dt=10.0/100,
                theta_dim=2
            )

            # Weight by batch size for proper averaging
            batch_size = batch_input.size(0)
            all_errors.append(batch_avg_error * batch_size)
            total_samples += batch_size

            # Calculate running average
            running_avg_error = sum(all_errors) / total_samples
            print(f"Batch {batch_idx + 1}: {batch_size} samples, batch error {batch_avg_error:.6f}, running avg {running_avg_error:.6f}")

        # Calculate weighted average across all batches
        avg_grad_error = sum(all_errors) / total_samples if total_samples > 0 else 0
        grad_time = time.time() - start_time
        print(f"Gradient method evaluated on {total_samples} samples with average error: {avg_grad_error:.6f}")

        return {
            'avg_error': avg_grad_error,
            'time': grad_time,
        }

    def run_beam_method():
        print("\nTesting beam search operator selection...")
        beam_results = []
        start_time = time.time()
        total_flops = 0
        flops_per_iteration = []
        max_operators = 5  # From line 419
        max_trials = 256 + args.beam_width*256*(max_operators-1)  # Upper bound estimate
        flops_per_trial = np.array([0.0]*max_trials)
        best_errors_per_trial = np.array([0.0]*max_trials)
        trial_counts = np.array([0]*max_trials)  # Track how many samples contributed to each trial
        number_of_samples = 0
        global_trials = 0
        global_flops_sum = 0.0
        global_error_sum = 0.0

        # Evaluate over test set (all 512 samples)
        sample_idx = 0
        for batch_idx, batch in enumerate(test_loader):
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['target'].to(DEVICE)

            for i in range(batch_input.size(0)):  # Process all samples in batch
                (
                    composition,
                    error,
                    pred,
                    flops,
                    op_time,
                    flops_trajectory_local,
                    error_trajectory_local,
                    trials_this_call,
                    flops_sum_this_call,
                    error_sum_this_call,
                ) = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=5,
                    dt=10.0/100,
                    batch_size=args.beam_batch_size
                )
                number_of_samples += 1
                total_flops += flops
                flops_per_iteration.append(flops)
                # Update global running stats - only use up to the expected number of trials
                n_trials = min(len(flops_trajectory_local), len(flops_per_trial))
                flops_per_trial[:n_trials] += flops_trajectory_local[:n_trials]
                best_errors_per_trial[:n_trials] += error_trajectory_local[:n_trials]
                trial_counts[:n_trials] += 1  # Increment count for trials that were actually used
                global_trials += trials_this_call
                global_flops_sum += flops_sum_this_call
                global_error_sum += error_sum_this_call

                # Calculate composed parameters
                composed_v = sum(operator_metadata[op_id]['advection_speed'] for op_id in composition)
                composed_D = sum(operator_metadata[op_id]['diffusion'] for op_id in composition)

                beam_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'composed_params': {'v': composed_v, 'D': composed_D},
                    'flops': flops
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                    print(f"Composed params: v={composed_v:.3f}, D={composed_D:.3f}")
                sample_idx += 1

        beam_time = time.time() - start_time
        avg_beam_error = sum(r['error'] for r in beam_results) / len(beam_results) if beam_results else 0
        print(f"Beam search evaluated on {len(beam_results)} samples")
        return {
            'avg_error': avg_beam_error,
            'time': beam_time,
            'flops': total_flops,
            'flops_per_iteration': flops_per_iteration,
            'avg_flops_per_iteration': (sum(flops_per_iteration) / len(flops_per_iteration)) if flops_per_iteration else 0,
            'flops_per_trial': (flops_per_trial / np.maximum(trial_counts, 1)).tolist(),  # Avoid division by zero
            'errors_per_trial': (best_errors_per_trial / np.maximum(trial_counts, 1)).tolist(),
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
        'equation_type': 'advection_diffusion',
        'experiment': args.experiment,
        'timestamp': datetime.now().isoformat(),
        'num_operators': args.num_operators,
        'num_samples': args.num_samples,
        'methods': {}
    }
    
    # Execute selected methods
    print(f"\nRunning methods: {', '.join(args.methods)}")
    for method_name in args.methods:
        if method_name in method_registry:
            results['methods'][method_name] = method_registry[method_name]()
        else:
            print(f"Warning: Unknown method '{method_name}', skipping...")
    
    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR ADVECTION-DIFFUSION ({args.experiment}):")
    for method_name, method_result in results['methods'].items():
        if 'error' in method_result:
            # Direct method format
            msg = f"{method_name.title()} prediction: {method_result['error']:.6f} (time: {method_result['time']:.2f}s"
            if 'wall_time' in method_result:
                msg += f", wall: {method_result['wall_time']:.2f}s"
            if 'flops' in method_result:
                msg += f", flops: {method_result['flops']:.2e}"
            msg += ")"
            print(msg)
        elif 'avg_error' in method_result:
            # Other methods format
            msg = f"{method_name.title()} selection: {method_result['avg_error']:.6f} (time: {method_result['time']:.2f}s"
            if 'flops' in method_result:
                msg += f", flops: {method_result['flops']:.2e}"
            msg += ")"
            print(msg)
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"advection_diffusion_{args.experiment}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()
