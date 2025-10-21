import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys

sys.path.append("/mnt/home/lserrano/disco-ball/")

from ttc_utils import (
    create_dataset_for_equation,
    save_results,
    GrayScottDatasetWrapper,
    DEVICE
)
from ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    gradient_selection_multi_operator,
    beam_search_operator_selection,
    beam_search_operator_selection_batch
)
from train.train_rd_aggregate import DISCOLitModule

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
    parser = argparse.ArgumentParser(description='Test time compute for reaction-diffusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_operators', type=int, default=20, help='Number of operators to encode')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        default=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        help='Methods to test (default: all methods)')
    parser.add_argument('--random_trials', type=int, default=100,
                        help='Number of random compositions to try per sample (default: 100)')
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--beam_batch_size', type=int, default=32,
                        help='Batch size for beam search operator selection (default: 32)')
    args = parser.parse_args()

    TRAINING_FILES = ["/mnt/home/lserrano/gray-scott-python/data/feed_20params_512traj_each.hdf5",
    "/mnt/home/lserrano/gray-scott-python/data/kill_20params_512traj_each.hdf5"]

    #TEST_FILES = ["/mnt/home/lserrano/gray-scott-python/data/gray_scott_10x10_params_16traj_each.hdf5"]
    TEST_FILES = ["/mnt/home/lserrano/gray-scott-python/data/gray_scott_10x10_params_16traj_each.hdf5"]
    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 32

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = GrayScottDatasetWrapper(
    hdf5_files=TRAINING_FILES,
    split='train',
    input_frames=N_INPUT_FRAMES,
    output_frames=N_OUTPUT_FRAMES,
    sub_x=1,
    sub_t=1,
    trajectories_per_environment=512,
)
    
    test_ds = GrayScottDatasetWrapper(
        hdf5_files=TEST_FILES,
        split='test',
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        trajectories_per_environment=16
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=64, 
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True, 
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=64,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )
    # Load model
    print("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return
    
    # Encode operators from training data
    print("\nEncoding operators from training data...")
    #theta_operators, theta_latent_operators, operator_metadata = encode_operators_from_training_data(
    #    model, 
    #    train_loader,
    #    num_operators=args.num_operators
    #)

    theta_latent_operators = lit_model.codebook
    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=2)

    operator_metadata = []
    cpt=0
    for index in range(0, len(train_ds), 512):
        sample = train_ds.__getitem__(index)
        operator_metadata.append({
                        'operator_id': cpt,
                        'equation_type': "ReactionDiffuson",
                        'trajectory_indices': [],
                        'f':sample['f'],
                        'k':sample['k'],
                    })
        cpt+=1
    
    # Get test samples
    test_batch = next(iter(test_loader))
    test_input = test_batch['input'][:args.num_samples].to(DEVICE)
    test_target = test_batch['output'][:args.num_samples].to(DEVICE)

    # Define method registry
    def run_direct_method():
        print("\nTesting direct prediction...")
        start_time = time.time()
        direct_error, direct_pred = test_direct_prediction(model, test_loader)
        direct_time = time.time() - start_time
        return {
            'error': direct_error,
            'time': direct_time
        }

    def run_greedy_method():
        print("\nTesting greedy operator selection...")
        greedy_results = []
        start_time = time.time()

        for i in range(args.num_samples):
            composition, error, pred = greedy_operator_selection(
                model, theta_latent_operators,
                test_input[i:i+1], test_target[i:i+1],
                max_operators=5
            )
            greedy_results.append({
                'sample_idx': i,
                'composition': composition,
                'error': error
            })
            print(f"Sample {i}: composition {composition}, error {error:.6f}")

        greedy_time = time.time() - start_time
        avg_greedy_error = sum(r['error'] for r in greedy_results) / len(greedy_results)
        return {
            'avg_error': avg_greedy_error,
            'time': greedy_time,
            'details': greedy_results
        }

    def run_random_method():
        print("\nTesting random operator selection...")
        random_results = []
        start_time = time.time()

        # Evaluate over entire test set
        sample_idx = 0
        for batch in test_loader:
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            for i in range(batch_input.size(0)):
                composition, error, pred = random_operator_selection(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    num_compositions=args.random_trials,
                    composition_lengths=[2]
                )
                random_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error
                })
                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    if sample_idx > 0:
                        recent_errors = [r['error'] for r in random_results[-100:]]
                        avg_recent_error = sum(recent_errors) / len(recent_errors)
                        print(f"Sample {sample_idx}: avg error over last 100 samples: {avg_recent_error:.6f}")
                    else:
                        print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                sample_idx += 1

        random_time = time.time() - start_time
        avg_random_error = sum(r['error'] for r in random_results) / len(random_results)
        print(f"Random method evaluated on {len(random_results)} samples")
        return {
            'avg_error': avg_random_error,
            'time': random_time,
            'details': random_results
        }

    def run_gradient_method():
        print("\nTesting gradient-based operator selection...")
        start_time = time.time()

        # Process each batch separately to avoid OOM
        all_errors = []
        total_samples = 0

        for batch_idx, batch in enumerate(test_loader):
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            # Run gradient optimization on this batch
            theta_latents, grad_pred, batch_avg_error = gradient_selection_multi_operator(
                model, theta_operators,
                batch_input, batch_target,
                num_operators=2,  # Reaction-diffusion might need fewer operators
                epochs=1000,
                lr=0.01,
                refinement_factor=1,
                splitting_method="strang"
            )

            # Weight by batch size for proper averaging
            batch_size = batch_input.size(0)
            all_errors.append(batch_avg_error * batch_size)
            total_samples += batch_size

            # Calculate running average
            running_avg_error = sum(all_errors) / total_samples
            print(f"Batch {batch_idx + 1}: {batch_size} samples, batch error {batch_avg_error:.6f}, running avg {running_avg_error:.6f}")

        # Calculate weighted average across all batches
        avg_grad_error = sum(all_errors) / total_samples
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

        # Evaluate over entire test set
        sample_idx = 0
        for batch in test_loader:
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            for i in range(batch_input.size(0)):
                composition, error, pred = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=5,
                    batch_size=args.beam_batch_size
                )
                beam_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error
                })
                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    if sample_idx > 0:
                        recent_errors = [r['error'] for r in beam_results[-100:]]
                        avg_recent_error = sum(recent_errors) / len(recent_errors)
                        print(f"Sample {sample_idx}: avg error over last 100 samples: {avg_recent_error:.6f}")
                    else:
                        print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                sample_idx += 1

        beam_time = time.time() - start_time
        avg_beam_error = sum(r['error'] for r in beam_results) / len(beam_results)
        print(f"Beam search evaluated on {len(beam_results)} samples")
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
        'equation_type': 'reaction_diffusion',
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
    print("SUMMARY FOR REACTION-DIFFUSION:")
    for method_name, method_result in results['methods'].items():
        if 'error' in method_result:
            # Direct method format
            print(f"{method_name.title()} prediction: {method_result['error']:.6f} (time: {method_result['time']:.2f}s)")
        elif 'avg_error' in method_result:
            # Other methods format
            print(f"{method_name.title()} selection: {method_result['avg_error']:.6f} (time: {method_result['time']:.2f}s)")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"reaction_diffusion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()