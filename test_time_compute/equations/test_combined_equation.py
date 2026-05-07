"""Test-time compute evaluation on the combined equation (reaction + advection + diffusion)."""
import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader


from test_time_compute.ttc_utils import (
    save_results,
    CombinedHDF5TemporalDataset,
    DEVICE,
    plot_results
)
from test_time_compute.ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    random_operator_selection_batch,
    gradient_selection_multi_operator,
    beam_search_operator_selection,
    beam_search_operator_selection_batch
)

from train.train_combined_aggregate import DISCOLitModule


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
    parser = argparse.ArgumentParser(description='Test time compute for combined equation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_time_compute/results', help='Output directory')
    parser.add_argument('--num_operators', type=int, default=20, help='Number of operators to encode')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['E_BG', 'E_ED', 'E_HE', 'E_ALL', 'E_EULER_OOD', 'E_DISP_OOD'],
                        help='Experiment type to run')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        default=['direct', 'greedy', 'random', 'gradient', 'beam'],
                        help='Methods to test (default: all methods)')
    parser.add_argument('--random_trials', type=int, default=200,
                        help='Number of random compositions to try per sample (default: 200)')
    parser.add_argument('--random_batch_size', type=int, default=16,
                        help='Batch size for random operator selection (default: 16)')
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--beam_batch_size', type=int, default=32,
                        help='Batch size for beam search operator selection (default: 32)')
    args = parser.parse_args()

    # Dataset mapping based on experiment type
    TRAINING_FILES = ["/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_EULER_train_gridparam256.h5",
                        "/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_HEAT_train_gridparam8192.h5",
                        "/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_DISP_train_gridparam8192.h5"
                        ]

    EXPERIMENT_FILES = {
        'E_BG': {
            'train': '/mnt/home/lserrano/ceph/E_BG_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_BG_test.h5'
        },
        'E_ED': {
            'train': '/mnt/home/lserrano/ceph/E_ED_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/ood/E_ED_test.h5'
        },
        'E_HE': {
            'train': '/mnt/home/lserrano/ceph/E_HE_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/ood/E_HE_test.h5'
        },
        'E_ALL': {
            'train': '/mnt/home/lserrano/ceph/E_ALL_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/test.h5'
        },
        'E_EULER_OOD': {
            'train': '/mnt/home/lserrano/ceph/E_EULER_OOD_train_envsize16.h5',
        },
        'E_DISP_OOD': {
            'train': '/mnt/home/lserrano/ceph/E_DISP_OOD_train_envsize16.h5',
        }
    }

    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 100
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return

    # Load datasets based on experiment type
    print(f"\nLoading datasets for experiment: {args.experiment}...")
    #train_file = EXPERIMENT_DATASETS[args.experiment]['train']
    test_file = EXPERIMENT_FILES[args.experiment]['train']

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    train_dataset = CombinedHDF5TemporalDataset(
        hdf5_files=TRAINING_FILES,
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        split='train'
    )

    test_dataset = CombinedHDF5TemporalDataset(
        hdf5_files=[test_file],
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        split='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True, # warning
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Encode operators from training data
    print("\nEncoding operators from training data...")
    #theta_operators, _, operator_metadata = encode_operators_from_training_data(
    #    model,
    #    train_file,
    #    num_operators=args.num_operators
    #)
    # to accelerate
    num_subsample = 4
    theta_latent_operators = lit_model.codebook[::num_subsample]
    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=1)

    operator_metadata = []
    cpt=0
    for index in range(0, len(train_dataset), 64*num_subsample): # 256 for later
        sample = train_dataset.__getitem__(index)
        operator_metadata.append({
                        'operator_id': cpt,
                        'equation_type': "CombinedEquaton",
                        'trajectory_indices': [],
                        'alpha':sample['alpha'],
                        'beta':sample['beta'],
                        'gamma':sample['gamma'],
                    })
        cpt+=1
    
    # Note: test samples will be processed batch by batch in each method
    
    # Define method registry
    def run_direct_method():
        print("\nTesting direct prediction...")
        start_time = time.time()
        direct_error, direct_pred = test_direct_prediction(model, test_loader, dt=10.0/100)
        direct_time = time.time() - start_time
        return {
            'error': direct_error,
            'time': direct_time
        }

    def run_greedy_method():
        print("\nTesting greedy operator selection...")
        greedy_results = []
        start_time = time.time()

        # Evaluate over entire test set
        sample_idx = 0
        for batch in test_loader:
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['output'].to(DEVICE)

            # Extract parameters if available
            batch_alpha = batch.get('alpha', torch.zeros(batch_input.size(0)))
            batch_beta = batch.get('beta', torch.zeros(batch_input.size(0)))
            batch_gamma = batch.get('gamma', torch.zeros(batch_input.size(0)))

            for i in range(batch_input.size(0)):
                composition, error, pred = greedy_operator_selection(
                    model, theta_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    max_operators=5,
                )

                # Calculate composed parameters
                composed_alpha = sum(operator_metadata[op_id]['alpha'] for op_id in composition)
                composed_beta = sum(operator_metadata[op_id]['beta'] for op_id in composition)
                composed_gamma = sum(operator_metadata[op_id]['gamma'] for op_id in composition)

                greedy_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'true_params': {'alpha': batch_alpha[i].item(), 'beta': batch_beta[i].item(), 'gamma': batch_gamma[i].item()},
                    'composed_params': {'alpha': composed_alpha, 'beta': composed_beta, 'gamma': composed_gamma}
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    if sample_idx > 0:
                        recent_errors = [r['error'] for r in greedy_results[-100:]]
                        avg_recent_error = sum(recent_errors) / len(recent_errors)
                        print(f"Sample {sample_idx}: avg error over last 100 samples: {avg_recent_error:.6f}")
                    else:
                        print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                        print(f"True params: alpha={batch_alpha[i]:.3f}, beta={batch_beta[i]:.3f}, gamma={batch_gamma[i]:.3f}")
                        print(f"Composed params: alpha={composed_alpha:.3f}, beta={composed_beta:.3f}, gamma={composed_gamma:.3f}")
                sample_idx += 1

        greedy_time = time.time() - start_time
        avg_greedy_error = sum(r['error'] for r in greedy_results) / len(greedy_results)
        print(f"Greedy method evaluated on {len(greedy_results)} samples")
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

            # Extract parameters if available
            batch_alpha = batch.get('alpha', torch.zeros(batch_input.size(0)))
            batch_beta = batch.get('beta', torch.zeros(batch_input.size(0)))
            batch_gamma = batch.get('gamma', torch.zeros(batch_input.size(0)))

            for i in range(batch_input.size(0)):
                composition, error, pred = random_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    num_compositions=args.random_trials,
                    composition_lengths=[2, 3], #3 for E_ALL
                    random_batch_size=args.random_batch_size
                )
                print('composition', composition)
                print('operator_metadata', len(operator_metadata))
                # Calculate composed parameters
                composed_alpha = sum(operator_metadata[op_id]['alpha'] for op_id in composition)
                composed_beta = sum(operator_metadata[op_id]['beta'] for op_id in composition)
                composed_gamma = sum(operator_metadata[op_id]['gamma'] for op_id in composition)

                random_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'true_params': {'alpha': batch_alpha[i].item(), 'beta': batch_beta[i].item(), 'gamma': batch_gamma[i].item()},
                    'composed_params': {'alpha': composed_alpha, 'beta': composed_beta, 'gamma': composed_gamma}
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    if sample_idx > 0:
                        recent_errors = [r['error'] for r in random_results[-100:]]
                        avg_recent_error = sum(recent_errors) / len(recent_errors)
                        print(f"Sample {sample_idx}: avg error over last 100 samples: {avg_recent_error:.6f}")
                    else:
                        print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                        print(f"True params: alpha={batch_alpha[i]:.3f}, beta={batch_beta[i]:.3f}, gamma={batch_gamma[i]:.3f}")
                        print(f"Composed params: alpha={composed_alpha:.3f}, beta={composed_beta:.3f}, gamma={composed_gamma:.3f}")
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
                num_operators=args.num_operators,
                epochs=300,
                lr=0.01,
                refinement_factor=1,
                splitting_method="strang",
                aux_loss_weight=0
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

            # Extract parameters if available
            batch_alpha = batch.get('alpha', torch.zeros(batch_input.size(0)))
            batch_beta = batch.get('beta', torch.zeros(batch_input.size(0)))
            batch_gamma = batch.get('gamma', torch.zeros(batch_input.size(0)))

            for i in range(batch_input.size(0)):
                composition, error, pred = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=3, #5, before but to go faster
                    batch_size=args.beam_batch_size
                )
                sample_idx+=1

                save_path = f"{args.output_dir}/plots/{args.experiment}/beam/"
                os.makedirs(save_path, exist_ok=True)
                plot_results(batch_target[i:i+1].cpu().squeeze(-2), pred.squeeze(-2).cpu().numpy(), error, f"{save_path}/{args.experiment}_prediction_{sample_idx}.pdf", num_plots=1)

                # Calculate composed parameters
                composed_alpha = sum(operator_metadata[op_id]['alpha'] for op_id in composition)
                composed_beta = sum(operator_metadata[op_id]['beta'] for op_id in composition)
                composed_gamma = sum(operator_metadata[op_id]['gamma'] for op_id in composition)

                beam_results.append({
                    'sample_idx': sample_idx,
                    'composition': composition,
                    'error': error,
                    'true_params': {'alpha': batch_alpha[i].item(), 'beta': batch_beta[i].item(), 'gamma': batch_gamma[i].item()},
                    'composed_params': {'alpha': composed_alpha, 'beta': composed_beta, 'gamma': composed_gamma}
                })

                if sample_idx % 100 == 0:  # Print progress every 100 samples
                    if sample_idx > 0:
                        recent_errors = [r['error'] for r in beam_results[-100:]]
                        avg_recent_error = sum(recent_errors) / len(recent_errors)
                        print(f"Sample {sample_idx}: avg error over last 100 samples: {avg_recent_error:.6f}")
                    else:
                        print(f"Sample {sample_idx}: composition {composition}, error {error:.6f}")
                        print(f"True params: alpha={batch_alpha[i]:.3f}, beta={batch_beta[i]:.3f}, gamma={batch_gamma[i]:.3f}")
                        print(f"Composed params: alpha={composed_alpha:.3f}, beta={composed_beta:.3f}, gamma={composed_gamma:.3f}")
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
        'equation_type': 'combined_equation',
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
    print(f"SUMMARY FOR COMBINED EQUATION ({args.experiment}):")
    for method_name, method_result in results['methods'].items():
        if 'error' in method_result:
            # Direct method format
            print(f"{method_name.title()} prediction: {method_result['error']:.6f} (time: {method_result['time']:.2f}s)")
        elif 'avg_error' in method_result:
            # Other methods format
            print(f"{method_name.title()} selection: {method_result['avg_error']:.6f} (time: {method_result['time']:.2f}s)")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"combined_equation_{args.experiment}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()