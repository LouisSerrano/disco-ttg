import torch
import argparse
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader


from test_time_compute.ttc_utils import (
    create_dataset_for_equation,
    save_results,
    GrayScottDatasetWrapper,
    DEVICE
)
from test_time_compute.ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
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


def save_prediction_data(output_dir, sample_idx, input_data, ground_truth, prediction, metadata=None):
    """Save prediction data in individual folder
    
    Args:
        output_dir: Base output directory
        sample_idx: Index of the sample
        input_data: Input tensor [T, C, H, W] or [T, C, H]
        ground_truth: Ground truth tensor [T, C, H, W] or [T, C, H]
        prediction: Model prediction tensor [T, C, H, W] or [T, C, H]
        metadata: Optional metadata dict (e.g., composition, error, parameters)
    """
    # Create folder for this sample
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    input_np = input_data.detach().cpu().numpy()
    ground_truth_np = ground_truth.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()
    
    # Save as .npy files
    np.save(os.path.join(sample_dir, "input.npy"), input_np)
    np.save(os.path.join(sample_dir, "ground_truth.npy"), ground_truth_np)
    np.save(os.path.join(sample_dir, "prediction.npy"), prediction_np)
    
    # Save metadata if provided
    if metadata is not None:
        import json
        with open(os.path.join(sample_dir, "metadata.json"), 'w') as f:
            # Convert any numpy types to Python native types
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump(convert_numpy(metadata), f, indent=2)
    
    print(f"Saved sample {sample_idx} to {sample_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run beam search and save predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./beam_search_predictions', 
                        help='Output directory for predictions')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of test samples to process')
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--max_operators', type=int, default=5,
                        help='Maximum number of operators in composition (default: 5)')
    parser.add_argument('--beam_batch_size', type=int, default=32,
                        help='Batch size for beam search operator selection (default: 32)')
    args = parser.parse_args()

    # Dataset configuration
    TRAINING_FILES = ["./datasets/gray-scott/feed_20params_512traj_each.hdf5",
                      "./datasets/gray-scott/kill_20params_512traj_each.hdf5"]
    TEST_FILES = ["./datasets/gray-scott/gray_scott_10x10_params_16traj_each.hdf5"]
    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 32

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"beam_search_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run configuration
    config = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'num_samples': args.num_samples,
        'beam_width': args.beam_width,
        'max_operators': args.max_operators,
        'beam_batch_size': args.beam_batch_size,
        'n_input_frames': N_INPUT_FRAMES,
        'n_output_frames': N_OUTPUT_FRAMES
    }
    
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load datasets
    print("Loading datasets...")
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
        batch_size=1,  # Process one sample at a time for easier saving
        shuffle=True, # for plots
        num_workers=2,
        pin_memory=True
    )

    # Load model
    print("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return
    
    # Get operators from codebook
    print("\nPreparing operators from codebook...")
    theta_latent_operators = lit_model.codebook
    with torch.no_grad():
        theta_operators = model.decode_theta(theta_latent_operators, dim=2)

    # Prepare operator metadata
    operator_metadata = []
    cpt = 0
    for index in range(0, len(train_ds), 512):
        sample = train_ds.__getitem__(index)
        operator_metadata.append({
            'operator_id': cpt,
            'equation_type': "ReactionDiffusion",
            'trajectory_indices': [],
            'f': sample['f'],
            'k': sample['k'],
        })
        cpt += 1
    
    # Process test samples
    print(f"\nProcessing {args.num_samples} test samples...")
    results_summary = []
    
    sample_idx = 0
    for batch_idx, batch in enumerate(test_loader):
        if sample_idx >= args.num_samples:
            break
            
        # Get data
        test_input = batch['input'].to(DEVICE)
        test_target = batch['output'].to(DEVICE)
        
        print(f"\nProcessing sample {sample_idx}/{args.num_samples}...")
        
        # Run beam search
        start_time = time.time()
        composition, error, pred = beam_search_operator_selection_batch(
            model, theta_latent_operators,
            test_input, test_target,
            beam_width=args.beam_width,
            max_operators=args.max_operators,
            batch_size=args.beam_batch_size
        )
        elapsed_time = time.time() - start_time
        
        print(f"  Composition: {composition}")
        print(f"  Error: {error:.6f}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        # Prepare metadata
        metadata = {
            'sample_idx': sample_idx,
            'composition': composition,
            'error': error,
            'elapsed_time': elapsed_time,
            'beam_width': args.beam_width,
            'max_operators': args.max_operators,
        }
        
        # Add physical parameters if available
        if 'f' in batch and 'k' in batch:
            metadata['f'] = float(batch['f'].item())
            metadata['k'] = float(batch['k'].item())
        
        # Save prediction data
        save_prediction_data(
            output_dir,
            sample_idx,
            test_input.squeeze(0),  # Remove batch dimension
            test_target.squeeze(0),
            pred.squeeze(0),
            metadata
        )
        
        # Add to summary
        results_summary.append(metadata)
        
        sample_idx += 1
    
    # Save overall summary
    summary_path = os.path.join(output_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config,
            'results': results_summary,
            'average_error': np.mean([r['error'] for r in results_summary]),
            'average_time': np.mean([r['elapsed_time'] for r in results_summary])
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Completed processing {len(results_summary)} samples")
    print(f"Average error: {np.mean([r['error'] for r in results_summary]):.6f}")
    print(f"Average time: {np.mean([r['elapsed_time'] for r in results_summary]):.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()