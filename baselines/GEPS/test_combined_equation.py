import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys

sys.path.append("/mnt/home/lserrano/disco-ttg/test_time_compute")

from ttc_utils import (
    save_results,
    CombinedHDF5TemporalDataset,
    DEVICE
)

from train_combined_equation import MPPLightning
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


def test_direct_prediction_mpp(model, test_loader, n_output_frames=50):
    """Test direct prediction for MPP model"""
    model.eval()
    rel_loss = RelativeL2()
    
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input = batch['input'].to(DEVICE)
            target = batch['output'].to(DEVICE)
            
            # Rearrange input for AViT1d model
            input = rearrange(input, "b t c h -> t b c h")
            
            predictions = [] 
            for _ in range(n_output_frames):
            # Direct prediction
                pred = model(input)
                predictions.append(pred)
                input = torch.cat([input[1:], pred.unsqueeze(0)], axis=0)

            predictions = torch.stack(predictions, axis=1)
            # Calculate error
            error = rel_loss(predictions, target)
            all_errors.append(error.item())
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Error = {error.item():.6f}")
    
    avg_error = sum(all_errors) / len(all_errors) if all_errors else 0
    return avg_error, None


def main():
    parser = argparse.ArgumentParser(description='Test MPP baseline for combined equation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./MPP/results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate (will process all available)')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['E_BG', 'E_ED', 'E_HE', 'E_ALL', 'E_EULER_OOD', 'E_DISP_OOD'],
                        help='Experiment type to run')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    args = parser.parse_args()

    EXPERIMENT_FILES = {
        'E_DEBUG': {
            'train': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_HEAT_valid.h5',
            #'test': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_BG_test.h5'
        },    
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
    N_OUTPUT_FRAMES = 50
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading MPP model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return

    # Load datasets based on experiment type
    print(f"\nLoading datasets for experiment: {args.experiment}...")
    test_file = EXPERIMENT_FILES[args.experiment]['train']

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    test_dataset = CombinedHDF5TemporalDataset(
        hdf5_files=[test_file],
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        split='train'  # Using train split for evaluation as specified
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # Test direct prediction
    print("\nTesting MPP direct prediction...")
    start_time = time.time()
    
    direct_error, _ = test_direct_prediction_mpp(model, test_loader)
    
    direct_time = time.time() - start_time
    
    results = {
        'equation_type': 'combined_equation',
        'model_type': 'MPP',
        'experiment': args.experiment,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_dataset),
        'direct_prediction': {
            'error': direct_error,
            'time': direct_time
        }
    }
    
    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR COMBINED EQUATION MPP BASELINE ({args.experiment}):")
    print(f"Direct prediction: {direct_error:.6f} (time: {direct_time:.2f}s)")
    print(f"Total samples evaluated: {len(test_dataset)}")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"combined_equation_mpp_{args.experiment}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()