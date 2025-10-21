import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys

sys.path.append("/mnt/home/lserrano/disco-ball/test-time-compute")

from ttc_utils import (
    save_results,
    DEVICE
)

from train.train import TemporalBatchDatasetFly
from train_advection_diffusion import MPPLightning
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


def test_direct_prediction_mpp(model, test_loader, n_output_frames=34):
    """Test direct prediction for MPP model"""
    model.eval()
    rel_loss = RelativeL2()
    
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input = batch['input'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
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
    parser = argparse.ArgumentParser(description='Test MPP baseline for advection-diffusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./MPP/results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of test samples to evaluate')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['E_AD_ALL', 'E_AD_v', 'E_AD_D'],
                        help='Experiment type: E_AD_ALL (v,D in [0,1]), E_AD_v (v in [1,3], D=0), E_AD_D (D in [1,3], v=0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
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
    print("Loading MPP model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return

    # Load datasets based on experiment type
    experiment_config = EXPERIMENT_CONFIGS[args.experiment]
    print(f"\nLoading datasets for experiment: {args.experiment}")
    print(f"Description: {experiment_config['description']}")
    print(f"v_range: {experiment_config['v_range']}, D_range: {experiment_config['D_range']}")

    # Create test dataset (using experiment-specific parameter ranges)
    # Calculate batches needed for desired number of samples
    test_n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    test_dataset = TemporalBatchDatasetFly(
        n_batches=test_n_batches,
        batch_size=args.batch_size,
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

    test_loader = test_dataset

    print(f"Test dataset: {test_n_batches * args.batch_size} samples")

    # Test direct prediction
    print("\nTesting MPP direct prediction...")
    start_time = time.time()
    
    direct_error, _ = test_direct_prediction_mpp(model, test_loader, N_OUTPUT_FRAMES)
    
    direct_time = time.time() - start_time
    
    results = {
        'equation_type': 'advection_diffusion',
        'model_type': 'MPP',
        'experiment': args.experiment,
        'timestamp': datetime.now().isoformat(),
        'num_samples': args.num_samples,
        'direct_prediction': {
            'error': direct_error,
            'time': direct_time
        }
    }
    
    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR ADVECTION-DIFFUSION MPP BASELINE ({args.experiment}):")
    print(f"Direct prediction: {direct_error:.6f} (time: {direct_time:.2f}s)")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"advection_diffusion_mpp_{args.experiment}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()