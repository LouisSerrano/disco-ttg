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
    GrayScottDatasetWrapper,
    DEVICE
)

from train_gray_scott import MPPLightning
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


def test_direct_prediction_mpp(model, test_loader, n_output_frames=32):
    """Test direct prediction for MPP model (2D Gray-Scott)"""
    model.eval()
    rel_loss = RelativeL2()
    
    all_errors = []
    total_samples = 0
    total_error = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input = batch['input'].to(DEVICE)
            target = batch['output'].to(DEVICE)
            
            # Rearrange input for AViT model (2D)
            input = rearrange(input, "b t c h w -> t b c h w")
            
            # Create labels and boundary conditions tensors
            batch_size = input.size(1)
            labels = torch.tensor([[0, 1]], device=DEVICE)#.expand(batch_size, -1)
            bcs = torch.tensor([[1, 1]], device=DEVICE)#.expand(batch_size, -1)
            
            predictions = []
            # Direct prediction
            for _ in range(n_output_frames):
                pred = model(input, labels, bcs)
                predictions.append(pred)
                input = torch.cat([input[1:], pred.unsqueeze(0)], axis=0)
            
            predictions = torch.stack(predictions, axis=1)

            error = rel_loss(predictions, target)
            all_errors.append(error)
            total_error += error * batch_size
            total_samples += batch_size
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Error = {error.item():.6f}")
    
    #avg_error = sum(all_errors) / len(all_errors) if all_errors else 0
    avg_error = total_error / total_samples
    return avg_error, None


def main():
    parser = argparse.ArgumentParser(description='Test MPP baseline for reaction-diffusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of test samples to evaluate (will process all available)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    TEST_FILES = ["/mnt/home/lserrano/gray-scott-python/data/gray_scott_10x10_params_16traj_each.hdf5"]
    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 32

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading MPP model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_ds = GrayScottDatasetWrapper(
        hdf5_files=TEST_FILES,
        split='test',
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        trajectories_per_environment=16
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_ds)} samples")
    
    # Test direct prediction
    print("\nTesting MPP direct prediction...")
    start_time = time.time()
    
    direct_error, _ = test_direct_prediction_mpp(model, test_loader)
    
    direct_time = time.time() - start_time
    
    results = {
        'equation_type': 'reaction_diffusion',
        'model_type': 'MPP',
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_ds),
        'direct_prediction': {
            'error': direct_error,
            'time': direct_time
        }
    }
    
    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR REACTION-DIFFUSION MPP BASELINE:")
    print(f"Direct prediction: {direct_error:.6f} (time: {direct_time:.2f}s)")
    print(f"Total samples evaluated: {len(test_ds)}")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"reaction_diffusion_mpp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)


if __name__ == "__main__":
    main()