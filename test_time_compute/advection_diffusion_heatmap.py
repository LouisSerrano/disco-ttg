import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from ttc_utils import (
    save_results,
    DEVICE,
    get_relative_l2_error,
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

from train.train import DISCOLitModule, TemporalBatchDatasetFly, advection_diffusion_analytical
from src.utils.advection_diffusion import Fractaloid, FractaloidPhase, AdvectionDiffusionExplicit



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
    parser.add_argument('--experiment', type=str,
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
    parser.add_argument('--beam_width', type=int, default=4,
                        help='Beam width for beam search (default: 3)')
    parser.add_argument('--beam_batch_size', type=int, default=128,
                        help='Batch size for beam search operator selection (default: 32)')
    args = parser.parse_args()

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

    # For advection-diffusion, we'll use the TemporalBatchDatasetFly

    # Create train dataset for operator encoding (using experiment-specific parameter ranges)
    train_dataset = TemporalBatchDatasetFly(
        n_batches=2,  # Adjust as needed # faster
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
    train_loader = train_dataset

    print(f"Train dataset: {100 * 64} samples (approx)")

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

    relative_l2_error = get_relative_l2_error()

    grid_size = 10
    num_samples = grid_size * grid_size
    batch_size = 4
    fractal_power = 3
    fractal_degree = 256
    L = 16.0
    nx = 256
    nt = 100
    T = 10.0
    input_frames = 16
    output_frames = 34

    advection_linspace = np.linspace(0.01, 1, grid_size)
    diffusion_linspace = np.linspace(0.001, 1, grid_size)
    advection_grid, diffusion_grid = np.meshgrid(advection_linspace, diffusion_linspace)

    direct_results = np.zeros((grid_size, grid_size))
    beam_results = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            avg_direct_error = 0
            avg_beam_error = 0
            
            for b in range(batch_size):
                advection_speed = advection_grid[i, j]
                diffusion = diffusion_grid[i, j]
                
                # Generate fractaloid initial condition
                fractaloid = FractaloidPhase(
                    degree=fractal_degree,
                    power=float(fractal_power),
                    size=nx,
                    patch_size=nx
                )
                u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
                
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=L, v=advection_speed, D=diffusion, nt=nt, T=T
                )
                
                inp = u_xt[:input_frames].copy()
                target = u_xt[input_frames: input_frames + output_frames].copy()

                inp = torch.tensor(inp).float()
                target = torch.tensor(target).float()

                # batch and channel dimension
                inp = inp.cuda().unsqueeze(0).unsqueeze(-2) 
                target = target.cuda().unsqueeze(0).unsqueeze(-2)

                #print('inp', inp.shape)
                #print('target', target.shape)
                
                # Direct prediction
                with torch.no_grad():
                    state_labels = torch.tensor([0], device=inp.device)
                    pred, *_ = model(inp, state_labels, n_future_steps=target.shape[1])
                    error = relative_l2_error(pred, target).item()
                    avg_direct_error += error
                
                print(f"Batch {b}: direct error {error}")
                
                # Beam prediction
                with torch.no_grad():
                    composition, error, pred = beam_search_operator_selection_batch(
                        model, theta_latent_operators,
                        inp, target,
                        beam_width=args.beam_width,
                        max_operators=3,
                        dt=10.0/100,
                        batch_size=args.beam_batch_size
                    )
                    avg_beam_error += error
                
                print(f"Batch {b}: beam error {error}")

            print(i, j,"direct error", avg_direct_error/batch_size)
            print(i, j,"beam error", avg_beam_error/batch_size)
            
            direct_results[i, j] = avg_direct_error / batch_size
            beam_results[i, j] = avg_beam_error / batch_size

    # Save results
    np.save("./test-time-compute/results/direct_prediction_error.npy", direct_results)
    np.save("./test-time-compute/results/beam_prediction_error.npy", beam_results)
    np.save("./test-time-compute/results/advection_grid.npy", advection_grid)
    np.save("./test-time-compute/results/diffusion_grid.npy", diffusion_grid)



if __name__ == "__main__":
    main()