import torch
import numpy as np
import h5py
import os
import argparse
from neural_ode_operators import AdvectionNeuralODE, DiffusionNeuralODE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', type=str, required=True)
    parser.add_argument('--ckpt2', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--method', type=str, default='rk4')
    parser.add_argument('--splitting_method', type=str, default='strang')
    parser.add_argument('--refinement_factor', type=int, default=1)
    parser.add_argument('--op1_name', type=str, default='Burgers')
    parser.add_argument('--op2_name', type=str, default='Heat')
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_dataset}...")
    with h5py.File(args.test_dataset, 'r') as f:
        trajectories = f['test']['u'][:][:args.num_samples]
        time_points = f['test']['t'][:]
        alphas = f['test']['alpha'][:][:args.num_samples]
        betas = f['test']['beta'][:][:args.num_samples]
        gammas = f['test']['gamma'][:][:args.num_samples]
    
    print(f"Loaded {len(trajectories)} samples")
    nx = trajectories.shape[2]
    
    # Create models
    model1 = CNN1DODE(spatial_dim=nx, hidden_channels=64, num_layers=4)
    model2 = CNN1DODE(spatial_dim=nx, hidden_channels=64, num_layers=4)
    
    # Load checkpoints
    model1.load_state_dict(torch.load(args.ckpt1, map_location='cpu'))
    model2.load_state_dict(torch.load(args.ckpt2, map_location='cpu'))
    
    model1.eval()
    model2.eval()
    
    # Time stepping
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    small_dt = dt / args.refinement_factor
    
    print(f"\nTesting with {args.splitting_method} splitting, refinement={args.refinement_factor}")
    
    # Run predictions
    errors = []
    
    for idx in range(len(trajectories)):
        print(f"\nSample {idx+1}/{len(trajectories)}")
        
        u0 = torch.tensor(trajectories[idx, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Neural splitting prediction
        u_pred = u0.clone()
        
        for t_idx in range(nt):
            for _ in range(args.refinement_factor):
                if args.splitting_method == 'strang':
                    # Strang splitting: half step op1, full step op2, half step op1
                    u_pred = model1(u_pred, small_dt/2)
                    u_pred = model2(u_pred, small_dt)
                    u_pred = model1(u_pred, small_dt/2)
                else:  # lie
                    # Lie splitting: full step op1, full step op2
                    u_pred = model1(u_pred, small_dt)
                    u_pred = model2(u_pred, small_dt)
        
        # Compute error
        u_true = torch.tensor(trajectories[idx, -1], dtype=torch.float32)
        u_pred_np = u_pred.squeeze().detach().numpy()
        
        mse = np.mean((u_pred_np - u_true.numpy())**2)
        rel_error = np.linalg.norm(u_pred_np - u_true.numpy()) / np.linalg.norm(u_true.numpy())
        
        errors.append(rel_error)
        print(f"  Relative error: {rel_error:.6f}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Summary for {args.op1_name} + {args.op2_name} with {args.splitting_method} splitting")
    print(f"Mean relative error: {np.mean(errors):.6f}")
    print(f"Std relative error: {np.std(errors):.6f}")
    print(f"Max relative error: {np.max(errors):.6f}")
    print("="*60)

if __name__ == '__main__':
    main()