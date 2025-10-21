"""
Neural Operator Splitting Test Script for Symbolic Regression

This script implements a clean version of the operator splitting approach from the 
symbolic regression notebook, focusing on:
1. Phase 1: Alpha parameter optimization with fixed neural networks
2. Phase 2: Joint fine-tuning with multi-step prediction

Based on notebooks/symbolic_regression_without_forgetting.ipynb
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from einops import rearrange

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import DISCO components
from src.operators.disco import DISCOHouse, vectors_to_parameters
from src.utils.advection_diffusion import Fractaloid, FractaloidPhase
from train.train import DISCOLitModule, advection_diffusion_analytical
from src.utils.plot_dataset_samples import plot_prediction_vs_ground_truth

# Import results management
from results_management import organize_results_by_parameters, save_results, convert_for_json


class RelativeL2(nn.Module):
    """Relative L2 loss function."""
    def forward(self, x, y, aggregate="mean"):
        x = rearrange(x, "b ... -> b (...)")
        y = rearrange(y, "b ... -> b (...)")
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)

        if aggregate == "mean":
            return (diff_norms / y_norms).mean()
        else:
            return (diff_norms / y_norms)


class TemporalBatchDatasetFly(IterableDataset):
    """Dataset for generating temporal batches on the fly."""
    def __init__(self, n_batches, batch_size, sub_x, sub_t, split="train", input_frames=16, output_frames=2,
                 L=16.0, nx=256, nt=100, T=10.0,
                 v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                 fractal_degree=8, fractal_power=2, seed=None,
                 fixed_params_mode=False, K=None):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.L = L
        self.nx = nx
        self.nt = nt
        self.T = T
        self.v_range = v_range
        self.D_range = D_range
        self.fractal_degree = fractal_degree
        self.fractal_power = fractal_power
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Fixed parameters mode for operator generation
        self.fixed_params_mode = fixed_params_mode
        self.K = K
        if fixed_params_mode and K is not None:
            self._setup_fixed_parameters()

    def _setup_fixed_parameters(self):
        """Setup fixed parameters for structured operator generation."""
        v_min, v_max = self.v_range if isinstance(self.v_range, (tuple, list)) else (0, self.v_range)
        D_min, D_max = self.D_range if isinstance(self.D_range, (tuple, list)) else (0, self.D_range)
        
        # K pure advection operators: v = linspace, D = 0
        v_values = np.linspace(v_min, v_max, self.K)
        advection_params = [(v, 0.0) for v in v_values]
        
        # K pure diffusion operators: v = 0, D = linspace  
        D_values = np.linspace(D_min, D_max, self.K)
        diffusion_params = [(0.0, D) for D in D_values]
        
        # Combine: 2*K total operators
        self.fixed_params = advection_params + diffusion_params
        self.param_index = 0
    
    def __iter__(self):
        for _ in range(self.n_batches):
            input_frames = self.input_frames
            batch_inputs = []
            batch_targets = []
            batch_v = []
            batch_d = []
            batch_init = []
            for _ in range(self.batch_size):
                # Sample advection speed and viscosity
                if self.fixed_params_mode:
                    # Use structured parameters for operator encoding
                    v, D = self.fixed_params[self.param_index % len(self.fixed_params)]
                    self.param_index += 1
                elif self.split == 'train':
                    if self.rng.random() < 0.5:
                        v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                        D = 0
                    else:
                        v = 0
                        D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                else:
                    v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                    D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                
                # Generate fractaloid initial condition
                fractaloid = FractaloidPhase(
                    degree=self.fractal_degree,
                    power=self.fractal_power,
                    size=self.nx,
                    patch_size=self.nx
                )
                u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=self.L, v=v, D=D, nt=self.nt, T=self.T
                )
                u_xt = u_xt[::self.sub_t, ::self.sub_x]
                input = u_xt[:input_frames].copy()
                target = u_xt[input_frames: input_frames + self.output_frames].copy()
                batch_inputs.append(torch.from_numpy(input).unsqueeze(-2).float())
                batch_targets.append(torch.from_numpy(target).unsqueeze(-2).float())
                batch_v.append(v)
                batch_d.append(D)
                batch_init.append(torch.from_numpy(u0))
            
            batch = {
                'input': torch.stack(batch_inputs),
                'target': torch.stack(batch_targets),
                'velocities': batch_v,
                'diffusivities': batch_d,
                'initial_conditions': torch.stack(batch_init)
            }
            yield batch


def get_batched_operators(theta_batch, model, dim=1):
    """
    Create a single function that processes all thetas at once.
    theta_batch: [num_operators, theta_dim]
    """
    from torch.func import functional_call, vmap
    
    base_opnn = model.opnns[str(dim)]
    param_dict = dict(base_opnn.named_parameters())
    batched_params_dict = vectors_to_parameters(theta_batch, param_dict)
    
    def batched_operator(x, state_labels):
        # x: [batch_size, channels, ...spatial]
        # state_labels: [num_states] or [batch_size, num_states]
        # theta_batch: [num_operators, theta_dim]
        
        num_operators = theta_batch.shape[0]
        
        # We need to replicate x and state_labels for each operator
        # x needs to become [num_operators, batch_size, channels, ...spatial]
        x_replicated = x.unsqueeze(0).expand(num_operators, -1, -1, -1)
        
        # state_labels needs to become [num_operators, ...]
        if state_labels.dim() == 1:
            # [num_states] -> [num_operators, num_states]
            state_labels_replicated = state_labels.unsqueeze(0).expand(num_operators, -1)
        else:
            # [batch_size, num_states] -> [num_operators, batch_size, num_states]  
            state_labels_replicated = state_labels.unsqueeze(0).expand(num_operators, -1, -1)
        
        # Now vmap over the operator dimension (first dim)
        return vmap(functional_call, in_dims=(None, 0, 0))(
            base_opnn, batched_params_dict, (x_replicated, state_labels_replicated)
        )
    
    return batched_operator


def mixture_of_operators(x, state_labels, alpha, operators):
    """
    Compute weighted mixture of operators using einsum.
    """
    # Get operator outputs: [num_operators, batch_size, channels, ...spatial]
    operator_outputs = operators(x, state_labels)
    
    if alpha.dim() == 1:
        # alpha: [num_operators]
        # Use einsum to weight and sum over operators
        return torch.einsum('k,k...->...', alpha, operator_outputs)
    else:
        # alpha: [batch_size, num_operators]
        # Transpose to [num_operators, batch_size] and use einsum
        alpha_t = alpha.t()  # [num_operators, batch_size]
        return torch.einsum('kb,kb...->b...', alpha_t, operator_outputs)


def solve_mixture_ode(x_input, operators, alpha, state_labels, 
                     integration_time=1.0, n_future_steps=1, 
                     solver='rk4', rtol=1e-7):
    """
    Solve neural ODE using mixture of operators.
    """
    # Create ODE function
    def ode_func(t, x):
        return mixture_of_operators(x, state_labels, alpha, operators)
    
    # Time grid
    t = torch.linspace(0, integration_time, n_future_steps + 1, device=x_input.device)
    
    # Solve ODE
    nsteps, solution = odeint(ode_func, x_input, t=t, rtol=rtol, method=solver)
    
    # Return future steps (excluding initial condition)
    return solution[1:, ...]


def sparsify_weights(weights, ratio_threshold=0.1):
    """
    Set weights to zero if they are less than ratio_threshold * max_weight
    
    Args:
        weights: tensor of weights
        ratio_threshold: threshold ratio (default 0.1 for 10%)
    
    Returns:
        sparsified weights
    """
    max_weight = torch.max(torch.abs(weights), dim=-1)[0]
    threshold = ratio_threshold * max_weight
    
    # Create mask for weights above threshold
    mask = torch.abs(weights) >= threshold
    
    # Apply mask
    sparsified_weights = weights * mask
    
    return sparsified_weights


class NeuralOperatorSplittingExperiment:
    """Main experiment class for neural operator splitting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the experiment with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative_l2_error = RelativeL2().to(self.device)
        
        # Create results directory
        self.results_dir = config['output_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load DISCO model
        self.model = self._load_disco_model()
        
        # Initialize data parameters
        self.batch_size = config.get('batch_size', 4)
        self.sub_x = config.get('sub_x', 1)
        self.sub_t = config.get('sub_t', 1)
        self.n_input_frames = config.get('n_input_frames', 16)
        self.n_output_frames = config.get('n_output_frames', 34)
        
        # Results storage
        self.results = {}
        
    def _load_disco_model(self):
        """Load the pre-trained DISCO model."""
        ckpt_path = self.config['checkpoint_path']
        print(f"Loading DISCO model from {ckpt_path}...")
        
        model = DISCOLitModule.load_from_checkpoint(ckpt_path, map_location=self.device)
        model = model.model.to(self.device)
        model.eval()
        
        return model
    
    def _create_dataset(self, split='train', v_range=(0.9, 1.0), D_range=(0.9, 1.0), 
                       fixed_params_mode=False, K=None):
        """Create dataset for the experiment."""
        n_batches = int(1000 // self.batch_size)
        
        dataset = TemporalBatchDatasetFly(
            n_batches=n_batches,
            batch_size=self.batch_size,
            sub_x=self.sub_x,
            sub_t=self.sub_t,
            split=split,
            input_frames=self.n_input_frames,
            output_frames=self.n_output_frames,
            L=16.0,
            nx=256,
            nt=100,
            T=10.0,
            fractal_power=3.0,
            fractal_degree=256,
            v_range=v_range,
            D_range=D_range,
            fixed_params_mode=fixed_params_mode,
            K=K,
        )
        
        return DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=True)
    
    def encode_structured_operators(self, K=4, n_trajectories_per_operator=1, v_range=(0.1, 1.0), D_range=(0.1, 1.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Encode 2*K operators from structured trajectories.
        
        Args:
            K: Number of operators per type (advection/diffusion)
            n_trajectories_per_operator: Number of trajectories to generate per operator (default: 1)
            v_range: Range for advection velocities
            D_range: Range for diffusion coefficients
        
        Returns:
            theta: [2*K, theta_dim] tensor of encoded operators
            inp_trajectories: [2*K*n_trajectories_per_operator, T, C, H] input trajectories for finetuning
            target_trajectories: [2*K*n_trajectories_per_operator, T_out, C, H] target trajectories for finetuning
            params: list of (v, D) tuples for each operator
        """
        print("=" * 60)
        print("ENCODING STRUCTURED OPERATORS")
        print("=" * 60)
        
        # Create dataset for structured operator encoding
        # Set batch_size = 2*K so each batch contains exactly one trajectory per operator
        total_operators = 2 * K
        
        # Temporarily store original batch_size and set to total_operators
        original_batch_size = self.batch_size
        self.batch_size = total_operators
        
        encoding_loader = self._create_dataset(
            split='train', 
            v_range=v_range, 
            D_range=D_range,
            fixed_params_mode=True, 
            K=K
        )
        
        # Collect n_trajectories_per_operator batches, each containing one trajectory per operator
        all_inp_trajectories = []
        all_target_trajectories = []
        all_params = []
        
        for traj_idx in range(n_trajectories_per_operator):
            batch = next(iter(encoding_loader))
            
            inp = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            velocities = batch["velocities"] 
            diffusivities = batch["diffusivities"]
            
            all_inp_trajectories.append(inp)
            all_target_trajectories.append(target)
            
            # Store params for all trajectories
            all_params.extend([(v, D) for v, D in zip(velocities, diffusivities)])
        
        # Restore original batch_size
        self.batch_size = original_batch_size
        
        # Concatenate all trajectories
        all_inp_trajectories = torch.cat(all_inp_trajectories, dim=0)
        all_target_trajectories = torch.cat(all_target_trajectories, dim=0)
        
        # Verify we have the right number of trajectories
        total_trajectories = total_operators * n_trajectories_per_operator
        assert all_inp_trajectories.shape[0] == total_trajectories, f"Expected {total_trajectories} trajectories, got {all_inp_trajectories.shape[0]}"
        
        print(f"Encoding {total_operators} operators using {len(all_params)} trajectories:")
        for i, (v, D) in enumerate(all_params):
            op_type = "Advection" if D == 0 else "Diffusion" 
            operator_idx = i // n_trajectories_per_operator
            traj_idx = i % n_trajectories_per_operator
            print(f"  Trajectory {i}: Operator {operator_idx} ({op_type}) trajectory {traj_idx} (v={v:.3f}, D={D:.3f})")
        
        # Encode operators using DISCO model
        state_labels = torch.tensor([0], device=self.device)
        
        with torch.no_grad():
            theta_latent, metadata = self.model.encode_theta_latent(all_inp_trajectories, state_labels)
            theta_all = self.model.decode_theta(theta_latent, dim=1)
            
            if n_trajectories_per_operator == 1:
                # No averaging needed - each trajectory corresponds to one operator
                theta = theta_all
            else:
                # Reshape and average theta parameters correctly
                # theta_all shape: [n_trajectories_per_operator * total_operators, theta_dim]
                # Reshape to: [n_trajectories_per_operator, total_operators, theta_dim]
                # Then average over trajectory dimension (dim=0)
                theta = theta_all.view(n_trajectories_per_operator, total_operators, -1).mean(dim=0)
        
        print(f"Encoded operators shape: {theta.shape}")
        print(f"Input trajectories shape: {all_inp_trajectories.shape}")
        print(f"Target trajectories shape: {all_target_trajectories.shape}")
        
        # Create unique params list for operators (remove duplicates)
        unique_params = []
        for op_idx in range(total_operators):
            start_idx = op_idx * n_trajectories_per_operator
            unique_params.append(all_params[start_idx])  # Take first trajectory's params for each operator
        
        return theta, all_inp_trajectories, all_target_trajectories, unique_params
    
    def random_operator_sampling(self, inp, target, theta, state_labels) -> Tuple[int, int, Dict]:
        """
        Random sampling approach to find best operator pair.
        
        Returns:
            Best operator indices and sampling history
        """
        print("=" * 60)
        print("RANDOM OPERATOR SAMPLING")
        print("=" * 60)
        
        num_operators = theta.shape[0]
        n_samples = self.config.get('n_random_samples', 1000)
        
        # Use a random time step for validation
        val_t = random.randint(0, self.n_input_frames - 2)
        x_val = inp[:, val_t].to(self.device)
        y_val = inp[:, val_t+1].to(self.device)
        
        best_error = float('inf')
        best_op1_idx = 0
        best_op2_idx = 0
        
        history = {
            'errors': [],
            'operator_pairs': [],
            'best_error_evolution': []
        }
        
        print(f"Sampling {n_samples} random operator pairs from {num_operators} operators...")
        
        for sample_idx in tqdm(range(n_samples), desc="Random Sampling"):
            # Randomly select two operators
            op1_idx = random.randint(0, num_operators - 1)
            op2_idx = random.randint(0, num_operators - 1)
            
            # Ensure different operators
            while op2_idx == op1_idx and num_operators > 1:
                op2_idx = random.randint(0, num_operators - 1)
            
            # Get individual operators
            with torch.no_grad():
                operator1 = get_batched_operators(theta[op1_idx:op1_idx+1], self.model, dim=1)
                operator2 = get_batched_operators(theta[op2_idx:op2_idx+1], self.model, dim=1)
                
                # Create simple alpha weights (equal weights for the selected operators)
                alpha1 = torch.ones(x_val.shape[0], 1, device=self.device)
                alpha2 = torch.ones(x_val.shape[0], 1, device=self.device)
                
                # Forward pass with operator splitting
                tmp = solve_mixture_ode(
                    x_val, operator1, alpha1, state_labels,
                    integration_time=1, n_future_steps=1)
                pred = solve_mixture_ode(
                    tmp[-1], operator2, alpha2, state_labels,
                    integration_time=1, n_future_steps=1)
                
                # Calculate error
                error = self.relative_l2_error(pred[-1], y_val).item()
                
                # Track history
                history['errors'].append(error)
                history['operator_pairs'].append((op1_idx, op2_idx))
                
                # Update best if found better combination
                if error < best_error:
                    best_error = error
                    best_op1_idx = op1_idx
                    best_op2_idx = op2_idx
                
                history['best_error_evolution'].append(best_error)
                
                # Print progress every 100 samples
                if (sample_idx + 1) % 100 == 0 or sample_idx == n_samples - 1:
                    print(f"Sample [{sample_idx+1}/{n_samples}] | "
                          f"Current Error: {error:.6f} | "
                          f"Best Error: {best_error:.6f} | "
                          f"Best Pair: ({best_op1_idx}, {best_op2_idx})")
        
        print(f"Random sampling completed. Best operator pair: ({best_op1_idx}, {best_op2_idx}) with error: {best_error:.6f}")
        return best_op1_idx, best_op2_idx, history
    
    def estimate_coefficients(self, best_composition: List[int], operator_params: List[Tuple[float, float]], ground_truth_params: List[Tuple[float, float]] = None) -> Dict:
        """
        Estimate PDE coefficients from the best operator composition and compare to ground truth.
        
        Args:
            best_composition: List of operator indices in the composition
            operator_params: List of (v, D) tuples for each operator
            ground_truth_params: List of ground truth (v, D) tuples for comparison
        
        Returns:
            Dictionary containing coefficient estimates, breakdown, and ground truth comparison
        """
        total_v = 0.0
        total_D = 0.0
        
        operator_breakdown = []
        for op_idx in best_composition:
            v, D = operator_params[op_idx]
            total_v += v
            total_D += D
            
            op_type = "Advection" if D == 0 else ("Diffusion" if v == 0 else "Mixed")
            operator_breakdown.append({
                'operator_index': op_idx,
                'operator_type': op_type,
                'v_contribution': v,
                'D_contribution': D,
                'parameters': (v, D)
            })
        
        result = {
            'estimated_v': total_v,
            'estimated_D': total_D,
            'composition_length': len(best_composition),
            'operator_breakdown': operator_breakdown,
            'total_contributions': {
                'advection_operators': sum(1 for op in operator_breakdown if op['operator_type'] == 'Advection'),
                'diffusion_operators': sum(1 for op in operator_breakdown if op['operator_type'] == 'Diffusion'),
                'mixed_operators': sum(1 for op in operator_breakdown if op['operator_type'] == 'Mixed')
            }
        }
        
        # Add ground truth comparison if available
        if ground_truth_params:
            ground_truth_v = ground_truth_params[0][0]  # Take first trajectory's parameters
            ground_truth_D = ground_truth_params[0][1]
            
            v_error = abs(total_v - ground_truth_v)
            D_error = abs(total_D - ground_truth_D)
            v_relative_error = v_error / max(abs(ground_truth_v), 1e-8) if ground_truth_v != 0 else float('inf') if total_v != 0 else 0.0
            D_relative_error = D_error / max(abs(ground_truth_D), 1e-8) if ground_truth_D != 0 else float('inf') if total_D != 0 else 0.0
            
            result['ground_truth_comparison'] = {
                'ground_truth_v': ground_truth_v,
                'ground_truth_D': ground_truth_D,
                'v_error': v_error,
                'D_error': D_error,
                'v_relative_error': v_relative_error,
                'D_relative_error': D_relative_error,
                'total_relative_error': (v_relative_error + D_relative_error) / 2
            }
            
            print(f"\nCoefficient Estimation Results:")
            print(f"  Ground Truth: v={ground_truth_v:.4f}, D={ground_truth_D:.4f}")
            print(f"  Estimated:    v={total_v:.4f}, D={total_D:.4f}")
            print(f"  Absolute Error: v_err={v_error:.4f}, D_err={D_error:.4f}")
            print(f"  Relative Error: v_rel={v_relative_error:.4f}, D_rel={D_relative_error:.4f}")
        
        return result
    
    def evaluate_model_with_best_operators(self, inp, target, theta, best_op1_idx, best_op2_idx, state_labels) -> Dict:
        """Evaluate the model using the best operator pair found during random sampling."""
        print("=" * 60)
        print("MODEL EVALUATION WITH BEST OPERATORS")
        print("=" * 60)
        
        x_test = inp[:, -1].to(self.device)
        
        # Get the best operators
        operator1 = get_batched_operators(theta[best_op1_idx:best_op1_idx+1], self.model, dim=1)
        operator2 = get_batched_operators(theta[best_op2_idx:best_op2_idx+1], self.model, dim=1)
        
        # Create simple alpha weights (equal weights)
        alpha1 = torch.ones(x_test.shape[0], 1, device=self.device)
        alpha2 = torch.ones(x_test.shape[0], 1, device=self.device)
        
        # Multi-step prediction with best operator pair
        pred = []
        with torch.no_grad():
            current = x_test
            for _ in range(self.n_output_frames):
                tmp = solve_mixture_ode(
                    current, operator1, alpha1, state_labels,
                    integration_time=1, n_future_steps=1)
                current = solve_mixture_ode(
                    tmp[-1], operator2, alpha2, state_labels,
                    integration_time=1, n_future_steps=1)
                pred.append(current)
                current = current[-1]
            
            pred = torch.cat(pred, axis=0)
        
        # Calculate error
        y_hat = pred.detach().cpu()
        y_true = rearrange(target.clone(), "b t c h -> t b c h").cpu()
        error = self.relative_l2_error(y_hat, y_true)
        
        print(f"Final extrapolation error with operators ({best_op1_idx}, {best_op2_idx}): {error.item():.6f}")
        
        return {
            'predictions': y_hat.numpy(),
            'ground_truth': y_true.numpy(),
            'error': error.item(),
            'best_operator_pair': (best_op1_idx, best_op2_idx)
        }
    
    
    def run_experiment(self) -> Dict:
        """Run the complete experiment."""
        print("Starting Neural Operator Splitting Experiment")
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Step 1: Encode structured operators (2*K from structured trajectories)
        K = self.config.get('num_operators', 4) // 2  # Divide by 2 since we get 2*K operators
        operator_v_range = self.config.get('operator_v_range', (0.1, 1.0))
        operator_D_range = self.config.get('operator_D_range', (0.1, 1.0))
        
        n_trajectories_per_operator = self.config.get('n_trajectories_per_operator', 1)
        theta, operator_inp, operator_target, operator_params = self.encode_structured_operators(
            K=K, n_trajectories_per_operator=n_trajectories_per_operator, 
            v_range=operator_v_range, D_range=operator_D_range
        )
        
        # Step 2: Get test data for optimization and evaluation  
        test_v_range = self.config.get('test_v_range', (0.5, 1.5))
        test_D_range = self.config.get('test_D_range', (0.5, 1.5))
        test_loader = self._create_dataset(split='test', v_range=test_v_range, D_range=test_D_range)
        
        # Collect enough test data trajectories
        all_inp, all_target = [], []
        trajectories_collected = 0
        n_test_trajectories = self.config.get('n_test_trajectories', 1)
        
        for batch in test_loader:
            inp_batch, target_batch = batch["input"], batch["target"]
            all_inp.append(inp_batch)
            all_target.append(target_batch)
            trajectories_collected += inp_batch.shape[0]
            
            if trajectories_collected >= n_test_trajectories:
                break
        
        # Concatenate and slice to exact number needed
        inp = torch.cat(all_inp, dim=0)[:n_test_trajectories].to(self.device)
        target = torch.cat(all_target, dim=0)[:n_test_trajectories].to(self.device)
        state_labels = torch.tensor([0], device=self.device)
        
        print(f"Using {theta.shape[0]} structured operators")
        print(f"Test data shape: {inp.shape}")
        
        # Random sampling to find best operator pair
        best_op1_idx, best_op2_idx, sampling_history = self.random_operator_sampling(
            inp, target, theta, state_labels
        )
        
        # Evaluation on test data using best operators
        eval_results = self.evaluate_model_with_best_operators(
            inp, target, theta, best_op1_idx, best_op2_idx, state_labels
        )
        
        # Get test parameters for ground truth
        test_loader = self._create_dataset(split='test', v_range=self.config['test_v_range'], D_range=self.config['test_D_range'])
        test_batch = next(iter(test_loader))
        ground_truth_params = [(v, D) for v, D in zip(test_batch["velocities"], test_batch["diffusivities"])][:self.config['n_test_trajectories']]
        
        # Estimate coefficients from best operator pair (composition)
        best_composition = [best_op1_idx, best_op2_idx]
        coefficient_estimates = self.estimate_coefficients(best_composition, operator_params, ground_truth_params)
        
        # Store all results
        results = {
            'operator_params': operator_params,
            'num_operators': theta.shape[0],
            'best_operators': (best_op1_idx, best_op2_idx),
            'best_composition': best_composition,
            'composition_length': len(best_composition),
            'sampling_history': sampling_history,
            'evaluation': eval_results,
            'coefficient_estimates': coefficient_estimates,
            'ground_truth_params': ground_truth_params,
            'config': self.config,
            'test_parameter_ranges': {
                'v_range': self.config['test_v_range'],
                'D_range': self.config['test_D_range']
            },
            'n_test_trajectories': self.config['n_test_trajectories'],
            'n_trajectories_per_operator': self.config['n_trajectories_per_operator']
        }
        
        return results
    
    def _compute_initial_error(self, inp, target, theta, state_labels):
        """Compute initial error before optimization."""
        with torch.no_grad():
            pred, metadata = self.model.solve_ode(
                inp[:, -1], theta, state_labels, dim=1,
                integration_time=self.n_output_frames, n_future_steps=self.n_output_frames, 
                predict_normed=False, metadata={}
            )
            return self.relative_l2_error(pred, target).item()


def create_trajectory_snapshots(predictions, ground_truth, output_dir: str, n_snapshots: int = 6):
    """
    Create trajectory snapshots showing prediction and ground truth in separate plots at different timestamps.
    
    Args:
        predictions: [time, batch, channel, spatial] prediction tensor
        ground_truth: [time, batch, channel, spatial] ground truth tensor
        output_dir: Directory to save plots
        n_snapshots: Number of time snapshots to show
    """
    sample_idx = 0  # Use first sample
    
    # Select time indices to plot
    time_indices = np.linspace(0, predictions.shape[0] - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    # Find global min/max for consistent color scaling
    vmin = min(np.min(predictions[:, sample_idx, 0, :]), np.min(ground_truth[:, sample_idx, 0, :]))
    vmax = max(np.max(predictions[:, sample_idx, 0, :]), np.max(ground_truth[:, sample_idx, 0, :]))
    
    # Create figure with 2 rows (prediction and ground truth) and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get spatial coordinates (assuming uniform grid)
    x = np.arange(predictions.shape[-1])
    
    # Row 1: Neural Prediction
    # Left: Heatmap of trajectory evolution
    pred_heatmap = predictions[:, sample_idx, 0, :].T  # Transpose for imshow
    im1 = axes[0, 0].imshow(pred_heatmap, aspect='auto', cmap='RdBu_r', 
                           extent=[0, predictions.shape[0]-1, x[-1], x[0]], 
                           vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Neural Prediction: Trajectory Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Spatial coordinate (x)')
    
    # Add vertical lines at snapshot times
    for t_idx in time_indices:
        axes[0, 0].axvline(x=t_idx, color='white', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Solution u(x,t)')
    
    # Right: Trajectory snapshots at different times
    for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
        axes[0, 1].plot(x, predictions[t_idx, sample_idx, 0, :], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f't = {t_idx}')
    
    axes[0, 1].set_title('Neural Prediction: Trajectory Snapshots')
    axes[0, 1].set_xlabel('Spatial coordinate (x)')
    axes[0, 1].set_ylabel('u(x,t)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Ground Truth
    # Left: Heatmap of trajectory evolution
    truth_heatmap = ground_truth[:, sample_idx, 0, :].T  # Transpose for imshow
    im2 = axes[1, 0].imshow(truth_heatmap, aspect='auto', cmap='RdBu_r', 
                           extent=[0, ground_truth.shape[0]-1, x[-1], x[0]], 
                           vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth: Trajectory Evolution')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Spatial coordinate (x)')
    
    # Add vertical lines at snapshot times
    for t_idx in time_indices:
        axes[1, 0].axvline(x=t_idx, color='white', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label('Solution u(x,t)')
    
    # Right: Trajectory snapshots at different times
    for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
        axes[1, 1].plot(x, ground_truth[t_idx, sample_idx, 0, :], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f't = {t_idx}')
    
    axes[1, 1].set_title('Ground Truth: Trajectory Snapshots')
    axes[1, 1].set_xlabel('Spatial coordinate (x)')
    axes[1, 1].set_ylabel('u(x,t)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_snapshots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_plots(results: Dict, output_dir: str):
    """Create comprehensive plots of the experiment results."""
    print("Creating plots...")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Random sampling results
    if 'sampling_history' in results:
        plt.figure(figsize=(15, 4))
        
        sampling_history = results['sampling_history']
        
        plt.subplot(1, 3, 1)
        plt.plot(sampling_history['errors'])
        plt.xlabel('Sample')
        plt.ylabel('Validation Error')
        plt.title('Random Sampling: Error per Sample')
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.plot(sampling_history['best_error_evolution'])
        plt.xlabel('Sample')
        plt.ylabel('Best Error So Far')
        plt.title('Random Sampling: Best Error Evolution')
        plt.yscale('log')
        
        # Plot operator pair frequency
        plt.subplot(1, 3, 3)
        pairs = sampling_history['operator_pairs']
        pair_counts = {}
        for pair in pairs:
            pair_key = f"({pair[0]},{pair[1]})"
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        
        # Show top 10 pairs
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        pair_labels, counts = zip(*sorted_pairs)
        
        plt.bar(range(len(pair_labels)), counts)
        plt.xticks(range(len(pair_labels)), pair_labels, rotation=45)
        plt.xlabel('Operator Pairs')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Sampled Operator Pairs')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'random_sampling.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Best operators and coefficient estimates
    if 'best_operators' in results and 'operator_params' in results:
        plt.figure(figsize=(15, 5))
        
        best_op1_idx, best_op2_idx = results['best_operators']
        operator_params = results['operator_params']
        
        # Plot operator parameters
        v_params = [param[0] for param in operator_params]
        d_params = [param[1] for param in operator_params]
        
        plt.subplot(1, 3, 1)
        plt.scatter(v_params, d_params, alpha=0.6, s=50, label='All operators')
        plt.scatter([operator_params[best_op1_idx][0]], [operator_params[best_op1_idx][1]], 
                   color='red', s=100, label=f'Best Op 1 (idx={best_op1_idx})')
        plt.scatter([operator_params[best_op2_idx][0]], [operator_params[best_op2_idx][1]], 
                   color='orange', s=100, label=f'Best Op 2 (idx={best_op2_idx})')
        plt.xlabel('Advection Parameter (v)')
        plt.ylabel('Diffusion Parameter (D)')
        plt.title('Operator Parameter Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        # Plot operator types
        op_types = []
        for v, D in operator_params:
            if D == 0:
                op_types.append('Advection')
            elif v == 0:
                op_types.append('Diffusion')
            else:
                op_types.append('Mixed')
        
        best_op1_type = op_types[best_op1_idx]
        best_op2_type = op_types[best_op2_idx]
        
        type_counts = {'Advection': op_types.count('Advection'), 
                      'Diffusion': op_types.count('Diffusion'),
                      'Mixed': op_types.count('Mixed')}
        
        colors = ['blue', 'green', 'purple']
        bars = plt.bar(type_counts.keys(), type_counts.values(), color=colors, alpha=0.6)
        
        # Highlight best operators
        plt.text(0.5, max(type_counts.values()) * 0.8, 
                f'Best pair: {best_op1_type} + {best_op2_type}', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.xlabel('Operator Type')
        plt.ylabel('Count')
        plt.title('Operator Type Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Coefficient estimates vs ground truth
        plt.subplot(1, 3, 3)
        if 'coefficient_estimates' in results and 'ground_truth_comparison' in results['coefficient_estimates']:
            coeff_data = results['coefficient_estimates']['ground_truth_comparison']
            
            # Create bar plot comparing estimated vs ground truth coefficients
            categories = ['Advection (v)', 'Diffusion (D)']
            ground_truth_values = [coeff_data['ground_truth_v'], coeff_data['ground_truth_D']]
            estimated_values = [results['coefficient_estimates']['estimated_v'], 
                              results['coefficient_estimates']['estimated_D']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, ground_truth_values, width, label='Ground Truth', alpha=0.7, color='blue')
            bars2 = plt.bar(x + width/2, estimated_values, width, label='Estimated', alpha=0.7, color='red')
            
            plt.xlabel('Coefficient Type')
            plt.ylabel('Coefficient Value')
            plt.title('Coefficient Estimates vs Ground Truth')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add text annotations with errors
            for i, (gt, est) in enumerate(zip(ground_truth_values, estimated_values)):
                error = abs(est - gt)
                rel_error = error / max(abs(gt), 1e-8) if gt != 0 else float('inf') if est != 0 else 0.0
                plt.text(i, max(gt, est) + 0.1 * max(ground_truth_values + estimated_values), 
                        f'Err: {error:.3f}\\nRel: {rel_error:.2f}', 
                        ha='center', va='bottom', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No coefficient\\ncomparison data', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.title('Coefficient Estimates vs Ground Truth')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'best_operators.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Trajectory snapshots for predictions and ground truth
    if 'evaluation' in results:
        eval_data = results['evaluation']
        predictions = eval_data['predictions']
        ground_truth = eval_data['ground_truth']
        
        # Create trajectory snapshots using the new function
        create_trajectory_snapshots(predictions, ground_truth, plots_dir)
        
        # Additional plot: Error evolution over time
        plt.figure(figsize=(10, 6))
        errors_over_time = []
        sample_idx = 0
        for t in range(predictions.shape[0]):
            pred_t = predictions[t, sample_idx, 0, :]
            true_t = ground_truth[t, sample_idx, 0, :]
            error_t = np.linalg.norm(pred_t - true_t) / np.linalg.norm(true_t)
            errors_over_time.append(error_t)
        
        plt.plot(errors_over_time, 'g-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Relative L2 Error')
        plt.title('Error Evolution Over Time')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {plots_dir}")


def save_random_sampling_results(results: Dict, output_dir: str):
    """Save experiment results using the organized results management approach."""
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'results_random_approach.json')
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    # Create enhanced summary similar to main_streamlined approach
    summary = {
        'method': 'random_approach',
        'approach_description': 'Random sampling operator selection method',
        'num_operators': results.get('num_operators', None),
        'best_operators': results.get('best_operators', []),
        'test_parameter_ranges': results.get('test_parameter_ranges', {}),
        'n_test_trajectories': results.get('n_test_trajectories', 1),
        'n_trajectories_per_operator': results.get('n_trajectories_per_operator', 1),
        'validation_error': min(results.get('sampling_history', {}).get('errors', [float('inf')])),
        'final_error': results.get('evaluation', {}).get('error', None),
        'best_operator_pair': results.get('evaluation', {}).get('best_operator_pair', None),
        'random_sampling_stats': {
            'n_samples': len(results.get('sampling_history', {}).get('errors', [])),
            'best_error_evolution': results.get('sampling_history', {}).get('best_error_evolution', [])
        }
    }
    
    # Calculate improvement if both errors available
    if summary['validation_error'] != float('inf') and summary['final_error'] is not None:
        summary['generalization_ratio'] = summary['final_error'] / summary['validation_error']
    
    summary_file = os.path.join(output_dir, 'summary_random_approach.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    
    print(f"Results saved to {output_dir}")
    val_str = f"{summary['validation_error']:.6f}" if summary['validation_error'] != float('inf') else "N/A"
    final_str = f"{summary['final_error']:.6f}" if summary['final_error'] is not None else "N/A"
    gen_ratio_str = f"{summary.get('generalization_ratio', 1.0):.2f}" if 'generalization_ratio' in summary else "N/A"
    
    print(f"Summary:")
    print(f"  Operators: {summary['num_operators']}, Best pair: {summary['best_operator_pair']}")
    print(f"  Validation error: {val_str}")
    print(f"  Final error:      {final_str}")
    print(f"  Generalization ratio: {gen_ratio_str}")
    print(f"  Random samples: {summary['random_sampling_stats']['n_samples']}")


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='Neural Operator Splitting Experiment')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to DISCO model checkpoint')
    parser.add_argument('--num-operators', type=int, default=4,
                        help='Number of operators to use (default: 4)')
    parser.add_argument('--n-trajectories-per-operator', type=int, default=1,
                        help='Number of trajectories per operator to avoid forgetting (default: 1)')
    parser.add_argument('--n-test-trajectories', type=int, default=1,
                        help='Number of test trajectories to use for fine-tuning (default: 1)')
    parser.add_argument('--rel-loss-coeff', type=float, default=1,
                        help='Coefficient for preservation loss (default: 0.01)')
    parser.add_argument('--sparsity-coeff', type=float, default=1e-2,
                        help='Coefficient for sparsity regularization (default: 1e-2)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of runs for statistics (default: 1)')
    parser.add_argument('--n-random-samples', type=int, default=1000,
                        help='Number of random operator pair samples (default: 1000)')
    # Test parameters
    parser.add_argument('--test-v-min', type=float, default=0.1, help='Test advection velocity min')
    parser.add_argument('--test-v-max', type=float, default=1.0, help='Test advection velocity max')
    parser.add_argument('--test-D-min', type=float, default=0.1, help='Test diffusion coefficient min')
    parser.add_argument('--test-D-max', type=float, default=1.0, help='Test diffusion coefficient max')
    parser.add_argument('--output-dir', type=str, default='./results/neural_operator_splitting',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create configuration
    config = {
        'checkpoint_path': args.checkpoint_path,
        'num_operators': args.num_operators,
        'rel_loss_2_coeff': args.rel_loss_coeff,
        'sparsity_coeff': args.sparsity_coeff,
        'n_random_samples': args.n_random_samples,
        'output_dir': args.output_dir,
        'seed': args.seed,
        'batch_size': 4,
        'n_input_frames': 16,
        'n_output_frames': 34,
        # Operator encoding parameters
        'n_trajectories_per_operator': args.n_trajectories_per_operator,  # Number of trajectories per operator to avoid forgetting
        'operator_v_range': (0.01, 1.0),
        'operator_D_range': (0.001, 1.0),
        # Test data parameters
        'n_test_trajectories': args.n_test_trajectories,  # Number of test trajectories for fine-tuning
        'test_v_range': (args.test_v_min, args.test_v_max),
        'test_D_range': (args.test_D_min, args.test_D_max),
    }
    
    # Create base timestamped results directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    config['output_dir'] = base_output_dir
    
    print("Neural Operator Splitting Experiment")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    if args.num_runs == 1:
        # Single run
        experiment = NeuralOperatorSplittingExperiment(config)
        results = experiment.run_experiment()
        
        # Organize results with improved structure
        organized_output_dir = organize_results_by_parameters(results, base_output_dir, timestamp, 1, args.n_test_trajectories, args.n_trajectories_per_operator)
        
        # Save results and create plots in organized directory
        save_random_sampling_results(results, organized_output_dir)
        create_plots(results, organized_output_dir)
        
        # Also save a copy in the timestamped directory for easy access
        save_random_sampling_results(results, config['output_dir'])
        
    else:
        # Multiple runs for statistics
        all_results = []
        
        for run_idx in range(args.num_runs):
            print(f"\n" + "="*60)
            print(f"RUN {run_idx + 1}/{args.num_runs}")
            print("="*60)
            
            # Update config for this run
            run_config = config.copy()
            run_config['seed'] = args.seed + run_idx
            
            # Set seed for this run
            torch.manual_seed(run_config['seed'])
            np.random.seed(run_config['seed'])
            random.seed(run_config['seed'])
            
            # Run experiment
            experiment = NeuralOperatorSplittingExperiment(run_config)
            results = experiment.run_experiment()
            
            # Organize results with improved structure
            organized_output_dir = organize_results_by_parameters(results, base_output_dir, f"{timestamp}_run_{run_idx:03d}", 1, args.n_test_trajectories, args.n_trajectories_per_operator)
            
            # Save individual run results in organized structure
            save_random_sampling_results(results, organized_output_dir)
            create_plots(results, organized_output_dir)
            
            all_results.append(results)
        
        # Aggregate statistics for random sampling approach
        print(f"\n" + "="*60)
        print("AGGREGATING STATISTICS")
        print("="*60)
        
        validation_errors = [min(r['sampling_history']['errors']) for r in all_results]
        final_errors = [r['evaluation']['error'] for r in all_results]
        n_samples_list = [len(r['sampling_history']['errors']) for r in all_results]
        best_operator_pairs = [r['best_operators'] for r in all_results]
        
        # Extract coefficient estimation statistics
        coefficient_stats = []
        for r in all_results:
            if 'coefficient_estimates' in r and 'ground_truth_comparison' in r['coefficient_estimates']:
                coeff_data = r['coefficient_estimates']['ground_truth_comparison']
                coefficient_stats.append({
                    'v_error': coeff_data['v_error'],
                    'D_error': coeff_data['D_error'], 
                    'v_relative_error': coeff_data['v_relative_error'],
                    'D_relative_error': coeff_data['D_relative_error'],
                    'total_relative_error': coeff_data['total_relative_error'],
                    'estimated_v': r['coefficient_estimates']['estimated_v'],
                    'estimated_D': r['coefficient_estimates']['estimated_D'],
                    'ground_truth_v': coeff_data['ground_truth_v'],
                    'ground_truth_D': coeff_data['ground_truth_D']
                })
        
        # Calculate coefficient estimation aggregate statistics
        coeff_aggregate = {}
        if coefficient_stats:
            coeff_aggregate = {
                'v_error': {
                    'mean': np.mean([c['v_error'] for c in coefficient_stats]),
                    'std': np.std([c['v_error'] for c in coefficient_stats]),
                    'min': np.min([c['v_error'] for c in coefficient_stats]),
                    'max': np.max([c['v_error'] for c in coefficient_stats])
                },
                'D_error': {
                    'mean': np.mean([c['D_error'] for c in coefficient_stats]),
                    'std': np.std([c['D_error'] for c in coefficient_stats]),
                    'min': np.min([c['D_error'] for c in coefficient_stats]),
                    'max': np.max([c['D_error'] for c in coefficient_stats])
                },
                'v_relative_error': {
                    'mean': np.mean([c['v_relative_error'] for c in coefficient_stats if c['v_relative_error'] != float('inf')]),
                    'std': np.std([c['v_relative_error'] for c in coefficient_stats if c['v_relative_error'] != float('inf')]),
                    'min': np.min([c['v_relative_error'] for c in coefficient_stats if c['v_relative_error'] != float('inf')]),
                    'max': np.max([c['v_relative_error'] for c in coefficient_stats if c['v_relative_error'] != float('inf')])
                },
                'D_relative_error': {
                    'mean': np.mean([c['D_relative_error'] for c in coefficient_stats if c['D_relative_error'] != float('inf')]),
                    'std': np.std([c['D_relative_error'] for c in coefficient_stats if c['D_relative_error'] != float('inf')]),
                    'min': np.min([c['D_relative_error'] for c in coefficient_stats if c['D_relative_error'] != float('inf')]),
                    'max': np.max([c['D_relative_error'] for c in coefficient_stats if c['D_relative_error'] != float('inf')])
                },
                'total_relative_error': {
                    'mean': np.mean([c['total_relative_error'] for c in coefficient_stats]),
                    'std': np.std([c['total_relative_error'] for c in coefficient_stats]),
                    'min': np.min([c['total_relative_error'] for c in coefficient_stats]),
                    'max': np.max([c['total_relative_error'] for c in coefficient_stats])
                },
                'num_runs_with_coeff_data': len(coefficient_stats)
            }
        
        stats = {
            'method': 'random_approach',
            'approach_description': 'Random sampling operator selection method - aggregate statistics',
            'num_runs': args.num_runs,
            'validation_error': {
                'mean': np.mean(validation_errors),
                'std': np.std(validation_errors),
                'min': np.min(validation_errors),
                'max': np.max(validation_errors)
            },
            'final_error': {
                'mean': np.mean(final_errors),
                'std': np.std(final_errors),
                'min': np.min(final_errors),
                'max': np.max(final_errors)
            },
            'generalization_ratio': {
                'mean': np.mean([f/v for f, v in zip(final_errors, validation_errors)]),
                'std': np.std([f/v for f, v in zip(final_errors, validation_errors)]),
                'min': np.min([f/v for f, v in zip(final_errors, validation_errors)]),
                'max': np.max([f/v for f, v in zip(final_errors, validation_errors)])
            },
            'random_sampling_stats': {
                'n_samples_per_run': {
                    'mean': np.mean(n_samples_list),
                    'std': np.std(n_samples_list),
                    'min': np.min(n_samples_list),
                    'max': np.max(n_samples_list)
                },
                'best_operator_pairs': best_operator_pairs
            },
            'coefficient_estimation_stats': coeff_aggregate,
            'test_parameter_ranges': all_results[0].get('test_parameter_ranges', {}) if all_results else {}
        }
        
        # Organize aggregate directory and save statistics
        first_result = all_results[0] if all_results else {}
        param_ranges = first_result.get('test_parameter_ranges', {})
        v_range = param_ranges.get('v_range', (0, 1))
        D_range = param_ranges.get('D_range', (0, 1))
        v_min, v_max = v_range
        D_min, D_max = D_range
        
        param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
        operator_dir = f"operators_{first_result.get('num_operators', 0)}"
        test_traj_dir = f"test_traj_{args.n_test_trajectories}"
        traj_per_op_dir = f"traj_per_op_{args.n_trajectories_per_operator}"
        integration_dir = f"integration_steps_1"
        
        aggregate_parent = os.path.join("./results/neural_operator_splitting", param_dir, operator_dir, test_traj_dir, traj_per_op_dir, integration_dir)
        os.makedirs(aggregate_parent, exist_ok=True)
        
        # Save aggregate statistics in organized structure  
        stats_file = os.path.join(aggregate_parent, 'aggregate_statistics_random_sampling.json')
        with open(stats_file, 'w') as f:
            json.dump(convert_for_json(stats), f, indent=2)
        
        # Also save copy in timestamped directory for backward compatibility
        stats_file_old = os.path.join(base_output_dir, 'aggregate_statistics_random_approach.json')  
        with open(stats_file_old, 'w') as f:
            json.dump(convert_for_json(stats), f, indent=2)
        
        print(f"Aggregate Statistics ({args.num_runs} runs):")
        print(f"  Validation Error:     {stats['validation_error']['mean']:.6f} ± {stats['validation_error']['std']:.6f}")
        print(f"  Final Error:          {stats['final_error']['mean']:.6f} ± {stats['final_error']['std']:.6f}")
        print(f"  Generalization Ratio: {stats['generalization_ratio']['mean']:.2f} ± {stats['generalization_ratio']['std']:.2f}")
        print(f"  Best Final Error:     {stats['final_error']['min']:.6f}")
        print(f"  Random Samples/Run:   {stats['random_sampling_stats']['n_samples_per_run']['mean']:.0f} ± {stats['random_sampling_stats']['n_samples_per_run']['std']:.0f}")
        print(f"  Parameter Range: v=[{v_min:.1f},{v_max:.1f}], D=[{D_min:.3f},{D_max:.3f}]")
        
        # Print coefficient estimation statistics if available
        if coeff_aggregate:
            print(f"\\nCoefficient Estimation Statistics:")
            print(f"  Advection Error (v):  {coeff_aggregate['v_error']['mean']:.4f} ± {coeff_aggregate['v_error']['std']:.4f}")
            print(f"  Diffusion Error (D):  {coeff_aggregate['D_error']['mean']:.4f} ± {coeff_aggregate['D_error']['std']:.4f}")
            print(f"  Advection Rel Error:  {coeff_aggregate['v_relative_error']['mean']:.4f} ± {coeff_aggregate['v_relative_error']['std']:.4f}")
            print(f"  Diffusion Rel Error:  {coeff_aggregate['D_relative_error']['mean']:.4f} ± {coeff_aggregate['D_relative_error']['std']:.4f}")
            print(f"  Total Rel Error:      {coeff_aggregate['total_relative_error']['mean']:.4f} ± {coeff_aggregate['total_relative_error']['std']:.4f}")
            print(f"  Runs with coeff data: {coeff_aggregate['num_runs_with_coeff_data']}/{args.num_runs}")
        
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config['output_dir'] if args.num_runs == 1 else base_output_dir}")


if __name__ == "__main__":
    main()