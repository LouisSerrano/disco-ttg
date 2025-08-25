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
from src.advection_diffusion import Fractaloid, FractaloidPhase
from train.train import DISCOLitModule, advection_diffusion_analytical
from src.plot_dataset_samples import plot_prediction_vs_ground_truth


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


def sequential_operator_composition(x, state_labels, operator_indices, theta, model, 
                                   integration_time=1.0, n_future_steps=1, solver='rk4', rtol=1e-7, idx_to_theta=None):
    """
    Apply operator composition at each time step Dt (where integration_time=1 advances by one Dt).
    
    Args:
        x: input tensor
        state_labels: state labels
        operator_indices: list of operator indices to apply sequentially
        theta: tensor of all operator parameters
        model: DISCO model
        integration_time: time step (1.0 = one Dt advance)
        n_future_steps: number of future time steps
        solver: ODE solver
        rtol: relative tolerance
    
    Returns:
        Tensor of shape [n_future_steps, batch, channel, height] for n_future_steps > 1,
        or [batch, channel, height] for n_future_steps = 1
    """
    current = x
    predictions = []
    
    # Apply the full composition at each time step
    for step in range(n_future_steps):
        # Apply all operators in sequence for this time step (each with integration_time=1)
        step_result = current
        for op_idx in operator_indices:
            # Get single operator - use mapping if provided, otherwise direct indexing
            if idx_to_theta is not None:
                theta_idx = idx_to_theta[op_idx]
                single_operator = get_batched_operators(theta[theta_idx:theta_idx+1], model, dim=1)
            else:
                single_operator = get_batched_operators(theta[op_idx:op_idx+1], model, dim=1)
            
            # Apply current operator for one Dt step
            step_result = solve_single_operator_ode(
                step_result, single_operator, state_labels,
                integration_time=1, n_future_steps=1,
                solver=solver, rtol=rtol
            )
            step_result = step_result[0]  # Take the single time step result
        
        # Store the result of this full composition step
        predictions.append(step_result)
        current = step_result  # Use this as input for next time step
    
    if n_future_steps == 1:
        return predictions[0]
    else:
        # Stack predictions: [n_future_steps, batch, channel, height]
        return torch.stack(predictions, dim=0)




def solve_single_operator_ode(x_input, operator, state_labels, 
                             integration_time=1.0, n_future_steps=1, 
                             solver='rk4', rtol=1e-7):
    """
    Solve neural ODE using a single operator (no mixing).
    """
    # Create ODE function for single operator
    def ode_func(t, x):
        # operator returns [1, batch_size, channels, ...spatial] since we use single operator
        # We need to squeeze the first dimension
        return operator(x, state_labels).squeeze(0)
    
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
    
    def greedy_iterative_operator_selection(self, inp, target, theta, state_labels, max_operators=5) -> Tuple[List[int], Dict]:
        """
        Greedy iterative approach: at each step, test ALL operators and select the best one to add.
        
        Args:
            inp: input data
            target: target data
            theta: operator parameters
            state_labels: state labels
            max_operators: maximum number of operators in composition (default: 5)
        
        Returns:
            Best operator composition (list of indices) and history
        """
        print("=" * 60)
        print("GREEDY ITERATIVE OPERATOR SELECTION")
        print("=" * 60)
        
        num_operators = theta.shape[0]
        
        # Use a random time step for validation
        val_t = random.randint(0, self.n_input_frames - 2)
        x_val = inp[:, val_t].to(self.device)
        y_val = inp[:, val_t+1].to(self.device)
        
        # Initialize tracking
        current_composition = []
        current_best_error = float('inf')
        
        history = {
            'compositions': [],
            'errors': [],
            'best_composition_per_length': {},
            'improvement_per_step': [],
            'step_details': []
        }
        
        print(f"Testing ALL {num_operators} operators at each step...")
        print(f"Maximum composition length: {max_operators}")
        
        # Iteratively build composition
        for comp_length in range(1, max_operators + 1):
            print(f"\n--- Step {comp_length}: Testing all operators ---")
            
            best_error_for_step = float('inf')
            best_composition_for_step = None
            best_operator_added = None
            step_candidates = []
            
            # Test ALL operators at this step
            for op_idx in range(num_operators):
                # Create new composition by adding this operator
                composition = current_composition + [op_idx]
                
                # Evaluate this composition
                with torch.no_grad():
                    try:
                        pred = sequential_operator_composition(
                            x_val, state_labels, composition, theta, self.model,
                            integration_time=1.0, n_future_steps=1
                        )
                        error = self.relative_l2_error(pred, y_val).item()
                        
                        # Store this candidate
                        candidate_info = {
                            'operator_added': op_idx,
                            'composition': composition.copy(),
                            'error': error
                        }
                        step_candidates.append(candidate_info)
                        
                        history['compositions'].append(composition.copy())
                        history['errors'].append(error)
                        
                        # Check if this is the best for this step
                        if error < best_error_for_step:
                            best_error_for_step = error
                            best_composition_for_step = composition.copy()
                            best_operator_added = op_idx
                        
                        if op_idx % 10 == 0:
                            print(f"  Tested operator {op_idx}/{num_operators-1}, error: {error:.6f}")
                        
                    except Exception as e:
                        print(f"Error evaluating operator {op_idx}: {e}")
                        continue
            
            # Store step details
            step_info = {
                'step': comp_length,
                'candidates_tested': len(step_candidates),
                'best_operator_added': best_operator_added,
                'best_error': best_error_for_step,
                'all_candidates': step_candidates
            }
            history['step_details'].append(step_info)
            
            # Update best composition for this length
            history['best_composition_per_length'][comp_length] = {
                'composition': best_composition_for_step,
                'error': best_error_for_step,
                'candidates_tested': len(step_candidates)
            }
            
            # Calculate improvement from previous step
            if comp_length == 1:
                improvement = 0  # No previous to compare to
                print(f"  Step {comp_length}: Best operator = {best_operator_added}, error = {best_error_for_step:.6f}")
            else:
                improvement = (current_best_error - best_error_for_step) / current_best_error * 100
                print(f"  Step {comp_length}: Best operator = {best_operator_added}, error = {best_error_for_step:.6f} "
                      f"(improvement: {improvement:+.2f}%)")
            
            history['improvement_per_step'].append(improvement)
            
            # Decide whether to continue: only if we got improvement or this is the first step
            if best_error_for_step < current_best_error:
                print(f"  ✓ Improvement found! Adding operator {best_operator_added}")
                print(f"    New composition: {best_composition_for_step}")
                current_composition = best_composition_for_step.copy()
                current_best_error = best_error_for_step
            else:
                print(f"  ✗ No improvement found. Stopping at length {comp_length - 1}")
                print(f"  Final composition: {current_composition}")
                break
        
        print(f"\nGreedy selection completed!")
        print(f"Final composition: {current_composition} with error: {current_best_error:.6f}")
        print(f"Composition length: {len(current_composition)}")
        
        return current_composition, history
    
    def evaluate_model_with_composition(self, inp, target, theta, best_composition, state_labels) -> Dict:
        """Evaluate the model using the best operator composition found during greedy selection."""
        print("=" * 60)
        print("MODEL EVALUATION WITH BEST COMPOSITION")
        print("=" * 60)
        
        x_test = inp[:, -1].to(self.device)
        
        print(f"Evaluating with composition: {best_composition}")
        print(f"Composition length: {len(best_composition)}")
        
        # Multi-step prediction with best operator composition
        pred = []
        with torch.no_grad():
            current = x_test
            for step in range(self.n_output_frames):
                # Apply the full composition for this time step
                current = sequential_operator_composition(
                    current, state_labels, best_composition, theta, self.model,
                    integration_time=1.0, n_future_steps=1
                )
                pred.append(current.unsqueeze(0))  # Add time dimension
                
                if (step + 1) % 10 == 0:
                    print(f"  Completed {step + 1}/{self.n_output_frames} time steps")
            
            pred = torch.cat(pred, axis=0)
        
        # Calculate error
        y_hat = pred.detach().cpu()
        y_true = rearrange(target.clone(), "b t c h -> t b c h").cpu()
        error = self.relative_l2_error(y_hat, y_true)
        
        print(f"Final extrapolation error with composition {best_composition}: {error.item():.6f}")
        
        return {
            'predictions': y_hat.numpy(),
            'ground_truth': y_true.numpy(),
            'error': error.item(),
            'best_composition': best_composition,
            'composition_length': len(best_composition)
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
        
        # Greedy iterative selection to find best operator composition
        max_operators = self.config.get('max_operators', 5)
        best_composition, selection_history = self.greedy_iterative_operator_selection(
            inp, target, theta, state_labels, max_operators=max_operators
        )
        
        # Evaluation on test data using best composition
        eval_results = self.evaluate_model_with_composition(
            inp, target, theta, best_composition, state_labels
        )
        
        # Store all results
        results = {
            'operator_params': operator_params,
            'num_operators': theta.shape[0],
            'best_composition': best_composition,
            'composition_length': len(best_composition),
            'selection_history': selection_history,
            'evaluation': eval_results,
            'config': self.config
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
    
    # Plot 1: Greedy selection results
    if 'selection_history' in results:
        plt.figure(figsize=(15, 10))
        
        selection_history = results['selection_history']
        
        # Plot composition errors over time
        plt.subplot(2, 3, 1)
        plt.plot(selection_history['errors'])
        plt.xlabel('Composition Evaluated')
        plt.ylabel('Validation Error')
        plt.title('Greedy Selection: Error per Composition')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot improvement per step
        plt.subplot(2, 3, 2)
        improvements = selection_history['improvement_per_step']
        composition_lengths = list(range(1, len(improvements) + 1))
        plt.bar(composition_lengths, improvements)
        plt.xlabel('Composition Length')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement per Composition Length')
        plt.grid(True, alpha=0.3)
        
        # Plot best error per length
        plt.subplot(2, 3, 3)
        best_per_length = selection_history['best_composition_per_length']
        lengths = list(best_per_length.keys())
        errors = [best_per_length[l]['error'] for l in lengths]
        plt.plot(lengths, errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Composition Length')
        plt.ylabel('Best Error')
        plt.title('Best Error vs Composition Length')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot composition frequencies
        plt.subplot(2, 3, 4)
        compositions = selection_history['compositions']
        comp_lengths = [len(comp) for comp in compositions]
        length_counts = {}
        for length in comp_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        lengths = sorted(length_counts.keys())
        counts = [length_counts[l] for l in lengths]
        plt.bar(lengths, counts)
        plt.xlabel('Composition Length')
        plt.ylabel('Number of Compositions Tested')
        plt.title('Compositions Tested by Length')
        plt.grid(True, alpha=0.3)
        
        # Plot candidates tested per length
        plt.subplot(2, 3, 5)
        lengths = list(best_per_length.keys())
        candidates = [best_per_length[l]['candidates_tested'] for l in lengths]
        plt.bar(lengths, candidates, alpha=0.7)
        plt.xlabel('Composition Length')
        plt.ylabel('Candidates Tested')
        plt.title('Search Effort per Length')
        plt.grid(True, alpha=0.3)
        
        # Show final best composition
        plt.subplot(2, 3, 6)
        best_composition = results.get('best_composition', [])
        if best_composition:
            plt.bar(range(len(best_composition)), best_composition, alpha=0.7)
            plt.xlabel('Position in Composition')
            plt.ylabel('Operator Index')
            plt.title(f'Best Composition: {best_composition}')
            plt.xticks(range(len(best_composition)))
            for i, op_idx in enumerate(best_composition):
                plt.text(i, op_idx + 0.1, str(op_idx), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'greedy_selection.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Best composition information
    if 'best_composition' in results and 'operator_params' in results:
        plt.figure(figsize=(15, 4))
        
        best_composition = results['best_composition']
        operator_params = results['operator_params']
        
        # Plot operator parameters with best composition highlighted
        v_params = [param[0] for param in operator_params]
        d_params = [param[1] for param in operator_params]
        
        plt.subplot(1, 3, 1)
        plt.scatter(v_params, d_params, alpha=0.6, s=50, label='All operators')
        
        # Highlight operators in best composition
        colors = plt.cm.Set1(np.linspace(0, 1, len(best_composition)))
        for i, op_idx in enumerate(best_composition):
            plt.scatter([operator_params[op_idx][0]], [operator_params[op_idx][1]], 
                       color=colors[i], s=150, 
                       label=f'Pos {i+1}: Op {op_idx}', 
                       edgecolor='black', linewidth=2)
        
        plt.xlabel('Advection Parameter (v)')
        plt.ylabel('Diffusion Parameter (D)')
        plt.title('Operator Parameter Space')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        # Plot operator types in composition
        op_types = []
        for v, D in operator_params:
            if D == 0:
                op_types.append('Advection')
            elif v == 0:
                op_types.append('Diffusion')
            else:
                op_types.append('Mixed')
        
        comp_types = [op_types[idx] for idx in best_composition]
        type_counts = {'Advection': comp_types.count('Advection'), 
                      'Diffusion': comp_types.count('Diffusion'),
                      'Mixed': comp_types.count('Mixed')}
        
        colors = ['blue', 'green', 'purple']
        bars = plt.bar(type_counts.keys(), type_counts.values(), color=colors, alpha=0.6)
        
        plt.xlabel('Operator Type')
        plt.ylabel('Count in Best Composition')
        plt.title('Operator Types in Best Composition')
        plt.grid(True, alpha=0.3)
        
        # Plot composition sequence
        plt.subplot(1, 3, 3)
        position_labels = [f"Pos {i+1}" for i in range(len(best_composition))]
        colors = ['red' if op_types[idx] == 'Advection' else 'blue' if op_types[idx] == 'Diffusion' else 'green' 
                 for idx in best_composition]
        
        bars = plt.bar(position_labels, best_composition, color=colors, alpha=0.7)
        plt.xlabel('Position in Composition')
        plt.ylabel('Operator Index')
        plt.title('Composition Sequence')
        
        # Add value labels on bars
        for bar, op_idx in zip(bars, best_composition):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(op_idx), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'best_composition.png'), dpi=300, bbox_inches='tight')
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


def save_results(results: Dict, output_dir: str):
    """Save experiment results."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    # Save summary
    summary = {
        'initial_error': results.get('initial_error', None),
        'final_error': results.get('evaluation', {}).get('error', None),
        'improvement': None,
        'composition_length': results.get('composition_length', None)
    }
    
    if summary['initial_error'] and summary['final_error']:
        summary['improvement'] = summary['initial_error'] / summary['final_error']
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    
    print(f"Results saved to {output_dir}")
    initial_str = f"{summary['initial_error']:.6f}" if summary['initial_error'] is not None else "N/A"
    final_str = f"{summary['final_error']:.6f}" if summary['final_error'] is not None else "N/A"
    improvement_str = f"{summary['improvement']:.2f}x" if summary['improvement'] is not None else "N/A"
    print(f"Summary: Initial error: {initial_str}, Final error: {final_str}, Improvement: {improvement_str}")


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
    parser.add_argument('--max-operators', type=int, default=5,
                        help='Maximum number of operators in composition (default: 5)')
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
        'max_operators': args.max_operators,
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
        #'test_v_range': (2.9, 3.00),#(0.9, 1.0),
        #'test_D_range': (0, 0),#(0.9, 1.0),
        'test_v_range': (1.9, 2.00),#(0.9, 1.0),
        'test_D_range': (0, 0),#(0.9, 1.0),
        #'test_v_range': (0.5, 0.5),
        #'test_D_range': (0.5, 0.5),
    }
    
    # Create timestamped results directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = os.path.join(args.output_dir, f'run_{timestamp}')
    
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
        
        # Save results and create plots
        save_results(results, config['output_dir'])
        create_plots(results, config['output_dir'])
        
    else:
        # Multiple runs for statistics
        all_results = []
        base_output_dir = config['output_dir']
        
        for run_idx in range(args.num_runs):
            print(f"\n" + "="*60)
            print(f"RUN {run_idx + 1}/{args.num_runs}")
            print("="*60)
            
            # Update config for this run
            run_config = config.copy()
            run_config['output_dir'] = os.path.join(base_output_dir, f'run_{run_idx:03d}')
            run_config['seed'] = args.seed + run_idx
            
            # Set seed for this run
            torch.manual_seed(run_config['seed'])
            np.random.seed(run_config['seed'])
            random.seed(run_config['seed'])
            
            # Run experiment
            experiment = NeuralOperatorSplittingExperiment(run_config)
            results = experiment.run_experiment()
            
            # Save individual run results
            save_results(results, run_config['output_dir'])
            create_plots(results, run_config['output_dir'])
            
            all_results.append(results)
        
        # Aggregate statistics
        print(f"\n" + "="*60)
        print("AGGREGATING STATISTICS")
        print("="*60)
        
        validation_errors = [min(r['selection_history']['errors']) for r in all_results]
        final_errors = [r['evaluation']['error'] for r in all_results]
        composition_lengths = [r['composition_length'] for r in all_results]
        
        stats = {
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
            'composition_length': {
                'mean': np.mean(composition_lengths),
                'std': np.std(composition_lengths),
                'min': np.min(composition_lengths),
                'max': np.max(composition_lengths)
            },
            'generalization_ratio': {
                'mean': np.mean([f/v for f, v in zip(final_errors, validation_errors)]),
                'std': np.std([f/v for f, v in zip(final_errors, validation_errors)]),
                'min': np.min([f/v for f, v in zip(final_errors, validation_errors)]),
                'max': np.max([f/v for f, v in zip(final_errors, validation_errors)])
            }
        }
        
        # Save aggregate statistics
        stats_file = os.path.join(base_output_dir, 'aggregate_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Aggregate Statistics ({args.num_runs} runs):")
        print(f"  Validation Error: {stats['validation_error']['mean']:.6f} ± {stats['validation_error']['std']:.6f}")
        print(f"  Final Error:      {stats['final_error']['mean']:.6f} ± {stats['final_error']['std']:.6f}")
        print(f"  Composition Length: {stats['composition_length']['mean']:.1f} ± {stats['composition_length']['std']:.1f}")
        print(f"  Generalization Ratio: {stats['generalization_ratio']['mean']:.2f} ± {stats['generalization_ratio']['std']:.2f}")
        print(f"  Best Final Error: {stats['final_error']['min']:.6f}")
        print(f"  Most Common Length: {int(np.round(stats['composition_length']['mean']))}")
        print(f"  Length Range: {int(stats['composition_length']['min'])} - {int(stats['composition_length']['max'])}")
        
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config['output_dir'] if args.num_runs == 1 else base_output_dir}")


if __name__ == "__main__":
    main()