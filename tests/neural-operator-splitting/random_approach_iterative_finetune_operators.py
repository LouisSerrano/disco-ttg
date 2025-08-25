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
from models import DISCOHouse, vectors_to_parameters
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
            # Reset parameter index at start of each batch for fixed_params_mode
            if self.fixed_params_mode:
                self.param_index = 0
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
                                   integration_time=1.0, n_future_steps=1, solver='rk4', rtol=1e-7, idx_to_theta=None, num_integration_steps=1):
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
        idx_to_theta: optional mapping from operator indices to theta indices
        num_integration_steps: number of sub-steps for finer dt integration (default: 1)
    
    Returns:
        Tensor of shape [n_future_steps, batch, channel, height] for n_future_steps > 1,
        or [batch, channel, height] for n_future_steps = 1
    """
    current = x
    predictions = []
    
    # Calculate finer dt for each operator application
    dt_per_operator = integration_time / num_integration_steps
    
    # Apply the full composition at each time step
    for step in range(n_future_steps):
        step_result = current
        
        # For finer integration, apply the full composition num_integration_steps times with smaller dt
        for substep in range(num_integration_steps):
            # Apply all operators in sequence for this sub-step
            for op_idx in operator_indices:
                # Get single operator - use mapping if provided, otherwise direct indexing
                if idx_to_theta is not None:
                    theta_idx = idx_to_theta[op_idx]
                    single_operator = get_batched_operators(theta[theta_idx:theta_idx+1], model, dim=1)
                else:
                    single_operator = get_batched_operators(theta[op_idx:op_idx+1], model, dim=1)
                
                # Apply current operator for smaller dt step
                step_result = solve_single_operator_ode(
                    step_result, single_operator, state_labels,
                    integration_time=dt_per_operator, n_future_steps=1,
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
    
    def encode_structured_operators(self, K=4, n_trajectories_per_operator=1, v_range=(0.1, 1.0), D_range=(0.1, 1.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, dict, torch.Tensor, torch.Tensor]:
        """
        Encode 2*K operators from structured trajectories.
        
        Args:
            K: Number of operators per type (advection/diffusion)
            n_trajectories_per_operator: Number of trajectories to generate per operator (default: 1)
            v_range: Range for advection velocities
            D_range: Range for diffusion coefficients
        
        Returns:
            theta: [2*K, theta_dim] tensor of encoded operators (averaged)
            theta_latent: [2*K, latent_dim] tensor of latent encodings (averaged)
            inp_trajectories: [2*K*n_trajectories_per_operator, T, C, H] input trajectories for finetuning
            target_trajectories: [2*K*n_trajectories_per_operator, T_out, C, H] target trajectories for finetuning
            params: list of (v, D) tuples for each operator
            operator_to_trajectories: dict mapping operator index to list of trajectory indices
            theta_all_individual: [2*K*n_trajectories_per_operator, theta_dim] individual trajectory thetas
            theta_latent_all_individual: [2*K*n_trajectories_per_operator, latent_dim] individual trajectory theta_latents
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
            theta_latent_all, metadata = self.model.encode_theta_latent(all_inp_trajectories, state_labels)
            theta_all = self.model.decode_theta(theta_latent_all, dim=1)
            
            # Store the individual trajectory thetas before averaging
            theta_all_individual = theta_all.clone()
            theta_latent_all_individual = theta_latent_all.clone()
            
            if n_trajectories_per_operator == 1:
                # No averaging needed - each trajectory corresponds to one operator
                theta = theta_all
                theta_latent = theta_latent_all
            else:
                # Reshape and average both theta and theta_latent parameters correctly
                # theta_all shape: [n_trajectories_per_operator * total_operators, theta_dim]
                # theta_latent_all shape: [n_trajectories_per_operator * total_operators, latent_dim]
                # Reshape to: [n_trajectories_per_operator, total_operators, -1]
                # Then average over trajectory dimension (dim=0)
                theta = theta_all.view(n_trajectories_per_operator, total_operators, -1).mean(dim=0)
                theta_latent = theta_latent_all.view(n_trajectories_per_operator, total_operators, -1).mean(dim=0)
        
        print(f"Encoded operators shape: {theta.shape}")
        print(f"Encoded latent operators shape: {theta_latent.shape}")
        print(f"Input trajectories shape: {all_inp_trajectories.shape}")
        print(f"Target trajectories shape: {all_target_trajectories.shape}")
        
        # Create operator-to-trajectories mapping for easier sampling
        # trajectories: [op0_traj0, op1_traj0, op2_traj0, op3_traj0, op0_traj1, op1_traj1, ...]
        # operator_to_trajectories: {0: [0, 4, 8, ...], 1: [1, 5, 9, ...], ...}
        operator_to_trajectories = {}
        for op_idx in range(total_operators):
            operator_to_trajectories[op_idx] = []
            for traj_batch in range(n_trajectories_per_operator):
                traj_global_idx = traj_batch * total_operators + op_idx
                operator_to_trajectories[op_idx].append(traj_global_idx)
        
        # Create unique params list for operators (remove duplicates)
        unique_params = []
        for op_idx in range(total_operators):
            start_idx = op_idx * n_trajectories_per_operator
            unique_params.append(all_params[start_idx])  # Take first trajectory's params for each operator
        
        return theta, theta_latent, all_inp_trajectories, all_target_trajectories, unique_params, operator_to_trajectories, theta_all_individual, theta_latent_all_individual
    
    def greedy_iterative_operator_selection(self, inp, target, theta, state_labels, max_operators=5, min_improvement_threshold=5.0) -> Tuple[List[int], Dict]:
        """
        Greedy iterative approach: at each step, test ALL operators and select the best one to add.
        
        Args:
            inp: input data
            target: target data
            theta: operator parameters
            state_labels: state labels
            max_operators: maximum number of operators in composition (default: 5)
            min_improvement_threshold: minimum improvement percentage required to add operator (default: 5.0)
        
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
        print(f"Minimum improvement threshold: {min_improvement_threshold}%")
        
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
                            integration_time=1.0, n_future_steps=1,
                            num_integration_steps=self.config.get('num_integration_steps', 1)
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
                # Always accept first operator
                should_continue = True
            else:
                improvement = (current_best_error - best_error_for_step) / current_best_error * 100
                print(f"  Step {comp_length}: Best operator = {best_operator_added}, error = {best_error_for_step:.6f} "
                      f"(improvement: {improvement:+.2f}%)")
                # Only continue if improvement meets threshold
                should_continue = improvement >= min_improvement_threshold
            
            history['improvement_per_step'].append(improvement)
            
            # Decide whether to continue: only if we got sufficient improvement or this is the first step
            if should_continue and best_error_for_step < current_best_error:
                if comp_length == 1:
                    print(f"  ✓ Adding first operator {best_operator_added}")
                else:
                    print(f"  ✓ Improvement {improvement:.2f}% >= {min_improvement_threshold}% threshold! Adding operator {best_operator_added}")
                print(f"    New composition: {best_composition_for_step}")
                current_composition = best_composition_for_step.copy()
                current_best_error = best_error_for_step
            elif best_error_for_step >= current_best_error:
                print(f"  ✗ No improvement found. Stopping at length {comp_length - 1}")
                print(f"  Final composition: {current_composition}")
                break
            else:
                print(f"  ✗ Improvement {improvement:.2f}% < {min_improvement_threshold}% threshold. Stopping at length {comp_length - 1}")
                print(f"  Final composition: {current_composition}")
                break
        
        print(f"\nGreedy selection completed!")
        print(f"Final composition: {current_composition} with error: {current_best_error:.6f}")
        print(f"Composition length: {len(current_composition)}")
        
        return current_composition, history
    
    def finetune_theta_with_composition(self, inp, target, theta, theta_latent, best_composition, state_labels, 
                                       operator_inp, operator_target, optimize_latent=False, operator_to_trajectories=None, theta_all_individual=None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Finetune the theta parameters of operators in the best composition while preserving
        the original operator capabilities.
        
        Args:
            inp: test input data
            target: test target data  
            theta: original operator parameters
            theta_latent: original latent operator parameters
            best_composition: list of operator indices in best composition
            state_labels: state labels
            operator_inp: original operator training inputs for preservation
            operator_target: original operator training targets for preservation
            optimize_latent: if True, optimize theta_latent instead of theta directly
            operator_to_trajectories: dict mapping operator index to list of trajectory indices
            theta_all_individual: individual trajectory thetas before averaging
            
        Returns:
            Finetuned theta parameters, finetuned theta_latent parameters, and training history
        """
        print("=" * 60)
        if optimize_latent:
            print("FINETUNING THETA LATENT PARAMETERS")
        else:
            print("FINETUNING THETA PARAMETERS")
        print("=" * 60)
        
        # Parameters
        epochs = self.config.get('finetune_epochs', 300)
        preservation_coeff = self.config.get('preservation_coeff', 1.0)
        noise_level = self.config.get('noise_level', 1e-3)
        
        # Only finetune operators that are in the best composition
        theta_finetuned = theta.clone().detach()
        theta_latent_finetuned = theta_latent.clone().detach()
        
        # Create mapping for unique operators in composition
        unique_ops = list(set(best_composition))
        idx_to_theta = {op_idx: i for i, op_idx in enumerate(unique_ops)}
        theta_to_idx = {i: op_idx for i, op_idx in enumerate(unique_ops)}
        
        print(f"Finetuning {len(unique_ops)} unique operators: {unique_ops}")
        print(f"Used in composition: {best_composition}")
        print(f"Out of {theta.shape[0]} total operators")
        print(f"Index mapping: {idx_to_theta}")
        print(f"Optimization mode: {'Latent space' if optimize_latent else 'Parameter space'}")
        
        # Collect ALL trajectories for operators in composition (do this once at the beginning)
        preservation_trajectories = None
        preservation_targets = None
        preservation_traj_to_op_mapping = None
        if preservation_coeff > 0 and operator_to_trajectories is not None and theta_all_individual is not None:
            all_traj_indices = []
            traj_to_op_mapping = []  # Maps each collected trajectory to its operator index
            
            for op_idx in unique_ops:
                if op_idx in operator_to_trajectories:
                    op_trajectories = operator_to_trajectories[op_idx]
                    all_traj_indices.extend(op_trajectories)
                    # Record which operator each trajectory belongs to
                    traj_to_op_mapping.extend([op_idx] * len(op_trajectories))
            
            if all_traj_indices:
                trajectories = torch.cat([operator_inp, operator_target],axis=1)
                preservation_trajectories = trajectories[all_traj_indices].to(self.device)
                #preservation_targets = operator_target[all_traj_indices].to(self.device)
                preservation_traj_to_op_mapping = torch.tensor(traj_to_op_mapping, device=self.device)
                print(f"Using {len(all_traj_indices)} trajectories for preservation loss")
        
        if optimize_latent:
            # Extract latent parameters for unique operators
            theta_latent_to_finetune = theta_latent_finetuned[unique_ops].clone().detach().requires_grad_()
            
            # Optimizer for latent parameters
            optimizer = torch.optim.AdamW([theta_latent_to_finetune], lr=1e-3, weight_decay=0) # 1e-3
            
            print(f"Latent theta shape: {theta_latent_to_finetune.shape}")
        else:
            # Extract only unique operators for finetuning
            theta_to_finetune = theta_finetuned[unique_ops].clone().detach().requires_grad_()
            
            # Optimizer for theta parameters
            optimizer = torch.optim.AdamW([theta_to_finetune], lr=1e-4, weight_decay=0) # 5e-4
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'epochs': [],
            'composition_loss': [],
            'preservation_loss': [],
            'total_loss': []
        }
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in tqdm(range(epochs), desc="Finetuning"):
            # Adaptive training horizon

            training_horizon = 1

            #if epoch < epochs // 3:
            #    training_horizon = 1
            #elif epoch < 2 * epochs // 3:
            #    training_horizon = 3
            #else:
                #training_horizon = 5 # 10
            
            # Composition loss: evaluate best composition on test data
            t1 = random.randint(0, self.n_input_frames - training_horizon - 1)
            x_test = inp[:, t1].to(self.device)
            x_test = x_test + torch.randn_like(x_test) * noise_level
            y_test = inp[:, t1+1:t1+1+training_horizon].to(self.device)
            y_test = rearrange(y_test, "b t c h -> t b c h")
            
            if optimize_latent:
                # Decode latent to get current theta parameters
                current_theta = self.model.decode_theta(theta_latent_to_finetune, dim=1)
            else:
                current_theta = theta_to_finetune
            
            # Multi-step prediction with best composition
            pred_composition = []
            current = x_test
            for _ in range(training_horizon):
                current = sequential_operator_composition(
                    current, state_labels, best_composition, current_theta, self.model,
                    integration_time=1.0, n_future_steps=1, idx_to_theta=idx_to_theta,
                    num_integration_steps=self.config.get('num_integration_steps', 1)
                )
                pred_composition.append(current.unsqueeze(0))
                
            pred_composition = torch.cat(pred_composition, dim=0)
            composition_loss = self.relative_l2_error(pred_composition, y_test)
            
            # Preservation loss: ensure operators maintain their original capabilities (BATCHED)
            preservation_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if preservation_coeff > 0 and preservation_trajectories is not None:
                # Sample random time step
                t2 = random.randint(0, preservation_trajectories.shape[1] - training_horizon - 1)
                
                # Get batch data
                batch_x = preservation_trajectories[:, t2] + torch.randn_like(preservation_trajectories[:, t2]) * noise_level
                batch_y = preservation_trajectories[:, t2+1:t2+1+training_horizon]
                batch_y = rearrange(batch_y, "b t c h -> t b c h")
                
                # Expand current theta to match batch size
                if optimize_latent:
                    # Use current latent parameters and decode
                    current_theta_unique = self.model.decode_theta(theta_latent_to_finetune, dim=1)
                else:
                    current_theta_unique = theta_to_finetune
                
                # Map from unique operators to full batch
                # preservation_traj_to_op_mapping: [batch_size] with operator indices
                # We need to map these to positions in unique_ops list
                op_to_unique_idx = {op_idx: i for i, op_idx in enumerate(unique_ops)}
                unique_indices = torch.tensor([op_to_unique_idx[op_idx.item()] for op_idx in preservation_traj_to_op_mapping], device=self.device)
                
                # Expand theta: [batch_size, theta_dim]
                batch_theta = current_theta_unique[unique_indices]
                
                # Use model.solve_ode for efficient batch processing
                pred_batch, _ = self.model.solve_ode(
                    batch_x, batch_theta, state_labels, dim=1,
                    n_future_steps=training_horizon, integration_time=training_horizon, 
                    predict_normed=False, metadata={}
                )
                
                # Compute preservation loss for entire batch
                # Fix dimension mismatch: pred_batch is (batch, time, ...) but batch_y is (time, batch, ...)
                pred_batch = pred_batch.transpose(0, 1)  # Convert to (time, batch, ...)
                preservation_loss = self.relative_l2_error(pred_batch, batch_y)
            
            # Total loss
            total_loss = composition_loss + preservation_coeff * preservation_loss
            #total_loss = preservation_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Store history
            if epoch % 50 == 0 or epoch == epochs - 1:
                history['epochs'].append(epoch)
                history['composition_loss'].append(composition_loss.item())
                history['preservation_loss'].append(preservation_loss.item() if isinstance(preservation_loss, torch.Tensor) else preservation_loss)
                history['total_loss'].append(total_loss.item())
                
                print(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Composition: {composition_loss.item():.6f} | "
                      f"Preservation: {preservation_loss.item() if isinstance(preservation_loss, torch.Tensor) else preservation_loss:.6f} | "
                      f"Total: {total_loss.item():.6f} | "
                      f"Horizon: {training_horizon}")
        
        # Update the full tensors with final finetuned parameters
        if optimize_latent:
            # Update latent parameters
            for i, op_idx in enumerate(unique_ops):
                theta_latent_finetuned[op_idx] = theta_latent_to_finetune[i].detach()
            
            # Decode final theta from latent
            with torch.no_grad():
                final_theta = self.model.decode_theta(theta_latent_finetuned, dim=1)
            theta_finetuned = final_theta
        else:
            # Update theta parameters directly
            for i, op_idx in enumerate(unique_ops):
                theta_finetuned[op_idx] = theta_to_finetune[i].detach()
        
        print(f"Finetuning completed.")
        return theta_finetuned.detach(), theta_latent_finetuned.detach(), history
    
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
                    integration_time=1.0, n_future_steps=1,
                    num_integration_steps=self.config.get('num_integration_steps', 1)
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
    
    
    def estimate_coefficients(self, best_composition, operator_params) -> Dict:
        """
        Estimate advection and diffusion coefficients by summing the actual parameter values
        from the operators in the best composition.
        
        Args:
            best_composition: list of operator indices in best composition
            operator_params: list of (v, D) tuples for each operator
            
        Returns:
            Dictionary containing coefficient estimates and analysis
        """
        print("=" * 60)
        print("ESTIMATING ADVECTION AND DIFFUSION COEFFICIENTS")
        print("=" * 60)
        
        if len(best_composition) == 0:
            return {'estimated_v': 0.0, 'estimated_D': 0.0, 'operator_contributions': []}
        
        # Simply sum the actual operator parameters
        operator_contributions = []
        total_v = 0.0
        total_D = 0.0
        
        for i, op_idx in enumerate(best_composition):
            # Get operator parameters directly from encoding
            v_param, D_param = operator_params[op_idx]
            
            # Add to totals
            total_v += v_param
            total_D += D_param
            
            operator_contributions.append({
                'operator_index': op_idx,
                'position_in_composition': i,
                'operator_params': (v_param, D_param),
                'v_contribution': v_param,
                'D_contribution': D_param,
                'operator_type': 'advection' if D_param == 0 else 'diffusion' if v_param == 0 else 'mixed'
            })
            
            print(f"  Operator {op_idx} (pos {i}): v={v_param:.3f}, D={D_param:.3f}")
        
        # The estimated coefficients are just the sums
        estimated_v = total_v
        estimated_D = total_D
        
        print(f"Estimated coefficients (sum): v={estimated_v:.3f}, D={estimated_D:.3f}")
        print(f"Individual contributions: v={[c['v_contribution'] for c in operator_contributions]}")
        print(f"Individual contributions: D={[c['D_contribution'] for c in operator_contributions]}")
        
        return {
            'estimated_v': estimated_v,
            'estimated_D': estimated_D,
            'total_v_sum': total_v,
            'total_D_sum': total_D,
            'operator_contributions': operator_contributions,
            'composition_length': len(best_composition)
        }

    def evaluate_before_finetuning(self, inp, target, theta, best_composition, state_labels) -> Dict:
        """Evaluate model performance before finetuning for comparison."""
        print("=" * 60)
        print("EVALUATING PERFORMANCE BEFORE FINETUNING")
        print("=" * 60)
        
        return self.evaluate_model_with_composition(inp, target, theta, best_composition, state_labels)

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
        theta, theta_latent, operator_inp, operator_target, operator_params, operator_to_trajectories, theta_all_individual, theta_latent_all_individual = self.encode_structured_operators(
            K=K, n_trajectories_per_operator=n_trajectories_per_operator, 
            v_range=operator_v_range, D_range=operator_D_range
        )
        
        # Step 2: Get test data for optimization and evaluation  
        test_v_range = self.config.get('test_v_range', (0.5, 1.5))
        test_D_range = self.config.get('test_D_range', (0.5, 1.5))
        test_loader = self._create_dataset(split='test', v_range=test_v_range, D_range=test_D_range)
        
        # Collect enough test data trajectories and store ground truth parameters
        all_inp, all_target = [], []
        ground_truth_params = []
        trajectories_collected = 0
        n_test_trajectories = self.config.get('n_test_trajectories', 1)
        
        for batch in test_loader:
            inp_batch, target_batch = batch["input"], batch["target"]
            velocities, diffusivities = batch["velocities"], batch["diffusivities"]
            all_inp.append(inp_batch)
            all_target.append(target_batch)
            ground_truth_params.extend([(v, D) for v, D in zip(velocities, diffusivities)])
            trajectories_collected += inp_batch.shape[0]
            
            if trajectories_collected >= n_test_trajectories:
                break
        
        # Concatenate and slice to exact number needed
        inp = torch.cat(all_inp, dim=0)[:n_test_trajectories].to(self.device)
        target = torch.cat(all_target, dim=0)[:n_test_trajectories].to(self.device)
        ground_truth_params = ground_truth_params[:n_test_trajectories]
        state_labels = torch.tensor([0], device=self.device)
        
        print(f"Using {theta.shape[0]} structured operators")
        print(f"Test data shape: {inp.shape}")
        print(f"Ground truth parameters: {ground_truth_params}")
        
        # Greedy iterative selection to find best operator composition
        max_operators = self.config.get('max_operators', 5)
        best_composition, selection_history = self.greedy_iterative_operator_selection(
            inp, target, theta, state_labels, max_operators=max_operators
        )
        
        # NEW: Estimate coefficients after finding best composition
        coefficient_estimates = self.estimate_coefficients(
            best_composition, operator_params
        )
        
        # NEW: Evaluate performance before finetuning
        eval_before_finetuning = self.evaluate_before_finetuning(
            inp, target, theta, best_composition, state_labels
        )
        
        # Finetuning phase: optimize theta parameters for best composition
        if self.config.get('enable_finetuning', True) and len(best_composition) > 0:
            optimize_latent = self.config.get('optimize_latent', False)
            theta_finetuned, theta_latent_finetuned, finetune_history = self.finetune_theta_with_composition(
                inp, target, theta, theta_latent, best_composition, state_labels, operator_inp, operator_target, optimize_latent, operator_to_trajectories, theta_all_individual
            )
        else:
            theta_finetuned = theta
            theta_latent_finetuned = theta_latent
            finetune_history = {}
            print("Skipping finetuning phase.")
        
        # Evaluation on test data using best composition and finetuned operators
        eval_after_finetuning = self.evaluate_model_with_composition(
            inp, target, theta_finetuned, best_composition, state_labels
        )
        
        # Calculate finetuning effectiveness
        improvement_ratio = eval_before_finetuning['error'] / eval_after_finetuning['error'] if eval_after_finetuning['error'] > 0 else 1.0
        error_reduction = eval_before_finetuning['error'] - eval_after_finetuning['error']
        relative_improvement = error_reduction / eval_before_finetuning['error'] * 100 if eval_before_finetuning['error'] > 0 else 0.0
        
        print(f"Finetuning Effectiveness:")
        print(f"  Before finetuning error: {eval_before_finetuning['error']:.6f}")
        print(f"  After finetuning error:  {eval_after_finetuning['error']:.6f}")
        print(f"  Improvement ratio: {improvement_ratio:.2f}x")
        print(f"  Relative improvement: {relative_improvement:.1f}%")
        
        # Calculate coefficient estimation accuracy for this run
        coefficient_estimation_accuracy = {}
        if ground_truth_params and coefficient_estimates:
            gt_v, gt_D = ground_truth_params[0]
            est_v, est_D = coefficient_estimates.get('estimated_v', 0), coefficient_estimates.get('estimated_D', 0)
            
            v_absolute_error = abs(est_v - gt_v)
            D_absolute_error = abs(est_D - gt_D)
            v_squared_error = (est_v - gt_v) ** 2
            D_squared_error = (est_D - gt_D) ** 2
            v_relative_error = v_absolute_error / max(gt_v, 1e-8) if gt_v != 0 else abs(est_v)
            D_relative_error = D_absolute_error / max(gt_D, 1e-8) if gt_D != 0 else abs(est_D)
            
            coefficient_estimation_accuracy = {
                'ground_truth_v': gt_v,
                'ground_truth_D': gt_D,
                'estimated_v': est_v,
                'estimated_D': est_D,
                'v_absolute_error': v_absolute_error,
                'D_absolute_error': D_absolute_error,
                'v_squared_error': v_squared_error,
                'D_squared_error': D_squared_error,
                'v_relative_error': v_relative_error,
                'D_relative_error': D_relative_error,
                'v_bias': est_v - gt_v,
                'D_bias': est_D - gt_D,
                'combined_mae': v_absolute_error + D_absolute_error,
                'combined_mse': v_squared_error + D_squared_error,
                'combined_rmse': np.sqrt(v_squared_error + D_squared_error)
            }
        
        # Store all results
        results = {
            'operator_params': operator_params,
            'num_operators': theta.shape[0],
            'best_composition': best_composition,
            'composition_length': len(best_composition),
            'selection_history': selection_history,
            'coefficient_estimates': coefficient_estimates,
            'ground_truth_params': ground_truth_params,
            'coefficient_estimation_accuracy': coefficient_estimation_accuracy,
            'evaluation_before_finetuning': eval_before_finetuning,
            'evaluation_after_finetuning': eval_after_finetuning,
            'finetuning_effectiveness': {
                'improvement_ratio': improvement_ratio,
                'error_reduction': error_reduction,
                'relative_improvement': relative_improvement
            },
            'finetune_history': finetune_history if 'finetune_history' in locals() else {},
            'test_parameter_ranges': {
                'v_range': test_v_range,
                'D_range': test_D_range,
                'operator_v_range': operator_v_range,
                'operator_D_range': operator_D_range
            },
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


def create_coefficient_estimation_plots(all_results: List[Dict], output_dir: str):
    """Create plots for coefficient estimation metrics across multiple runs."""
    print("Creating coefficient estimation plots...")
    
    plots_dir = os.path.join(output_dir, 'coefficient_estimation_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect data from all runs
    v_estimates, D_estimates = [], []
    v_ground_truth, D_ground_truth = [], []
    v_errors, D_errors = [], []
    run_indices = []
    
    for i, r in enumerate(all_results):
        if 'coefficient_estimates' in r and 'ground_truth_params' in r and r['ground_truth_params']:
            coeff_est = r['coefficient_estimates']
            gt_params = r['ground_truth_params'][0]
            
            est_v = coeff_est.get('estimated_v', 0.0)
            est_D = coeff_est.get('estimated_D', 0.0)
            gt_v, gt_D = gt_params
            
            v_estimates.append(est_v)
            D_estimates.append(est_D)
            v_ground_truth.append(gt_v)
            D_ground_truth.append(gt_D)
            v_errors.append(est_v - gt_v)
            D_errors.append(est_D - gt_D)
            run_indices.append(i)
    
    if not v_estimates:  # No data to plot
        return
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Estimates vs Ground Truth for Advection
    axes[0, 0].scatter(v_ground_truth, v_estimates, alpha=0.7, color='red')
    min_v, max_v = min(min(v_ground_truth), min(v_estimates)), max(max(v_ground_truth), max(v_estimates))
    axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.7, label='Perfect estimation')
    axes[0, 0].set_xlabel('Ground Truth Advection (v)')
    axes[0, 0].set_ylabel('Estimated Advection (v)')
    axes[0, 0].set_title('Advection Coefficient: Estimates vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Estimates vs Ground Truth for Diffusion
    axes[0, 1].scatter(D_ground_truth, D_estimates, alpha=0.7, color='blue')
    min_D, max_D = min(min(D_ground_truth), min(D_estimates)), max(max(D_ground_truth), max(D_estimates))
    axes[0, 1].plot([min_D, max_D], [min_D, max_D], 'k--', alpha=0.7, label='Perfect estimation')
    axes[0, 1].set_xlabel('Ground Truth Diffusion (D)')
    axes[0, 1].set_ylabel('Estimated Diffusion (D)')
    axes[0, 1].set_title('Diffusion Coefficient: Estimates vs Ground Truth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution for Advection
    axes[0, 2].hist(v_errors, bins=min(10, len(v_errors)), alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].axvline(0, color='black', linestyle='--', alpha=0.7, label='Perfect estimation')
    axes[0, 2].set_xlabel('Estimation Error (v)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Advection Coefficient Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution for Diffusion
    axes[1, 0].hist(D_errors, bins=min(10, len(D_errors)), alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Perfect estimation')
    axes[1, 0].set_xlabel('Estimation Error (D)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Diffusion Coefficient Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Errors over Runs
    axes[1, 1].plot(run_indices, v_errors, 'ro-', alpha=0.7, label='Advection Error', markersize=4)
    axes[1, 1].plot(run_indices, D_errors, 'bo-', alpha=0.7, label='Diffusion Error', markersize=4)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Run Index')
    axes[1, 1].set_ylabel('Estimation Error')
    axes[1, 1].set_title('Estimation Errors Across Runs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Combined Error Metrics
    v_mae = np.mean(np.abs(v_errors))
    D_mae = np.mean(np.abs(D_errors))
    v_mse = np.mean(np.array(v_errors) ** 2)
    D_mse = np.mean(np.array(D_errors) ** 2)
    v_rmse = np.sqrt(v_mse)
    D_rmse = np.sqrt(D_mse)
    
    metrics = ['MAE', 'MSE', 'RMSE']
    v_metrics = [v_mae, v_mse, v_rmse]
    D_metrics = [D_mae, D_mse, D_rmse]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, v_metrics, width, label='Advection (v)', alpha=0.7, color='red')
    axes[1, 2].bar(x + width/2, D_metrics, width, label='Diffusion (D)', alpha=0.7, color='blue')
    axes[1, 2].set_ylabel('Error Value')
    axes[1, 2].set_title('Error Metrics Summary')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    # Add value labels on bars
    for i, (v_val, D_val) in enumerate(zip(v_metrics, D_metrics)):
        axes[1, 2].text(i - width/2, v_val * 1.1, f'{v_val:.4f}', ha='center', va='bottom', fontsize=8)
        axes[1, 2].text(i + width/2, D_val * 1.1, f'{D_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Coefficient Estimation Analysis ({len(v_estimates)} runs)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'coefficient_estimation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Coefficient estimation plots saved to {plots_dir}")


def create_plots(results: Dict, output_dir: str):
    """Create comprehensive plots of the experiment results."""
    print("Creating plots...")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 0: NEW - Before/After Finetuning Comparison
    if 'evaluation_before_finetuning' in results and 'evaluation_after_finetuning' in results:
        plt.figure(figsize=(15, 5))
        
        before_error = results['evaluation_before_finetuning']['error']
        after_error = results['evaluation_after_finetuning']['error']
        improvement_ratio = results.get('finetuning_effectiveness', {}).get('improvement_ratio', 1.0)
        
        # Plot 1: Error comparison
        plt.subplot(1, 3, 1)
        errors = [before_error, after_error]
        labels = ['Before Finetuning', 'After Finetuning']
        colors = ['red', 'green']
        bars = plt.bar(labels, errors, color=colors, alpha=0.7)
        plt.ylabel('Relative L2 Error')
        plt.title('Finetuning Effectiveness')
        plt.yscale('log')
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{error:.4f}', ha='center', va='bottom')
        
        # Add improvement text
        plt.text(0.5, max(errors) * 0.5, f'Improvement: {improvement_ratio:.2f}x', 
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient Estimation Accuracy
        plt.subplot(1, 3, 2)
        if 'coefficient_estimates' in results and 'ground_truth_params' in results and results['ground_truth_params']:
            coeff_est = results['coefficient_estimates']
            gt_params = results['ground_truth_params'][0]
            
            gt_v, gt_D = gt_params
            est_v, est_D = coeff_est.get('estimated_v', 0), coeff_est.get('estimated_D', 0)
            
            categories = ['Advection (v)', 'Diffusion (D)']
            ground_truth = [gt_v, gt_D]
            estimated = [est_v, est_D]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, ground_truth, width, label='Ground Truth', alpha=0.7, color='blue')
            plt.bar(x + width/2, estimated, width, label='Estimated', alpha=0.7, color='orange')
            
            plt.ylabel('Coefficient Value')
            plt.title('Coefficient Estimation Accuracy')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add error text
            v_error = abs(gt_v - est_v) / max(gt_v, 1e-8) if gt_v != 0 else abs(est_v)
            D_error = abs(gt_D - est_D) / max(gt_D, 1e-8) if gt_D != 0 else abs(est_D)
            plt.text(0, max(max(ground_truth), max(estimated)) * 0.8, 
                    f'v error: {v_error*100:.1f}%\nD error: {D_error*100:.1f}%', 
                    ha='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            plt.text(0.5, 0.5, 'No coefficient estimation data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Coefficient Estimation Accuracy')
        
        # Plot 3: Operator Contributions
        plt.subplot(1, 3, 3)
        if 'coefficient_estimates' in results and 'operator_contributions' in results['coefficient_estimates']:
            contributions = results['coefficient_estimates']['operator_contributions']
            
            if contributions:
                op_indices = [c['operator_index'] for c in contributions]
                v_contributions = [c['v_contribution'] for c in contributions]
                D_contributions = [c['D_contribution'] for c in contributions]
                colors = ['red' if c['operator_type'] == 'advection' else 'blue' if c['operator_type'] == 'diffusion' else 'green' 
                         for c in contributions]
                
                # Stack v and D contributions
                x_pos = range(len(op_indices))
                plt.bar(x_pos, v_contributions, color='red', alpha=0.7, label='Advection (v)')
                plt.bar(x_pos, D_contributions, bottom=v_contributions, color='blue', alpha=0.7, label='Diffusion (D)')
                
                plt.xlabel('Position in Composition')
                plt.ylabel('Parameter Value')
                plt.title('Operator Parameter Contributions')
                plt.xticks(x_pos, [f'Op {idx}' for idx in op_indices], rotation=45)
                
                # Add legend
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No operator contribution data', ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12)
        else:
            plt.text(0.5, 0.5, 'No operator contribution data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Individual Operator Contributions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'finetuning_and_coefficients.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
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
    eval_key = 'evaluation_after_finetuning' if 'evaluation_after_finetuning' in results else 'evaluation'
    if eval_key in results:
        eval_data = results[eval_key]
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


def organize_results_by_parameters(results: Dict, base_output_dir: str) -> str:
    """
    Organize results by parameter ranges and operator count.
    
    Args:
        results: experiment results dictionary
        base_output_dir: base directory for results
        
    Returns:
        organized output directory path
    """
    # Extract parameter information
    test_ranges = results.get('test_parameter_ranges', {})
    v_range = test_ranges.get('v_range', (0, 1))
    D_range = test_ranges.get('D_range', (0, 1))
    num_operators = results.get('num_operators', 0)
    
    # Create organized directory structure
    v_min, v_max = v_range
    D_min, D_max = D_range
    
    # Format parameter ranges for directory names
    param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
    operator_dir = f"operators_{num_operators}"
    
    # Create full organized path
    organized_dir = os.path.join(base_output_dir, operator_dir, param_dir)
    os.makedirs(organized_dir, exist_ok=True)
    
    return organized_dir

def organize_results_by_parameters(results: Dict, base_output_dir: str, run_timestamp: str, num_integration_steps: int = None) -> str:
    """
    Organize results by parameter ranges, operator count, integration steps, and run timestamp.
    
    Better structure: v_min_max_D_min_max/operators_N/integration_steps_N/run_TIMESTAMP/
    
    Args:
        results: experiment results dictionary
        base_output_dir: base directory for results
        run_timestamp: timestamp for this experimental run
        num_integration_steps: number of integration steps used
        
    Returns:
        organized output directory path
    """
    # Extract parameter information
    test_ranges = results.get('test_parameter_ranges', {})
    v_range = test_ranges.get('v_range', (0, 1))
    D_range = test_ranges.get('D_range', (0, 1))
    num_operators = results.get('num_operators', 0)
    
    # Create organized directory structure: v_range/D_range -> operators -> integration_steps -> run_timestamp
    v_min, v_max = v_range
    D_min, D_max = D_range
    
    # Format parameter ranges for directory names
    param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
    operator_dir = f"operators_{num_operators}"
    integration_dir = f"integration_steps_{num_integration_steps}" if num_integration_steps else "integration_steps_default"
    run_dir = f"run_{run_timestamp}"
    
    # Create full organized path: param_dir/operator_dir/integration_dir/run_dir
    # But use the original base directory structure as the root
    organized_dir = os.path.join("./results/neural_operator_splitting", param_dir, operator_dir, integration_dir, run_dir)
    os.makedirs(organized_dir, exist_ok=True)
    
    return organized_dir


def convert_for_json(obj):
    """Convert numpy arrays and tensors to JSON-serializable format."""
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


def save_results(results: Dict, output_dir: str):
    """Save experiment results with organized structure."""
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    # Enhanced summary with before/after finetuning comparison
    before_error = results.get('evaluation_before_finetuning', {}).get('error', None)
    after_error = results.get('evaluation_after_finetuning', {}).get('error', None)
    finetuning_effectiveness = results.get('finetuning_effectiveness', {})
    
    summary = {
        'num_operators': results.get('num_operators', None),
        'composition_length': results.get('composition_length', None),
        'best_composition': results.get('best_composition', []),
        'test_parameter_ranges': results.get('test_parameter_ranges', {}),
        'coefficient_estimates': results.get('coefficient_estimates', {}),
        'ground_truth_params': results.get('ground_truth_params', []),
        'before_finetuning_error': before_error,
        'after_finetuning_error': after_error,
        'finetuning_improvement_ratio': finetuning_effectiveness.get('improvement_ratio', None),
        'finetuning_relative_improvement_pct': finetuning_effectiveness.get('relative_improvement', None),
        'error_reduction': finetuning_effectiveness.get('error_reduction', None)
    }
    
    # Add coefficient estimation accuracy if ground truth available
    if results.get('ground_truth_params') and results.get('coefficient_estimates'):
        gt_params = results['ground_truth_params'][0] if results['ground_truth_params'] else (0, 0)
        estimates = results['coefficient_estimates']
        
        gt_v, gt_D = gt_params
        est_v, est_D = estimates.get('estimated_v', 0), estimates.get('estimated_D', 0)
        
        v_error = abs(gt_v - est_v) / max(gt_v, 1e-8) if gt_v != 0 else abs(est_v)
        D_error = abs(gt_D - est_D) / max(gt_D, 1e-8) if gt_D != 0 else abs(est_D)
        
        # Calculate detailed metrics for single run
        v_absolute_error = abs(est_v - gt_v)
        D_absolute_error = abs(est_D - gt_D)
        v_squared_error = (est_v - gt_v) ** 2
        D_squared_error = (est_D - gt_D) ** 2
        
        summary['coefficient_estimation_accuracy'] = {
            'ground_truth_v': gt_v,
            'ground_truth_D': gt_D,
            'estimated_v': est_v,
            'estimated_D': est_D,
            'v_absolute_error': v_absolute_error,
            'D_absolute_error': D_absolute_error,
            'v_squared_error': v_squared_error,
            'D_squared_error': D_squared_error,
            'v_relative_error': v_error,
            'D_relative_error': D_error,
            'v_bias': est_v - gt_v,
            'D_bias': est_D - gt_D,
            'combined_mae': v_absolute_error + D_absolute_error,
            'combined_mse': v_squared_error + D_squared_error,
            'combined_rmse': np.sqrt(v_squared_error + D_squared_error)
        }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    
    # Save detailed coefficient estimation results if available
    if results.get('coefficient_estimates'):
        coeff_file = os.path.join(output_dir, 'coefficient_estimation.json')
        coeff_data = {
            'coefficient_estimates': results['coefficient_estimates'],
            'ground_truth_params': results.get('ground_truth_params', []),
            'estimation_accuracy': summary.get('coefficient_estimation_accuracy', {}),
            'operator_contributions': results['coefficient_estimates'].get('operator_contributions', []),
            'best_composition': results.get('best_composition', []),
            'operator_params': results.get('operator_params', [])
        }
        with open(coeff_file, 'w') as f:
            json.dump(convert_for_json(coeff_data), f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    # Enhanced summary printout
    before_str = f"{before_error:.6f}" if before_error is not None else "N/A"
    after_str = f"{after_error:.6f}" if after_error is not None else "N/A"
    improvement_str = f"{finetuning_effectiveness.get('improvement_ratio', 1.0):.2f}x" if finetuning_effectiveness.get('improvement_ratio') else "N/A"
    
    print(f"Summary:")
    print(f"  Operators: {results.get('num_operators', 'N/A')}, Composition length: {results.get('composition_length', 'N/A')}")
    print(f"  Before finetuning error: {before_str}")
    print(f"  After finetuning error:  {after_str}")
    print(f"  Finetuning improvement: {improvement_str}")
    
    if 'coefficient_estimation_accuracy' in summary:
        acc = summary['coefficient_estimation_accuracy']
        print(f"  Coefficient estimation - v: {acc['estimated_v']:.3f} (GT: {acc['ground_truth_v']:.3f}, err: {acc['v_relative_error']*100:.1f}%)")
        print(f"                          - D: {acc['estimated_D']:.3f} (GT: {acc['ground_truth_D']:.3f}, err: {acc['D_relative_error']*100:.1f}%)")


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
    parser.add_argument('--num-integration-steps', type=int, default=1,
                        help='Number of integration sub-steps for finer dt (default: 1)')
    parser.add_argument('--enable-finetuning', action='store_true', default=True,
                        help='Enable theta parameter finetuning (default: True)')
    parser.add_argument('--optimize-latent', action='store_true', default=False,
                        help='Optimize theta_latent instead of theta directly (default: False)')
    parser.add_argument('--finetune-epochs', type=int, default=300,
                        help='Number of epochs for finetuning (default: 300)')
    parser.add_argument('--preservation-coeff', type=float, default=1.0,
                        help='Coefficient for preservation loss during finetuning (default: 1.0)')
    parser.add_argument('--noise-level', type=float, default=1e-3, #1e-3
                        help='Noise level for data augmentation during finetuning (default: 1e-3)')
    parser.add_argument('--output-dir', type=str, default='./results/neural_operator_splitting',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--test-v-min', type=float, default=1.9,
                        help='Minimum value for test advection velocity range (default: 1.9)')
    parser.add_argument('--test-v-max', type=float, default=2.0,
                        help='Maximum value for test advection velocity range (default: 2.0)')
    parser.add_argument('--test-D-min', type=float, default=0.0,
                        help='Minimum value for test diffusion coefficient range (default: 0.0)')
    parser.add_argument('--test-D-max', type=float, default=0.0,
                        help='Maximum value for test diffusion coefficient range (default: 0.0)')
    
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
        'num_integration_steps': args.num_integration_steps,
        'enable_finetuning': args.enable_finetuning,
        'optimize_latent': args.optimize_latent,
        'finetune_epochs': args.finetune_epochs,
        'preservation_coeff': args.preservation_coeff,
        'noise_level': args.noise_level,
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
        
        # Organize results by parameters and operator count with improved structure
        organized_output_dir = organize_results_by_parameters(results, base_output_dir, timestamp, args.num_integration_steps)
        
        # Save results and create plots in organized directory
        save_results(results, organized_output_dir)
        create_plots(results, organized_output_dir)
        
        # Also save a copy in the timestamped directory for easy access
        save_results(results, base_output_dir)
        
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
            
            # Organize results by parameters for this run with improved structure
            organized_output_dir = organize_results_by_parameters(results, base_output_dir, f"{timestamp}_run_{run_idx:03d}", args.num_integration_steps)
            
            # Save individual run results in organized structure
            save_results(results, organized_output_dir)
            create_plots(results, organized_output_dir)
            
            all_results.append(results)
        
        # Create aggregate plots for coefficient estimation metrics in organized structure
        if len(all_results) > 1:
            # Save aggregate plots in the parameter-organized integration directory for easy comparison
            first_result = all_results[0]
            
            # Create aggregate directory at integration level: v_range/operators_N/integration_steps_N/
            test_ranges = first_result.get('test_parameter_ranges', {})
            v_range = test_ranges.get('v_range', (0, 1))
            D_range = test_ranges.get('D_range', (0, 1))
            v_min, v_max = v_range
            D_min, D_max = D_range
            
            param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
            operator_dir = f"operators_{first_result.get('num_operators', 0)}"
            integration_dir = f"integration_steps_{args.num_integration_steps}" if args.num_integration_steps else "integration_steps_default"
            
            aggregate_parent = os.path.join("./results/neural_operator_splitting", param_dir, operator_dir, integration_dir)
            os.makedirs(aggregate_parent, exist_ok=True)
            
            create_coefficient_estimation_plots(all_results, aggregate_parent)
        
        # Aggregate statistics
        print(f"\n" + "="*60)
        print("AGGREGATING STATISTICS")
        print("="*60)
        
        validation_errors = [min(r['selection_history']['errors']) for r in all_results]
        final_errors = [r['evaluation_after_finetuning']['error'] for r in all_results]
        before_finetuning_errors = [r['evaluation_before_finetuning']['error'] for r in all_results]
        composition_lengths = [r['composition_length'] for r in all_results]
        improvement_ratios = [r['finetuning_effectiveness']['improvement_ratio'] for r in all_results]
        
        # NEW: Coefficient estimation metrics
        coefficient_metrics = {
            'v_estimates': [],
            'D_estimates': [],
            'v_ground_truth': [],
            'D_ground_truth': [],
            'v_errors': [],
            'D_errors': [],
            'v_squared_errors': [],
            'D_squared_errors': []
        }
        
        # Collect coefficient estimation data from all runs
        for r in all_results:
            if 'coefficient_estimates' in r and 'ground_truth_params' in r and r['ground_truth_params']:
                coeff_est = r['coefficient_estimates']
                gt_params = r['ground_truth_params'][0]  # Assuming single trajectory per run
                
                est_v = coeff_est.get('estimated_v', 0.0)
                est_D = coeff_est.get('estimated_D', 0.0)
                gt_v, gt_D = gt_params
                
                coefficient_metrics['v_estimates'].append(est_v)
                coefficient_metrics['D_estimates'].append(est_D)
                coefficient_metrics['v_ground_truth'].append(gt_v)
                coefficient_metrics['D_ground_truth'].append(gt_D)
                
                # Calculate individual errors
                v_error = est_v - gt_v
                D_error = est_D - gt_D
                coefficient_metrics['v_errors'].append(v_error)
                coefficient_metrics['D_errors'].append(D_error)
                coefficient_metrics['v_squared_errors'].append(v_error ** 2)
                coefficient_metrics['D_squared_errors'].append(D_error ** 2)
        
        # Calculate coefficient estimation statistics
        coeff_stats = {}
        if coefficient_metrics['v_estimates']:  # Only if we have coefficient data
            # Convert to numpy arrays for easier computation
            v_estimates = np.array(coefficient_metrics['v_estimates'])
            D_estimates = np.array(coefficient_metrics['D_estimates'])
            v_ground_truth = np.array(coefficient_metrics['v_ground_truth'])
            D_ground_truth = np.array(coefficient_metrics['D_ground_truth'])
            v_errors = np.array(coefficient_metrics['v_errors'])
            D_errors = np.array(coefficient_metrics['D_errors'])
            
            coeff_stats = {
                'advection_coefficient': {
                    'mse': np.mean(v_errors ** 2),
                    'mae': np.mean(np.abs(v_errors)),
                    'rmse': np.sqrt(np.mean(v_errors ** 2)),
                    'mean_estimate': np.mean(v_estimates),
                    'std_estimate': np.std(v_estimates),
                    'mean_ground_truth': np.mean(v_ground_truth),
                    'relative_mae': np.mean(np.abs(v_errors / np.maximum(v_ground_truth, 1e-8))),
                    'bias': np.mean(v_errors)
                },
                'diffusion_coefficient': {
                    'mse': np.mean(D_errors ** 2),
                    'mae': np.mean(np.abs(D_errors)),
                    'rmse': np.sqrt(np.mean(D_errors ** 2)),
                    'mean_estimate': np.mean(D_estimates),
                    'std_estimate': np.std(D_estimates),
                    'mean_ground_truth': np.mean(D_ground_truth),
                    'relative_mae': np.mean(np.abs(D_errors / np.maximum(D_ground_truth, 1e-8))),
                    'bias': np.mean(D_errors)
                },
                'overall': {
                    'combined_mse': np.mean(v_errors ** 2) + np.mean(D_errors ** 2),
                    'combined_mae': np.mean(np.abs(v_errors)) + np.mean(np.abs(D_errors)),
                    'num_runs_with_coeff_data': len(v_estimates)
                }
            }
        
        stats = {
            'num_runs': args.num_runs,
            'validation_error': {
                'mean': np.mean(validation_errors),
                'std': np.std(validation_errors),
                'min': np.min(validation_errors),
                'max': np.max(validation_errors)
            },
            'before_finetuning_error': {
                'mean': np.mean(before_finetuning_errors),
                'std': np.std(before_finetuning_errors),
                'min': np.min(before_finetuning_errors),
                'max': np.max(before_finetuning_errors)
            },
            'after_finetuning_error': {
                'mean': np.mean(final_errors),
                'std': np.std(final_errors),
                'min': np.min(final_errors),
                'max': np.max(final_errors)
            },
            'finetuning_improvement_ratio': {
                'mean': np.mean(improvement_ratios),
                'std': np.std(improvement_ratios),
                'min': np.min(improvement_ratios),
                'max': np.max(improvement_ratios)
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
            },
            'coefficient_estimation': coeff_stats
        }
        
        # Save aggregate statistics in organized structure  
        aggregate_parent = os.path.join("./results/neural_operator_splitting", param_dir, operator_dir, integration_dir)
        stats_file = os.path.join(aggregate_parent, 'aggregate_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
        
        # Also save copy in timestamped directory for backward compatibility
        stats_file_old = os.path.join(base_output_dir, 'aggregate_statistics.json')  
        with open(stats_file_old, 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
        
        # Save detailed coefficient estimation data across all runs in organized structure
        if coeff_stats:
            coeff_aggregate_file = os.path.join(aggregate_parent, 'coefficient_estimation_aggregate.json')
            detailed_coeff_data = {
                'statistics': coeff_stats,
                'raw_data': {
                    'v_estimates': coefficient_metrics['v_estimates'],
                    'D_estimates': coefficient_metrics['D_estimates'],
                    'v_ground_truth': coefficient_metrics['v_ground_truth'],
                    'D_ground_truth': coefficient_metrics['D_ground_truth'],
                    'v_errors': coefficient_metrics['v_errors'],
                    'D_errors': coefficient_metrics['D_errors']
                },
                'run_details': []
            }
            
            # Add per-run details
            for i, r in enumerate(all_results):
                if 'coefficient_estimates' in r and 'ground_truth_params' in r and r['ground_truth_params']:
                    run_detail = {
                        'run_index': i,
                        'best_composition': r.get('best_composition', []),
                        'coefficient_estimates': r['coefficient_estimates'],
                        'ground_truth_params': r['ground_truth_params'],
                        'estimation_accuracy': r.get('coefficient_estimation_accuracy', {}),
                        'test_parameter_ranges': r.get('test_parameter_ranges', {})
                    }
                    detailed_coeff_data['run_details'].append(run_detail)
            
            with open(coeff_aggregate_file, 'w') as f:
                json.dump(convert_for_json(detailed_coeff_data), f, indent=2)
            
            # Also save copy in timestamped directory for backward compatibility
            coeff_aggregate_file_old = os.path.join(base_output_dir, 'coefficient_estimation_aggregate.json')
            with open(coeff_aggregate_file_old, 'w') as f:
                json.dump(convert_for_json(detailed_coeff_data), f, indent=2)
        
        print(f"Aggregate Statistics ({args.num_runs} runs):")
        print(f"  Validation Error:         {stats['validation_error']['mean']:.6f} ± {stats['validation_error']['std']:.6f}")
        print(f"  Before Finetuning Error:  {stats['before_finetuning_error']['mean']:.6f} ± {stats['before_finetuning_error']['std']:.6f}")
        print(f"  After Finetuning Error:   {stats['after_finetuning_error']['mean']:.6f} ± {stats['after_finetuning_error']['std']:.6f}")
        print(f"  Finetuning Improvement:   {stats['finetuning_improvement_ratio']['mean']:.2f}x ± {stats['finetuning_improvement_ratio']['std']:.2f}")
        print(f"  Composition Length:       {stats['composition_length']['mean']:.1f} ± {stats['composition_length']['std']:.1f}")
        print(f"  Generalization Ratio:     {stats['generalization_ratio']['mean']:.2f} ± {stats['generalization_ratio']['std']:.2f}")
        print(f"  Best After-Finetuning Error: {stats['after_finetuning_error']['min']:.6f}")
        print(f"  Best Improvement Ratio:   {stats['finetuning_improvement_ratio']['max']:.2f}x")
        print(f"  Length Range: {int(stats['composition_length']['min'])} - {int(stats['composition_length']['max'])}")
        
        # Print coefficient estimation metrics if available
        if coeff_stats:
            print(f"\n  Coefficient Estimation Metrics ({coeff_stats['overall']['num_runs_with_coeff_data']} runs with data):")
            print(f"  Advection Coefficient (v):")
            print(f"    MSE:  {coeff_stats['advection_coefficient']['mse']:.6f}")
            print(f"    MAE:  {coeff_stats['advection_coefficient']['mae']:.6f}")
            print(f"    RMSE: {coeff_stats['advection_coefficient']['rmse']:.6f}")
            print(f"    Bias: {coeff_stats['advection_coefficient']['bias']:.6f}")
            print(f"    Relative MAE: {coeff_stats['advection_coefficient']['relative_mae']*100:.1f}%")
            print(f"    Mean Estimate: {coeff_stats['advection_coefficient']['mean_estimate']:.3f} ± {coeff_stats['advection_coefficient']['std_estimate']:.3f}")
            print(f"    Mean Ground Truth: {coeff_stats['advection_coefficient']['mean_ground_truth']:.3f}")
            
            print(f"  Diffusion Coefficient (D):")
            print(f"    MSE:  {coeff_stats['diffusion_coefficient']['mse']:.6f}")
            print(f"    MAE:  {coeff_stats['diffusion_coefficient']['mae']:.6f}")
            print(f"    RMSE: {coeff_stats['diffusion_coefficient']['rmse']:.6f}")
            print(f"    Bias: {coeff_stats['diffusion_coefficient']['bias']:.6f}")
            print(f"    Relative MAE: {coeff_stats['diffusion_coefficient']['relative_mae']*100:.1f}%")
            print(f"    Mean Estimate: {coeff_stats['diffusion_coefficient']['mean_estimate']:.3f} ± {coeff_stats['diffusion_coefficient']['std_estimate']:.3f}")
            print(f"    Mean Ground Truth: {coeff_stats['diffusion_coefficient']['mean_ground_truth']:.3f}")
            
            print(f"  Overall:")
            print(f"    Combined MSE: {coeff_stats['overall']['combined_mse']:.6f}")
            print(f"    Combined MAE: {coeff_stats['overall']['combined_mae']:.6f}")
        else:
            print(f"\n  No coefficient estimation data available across runs.")
        
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config['output_dir'] if args.num_runs == 1 else base_output_dir}")


if __name__ == "__main__":
    main()