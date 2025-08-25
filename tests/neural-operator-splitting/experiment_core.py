"""
Neural Operator Splitting Experiment Core Module

This module contains the main NeuralOperatorSplittingExperiment class
extracted from the random_approach_iterative_finetune_operators.py file.

The class implements:
1. Structured operator encoding from trajectories
2. Greedy iterative operator selection for composition
3. Theta parameter finetuning with preservation loss
4. Coefficient estimation and evaluation
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import random
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
from train.train import DISCOLitModule
from src.plot_dataset_samples import plot_prediction_vs_ground_truth

# Import from our modular components (absolute imports to work when running directly)
from data_generation import RelativeL2, TemporalBatchDatasetFly
from operator_utils import (
    get_batched_operators, 
    sequential_operator_composition, 
    solve_single_operator_ode, 
    sparsify_weights
)
from visualization import create_plots, create_coefficient_estimation_plots
from results_management import organize_results_by_parameters, save_results



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
            # The correct operator index calculation for batched trajectory collection
            traj_batch_idx = i // total_operators  # Which batch of trajectories (0, 1, 2, ...)
            operator_idx = i % total_operators      # Which operator within the batch (0, 1, 2, ..., 2*K-1)
            print(f"  Trajectory {i}: Batch {traj_batch_idx}, Operator {operator_idx} ({op_type}) (v={v:.3f}, D={D:.3f})")
        
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
        
        # Create unique params list for operators - ensure consistency with theta averaging
        unique_params = []
        for op_idx in range(total_operators):
            if n_trajectories_per_operator == 1:
                # Single trajectory per operator - use directly
                unique_params.append(all_params[op_idx])
            else:
                # Multiple trajectories per operator - collect all trajectories for this operator
                traj_params = []
                for traj_batch_idx in range(n_trajectories_per_operator):
                    traj_global_idx = traj_batch_idx * total_operators + op_idx
                    traj_params.append(all_params[traj_global_idx])
                
                # Check if all trajectories for this operator have the same parameters
                first_params = traj_params[0]
                all_same = all(abs(p[0] - first_params[0]) < 1e-10 and abs(p[1] - first_params[1]) < 1e-10 for p in traj_params)
                
                if all_same:
                    # Parameters are identical - use the first one
                    unique_params.append(first_params)
                else:
                    # Parameters differ - average them to match theta averaging
                    avg_v = sum(p[0] for p in traj_params) / len(traj_params)
                    avg_D = sum(p[1] for p in traj_params) / len(traj_params) 
                    unique_params.append((avg_v, avg_D))
                    
                    # Debug: Print parameter averaging info
                    v_values = [p[0] for p in traj_params]
                    D_values = [p[1] for p in traj_params]
                    print(f"  WARNING: Operator {op_idx} has inconsistent parameters across trajectories!")
                    print(f"    v={v_values} -> avg={avg_v:.6f}")
                    print(f"    D={D_values} -> avg={avg_D:.6f}")
                    print(f"    This suggests a bug in the data generation process.")
        
        print(f"\nFinal unique operator parameters (used for coefficient estimation):")
        for op_idx, (v, D) in enumerate(unique_params):
            op_type = "Advection" if D == 0 else "Diffusion"
            print(f"  Operator {op_idx} ({op_type}): v={v:.6f}, D={D:.6f}")
        
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
        
        print(f"\nExperiment completed successfully!")
        print(f"Results: {results}")
        
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
            
            all_results.append(results)
        
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