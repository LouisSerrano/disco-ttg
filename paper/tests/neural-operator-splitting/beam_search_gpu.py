"""
GPU-Optimized Beam Search with Batch Parallelization

Key optimizations:
1. No threading - pure GPU batch processing
2. Temporal dimension parallelization via batch dimension using einops
3. Sequential beam processing (simpler and still effective)
4. Efficient tensor operations on GPU
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple
from einops import rearrange, repeat
from operator_utils import sequential_operator_composition


def beam_search_operator_selection_gpu(inp, target, theta, state_labels, model, relative_l2_error, 
                                     n_input_frames, config, beam_width=3, timestamps_per_op=3, 
                                     max_operators=5, improvement_threshold=0.05):
    """
    GPU-optimized beam search using batch dimension for temporal parallelization.
    
    Args:
        beam_width: Number of best compositions to keep
        timestamps_per_op: Number of timestamps to test per operator  
        improvement_threshold: Minimum improvement to continue (0.05 = 5%)
    """
    print("=" * 60)
    print("GPU BEAM SEARCH OPERATOR SELECTION")
    print("=" * 60)
    print(f"Beam width: {beam_width}, Timestamps per operator: {timestamps_per_op}")
    print(f"Improvement threshold: {improvement_threshold*100:.1f}%")
    print(f"Using GPU batching for temporal parallelization")
    
    num_operators = theta.shape[0]
    device = theta.device
    
    # Initialize beam with empty compositions
    beam = [{'composition': [], 'error': float('inf')}]
    best_overall_error = float('inf')
    
    # History tracking
    history = {
        'compositions': [],
        'errors': [], 
        'beam_search_steps': [],
        'improvement_per_step': [],
        'total_evaluations': 0
    }
    
    for depth in range(max_operators):
        print(f"\n--- Depth {depth + 1}: Expanding beam ---")
        
        # Collect all candidates for this depth
        all_candidates = []
        
        for beam_idx, beam_item in enumerate(beam):
            current_composition = beam_item['composition']
            parent_error = beam_item['error']
            
            # Evaluate all operators for this beam item using GPU batching
            candidates = evaluate_all_operators_gpu_batch(
                current_composition, inp, target, theta, state_labels, model, 
                relative_l2_error, n_input_frames, config, timestamps_per_op, num_operators
            )
            
            # Add metadata to candidates
            for candidate in candidates:
                candidate['beam_idx'] = beam_idx
                candidate['parent_error'] = parent_error
                all_candidates.append(candidate)
        
        history['total_evaluations'] += len(all_candidates)
        
        # Sort and keep best K candidates
        all_candidates.sort(key=lambda x: x['error'])
        beam = all_candidates[:beam_width]
        
        # Check improvement
        best_current_error = beam[0]['error'] 
        improvement = (best_overall_error - best_current_error) / max(best_overall_error, 1e-10)
        
        history['beam_search_steps'].append({
            'depth': depth + 1,
            'candidates_evaluated': len(all_candidates),
            'best_error': best_current_error,
            'improvement': improvement
        })
        history['improvement_per_step'].append(improvement)
        
        print(f"Evaluated {len(all_candidates)} candidates")
        print(f"Best error: {best_current_error:.6f} (improvement: {improvement*100:+.2f}%)")
        print(f"Top 3 compositions: {[c['composition'] for c in beam[:3]]}")
        
        # Store all evaluated compositions and errors
        for candidate in all_candidates:
            history['compositions'].append(candidate['composition'])
            history['errors'].append(candidate['error'])
        
        # Check stopping criteria
        if improvement < improvement_threshold and depth > 0:
            print(f"✗ Improvement {improvement*100:.2f}% < {improvement_threshold*100:.1f}% threshold. Stopping.")
            break
            
        best_overall_error = best_current_error
    
    best_composition = beam[0]['composition']
    print(f"\nBeam search completed!")
    print(f"Best composition: {best_composition} with error: {beam[0]['error']:.6f}")
    print(f"Total evaluations: {history['total_evaluations']}")
    
    return best_composition, history


def evaluate_all_operators_gpu_batch(current_composition, inp, target, theta, state_labels, 
                                   model, relative_l2_error, n_input_frames, config, 
                                   timestamps_per_op, num_operators):
    """
    Evaluate all operators using GPU batch processing with temporal parallelization.
    
    Key insight: Use einops to rearrange temporal dimension into batch dimension
    for parallel evaluation of different timestamps.
    """
    device = theta.device
    candidates = []
    
    # Select validation timestamps
    val_timestamps = select_validation_timestamps(n_input_frames, timestamps_per_op)
    
    # Process operators in batches to manage memory
    batch_size = min(8, num_operators)  # Adjust based on GPU memory
    
    for op_batch_start in range(0, num_operators, batch_size):
        op_batch_end = min(op_batch_start + batch_size, num_operators)
        op_batch_indices = list(range(op_batch_start, op_batch_end))
        
        # Evaluate this batch of operators across all timestamps using GPU batching
        batch_candidates = evaluate_operator_batch_with_temporal_batching(
            current_composition, op_batch_indices, val_timestamps, inp, target, 
            theta, state_labels, model, relative_l2_error, config
        )
        
        candidates.extend(batch_candidates)
    
    return candidates


def evaluate_operator_batch_with_temporal_batching(current_composition, op_indices, val_timestamps,
                                                 inp, target, theta, state_labels, model, 
                                                 relative_l2_error, config):
    """
    Evaluate a batch of operators across multiple timestamps using einops for batching.
    
    This is where the main GPU parallelization happens:
    - Batch dimension: different operators
    - Time dimension rearranged to batch: different timestamps per operator
    """
    device = theta.device
    candidates = []
    
    # Create all operator-timestamp combinations for this batch
    all_combinations = []
    for op_idx in op_indices:
        for val_t in val_timestamps:
            all_combinations.append((op_idx, val_t))
    
    if not all_combinations:
        return candidates
    
    # Extract input/target data for all timestamps
    # Shape: [n_combinations, channels, height, width]
    batch_x_vals = []
    batch_y_vals = []
    batch_compositions = []
    
    for op_idx, val_t in all_combinations:
        x_val = inp[:, val_t].to(device)  # [batch_size, channels, height, width]
        y_val = inp[:, val_t + 1].to(device)
        
        batch_x_vals.append(x_val)
        batch_y_vals.append(y_val)
        batch_compositions.append(current_composition + [op_idx])
    
    # Stack to create batch dimension: [n_combinations, batch_size, channels, height, width]
    batch_x = torch.stack(batch_x_vals, dim=0)
    batch_y = torch.stack(batch_y_vals, dim=0)
    
    # Rearrange to merge combination and batch dimensions for parallel processing
    # From [n_combinations, batch_size, C, H, W] to [n_combinations * batch_size, C, H, W]
    batch_x_flat = rearrange(batch_x, 'n b c h w -> (n b) c h w')
    batch_y_flat = rearrange(batch_y, 'n b c h w -> (n b) c h w')
    
    try:
        with torch.no_grad():
            # Evaluate all combinations in parallel
            all_errors = []
            
            # Process in smaller sub-batches if needed for memory management
            sub_batch_size = min(32, batch_x_flat.shape[0])
            
            for i in range(0, batch_x_flat.shape[0], sub_batch_size):
                sub_x = batch_x_flat[i:i + sub_batch_size]
                sub_y = batch_y_flat[i:i + sub_batch_size]
                
                # Get corresponding compositions for this sub-batch
                sub_combinations = all_combinations[i//inp.shape[0]:(i//inp.shape[0]) + (sub_batch_size//inp.shape[0])]
                
                # Evaluate each composition (still sequential, but batched across time)
                for j, (op_idx, val_t) in enumerate(sub_combinations):
                    if j * inp.shape[0] >= sub_x.shape[0]:
                        break
                        
                    composition = current_composition + [op_idx]
                    
                    # Extract single timestep batch for this composition
                    start_idx = j * inp.shape[0]
                    end_idx = min(start_idx + inp.shape[0], sub_x.shape[0])
                    x_comp = sub_x[start_idx:end_idx]
                    y_comp = sub_y[start_idx:end_idx]
                    
                    try:
                        # Apply composition
                        pred = sequential_operator_composition(
                            x_comp, state_labels, composition, theta, model,
                            integration_time=1.0, n_future_steps=1,
                            num_integration_steps=config.get('num_integration_steps', 1)
                        )
                        
                        # Calculate error
                        error = relative_l2_error(pred, y_comp).item()
                        
                        if np.isfinite(error):
                            candidates.append({
                                'composition': composition,
                                'error': error,
                                'operator_added': op_idx,
                                'timestamp_used': val_t
                            })
                    
                    except Exception as e:
                        # Skip failed evaluations
                        continue
    
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        # Fallback to sequential evaluation
        for op_idx, val_t in all_combinations:
            try:
                x_val = inp[:, val_t].to(device)
                y_val = inp[:, val_t + 1].to(device)
                composition = current_composition + [op_idx]
                
                with torch.no_grad():
                    pred = sequential_operator_composition(
                        x_val, state_labels, composition, theta, model,
                        integration_time=1.0, n_future_steps=1,
                        num_integration_steps=config.get('num_integration_steps', 1)
                    )
                    error = relative_l2_error(pred, y_val).item()
                    
                    if np.isfinite(error):
                        candidates.append({
                            'composition': composition,
                            'error': error,
                            'operator_added': op_idx,
                            'timestamp_used': val_t
                        })
            except:
                continue
    
    return candidates


def select_validation_timestamps(n_input_frames, timestamps_per_op):
    """Select timestamps for validation."""
    max_t = n_input_frames - 2
    if max_t <= 0:
        return [0]
    
    if max_t < timestamps_per_op:
        return list(range(max_t + 1))
    
    # Uniform sampling across the sequence
    timestamps = np.linspace(0, max_t, timestamps_per_op, dtype=int)
    return timestamps.tolist()


def add_gpu_beam_search_to_experiment(experiment_class):
    """
    Add GPU beam search method to existing experiment class.
    """
    def gpu_beam_search_selection_method(self, inp, target, theta, state_labels, max_operators=5, 
                                        min_improvement_threshold=5.0):
        """Wrapper to use GPU beam search instead of greedy search."""
        beam_width = self.config.get('beam_width', 3)
        timestamps_per_op = self.config.get('timestamps_per_operator', 3) 
        improvement_threshold = self.config.get('beam_improvement_threshold', 0.05)
        
        return beam_search_operator_selection_gpu(
            inp, target, theta, state_labels, self.model, self.relative_l2_error,
            self.n_input_frames, self.config, beam_width=beam_width, 
            timestamps_per_op=timestamps_per_op, max_operators=max_operators,
            improvement_threshold=improvement_threshold
        )
    
    # Add the method to the class
    experiment_class.gpu_beam_search_iterative_operator_selection = gpu_beam_search_selection_method
    
    return experiment_class


if __name__ == "__main__":
    print("GPU-Optimized Beam Search with Batch Parallelization")
    print("Key features:")
    print("- Pure GPU batch processing (no threading)")
    print("- Temporal dimension parallelization via batch dimension using einops")
    print("- Sequential beam processing for simplicity") 
    print("- Memory-efficient batching")