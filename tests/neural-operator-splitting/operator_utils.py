"""
Operator utilities for neural operator splitting experiments.
"""

import torch
from torchdiffeq import odeint
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.operators.disco import vectors_to_parameters


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


def strang_splitting_composition(x, state_labels, operator_indices, theta, model, 
                                integration_time=1.0, n_future_steps=1, solver='rk4', rtol=1e-7, idx_to_theta=None, num_integration_steps=1):
    """
    Apply Strang splitting for operator composition.
    
    For operators [i1, i2, i3, ..., iN], constructs the sequence:
    [iN, iN-1, ..., i2, i1, i2, ..., iN-1, iN]
    
    Time steps: dt/2 for outer operators, dt for middle operator i1
    
    Args:
        x: input tensor
        state_labels: state labels
        operator_indices: list of operator indices
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
    
    n_ops = len(operator_indices)
    if n_ops < 1:
        raise ValueError("Strang splitting requires at least 1 operator")
    
    # Construct Strang splitting sequence: [iN, iN-1, ..., i2, i1, i2, ..., iN-1, iN]
    if n_ops == 1:
        # Special case: single operator gets full dt
        strang_sequence = [operator_indices[0]]
        time_steps = [dt_per_operator]
    else:
        # Backward sequence: [iN, iN-1, ..., i2] with dt/2
        backward_seq = operator_indices[1:][::-1]  # Skip i1, reverse the rest
        
        # Forward sequence: [i2, ..., iN-1, iN] with dt/2
        forward_seq = operator_indices[1:]  # Skip i1
        
        # Full sequence: backward + [i1] + forward
        strang_sequence = backward_seq + [operator_indices[0]] + forward_seq
        
        # Time steps: dt/2 for all except middle operator i1 which gets dt
        time_steps = [dt_per_operator/2] * len(backward_seq) + [dt_per_operator] + [dt_per_operator/2] * len(forward_seq)
    
    # Apply the Strang splitting composition at each time step
    for step in range(n_future_steps):
        step_result = current
        
        # For finer integration, apply the full Strang composition num_integration_steps times
        for substep in range(num_integration_steps):
            # Apply operators in Strang sequence
            for op_idx, dt in zip(strang_sequence, time_steps):
                if idx_to_theta is not None:
                    theta_idx = idx_to_theta[op_idx]
                    single_operator = get_batched_operators(theta[theta_idx:theta_idx+1], model, dim=1)
                else:
                    single_operator = get_batched_operators(theta[op_idx:op_idx+1], model, dim=1)
                
                step_result = solve_single_operator_ode(
                    step_result, single_operator, state_labels,
                    integration_time=dt, n_future_steps=1,
                    solver=solver, rtol=rtol
                )
                step_result = step_result[0]
        
        # Store the result of this full Strang composition step
        predictions.append(step_result)
        current = step_result  # Use this as input for next time step
    
    if n_future_steps == 1:
        return predictions[0]
    else:
        # Stack predictions: [n_future_steps, batch, channel, height]
        return torch.stack(predictions, dim=0)


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