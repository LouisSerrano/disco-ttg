"""Core test-time compute methods: direct prediction, greedy/random/beam/gradient operator selection."""
import torch
import random
import logging
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm


from test_time_compute.ttc_utils import DEVICE, N_INPUT_FRAMES, N_OUTPUT_FRAMES, get_relative_l2_error
from einops import rearrange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_state_labels(x):
    """Get appropriate state labels based on number of input channels.

    Args:
        x: Input tensor with shape (b, t, c, h) for 1D or (b, t, c, h, w) for 2D
           where c is the number of channels/state variables

    Returns:
        state_labels: torch.tensor with indices [0, 1, ..., c-1] for c channels
    """
    if len(x.shape) == 4:  # 1D case: (b, t, c, h)
        num_channels = x.shape[2]
    elif len(x.shape) == 5:  # 2D case: (b, t, c, h, w)
        num_channels = x.shape[2]
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}. Expected 4D (1D data) or 5D (2D data)")

    return torch.tensor(list(range(num_channels)), device=x.device)


def get_target_from_batch(batch, device):
    """Get target tensor from batch, checking for both 'target' and 'output' keys.

    Args:
        batch: Batch dictionary that may contain 'target' or 'output' key
        device: Device to move tensor to

    Returns:
        target tensor moved to specified device
    """
    if 'target' in batch:
        return batch['target'].to(device)
    elif 'output' in batch:
        return batch['output'].to(device)
    else:
        raise KeyError("Batch must contain either 'target' or 'output' key")


def test_direct_prediction(model, dataloader, dt):
    """Test standard model prediction"""
    relative_l2_error = get_relative_l2_error()
    total_error = 0
    samples = 0
    predictions = []

    for batch_idx, batch in enumerate(dataloader):
        inp = batch["input"].to(DEVICE)
        target = get_target_from_batch(batch, DEVICE)
        state_labels = get_state_labels(inp)

        with torch.no_grad():
            pred, _ = model(inp, state_labels, n_future_steps=target.shape[1])
            predictions.append(pred.cpu())
            error = relative_l2_error(pred, target).item()
            total_error += error * inp.shape[0]
            samples += inp.shape[0]

        # Print progress
        running_avg = total_error / samples
        logger.info(f"Batch {batch_idx + 1}: {samples} samples processed, running avg error: {running_avg:.6f}")

    avg_error = total_error / samples
    logger.info(f"Direct prediction complete: {samples} total samples, final avg error: {avg_error:.6f}")
    return avg_error, torch.cat(predictions)


def encode_operators_from_training_data(model, dataloader, num_operators=20, n_trajectories_per_operator=4):
    """Encode operators from training trajectories"""
    logger.info(f"Encoding {num_operators} operators from training data...")
    
    all_trajectories = []
    operator_metadata = []
    
    collected = 0
    target_trajectories = num_operators * n_trajectories_per_operator
    
    for batch in dataloader:
        if collected >= target_trajectories:
            break

        input_seq = batch['input']
        batch_size = input_seq.shape[0]

        for sample_idx in range(batch_size):
            if collected >= target_trajectories:
                break

            trajectory = input_seq[sample_idx:sample_idx+1]
            all_trajectories.append(trajectory)

            if collected % n_trajectories_per_operator == 0:
                # Create new operator metadata with operator_id and trajectory_indices
                operator_meta = {
                    'operator_id': collected // n_trajectories_per_operator,
                    'trajectory_indices': [],
                }

                # Add all relevant keys from the batch (excluding 'input' and non-parameter keys)
                for key, value in batch.items():
                    if key != 'input' and isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                        # Extract the value for this specific sample
                        if value.dim() == 1:  # 1D tensor (parameter values)
                            operator_meta[key] = value[sample_idx].item()
                        else:  # Multi-dimensional tensor, take the sample
                            operator_meta[key] = value[sample_idx].cpu().numpy().tolist()

                operator_metadata.append(operator_meta)

            operator_metadata[-1]['trajectory_indices'].append(len(all_trajectories) - 1)
            collected += 1
    
    # Encode all trajectories
    all_theta = []
    # Get state labels from the first trajectory to determine dimensionality
    state_labels = get_state_labels(all_trajectories[0])
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_trajectories), 32):
            batch_trajectories = all_trajectories[i:i+32]
            batch_input = torch.cat(batch_trajectories, dim=0).to(DEVICE)
            
            theta_latent_batch, _ = model.encode_theta_latent(batch_input, state_labels)
            # Determine dim based on input shape: 1 for 1D data, 2 for 2D data
            dim = 1 if len(batch_input.shape) == 4 else 2
            theta_batch = model.decode_theta(theta_latent_batch, dim=dim)
            all_theta.append(theta_batch.cpu())
    
    all_theta = torch.cat(all_theta, dim=0)
    
    # Average parameters for each operator
    theta_operators = torch.zeros(num_operators, all_theta.shape[1])
    
    for op_idx, op_meta in enumerate(operator_metadata):
        traj_indices = op_meta['trajectory_indices']
        theta_operators[op_idx] = all_theta[traj_indices].mean(dim=0)
    
    return theta_operators, None, operator_metadata


def greedy_operator_selection(model, theta_latent_operators, inp, target, max_operators=5, min_improvement_threshold=5.0, dt=4/250, splitting_method="strang", refinement_factor=1):
    """Greedy operator selection"""
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    #print('inp', inp.shape)
    #print('target', target.shape)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")

    #print('x_val', x_val.shape)
    #print('y_val', y_val.shape)

    #print('theta_operators_latent', theta_latent_operators.shape)
    
    current_composition = []
    current_best_error = float('inf')

    model.eval()
    
    history = {
        'compositions': [],
        'errors': [],
        'method': 'greedy'
    }
    
    best_composition = []
    best_error = float('inf')
    best_pred = None
    
    current_composition = []
    current_operators = []
    current_latent_operators = []

    with torch.no_grad():
        dim = 1 if len(inp.shape) == 4 else 2
        all_operators = model.decode_theta(theta_latent_operators, dim)
    
    for step in range(max_operators):
        best_error_for_step = float('inf')
        best_composition_for_step = None
        best_operator_added = None
        
        # Try adding each operator
        for op_idx in range(theta_latent_operators.shape[0]):
            test_composition = current_composition + [op_idx]
            #print('test_composition', test_composition)
            test_latent_operators = current_latent_operators + [theta_latent_operators[op_idx].unsqueeze(0).repeat(x_val.shape[0], 1)]
            test_operators = current_operators + [all_operators[op_idx].unsqueeze(0).repeat(x_val.shape[0], 1)]
            
            # Predict with this composition
            if len(test_latent_operators) == 1:
                with torch.no_grad():
                    pred, _ = model.solve_ode(x_val, test_operators[0], state_labels,
                                            dim=dim, integration_time=dt, n_future_steps=1, dt=dt)
            else:
                with torch.no_grad():
                    dim = 1 if len(inp.shape) == 4 else 2
                    pred = multi_operator_splitting(model, test_latent_operators, x_val,
                                                         nt=1, dt=dt, refinement_factor=refinement_factor,
                                                         splitting_method=splitting_method)
            
            error = relative_l2_error(pred, y_val).item()
            #print('error', error)
            
            if error < best_error_for_step:
                best_error_for_step = error
                #best_composition_for_step = composition.copy()
                best_operator_added = op_idx
                        
                best_composition = test_composition.copy()
                best_latent_operators = test_latent_operators.copy()

        if step == 0:
            improvement = 0
            should_continue = True
        else:
            improvement = (current_best_error - best_error_for_step) / current_best_error * 100
            should_continue = improvement >= min_improvement_threshold

        logger.info(f"  Step {step}: Best operator: {best_operator_added}, error: {best_error_for_step:.6f}")
        if step > 0:
            logger.info(f"  Improvement: {improvement:+.2f}% (threshold: {min_improvement_threshold}%)")

        if should_continue and best_error_for_step < current_best_error:
            current_composition = best_composition.copy()
            current_best_error = best_error_for_step
            current_operators = [all_operators[idx].unsqueeze(0).repeat(x_val.shape[0], 1) for idx in current_composition]
            current_latent_operators = [theta_latent_operators[idx].unsqueeze(0).repeat(x_val.shape[0], 1) for idx in current_composition]
        else:
            logger.info(f"  Stopping - insufficient improvement (need {min_improvement_threshold}%, got {improvement:.2f}%)")
            break

    with torch.no_grad():
        logger.info(f'current_composition: {current_composition}')
        if len(current_composition) > 1:
            best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in current_composition]
            pred = multi_operator_splitting(model, best_latent_operators, inp[:, -1], nt=target.shape[1], dt=dt,
                                          refinement_factor=refinement_factor, splitting_method=splitting_method)
        else:
            best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in current_composition]
            best_operators = model.decode_theta(best_latent_operators[0], dim)
            pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                            dim=dim, integration_time=dt, n_future_steps=target.shape[1], dt=dt)
        #pred = rearrange(pred, 't b ... -> b t c h')
        #print('pred', pred.shape)
        test_error = relative_l2_error(pred, target).item()
        logger.info(f'test_error: {test_error}')
    
    return best_composition, best_error, best_pred


def random_operator_selection(model, theta_latent_operators, inp, target, num_compositions=100, composition_lengths=[2], dt=4/250, splitting_method="strang", refinement_factor=1):
    """Random operator selection"""
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")

    model.eval()

    best_composition = []
    best_error = float('inf')
    best_pred = None
    best_latent_operators = []

    dim = 1 if len(inp.shape) == 4 else 2

    for _ in range(num_compositions):
        # Random composition length
        length = random.choice(composition_lengths)

        # Random operator indices
        composition = [random.randint(0, theta_latent_operators.shape[0]-1) for _ in range(length)]
        test_latent_operators = [theta_latent_operators[idx].unsqueeze(0).repeat(x_val.shape[0], 1) for idx in composition]

        # Predict
        if len(test_latent_operators) == 1:
            with torch.no_grad():
                # Decode the single operator for this batch
                #print('test_latent_operators', len(test_latent_operators), test_latent_operators[0].shape)
                test_operators = model.decode_theta(test_latent_operators[0], dim)
                pred, _ = model.solve_ode(x_val, test_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=1, dt=dt)
        else:
            with torch.no_grad():
                #print(test_latent_operators[0].shape)
                #print(test_latent_operators[1].shape)
                pred = multi_operator_splitting(model, test_latent_operators, x_val,
                                                     nt=1, dt=dt)

        error = relative_l2_error(pred, y_val).item()

        if error < best_error:
            best_error = error
            best_composition = composition
            # Store the latent operators for final prediction
            best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in composition]

    # Final prediction on full sequence
    if len(best_composition) > 0:
        with torch.no_grad():
            if len(best_composition) > 1:
                pred = multi_operator_splitting(model, best_latent_operators, inp[:, -1], nt=target.shape[1], dt=dt,
                                              refinement_factor=refinement_factor, splitting_method=splitting_method)
            else:
                best_operators = model.decode_theta(best_latent_operators[0], dim)
                pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=target.shape[1], dt=dt)
    else:
        # Fallback: return zeros if no valid composition found
        pred = torch.zeros_like(target)

    test_error = relative_l2_error(pred, target).item()
    logger.info(f'test_error: {test_error}')

    return best_composition, test_error, pred



def random_operator_selection_batch(model, theta_latent_operators, inp, target, num_compositions=128, composition_lengths=[2], dt=4/250, random_batch_size=32, splitting_method="strang", refinement_factor=1):
    """Random operator selection"""
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")

    original_shape = x_val.shape
    repeat_pattern = (random_batch_size,) + (1,) * len(original_shape)
    x_val = x_val.unsqueeze(0).repeat(repeat_pattern)
    x_val = rearrange(x_val, 'b t ... -> (b t) ...')

    model.eval()

    best_composition = []
    best_error = float('inf')
    best_pred = None
    best_latent_operators = []

    dim = 1 if len(inp.shape) == 4 else 2

    for iter_idx in range(0, num_compositions, random_batch_size):
        # Random composition length
        length = random.choice(composition_lengths)

        # Random operator indices
        compositions = [[random.randint(0, theta_latent_operators.shape[0]-1) for _ in range(length)] for _ in range(random_batch_size)]

        # Create test latent operators for all compositions
        test_latent_operators_batch = []
        for i in range(length):
            comp_operators = torch.stack([theta_latent_operators[compositions[k][i]].unsqueeze(0).repeat(original_shape[0], 1) for k in range(random_batch_size)]) #.repeat(x_val.shape[0], 1)
            comp_operators = rearrange(comp_operators, 'b t c -> (b t) c')
            test_latent_operators_batch.append(comp_operators)

        # we have a list of shape [B, C]
        # and x_val is of shape (T-1, C, H)
        # so we need to expand to (T-1, B, H)
        # so we need to expand to (B, T-1, H)
        # then we need to reshape the predictions and average them to get B scores

        
        #y_val = rearrange(y_val, 'b t c -> (b t) c')
        # Predict

        if length == 1:
            with torch.no_grad():
                test_operators = model.decode_theta(test_latent_operators_batch[0], dim)
                pred, _ = model.solve_ode(x_val, test_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=1, dt=dt)
                pred = rearrange(pred, '(b t) ... -> b t ...', b=random_batch_size)

        else:
            with torch.no_grad():
                pred = multi_operator_splitting(model, test_latent_operators_batch, x_val,
                                                     nt=1, dt=dt, refinement_factor=refinement_factor,
                                                     splitting_method=splitting_method)
                pred = rearrange(pred, '(b t) ... -> b t ...', b=random_batch_size)

        errors = [relative_l2_error(pred[k], y_val).item() for k in range(random_batch_size)]

        for k, error in enumerate(errors):
            if error < best_error:
                best_error = error
                best_composition = compositions[k]
                # Store the latent operators for final prediction
                best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in best_composition]

    # Final prediction on full sequence
    if len(best_composition) > 0:
        with torch.no_grad():
            if len(best_composition) > 1:
                pred = multi_operator_splitting(model, best_latent_operators, inp[:, -1], nt=target.shape[1], dt=dt,
                                              refinement_factor=refinement_factor, splitting_method=splitting_method)
            else:
                best_operators = model.decode_theta(best_latent_operators[0], dim)
                pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=target.shape[1], dt=dt)
    else:
        # Fallback: return zeros if no valid composition found
        pred = torch.zeros_like(target)

    test_error = relative_l2_error(pred, target).item()
    logger.info(f'test_error: {test_error}')

    return best_composition, test_error, pred





def multi_operator_splitting_simple(model, theta_operators, x, nt=10, dt=4/250, refinement_factor=5):
    """Simple multi-operator splitting using Lie splitting"""
    state_labels = get_state_labels(x.unsqueeze(1))  # Add time dimension for shape detection
    small_dt = dt / refinement_factor
    
    pred = x
    trajectory_pred = []
    
    for t_idx in range(nt):
        for ref_step in range(refinement_factor):
            # Sequential application of operators
            for theta in theta_operators:
                dim = 1 if len(x.shape) == 3 else 2
                pred, _ = model.solve_ode(pred, theta, state_labels, dim=dim,
                                        n_future_steps=1, integration_time=small_dt, dt=small_dt)
                pred = pred[:, -1]
        
        trajectory_pred.append(pred)
    
    return torch.stack(trajectory_pred, dim=1)


def multi_operator_splitting(model, theta_latents, x, nt=1, dt=4/250, refinement_factor=1, splitting_method="strang"):
    """Full multi-operator splitting with different methods"""
    state_labels = get_state_labels(x.unsqueeze(1))  # Add time dimension for shape detection
    dim = 1 if len(x.shape) == 3 else 2  # dim=1 for 1D data, dim=2 for 2D data
    
    small_dt = dt / refinement_factor
    k_operators = len(theta_latents)
    
    # Decode all operators
    thetas = [model.decode_theta(theta_latent, dim) for theta_latent in theta_latents]
    
    pred = x
    trajectory_pred = []
    
    for t_idx in range(nt):
        for ref_step in range(refinement_factor):
            
            if splitting_method == 'strang' and k_operators > 1:
                # Strang splitting
                # Forward pass
                for i in range(k_operators - 1):
                    pred, _ = model.solve_ode(pred, thetas[i], state_labels, dim, 
                                            n_future_steps=1, integration_time=small_dt/2, dt=small_dt/2)
                    pred = pred[:, -1]
                
                # Full step for last operator
                pred, _ = model.solve_ode(pred, thetas[-1], state_labels, dim, 
                                        n_future_steps=1, integration_time=small_dt, dt=small_dt)
                pred = pred[:, -1]
                
                # Backward pass
                for i in range(k_operators - 2, -1, -1):
                    pred, _ = model.solve_ode(pred, thetas[i], state_labels, dim, 
                                            n_future_steps=1, integration_time=small_dt/2, dt=small_dt/2)
                    pred = pred[:, -1]
            
            else:  # Default to Lie splitting
                for theta in thetas:
                    pred, _ = model.solve_ode(pred, theta, state_labels, dim, 
                                            n_future_steps=1, integration_time=small_dt, dt=small_dt)
                    pred = pred[:, -1]
        
        trajectory_pred.append(pred)
    
    trajectory_pred = torch.stack(trajectory_pred, dim=1)
    return trajectory_pred


def beam_search_operator_selection(model, theta_latent_operators, inp, target,
                                  beam_width=3, max_operators=5, min_improvement_threshold=5.0, dt=4/250,
                                  splitting_method="strang", refinement_factor=1):
    """Beam search operator selection that maintains top-k combinations at each step"""
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")

    model.eval()
    dim = 1 if len(inp.shape) == 4 else 2

    # Decode all operators once
    with torch.no_grad():
        all_operators = model.decode_theta(theta_latent_operators, dim)

    # Initialize beam with empty compositions
    # Each beam item: (composition, error, latent_operators, operators)
    # Start with empty beam for step 0
    current_beams = [([], float('inf'), [], [])]

    best_overall_composition = []
    best_overall_error = float('inf')
    best_overall_pred = None

    logger.info(f"Starting beam search with beam width {beam_width}")

    for step in range(max_operators):
        next_beams = []

        # For each current beam, try adding each operator
        for beam_composition, beam_error, beam_latent_ops, beam_ops in current_beams:
            for op_idx in range(theta_latent_operators.shape[0]):
                # Skip if operator already in composition
                if op_idx in beam_composition:
                    continue

                new_composition = beam_composition + [op_idx]
                new_latent_ops = beam_latent_ops + [theta_latent_operators[op_idx].unsqueeze(0).repeat(x_val.shape[0], 1)]
                new_ops = beam_ops + [all_operators[op_idx].unsqueeze(0).repeat(x_val.shape[0], 1)]

                # Predict with this composition
                if len(new_latent_ops) == 1:
                    with torch.no_grad():
                        pred, _ = model.solve_ode(x_val, new_ops[0], state_labels,
                                                dim=dim, integration_time=dt, n_future_steps=1, dt=dt)
                else:
                    with torch.no_grad():
                        pred = multi_operator_splitting(model, new_latent_ops, x_val,
                                                       nt=1, dt=dt, refinement_factor=refinement_factor,
                                                       splitting_method=splitting_method)

                error = relative_l2_error(pred, y_val).item()
                next_beams.append((new_composition, error, new_latent_ops, new_ops))

        # Sort beams by error and keep top beam_width
        next_beams.sort(key=lambda x: x[1])
        current_beams = next_beams[:beam_width]

        # Check improvement criterion
        if step > 0 and len(current_beams) > 0:
            best_current_error = current_beams[0][1]
            # Compare with the best error from the previous step
            if best_current_error < best_overall_error:
                improvement = (best_overall_error - best_current_error) / best_overall_error * 100
            else:
                improvement = 0.0

            if len(current_beams[0][0]) > 1 and improvement < min_improvement_threshold:
                logger.info(f"  Step {step}: Stopping - insufficient improvement ({improvement:.2f}% < {min_improvement_threshold}%)")
                break

        # Print current beam status
        logger.info(f"  Step {step}: Top beams (threshold: {min_improvement_threshold}%):")
        for i, (composition, error, _, _) in enumerate(current_beams[:beam_width]):  # Show top 3
            logger.info(f"    Beam {i+1}: {composition} -> error: {error:.6f}")

        # Update best overall
        if len(current_beams) > 0 and current_beams[0][1] < best_overall_error:
            best_overall_composition = current_beams[0][0].copy()
            best_overall_error = current_beams[0][1]

    # Final prediction with best composition
    if len(best_overall_composition) > 0:
        with torch.no_grad():
            logger.info(f'Final best composition: {best_overall_composition}')
            if len(best_overall_composition) > 1:
                best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in best_overall_composition]
                pred = multi_operator_splitting(model, best_latent_operators, inp[:, -1], nt=target.shape[1], dt=dt,
                                              refinement_factor=refinement_factor, splitting_method=splitting_method)
            else:
                best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in best_overall_composition]
                best_operators = model.decode_theta(best_latent_operators[0], dim)
                pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                        dim=dim, integration_time=dt, n_future_steps=target.shape[1], dt=dt)
            test_error = relative_l2_error(pred, target).item()
            logger.info(f'Final test error: {test_error}')
            best_overall_pred = pred

    return best_overall_composition, best_overall_error, best_overall_pred


def beam_search_operator_selection_batch(model, theta_latent_operators, inp, target,
                                        beam_width=3, max_operators=5, min_improvement_threshold=5.0,
                                        dt=4/250, batch_size=32, splitting_method="strang", refinement_factor=1):
    """Beam search operator selection with batched evaluation for acceleration"""
    relative_l2_error = get_relative_l2_error()
    theta_latent_operators = theta_latent_operators.to(DEVICE)
    state_labels = get_state_labels(inp)

    x_val = rearrange(inp[:, :-1], "b t ... -> (b t) ...")
    y_val = rearrange(inp[:, 1:], "b t ... -> (b t) ...")
    original_shape = x_val.shape

    model.eval()
    dim = 1 if len(inp.shape) == 4 else 2

    # Decode all operators once
    with torch.no_grad():
        all_operators = model.decode_theta(theta_latent_operators, dim)

    # Initialize beam with empty compositions
    # Each beam item: (composition, error, latent_operators, operators)
    current_beams = [([], float('inf'), [], [])]

    best_overall_composition = []
    best_overall_error = float('inf')
    best_overall_pred = None

    logger.info(f"Starting batched beam search with beam width {beam_width}, batch size {batch_size}")

    for step in range(max_operators):
        # Collect all candidate compositions for this step
        candidates = []

        for beam_composition, beam_error, beam_latent_ops, beam_ops in current_beams:
            for op_idx in range(theta_latent_operators.shape[0]):
                # Skip if operator already in composition
                if op_idx in beam_composition:
                    continue

                new_composition = beam_composition + [op_idx]
                new_latent_ops = beam_latent_ops + [theta_latent_operators[op_idx]]
                new_ops = beam_ops + [all_operators[op_idx]]

                candidates.append((new_composition, new_latent_ops, new_ops))

        if not candidates:
            break

        # Process candidates in batches
        evaluated_candidates = []

        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            current_batch_size = len(batch_candidates)

            # Prepare batched inputs
            repeat_pattern = (current_batch_size,) + (1,) * len(original_shape)
            x_val_batch = x_val.unsqueeze(0).repeat(repeat_pattern)
            x_val_batch = rearrange(x_val_batch, 'b t ... -> (b t) ...')

            # Group by composition length for efficient processing
            length_groups = {}
            for i, (composition, latent_ops, ops) in enumerate(batch_candidates):
                length = len(composition)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append((i, composition, latent_ops, ops))

            # Process each length group
            batch_errors = [float('inf')] * current_batch_size

            for length, group in length_groups.items():
                if length == 1:
                    # Single operator case
                    group_ops = torch.stack([ops[0] for _, _, _, ops in group])
                    group_ops = group_ops.unsqueeze(1).repeat(1, original_shape[0], 1)
                    group_ops = rearrange(group_ops, 'b t c -> (b t) c')

                    with torch.no_grad():
                        pred, _ = model.solve_ode(x_val_batch[:len(group)*original_shape[0]],
                                                group_ops, state_labels,
                                                dim=dim, integration_time=dt,
                                                n_future_steps=1, dt=dt)
                        pred = rearrange(pred, '(b t) ... -> b t ...', b=len(group))
                else:
                    # Multi-operator case - prepare latent operators for batch processing
                    group_latent_ops = []
                    for op_idx in range(length):
                        op_stack = torch.stack([latent_ops[op_idx] for _, _, latent_ops, _ in group])
                        op_stack = op_stack.unsqueeze(1).repeat(1, original_shape[0], 1)
                        op_stack = rearrange(op_stack, 'b t c -> (b t) c')
                        group_latent_ops.append(op_stack)

                    with torch.no_grad():
                        pred = multi_operator_splitting(model, group_latent_ops,
                                                      x_val_batch[:len(group)*original_shape[0]],
                                                      nt=1, dt=dt, refinement_factor=refinement_factor,
                                                      splitting_method=splitting_method)
                        pred = rearrange(pred, '(b t) ... -> b t ...', b=len(group))

                # Calculate errors for this group
                for j, (batch_idx, _, _, _) in enumerate(group):
                    error = relative_l2_error(pred[j], y_val).item()
                    batch_errors[batch_idx] = error

            # Add evaluated candidates
            for i, (composition, latent_ops, ops) in enumerate(batch_candidates):
                evaluated_candidates.append((composition, batch_errors[i], latent_ops, ops))

        # Sort all candidates by error and keep top beam_width
        evaluated_candidates.sort(key=lambda x: x[1])
        current_beams = evaluated_candidates[:beam_width]

        # Check improvement criterion
        if step > 0 and len(current_beams) > 0:
            best_current_error = current_beams[0][1]
            if best_current_error < best_overall_error:
                improvement = (best_overall_error - best_current_error) / best_overall_error * 100
            else:
                improvement = 0.0

            if len(current_beams[0][0]) > 1 and improvement < min_improvement_threshold:
                logger.info(f"  Step {step}: Stopping - insufficient improvement ({improvement:.2f}% < {min_improvement_threshold}%)")
                break

        # Print current beam status
        logger.info(f"  Step {step}: Top beams (threshold: {min_improvement_threshold}%):")
        for i, (composition, error, _, _) in enumerate(current_beams[:3]):
            logger.info(f"    Beam {i+1}: {composition} -> error: {error:.6f}")

        # Update best overall
        if len(current_beams) > 0 and current_beams[0][1] < best_overall_error:
            best_overall_composition = current_beams[0][0].copy()
            best_overall_error = current_beams[0][1]

    # Final prediction with best composition
    if len(best_overall_composition) > 0:
        with torch.no_grad():
            best_latent_operators = [theta_latent_operators[idx].unsqueeze(0) for idx in best_overall_composition]
            if len(best_overall_composition) > 1:
                pred = multi_operator_splitting(model, best_latent_operators, inp[:, -1],
                                              nt=target.shape[1], dt=dt, refinement_factor=refinement_factor,
                                              splitting_method=splitting_method)
            else:
                best_operators = model.decode_theta(best_latent_operators[0], dim)
                pred, _ = model.solve_ode(inp[:, -1], best_operators, state_labels,
                                        dim=dim, integration_time=dt,
                                        n_future_steps=target.shape[1], dt=dt)

            test_error = relative_l2_error(pred, target).item()
            logger.info(f'Final test error: {test_error}')
            best_overall_pred = pred
    else:
        test_error = float('inf')
        best_overall_pred = torch.zeros_like(target)

    return best_overall_composition, test_error, best_overall_pred


def gradient_selection_multi_operator(model, theta_operators, test_input, test_target,
                                    num_operators=3, epochs=500, lr=0.01,
                                    refinement_factor=1, splitting_method="strang",
                                    aux_loss_weight=0, dt=4/250, theta_dim=3):
    """Multi-operator gradient selection"""
    logger.info(f"Running gradient based operator selection with {num_operators} operators...")
    
    theta_operators = theta_operators.to(DEVICE)
    test_input = test_input.to(DEVICE)
    test_target = test_target.to(DEVICE)

    # Get state labels based on test input shape
    state_labels = get_state_labels(test_input)

    loss_fn = get_relative_l2_error()
    
    # Initialize theta latents
    theta_latents = []
    for i in range(num_operators):
        theta_latent = (torch.randn((test_input.shape[0], theta_dim), device=DEVICE) * 0.01).requires_grad_()
        theta_latents.append(theta_latent)
    
    #optimizer = torch.optim.AdamW(theta_latents, lr=lr, weight_decay=0.0, betas=(0., 0.5))
    optimizer = torch.optim.AdamW(theta_latents, lr=lr, weight_decay=0.0, betas=(0.5, 0.5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for step in range(epochs):
        # Random time step
        npred = 1
        t = random.randint(0, test_input.shape[1] - npred - 1)
        x_val = test_input[:, t]
        y_val = test_input[:, t + 1:t+1+npred]
        
        # Predict using multi-operator splitting
        pred = multi_operator_splitting(model, theta_latents, x_val, nt=npred, dt=dt,
                                      refinement_factor=refinement_factor,
                                      splitting_method=splitting_method,
                                      )
        
        # Main loss
        loss = loss_fn(pred, y_val)
        
        # Simple manifold loss
        if aux_loss_weight > 0:
            total_manifold_loss = 0
            for theta_latent in theta_latents:
                dim = 1 if len(test_input.shape) == 4 else 2
                theta_decoded = model.decode_theta(theta_latent, dim=dim)
                distances = torch.cdist(theta_decoded, theta_operators)
                min_dist = torch.min(distances, dim=1)[0]
                manifold_loss = torch.relu(min_dist - 0.01).mean()
                total_manifold_loss += aux_loss_weight * manifold_loss

        # this might slow down the finetuning
        else:
            total_manifold_loss = torch.tensor([0], device=pred.device)
        
        #total_loss = loss + total_manifold_loss
        
        total_loss = loss + total_manifold_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print progress
        if step % 50 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.6f}, Manifold loss: {total_manifold_loss.item():.6f}")

        # Evaluation
        if step % 100 == 0:
            with torch.no_grad():
                pred_test = multi_operator_splitting(model, theta_latents, test_input[:, -1],
                                                   nt=test_target.shape[1], dt=dt,
                                                   refinement_factor=refinement_factor,
                                                   splitting_method=splitting_method)
            test_error = loss_fn(pred_test, test_target).item()
            logger.info(f"Test error for {test_target.shape[1]} frames: {test_error:.6f}")
    
    return theta_latents, pred_test, test_error
