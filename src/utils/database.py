from typing import Tuple
import sys
import contextlib
from typing import List
import numpy as np
from skimage.transform import resize
import torch 


@contextlib.contextmanager
def set_seed(seed):
    if seed is not None:
        saved_state = torch.random.get_rng_state()
        # Set the new seed
        torch.manual_seed(seed)
        try:
            yield
        finally:
            torch.random.set_rng_state(saved_state)
    else:
        # If seed is None, do nothing special
        yield


def to_torch(x: np.ndarray | torch.Tensor) -> torch.Tensor: 
    if isinstance(x, torch.Tensor):
        return x.to('cuda:0')
    return torch.tensor(x, device='cuda:0')


def batch_roll(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """ 
    Shift each column of x by the corresponding shift in shifts.

    :param x: tensor of shape B x T
    :param shifts: tensor of shape B
    """
    
    assert x.ndim == 2 and shifts.ndim == 1 and x.shape[0] == shifts.shape[0]

    idx = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1)
    shifts = shifts.unsqueeze(1)

    shifted_idx = (idx - shifts) % x.shape[1]

    return x.gather(1, shifted_idx)


def list_split(input_list: List, num_splits: int) -> List[List]:
    """ Split a list into multiple sub-lists. """

    if num_splits > len(input_list):
        raise ValueError("Cannot split a list with more splits than its actual size.")

    # calculate the approximate size of each sublist
    avg_size = len(input_list) // num_splits
    remainder = len(input_list) % num_splits

    # initialize variables
    start = 0
    end = avg_size
    sublists = []

    for i in range(num_splits):
        # adjust sublist size for the remainder
        if i < remainder:
            end += 1

        # create a sublist and add it to the result
        sublist = input_list[start:end]
        sublists.append(sublist)

        # update the start and end indices for the next sublist
        start = end
        end += avg_size

    return sublists


def divide_grid(grid: List, n_total: int, task_id: int) -> List:
    """ Divide a grid into n_total tasks and return the task corresponding to task_id.

    :param grid: list of arguments for each call
    :param n_total: number of workers
    :param task_id: id of this worker
    """
    # total number of tasks exceeds the grid size
    if n_total > len(grid):
        n_total = len(grid)
    # task id invalid: trying to complete null tasks
    if task_id > n_total:
        return []
    # task_id corresponds to a chunk of the grid
    return list_split(grid, n_total)[task_id]


def is_debug() -> bool:
    """ Detects whether the code is running on a local system file and not on the cluster. """
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    return False


def normalize(x: torch.Tensor, dims: Tuple, return_stats: bool = False):
    with torch.no_grad():
        l2 = x.pow(2.0).mean(dims, keepdims=True).sqrt()
        l2 = l2 + 1e-7
    if return_stats:
        return x / l2, l2
    return x / l2


def standardize(x: torch.Tensor, dims: Tuple, return_stats: bool = False):
    with torch.no_grad():
        x_std, x_mean = torch.std_mean(x, dim=dims, keepdims=True)
        x_std = x_std + 1e-7
    if return_stats:
        return (x - x_mean) / x_std, x_mean, x_std
    return (x - x_mean) / x_std


def region_standardize(x: torch.Tensor, separators: List, return_stats: bool = False): 
    """ Standardize the values of x based on the regions defined by the separators. """
    mean = torch.empty_like(x)
    std = torch.empty_like(x)
    for left, right in zip(separators[:-1], separators[1:]):
        mean[...,left:right] = x[...,left:right].mean(-1)
        std[...,left:right] = x[...,left:right].std(-1)
    if return_stats:
        return (x - mean) / std, mean, std
    return (x - mean) / std


def nrmpe(x_ref, x, dims, stand=False, p=1.0, avg=True, normalize=True): 
    x, y = x_ref.clone(), x.clone()
    if stand:
        x, y = standardize(x, dims=dims), standardize(y, dims=dims)
    residual = x - y
    loss = residual.abs().pow(p).mean(dims)
    if normalize:
        loss /= x.abs().pow(p).mean(dims)
    loss = loss.pow(1/p)
    if avg:
        return loss.mean()
    return loss


def subsample(x, reso):
    """ Subsample a numpy array (or cpu tensor) to a given resolution 
    using antialiasing Gaussian filter. """
    ndim = len(reso)
    output_shape = x.shape[:-ndim] + tuple(reso)
    if any(output_shape[d] > x.shape[d] for d in range(-ndim, 0)):
        return x
    if output_shape == x.shape:
        return x
    if isinstance(x, np.ndarray):
        return resize(x, output_shape, anti_aliasing=True)
    elif isinstance(x, torch.Tensor):
        x = x.numpy()
        x = resize(x, output_shape, anti_aliasing=True)
        return torch.tensor(x)


def clip_quantiles(data, lower_quantile=0.1, upper_quantile=0.9):
    """ Clip the data between the lower and upper quantiles. """
    # Compute the 0.1 and 0.9 quantiles
    lower_bound = torch.quantile(data, lower_quantile)
    upper_bound = torch.quantile(data, upper_quantile)
    
    # Clip the data between the computed quantiles
    clipped_data = torch.clamp(data, min=lower_bound.item(), max=upper_bound.item())
    
    return clipped_data
