"""
Data generation utilities for neural operator splitting experiments.
"""

import torch
from torch.utils.data import IterableDataset
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.advection_diffusion import FractaloidPhase
from train.train import advection_diffusion_analytical


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