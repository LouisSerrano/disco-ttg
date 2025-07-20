""" Examples of simple operators. The 'operator' here is the evolution operator
that takes initial conditions u0 and returns the solution at time t. """
from typing import *
from abc import ABC, abstractmethod
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve_banded

@contextlib.contextmanager
def set_seed(seed, backend='numpy'):
    if seed is not None:
        if backend == 'numpy':
            saved_state = np.random.get_state()
            # Set the new seed
            np.random.seed(seed)
            try:
                yield
            finally:
                np.random.set_state(saved_state)
        elif backend == 'torch':
            saved_state = torch.random.get_rng_state()
            # Set the new seed
            torch.manual_seed(seed)
            try:
                yield
            finally:
                torch.random.set_rng_state(saved_state)
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'torch' or 'numpy'")
    else:
        # If seed is None, do nothing special
        yield


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

#### Initial conditions ####

class InitialCondition(ABC):

    @abstractmethod
    def generate(self, batch_size: int, seed: int | None) -> torch.Tensor:
        pass


class ShiftedPattern(InitialCondition):
    """ Generate a shifted pattern, i.e. a space-localized signal, 
    that is randomly shifted within/across patches. """

    def __init__(self, 
        size: int, 
        patch_size: int | None, 
        pattern_size: int | None, 
    ):
        patch_size = patch_size or size
        self.pattern_size = pattern_size or patch_size
        if not size % patch_size == 0:
            raise ValueError("The length must be divisible by the patch size.")
        if patch_size is not None and self.pattern_size > patch_size:
            raise ValueError("The pattern must fit inside a patch.")
        self.size = size
        self.patch_size = patch_size

    @abstractmethod
    def generate_pattern(self, batch_size: int, seed: int | None) -> torch.Tensor:
        pass
    
    def generate(self, batch_size: int, seed: int | None) -> torch.Tensor:
        # generate the pattern (that fits within a patch)
        u0 = self.generate_pattern(batch_size, seed)

        # random shifts
        with set_seed(seed):
            # within a patch
            shift1 = torch.randint(0, max(1, self.patch_size - self.pattern_size), (batch_size,))
            # across different patches
            shift2 = self.patch_size * torch.randint(0, self.size, (batch_size,))
            shifts = (shift1 + shift2) % self.size
            u0 = batch_roll(u0, shifts)
            # multiply by a random sign
            sign = (torch.randn(batch_size) > 0) * 2  - 1
            u0 *= sign[:,None]

        return u0
    

class WhiteNoise(ShiftedPattern):

    def __init__(self, 
        amplitude: float,
        size: int, 
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, 1)
        self.amplitude = amplitude

    def generate_pattern(self, batch_size: int, seed: int | None):
        """ Generate half-circle. """
        u0 = torch.zeros(batch_size, self.size)

        with set_seed(seed):
            u0[:,:self.patch_size] = torch.randn(batch_size, self.patch_size) * self.amplitude

        return u0


class Dirac(ShiftedPattern):

    def __init__(self, 
        amplitude: float,
        size: int, 
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, 1)
        self.amplitude = amplitude

    def generate_pattern(self, batch_size: int, seed: int | None):
        """ Generate half-circle. """
        u0 = torch.zeros(batch_size, self.size)

        u0[:, 0] = self.amplitude

        return u0


class Rectangle(ShiftedPattern):

    def __init__(self, 
        end: int, 
        amplitude: float, 
        size: int, 
        patch_size: int | None = None
    ):
        self.end = end
        self.amplitude = amplitude
        super().__init__(size, patch_size, end)

    def generate_pattern(self, batch_size: int, seed: int | None):
        """ Generate rectangle. """
        u0 = torch.zeros(batch_size, self.size)
        u0[:, :self.end] = self.amplitude

        return u0


class Triangle(ShiftedPattern):

    def __init__(self, 
        center: int, 
        end: int, 
        amplitude: float,
        nb_triangles: int, 
        size: int, 
        patch_size: int | None = None
    ):
        if not (0 < center < end < size):
            raise ValueError("The triangle parameters must satisfy 0 <= t0 < t1 < t2 < length.")
        self.center = center
        self.end = end
        self.amplitude = amplitude
        self.nb_triangles = nb_triangles
        super().__init__(size, patch_size, end*nb_triangles)

    def generate_pattern(self, batch_size: int, seed: int | None):
        """ Generate triangle. """
        u0 = torch.zeros(batch_size, self.size)

        # single triangle
        triangle = torch.zeros(self.end+1)
        triangle[:self.center+1] = torch.linspace(0, self.amplitude, self.center+1)
        triangle[self.center:self.end+1] = torch.linspace(self.amplitude, 0, self.end-self.center+1)

        # multiple triangles
        pattern_size = self.end
        for k in range(self.nb_triangles):
            u0[:, k*pattern_size : (k+1)*pattern_size+1] = triangle[None,:]

        return u0


class HalfEllipse(ShiftedPattern):

    def __init__(self, 
        diameter: int, 
        amplitude: float,
        size: int, 
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, diameter)
        self.diameter = diameter
        self.amplitude = amplitude

    def generate_pattern(self, batch_size: int, seed: int | None):
        """ Generate half-circle. """
        u0 = torch.zeros(batch_size, self.size)

        theta = torch.linspace(0, torch.pi, self.diameter)
        u0[:,:self.diameter] = self.amplitude * torch.sin(theta)

        return u0


class Sine(ShiftedPattern):

    def __init__(self, 
        periods: int, 
        amplitude: float, 
        size: int, 
        patch_size: int | None = None,
        L: float = 16.0,
        num_points: int = 256
    ):
        super().__init__(size, patch_size, patch_size)
        self.amplitude = amplitude
        self.periods = periods
        self.L = L
        self.num_points = num_points

    def generate_pattern(self, batch_size: int, seed: int | None=None):
        """ Generate sinusoid. """
        np.random.seed(seed)
        ks = np.random.randint(1, 5, size=batch_size)
        phases = 2 * np.pi * np.random.rand(batch_size)
        x = np.linspace(0, self.L, self.num_points, endpoint=False)
        # Initial conditions: shape (batch_size, Nx)
        u0 = np.array([np.sin(2 * np.pi * k * x / self.L + phase) for k, phase in zip(ks, phases)])

        return torch.from_numpy(u0)


class GaussianMixtures(ShiftedPattern):

    def __init__(self,
        n_gaussians: int, 
        size: int,
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, patch_size)
        self.n_gaussians = n_gaussians

    def generate_pattern(self, batch_size: int, seed: int | None) -> torch.Tensor:
        """ Generate a mixture of Gaussians. """
        n = self.n_gaussians  # number of gaussians
        p = self.patch_size  # patch size

        with set_seed(seed):
            # define the gaussians
            centers = p * torch.rand(batch_size, n)
            sigma_min = p // 32
            sigma_max = p // 8
            sigmas = sigma_min + (sigma_max - sigma_min) * torch.rand(batch_size, n)
            xs = torch.arange(p)
            gaussians = (-(xs[None,None,:] - centers[:,:,None]).pow(2.0) / 2 / sigmas[:,:,None] ** 2).exp()  # b n p

            # random sum of the Gaussians
            summation = torch.randn(batch_size, n)
            v0 = (summation[:,:,None] * gaussians).sum(1) / np.sqrt(n)
    
        # make sure the signal is smoothly decaying at the edges
        v0 = v0 * torch.hann_window(v0.shape[-1])[None,:]

        # insert the pattern in the full signal
        u0 = torch.zeros(batch_size, self.size)
        u0[:, :p] = v0

        return u0


class SmoothPlateau(ShiftedPattern):

    def __init__(self,
        width_ratio: float, 
        pattern_size: int,
        size: int,
        patch_size: int | None = None,
    ):
        super().__init__(size, patch_size, pattern_size)
        self.width_ratio = width_ratio

    def generate_pattern(self, batch_size: int, seed: int | None) -> torch.Tensor:
        """ Generate a smooth plateau. """
        p = self.pattern_size
        rho = self.width_ratio

        x = torch.linspace(0,1,p)
        v0 = torch.ones(p)
        v0 *= torch.sigmoid(100*rho*(x-0.1))
        v0 *= torch.sigmoid(100*rho*(1-x-0.1))

        # insert the pattern in the full signal
        u0 = torch.zeros(batch_size, self.size)
        u0[:, :p] = v0

        return u0


class FourierBased(ShiftedPattern):
    """ Pattern reconstructed from independent Gaussian Fourier coefficients
        with a given power-spectrum (i.e. relative amplitude of these modes).  
        This is achieved through trigonometric polynomial 
        u0(sin(\theta})) = \sum_{k=1}^{degree} a_k P_k sin(k \theta)
        where P_k is the power-spectrum and a_k are independent Gaussian coefficients.""" 

    def __init__(self, 
        size: int, 
        patch_size: int | None, 
        degree: int
    ):
        super().__init__(size, patch_size, patch_size)
        self.degree = degree

    @abstractmethod
    def power_spectrum(self, degree: int) -> torch.Tensor:
        pass

    def generate_pattern(self, batch_size: int, seed: int | None) -> torch.Tensor:
        """ Generate fractaloid. """
        u0 = torch.zeros(batch_size, self.size)

        # phases 0, \theta, 2x\theta, ... degree x \theta
        #phase = torch.linspace(0, 2*torch.pi, self.patch_size)
        phase = np.linspace(0, 2*torch.pi, self.patch_size, endpoint=False)
        phase = torch.from_numpy(phase).float()
        phase = torch.arange(1,self.degree+1)[:,None] * phase[None,:]

        # random coefficients a_0, a_1, ... a_degree
        with set_seed(seed):
            proj = torch.randn(batch_size, self.degree)

        # apply the desired power spectrum 
        power_spectrum = self.power_spectrum(self.degree)
        proj = proj * power_spectrum[None,:]

        u0[:, :self.patch_size] = proj @ torch.sin(phase)

        return u0


class Fractaloid(FourierBased):
    """ Generate power-law power-spectrum random signals. Such signals are self-similar. """

    def __init__(self, 
        degree: int,
        power: float,
        size: int, 
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, degree)
        self.degree = degree
        self.power = power  # the larger the smoother the signal

    def power_spectrum(self, degree: int, batch_size: int | None = None) -> torch.Tensor:
        """ Power-spectrum of the fractaloid. """
        if batch_size is None:
            return torch.arange(1, degree+1) ** (-self.power)
        else:
            return torch.arange(1, degree+1) ** (-self.power)


class FourierSmooth(FourierBased):
    """ Generate exponential power-spectrum random signals. Such signals are smooth. """

    def __init__(self, 
        degree: int,
        smoothness: float,
        size: int, 
        patch_size: int | None = None
    ):
        """ smoothness: between 0 and 1, 0: white noise, 1: constant signal  """
        super().__init__(size, patch_size, degree)
        self.degree = degree
        self.smoothness = smoothness  # the larger the smoother the signal

    def power_spectrum(self, degree: int) -> torch.Tensor:
        """ Power-spectrum of the fractaloid. """
        fs = torch.linspace(0,1,degree)
        lam = 2 * self.smoothness * degree
        return torch.exp(-lam * fs)
    

#### Evolution operator ####

class Operator(ABC):

    def _input_checks(self, t: torch.Tensor, u: torch.Tensor):
        assert t.ndim in [0, 1]
        assert u.ndim in [3, 4], "u should have 1 or 2 space dimension max"

    @abstractmethod
    def __call__(self, t: torch.Tensor, u: torch.Tensor, seed: int | None) -> torch.Tensor:
        """ Take b c h w and return b t c h w  """
        pass


class AdvectionDiffusion(Operator):
    """ Also called AdvectionDiffusion, this is just the composition 
    of Translation and Diffusion (both commuting). """

    def __init__(self, velocity: Tuple, diffusivity: float):
        self.velocity = velocity
        self.diffusivity = diffusivity
    
    @staticmethod
    def _gaussian_kernels(ksize, sigmas, ndim):
        """ Init Gaussian kernels of size ksize and scale sigmas.

        :param ksize: int
        :param sigmas: 1d tensor
        :param ndim: spatial dimensions e.g. 1 for burgers, 2 for NS
        """
        coordinate = list(range(ksize//2+1)) + list(range(1,ksize//2+ksize%2))[::-1]
        # coordinates = [torch.arange(-ksize//2+1, ksize//2+1, device=sigmas.device)] * ndim
        coordinates = [torch.tensor(coordinate, device=sigmas.device)] * ndim
        grid = torch.meshgrid(*coordinates, indexing='ij')
        #distances_squared = sum((coord / sigmas[:,*(None,)*ndim]) ** 2 for coord in grid)
        distances_squared = sum((coord / sigmas.view(-1, *([1] * ndim))) ** 2 for coord in grid)
        print('distances_squared.shape', distances_squared.shape)
        kernel = torch.exp(-0.5*distances_squared)
        print('kernel.shape', kernel.shape)
        spatial_dims = tuple(range(ndim+1))[1:]
        kernel = kernel / kernel.sum(dim=spatial_dims, keepdim=True)
        print('kernel.shape', kernel.shape)

        return kernel

    def __call__(self, t: torch.Tensor, u: torch.Tensor, seed: int | None=None) -> torch.Tensor:
        
        self._input_checks(t, u)

        d = u.ndim - 2  # u should be b c h (w)
        dims = (-1,) if d == 1 else (-1, -2)

        print('t.shape', t.shape)
        print('u.shape', u.shape)

        # translation
        ut = []
        for it in t.float():
            #shift = [it.item() * vel for vel in self.velocity]
            shift = it * self.velocity
            if all(abs(vel) < 1e-9 for vel in shift):
                ut.append(u)
                continue
            if not all(vel.is_integer() for vel in shift):
                raise ValueError("Translation shift should be integer")
            shift = [int(s) for s in shift]
            ut.append(torch.roll(u, shifts=shift, dims=dims))
        ut = torch.stack(ut, dim=1)

        # diffusion
        if abs(self.diffusivity) < 1e-9:
            return ut
        sigmas = (2 * self.diffusivity * t) ** 0.5
        sigmas[sigmas<1e-9] = 1e-9
        print('sigmas.shape', sigmas.shape)
        k = self._gaussian_kernels(u.shape[-1], sigmas, d)

        # convolve in Fourier space
        ut = torch.fft.rfftn(ut, dim=dims)
        kf = torch.fft.rfftn(k.unsqueeze(1).unsqueeze(0), dim=dims)

        print('ut.shape', ut.shape)
        print('kf.shape', kf.shape)

        ut = torch.fft.irfftn(kf*ut, dim=dims)
        
        return ut


class AdvectionDiffusionExplicit(Operator):
    """Explicit finite difference scheme for 1D advection-diffusion, batch-compatible."""
    def __init__(self, velocity: float, diffusivity: float, length_of_domain: float, total_simulation_time: float, number_of_time_steps: int, number_of_snapshots: int):
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.length_of_domain = length_of_domain
        self.total_simulation_time = total_simulation_time
        self.number_of_time_steps = number_of_time_steps
        self.number_of_snapshots = number_of_snapshots

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # u: [batch_size, Nx]
        batch_size, Nx = u.shape
        L = self.length_of_domain
        T = self.total_simulation_time
        Nt = self.number_of_time_steps
        N_snapshots = self.number_of_snapshots
        dx = L / Nx
        dt = T / Nt
        freq = Nt // N_snapshots
        v = self.velocity
        D = self.diffusivity

        u_n = u.clone().cpu().numpy()  # work in numpy for easy rolling
        u_history = [u_n.copy()]
        for n in range(Nt - 1):
            u_left = np.roll(u_n, 1, axis=1)
            u_right = np.roll(u_n, -1, axis=1)
            advection = -v * (dt / (2 * dx)) * (u_right - u_left)
            diffusion = D * (dt / dx**2) * (u_right - 2 * u_n + u_left)
            u_n = u_n + advection + diffusion
            if n % freq == 0 and n > 0:
                u_history.append(u_n.copy())
        u_hist_np = np.stack(u_history, axis=1)  # [batch_size, N_snapshots, Nx]
        return torch.from_numpy(u_hist_np).to(u.device)


class AdvectionDiffusionLaxWendroff(Operator):
    """Explicit finite difference scheme for 1D advection-diffusion
    using Lax-Wendroff for advection, batch-compatible.
    """
    def __init__(self, velocity: float, diffusivity: float, length_of_domain: float, total_simulation_time: float, number_of_time_steps: int, number_of_snapshots: int):
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.length_of_domain = length_of_domain
        self.total_simulation_time = total_simulation_time
        self.number_of_time_steps = number_of_time_steps
        self.number_of_snapshots = number_of_snapshots

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # u: [batch_size, Nx]
        batch_size, Nx = u.shape
        L = self.length_of_domain
        T = self.total_simulation_time
        Nt = self.number_of_time_steps
        N_snapshots = self.number_of_snapshots
        dx = L / Nx
        dt = T / Nt
        freq = Nt // N_snapshots
        v = self.velocity
        D = self.diffusivity

        # Calculate the Courant number for convenience
        C = v * dt / dx

        u_n = u.clone().cpu().numpy()  # work in numpy for easy rolling
        u_history = [u_n.copy()]

        for n in range(Nt - 1):
            # Roll for neighbor values
            u_left = np.roll(u_n, 1, axis=1)
            u_right = np.roll(u_n, -1, axis=1)

            # Lax-Wendroff Advection Term:
            # -v * (dt / (2 * dx)) * (u_right - u_left)  (first-order central difference part)
            # + 0.5 * v^2 * (dt^2 / dx^2) * (u_right - 2 * u_n + u_left) (second-order correction)
            advection_lw = -C / 2 * (u_right - u_left) + 0.5 * C**2 * (u_right - 2 * u_n + u_left)

            # Explicit Diffusion Term (same as before)
            diffusion = D * (dt / dx**2) * (u_right - 2 * u_n + u_left)

            # Update u_n
            u_n = u_n + advection_lw + diffusion

            # Store snapshots
            if (n + 1) % freq == 0 and n > 0: # Check (n+1) because we want to save after the update
                u_history.append(u_n.copy())
        
        # Ensure the last snapshot is always included if N_snapshots is such that freq doesn't align
        # or if N_snapshots is very small
        if len(u_history) < N_snapshots:
             u_history.append(u_n.copy()) # Add the very last state
        elif len(u_history) > N_snapshots: # If somehow more than N_snapshots are collected (shouldn't happen with correct freq calculation)
             u_history = u_history[:N_snapshots]


        u_hist_np = np.stack(u_history, axis=1)  # [batch_size, N_snapshots, Nx]
        return torch.from_numpy(u_hist_np).to(u.device)
    
class AdvectionDiffusionImplicit(Operator):
    """Implicit finite difference scheme for 1D advection-diffusion
    using Crank-Nicolson with PERIODIC boundary conditions, batch-compatible.
    Uses central differencing for advection and diffusion.
    """
    def __init__(self, velocity: float, diffusivity: float, length_of_domain: float, total_simulation_time: float, number_of_time_steps: int, number_of_snapshots: int):
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.length_of_domain = length_of_domain
        self.total_simulation_time = total_simulation_time
        self.number_of_time_steps = number_of_time_steps
        self.number_of_snapshots = number_of_snapshots

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # u: [batch_size, Nx]
        batch_size, Nx = u.shape
        L = self.length_of_domain
        T = self.total_simulation_time
        Nt = self.number_of_time_steps
        N_snapshots = self.number_of_snapshots
        dx = L / Nx # For periodic, we typically consider Nx points representing Nx intervals, dx = L/Nx
        dt = T / Nt
        freq = Nt // N_snapshots
        v = self.velocity
        D = self.diffusivity

        # Coefficients for Crank-Nicolson
        alpha = D * dt / (2 * dx**2) # Diffusion coefficient
        beta = v * dt / (4 * dx)     # Advection coefficient

        u_n = u.clone().cpu().numpy().astype(np.float64)
        u_history = [u_n.copy()]

        # Define the main diagonals for the tridiagonal matrix A
        # The structure of the matrix for periodic BCs is:
        # B C 0 ... 0 A
        # A B C ... 0 0
        # 0 A B ... 0 0
        # ...
        # 0 0 0 ... A B C
        # C 0 0 ... 0 A B

        # Main diagonal terms for the tridiagonal part (B_i)
        main_diag_val = 1 + 2 * alpha
        # Upper diagonal terms (C_i)
        upper_diag_val = -alpha + beta
        # Lower diagonal terms (A_i)
        lower_diag_val = -alpha - beta

        # Arrays for the three main diagonals of the almost-tridiagonal system
        # (excluding the wrap-around terms for now)
        main_diag = np.full(Nx, main_diag_val)
        upper_diag = np.full(Nx, upper_diag_val)
        lower_diag = np.full(Nx, lower_diag_val)

        for n_step in range(Nt):
            u_next = np.zeros_like(u_n)

            for b in range(batch_size):
                u_current_batch = u_n[b, :]

                # Construct the right-hand side (RHS) vector for all Nx points
                # (alpha + beta) u_{i-1}^n + (1 - 2*alpha) u_i^n + (alpha - beta) u_{i+1}^n
                # Need to use periodic indexing for u_current_batch
                u_left_n = np.roll(u_current_batch, 1)
                u_right_n = np.roll(u_current_batch, -1)

                rhs_vector = (alpha + beta) * u_left_n + \
                             (1 - 2 * alpha) * u_current_batch + \
                             (alpha - beta) * u_right_n

                # --- Cyclic Tridiagonal Matrix Algorithm (TDMA) ---
                # This solves Ax = b where A is a cyclic tridiagonal matrix
                # The general idea is to modify the system into a standard tridiagonal system,
                # solve it twice, and combine the solutions.

                # 1. Solve A'x' = b (standard tridiagonal system)
                # A' is A without the (0, Nx-1) and (Nx-1, 0) elements.
                # This means the matrix for solve_banded will be Nx-1 x Nx-1 (ignoring last row/col)
                # Or, we can modify the first and last rows after construction.

                # Construct the banded matrix for scipy.linalg.solve_banded
                # This function expects a matrix with 3 rows:
                # row 0: upper diagonal (c_i) with a 0 at the start (Nx-1 elements)
                # row 1: main diagonal (b_i) (Nx elements)
                # row 2: lower diagonal (a_i) with a 0 at the end (Nx-1 elements)

                # For the standard TDMA part (ignoring wrap-around terms for a moment)
                ab = np.zeros((3, Nx))
                ab[0, 1:] = upper_diag[:-1]  # Upper diagonal (c_i)
                ab[1, :] = main_diag         # Main diagonal (b_i)
                ab[2, :-1] = lower_diag[1:]  # Lower diagonal (a_i)

                # Solve the regular tridiagonal system (ignoring last row for now)
                # We solve for u_prime using rhs_vector and then for y_prime using special vectors
                # This is simplified in the 'solve_banded' context by handling the boundary terms.

                # The cyclic TDMA involves solving two systems with slightly modified RHS vectors.
                # Let's call the full cyclic matrix A_c.
                # A_c x = b.  We can write A_c = T + uv^T where T is tridiagonal and uv^T is the perturbation.
                # (T is the matrix we'd have for fixed BCs where x_0 and x_{N-1} are not linked.)

                # Create two RHS vectors for the two sub-problems in cyclic TDMA:
                # d_vec (original RHS) and e_vec (a vector with 1 at start/end, 0 elsewhere)
                d_vec = rhs_vector.copy()
                e_vec = np.zeros(Nx)
                e_vec[0] = (-alpha - beta) # The coefficient A_N-1,0 in the original matrix
                e_vec[Nx-1] = (-alpha + beta) # The coefficient C_0,N-1 in the original matrix

                # Temporarily modify the main diagonal for the 'solve_banded' call
                # to represent fixed BCs for the sub-problems.
                # We use the full Nx system and then correct.

                # Step 1: Solve Ty_d = d_vec (original RHS)
                # Here, T is the matrix with fixed BCs, so we solve for all Nx points assuming fixed.
                # Adjusting ab based on the full matrix with 0s for wrap around for solve_banded
                ab_fixed_bc = np.zeros((3, Nx))
                ab_fixed_bc[0, 1:] = upper_diag[:-1] # c
                ab_fixed_bc[1, :] = main_diag         # b
                ab_fixed_bc[2, :-1] = lower_diag[1:] # a

                # Fix first and last rows to avoid linking them (treat as fixed BCs temporarily)
                # This usually means setting a_1 = 0 and c_{N-2} = 0, and handling the RHS.
                # For scipy.linalg.solve_banded, we just provide the diagonals as if they are
                # for a standard tridiagonal matrix of size Nx.

                # The standard way to implement cyclic TDMA (simplified here for Python)
                # is to solve Ax=b for the first N-1 equations, then use the last equation.
                # Or, solve twice: Ax=b and Ax=e where e has non-zero only at ends.

                # A more direct approach with solve_banded for cyclic:
                # 1. Define the almost-tridiagonal matrix without the corner elements.
                # 2. Add the corner elements (a[0,N-1] and a[N-1,0]) to a rank-2 matrix.
                # 3. Use Sherman-Morrison to invert (A_tridiagonal + uvT).
                # This is equivalent to solving two tridiagonal systems.

                # Let's implement using the two-solve approach
                # Solve the 'fixed boundary' system T x_hat = b_hat
                # where T is the tridiagonal matrix without the periodic wrap-around terms.
                # And solve T y_hat = e_hat where e_hat has 1 at ends.

                # Create the standard tridiagonal matrix diagonals (Nx-1 internal points)
                # The first row (index 0) and last row (index Nx-1) are special.
                # We solve for 0..Nx-1 by treating it as a slightly modified system.
                # For `solve_banded`, the `ab` matrix format is key.

                # The `solve_banded` function needs `m_l` lower diagonals and `m_u` upper diagonals.
                # For a tridiagonal system, `m_l=1`, `m_u=1`.
                # The `ab` array has dimensions (m_l + m_u + 1, N).

                # Build the diagonals for the standard tridiagonal system of size Nx:
                a_coeffs = np.full(Nx - 1, lower_diag_val)
                b_coeffs = np.full(Nx, main_diag_val)
                c_coeffs = np.full(Nx - 1, upper_diag_val)

                # Setup 'ab' for `solve_banded`:
                # row 0: upper diagonal (c_coeffs) shifted by 1
                # row 1: main diagonal (b_coeffs)
                # row 2: lower diagonal (a_coeffs) shifted by 0
                ab_standard = np.zeros((3, Nx))
                ab_standard[0, 1:] = c_coeffs
                ab_standard[1, :] = b_coeffs
                ab_standard[2, :-1] = a_coeffs

                # Solve the first system: T * u_tilde = rhs_vector
                u_tilde = solve_banded((1, 1), ab_standard, rhs_vector)

                # Solve the second system: T * y_tilde = e_vector
                # e_vector has specific values at the ends to represent the periodic wrap-around
                e_vector = np.zeros(Nx)
                e_vector[0] = main_diag_val * lower_diag_val + upper_diag_val * main_diag_val # Simplified for illustration; typically 1 or specific values
                e_vector[Nx-1] = 1.0 # Or more specific values depending on formulation

                # More accurate e_vector for cyclic TDMA
                e_vector = np.zeros(Nx)
                e_vector[0] = lower_diag_val # The a_0 coefficient from the cyclic matrix
                e_vector[Nx-1] = upper_diag_val # The c_{N-1} coefficient from the cyclic matrix

                y_tilde = solve_banded((1, 1), ab_standard, e_vector)

                # Correction factor for periodic BCs
                # This formula comes from applying Sherman-Morrison to T + uv^T
                gamma = (lower_diag_val * u_tilde[Nx-1] + upper_diag_val * u_tilde[0]) / \
                        (1 + lower_diag_val * y_tilde[Nx-1] + upper_diag_val * y_tilde[0])

                u_next[b, :] = u_tilde - gamma * y_tilde

            u_n = u_next # Update for the next time step

            # Store snapshots
            if (n_step + 1) % freq == 0:
                u_history.append(u_n.copy())

        # Ensure the last snapshot is always included if N_snapshots is such that freq doesn't align
        if len(u_history) < N_snapshots:
             u_history.append(u_n.copy())
        elif len(u_history) > N_snapshots:
             u_history = u_history[:N_snapshots]

        u_hist_np = np.stack(u_history, axis=1)  # [batch_size, N_snapshots, Nx]
        return torch.from_numpy(u_hist_np).to(u.device)

def calculate_numbers(velocities, diffusivities, set_name, dt=1/1000, dx=1/1024):
    print(f"--- {set_name} Set Calculations ---")
    results = []
    for v_idx, v in enumerate(velocities):
        for d_idx, D in enumerate(diffusivities):
            C = (abs(v) * dt) / dx
            Fo = (D * dt) / (dx**2)
            
            # Pe_Delta calculation: Handle D=0 case gracefully
            if D == 0:
                Pe_Delta = np.inf # Pure advection
            else:
                Pe_Delta = (abs(v) * dx) / D
            
            result = {
                "v": v,
                "D": D,
                "Courant (C)": f"{C:.4f}",
                "Diffusion Number (Fo)": f"{Fo:.4f}",
                "Grid Peclet (Pe_Delta)": f"{Pe_Delta:.4f}" if np.isfinite(Pe_Delta) else "Infinity (pure advection)"
            }
            results.append(result)
            print(f"v={v:<6.3f}, D={D:<7.4f}: C={C:.4f}, Fo={Fo:.4f}, Pe_Delta={Pe_Delta:.4f}" if np.isfinite(Pe_Delta) else f"v={v:<6.3f}, D={D:<7.4f}: C={C:.4f}, Fo={Fo:.4f}, Pe_Delta=Infinity (pure advection)")
    print("\n")
    return results


if __name__ == "__main__":
    batch_size = 7
    spatial_size = 128
    time_steps = 16
    t = torch.arange(time_steps)
    os.makedirs("plots", exist_ok=True)

    # List of initial condition classes and their parameters
    initial_conditions = [
        (Fractaloid, dict(degree=spatial_size, power=2.0, size=spatial_size, patch_size=spatial_size)),
        (FourierSmooth, dict(degree=spatial_size, smoothness=0.7, size=spatial_size, patch_size=spatial_size)),
        (WhiteNoise, dict(amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
        (Dirac, dict(amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
        (Rectangle, dict(end=32, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
        (Triangle, dict(center=16, end=32, amplitude=1.0, nb_triangles=2, size=spatial_size, patch_size=spatial_size)),
        (HalfEllipse, dict(diameter=32, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
        (Sine, dict(periods=3, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
        (GaussianMixtures, dict(n_gaussians=3, size=spatial_size, patch_size=spatial_size)),
        (SmoothPlateau, dict(width_ratio=0.5, pattern_size=32, size=spatial_size, patch_size=spatial_size)),
    ]

    operator = AdvectionDiffusion(velocity=(1,), diffusivity=0.1)

    for ic_cls, ic_kwargs in initial_conditions:
        try:
            ic = ic_cls(**ic_kwargs)
            u0 = ic.generate(batch_size, None)[:, None, :]
            ut = operator(t, u0, seed=None)  # shape: [B, T, C, H, (W)]
            ut = ut.squeeze(2)  # remove channel dim if present

            # Save trajectory as .npy
            np.save(f"plots/trajectory_{ic_cls.__name__}.npy", ut.cpu().numpy())

            # Plot a few trajectories through time
            for b in range(min(3, batch_size)):
                plt.figure(figsize=(10, 6))
                for ti in range(time_steps):
                    plt.plot(ut[b, ti].cpu(), label=f"t={ti}", alpha=0.5)
                plt.title(f"{ic_cls.__name__} Trajectory (batch {b})")
                plt.xlabel("Space")
                plt.ylabel("u")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"plots/{ic_cls.__name__}_batch{b}.png")
                plt.close()
        except Exception as e:
            print(f"Failed for {ic_cls.__name__}: {e}")
