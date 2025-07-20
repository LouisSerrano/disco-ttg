""" Examples of simple operators. The 'operator' here is the evolution operator
that takes initial conditions u0 and returns the solution at time t. """
from typing import *
from abc import ABC, abstractmethod
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


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
        patch_size: int | None = None
    ):
        super().__init__(size, patch_size, patch_size)
        self.amplitude = amplitude
        self.periods = periods

    def generate_pattern(self, batch_size: int, seed: int | None=None):
        """ Generate sinusoid. """
        u0 = torch.zeros(batch_size, self.size)

        phase = torch.linspace(0, 2*torch.pi*self.periods, self.patch_size+1)[:-1]
        u0[:,:self.patch_size] = self.amplitude * torch.sin(phase)

        return u0


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
        phase = torch.linspace(0, 2*torch.pi, self.patch_size)
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

    def power_spectrum(self, degree: int) -> torch.Tensor:
        """ Power-spectrum of the fractaloid. """
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
            shift = self.velocity
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
