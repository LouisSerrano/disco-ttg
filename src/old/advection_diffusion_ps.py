import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff
import concurrent.futures

# Top-level worker function for multiprocessing
def solve_one_ivp_worker(args):
    u0, v, D, L, T, t_eval = args
    from scipy.fftpack import diff as psdiff  # ensure import in subprocess
    from scipy.integrate import solve_ivp
    def rhs(t, u):
        ux = psdiff(u, period=L, order=1)
        uxx = psdiff(u, period=L, order=2)
        return -v * ux + D * uxx
    sol = solve_ivp(rhs, [0, T], u0, t_eval=t_eval, method='Radau', atol=1e-12, rtol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y.T  # shape [N_snapshots, Nx]

class AdvectionDiffusionPseudospectral:
    """
    Pseudospectral (Fourier) method for 1D advection-diffusion, batch-compatible.
    Solves: u_t + v u_x = D u_xx
    Uses solve_ivp for time integration and psdiff for spatial derivatives.

    Args:
        parallel (bool): If True, parallelize across batch using multiple CPU cores.
    """
    def __init__(self, velocity: float, diffusivity: float, length_of_domain: float, total_simulation_time: float, number_of_time_steps: int, number_of_snapshots: int):
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.length_of_domain = length_of_domain
        self.total_simulation_time = total_simulation_time
        self.number_of_time_steps = number_of_time_steps
        self.number_of_snapshots = number_of_snapshots

    def __call__(self, u: torch.Tensor, parallel: bool = True) -> torch.Tensor:
        """
        Args:
            u (torch.Tensor): Initial conditions, shape [batch_size, Nx]
            parallel (bool): If True, use multiple CPU cores for batch ODE solves.
        Returns:
            torch.Tensor: Trajectories, shape [batch_size, n_snapshots, Nx]
        """
        batch_size, Nx = u.shape
        L = self.length_of_domain
        T = self.total_simulation_time
        N_snapshots = self.number_of_snapshots
        t_eval = np.linspace(0, T, N_snapshots)
        v = self.velocity
        D = self.diffusivity
        u_np = u.detach().cpu().numpy()
        args_list = [(u_np[b], v, D, L, T, t_eval) for b in range(batch_size)]
        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(solve_one_ivp_worker, args_list))
        else:
            results = [solve_one_ivp_worker(args) for args in args_list]
        results = np.stack(results, axis=0)  # [batch_size, N_snapshots, Nx]
        return torch.from_numpy(results).to(u.device)

    @staticmethod
    def _pseudospectral_rhs(u, v, D, L):
        # u: [batch_size, Nx]
        batch_size, Nx = u.shape
        k = 2 * np.pi * np.fft.fftfreq(Nx, d=L/Nx)  # [Nx]
        ik = 1j * k
        k2 = k ** 2
        dudt = np.zeros_like(u, dtype=np.float64)
        for b in range(batch_size):
            u_hat = np.fft.fft(u[b])
            ux_hat = ik * u_hat
            uxx_hat = -k2 * u_hat
            ux = np.fft.ifft(ux_hat).real
            uxx = np.fft.ifft(uxx_hat).real
            dudt[b] = -v * ux + D * uxx
        return dudt 