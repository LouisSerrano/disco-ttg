import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from typing import Tuple, Callable
import time


class AdvectionDiffusionSolver:
    """
    Solver for the 2D advection-diffusion equation:
    ∂u/∂t + v_x * ∂u/∂x + v_y * ∂u/∂y = D * (∂²u/∂x² + ∂²u/∂y²)
    
    Supports various operator splitting methods and ground truth FFT solver.
    """
    
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, 
                 vx: float, vy: float, D: float):
        """
        Initialize the solver with domain and parameters.
        
        Args:
            nx, ny: Grid points in x and y
            Lx, Ly: Domain lengths in x and y
            vx, vy: Advection velocities in x and y
            D: Diffusion coefficient
        """
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.vx, self.vy = vx, vy
        self.D = D
        
        # Create spatial grids
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Frequency grids for FFT
        self.kx = 2 * np.pi * fftfreq(nx, self.dx)
        self.ky = 2 * np.pi * fftfreq(ny, self.dy)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky)
        self.K2 = self.Kx**2 + self.Ky**2
    
    def initial_condition_gaussian(self, x0: float = None, y0: float = None, 
                                 sigma: float = 0.1) -> np.ndarray:
        """Create a Gaussian initial condition."""
        if x0 is None:
            x0 = self.Lx / 4
        if y0 is None:
            y0 = self.Ly / 2
            
        return np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))
    
    def fft_ground_truth(self, u0: np.ndarray, dt: float, nt: int) -> np.ndarray:
        """
        Solve the full advection-diffusion equation using spectral methods (FFT).
        This serves as our ground truth solution.
        """
        u_hat = fft2(u0)
        
        # Pre-compute the linear operator in frequency space
        # For ∂u/∂t + vx*∂u/∂x + vy*∂u/∂y = D*(∂²u/∂x² + ∂²u/∂y²)
        # In frequency space: ∂û/∂t + i*vx*kx*û + i*vy*ky*û = -D*k²*û
        # Solution: û(t) = û(0) * exp((-i*(vx*kx + vy*ky) - D*k²)*t)
        linear_operator = -1j * (self.vx * self.Kx + self.vy * self.Ky) - self.D * self.K2
        
        solutions = [u0.copy()]
        
        for n in range(nt):
            # Apply the exact solution operator
            u_hat *= np.exp(linear_operator * dt)
            u = np.real(ifft2(u_hat))
            solutions.append(u.copy())
            
        return solutions
    
    def advection_step_upwind(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation ∂u/∂t + vx*∂u/∂x + vy*∂u/∂y = 0
        using upwind finite differences.
        """
        u_new = u.copy()
        
        # X-direction advection
        if self.vx > 0:
            u_new[1:, :] -= self.vx * dt / self.dx * (u[1:, :] - u[:-1, :])
            u_new[0, :] -= self.vx * dt / self.dx * (u[0, :] - u[-1, :])  # Periodic BC
        else:
            u_new[:-1, :] -= self.vx * dt / self.dx * (u[1:, :] - u[:-1, :])
            u_new[-1, :] -= self.vx * dt / self.dx * (u[0, :] - u[-1, :])  # Periodic BC
            
        # Y-direction advection
        if self.vy > 0:
            u_new[:, 1:] -= self.vy * dt / self.dy * (u[:, 1:] - u[:, :-1])
            u_new[:, 0] -= self.vy * dt / self.dy * (u[:, 0] - u[:, -1])  # Periodic BC
        else:
            u_new[:, :-1] -= self.vy * dt / self.dy * (u[:, 1:] - u[:, :-1])
            u_new[:, -1] -= self.vy * dt / self.dy * (u[:, 0] - u[:, -1])  # Periodic BC
            
        return u_new
    
    def advection_step_spectral(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation using spectral methods.
        More accurate than upwind for smooth solutions.
        """
        u_hat = fft2(u)
        # In frequency space: ∂û/∂t + i*(vx*kx + vy*ky)*û = 0
        advection_operator = -1j * (self.vx * self.Kx + self.vy * self.Ky)
        u_hat *= np.exp(advection_operator * dt)
        return np.real(ifft2(u_hat))
    
    def diffusion_step_spectral(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation ∂u/∂t = D*(∂²u/∂x² + ∂²u/∂y²)
        using spectral methods.
        """
        u_hat = fft2(u)
        # In frequency space: ∂û/∂t = -D*k²*û
        diffusion_operator = -self.D * self.K2
        u_hat *= np.exp(diffusion_operator * dt)
        return np.real(ifft2(u_hat))
    
    def advection_x_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Solve advection in x-direction only: ∂u/∂t + vx*∂u/∂x = 0"""
        u_hat = fft2(u)
        advection_x_operator = -1j * self.vx * self.Kx
        u_hat *= np.exp(advection_x_operator * dt)
        return np.real(ifft2(u_hat))
    
    def advection_y_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Solve advection in y-direction only: ∂u/∂t + vy*∂u/∂y = 0"""
        u_hat = fft2(u)
        advection_y_operator = -1j * self.vy * self.Ky
        u_hat *= np.exp(advection_y_operator * dt)
        return np.real(ifft2(u_hat))
    
    def diffusion_x_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Solve diffusion in x-direction only: ∂u/∂t = D*∂²u/∂x²"""
        u_hat = fft2(u)
        diffusion_x_operator = -self.D * self.Kx**2
        u_hat *= np.exp(diffusion_x_operator * dt)
        return np.real(ifft2(u_hat))
    
    def diffusion_y_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Solve diffusion in y-direction only: ∂u/∂t = D*∂²u/∂y²"""
        u_hat = fft2(u)
        diffusion_y_operator = -self.D * self.Ky**2
        u_hat *= np.exp(diffusion_y_operator * dt)
        return np.real(ifft2(u_hat))


class OperatorSplittingMethods:
    """Collection of operator splitting methods."""
    
    def __init__(self, solver: AdvectionDiffusionSolver):
        self.solver = solver
    
    def lie_splitting(self, u0: np.ndarray, dt: float, nt: int, 
                     use_spectral: bool = True) -> list:
        """
        First-order Lie splitting: A(dt) ∘ D(dt)
        Solve advection for full timestep, then diffusion for full timestep.
        """
        u = u0.copy()
        solutions = [u.copy()]
        
        advection_step = (self.solver.advection_step_spectral if use_spectral 
                         else self.solver.advection_step_upwind)
        
        for n in range(nt):
            # Step 1: Solve advection
            u = advection_step(u, dt)
            # Step 2: Solve diffusion
            u = self.solver.diffusion_step_spectral(u, dt)
            solutions.append(u.copy())
            
        return solutions
    
    def strang_splitting(self, u0: np.ndarray, dt: float, nt: int,
                        use_spectral: bool = True) -> list:
        """
        Second-order Strang splitting: D(dt/2) ∘ A(dt) ∘ D(dt/2)
        """
        u = u0.copy()
        solutions = [u.copy()]
        
        advection_step = (self.solver.advection_step_spectral if use_spectral 
                         else self.solver.advection_step_upwind)
        
        for n in range(nt):
            # Step 1: Solve diffusion for half timestep
            u = self.solver.diffusion_step_spectral(u, dt/2)
            # Step 2: Solve advection for full timestep
            u = advection_step(u, dt)
            # Step 3: Solve diffusion for half timestep
            u = self.solver.diffusion_step_spectral(u, dt/2)
            solutions.append(u.copy())
            
        return solutions
    
    def advection_advection_splitting(self, u0: np.ndarray, dt: float, nt: int) -> list:
        """
        Split advection into x and y components: Ax(dt) ∘ Ay(dt)
        """
        u = u0.copy()
        solutions = [u.copy()]
        
        for n in range(nt):
            # Step 1: Solve advection in x-direction
            u = self.solver.advection_x_step(u, dt)
            # Step 2: Solve advection in y-direction
            u = self.solver.advection_y_step(u, dt)
            solutions.append(u.copy())
            
        return solutions
    
    def diffusion_diffusion_splitting(self, u0: np.ndarray, dt: float, nt: int) -> list:
        """
        Split diffusion into x and y components: Dx(dt) ∘ Dy(dt)
        """
        u = u0.copy()
        solutions = [u.copy()]
        
        for n in range(nt):
            # Step 1: Solve diffusion in x-direction
            u = self.solver.diffusion_x_step(u, dt)
            # Step 2: Solve diffusion in y-direction
            u = self.solver.diffusion_y_step(u, dt)
            solutions.append(u.copy())
            
        return solutions


def compute_error_metrics(solution: np.ndarray, reference: np.ndarray) -> dict:
    """Compute various error metrics between solution and reference."""
    error = solution - reference
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(np.abs(error))
    relative_l2 = l2_error / np.sqrt(np.mean(reference**2))
    
    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'relative_l2': relative_l2
    }


if __name__ == "__main__":
    # Example usage and basic test
    print("Operator Splitting Investigation for Advection-Diffusion")
    print("=" * 60)
    
    # Parameters
    nx, ny = 64, 64
    Lx, Ly = 2*np.pi, 2*np.pi
    vx, vy = 1.0, 0.5
    D = 0.01
    dt = 0.01
    nt = 50
    
    # Initialize solver
    solver = AdvectionDiffusionSolver(nx, ny, Lx, Ly, vx, vy, D)
    splitting = OperatorSplittingMethods(solver)
    
    # Initial condition
    u0 = solver.initial_condition_gaussian()
    
    print(f"Domain: {Lx} x {Ly}, Grid: {nx} x {ny}")
    print(f"Velocities: vx={vx}, vy={vy}")
    print(f"Diffusion: D={D}")
    print(f"Time step: dt={dt}, Steps: nt={nt}")
    print()
    
    # Compute solutions
    print("Computing solutions...")
    start_time = time.time()
    ground_truth = solver.fft_ground_truth(u0, dt, nt)
    gt_time = time.time() - start_time
    
    start_time = time.time()
    lie_solution = splitting.lie_splitting(u0, dt, nt)
    lie_time = time.time() - start_time
    
    start_time = time.time()
    strang_solution = splitting.strang_splitting(u0, dt, nt)
    strang_time = time.time() - start_time
    
    # Error analysis at final time
    final_gt = ground_truth[-1]
    final_lie = lie_solution[-1]
    final_strang = strang_solution[-1]
    
    lie_errors = compute_error_metrics(final_lie, final_gt)
    strang_errors = compute_error_metrics(final_strang, final_gt)
    
    print("Results:")
    print(f"Ground truth time: {gt_time:.4f}s")
    print(f"Lie splitting time: {lie_time:.4f}s")
    print(f"Strang splitting time: {strang_time:.4f}s")
    print()
    print("Final time errors:")
    print(f"Lie splitting - L2: {lie_errors['l2_error']:.6e}, L∞: {lie_errors['linf_error']:.6e}")
    print(f"Strang splitting - L2: {strang_errors['l2_error']:.6e}, L∞: {strang_errors['linf_error']:.6e}")