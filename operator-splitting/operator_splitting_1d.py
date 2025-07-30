import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from typing import List, Dict, Callable
import time


class AdvectionDiffusion1DSolver:
    """
    Solver for the 1D advection-diffusion equation:
    ∂u/∂t + v * ∂u/∂x = D * ∂²u/∂x²
    
    Supports various operator splitting methods and ground truth FFT solver.
    """
    
    def __init__(self, nx: int, L: float, v: float, D: float):
        """
        Initialize the solver with domain and parameters.
        
        Args:
            nx: Number of grid points
            L: Domain length
            v: Advection velocity
            D: Diffusion coefficient
        """
        self.nx = nx
        self.L = L
        self.v = v
        self.D = D
        
        # Create spatial grid
        self.dx = L / nx
        self.x = np.linspace(0, L, nx, endpoint=False)
        
        # Frequency grid for FFT
        self.k = 2 * np.pi * fftfreq(nx, self.dx)
        self.k2 = self.k**2
    
    def initial_condition_gaussian(self, x0: float = None, sigma: float = 0.1) -> np.ndarray:
        """Create a Gaussian initial condition."""
        if x0 is None:
            x0 = self.L / 4
        return np.exp(-(self.x - x0)**2 / (2 * sigma**2))
    
    def initial_condition_sine(self, n_modes: int = 1) -> np.ndarray:
        """Create a sine wave initial condition."""
        return np.sin(2 * np.pi * n_modes * self.x / self.L)
    
    def initial_condition_complex_sines(self, modes: List[int] = None, amplitudes: List[float] = None) -> np.ndarray:
        """Create a complex initial condition as sum of sines with high frequencies."""
        if modes is None:
            modes = [3, 7, 11, 15, 19]  # High frequency modes
        if amplitudes is None:
            amplitudes = [1.0, 0.8, 0.6, 0.4, 0.3]  # Decreasing amplitudes
        
        u = np.zeros_like(self.x)
        for mode, amp in zip(modes, amplitudes):
            u += amp * np.sin(2 * np.pi * mode * self.x / self.L)
        
        return u
    
    def initial_condition_step(self, x0: float = None, width: float = 0.2) -> np.ndarray:
        """Create a step function initial condition."""
        if x0 is None:
            x0 = self.L / 4
        u = np.zeros_like(self.x)
        mask = (self.x >= x0) & (self.x <= x0 + width)
        u[mask] = 1.0
        return u
    
    def fft_ground_truth(self, u0: np.ndarray, dt: float, nt: int) -> List[np.ndarray]:
        """
        Solve the full advection-diffusion equation using spectral methods (FFT).
        This serves as our ground truth solution.
        
        For ∂u/∂t + v*∂u/∂x = D*∂²u/∂x²
        In frequency space: ∂û/∂t + i*v*k*û = -D*k²*û
        Solution: û(t) = û(0) * exp((-i*v*k - D*k²)*t)
        """
        # Handle batched inputs with sequential evaluation
        if u0.ndim == 2:  # Batched input: shape [batch_size, nx]
            batch_size, nx = u0.shape
            batch_solutions = []
            
            for i in range(batch_size):
                # Process each sample sequentially
                single_solutions = self.fft_ground_truth(u0[i], dt, nt)
                batch_solutions.append(single_solutions)
            
            # Reshape to [nt+1, batch_size, nx] format
            results = []
            for t in range(nt + 1):
                time_step_batch = np.array([batch_solutions[i][t] for i in range(batch_size)])
                results.append(time_step_batch)
            
            return results
        
        # Single sample processing (original logic)
        u_hat = fft(u0)
        
        # Debug prints
        #print(f"self.nx: {self.nx}")
        #print(f"self.k shape: {self.k.shape}")
        #print(f"self.k2 shape: {self.k2.shape}")
        #print(f"u_hat shape: {u_hat.shape}")
        
        # Linear operator in frequency space
        linear_operator = -1j * self.v * self.k - self.D * self.k2
        
        solutions = [u0.copy()]
        
        for n in range(nt):
            # Apply the exact solution operator
            u_hat *= np.exp(linear_operator * dt)
            u = np.real(ifft(u_hat))
            solutions.append(u.copy())
            
        return solutions
    
    def advection_step_spectral(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation ∂u/∂t + v*∂u/∂x = 0
        using spectral methods.
        """
        u_hat = fft(u)
        # In frequency space: ∂û/∂t + i*v*k*û = 0
        advection_operator = -1j * self.v * self.k
        u_hat *= np.exp(advection_operator * dt)
        return np.real(ifft(u_hat))
    
    def advection_step_upwind(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation using upwind finite differences.
        """
        u_new = u.copy()
        
        if self.v > 0:
            # Forward difference
            u_new[1:] -= self.v * dt / self.dx * (u[1:] - u[:-1])
            u_new[0] -= self.v * dt / self.dx * (u[0] - u[-1])  # Periodic BC
        else:
            # Backward difference
            u_new[:-1] -= self.v * dt / self.dx * (u[1:] - u[:-1])
            u_new[-1] -= self.v * dt / self.dx * (u[0] - u[-1])  # Periodic BC
            
        return u_new
    
    def advection_step_centered(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation using second-order centered differences in space.
        Uses RK2 (midpoint method) for second-order accuracy in time.
        
        For ∂u/∂t + v*∂u/∂x = 0
        Spatial discretization: ∂u/∂x ≈ (u[i+1] - u[i-1])/(2*dx)
        Time discretization: RK2 midpoint method
        """
        # RK2 (midpoint method) for time stepping
        # k1 = f(u_n)
        k1 = self._advection_rhs_centered(u)
        
        # k2 = f(u_n + dt/2 * k1)
        u_mid = u + (dt/2) * k1
        k2 = self._advection_rhs_centered(u_mid)
        
        # u_{n+1} = u_n + dt * k2
        return u + dt * k2
    
    def _advection_rhs_centered(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the advection equation using centered differences.
        Returns -v * ∂u/∂x
        """
        dudx = np.zeros_like(u)
        
        # Interior points: centered difference
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * self.dx)
        
        # Periodic boundary conditions
        dudx[0] = (u[1] - u[-1]) / (2 * self.dx)
        dudx[-1] = (u[0] - u[-2]) / (2 * self.dx)
        
        return -self.v * dudx
    
    def advection_step_lax_wendroff(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve advection equation using Lax-Wendroff scheme (second-order in space and time).
        
        For ∂u/∂t + v*∂u/∂x = 0
        Lax-Wendroff: u[i]^{n+1} = u[i]^n - (v*dt)/(2*dx) * (u[i+1]^n - u[i-1]^n) 
                                            + (v*dt)^2/(2*dx^2) * (u[i+1]^n - 2*u[i]^n + u[i-1]^n)
        """
        u_new = u.copy()
        r = self.v * dt / self.dx  # CFL number
        
        # Interior points
        u_new[1:-1] = (u[1:-1] 
                       - 0.5 * r * (u[2:] - u[:-2])
                       + 0.5 * r**2 * (u[2:] - 2*u[1:-1] + u[:-2]))
        
        # Periodic boundary conditions
        u_new[0] = (u[0] 
                    - 0.5 * r * (u[1] - u[-1])
                    + 0.5 * r**2 * (u[1] - 2*u[0] + u[-1]))
        
        u_new[-1] = (u[-1] 
                     - 0.5 * r * (u[0] - u[-2])
                     + 0.5 * r**2 * (u[0] - 2*u[-1] + u[-2]))
        
        return u_new
    
    def diffusion_step_spectral(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation ∂u/∂t = D*∂²u/∂x²
        using spectral methods.
        """
        u_hat = fft(u)
        # In frequency space: ∂û/∂t = -D*k²*û
        diffusion_operator = -self.D * self.k2
        u_hat *= np.exp(diffusion_operator * dt)
        return np.real(ifft(u_hat))
    
    def diffusion_step_finite_diff(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation using finite differences (Crank-Nicolson).
        """
        alpha = self.D * dt / (2 * self.dx**2)
        
        # Build tridiagonal matrix for implicit step
        # (I - α*L) u^{n+1} = (I + α*L) u^n
        # where L is the discrete Laplacian
        
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nx))
        
        # Main diagonal
        A[range(self.nx), range(self.nx)] = 1 + 2*alpha
        B[range(self.nx), range(self.nx)] = 1 - 2*alpha
        
        # Off-diagonals (with periodic boundary conditions)
        for i in range(self.nx):
            A[i, (i-1) % self.nx] = -alpha
            A[i, (i+1) % self.nx] = -alpha
            B[i, (i-1) % self.nx] = alpha
            B[i, (i+1) % self.nx] = alpha
        
        rhs = B @ u
        return np.linalg.solve(A, rhs)
    
    def diffusion_step_explicit(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation using explicit second-order finite differences.
        Uses forward Euler in time and centered differences in space.
        
        For ∂u/∂t = D*∂²u/∂x²
        Explicit scheme: u[i]^{n+1} = u[i]^n + D*dt/dx² * (u[i+1]^n - 2*u[i]^n + u[i-1]^n)
        
        WARNING: This scheme has stability constraint dt ≤ dx²/(2*D)
        """
        u_new = u.copy()
        alpha = self.D * dt / (self.dx**2)
        
        # Check stability condition
        if alpha > 0.5:
            print(f"WARNING: Stability condition violated! α = {alpha:.3f} > 0.5")
            print(f"Consider reducing dt below {0.5 * self.dx**2 / self.D:.6f}")
        
        # Interior points
        u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2*u[1:-1] + u[:-2])
        
        # Periodic boundary conditions
        u_new[0] = u[0] + alpha * (u[1] - 2*u[0] + u[-1])
        u_new[-1] = u[-1] + alpha * (u[0] - 2*u[-1] + u[-2])
        
        return u_new
    
    def diffusion_step_rk2(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation using RK2 (midpoint method) in time with 
        second-order centered differences in space.
        
        For ∂u/∂t = D*∂²u/∂x²
        More stable than forward Euler, still explicit.
        """
        # RK2 (midpoint method)
        # k1 = f(u_n)
        k1 = self._diffusion_rhs(u)
        
        # k2 = f(u_n + dt/2 * k1)
        u_mid = u + (dt/2) * k1
        k2 = self._diffusion_rhs(u_mid)
        
        # u_{n+1} = u_n + dt * k2
        return u + dt * k2
    
    def _diffusion_rhs(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the diffusion equation using centered differences.
        Returns D * ∂²u/∂x²
        """
        d2udx2 = np.zeros_like(u)
        
        # Interior points: centered second difference
        d2udx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (self.dx**2)
        
        # Periodic boundary conditions
        d2udx2[0] = (u[1] - 2*u[0] + u[-1]) / (self.dx**2)
        d2udx2[-1] = (u[0] - 2*u[-1] + u[-2]) / (self.dx**2)
        
        return self.D * d2udx2
    
    def diffusion_step_backward_euler(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation using backward Euler (fully implicit).
        Unconditionally stable but only first-order accurate in time.
        
        For ∂u/∂t = D*∂²u/∂x²
        Backward Euler: (I - dt*D*L) u^{n+1} = u^n
        where L is the discrete Laplacian operator
        """
        alpha = self.D * dt / (self.dx**2)
        
        # Build matrix (I - dt*D*L)
        A = np.zeros((self.nx, self.nx))
        
        # Main diagonal
        A[range(self.nx), range(self.nx)] = 1 + 2*alpha
        
        # Off-diagonals (with periodic boundary conditions)
        for i in range(self.nx):
            A[i, (i-1) % self.nx] = -alpha
            A[i, (i+1) % self.nx] = -alpha
        
        return np.linalg.solve(A, u)


class OperatorSplitting1DMethods:
    """Collection of 1D operator splitting methods."""
    
    def __init__(self, solver: AdvectionDiffusion1DSolver):
        self.solver = solver
    
    def lie_splitting(self, u0: np.ndarray, dt: float, nt: int, 
                     discretization: str = "centered") -> List[np.ndarray]:
        """
        First-order Lie splitting: A(dt) ∘ D(dt)
        Solve advection for full timestep, then diffusion for full timestep.
        
        Args:
            discretization: "spectral", "upwind", "centered", "lax-wendroff", "explicit", "rk2", "backward-euler"
        """
        # Handle batched inputs with sequential evaluation
        if u0.ndim == 2:  # Batched input: shape [batch_size, nx]
            batch_size, nx = u0.shape
            batch_solutions = []
            
            for i in range(batch_size):
                # Process each sample sequentially
                single_solutions = self.lie_splitting(u0[i], dt, nt, discretization)
                batch_solutions.append(single_solutions)
            
            # Reshape to [nt+1, batch_size, nx] format
            results = []
            for t in range(nt + 1):
                time_step_batch = np.array([batch_solutions[i][t] for i in range(batch_size)])
                results.append(time_step_batch)
            
            return results
        
        # Single sample processing (original logic)
        u = u0.copy()
        solutions = [u.copy()]
        
        # Select advection method
        if discretization == "spectral":
            advection_step = self.solver.advection_step_spectral
        elif discretization in ["upwind", "explicit", "backward-euler"]:
            advection_step = self.solver.advection_step_upwind
        elif discretization == "centered":
            advection_step = self.solver.advection_step_centered
        elif discretization == "lax-wendroff":
            advection_step = self.solver.advection_step_lax_wendroff
        elif discretization == "rk2":
            advection_step = self.solver.advection_step_centered  # RK2 with centered differences
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        # Select diffusion method
        if discretization == "spectral":
            diffusion_step = self.solver.diffusion_step_spectral
        elif discretization in ["upwind", "centered", "lax-wendroff"]:
            diffusion_step = self.solver.diffusion_step_finite_diff  # Crank-Nicolson
        elif discretization == "explicit":
            diffusion_step = self.solver.diffusion_step_explicit
        elif discretization == "rk2":
            diffusion_step = self.solver.diffusion_step_rk2
        elif discretization == "backward-euler":
            diffusion_step = self.solver.diffusion_step_backward_euler
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        for n in range(nt):
            # Step 1: Solve advection
            u = advection_step(u, dt)
            # Step 2: Solve diffusion
            u = diffusion_step(u, dt)
            solutions.append(u.copy())
            
        return solutions
    
    def strang_splitting(self, u0: np.ndarray, dt: float, nt: int,
                        discretization: str = "centered") -> List[np.ndarray]:
        """
        Second-order Strang splitting: D(dt/2) ∘ A(dt) ∘ D(dt/2)
        
        Args:
            discretization: "spectral", "upwind", "centered", "lax-wendroff", "explicit", "rk2", "backward-euler"
        """
        # Handle batched inputs with sequential evaluation
        if u0.ndim == 2:  # Batched input: shape [batch_size, nx]
            batch_size, nx = u0.shape
            batch_solutions = []
            
            for i in range(batch_size):
                # Process each sample sequentially
                single_solutions = self.strang_splitting(u0[i], dt, nt, discretization)
                batch_solutions.append(single_solutions)
            
            # Reshape to [nt+1, batch_size, nx] format
            results = []
            for t in range(nt + 1):
                time_step_batch = np.array([batch_solutions[i][t] for i in range(batch_size)])
                results.append(time_step_batch)
            
            return results
        
        # Single sample processing (original logic)
        u = u0.copy()
        solutions = [u.copy()]
        
        # Select advection method
        if discretization == "spectral":
            advection_step = self.solver.advection_step_spectral
        elif discretization in ["upwind", "explicit", "backward-euler"]:
            advection_step = self.solver.advection_step_upwind
        elif discretization == "centered":
            advection_step = self.solver.advection_step_centered
        elif discretization == "lax-wendroff":
            advection_step = self.solver.advection_step_lax_wendroff
        elif discretization == "rk2":
            advection_step = self.solver.advection_step_centered
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        # Select diffusion method
        if discretization == "spectral":
            diffusion_step = self.solver.diffusion_step_spectral
        elif discretization in ["upwind", "centered", "lax-wendroff"]:
            diffusion_step = self.solver.diffusion_step_finite_diff  # Crank-Nicolson
        elif discretization == "explicit":
            diffusion_step = self.solver.diffusion_step_explicit
        elif discretization == "rk2":
            diffusion_step = self.solver.diffusion_step_rk2
        elif discretization == "backward-euler":
            diffusion_step = self.solver.diffusion_step_backward_euler
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        for n in range(nt):
            # Step 1: Solve diffusion for half timestep
            u = diffusion_step(u, dt/2)
            # Step 2: Solve advection for full timestep
            u = advection_step(u, dt)
            # Step 3: Solve diffusion for half timestep
            u = diffusion_step(u, dt/2)
            solutions.append(u.copy())
            
        return solutions
    
    def alternating_splitting(self, u0: np.ndarray, dt: float, nt: int,
                             discretization: str = "centered") -> List[np.ndarray]:
        """
        Alternating splitting: A(dt/2) ∘ D(dt) ∘ A(dt/2)
        
        Args:
            discretization: "spectral", "upwind", "centered", "lax-wendroff", "explicit", "rk2", "backward-euler"
        """
        u = u0.copy()
        solutions = [u.copy()]
        
        # Select advection method
        if discretization == "spectral":
            advection_step = self.solver.advection_step_spectral
        elif discretization in ["upwind", "explicit", "backward-euler"]:
            advection_step = self.solver.advection_step_upwind
        elif discretization == "centered":
            advection_step = self.solver.advection_step_centered
        elif discretization == "lax-wendroff":
            advection_step = self.solver.advection_step_lax_wendroff
        elif discretization == "rk2":
            advection_step = self.solver.advection_step_centered
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        # Select diffusion method
        if discretization == "spectral":
            diffusion_step = self.solver.diffusion_step_spectral
        elif discretization in ["upwind", "centered", "lax-wendroff"]:
            diffusion_step = self.solver.diffusion_step_finite_diff  # Crank-Nicolson
        elif discretization == "explicit":
            diffusion_step = self.solver.diffusion_step_explicit
        elif discretization == "rk2":
            diffusion_step = self.solver.diffusion_step_rk2
        elif discretization == "backward-euler":
            diffusion_step = self.solver.diffusion_step_backward_euler
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
        
        for n in range(nt):
            # Step 1: Solve advection for half timestep
            u = advection_step(u, dt/2)
            # Step 2: Solve diffusion for full timestep
            u = diffusion_step(u, dt)
            # Step 3: Solve advection for half timestep
            u = advection_step(u, dt/2)
            solutions.append(u.copy())
            
        return solutions


def compute_error_metrics_1d(solution: np.ndarray, reference: np.ndarray) -> Dict:
    """Compute various error metrics between solution and reference."""
    error = solution - reference
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(np.abs(error))
    relative_l2 = l2_error / (np.sqrt(np.mean(reference**2)) + 1e-16)
    
    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'relative_l2': relative_l2
    }


def analyze_1d_convergence(nx: int = 128, L: float = 2*np.pi, v: float = 1.0, D: float = 0.01,
                          T: float = 0.5, dt_values: List[float] = None, save_predictions: bool = True,
                          output_dir: str = "convergence_predictions") -> Dict:
    """Study convergence of 1D splitting methods with respect to time step."""
    import os
    if dt_values is None:
        dt_values = [0.02, 0.01, 0.005, 0.0025]
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    
    results = {
        'dt_values': dt_values,
        'lie_errors': [],
        'strang_errors': [],
        'alternating_errors': [],
        'predictions': {}  # Store all predictions for analysis
    }
    
    # Create directory for saving predictions
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    
    print("1D Convergence Study:")
    print("=" * 50)
    
    for i, dt in enumerate(dt_values):
        nt = int(T / dt)
        u0 = solver.initial_condition_complex_sines()
        
        print(f"Case {i+1}/{len(dt_values)}: dt={dt:.4f}, nt={nt}")
        
        # Ground truth
        ground_truth = solver.fft_ground_truth(u0, dt, nt)
        final_gt = ground_truth[-1]
        
        # Splitting methods
        lie_solution = splitting.lie_splitting(u0, dt, nt)
        strang_solution = splitting.strang_splitting(u0, dt, nt)
        alt_solution = splitting.alternating_splitting(u0, dt, nt)
        
        # Store predictions for this dt value
        case_key = f"dt_{dt:.6f}"
        results['predictions'][case_key] = {
            'dt': dt,
            'nt': nt,
            'x': solver.x.copy(),
            'time_points': np.linspace(0, T, nt+1),
            'ground_truth': [u.copy() for u in ground_truth],
            'lie_solution': [u.copy() for u in lie_solution],
            'strang_solution': [u.copy() for u in strang_solution],
            'alternating_solution': [u.copy() for u in alt_solution],
            'initial_condition': u0.copy()
        }
        
        # Save predictions to files if requested
        if save_predictions:
            case_dir = os.path.join(output_dir, f"case_dt_{dt:.6f}")
            os.makedirs(case_dir, exist_ok=True)
            
            # Save as numpy arrays for easy loading
            np.save(os.path.join(case_dir, "x_grid.npy"), solver.x)
            np.save(os.path.join(case_dir, "time_points.npy"), np.linspace(0, T, nt+1))
            np.save(os.path.join(case_dir, "initial_condition.npy"), u0)
            np.save(os.path.join(case_dir, "ground_truth.npy"), np.array(ground_truth))
            np.save(os.path.join(case_dir, "lie_solution.npy"), np.array(lie_solution))
            np.save(os.path.join(case_dir, "strang_solution.npy"), np.array(strang_solution))
            np.save(os.path.join(case_dir, "alternating_solution.npy"), np.array(alt_solution))
            
            # Save metadata
            metadata = {
                'dt': dt,
                'nt': nt,
                'T': T,
                'nx': nx,
                'L': L,
                'v': v,
                'D': D,
                'Pe': v * L / D
            }
            import json
            with open(os.path.join(case_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create and save plot for this dt case
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot solutions at different times
            time_indices = [0, nt//4, nt//2, nt]
            colors = ['blue', 'green', 'orange', 'red']
            
            for j, t_idx in enumerate(time_indices):
                if t_idx < len(ground_truth):
                    axes[0,0].plot(solver.x, ground_truth[t_idx], colors[j], 
                                  label=f't={t_idx*dt:.2f}', alpha=0.8)
            axes[0,0].set_title('Ground Truth Evolution')
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('u')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Final time comparison
            axes[0,1].plot(solver.x, ground_truth[-1], 'k-', label='Ground Truth', linewidth=2)
            axes[0,1].plot(solver.x, lie_solution[-1], 'r--', label='Lie Splitting', linewidth=2)
            axes[0,1].plot(solver.x, strang_solution[-1], 'b:', label='Strang Splitting', linewidth=2)
            axes[0,1].plot(solver.x, alt_solution[-1], 'g-.', label='Alternating Splitting', linewidth=2)
            axes[0,1].set_title(f'Final Time Comparison (t={T})')
            axes[0,1].set_xlabel('x')
            axes[0,1].set_ylabel('u')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Error plots
            axes[1,0].plot(solver.x, lie_solution[-1] - ground_truth[-1], 'r-', label='Lie Error', linewidth=2)
            axes[1,0].plot(solver.x, strang_solution[-1] - ground_truth[-1], 'b-', label='Strang Error', linewidth=2)
            axes[1,0].plot(solver.x, alt_solution[-1] - ground_truth[-1], 'g-', label='Alternating Error', linewidth=2)
            axes[1,0].set_title('Spatial Error Distribution')
            axes[1,0].set_xlabel('x')
            axes[1,0].set_ylabel('Error')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Time evolution of L2 error with instability detection
            time_points = np.linspace(0, T, nt+1)
            lie_errors_time = [compute_error_metrics_1d(lie_solution[j], ground_truth[j])['l2_error'] 
                              for j in range(len(ground_truth))]
            strang_errors_time = [compute_error_metrics_1d(strang_solution[j], ground_truth[j])['l2_error'] 
                                 for j in range(len(ground_truth))]
            alt_errors_time = [compute_error_metrics_1d(alt_solution[j], ground_truth[j])['l2_error'] 
                              for j in range(len(ground_truth))]
            
            # Detect instability (rapid error growth)
            instability_threshold = 1e-1  # Flag cases with errors above this
            lie_unstable = any(err > instability_threshold for err in lie_errors_time)
            strang_unstable = any(err > instability_threshold for err in strang_errors_time)
            alt_unstable = any(err > instability_threshold for err in alt_errors_time)
            
            # Plot with color coding for stability
            lie_color = 'red' if lie_unstable else 'darkred'
            strang_color = 'red' if strang_unstable else 'blue'
            alt_color = 'red' if alt_unstable else 'green'
            
            axes[1,1].semilogy(time_points, lie_errors_time, 'o-', color=lie_color, 
                              label=f'Lie{" (UNSTABLE)" if lie_unstable else ""}', markersize=4)
            axes[1,1].semilogy(time_points, strang_errors_time, 's-', color=strang_color, 
                              label=f'Strang{" (UNSTABLE)" if strang_unstable else ""}', markersize=4)
            axes[1,1].semilogy(time_points, alt_errors_time, '^-', color=alt_color, 
                              label=f'Alternating{" (UNSTABLE)" if alt_unstable else ""}', markersize=4)
            
            # Add instability threshold line
            axes[1,1].axhline(y=instability_threshold, color='red', linestyle='--', alpha=0.7, 
                             label=f'Instability Threshold ({instability_threshold})')
            
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('L2 Error')
            axes[1,1].set_title('Error Evolution & Instability Detection')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            plt.suptitle(f'Convergence Analysis: dt={dt:.6f}, v={v}, D={D}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'convergence_dt_{dt:.6f}_v{v}_D{D}.png'
            plot_filepath = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Predictions saved to: {case_dir}")
            print(f"  Plot saved to: {plot_filepath}")
        
        # Compute errors
        lie_error = compute_error_metrics_1d(lie_solution[-1], final_gt)
        strang_error = compute_error_metrics_1d(strang_solution[-1], final_gt)
        alt_error = compute_error_metrics_1d(alt_solution[-1], final_gt)
        
        results['lie_errors'].append(lie_error['l2_error'])
        results['strang_errors'].append(strang_error['l2_error'])
        results['alternating_errors'].append(alt_error['l2_error'])
        
        print(f"  Lie={lie_error['l2_error']:.2e}, "
              f"Strang={strang_error['l2_error']:.2e}, "
              f"Alt={alt_error['l2_error']:.2e}")
    
    # Create instability detection dashboard
    print("\n" + "="*50)
    print("CREATING INSTABILITY DETECTION DASHBOARD...")
    dashboard_results = create_instability_detection_dashboard(results, output_dir)
    results['instability_analysis'] = dashboard_results
    
    return results


def create_instability_detection_dashboard(results: Dict, output_dir: str = "convergence_predictions"):
    """
    Create comprehensive dashboard plots for rapid instability detection.
    Replaces CSV-based analysis with visual tools for quick assessment.
    """
    import os
    
    dashboard_dir = os.path.join(output_dir, "instability_dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    dt_values = results['dt_values']
    predictions = results['predictions']
    
    # 1. Create overview thumbnail grid for all cases
    n_cases = len(dt_values)
    cols = min(4, n_cases)
    rows = (n_cases + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_cases > 1 else [axes]
    
    instability_detected = []
    
    for i, dt in enumerate(dt_values):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        case_key = f"dt_{dt:.6f}"
        case_data = predictions[case_key]
        
        # Plot final solutions
        x = case_data['x']
        ground_truth_final = case_data['ground_truth'][-1]
        strang_final = case_data['strang_solution'][-1]
        
        # Calculate max error for color coding
        max_error = np.max(np.abs(strang_final - ground_truth_final))
        instability_threshold = 1e-1
        is_unstable = max_error > instability_threshold
        instability_detected.append(is_unstable)
        
        # Color-coded plot based on stability
        color = 'red' if is_unstable else 'blue'
        ax.plot(x, ground_truth_final, 'k-', alpha=0.7, linewidth=1, label='Truth')
        ax.plot(x, strang_final, color=color, linewidth=2, 
                label=f'Strang ({"UNSTABLE" if is_unstable else "Stable"})')
        
        ax.set_title(f'dt={dt:.4f}\nMax Error: {max_error:.1e}', 
                    color='red' if is_unstable else 'black', fontweight='bold' if is_unstable else 'normal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_cases, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('INSTABILITY DETECTION OVERVIEW\n(Red = Unstable, Blue = Stable)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    overview_path = os.path.join(dashboard_dir, 'instability_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Create error growth rate analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convergence plot with stability indicators
    stable_dt = []
    stable_errors = []
    unstable_dt = []
    unstable_errors = []
    
    for i, dt in enumerate(dt_values):
        error = results['strang_errors'][i]
        if instability_detected[i]:
            unstable_dt.append(dt)
            unstable_errors.append(error)
        else:
            stable_dt.append(dt)
            stable_errors.append(error)
    
    if stable_dt:
        ax1.loglog(stable_dt, stable_errors, 'bo-', label='Stable Cases', markersize=8)
    if unstable_dt:
        ax1.loglog(unstable_dt, unstable_errors, 'ro-', label='UNSTABLE Cases', markersize=10, markerfacecolor='red')
    
    # Add theoretical convergence lines
    if len(dt_values) > 1 and stable_dt:
        dt_range = np.array([min(dt_values), max(dt_values)])
        # First order convergence reference
        first_order = stable_errors[0] * (dt_range / stable_dt[0])
        ax1.loglog(dt_range, first_order, 'k--', alpha=0.5, label='1st Order Reference')
        # Second order convergence reference  
        second_order = first_order * (dt_range / dt_range[0])
        ax1.loglog(dt_range, second_order, 'k:', alpha=0.5, label='2nd Order Reference')
    
    ax1.set_xlabel('Time Step (dt)')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Convergence Analysis with Stability Detection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error evolution for problematic cases
    for i, dt in enumerate(dt_values):
        if instability_detected[i]:
            case_key = f"dt_{dt:.6f}"
            case_data = predictions[case_key] 
            time_points = case_data['time_points']
            
            # Calculate error evolution
            errors_time = []
            for j in range(len(case_data['ground_truth'])):
                error = compute_error_metrics_1d(case_data['strang_solution'][j], 
                                               case_data['ground_truth'][j])['l2_error']
                errors_time.append(error)
            
            ax2.semilogy(time_points, errors_time, 'r-', alpha=0.7, linewidth=2,
                        label=f'dt={dt:.4f} (UNSTABLE)')
    
    if any(instability_detected):
        ax2.axhline(y=instability_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Instability Threshold ({instability_threshold})')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('L2 Error')
        ax2.set_title('Error Evolution for Unstable Cases')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No unstable cases detected', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, color='green')
        ax2.set_title('Error Evolution for Unstable Cases')
    
    plt.tight_layout()
    
    convergence_path = os.path.join(dashboard_dir, 'convergence_stability_analysis.png')
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Create summary report (visual instead of CSV)
    unstable_count = sum(instability_detected)
    stable_count = len(dt_values) - unstable_count
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Title
    title_text = "INSTABILITY DETECTION SUMMARY REPORT"
    ax.text(0.5, 0.95, title_text, fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Summary statistics
    summary_lines = [
        f"Total test cases: {len(dt_values)}",
        f"Stable cases: {stable_count} ({100*stable_count/len(dt_values):.1f}%)",
        f"Unstable cases: {unstable_count} ({100*unstable_count/len(dt_values):.1f}%)",
        "",
        "STABLE CASES (✓):",
    ]
    
    for i, dt in enumerate(dt_values):
        if not instability_detected[i]:
            error = results['strang_errors'][i]
            summary_lines.append(f"  • dt = {dt:.6f}, L2 error = {error:.2e}")
    
    if unstable_count > 0:
        summary_lines.extend(["", "UNSTABLE CASES (⚠):", ""])
        for i, dt in enumerate(dt_values):
            if instability_detected[i]:
                error = results['strang_errors'][i]
                summary_lines.append(f"  • dt = {dt:.6f}, L2 error = {error:.2e} (CRITICAL)")
    
    stable_dt_vals = [dt for i, dt in enumerate(dt_values) if not instability_detected[i]]
    min_stable_dt = min(stable_dt_vals) if stable_dt_vals else min(dt_values)
    
    summary_lines.extend([
        "",
        "RECOMMENDATIONS:",
        f"• Use time step ≤ {min_stable_dt:.6f} for stable results",
        f"• Avoid time steps that exceed {instability_threshold:.1e} error threshold",
        "• Monitor error evolution plots for early instability detection"
    ])
    
    # Display text
    text_y = 0.85
    for line in summary_lines:
        if "UNSTABLE" in line or "CRITICAL" in line:
            color = 'red'
            weight = 'bold'
        elif "✓" in line or "Stable" in line:
            color = 'green'
            weight = 'bold'
        elif line.startswith("  •"):
            color = 'red' if "CRITICAL" in line else 'blue'
            weight = 'normal'
        else:
            color = 'black'
            weight = 'bold' if line and not line.startswith("  ") else 'normal'
        
        ax.text(0.05, text_y, line, fontsize=12, fontweight=weight, color=color, 
               transform=ax.transAxes, family='monospace')
        text_y -= 0.04
    
    # Add border
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor='black', 
                              linewidth=2, transform=ax.transAxes))
    
    summary_path = os.path.join(dashboard_dir, 'instability_summary_report.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎯 INSTABILITY DETECTION DASHBOARD CREATED:")
    print(f"├── Overview: {overview_path}")
    print(f"├── Analysis: {convergence_path}")  
    print(f"└── Summary: {summary_path}")
    print(f"\n📊 Quick Assessment: {unstable_count}/{len(dt_values)} cases unstable")
    
    return {
        'instability_detected': instability_detected,
        'stable_cases': stable_count,
        'unstable_cases': unstable_count,
        'dashboard_dir': dashboard_dir
    }


def compare_initial_conditions(nx: int = 128, L: float = 2*np.pi, v: float = 1.0, D: float = 0.01,
                              dt: float = 0.01, T: float = 0.5):
    """Compare splitting methods for different initial conditions."""
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    nt = int(T / dt)
    
    initial_conditions = {
        'Gaussian': solver.initial_condition_gaussian(),
        'Sine': solver.initial_condition_sine(n_modes=2),
        'Step': solver.initial_condition_step()
    }
    
    print("\n1D Initial Condition Comparison:")
    print("=" * 50)
    
    for ic_name, u0 in initial_conditions.items():
        print(f"\n{ic_name} Initial Condition:")
        
        # Ground truth
        ground_truth = solver.fft_ground_truth(u0, dt, nt)
        final_gt = ground_truth[-1]
        
        # Splitting methods
        methods = {
            'Lie': splitting.lie_splitting(u0, dt, nt),
            'Strang': splitting.strang_splitting(u0, dt, nt),
            'Alternating': splitting.alternating_splitting(u0, dt, nt)
        }
        
        for method_name, solution in methods.items():
            error = compute_error_metrics_1d(solution[-1], final_gt)
            print(f"  {method_name:12}: L2={error['l2_error']:.2e}, L∞={error['linf_error']:.2e}")


def compare_complex_initial_condition(nx: int = 512, L: float = 2*np.pi, v: float = 1.0, D: float = 0.01,
                                     dt: float = 0.005, T: float = 10.0):
    """Compare splitting methods for complex initial condition with high frequencies."""
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    nt = int(T / dt)
    
    print("\nComplex High-Frequency Initial Condition Analysis:")
    print("=" * 50)
    
    # Complex initial condition with multiple high frequency modes
    u0 = solver.initial_condition_complex_sines()
    
    print(f"Initial condition: Sum of sines with modes [3, 7, 11, 15, 19]")
    print(f"Spatial resolution: nx={nx}, Temporal resolution: dt={dt}")
    print(f"Integration time: T={T}")
    
    # Ground truth
    print("Computing ground truth solution...")
    ground_truth = solver.fft_ground_truth(u0, dt, nt)
    final_gt = ground_truth[-1]
    
    # Splitting methods
    print("Computing operator splitting solutions...")
    methods = {
        'Lie': splitting.lie_splitting(u0, dt, nt),
        'Strang': splitting.strang_splitting(u0, dt, nt),
        'Alternating': splitting.alternating_splitting(u0, dt, nt)
    }
    
    print(f"\nFinal time errors (T={T}):")
    for method_name, solution in methods.items():
        error = compute_error_metrics_1d(solution[-1], final_gt)
        print(f"  {method_name:12}: L2={error['l2_error']:.2e}, L∞={error['linf_error']:.2e}, RelL2={error['relative_l2']:.2e}")


def create_1d_visualizations(nx: int = 512, L: float = 2*np.pi, v: float = 1.0, D: float = 0.01,
                            dt: float = 0.005, T: float = 10.0, output_dir: str = "results"):
    """Create 1D visualization plots comparing different methods."""
    import os
    
    # Create organized directory structure
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    nt = int(T / dt)
    
    u0 = solver.initial_condition_complex_sines()
    
    print("\nGenerating 1D visualizations...")
    ground_truth = solver.fft_ground_truth(u0, dt, nt)
    lie_solution = splitting.lie_splitting(u0, dt, nt)
    strang_solution = splitting.strang_splitting(u0, dt, nt)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot solutions at different times
    time_indices = [0, nt//4, nt//2, nt]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, t_idx in enumerate(time_indices):
        if t_idx < len(ground_truth):
            axes[0,0].plot(solver.x, ground_truth[t_idx], colors[i], 
                          label=f't={t_idx*dt:.2f}', alpha=0.8)
    axes[0,0].set_title('Ground Truth (FFT)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('u')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Final time comparison
    axes[0,1].plot(solver.x, ground_truth[-1], 'k-', label='Ground Truth', linewidth=2)
    axes[0,1].plot(solver.x, lie_solution[-1], 'r--', label='Lie Splitting', linewidth=2)
    axes[0,1].plot(solver.x, strang_solution[-1], 'b:', label='Strang Splitting', linewidth=2)
    axes[0,1].set_title(f'Final Time Comparison (t={T})')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('u')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Error plots
    lie_error = np.array([lie_solution[i] - ground_truth[i] for i in range(len(ground_truth))])
    strang_error = np.array([strang_solution[i] - ground_truth[i] for i in range(len(ground_truth))])
    
    axes[1,0].plot(solver.x, lie_error[-1], 'r-', label='Lie Error', linewidth=2)
    axes[1,0].plot(solver.x, strang_error[-1], 'b-', label='Strang Error', linewidth=2)
    axes[1,0].set_title('Spatial Error Distribution')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('Error')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Time evolution of L2 error
    time_points = np.linspace(0, T, nt+1)
    lie_errors_time = [compute_error_metrics_1d(lie_solution[i], ground_truth[i])['l2_error'] 
                      for i in range(len(ground_truth))]
    strang_errors_time = [compute_error_metrics_1d(strang_solution[i], ground_truth[i])['l2_error'] 
                         for i in range(len(ground_truth))]
    
    axes[1,1].semilogy(time_points, lie_errors_time, 'ro-', label='Lie Splitting', markersize=4)
    axes[1,1].semilogy(time_points, strang_errors_time, 'bs-', label='Strang Splitting', markersize=4)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('L2 Error')
    axes[1,1].set_title('Error Evolution')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    # Save with organized naming
    Pe = v * L / D
    viz_filename = f'operator_splitting_comparison_v{v}_D{D}_Pe{Pe:.2f}.png'
    viz_filepath = os.path.join(viz_dir, viz_filename)
    plt.savefig(viz_filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {viz_filepath}")
    
    return viz_filepath


def create_parameter_study_dashboard(df, output_dir):
    """
    Create visual dashboard for parameter study results instead of CSV.
    Provides rapid instability detection across parameter ranges.
    """
    import os
    
    dashboard_dir = os.path.join(output_dir, "parameter_dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Define instability thresholds for different methods
    instability_thresholds = {
        'lie_l2': 1e-1,
        'strang_l2': 1e-1, 
        'alt_l2': 1e-1
    }
    
    # 1. Create overview heatmap of all errors
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['lie_l2', 'strang_l2', 'alt_l2']
    method_names = ['Lie Splitting', 'Strang Splitting', 'Alternating Splitting']
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        # Create color map with instability detection
        errors = df[method].values
        unstable_mask = errors > instability_thresholds[method]
        
        # Plot parameter vs error with color coding
        v_values = df['v'].values
        colors = ['red' if unstable else 'blue' for unstable in unstable_mask]
        
        axes[i].scatter(v_values, errors, c=colors, s=100, alpha=0.7)
        axes[i].axhline(y=instability_thresholds[method], color='red', linestyle='--', 
                       alpha=0.7, label=f'Instability Threshold')
        axes[i].set_xlabel('Advection Velocity (v)')
        axes[i].set_ylabel('L2 Error')
        axes[i].set_title(f'{name}\n(Red = Unstable)')
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add text annotations for unstable cases
        for j, (v, error, is_unstable) in enumerate(zip(v_values, errors, unstable_mask)):
            if is_unstable:
                axes[i].annotate(f'v={v}\nUNSTABLE', (v, error), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='red', weight='bold')
    
    plt.suptitle('PARAMETER STUDY: INSTABILITY DETECTION OVERVIEW', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    overview_path = os.path.join(dashboard_dir, 'parameter_instability_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Create summary table as visual plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Count stable vs unstable cases for each method
    stability_summary = {}
    for method in methods:
        errors = df[method].values
        unstable_count = sum(errors > instability_thresholds[method])
        stable_count = len(errors) - unstable_count
        stability_summary[method] = {'stable': stable_count, 'unstable': unstable_count}
    
    # Title
    title_text = "PARAMETER STUDY STABILITY SUMMARY"
    ax.text(0.5, 0.95, title_text, fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Create table-like display
    summary_lines = [
        f"Total parameter combinations tested: {len(df)}",
        f"Advection velocities: {df['v'].min():.1f} to {df['v'].max():.1f}",
        f"Fixed diffusion coefficient: D = {df['D'].iloc[0]:.3f}",
        "",
        "STABILITY ANALYSIS BY METHOD:",
        ""
    ]
    
    for method, name in zip(methods, method_names):
        stable = stability_summary[method]['stable']
        unstable = stability_summary[method]['unstable']
        stable_pct = 100 * stable / len(df)
        unstable_pct = 100 * unstable / len(df)
        
        summary_lines.extend([
            f"{name}:",
            f"  ✓ Stable cases:   {stable:2d} ({stable_pct:5.1f}%)",
            f"  ⚠ Unstable cases: {unstable:2d} ({unstable_pct:5.1f}%)",
            ""
        ])
    
    # List problematic parameter combinations
    summary_lines.append("CRITICAL PARAMETER COMBINATIONS:")
    any_critical = False
    for _, row in df.iterrows():
        critical_methods = []
        for method in methods:
            if row[method] > instability_thresholds[method]:
                critical_methods.append(method.replace('_l2', '').upper())
        
        if critical_methods:
            any_critical = True
            methods_str = ', '.join(critical_methods)
            summary_lines.append(f"  • v={row['v']:.1f}, Pe={row['Pe']:.1f}: {methods_str} UNSTABLE")
    
    if not any_critical:
        summary_lines.append("  ✓ No critical parameter combinations detected")
    
    summary_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "• Use parameter combinations marked as stable (blue points)",
        "• Avoid high advection velocities that cause instability",
        "• Monitor Peclet number (Pe = vL/D) for stability assessment"
    ])
    
    # Display text
    text_y = 0.85
    for line in summary_lines:
        if "UNSTABLE" in line or "CRITICAL" in line:
            color = 'red'
            weight = 'bold'
        elif "✓" in line or "Stable" in line:
            color = 'green' 
            weight = 'bold'
        elif line.startswith("  •") and "UNSTABLE" in line:
            color = 'red'
            weight = 'normal'
        elif line.endswith(":"):
            color = 'black'
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'
        
        ax.text(0.05, text_y, line, fontsize=11, fontweight=weight, color=color,
               transform=ax.transAxes, family='monospace')
        text_y -= 0.04
    
    # Add border
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor='black',
                              linewidth=2, transform=ax.transAxes))
    
    summary_path = os.path.join(dashboard_dir, 'parameter_stability_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Create Peclet number analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Peclet vs Error for best method (Strang)
    Pe_values = df['Pe'].values
    strang_errors = df['strang_l2'].values
    unstable_strang = strang_errors > instability_thresholds['strang_l2']
    
    stable_Pe = Pe_values[~unstable_strang]
    stable_errors = strang_errors[~unstable_strang]
    unstable_Pe = Pe_values[unstable_strang]
    unstable_errors_pe = strang_errors[unstable_strang]
    
    if len(stable_Pe) > 0:
        ax1.loglog(stable_Pe, stable_errors, 'bo', markersize=8, label='Stable Cases')
    if len(unstable_Pe) > 0:
        ax1.loglog(unstable_Pe, unstable_errors_pe, 'ro', markersize=10, label='UNSTABLE Cases')
    
    ax1.set_xlabel('Peclet Number (Pe = vL/D)')
    ax1.set_ylabel('L2 Error (Strang Method)')
    ax1.set_title('Stability vs Peclet Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Method comparison for worst case
    worst_case_idx = df['v'].idxmax()  # Highest advection velocity
    worst_case = df.iloc[worst_case_idx]
    
    method_errors = [worst_case['lie_l2'], worst_case['strang_l2'], worst_case['alt_l2']]
    colors = ['red' if err > instability_thresholds[methods[i]] else 'blue' 
              for i, err in enumerate(method_errors)]
    
    bars = ax2.bar(method_names, method_errors, color=colors, alpha=0.7)
    ax2.set_ylabel('L2 Error')
    ax2.set_title(f'Method Comparison (Worst Case: v={worst_case["v"]:.1f})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    for i, method in enumerate(methods):
        ax2.axhline(y=instability_thresholds[method], color='red', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, error in zip(bars, method_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.1e}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    peclet_path = os.path.join(dashboard_dir, 'peclet_stability_analysis.png')
    plt.savefig(peclet_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎯 PARAMETER STUDY DASHBOARD CREATED:")
    print(f"├── Overview: {overview_path}")
    print(f"├── Summary: {summary_path}")
    print(f"└── Peclet Analysis: {peclet_path}")
    
    # Print quick assessment
    total_unstable = sum(sum(df[method] > instability_thresholds[method]) for method in methods)
    total_tests = len(df) * len(methods)
    print(f"\n📊 Quick Assessment: {total_unstable}/{total_tests} method-parameter combinations unstable")
    
    return dashboard_dir


def parameter_study_1d(nx: int = 512, L: float = 2*np.pi, T: float = 10.0, dt: float = 0.005,
                      output_dir: str = "results"):
    """
    Parameter study varying advection speed with fixed diffusion coefficient.
    Tests 10 different parameter combinations and saves plots for each case.
    """
    import os
    
    # Create organized directory structure
    parameter_dir = os.path.join(output_dir, "parameter_study")
    cases_dir = os.path.join(parameter_dir, "individual_cases")
    os.makedirs(cases_dir, exist_ok=True)
    
    # Parameter ranges - 10 different advection speeds
    v_values = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    # Fixed diffusion coefficient
    D = 0.01  # Fixed at 0.01
    
    results_summary = []
    case_number = 0
    
    print("=" * 70)
    print("PARAMETER STUDY FOR 1D OPERATOR SPLITTING")
    print("=" * 70)
    print(f"Testing {len(v_values)} different advection velocities")
    print(f"Fixed diffusion coefficient: D = {D}")
    print(f"Spatial resolution: nx={nx}")
    print(f"Temporal resolution: dt={dt}")
    print(f"Integration time: T={T}")
    print("=" * 70)
    
    for i, v in enumerate(v_values):
        case_number += 1
        print(f"\nCase {case_number}/10: v={v}, D={D}")
        
        # Calculate Peclet number (advection vs diffusion strength)
        Pe = v * L / D
        print(f"  Peclet number: Pe = vL/D = {Pe:.1f}")
        
        # Initialize solver with current parameters
        solver = AdvectionDiffusion1DSolver(nx, L, v, D)
        splitting = OperatorSplitting1DMethods(solver)
        nt = int(T / dt)
        
        # Use complex initial condition
        u0 = solver.initial_condition_complex_sines()
        
        # Compute solutions
        print("  Computing solutions...")
        ground_truth = solver.fft_ground_truth(u0, dt, nt)
        lie_solution = splitting.lie_splitting(u0, dt, nt)
        strang_solution = splitting.strang_splitting(u0, dt, nt)
        alt_solution = splitting.alternating_splitting(u0, dt, nt)
        
        # Compute errors
        final_gt = ground_truth[-1]
        lie_error = compute_error_metrics_1d(lie_solution[-1], final_gt)
        strang_error = compute_error_metrics_1d(strang_solution[-1], final_gt)
        alt_error = compute_error_metrics_1d(alt_solution[-1], final_gt)
        
        print(f"  Lie L2 error: {lie_error['l2_error']:.2e}")
        print(f"  Strang L2 error: {strang_error['l2_error']:.2e}")
        print(f"  Alternating L2 error: {alt_error['l2_error']:.2e}")
        
        # Store results
        case_result = {
            'case': case_number,
            'v': v,
            'D': D,
            'Pe': Pe,
            'lie_l2': lie_error['l2_error'],
            'strang_l2': strang_error['l2_error'],
            'alt_l2': alt_error['l2_error'],
            'lie_linf': lie_error['linf_error'],
            'strang_linf': strang_error['linf_error'],
            'alt_linf': alt_error['linf_error']
        }
        results_summary.append(case_result)
        
        # Create visualization for this case
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot initial condition
        axes[0,0].plot(solver.x, u0, 'k-', linewidth=2)
        axes[0,0].set_title('Initial Condition\n(Sum of High-Frequency Sines)')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('u')
        axes[0,0].grid(True)
        
        # Plot final solutions comparison
        axes[0,1].plot(solver.x, ground_truth[-1], 'k-', label='Ground Truth', linewidth=2)
        axes[0,1].plot(solver.x, lie_solution[-1], 'r--', label='Lie', linewidth=2)
        axes[0,1].plot(solver.x, strang_solution[-1], 'b:', label='Strang', linewidth=2)
        axes[0,1].plot(solver.x, alt_solution[-1], 'g-.', label='Alternating', linewidth=2)
        axes[0,1].set_title(f'Final Solutions (T={T})')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('u')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot errors
        axes[0,2].plot(solver.x, lie_solution[-1] - ground_truth[-1], 'r-', label='Lie Error', linewidth=2)
        axes[0,2].plot(solver.x, strang_solution[-1] - ground_truth[-1], 'b-', label='Strang Error', linewidth=2)
        axes[0,2].plot(solver.x, alt_solution[-1] - ground_truth[-1], 'g-', label='Alt Error', linewidth=2)
        axes[0,2].set_title('Spatial Error Distribution')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('Error')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # Time evolution plots (subsample for efficiency)
        time_indices = np.linspace(0, len(ground_truth)-1, 100, dtype=int)
        time_points = np.array([i * dt for i in time_indices])
        
        # Ground truth evolution
        for idx in [0, len(time_indices)//4, len(time_indices)//2, 3*len(time_indices)//4, -1]:
            t_idx = time_indices[idx]
            alpha = 0.3 + 0.7 * idx / (len(time_indices)-1)
            axes[1,0].plot(solver.x, ground_truth[t_idx], alpha=alpha, 
                         label=f't={t_idx*dt:.1f}')
        axes[1,0].set_title('Ground Truth Evolution')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('u')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Error evolution over time
        lie_errors_time = [compute_error_metrics_1d(lie_solution[i], ground_truth[i])['l2_error'] 
                          for i in time_indices]
        strang_errors_time = [compute_error_metrics_1d(strang_solution[i], ground_truth[i])['l2_error'] 
                             for i in time_indices]
        alt_errors_time = [compute_error_metrics_1d(alt_solution[i], ground_truth[i])['l2_error'] 
                          for i in time_indices]
        
        axes[1,1].semilogy(time_points, lie_errors_time, 'ro-', label='Lie', markersize=3)
        axes[1,1].semilogy(time_points, strang_errors_time, 'bs-', label='Strang', markersize=3)
        axes[1,1].semilogy(time_points, alt_errors_time, 'g^-', label='Alternating', markersize=3)
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('L2 Error')
        axes[1,1].set_title('Error Evolution')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Parameter info and summary
        axes[1,2].text(0.05, 0.95, f'Case {case_number}/10', transform=axes[1,2].transAxes, 
                      fontsize=14, fontweight='bold', va='top')
        axes[1,2].text(0.05, 0.85, f'Advection velocity: v = {v}', transform=axes[1,2].transAxes, 
                      fontsize=12, va='top')
        axes[1,2].text(0.05, 0.75, f'Diffusion coefficient: D = {D}', transform=axes[1,2].transAxes, 
                      fontsize=12, va='top')
        axes[1,2].text(0.05, 0.65, f'Peclet number: Pe = {Pe:.1f}', transform=axes[1,2].transAxes, 
                      fontsize=12, va='top')
        
        axes[1,2].text(0.05, 0.5, 'Final L2 Errors:', transform=axes[1,2].transAxes, 
                      fontsize=12, fontweight='bold', va='top')
        axes[1,2].text(0.05, 0.4, f'Lie: {lie_error["l2_error"]:.2e}', transform=axes[1,2].transAxes, 
                      fontsize=11, va='top')
        axes[1,2].text(0.05, 0.3, f'Strang: {strang_error["l2_error"]:.2e}', transform=axes[1,2].transAxes, 
                      fontsize=11, va='top')
        axes[1,2].text(0.05, 0.2, f'Alternating: {alt_error["l2_error"]:.2e}', transform=axes[1,2].transAxes, 
                      fontsize=11, va='top')
        
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        
        plt.suptitle(f'1D Operator Splitting Analysis - Case {case_number}: v={v}, D={D}, Pe={Pe:.1f}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot with descriptive filename
        filename = f'case_{case_number:02d}_v{v}_D{D}_Pe{Pe:.1f}.png'
        filepath = os.path.join(cases_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved: {filepath}")
    
    # Create summary analysis
    create_summary_analysis(results_summary, parameter_dir)
    
    return results_summary


def test_dt_convergence_specific_case(nx: int = 512, L: float = 2*np.pi, v: float = 0.8, D: float = 0.5,
                                     T: float = 1.0, dt_min: float = 0.001, dt_max: float = 0.1, 
                                     n_dt_values: int = 10, output_dir: str = "results", 
                                     save_predictions: bool = True) -> Dict:
    """
    Test convergence behavior for varying dt with specific advection speed and diffusion coefficient.
    
    Args:
        nx: Number of spatial grid points
        L: Domain length
        v: Advection speed (default 0.8)
        D: Diffusion coefficient (default 0.5)
        T: Final time
        dt_min: Minimum time step
        dt_max: Maximum time step
        n_dt_values: Number of dt values to test
        output_dir: Base directory for saving plots
    
    Returns:
        Dictionary containing dt values and corresponding errors
    """
    import os
    
    # Create organized directory structure
    convergence_dir = os.path.join(output_dir, "convergence_analysis")
    os.makedirs(convergence_dir, exist_ok=True)
    # Generate dt values logarithmically spaced
    dt_values = np.logspace(np.log10(dt_min), np.log10(dt_max), n_dt_values)
    dt_values = sorted(dt_values, reverse=True)  # Start with larger dt
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    
    # Calculate Peclet number
    Pe = v * L / D
    
    results = {
        'dt_values': dt_values,
        'lie_errors': [],
        'strang_errors': [],
        'alternating_errors': [],
        'parameters': {'v': v, 'D': D, 'Pe': Pe, 'T': T},
        'predictions': {}  # Store all predictions for analysis
    }
    
    print(f"dt Convergence Study for v={v}, D={D} (Pe={Pe:.2f})")
    print("=" * 60)
    print(f"Testing {n_dt_values} dt values from {dt_max} to {dt_min}")
    print("=" * 60)
    
    # Use complex initial condition for challenging test
    u0 = solver.initial_condition_complex_sines()
    
    # Create prediction directory if saving
    if save_predictions:
        pred_dir = os.path.join(convergence_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
    
    for i, dt in enumerate(dt_values):
        nt = int(T / dt)
        print(f"Case {i+1}/{n_dt_values}: dt={dt:.6f}, nt={nt}")
        
        # Compute ground truth with current dt
        ground_truth = solver.fft_ground_truth(u0, dt, nt)
        final_gt = ground_truth[-1]
        
        # Compute splitting solutions
        lie_solution = splitting.lie_splitting(u0, dt, nt)
        strang_solution = splitting.strang_splitting(u0, dt, nt)
        alt_solution = splitting.alternating_splitting(u0, dt, nt)
        
        # Store predictions for this dt value
        case_key = f"dt_{dt:.6f}"
        results['predictions'][case_key] = {
            'dt': dt,
            'nt': nt,
            'x': solver.x.copy(),
            'time_points': np.linspace(0, T, nt+1),
            'ground_truth': [u.copy() for u in ground_truth],
            'lie_solution': [u.copy() for u in lie_solution],
            'strang_solution': [u.copy() for u in strang_solution],
            'alternating_solution': [u.copy() for u in alt_solution],
            'initial_condition': u0.copy()
        }
        
        # Save predictions to files if requested
        if save_predictions:
            case_dir = os.path.join(pred_dir, f"case_dt_{dt:.6f}")
            os.makedirs(case_dir, exist_ok=True)
            
            # Save as numpy arrays for easy loading
            np.save(os.path.join(case_dir, "x_grid.npy"), solver.x)
            np.save(os.path.join(case_dir, "time_points.npy"), np.linspace(0, T, nt+1))
            np.save(os.path.join(case_dir, "initial_condition.npy"), u0)
            np.save(os.path.join(case_dir, "ground_truth.npy"), np.array(ground_truth))
            np.save(os.path.join(case_dir, "lie_solution.npy"), np.array(lie_solution))
            np.save(os.path.join(case_dir, "strang_solution.npy"), np.array(strang_solution))
            np.save(os.path.join(case_dir, "alternating_solution.npy"), np.array(alt_solution))
            
            # Create comprehensive prediction plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Initial condition
            axes[0,0].plot(solver.x, u0, 'k-', label='Initial Condition', linewidth=2)
            axes[0,0].set_title(f'Initial Condition')
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('u')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Final time comparison
            axes[0,1].plot(solver.x, ground_truth[-1], 'k-', label='Ground Truth', linewidth=2)
            axes[0,1].plot(solver.x, lie_solution[-1], 'r--', label='Lie Splitting', linewidth=2)
            axes[0,1].plot(solver.x, strang_solution[-1], 'b:', label='Strang Splitting', linewidth=2)
            axes[0,1].plot(solver.x, alt_solution[-1], 'g-.', label='Alternating Splitting', linewidth=2)
            axes[0,1].set_title(f'Final Time Comparison (t={T})')
            axes[0,1].set_xlabel('x')
            axes[0,1].set_ylabel('u')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Error plots
            axes[1,0].plot(solver.x, lie_solution[-1] - ground_truth[-1], 'r-', label='Lie Error', linewidth=2)
            axes[1,0].plot(solver.x, strang_solution[-1] - ground_truth[-1], 'b-', label='Strang Error', linewidth=2)
            axes[1,0].plot(solver.x, alt_solution[-1] - ground_truth[-1], 'g-', label='Alternating Error', linewidth=2)
            axes[1,0].set_title('Spatial Error Distribution')
            axes[1,0].set_xlabel('x')
            axes[1,0].set_ylabel('Error')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Time evolution of L2 error with instability detection
            time_points = np.linspace(0, T, nt+1)
            lie_errors_time = [compute_error_metrics_1d(lie_solution[j], ground_truth[j])['l2_error'] 
                              for j in range(len(ground_truth))]
            strang_errors_time = [compute_error_metrics_1d(strang_solution[j], ground_truth[j])['l2_error'] 
                                 for j in range(len(ground_truth))]
            alt_errors_time = [compute_error_metrics_1d(alt_solution[j], ground_truth[j])['l2_error'] 
                              for j in range(len(ground_truth))]
            
            # Detect instability (rapid error growth)
            instability_threshold = 1e-1  # Flag cases with errors above this
            lie_unstable = any(err > instability_threshold for err in lie_errors_time)
            strang_unstable = any(err > instability_threshold for err in strang_errors_time)
            alt_unstable = any(err > instability_threshold for err in alt_errors_time)
            
            # Plot with color coding for stability
            lie_color = 'red' if lie_unstable else 'darkred'
            strang_color = 'red' if strang_unstable else 'blue'
            alt_color = 'red' if alt_unstable else 'green'
            
            axes[1,1].semilogy(time_points, lie_errors_time, 'o-', color=lie_color, 
                              label=f'Lie{" (UNSTABLE)" if lie_unstable else ""}', markersize=4)
            axes[1,1].semilogy(time_points, strang_errors_time, 's-', color=strang_color, 
                              label=f'Strang{" (UNSTABLE)" if strang_unstable else ""}', markersize=4)
            axes[1,1].semilogy(time_points, alt_errors_time, '^-', color=alt_color, 
                              label=f'Alternating{" (UNSTABLE)" if alt_unstable else ""}', markersize=4)
            
            # Add instability threshold line
            axes[1,1].axhline(y=instability_threshold, color='red', linestyle='--', alpha=0.7, 
                             label=f'Instability Threshold ({instability_threshold})')
            
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('L2 Error')
            axes[1,1].set_title('Error Evolution & Instability Detection')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            plt.suptitle(f'dt Convergence Analysis: dt={dt:.6f}, v={v}, D={D}, Pe={Pe:.2f}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'predictions_dt_{dt:.6f}_v{v}_D{D}.png'
            plot_filepath = os.path.join(case_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save metadata
            metadata = {
                'dt': dt,
                'nt': nt,
                'T': T,
                'nx': nx,
                'L': L,
                'v': v,
                'D': D,
                'Pe': Pe
            }
            import json
            with open(os.path.join(case_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Compute errors
        lie_error = compute_error_metrics_1d(lie_solution[-1], final_gt)
        strang_error = compute_error_metrics_1d(strang_solution[-1], final_gt)
        alt_error = compute_error_metrics_1d(alt_solution[-1], final_gt)
        
        results['lie_errors'].append(lie_error['l2_error'])
        results['strang_errors'].append(strang_error['l2_error'])
        results['alternating_errors'].append(alt_error['l2_error'])
        
        print(f"  Lie: {lie_error['l2_error']:.2e}, Strang: {strang_error['l2_error']:.2e}, Alt: {alt_error['l2_error']:.2e}")
        if save_predictions:
            print(f"  Predictions saved to: {case_dir}")
            print(f"  Plot saved to: {plot_filepath}")
    
    # Create convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error vs dt
    ax1.loglog(dt_values, results['lie_errors'], 'ro-', label='Lie Splitting', markersize=6)
    ax1.loglog(dt_values, results['strang_errors'], 'bs-', label='Strang Splitting', markersize=6)
    ax1.loglog(dt_values, results['alternating_errors'], 'g^-', label='Alternating Splitting', markersize=6)
    
    # Add theoretical convergence lines
    dt_ref = dt_values[len(dt_values)//2]
    error_ref = results['strang_errors'][len(dt_values)//2]
    
    # First-order line (for Lie splitting)
    first_order = error_ref * (np.array(dt_values) / dt_ref)
    ax1.loglog(dt_values, first_order, 'k--', alpha=0.5, label='O(dt)')
    
    # Second-order line (for Strang splitting)
    second_order = error_ref * (np.array(dt_values) / dt_ref)**2
    ax1.loglog(dt_values, second_order, 'k:', alpha=0.5, label='O(dt²)')
    
    ax1.set_xlabel('Time Step (dt)')
    ax1.set_ylabel('L2 Error')
    ax1.set_title(f'Convergence Analysis: v={v}, D={D}, Pe={Pe:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Smaller dt on the right
    
    # Plot 2: Error reduction ratio
    if len(dt_values) > 1:
        dt_ratios = []
        lie_ratios = []
        strang_ratios = []
        alt_ratios = []
        
        for i in range(1, len(dt_values)):
            dt_ratio = dt_values[i-1] / dt_values[i]
            lie_ratio = results['lie_errors'][i-1] / results['lie_errors'][i]
            strang_ratio = results['strang_errors'][i-1] / results['strang_errors'][i]
            alt_ratio = results['alternating_errors'][i-1] / results['alternating_errors'][i]
            
            dt_ratios.append(dt_ratio)
            lie_ratios.append(lie_ratio)
            strang_ratios.append(strang_ratio)
            alt_ratios.append(alt_ratio)
        
        ax2.semilogx(dt_ratios, lie_ratios, 'ro-', label='Lie Splitting', markersize=6)
        ax2.semilogx(dt_ratios, strang_ratios, 'bs-', label='Strang Splitting', markersize=6)
        ax2.semilogx(dt_ratios, alt_ratios, 'g^-', label='Alternating Splitting', markersize=6)
        
        # Add theoretical lines
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='O(1) - No convergence')
        ax2.plot(dt_ratios, dt_ratios, 'k--', alpha=0.5, label='O(dt) - First order')
        ax2.plot(dt_ratios, np.array(dt_ratios)**2, 'k:', alpha=0.5, label='O(dt²) - Second order')
    
    ax2.set_xlabel('dt Refinement Ratio')
    ax2.set_ylabel('Error Reduction Ratio')
    ax2.set_title('Convergence Order Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized naming
    plot_filename = f'dt_convergence_v{v}_D{D}_Pe{Pe:.2f}.png'
    plot_filepath = os.path.join(convergence_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to avoid display and save memory
    
    print(f"Convergence plot saved: {plot_filepath}")
    
    # Print summary
    print("\nConvergence Analysis Summary:")
    print("=" * 40)
    print(f"Parameters: v={v}, D={D}, Pe={Pe:.2f}")
    print(f"Time step range: {dt_min} to {dt_max}")
    print(f"Final time: T={T}")
    print("\nFinal errors (smallest dt):")
    print(f"  Lie Splitting:        {results['lie_errors'][-1]:.2e}")
    print(f"  Strang Splitting:     {results['strang_errors'][-1]:.2e}")
    print(f"  Alternating Splitting: {results['alternating_errors'][-1]:.2e}")
    
    return results


def analyze_saved_predictions(prediction_dir: str, dt_value: float = None, method: str = "strang"):
    """
    Analyze saved predictions to identify instabilities and problematic behaviors.
    
    Args:
        prediction_dir: Directory containing saved predictions
        dt_value: Specific dt value to analyze (if None, analyzes all)
        method: Which method to analyze ('lie', 'strang', 'alternating', 'ground_truth')
    
    Returns:
        Dictionary with analysis results
    """
    import os
    import json
    
    if not os.path.exists(prediction_dir):
        print(f"Prediction directory {prediction_dir} does not exist")
        return None
    
    # Find all cases
    cases = [d for d in os.listdir(prediction_dir) if d.startswith("case_dt_")]
    cases.sort()
    
    if dt_value is not None:
        # Filter for specific dt value
        target_case = f"case_dt_{dt_value:.6f}"
        cases = [c for c in cases if c == target_case]
        if not cases:
            print(f"No case found for dt={dt_value}")
            return None
    
    analysis_results = {}
    
    for case in cases:
        case_dir = os.path.join(prediction_dir, case)
        
        # Load metadata
        with open(os.path.join(case_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        dt = metadata['dt']
        print(f"\nAnalyzing case: {case} (dt={dt})")
        
        # Load spatial grid and time points
        x = np.load(os.path.join(case_dir, "x_grid.npy"))
        time_points = np.load(os.path.join(case_dir, "time_points.npy"))
        
        # Load solutions
        method_files = {
            'lie': 'lie_solution.npy',
            'strang': 'strang_solution.npy', 
            'alternating': 'alternating_solution.npy',
            'ground_truth': 'ground_truth.npy'
        }
        
        if method not in method_files:
            print(f"Unknown method: {method}")
            continue
            
        solution = np.load(os.path.join(case_dir, method_files[method]))
        ground_truth = np.load(os.path.join(case_dir, "ground_truth.npy"))
        
        # Compute diagnostics
        diagnostics = {
            'dt': dt,
            'max_values': np.max(np.abs(solution), axis=1),  # Max absolute value at each time step
            'l2_norms': np.sqrt(np.mean(solution**2, axis=1)),  # L2 norm at each time step
            'energy': 0.5 * np.sum(solution**2, axis=1) * (x[1] - x[0]),  # Discrete energy
            'max_gradient': [],  # Maximum spatial gradient at each time
            'instability_onset': None,  # Time when instability starts
            'error_evolution': []  # Error relative to ground truth over time
        }
        
        # Compute spatial gradients and errors
        dx = x[1] - x[0]
        for t_idx in range(len(solution)):
            # Spatial gradient (central differences with periodic BC)
            grad = np.zeros_like(x)
            grad[1:-1] = (solution[t_idx, 2:] - solution[t_idx, :-2]) / (2 * dx)
            grad[0] = (solution[t_idx, 1] - solution[t_idx, -1]) / (2 * dx)  # Periodic
            grad[-1] = (solution[t_idx, 0] - solution[t_idx, -2]) / (2 * dx)  # Periodic
            diagnostics['max_gradient'].append(np.max(np.abs(grad)))
            
            # Error evolution
            error = np.sqrt(np.mean((solution[t_idx] - ground_truth[t_idx])**2))
            diagnostics['error_evolution'].append(error)
        
        diagnostics['max_gradient'] = np.array(diagnostics['max_gradient'])
        diagnostics['error_evolution'] = np.array(diagnostics['error_evolution'])
        
        # Detect instability onset (when L2 norm starts growing exponentially)
        # Look for sudden jumps in max values or L2 norms
        max_val_growth = np.diff(np.log(diagnostics['max_values'] + 1e-16))
        l2_growth = np.diff(np.log(diagnostics['l2_norms'] + 1e-16))
        
        # Find first time when growth rate exceeds threshold
        growth_threshold = 0.1  # 10% growth per time step
        instability_indices = np.where((max_val_growth > growth_threshold) | 
                                     (l2_growth > growth_threshold))[0]
        
        if len(instability_indices) > 0:
            onset_idx = instability_indices[0]
            diagnostics['instability_onset'] = time_points[onset_idx]
            print(f"  Instability detected at t={time_points[onset_idx]:.4f} (step {onset_idx})")
        else:
            print(f"  No instability detected")
        
        # Summary statistics
        print(f"  Max absolute value: {np.max(diagnostics['max_values']):.2e}")
        print(f"  Final L2 norm: {diagnostics['l2_norms'][-1]:.2e}")
        print(f"  Max spatial gradient: {np.max(diagnostics['max_gradient']):.2e}")
        print(f"  Final error: {diagnostics['error_evolution'][-1]:.2e}")
        
        analysis_results[case] = diagnostics
    
    return analysis_results


def create_summary_analysis(results_summary, plot_dir):
    """Create summary plots and analysis across all parameter combinations."""
    
    print("\n" + "="*50)
    print("CREATING SUMMARY ANALYSIS")
    print("="*50)
    
    import pandas as pd
    df = pd.DataFrame(results_summary)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Error vs Peclet number
    axes[0,0].loglog(df['Pe'], df['lie_l2'], 'ro-', label='Lie', markersize=6)
    axes[0,0].loglog(df['Pe'], df['strang_l2'], 'bs-', label='Strang', markersize=6)
    axes[0,0].loglog(df['Pe'], df['alt_l2'], 'g^-', label='Alternating', markersize=6)
    axes[0,0].set_xlabel('Peclet Number (Pe = vL/D)')
    axes[0,0].set_ylabel('L2 Error')
    axes[0,0].set_title('Error vs Peclet Number')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Error vs advection velocity
    axes[0,1].semilogy(df['v'], df['lie_l2'], 'ro-', label='Lie', markersize=6)
    axes[0,1].semilogy(df['v'], df['strang_l2'], 'bs-', label='Strang', markersize=6)
    axes[0,1].semilogy(df['v'], df['alt_l2'], 'g^-', label='Alternating', markersize=6)
    axes[0,1].set_xlabel('Advection Velocity (v)')
    axes[0,1].set_ylabel('L2 Error')
    axes[0,1].set_title('Error vs Advection Velocity')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Error vs diffusion coefficient
    axes[0,2].loglog(df['D'], df['lie_l2'], 'ro-', label='Lie', markersize=6)
    axes[0,2].loglog(df['D'], df['strang_l2'], 'bs-', label='Strang', markersize=6)
    axes[0,2].loglog(df['D'], df['alt_l2'], 'g^-', label='Alternating', markersize=6)
    axes[0,2].set_xlabel('Diffusion Coefficient (D)')
    axes[0,2].set_ylabel('L2 Error')
    axes[0,2].set_title('Error vs Diffusion Coefficient')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Heatmap of errors for Strang splitting
    v_unique = sorted(df['v'].unique())
    D_unique = sorted(df['D'].unique())
    error_matrix = np.zeros((len(D_unique), len(v_unique)))
    
    for i, D in enumerate(D_unique):
        for j, v in enumerate(v_unique):
            error_val = df[(df['v'] == v) & (df['D'] == D)]['strang_l2'].iloc[0]
            error_matrix[i, j] = np.log10(error_val)
    
    im = axes[1,0].imshow(error_matrix, aspect='auto', cmap='viridis')
    axes[1,0].set_xticks(range(len(v_unique)))
    axes[1,0].set_xticklabels([f'{v}' for v in v_unique])
    axes[1,0].set_yticks(range(len(D_unique)))
    axes[1,0].set_yticklabels([f'{D}' for D in D_unique])
    axes[1,0].set_xlabel('Advection Velocity (v)')
    axes[1,0].set_ylabel('Diffusion Coefficient (D)')
    axes[1,0].set_title('Strang Error Heatmap (log10)')
    plt.colorbar(im, ax=axes[1,0])
    
    # Method comparison
    methods = ['lie_l2', 'strang_l2', 'alt_l2']
    method_names = ['Lie', 'Strang', 'Alternating']
    mean_errors = [df[method].mean() for method in methods]
    
    axes[1,1].bar(method_names, mean_errors, color=['red', 'blue', 'green'], alpha=0.7)
    axes[1,1].set_ylabel('Mean L2 Error')
    axes[1,1].set_title('Average Performance Across All Cases')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1,2].axis('off')
    summary_text = f"""PARAMETER STUDY SUMMARY
    
Total cases analyzed: {len(df)}

Advection velocities tested:
{', '.join([str(v) for v in sorted(df['v'].unique())])}

Diffusion coefficients tested:
{', '.join([str(D) for D in sorted(df['D'].unique())])}

Peclet number range:
{df['Pe'].min():.1f} - {df['Pe'].max():.1f}

Best performing method (lowest mean error):
{method_names[np.argmin(mean_errors)]}

Worst case error:
{df[['lie_l2', 'strang_l2', 'alt_l2']].max().max():.2e}

Best case error:
{df[['lie_l2', 'strang_l2', 'alt_l2']].min().min():.2e}
"""
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   fontsize=11, va='top', ha='left', family='monospace')
    
    plt.suptitle('1D Operator Splitting Parameter Study - Summary Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_filepath = os.path.join(plot_dir, 'summary_analysis.png')
    plt.savefig(summary_filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary analysis saved: {summary_filepath}")
    
    # Create visual summary dashboard instead of CSV
    print("\n" + "="*50)
    print("CREATING PARAMETER STUDY DASHBOARD...")
    create_parameter_study_dashboard(df, plot_dir)
    print(f"✓ Visual dashboard created instead of CSV for better instability detection")


if __name__ == "__main__":
    print("1D OPERATOR SPLITTING INVESTIGATION FOR ADVECTION-DIFFUSION")
    print("Complex Initial Condition with High Frequencies")
    print("=" * 70)
    print("This script will create an organized directory structure with all results:")
    print("├── operator_splitting_results/")
    print("│   ├── convergence_analysis/     # dt convergence plots")
    print("│   ├── visualizations/           # Basic comparison plots")
    print("│   └── parameter_study/")
    print("│       ├── individual_cases/     # 25 individual parameter cases")
    print("│       ├── summary_analysis.png  # Overall analysis summary")
    print("│       └── parameter_dashboard/           # Visual instability analysis")
    print("=" * 70)
    
    # Parameters - Finer discretization and longer integration time
    nx = 512  # Finer spatial discretization
    L = 2*np.pi
    v = 1.0
    D = 0.01
    T = 10.0  # Final integration time
    dt = 0.005  # Adjusted temporal discretization for stability and accuracy
    nt = int(T / dt)
    
    # Initialize solver
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    
    print(f"Domain: [0, {L:.2f}], Grid: {nx} points")
    print(f"Velocity: v={v}")
    print(f"Diffusion: D={D}")
    print(f"Time step: dt={dt}, Steps: nt={nt}, Final time: T={T}")
    print()
    
    # Use complex initial condition with high frequencies
    u0 = solver.initial_condition_complex_sines()
    
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
    
    lie_errors = compute_error_metrics_1d(final_lie, final_gt)
    strang_errors = compute_error_metrics_1d(final_strang, final_gt)
    
    print("Basic Results:")
    print(f"Ground truth time: {gt_time:.4f}s")
    print(f"Lie splitting time: {lie_time:.4f}s")
    print(f"Strang splitting time: {strang_time:.4f}s")
    print()
    print("Final time errors:")
    print(f"Lie splitting - L2: {lie_errors['l2_error']:.6e}, L∞: {lie_errors['linf_error']:.6e}")
    print(f"Strang splitting - L2: {strang_errors['l2_error']:.6e}, L∞: {strang_errors['linf_error']:.6e}")
    
    print("\n" + "="*70)
    print("RUNNING dt CONVERGENCE TEST")
    print("="*70)
    
    # Create organized output directory structure
    output_dir = "operator_splitting_results"
    
    # Test dt convergence for specific case: v=0.8, D=0.5
    dt_convergence_results = test_dt_convergence_specific_case(
        nx=nx, L=L, v=0.8, D=0.5, T=1.0, 
        dt_min=0.001, dt_max=0.1, n_dt_values=10,
        output_dir=output_dir
    )
    
    # Create basic visualizations
    create_1d_visualizations(nx=nx, L=L, v=v, D=D, dt=dt, T=T, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE PARAMETER STUDY")
    print("="*70)
    
    # Run comprehensive parameter study
    results = parameter_study_1d(nx=nx, L=L, T=T, dt=dt, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("PARAMETER STUDY COMPLETE!")
    print("="*70)
    print(f"✓ Generated 25 individual case plots in '{output_dir}/parameter_study/individual_cases/' directory")
    print(f"✓ Created summary analysis plot: '{output_dir}/parameter_study/summary_analysis.png'")
    print(f"✓ Created visual dashboards instead of CSV for better instability detection")
    print(f"✓ dt convergence analysis saved in: '{output_dir}/convergence_analysis/'")
    print(f"✓ Basic visualizations saved in: '{output_dir}/visualizations/'")
    print("✓ All plots saved with descriptive filenames indicating parameters and Peclet numbers")
    
    # Show some key findings
    import pandas as pd
    df = pd.DataFrame(results)
    print(f"\nKey Findings:")
    print(f"- Peclet number range tested: {df['Pe'].min():.1f} to {df['Pe'].max():.1f}")
    print(f"- Best overall performance: {['Lie', 'Strang', 'Alternating'][np.argmin([df['lie_l2'].mean(), df['strang_l2'].mean(), df['alt_l2'].mean()])]}")
    print(f"- Lowest error achieved: {df[['lie_l2', 'strang_l2', 'alt_l2']].min().min():.2e}")
    print(f"- Highest error encountered: {df[['lie_l2', 'strang_l2', 'alt_l2']].max().max():.2e}")