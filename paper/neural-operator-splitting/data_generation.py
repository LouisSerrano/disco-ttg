"""
Data Generation for Neural Operator Splitting

This module generates training data for neural operators by using existing FFT solvers
to create ground truth trajectories for pure advection and diffusion operators.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Add the parent directory to path to import from operator-splitting
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Now import from the symlinked directory
try:
    from operator_splitting.operator_splitting_1d import AdvectionDiffusion1DSolver
except ImportError:
    # Fallback: import directly from the actual directory
    operator_splitting_path = os.path.join(parent_dir, 'operator-splitting')
    sys.path.insert(0, operator_splitting_path)
    from operator_splitting_1d import AdvectionDiffusion1DSolver

# Import Fractaloid for aligned initial conditions
from src.advection_diffusion import Fractaloid


class TrajectoryGenerator:
    """Generate training trajectories for individual operators using FFT ground truth."""
    
    def __init__(self, nx: int = 128, L: float = 2*np.pi):
        """
        Initialize trajectory generator.
        
        Args:
            nx: Number of spatial grid points
            L: Domain length
        """
        self.nx = nx
        self.L = L
        self.x = np.linspace(0, L, nx, endpoint=False)
        
    def generate_advection_trajectories(self,
                                      beta_values: List[float],
                                      u0: np.ndarray,
                                      dt: float = 0.01,
                                      T: float = 1.0,
                                      n_initial_conditions: int = 5) -> Dict:
        """
        Generate pure advection trajectories with different coefficients and given initial condition.
        
        Args:
            beta_values: List of advection coefficients
            u0: Initial condition array
            dt: Time step
            T: Final time
            n_initial_conditions: Number of different initial conditions per coefficient
            
        Returns:
            Dictionary containing trajectories, parameters, and metadata
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions = []
        
        print(f"Generating advection trajectories...")
        print(f"Beta values: {beta_values}")
        print(f"Initial conditions per beta: {n_initial_conditions}")
        
        for beta in beta_values:
            # Create solver with pure advection (D=0)
            solver = AdvectionDiffusion1DSolver(self.nx, self.L, beta, D=0.0)
            
            # Generate trajectory using FFT ground truth with provided initial condition
            trajectory = solver.fft_ground_truth(u0, dt, nt)
            trajectory = np.array(trajectory)  # Shape: (nt+1, nx)
            
            trajectories.append(trajectory)
            parameters.append({'beta': beta, 'D': 0.0, 'type': 'advection'})
            initial_conditions.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }
    
    def generate_diffusion_trajectories(self,
                                      nu_values: List[float],
                                      u0: np.ndarray,
                                      dt: float = 0.01,
                                      T: float = 1.0,
                                      n_initial_conditions: int = 5) -> Dict:
        """
        Generate pure diffusion trajectories with different coefficients and given initial condition.
        
        Args:
            nu_values: List of diffusion coefficients
            u0: Initial condition array
            dt: Time step  
            T: Final time
            n_initial_conditions: Number of different initial conditions per coefficient
            
        Returns:
            Dictionary containing trajectories, parameters, and metadata
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions = []
        
        print(f"Generating diffusion trajectories...")
        print(f"Nu values: {nu_values}")
        print(f"Initial conditions per nu: {n_initial_conditions}")
        
        for nu in nu_values:
            # Create solver with pure diffusion (v=0)
            solver = AdvectionDiffusion1DSolver(self.nx, self.L, v=0.0, D=nu)
            
            # Generate trajectory using FFT ground truth with provided initial condition
            trajectory = solver.fft_ground_truth(u0, dt, nt)
            trajectory = np.array(trajectory)  # Shape: (nt+1, nx)
            
            trajectories.append(trajectory)
            parameters.append({'beta': 0.0, 'D': nu, 'type': 'diffusion'})
            initial_conditions.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }
    
    def generate_combined_trajectories(self,
                                     beta_values: List[float],
                                     nu_values: List[float],
                                     u0: np.ndarray,
                                     dt: float = 0.01,
                                     T: float = 1.0,
                                     n_initial_conditions: int = 3) -> Dict:
        """
        Generate combined advection-diffusion trajectories for comparison.
        
        Args:
            beta_values: List of advection coefficients
            nu_values: List of diffusion coefficients
            u0: Initial condition array
            dt: Time step
            T: Final time
            n_initial_conditions: Number of initial conditions per parameter combination
            
        Returns:
            Dictionary containing combined trajectories for ground truth comparison
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions = []
        
        print(f"Generating combined advection-diffusion trajectories...")
        
        for beta in beta_values:
            for nu in nu_values:
                solver = AdvectionDiffusion1DSolver(self.nx, self.L, beta, nu)
                
                # Generate trajectory using FFT ground truth with provided initial condition
                trajectory = solver.fft_ground_truth(u0, dt, nt)
                trajectory = np.array(trajectory)
                
                trajectories.append(trajectory)
                parameters.append({'beta': beta, 'D': nu, 'type': 'combined'})
                initial_conditions.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }
    
    def generate_advection_trajectories_multiple_ic(self,
                                                  beta_values: List[float],
                                                  initial_conditions: List[np.ndarray],
                                                  dt: float = 0.01,
                                                  T: float = 1.0) -> Dict:
        """
        Generate pure advection trajectories with different coefficients and multiple initial conditions.
        
        Args:
            beta_values: List of advection coefficients
            initial_conditions: List of initial condition arrays
            dt: Time step
            T: Final time
            
        Returns:
            Dictionary containing trajectories, parameters, and metadata
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions_out = []
        
        print(f"Generating advection trajectories...")
        print(f"Beta values: {beta_values}")
        print(f"Initial conditions: {len(initial_conditions)}")
        
        for beta in beta_values:
            for u0 in initial_conditions:
                # Create solver with pure advection (D=0)
                solver = AdvectionDiffusion1DSolver(self.nx, self.L, beta, D=0.0)
                
                # Generate trajectory using FFT ground truth with provided initial condition
                trajectory = solver.fft_ground_truth(u0, dt, nt)
                trajectory = np.array(trajectory)  # Shape: (nt+1, nx)
                
                trajectories.append(trajectory)
                parameters.append({'beta': beta, 'D': 0.0, 'type': 'advection'})
                initial_conditions_out.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions_out,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }
    
    def generate_diffusion_trajectories_multiple_ic(self,
                                                  nu_values: List[float],
                                                  initial_conditions: List[np.ndarray],
                                                  dt: float = 0.01,
                                                  T: float = 1.0) -> Dict:
        """
        Generate pure diffusion trajectories with different coefficients and multiple initial conditions.
        
        Args:
            nu_values: List of diffusion coefficients
            initial_conditions: List of initial condition arrays
            dt: Time step  
            T: Final time
            
        Returns:
            Dictionary containing trajectories, parameters, and metadata
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions_out = []
        
        print(f"Generating diffusion trajectories...")
        print(f"Nu values: {nu_values}")
        print(f"Initial conditions: {len(initial_conditions)}")
        
        for nu in nu_values:
            for u0 in initial_conditions:
                # Create solver with pure diffusion (v=0)
                solver = AdvectionDiffusion1DSolver(self.nx, self.L, v=0.0, D=nu)
                
                # Generate trajectory using FFT ground truth with provided initial condition
                trajectory = solver.fft_ground_truth(u0, dt, nt)
                trajectory = np.array(trajectory)  # Shape: (nt+1, nx)
                
                trajectories.append(trajectory)
                parameters.append({'beta': 0.0, 'D': nu, 'type': 'diffusion'})
                initial_conditions_out.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions_out,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }
    
    def generate_combined_trajectories_multiple_ic(self,
                                                 beta_values: List[float],
                                                 nu_values: List[float],
                                                 initial_conditions: List[np.ndarray],
                                                 dt: float = 0.01,
                                                 T: float = 1.0) -> Dict:
        """
        Generate combined advection-diffusion trajectories with multiple initial conditions.
        
        Args:
            beta_values: List of advection coefficients
            nu_values: List of diffusion coefficients
            initial_conditions: List of initial condition arrays
            dt: Time step
            T: Final time
            
        Returns:
            Dictionary containing combined trajectories for ground truth comparison
        """
        nt = int(T / dt)
        trajectories = []
        parameters = []
        initial_conditions_out = []
        
        print(f"Generating combined advection-diffusion trajectories...")
        
        for beta in beta_values:
            for nu in nu_values:
                for u0 in initial_conditions:
                    solver = AdvectionDiffusion1DSolver(self.nx, self.L, beta, nu)
                    
                    # Generate trajectory using FFT ground truth with provided initial condition
                    trajectory = solver.fft_ground_truth(u0, dt, nt)
                    trajectory = np.array(trajectory)
                    
                    trajectories.append(trajectory)
                    parameters.append({'beta': beta, 'D': nu, 'type': 'combined'})
                    initial_conditions_out.append(u0.copy())
        
        return {
            'trajectories': trajectories,
            'parameters': parameters,
            'initial_conditions': initial_conditions_out,
            'metadata': {
                'nx': self.nx,
                'L': self.L,
                'dt': dt,
                'T': T,
                'nt': nt,
                'x': self.x,
                'time': np.linspace(0, T, nt+1)
            }
        }

    def visualize_trajectories(self, data: Dict, max_trajectories: int = 3):
        """Visualize a few example trajectories."""
        trajectories = data['trajectories'][:max_trajectories]
        parameters = data['parameters'][:max_trajectories]
        x = data['metadata']['x']
        time = data['metadata']['time']
        
        fig, axes = plt.subplots(len(trajectories), 2, figsize=(12, 4*len(trajectories)))
        if len(trajectories) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (traj, params) in enumerate(zip(trajectories, parameters)):
            # Plot initial and final states
            axes[i, 0].plot(x, traj[0], 'b-', label='Initial', linewidth=2)
            axes[i, 0].plot(x, traj[-1], 'r-', label='Final', linewidth=2)
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('u')
            axes[i, 0].set_title(f'{params["type"]}: β={params["beta"]}, ν={params["D"]}')
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            
            # Plot spacetime evolution
            X, T = np.meshgrid(x, time)
            im = axes[i, 1].contourf(X, T, traj, levels=20, cmap='viridis')
            axes[i, 1].set_xlabel('x')
            axes[i, 1].set_ylabel('time')
            axes[i, 1].set_title('Spacetime Evolution')
            plt.colorbar(im, ax=axes[i, 1])
        
        plt.tight_layout()
        return fig


def advection_diffusion_analytical(u0, L=16.0, v=0.1, D=0.5, nt=100, T=10.0):
    """
    Compute the analytical solution of the 1D advection-diffusion equation
    with periodic boundary conditions using the Fourier spectral method.

    Parameters:
        u0 (ndarray): Initial condition, array of shape (nx,)
        L (float): Domain length
        v (float): Advection speed
        D (float): Diffusion coefficient
        nt (int): Number of time steps
        T (float): Final time

    Returns:
        u_xt (ndarray): Solution array of shape (nt, nx)
        x (ndarray): Spatial grid of shape (nx,)
        t (ndarray): Time grid of shape (nt,)
    """
    nx = len(u0)  # infer spatial resolution from input
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt)

    # Fourier wavenumbers
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    k = 1j * k  # complex wavenumber for exponential form

    # FFT of initial condition
    u0_hat = np.fft.fft(u0)

    # Allocate solution array
    u_xt = np.zeros((nt, nx))

    # Time evolution in spectral space
    for i, ti in enumerate(t):
        decay = np.exp(D * (k**2) * ti) * np.exp(-k * v * ti)
        u_hat_t = u0_hat * decay
        u_xt[i] = np.fft.ifft(u_hat_t).real  # keep only real part

    return u_xt, x, t


def generate_training_data(nx: int = 256,
                          L: float = 16.0,
                          beta_values: Optional[List[float]] = None,
                          nu_values: Optional[List[float]] = None,
                          dt: float = 0.01,
                          T: float = 10.0,
                          nt: int = 100,
                          n_initial_conditions: int = 5,
                          fractal_degree: int = 8,
                          fractal_power_range: Tuple[float, float] = (1.5, 2.5)) -> Tuple[Dict, Dict, Dict]:
    """
    Generate complete training dataset for neural operator splitting using Fractaloid initial conditions
    and analytical solutions, aligned with train/train.py approach.
    
    Args:
        nx: Number of spatial grid points
        L: Domain length
        beta_values: Advection coefficients (v in analytical solution)
        nu_values: Diffusion coefficients (D in analytical solution)
        dt: Time step (for compatibility, actual dt = T/nt)
        T: Final time
        nt: Number of time steps
        n_initial_conditions: Number of initial conditions per parameter
        fractal_degree: Degree for Fractaloid generation
        fractal_power_range: Range for fractal power parameter
        
    Returns:
        Tuple of (advection_data, diffusion_data, combined_data)
    """
    if beta_values is None:
        beta_values = [1.0]  # Aligned with train/train.py range
    if nu_values is None:
        nu_values = [0.5]   # Aligned with train/train.py range
    
    # Generate multiple initial conditions using Fractaloid (same as train/train.py)
    initial_conditions = []
    rng = np.random.default_rng()
    
    for i in range(n_initial_conditions):
        # Generate fractal power in specified range
        fractal_power = rng.uniform(*fractal_power_range)
        fractaloid = Fractaloid(
            degree=fractal_degree,
            power=fractal_power,
            size=nx,
            patch_size=nx
        )
        u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
        # Normalize like in train/train.py
        u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
        initial_conditions.append(u0)
    
    # Generate trajectories using analytical solution
    advection_trajectories = []
    advection_parameters = []
    diffusion_trajectories = []
    diffusion_parameters = []
    combined_trajectories = []
    combined_parameters = []
    
    # Time array
    time_points = np.linspace(0, T, nt)
    
    # Generate pure advection trajectories (D=0)
    for beta in beta_values:
        for u0 in initial_conditions:
            u_xt, x, t = advection_diffusion_analytical(
                u0, L=L, v=beta, D=0.0, nt=nt, T=T
            )
            advection_trajectories.append(u_xt)
            advection_parameters.append({
                'beta': beta,
                'D': 0.0,
                'type': 'advection'
            })
    
    # Generate pure diffusion trajectories (v=0)
    for nu in nu_values:
        for u0 in initial_conditions:
            u_xt, x, t = advection_diffusion_analytical(
                u0, L=L, v=0.0, D=nu, nt=nt, T=T
            )
            diffusion_trajectories.append(u_xt)
            diffusion_parameters.append({
                'beta': 0.0,
                'D': nu,
                'type': 'diffusion'
            })
    
    # Generate combined trajectories for comparison (first 2 values of each)
    for beta in beta_values[:min(2, len(beta_values))]:
        for nu in nu_values[:min(2, len(nu_values))]:
            for u0 in initial_conditions:
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=L, v=beta, D=nu, nt=nt, T=T
                )
                combined_trajectories.append(u_xt)
                combined_parameters.append({
                    'beta': beta,
                    'D': nu,
                    'type': 'combined'
                })
    
    # Create data dictionaries
    x_grid = np.linspace(0, L, nx, endpoint=False)
    metadata = {
        'nx': nx,
        'L': L,
        'nt': nt,
        'T': T,
        'time': time_points,
        'x': x_grid,
        'dt': T / (nt - 1),
        'dx': L / nx
    }
    
    advection_data = {
        'trajectories': advection_trajectories,
        'parameters': advection_parameters,
        'initial_conditions': initial_conditions,
        'metadata': metadata
    }
    
    diffusion_data = {
        'trajectories': diffusion_trajectories,
        'parameters': diffusion_parameters,
        'initial_conditions': initial_conditions,
        'metadata': metadata
    }
    
    combined_data = {
        'trajectories': combined_trajectories,
        'parameters': combined_parameters,
        'initial_conditions': initial_conditions,
        'metadata': metadata
    }
    
    print(f"\nData generation complete:")
    print(f"Advection trajectories: {len(advection_data['trajectories'])}")
    print(f"Diffusion trajectories: {len(diffusion_data['trajectories'])}")
    print(f"Combined trajectories: {len(combined_data['trajectories'])}")
    
    return advection_data, diffusion_data, combined_data


if __name__ == "__main__":
    # Generate example data
    print("Generating training data for neural operator splitting...")
    
    advection_data, diffusion_data, combined_data = generate_training_data(
        nx=64, T=0.5, dt=0.01, n_initial_conditions=3
    )
    
    # Visualize some examples
    generator = TrajectoryGenerator()
    
    print("\nVisualizing advection trajectories...")
    fig1 = generator.visualize_trajectories(advection_data, max_trajectories=2)
    plt.savefig('/mnt/home/lserrano/disco-ttg/neural-operator-splitting/advection_examples.png', 
                dpi=150, bbox_inches='tight')
    
    print("Visualizing diffusion trajectories...")
    fig2 = generator.visualize_trajectories(diffusion_data, max_trajectories=2)
    plt.savefig('/mnt/home/lserrano/disco-ttg/neural-operator-splitting/diffusion_examples.png',
                dpi=150, bbox_inches='tight')
    
    print("Data generation and visualization complete!")