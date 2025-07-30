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


def generate_training_data(nx: int = 128,
                          L: float = 2*np.pi,
                          beta_values: Optional[List[float]] = None,
                          nu_values: Optional[List[float]] = None,
                          dt: float = 0.01,
                          T: float = 1.0,
                          n_initial_conditions: int = 5) -> Tuple[Dict, Dict, Dict]:
    """
    Generate complete training dataset for neural operator splitting.
    
    Args:
        nx: Number of spatial grid points
        L: Domain length
        beta_values: Advection coefficients
        nu_values: Diffusion coefficients
        dt: Time step
        T: Final time
        n_initial_conditions: Number of initial conditions per parameter
        
    Returns:
        Tuple of (advection_data, diffusion_data, combined_data)
    """
    if beta_values is None:
        beta_values = [0.5]
    if nu_values is None:
        nu_values = [0.1]
    
    generator = TrajectoryGenerator(nx, L)
    
    # Generate multiple initial conditions
    temp_solver = AdvectionDiffusion1DSolver(nx, L, 1.0, 1.0)  # Temporary solver for initial condition
    initial_conditions = []
    for i in range(n_initial_conditions):
        modes = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,14,15,16], size=6, replace=False)
        amplitudes = np.random.uniform(0.3, 1.0, size=6)
        u0 = temp_solver.initial_condition_complex_sines(modes.tolist(), amplitudes.tolist())
        initial_conditions.append(u0)
    
    # Generate individual operator trajectories with multiple initial conditions
    advection_data = generator.generate_advection_trajectories_multiple_ic(
        beta_values, initial_conditions, dt, T)
    
    diffusion_data = generator.generate_diffusion_trajectories_multiple_ic(
        nu_values, initial_conditions, dt, T)
    
    # Generate combined trajectories for comparison
    combined_data = generator.generate_combined_trajectories_multiple_ic(
        beta_values[:2], nu_values[:2], initial_conditions, dt, T)  # Smaller set for comparison
    
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
    plt.savefig('/mnt/home/lserrano/disco-ball/neural-operator-splitting/advection_examples.png', 
                dpi=150, bbox_inches='tight')
    
    print("Visualizing diffusion trajectories...")
    fig2 = generator.visualize_trajectories(diffusion_data, max_trajectories=2)
    plt.savefig('/mnt/home/lserrano/disco-ball/neural-operator-splitting/diffusion_examples.png',
                dpi=150, bbox_inches='tight')
    
    print("Data generation and visualization complete!")