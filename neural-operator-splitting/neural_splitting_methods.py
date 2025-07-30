"""
Neural Operator Splitting Methods

This module implements operator splitting methods (Lie, Strang) using trained
neural ODE operators for advection and diffusion. It also provides comparison
methods with classical operator splitting from the existing framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
from tqdm import tqdm

# Add parent directory to import existing operator splitting
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Now import from the symlinked directory
try:
    from operator_splitting.operator_splitting_1d import AdvectionDiffusion1DSolver, compute_error_metrics_1d as compute_error_metrics, OperatorSplitting1DMethods
except ImportError:
    # Fallback: import directly from the actual directory
    operator_splitting_path = os.path.join(parent_dir, 'operator-splitting')
    sys.path.insert(0, operator_splitting_path)
    from operator_splitting_1d import OperatorSplitting1DMethods, AdvectionDiffusion1DSolver, compute_error_metrics_1d as compute_error_metrics

from neural_ode_operators import AdvectionNeuralODE, DiffusionNeuralODE


class NeuralOperatorSplitting:
    """Neural operator splitting methods using trained neural ODEs."""
    
    def __init__(self, 
                 advection_model: AdvectionNeuralODE,
                 diffusion_model: DiffusionNeuralODE,
                 device: str = 'auto'):
        """
        Initialize neural operator splitting.
        
        Args:
            advection_model: Trained advection neural ODE
            diffusion_model: Trained diffusion neural ODE  
            device: Device to run models on
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.advection_model = advection_model.to(device)
        self.diffusion_model = diffusion_model.to(device)
        
        # Set models to evaluation mode
        self.advection_model.eval()
        self.diffusion_model.eval()
        
        # Get spatial parameters
        self.nx = advection_model.nx
        self.L = advection_model.L
        self.dx = self.L / self.nx
    
    def lie_splitting(self, 
                     u0: np.ndarray,
                     dt: float,
                     nt: int,
                     method: str = 'rk4',
                     rtol: float = 1e-7,
                     atol: float = 1e-9,
                     save_intermediate: bool = True,
                     num_save_steps: int = None) -> List[np.ndarray]:
        """
        First-order Lie splitting: A(dt) ∘ D(dt)
        Apply advection for full timestep, then diffusion for full timestep.
        
        Args:
            u0: Initial condition (nx,)
            dt: Time step
            nt: Number of time steps
            method: ODE solver method
            rtol: Relative tolerance  
            atol: Absolute tolerance
            save_intermediate: If True, save intermediate predictions at regular intervals
            num_save_steps: Number of intermediate steps to save (if None, saves every step)
            
        Returns:
            List of solutions at each time step (or at save intervals if save_intermediate=True)
        """
        if u0.ndim ==1:
            u0[None, ...]
        solutions = [u0.copy()]
        u0 = u0[:, None]
        u = torch.from_numpy(u0.copy()).float().to(self.device)  # (1, nx)
        print('------- Lie splitting --------')
        #print('u0', u.shape)
        
        # Set default num_save_steps to nt if not provided
        if num_save_steps is None:
            num_save_steps = nt
            
        # Determine save frequency
        if save_intermediate and num_save_steps is not None:
            save_every = max(1, nt // num_save_steps)
            print(f'Saving every {save_every} steps for {num_save_steps} intermediate saves')
        else:
            save_every = 1  # Save every step (default behavior)


        num_steps = self.advection_model.num_steps
        #nt = nt*num_steps
        #dt = 1/num_steps
        
        with torch.no_grad():
            for n in range(nt):
                # Step 1: Apply advection operator for dt
                # Use explicit t_span for operator splitting timesteps to maintain classical behavior
                #print('u', u.shape)
                t_span = torch.tensor([0.0, dt], device=self.device)
                n_steps, u_after_advection = self.advection_model(u, t_span=t_span, method=method, rtol=rtol, atol=atol)
                u = u_after_advection[-1]  # Take final time point

                #print('u after advection', u_after_advection.shape)
                
                # Step 2: Apply diffusion operator for dt
                n_steps, u_after_diffusion = self.diffusion_model(u, t_span=t_span, method=method, rtol=rtol, atol=atol)
                u = u_after_diffusion[-1]  # Take final time point

                #print('u after diffusion', u_after_diffusion.shape)
                
                # Save solution at specified intervals
                if (n + 1) % save_every == 0 or n == nt - 1:  # Save at intervals or final step
                    u_np = u.squeeze(0).squeeze(1).cpu().numpy()
                    solutions.append(u_np.copy())
                    #print('u_np', n, u_np.shape)

            #solutions = np.concatenate(solutions, axis=1)
            #print('solution', solutions.shape)
        
        return solutions
    
    def strang_splitting(self,
                        u0: np.ndarray,
                        dt: float,
                        nt: int,
                        method: str = 'rk4',
                        rtol: float = 1e-7,
                        atol: float = 1e-9,
                        save_intermediate: bool = False,
                        num_save_steps: int = None) -> List[np.ndarray]:
        """
        Second-order Strang splitting: D(dt/2) ∘ A(dt) ∘ D(dt/2)
        Apply diffusion for half timestep, advection for full timestep, 
        then diffusion for half timestep.
        
        Args:
            u0: Initial condition (nx,)
            dt: Time step
            nt: Number of time steps
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            save_intermediate: If True, save intermediate predictions at regular intervals
            num_save_steps: Number of intermediate steps to save (if None, defaults to nt)
            
        Returns:
            List of solutions at each time step (or at save intervals if save_intermediate=True)
        """
        u = torch.from_numpy(u0.copy()).float().unsqueeze(0).to(self.device)  # (1, nx)
        solutions = [u0.copy()]
        
        # Set default num_save_steps to nt if not provided
        if num_save_steps is None:
            num_save_steps = nt
            
        # Determine save frequency
        if save_intermediate and num_save_steps is not None:
            save_every = max(1, nt // num_save_steps)
            print(f'Saving every {save_every} steps for {num_save_steps} intermediate saves')
        else:
            save_every = 1  # Save every step (default behavior)
        
        t_span_half = torch.tensor([0.0, dt/2], device=self.device)
        t_span_full = torch.tensor([0.0, dt], device=self.device)
        
        with torch.no_grad():
            for n in range(nt):
                # Step 1: Apply diffusion for dt/2
                n_steps, u_after_diff1 = self.diffusion_model(u, t_span=t_span_half, method=method, rtol=rtol, atol=atol)
                u = u_after_diff1[-1]
                
                # Step 2: Apply advection for dt
                n_steps, u_after_advection = self.advection_model(u, t_span=t_span_full, method=method, rtol=rtol, atol=atol)
                u = u_after_advection[-1]
                
                # Step 3: Apply diffusion for dt/2
                n_steps, u_after_diff2 = self.diffusion_model(u, t_span=t_span_half, method=method, rtol=rtol, atol=atol)
                u = u_after_diff2[-1]
                
                # Convert back to numpy and store (with save frequency logic)
                u_np = u.squeeze(0).cpu().numpy()
                if (n + 1) % save_every == 0 or n == nt - 1:
                    solutions.append(u_np.copy())
        
        return solutions
    
    def alternating_splitting(self,
                            u0: np.ndarray,
                            dt: float,
                            nt: int,
                            method: str = 'rk4',
                            start_with_advection: bool = True) -> List[np.ndarray]:
        """
        Alternating splitting: switches order at each time step.
        Even steps: A(dt) ∘ D(dt), Odd steps: D(dt) ∘ A(dt)
        
        Args:
            u0: Initial condition
            dt: Time step  
            nt: Number of time steps
            method: ODE solver method
            start_with_advection: Whether to start with advection
            
        Returns:
            List of solutions at each time step
        """
        u = torch.from_numpy(u0.copy()).float().unsqueeze(0).to(self.device)
        solutions = [u0.copy()]
        
        t_span = torch.tensor([0.0, dt], device=self.device)
        
        with torch.no_grad():
            for n in range(nt):
                advection_first = start_with_advection if n % 2 == 0 else not start_with_advection
                
                if advection_first:
                    # A(dt) ∘ D(dt)
                    u_after_advection = self.advection_model(u, t_span)
                    u = u_after_advection[-1]
                    u_after_diffusion = self.diffusion_model(u, t_span)
                    u = u_after_diffusion[-1]
                else:
                    # D(dt) ∘ A(dt)
                    u_after_diffusion = self.diffusion_model(u, t_span)
                    u = u_after_diffusion[-1]
                    u_after_advection = self.advection_model(u, t_span)
                    u = u_after_advection[-1]
                
                u_np = u.squeeze(0).cpu().numpy()
                solutions.append(u_np.copy())
        
        return solutions
    
    def predict_with_default_T(self, 
                              u0: np.ndarray,
                              operator: str = 'both',
                              T: float = 1.0,
                              n_intermediate: int = 10,
                              method: str = 'rk4') -> Dict[str, np.ndarray]:
        """
        Make predictions using neural ODE with default T=1 and intermediate steps.
        This demonstrates the new default behavior for neural ODE operators.
        
        Args:
            u0: Initial condition (nx,)
            operator: Which operator to use ('advection', 'diffusion', 'both')
            T: Final time (default 1.0)
            n_intermediate: Number of intermediate evaluation points
            method: ODE solver method
            
        Returns:
            Dictionary with predictions from each operator
        """
        u = torch.from_numpy(u0.copy()).float().unsqueeze(0).to(self.device)  # (1, nx)
        results = {}
        
        with torch.no_grad():
            if operator in ['advection', 'both']:
                print(f"Neural advection prediction: T={T}, intermediate_steps={n_intermediate}")
                n_steps, advection_solution = self.advection_model(
                    u, T=T, num_steps=n_intermediate, method=method
                )
                results['advection'] = advection_solution.squeeze(1).cpu().numpy()  # (time_steps, nx)
                print(f"  Output shape: {results['advection'].shape}")
            
            if operator in ['diffusion', 'both']:
                print(f"Neural diffusion prediction: T={T}, intermediate_steps={n_intermediate}")
                n_steps, diffusion_solution = self.diffusion_model(
                    u, T=T, num_steps=n_intermediate, method=method
                )
                results['diffusion'] = diffusion_solution.squeeze(1).cpu().numpy()  # (time_steps, nx)
                print(f"  Output shape: {results['diffusion'].shape}")
        
        return results
    
    def lie_splitting_with_predictions(self,
                                     u0: np.ndarray,
                                     T_total: float = 1.0,
                                     num_predictions: int = 50,
                                     method: str = 'rk4') -> Dict[str, np.ndarray]:
        """
        Lie splitting with automatic dt calculation for desired prediction frequency.
        
        Args:
            u0: Initial condition (nx,)
            T_total: Total integration time
            num_predictions: Number of prediction points to save
            method: ODE solver method
            
        Returns:
            Dictionary with solutions and time points
        """
        # Calculate dt and nt for desired prediction frequency
        dt = T_total / num_predictions
        nt = num_predictions
        
        print(f"Lie splitting with T={T_total}, dt={dt:.6f}, nt={nt}")
        
        # Run splitting and save every step (since dt is already set correctly)
        solutions = self.lie_splitting(u0, dt, nt, method, save_intermediate=False)
        
        # Create time points
        time_points = np.linspace(0, T_total, len(solutions))
        
        return {
            'solutions': np.array(solutions),
            'time_points': time_points,
            'dt': dt,
            'T_total': T_total,
            'num_predictions': len(solutions)
        }
    
    def strang_splitting_with_predictions(self,
                                        u0: np.ndarray,
                                        T_total: float = 1.0,
                                        num_predictions: int = 50,
                                        method: str = 'rk4') -> Dict[str, np.ndarray]:
        """
        Strang splitting with automatic dt calculation for desired prediction frequency.
        
        Args:
            u0: Initial condition (nx,)
            T_total: Total integration time
            num_predictions: Number of prediction points to save
            method: ODE solver method
            
        Returns:
            Dictionary with solutions and time points
        """
        # Calculate dt and nt for desired prediction frequency
        dt = T_total / num_predictions
        nt = num_predictions
        
        print(f"Strang splitting with T={T_total}, dt={dt:.6f}, nt={nt}")
        
        # Run splitting and save every step (since dt is already set correctly)
        solutions = self.strang_splitting(u0, dt, nt, method)
        
        # Create time points
        time_points = np.linspace(0, T_total, len(solutions))
        
        return {
            'solutions': np.array(solutions),
            'time_points': time_points,
            'dt': dt,
            'T_total': T_total,
            'num_predictions': len(solutions)
        }


class ComparisonFramework:
    """Framework for comparing neural and classical operator splitting methods."""
    
    def __init__(self, 
                 neural_splitting: NeuralOperatorSplitting,
                 nx: int,
                 L: float,
                 beta: float,
                 nu: float):
        """
        Initialize comparison framework.
        
        Args:
            neural_splitting: Neural operator splitting instance
            nx: Number of spatial grid points  
            L: Domain length
            beta: Advection coefficient
            nu: Diffusion coefficient
        """
        self.neural_splitting = neural_splitting
        self.nx = nx
        self.L = L
        self.beta = beta
        self.nu = nu
        
        # Create classical solver for comparison
        self.classical_solver = AdvectionDiffusion1DSolver(nx, L, beta, nu)
        
    def compare_methods(self,
                       u0: np.ndarray,
                       dt: float,
                       nt: int,
                       methods: List[str] = None,
                       ode_method: str = 'rk4') -> Dict:
        """
        Compare different splitting methods with ground truth.
        
        Args:
            u0: Initial condition
            dt: Time step
            nt: Number of time steps
            methods: List of methods to compare
            ode_method: ODE solver method for neural operators
            
        Returns:
            Dictionary with solutions and error metrics
        """
        if methods is None:
            methods = ['neural_lie', 'neural_strang', 'classical_lie', 'classical_strang', 'ground_truth']
        
        results = {}
        
        print("Computing solutions...")
        
        # Ground truth using FFT
        if 'ground_truth' in methods:
            print("  Computing ground truth (FFT)...")
            ground_truth = self.classical_solver.fft_ground_truth(u0.copy(), dt, nt)
            results['ground_truth'] = ground_truth
        
        # Classical operator splitting methods
        if 'classical_lie' in methods or 'classical_strang' in methods:
            classical_splitting = OperatorSplitting1DMethods(self.classical_solver)
            
            if 'classical_lie' in methods:
                print("  Computing classical Lie splitting...")
                classical_lie = classical_splitting.lie_splitting(u0.copy(), dt, nt)
                results['classical_lie'] = classical_lie
            
            if 'classical_strang' in methods:
                print("  Computing classical Strang splitting...")
                classical_strang = classical_splitting.strang_splitting(u0.copy(), dt, nt)
                results['classical_strang'] = classical_strang
        
        # Neural operator splitting methods
        if 'neural_lie' in methods:
            print("  Computing neural Lie splitting...")
            num_steps = self.neural_splitting.advection_model.num_steps
            neural_lie = self.neural_splitting.lie_splitting(u0.copy(), dt/num_steps, nt*num_steps, ode_method, save_intermediate=True, num_save_steps=nt)
            results['neural_lie'] = neural_lie
        
        if 'neural_strang' in methods:
            print("  Computing neural Strang splitting...")
            neural_strang = self.neural_splitting.strang_splitting(u0.copy(), dt/num_steps, nt*num_steps, ode_method, save_intermediate=True, num_save_steps=nt)
            results['neural_strang'] = neural_strang
        
        if 'neural_alternating' in methods:
            print("  Computing neural alternating splitting...")
            neural_alt = self.neural_splitting.alternating_splitting(u0.copy(), dt, nt, ode_method)
            results['neural_alternating'] = neural_alt
        
        # Compute error metrics if ground truth is available
        if 'ground_truth' in results:
            print("Computing error metrics...")
            ground_truth = results['ground_truth']
            final_gt = ground_truth[-1]
            
            errors = {}
            for method_name, solution in results.items():
                if method_name == 'ground_truth' or solution is None:
                    continue
                
                final_solution = solution[-1]
                error_metrics = compute_error_metrics(final_solution, final_gt)
                errors[method_name] = error_metrics
                
                print(f"  {method_name}: L2 = {error_metrics['l2_error']:.2e}, "
                      f"L∞ = {error_metrics['linf_error']:.2e}")
            
            results['errors'] = errors
        
        return results
    
    def dt_convergence_study(self,
                           u0: np.ndarray,
                           T: float = 1.0,
                           dt_values: List[float] = None,
                           methods: List[str] = None,
                           ode_method: str = 'rk4') -> Dict:
        """
        Study convergence with respect to time step size.
        
        Args:
            u0: Initial condition
            T: Final time
            dt_values: List of time step values
            methods: Methods to compare
            ode_method: ODE solver method
            
        Returns:
            Convergence study results
        """
        if dt_values is None:
            dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]
        
        if methods is None:
            methods = ['neural_lie', 'neural_strang', 'classical_lie', 'classical_strang']
        
        convergence_results = {
            'dt_values': dt_values,
            'errors': {method: [] for method in methods}
        }
        
        print(f"dt convergence study with T={T}")
        print(f"dt values: {dt_values}")
        
        for dt in dt_values:
            nt = int(T / dt)
            print(f"\nTesting dt = {dt:.5f} (nt = {nt})")
            
            # Compute ground truth with smallest dt for reference
            if dt == min(dt_values):
                ground_truth = self.classical_solver.fft_ground_truth(u0, dt, nt)
                reference_solution = ground_truth[-1]
                convergence_results['reference_solution'] = reference_solution
            else:
                # Use the reference solution from smallest dt
                reference_solution = convergence_results.get('reference_solution')
                if reference_solution is None:
                    # Compute ground truth for this dt
                    ground_truth = self.classical_solver.fft_ground_truth(u0, dt, nt)
                    reference_solution = ground_truth[-1]
            
            # Test each method
            for method in methods:
                try:
                    if method.startswith('neural_'):
                        if method == 'neural_lie':
                            solution = self.neural_splitting.lie_splitting(u0, dt, nt, ode_method)
                        elif method == 'neural_strang':
                            solution = self.neural_splitting.strang_splitting(u0, dt, nt, ode_method)
                        else:
                            continue
                    else:  # classical methods
                        from operator_splitting.operator_splitting_1d import OperatorSplittingMethods1D
                        classical_splitting = OperatorSplittingMethods1D(self.classical_solver)
                        
                        if method == 'classical_lie':
                            solution = classical_splitting.lie_splitting(u0, dt, nt)
                        elif method == 'classical_strang':
                            solution = classical_splitting.strang_splitting(u0, dt, nt)
                        else:
                            continue
                    
                    final_solution = solution[-1]
                    error_metrics = compute_error_metrics(final_solution, reference_solution)
                    convergence_results['errors'][method].append(error_metrics['l2_error'])
                    
                    print(f"  {method}: L2 error = {error_metrics['l2_error']:.2e}")
                    
                except Exception as e:
                    print(f"  {method} failed: {e}")
                    convergence_results['errors'][method].append(float('inf'))
        
        return convergence_results
    
    def solver_comparison(self,
                         u0: np.ndarray,
                         dt: float,
                         nt: int,
                         ode_methods: List[str] = None) -> Dict:
        """
        Compare different ODE solver methods for neural operators.
        
        Args:
            u0: Initial condition
            dt: Time step
            nt: Number of time steps
            ode_methods: List of ODE solver methods to test
            
        Returns:
            Solver comparison results
        """
        if ode_methods is None:
            ode_methods = ['euler', 'rk4', 'dopri5']
        
        print(f"Comparing ODE solvers: {ode_methods}")
        
        # Compute ground truth
        ground_truth = self.classical_solver.fft_ground_truth(u0, dt, nt)
        final_gt = ground_truth[-1]
        
        results = {
            'methods': ode_methods,
            'neural_lie': {},
            'neural_strang': {},
            'ground_truth': ground_truth
        }
        
        for method in ode_methods:
            print(f"\nTesting {method} solver...")
            
            try:
                # Test Lie splitting
                lie_solution = self.neural_splitting.lie_splitting(u0, dt, nt, method)
                lie_error = compute_error_metrics(lie_solution[-1], final_gt)
                results['neural_lie'][method] = {
                    'solution': lie_solution,
                    'error': lie_error
                }
                print(f"  Lie splitting: L2 = {lie_error['l2_error']:.2e}")
                
            except Exception as e:
                print(f"  Lie splitting with {method} failed: {e}")
                results['neural_lie'][method] = None
            
            try:
                # Test Strang splitting
                strang_solution = self.neural_splitting.strang_splitting(u0, dt, nt, method)
                strang_error = compute_error_metrics(strang_solution[-1], final_gt)
                results['neural_strang'][method] = {
                    'solution': strang_solution,
                    'error': strang_error
                }
                print(f"  Strang splitting: L2 = {strang_error['l2_error']:.2e}")
                
            except Exception as e:
                print(f"  Strang splitting with {method} failed: {e}")
                results['neural_strang'][method] = None
        
        return results


if __name__ == "__main__":
    print("Testing Neural Operator Splitting Methods...")
    
    # This is a placeholder test - in practice, you would load trained models
    print("Note: This test requires trained neural ODE models.")
    print("Run training.py first to train the models, then load them here.")
    
    # Example of how to use the framework:
    # from neural_ode_operators import create_neural_operators
    # 
    # # Load trained models
    # operators = create_neural_operators(nx=64, L=2*np.pi)
    # operators['advection'].load_state_dict(torch.load('models/advection_model.pth'))
    # operators['diffusion'].load_state_dict(torch.load('models/diffusion_model.pth'))
    # 
    # # Create splitting framework
    # neural_splitting = NeuralOperatorSplitting(
    #     operators['advection'], 
    #     operators['diffusion']
    # )
    # 
    # # Create comparison framework
    # comparison = ComparisonFramework(neural_splitting, 64, 2*np.pi, 1.0, 0.1)
    # 
    # # Test on initial condition
    # x = np.linspace(0, 2*np.pi, 64, endpoint=False)
    # u0 = np.sin(x)
    # 
    # # Compare methods
    # results = comparison.compare_methods(u0, dt=0.01, nt=50)
    
    print("Neural operator splitting framework ready!")