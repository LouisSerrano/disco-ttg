import numpy as np
import matplotlib.pyplot as plt
from operator_splitting import AdvectionDiffusionSolver, OperatorSplittingMethods, compute_error_metrics
import time
from typing import Dict, List, Tuple


class OperatorSplittingAnalysis:
    """Comprehensive analysis framework for operator splitting methods."""
    
    def __init__(self, nx: int = 64, ny: int = 64, Lx: float = 2*np.pi, Ly: float = 2*np.pi):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        
    def convergence_study(self, vx: float = 1.0, vy: float = 0.5, D: float = 0.01,
                         T: float = 0.5, dt_values: List[float] = None) -> Dict:
        """Study convergence of splitting methods with respect to time step."""
        if dt_values is None:
            dt_values = [0.02, 0.01, 0.005, 0.0025]
            
        results = {
            'dt_values': dt_values,
            'lie_errors': [],
            'strang_errors': [],
            'advection_advection_errors': [],
            'diffusion_diffusion_errors': []
        }
        
        print("Convergence Study:")
        print("=" * 50)
        
        for dt in dt_values:
            nt = int(T / dt)
            solver = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, vx, vy, D)
            splitting = OperatorSplittingMethods(solver)
            
            u0 = solver.initial_condition_gaussian()
            
            # Ground truth
            ground_truth = solver.fft_ground_truth(u0, dt, nt)
            final_gt = ground_truth[-1]
            
            # Splitting methods
            lie_solution = splitting.lie_splitting(u0, dt, nt)
            strang_solution = splitting.strang_splitting(u0, dt, nt)
            aa_solution = splitting.advection_advection_splitting(u0, dt, nt)
            dd_solution = splitting.diffusion_diffusion_splitting(u0, dt, nt)
            
            # Compute errors
            lie_error = compute_error_metrics(lie_solution[-1], final_gt)
            strang_error = compute_error_metrics(strang_solution[-1], final_gt)
            aa_error = compute_error_metrics(aa_solution[-1], final_gt)
            dd_error = compute_error_metrics(dd_solution[-1], final_gt)
            
            results['lie_errors'].append(lie_error['l2_error'])
            results['strang_errors'].append(strang_error['l2_error'])
            results['advection_advection_errors'].append(aa_error['l2_error'])
            results['diffusion_diffusion_errors'].append(dd_error['l2_error'])
            
            print(f"dt = {dt:.4f}: Lie={lie_error['l2_error']:.2e}, "
                  f"Strang={strang_error['l2_error']:.2e}, "
                  f"A-A={aa_error['l2_error']:.2e}, D-D={dd_error['l2_error']:.2e}")
        
        return results
    
    def parameter_study(self, dt: float = 0.01, T: float = 0.5) -> Dict:
        """Study behavior across different parameter regimes."""
        parameter_sets = [
            {'vx': 1.0, 'vy': 0.5, 'D': 0.01, 'name': 'Balanced'},
            {'vx': 5.0, 'vy': 2.5, 'D': 0.01, 'name': 'Advection dominated'},
            {'vx': 0.2, 'vy': 0.1, 'D': 0.1, 'name': 'Diffusion dominated'},
            {'vx': 10.0, 'vy': 0.0, 'D': 0.001, 'name': 'Pure advection (x)'},
            {'vx': 0.0, 'vy': 0.0, 'D': 0.05, 'name': 'Pure diffusion'}
        ]
        
        results = {
            'parameter_sets': parameter_sets,
            'errors': {},
            'times': {}
        }
        
        nt = int(T / dt)
        print("\nParameter Study:")
        print("=" * 50)
        
        for params in parameter_sets:
            solver = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, 
                                            params['vx'], params['vy'], params['D'])
            splitting = OperatorSplittingMethods(solver)
            
            u0 = solver.initial_condition_gaussian()
            
            # Time and solve each method
            methods = {
                'ground_truth': lambda: solver.fft_ground_truth(u0, dt, nt),
                'lie': lambda: splitting.lie_splitting(u0, dt, nt),
                'strang': lambda: splitting.strang_splitting(u0, dt, nt),
                'advection_advection': lambda: splitting.advection_advection_splitting(u0, dt, nt),
                'diffusion_diffusion': lambda: splitting.diffusion_diffusion_splitting(u0, dt, nt)
            }
            
            param_errors = {}
            param_times = {}
            
            # Ground truth first
            start_time = time.time()
            ground_truth = methods['ground_truth']()
            param_times['ground_truth'] = time.time() - start_time
            final_gt = ground_truth[-1]
            
            # Other methods
            for method_name, method_func in methods.items():
                if method_name == 'ground_truth':
                    continue
                    
                start_time = time.time()
                solution = method_func()
                param_times[method_name] = time.time() - start_time
                
                error_metrics = compute_error_metrics(solution[-1], final_gt)
                param_errors[method_name] = error_metrics
            
            results['errors'][params['name']] = param_errors
            results['times'][params['name']] = param_times
            
            print(f"\n{params['name']} (vx={params['vx']}, vy={params['vy']}, D={params['D']}):")
            for method in ['lie', 'strang', 'advection_advection', 'diffusion_diffusion']:
                error = param_errors[method]['l2_error']
                runtime = param_times[method]
                print(f"  {method:20}: L2={error:.2e}, Time={runtime:.4f}s")
        
        return results
    
    def pure_operator_comparison(self, dt: float = 0.01, T: float = 0.5):
        """Compare splitting methods for pure advection and pure diffusion cases."""
        print("\nPure Operator Comparison:")
        print("=" * 50)
        
        nt = int(T / dt)
        
        # Pure advection case
        print("\n1. Pure Advection (D=0):")
        solver_adv = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, 1.0, 0.5, 0.0)
        splitting_adv = OperatorSplittingMethods(solver_adv)
        u0 = solver_adv.initial_condition_gaussian()
        
        gt_adv = solver_adv.fft_ground_truth(u0, dt, nt)
        aa_adv = splitting_adv.advection_advection_splitting(u0, dt, nt)
        
        aa_error_adv = compute_error_metrics(aa_adv[-1], gt_adv[-1])
        print(f"  Advection-Advection splitting error: L2={aa_error_adv['l2_error']:.2e}")
        
        # Pure diffusion case
        print("\n2. Pure Diffusion (v=0):")
        solver_diff = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, 0.0, 0.0, 0.05)
        splitting_diff = OperatorSplittingMethods(solver_diff)
        u0 = solver_diff.initial_condition_gaussian()
        
        gt_diff = solver_diff.fft_ground_truth(u0, dt, nt)
        dd_diff = splitting_diff.diffusion_diffusion_splitting(u0, dt, nt)
        
        dd_error_diff = compute_error_metrics(dd_diff[-1], gt_diff[-1])
        print(f"  Diffusion-Diffusion splitting error: L2={dd_error_diff['l2_error']:.2e}")
    
    def stability_analysis(self, vx: float = 2.0, vy: float = 1.0, D: float = 0.01):
        """Analyze stability limits of different methods."""
        print("\nStability Analysis:")
        print("=" * 50)
        
        # Test a range of time steps
        dt_values = np.logspace(-3, -1, 20)  # 0.001 to 0.1
        T = 1.0
        
        stable_methods = {
            'lie': [],
            'strang': [],
            'advection_advection': [],
            'diffusion_diffusion': []
        }
        
        solver = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, vx, vy, D)
        splitting = OperatorSplittingMethods(solver)
        u0 = solver.initial_condition_gaussian()
        
        for dt in dt_values:
            nt = int(T / dt)
            if nt < 10:  # Skip very small number of steps
                continue
                
            try:
                # Test each method for stability (solution shouldn't blow up)
                methods_to_test = {
                    'lie': lambda: splitting.lie_splitting(u0, dt, nt),
                    'strang': lambda: splitting.strang_splitting(u0, dt, nt),
                    'advection_advection': lambda: splitting.advection_advection_splitting(u0, dt, nt),
                    'diffusion_diffusion': lambda: splitting.diffusion_diffusion_splitting(u0, dt, nt)
                }
                
                for method_name, method_func in methods_to_test.items():
                    solution = method_func()
                    final_solution = solution[-1]
                    
                    # Check for stability (no NaN, not too large)
                    if (not np.any(np.isnan(final_solution)) and 
                        np.max(np.abs(final_solution)) < 100):
                        stable_methods[method_name].append(dt)
                        
            except Exception as e:
                # Method became unstable
                continue
        
        print("Maximum stable time steps:")
        for method, dt_stable in stable_methods.items():
            if dt_stable:
                max_dt = max(dt_stable)
                print(f"  {method:20}: dt_max ≈ {max_dt:.4f}")
            else:
                print(f"  {method:20}: Unstable for all tested dt")
    
    def create_visualizations(self, vx: float = 1.0, vy: float = 0.5, D: float = 0.01,
                            dt: float = 0.01, T: float = 0.5):
        """Create visualization plots comparing different methods."""
        nt = int(T / dt)
        solver = AdvectionDiffusionSolver(self.nx, self.ny, self.Lx, self.Ly, vx, vy, D)
        splitting = OperatorSplittingMethods(solver)
        
        u0 = solver.initial_condition_gaussian()
        
        # Compute all solutions
        print("\nGenerating visualizations...")
        ground_truth = solver.fft_ground_truth(u0, dt, nt)
        lie_solution = splitting.lie_splitting(u0, dt, nt)
        strang_solution = splitting.strang_splitting(u0, dt, nt)
        
        # Plot comparison at final time
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Solutions
        vmin, vmax = 0, np.max(ground_truth[-1])
        
        im1 = axes[0,0].imshow(ground_truth[-1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0,0].set_title('Ground Truth (FFT)')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(lie_solution[-1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0,1].set_title('Lie Splitting')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].imshow(strang_solution[-1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0,2].set_title('Strang Splitting')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Error plots
        lie_error = lie_solution[-1] - ground_truth[-1]
        strang_error = strang_solution[-1] - ground_truth[-1]
        error_max = max(np.max(np.abs(lie_error)), np.max(np.abs(strang_error)))
        
        im4 = axes[1,0].imshow(lie_error, vmin=-error_max, vmax=error_max, cmap='RdBu_r')
        axes[1,0].set_title('Lie Splitting Error')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].imshow(strang_error, vmin=-error_max, vmax=error_max, cmap='RdBu_r')
        axes[1,1].set_title('Strang Splitting Error')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('y')
        plt.colorbar(im5, ax=axes[1,1])
        
        # Time evolution of L2 error
        time_points = np.linspace(0, T, nt+1)
        lie_errors_time = [compute_error_metrics(lie_solution[i], ground_truth[i])['l2_error'] 
                          for i in range(len(ground_truth))]
        strang_errors_time = [compute_error_metrics(strang_solution[i], ground_truth[i])['l2_error'] 
                             for i in range(len(ground_truth))]
        
        axes[1,2].semilogy(time_points, lie_errors_time, 'o-', label='Lie Splitting')
        axes[1,2].semilogy(time_points, strang_errors_time, 's-', label='Strang Splitting')
        axes[1,2].set_xlabel('Time')
        axes[1,2].set_ylabel('L2 Error')
        axes[1,2].set_title('Error Evolution')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig('operator_splitting_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    """Run comprehensive analysis of operator splitting methods."""
    print("Comprehensive Operator Splitting Analysis")
    print("=" * 60)
    
    # Initialize analysis framework
    analysis = OperatorSplittingAnalysis(nx=64, ny=64)
    
    # 1. Convergence study
    convergence_results = analysis.convergence_study()
    
    # 2. Parameter study
    parameter_results = analysis.parameter_study()
    
    # 3. Pure operator comparison
    analysis.pure_operator_comparison()
    
    # 4. Stability analysis
    analysis.stability_analysis()
    
    # 5. Create visualizations
    analysis.create_visualizations()
    
    print("\nAnalysis complete! Check 'operator_splitting_comparison.png' for visualizations.")


if __name__ == "__main__":
    main()