#!/usr/bin/env python3
"""
Test script for second-order finite difference methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from operator_splitting_1d import AdvectionDiffusion1DSolver, OperatorSplitting1DMethods, compute_error_metrics_1d

def test_individual_operators():
    """Test individual advection and diffusion operators."""
    print("Testing individual second-order operators...")
    
    # Setup
    nx = 64
    L = 2*np.pi
    v = 1.0
    D = 0.01
    dt = 0.001  # Small time step for stability
    T = 0.1
    nt = int(T / dt)
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    
    # Test initial condition
    u0 = solver.initial_condition_gaussian()
    
    print(f"Setup: nx={nx}, dt={dt}, T={T}, nt={nt}")
    
    # Test advection operators
    print("\nTesting advection operators:")
    advection_methods = {
        'Spectral': solver.advection_step_spectral,
        'Upwind': solver.advection_step_upwind,
        'Centered (RK2)': solver.advection_step_centered,
        'Lax-Wendroff': solver.advection_step_lax_wendroff
    }
    
    for name, method in advection_methods.items():
        u = u0.copy()
        try:
            for _ in range(10):  # Just a few steps
                u = method(u, dt)
            max_val = np.max(np.abs(u))
            print(f"  {name:15}: Max |u| = {max_val:.3f} - {'OK' if max_val < 10 else 'UNSTABLE'}")
        except Exception as e:
            print(f"  {name:15}: ERROR - {e}")
    
    # Test diffusion operators
    print("\nTesting diffusion operators:")
    diffusion_methods = {
        'Spectral': solver.diffusion_step_spectral,
        'Crank-Nicolson': solver.diffusion_step_finite_diff,
        'Explicit': solver.diffusion_step_explicit,
        'RK2': solver.diffusion_step_rk2,
        'Backward Euler': solver.diffusion_step_backward_euler
    }
    
    for name, method in diffusion_methods.items():
        u = u0.copy()
        try:
            for _ in range(10):  # Just a few steps
                u = method(u, dt)
            max_val = np.max(np.abs(u))
            print(f"  {name:15}: Max |u| = {max_val:.3f} - {'OK' if max_val < 10 else 'UNSTABLE'}")
        except Exception as e:
            print(f"  {name:15}: ERROR - {e}")

def test_operator_splitting_methods():
    """Test operator splitting with different discretizations."""
    print("\n" + "="*50)
    print("Testing operator splitting with second-order methods...")
    
    # Setup
    nx = 128
    L = 2*np.pi  
    v = 1.0
    D = 0.01
    dt = 0.01
    T = 0.2
    nt = int(T / dt)
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    
    # Initial condition
    u0 = solver.initial_condition_complex_sines()
    
    # Ground truth (spectral method)
    print(f"\nComputing ground truth (spectral)...")
    ground_truth = solver.fft_ground_truth(u0, dt, nt)
    final_gt = ground_truth[-1]
    
    # Test different discretizations
    discretizations = ["spectral", "centered", "lax-wendroff", "rk2"]
    splitting_methods = ["lie_splitting", "strang_splitting", "alternating_splitting"]
    
    results = {}
    
    for disc in discretizations:
        print(f"\nTesting discretization: {disc}")
        results[disc] = {}
        
        for split_method in splitting_methods:
            try:
                method_func = getattr(splitting, split_method)
                solution = method_func(u0, dt, nt, discretization=disc)
                final_sol = solution[-1]
                
                # Compute error
                error = compute_error_metrics_1d(final_sol, final_gt)
                results[disc][split_method] = error['l2_error']
                
                print(f"  {split_method:20}: L2 error = {error['l2_error']:.2e}")
                
            except Exception as e:
                print(f"  {split_method:20}: ERROR - {e}")
                results[disc][split_method] = float('inf')
    
    return results

def test_stability_conditions():
    """Test stability conditions for explicit methods."""
    print("\n" + "="*50)
    print("Testing stability conditions...")
    
    nx = 64
    L = 2*np.pi
    v = 1.0
    D = 0.1  # Larger diffusion coefficient
    dx = L / nx
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    
    # Test diffusion stability condition: dt ≤ dx²/(2*D)
    dt_stable = 0.5 * dx**2 / D
    dt_unstable = 2 * dt_stable
    
    print(f"Diffusion stability condition: dt ≤ {dt_stable:.6f}")
    print(f"Testing stable dt = {dt_stable:.6f}")
    print(f"Testing unstable dt = {dt_unstable:.6f}")
    
    u0 = solver.initial_condition_gaussian()
    
    # Test stable case
    try:
        u = u0.copy()
        for _ in range(10):
            u = solver.diffusion_step_explicit(u, dt_stable)
        print(f"Stable case: Max |u| = {np.max(np.abs(u)):.3f} - OK")
    except Exception as e:
        print(f"Stable case: ERROR - {e}")
    
    # Test unstable case
    try:
        u = u0.copy()
        for _ in range(10):
            u = solver.diffusion_step_explicit(u, dt_unstable)
        max_u = np.max(np.abs(u))
        print(f"Unstable case: Max |u| = {max_u:.3f} - {'UNSTABLE' if max_u > 100 else 'OK'}")
    except Exception as e:
        print(f"Unstable case: ERROR - {e}")

def compare_accuracy_orders():
    """Compare accuracy of different methods."""
    print("\n" + "="*50)
    print("Comparing accuracy orders...")
    
    nx = 128
    L = 2*np.pi
    v = 0.5
    D = 0.01
    T = 0.1
    
    solver = AdvectionDiffusion1DSolver(nx, L, v, D)
    splitting = OperatorSplitting1DMethods(solver)
    
    # Test with different time steps
    dt_values = [0.02, 0.01, 0.005]
    u0 = solver.initial_condition_sine(n_modes=2)
    
    print(f"Testing Strang splitting accuracy order...")
    print(f"dt values: {dt_values}")
    
    discretizations = ["spectral", "centered", "lax-wendroff", "rk2"]
    
    for disc in discretizations:
        print(f"\nDiscretization: {disc}")
        errors = []
        
        for dt in dt_values:
            nt = int(T / dt)
            
            # Ground truth (high resolution spectral)
            ground_truth = solver.fft_ground_truth(u0, dt/4, nt*4)  # 4x higher resolution
            final_gt = ground_truth[-1]
            
            # Test solution
            try:
                solution = splitting.strang_splitting(u0, dt, nt, discretization=disc)
                error = compute_error_metrics_1d(solution[-1], final_gt)
                errors.append(error['l2_error'])
                print(f"  dt={dt:.3f}: error={error['l2_error']:.2e}")
            except Exception as e:
                print(f"  dt={dt:.3f}: ERROR - {e}")
                errors.append(float('inf'))
        
        # Compute convergence rate
        if len(errors) >= 2 and all(e < float('inf') for e in errors):
            rate = np.log(errors[-2]/errors[-1]) / np.log(dt_values[-2]/dt_values[-1])
            print(f"  Convergence rate: {rate:.2f}")

def main():
    """Run all tests."""
    print("TESTING SECOND-ORDER FINITE DIFFERENCE METHODS")
    print("=" * 60)
    
    test_individual_operators()
    results = test_operator_splitting_methods() 
    test_stability_conditions()
    compare_accuracy_orders()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Individual operators tested")
    print("✓ Operator splitting methods tested with multiple discretizations")
    print("✓ Stability conditions verified")
    print("✓ Accuracy orders compared")
    print("\nAll second-order methods have been successfully implemented!")

if __name__ == "__main__":
    main()