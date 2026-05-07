#!/usr/bin/env python3
"""
Test script to verify that prediction saving modifications work correctly.
"""

import numpy as np
import os
import sys
from operator_splitting_1d import analyze_1d_convergence, analyze_saved_predictions

def test_prediction_saving():
    """Test the prediction saving functionality."""
    print("Testing prediction saving functionality...")
    
    # Small test case
    nx = 64
    L = 2*np.pi
    v = 1.0
    D = 0.01
    T = 0.1  # Short time for quick test
    dt_values = [0.02, 0.01]  # Just two values for quick test
    
    # Test directory
    test_dir = "test_predictions"
    
    # Run convergence analysis with prediction saving
    print("\nRunning convergence analysis with prediction saving...")
    results = analyze_1d_convergence(
        nx=nx, L=L, v=v, D=D, T=T, 
        dt_values=dt_values,
        save_predictions=True,
        output_dir=test_dir
    )
    
    # Check that files were created
    print("\nChecking that prediction files were created...")
    if not os.path.exists(test_dir):
        print("ERROR: Test directory was not created!")
        return False
    
    # Check each case directory
    for dt in dt_values:
        case_dir = os.path.join(test_dir, f"case_dt_{dt:.6f}")
        if not os.path.exists(case_dir):
            print(f"ERROR: Case directory {case_dir} was not created!")
            return False
        
        # Check that all required files exist
        required_files = [
            "x_grid.npy",
            "time_points.npy", 
            "initial_condition.npy",
            "ground_truth.npy",
            "lie_solution.npy",
            "strang_solution.npy",
            "alternating_solution.npy",
            "metadata.json"
        ]
        
        for filename in required_files:
            filepath = os.path.join(case_dir, filename)
            if not os.path.exists(filepath):
                print(f"ERROR: Required file {filepath} was not created!")
                return False
        
        print(f"✓ All files created for dt={dt}")
    
    # Test loading and analyzing predictions
    print("\nTesting prediction analysis...")
    analysis = analyze_saved_predictions(test_dir, method="strang")
    
    if analysis is None:
        print("ERROR: Prediction analysis failed!")
        return False
    
    print(f"✓ Analysis completed for {len(analysis)} cases")
    
    # Test accessing predictions from results dict
    print("\nTesting in-memory prediction access...")
    if 'predictions' not in results:
        print("ERROR: Predictions not stored in results dict!")
        return False
    
    for dt in dt_values:
        case_key = f"dt_{dt:.6f}"
        if case_key not in results['predictions']:
            print(f"ERROR: Case {case_key} not found in predictions!")
            return False
        
        case_data = results['predictions'][case_key]
        required_keys = ['dt', 'nt', 'x', 'time_points', 'ground_truth', 
                        'lie_solution', 'strang_solution', 'alternating_solution']
        
        for key in required_keys:
            if key not in case_data:
                print(f"ERROR: Key {key} missing from case data!")
                return False
        
        print(f"✓ In-memory data complete for dt={dt}")
    
    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("✓ Cleanup complete")
    
    print("\n🎉 All tests passed! Prediction saving is working correctly.")
    return True

if __name__ == "__main__":
    success = test_prediction_saving()
    sys.exit(0 if success else 1)