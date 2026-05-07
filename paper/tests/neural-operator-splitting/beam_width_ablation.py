"""
Beam Width Ablation Study Script

This script runs experiments with different beam widths to study the effect
of beam width on performance vs computational cost.
"""

import os
import sys
import subprocess
import json
import time
from typing import List, Dict


def run_beam_width_ablation(checkpoint_path: str, beam_widths: List[int] = [1, 3, 5, 8],
                          timestamps_per_op: int = 3, num_runs: int = 3,
                          base_output_dir: str = "./results/beam_width_ablation"):
    """
    Run ablation study on beam width.
    
    Args:
        checkpoint_path: Path to DISCO checkpoint
        beam_widths: List of beam widths to test (1 = greedy baseline)
        timestamps_per_op: Number of timestamps per operator
        num_runs: Number of runs per configuration
        base_output_dir: Base output directory
    """
    
    print("=" * 70)
    print("BEAM WIDTH ABLATION STUDY")
    print("=" * 70)
    print(f"Testing beam widths: {beam_widths}")
    print(f"Timestamps per operator: {timestamps_per_op}")
    print(f"Runs per configuration: {num_runs}")
    print("=" * 70)
    
    results_summary = {}
    
    for beam_width in beam_widths:
        print(f"\n{'='*50}")
        print(f"TESTING BEAM WIDTH = {beam_width}")
        print(f"{'='*50}")
        
        # Determine selection method
        selection_method = "greedy" if beam_width == 1 else "beam_search"
        
        # Create output directory for this configuration
        config_dir = f"beam_width_{beam_width}"
        output_dir = os.path.join(base_output_dir, config_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "main_with_beam_search.py",
            "--checkpoint-path", checkpoint_path,
            "--selection-method", selection_method,
            "--beam-width", str(beam_width),
            "--timestamps-per-operator", str(timestamps_per_op),
            "--num-runs", str(num_runs),
            "--output-dir", output_dir,
            "--num-operators", "20",  # Use 20 operators for better selection
            "--max-operators", "5",
            "--finetune-epochs", "5",  # Quick finetuning for ablation
            "--seed", "42"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            end_time = time.time()
            runtime = end_time - start_time
            
            if result.returncode == 0:
                print(f"✓ Beam width {beam_width} completed successfully in {runtime:.1f}s")
                
                # Try to load aggregate statistics
                stats_files = []
                for root, dirs, files in os.walk(output_dir):
                    if "aggregate_stats.json" in files:
                        stats_files.append(os.path.join(root, "aggregate_stats.json"))
                
                if stats_files:
                    with open(stats_files[0], 'r') as f:
                        stats = json.load(f)
                    
                    results_summary[beam_width] = {
                        'method': selection_method,
                        'beam_width': beam_width,
                        'runtime_seconds': runtime,
                        'final_error_mean': stats['final_error']['mean'],
                        'final_error_std': stats['final_error']['std'],
                        'final_error_min': stats['final_error']['min'],
                        'composition_length_mean': stats['composition_length']['mean'],
                        'validation_error_mean': stats['validation_error']['mean']
                    }
                    
                    print(f"  Final error: {stats['final_error']['mean']:.6f} ± {stats['final_error']['std']:.6f}")
                    print(f"  Best final error: {stats['final_error']['min']:.6f}")
                    print(f"  Avg composition length: {stats['composition_length']['mean']:.1f}")
                else:
                    print(f"  Warning: Could not find aggregate statistics")
                    results_summary[beam_width] = {
                        'method': selection_method,
                        'beam_width': beam_width,
                        'runtime_seconds': runtime,
                        'error': 'Statistics not found'
                    }
            else:
                print(f"✗ Beam width {beam_width} failed")
                print(f"Error output: {result.stderr}")
                results_summary[beam_width] = {
                    'method': selection_method,
                    'beam_width': beam_width,
                    'runtime_seconds': runtime,
                    'error': result.stderr
                }
                
        except Exception as e:
            print(f"✗ Exception running beam width {beam_width}: {e}")
            results_summary[beam_width] = {
                'method': selection_method,
                'beam_width': beam_width,
                'error': str(e)
            }
    
    # Save summary results
    summary_file = os.path.join(base_output_dir, "beam_width_ablation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Beam Width':<12} {'Method':<12} {'Runtime(s)':<12} {'Final Error':<15} {'Best Error':<12} {'Comp Length':<12}")
    print("-"*70)
    
    for beam_width in beam_widths:
        if beam_width in results_summary:
            result = results_summary[beam_width]
            if 'error' not in result or 'final_error_mean' in result:
                print(f"{beam_width:<12} {result.get('method', 'N/A'):<12} "
                      f"{result.get('runtime_seconds', 0):<12.1f} "
                      f"{result.get('final_error_mean', 0):<15.6f} "
                      f"{result.get('final_error_min', 0):<12.6f} "
                      f"{result.get('composition_length_mean', 0):<12.1f}")
            else:
                print(f"{beam_width:<12} {'FAILED':<12} {result.get('runtime_seconds', 0):<12.1f} {'N/A':<15} {'N/A':<12} {'N/A':<12}")
    
    print(f"\nDetailed results saved to: {summary_file}")
    print("="*70)
    
    return results_summary


def analyze_results(results_summary: Dict):
    """Analyze ablation results and provide insights."""
    print("\nANALYSIS:")
    print("-"*40)
    
    # Find best performing configuration
    valid_results = {k: v for k, v in results_summary.items() 
                    if 'final_error_min' in v}
    
    if valid_results:
        best_config = min(valid_results.items(), key=lambda x: x[1]['final_error_min'])
        print(f"Best performance: Beam width {best_config[0]} with error {best_config[1]['final_error_min']:.6f}")
        
        # Runtime analysis
        greedy_runtime = results_summary.get(1, {}).get('runtime_seconds', 0)
        if greedy_runtime > 0:
            print(f"Runtime comparison (vs greedy baseline):")
            for beam_width, result in valid_results.items():
                if beam_width != 1:
                    speedup_factor = result['runtime_seconds'] / greedy_runtime
                    print(f"  Beam width {beam_width}: {speedup_factor:.1f}x slower")
        
        # Performance improvement
        greedy_error = results_summary.get(1, {}).get('final_error_min', float('inf'))
        if greedy_error < float('inf'):
            print(f"Performance improvement (vs greedy baseline):")
            for beam_width, result in valid_results.items():
                if beam_width != 1:
                    improvement = (greedy_error - result['final_error_min']) / greedy_error * 100
                    print(f"  Beam width {beam_width}: {improvement:+.1f}% improvement")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Beam Width Ablation Study')
    parser.add_argument('--checkpoint-path', required=True, help='Path to DISCO checkpoint')
    parser.add_argument('--beam-widths', nargs='+', type=int, default=[1, 3, 5, 8], 
                       help='Beam widths to test')
    parser.add_argument('--timestamps-per-op', type=int, default=3, 
                       help='Timestamps per operator')
    parser.add_argument('--num-runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--output-dir', default='./results/beam_width_ablation', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    results = run_beam_width_ablation(
        checkpoint_path=args.checkpoint_path,
        beam_widths=args.beam_widths,
        timestamps_per_op=args.timestamps_per_op,
        num_runs=args.num_runs,
        base_output_dir=args.output_dir
    )
    
    analyze_results(results)