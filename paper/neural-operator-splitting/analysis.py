"""
Analysis and Visualization for Neural Operator Splitting

This module provides comprehensive analysis and visualization tools for 
neural operator splitting results, including error analysis, convergence plots,
trajectory comparisons, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class NeuralSplittingAnalyzer:
    """Comprehensive analyzer for neural operator splitting results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing test results
        """
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Load results if available
        self.results = self._load_results()
        
    def _load_results(self) -> Optional[Dict]:
        """Load test results from JSON file."""
        results_file = os.path.join(self.results_dir, 'full_pipeline_results.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading results: {e}")
        return None
    
    def plot_training_convergence(self, 
                                training_results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training convergence for both advection and diffusion models.
        
        Args:
            training_results: Training results dictionary
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Advection training
        adv_history = training_results['advection']['history']
        axes[0, 0].plot(adv_history['train_losses'], label='Train', linewidth=2)
        axes[0, 0].plot(adv_history['val_losses'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Advection Model Training')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diffusion training
        diff_history = training_results['diffusion']['history']
        axes[0, 1].plot(diff_history['train_losses'], label='Train', linewidth=2)
        axes[0, 1].plot(diff_history['val_losses'], label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Diffusion Model Training')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training comparison
        axes[1, 0].plot(adv_history['val_losses'], label='Advection', linewidth=2)
        axes[1, 0].plot(diff_history['val_losses'], label='Diffusion', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss Comparison')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final losses bar plot
        final_losses = [
            adv_history['best_val_loss'],
            diff_history['best_val_loss']
        ]
        axes[1, 1].bar(['Advection', 'Diffusion'], final_losses, 
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1, 1].set_ylabel('Best Validation Loss')
        axes[1, 1].set_title('Final Model Performance')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add text annotations
        for i, loss in enumerate(final_losses):
            axes[1, 1].text(i, loss * 1.1, f'{loss:.2e}', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_method_comparison(self,
                             splitting_results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different splitting methods across test cases.
        
        Args:
            splitting_results: Results from splitting method comparison
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract error data
        methods = []
        test_cases = []
        l2_errors = []
        linf_errors = []
        
        for case_name, case_results in splitting_results.items():
            if 'errors' not in case_results:
                continue
                
            for method_name, error_dict in case_results['errors'].items():
                if error_dict is not None:
                    methods.append(method_name)
                    test_cases.append(case_name)
                    l2_errors.append(error_dict['l2_error'])
                    linf_errors.append(error_dict['linf_error'])
        
        if not methods:
            print("No valid error data found for plotting")
            return plt.figure()
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Method': methods,
            'Test Case': test_cases,
            'L2 Error': l2_errors,
            'L∞ Error': linf_errors
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # L2 Error comparison by method
        method_order = ['ground_truth', 'classical_lie', 'classical_strang', 'neural_lie', 'neural_strang']
        method_order = [m for m in method_order if m in df['Method'].unique()]
        
        sns.boxplot(data=df, x='Method', y='L2 Error', order=method_order, ax=axes[0, 0])
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('L2 Error Distribution by Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # L∞ Error comparison by method
        sns.boxplot(data=df, x='Method', y='L∞ Error', order=method_order, ax=axes[0, 1])
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('L∞ Error Distribution by Method')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Heatmap of L2 errors
        pivot_l2 = df.pivot(index='Test Case', columns='Method', values='L2 Error')
        pivot_l2 = pivot_l2.reindex(columns=[m for m in method_order if m in pivot_l2.columns])
        
        sns.heatmap(np.log10(pivot_l2), annot=True, fmt='.1f', cmap='viridis', 
                   ax=axes[1, 0], cbar_kws={'label': 'log10(L2 Error)'})
        axes[1, 0].set_title('L2 Error Heatmap (log scale)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Method performance ranking
        method_means = df.groupby('Method')['L2 Error'].mean().sort_values()
        axes[1, 1].barh(range(len(method_means)), method_means.values, 
                       color='lightblue', alpha=0.7)
        axes[1, 1].set_yticks(range(len(method_means)))
        axes[1, 1].set_yticklabels(method_means.index)
        axes[1, 1].set_xlabel('Mean L2 Error')
        axes[1, 1].set_title('Average Method Performance')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_study(self,
                             convergence_results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot dt convergence study results.
        
        Args:
            convergence_results: Convergence study results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'errors' not in convergence_results:
            print("No convergence data available")
            return plt.figure()
        
        dt_values = convergence_results['dt_values']
        errors = convergence_results['errors']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main convergence plot
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (method, error_list) in enumerate(errors.items()):
            # Filter out infinite errors
            valid_errors = [(dt, err) for dt, err in zip(dt_values, error_list) 
                           if err != float('inf') and not np.isnan(err)]
            
            if valid_errors:
                valid_dt, valid_err = zip(*valid_errors)
                axes[0, 0].loglog(valid_dt, valid_err, 
                                 color=colors[i % len(colors)], 
                                 marker=markers[i % len(markers)], 
                                 label=method, linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('Time Step (dt)')
        axes[0, 0].set_ylabel('L2 Error')
        axes[0, 0].set_title('Convergence Study: Error vs Time Step')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add theoretical convergence lines
        dt_theory = np.array(dt_values)
        axes[0, 0].loglog(dt_theory, dt_theory, '--', color='gray', alpha=0.7, label='O(dt)')
        axes[0, 0].loglog(dt_theory, dt_theory**2, '--', color='lightgray', alpha=0.7, label='O(dt²)')
        
        # Convergence rates
        rates = {}
        for method, error_list in errors.items():
            valid_data = [(dt, err) for dt, err in zip(dt_values, error_list) 
                         if err != float('inf') and not np.isnan(err)]
            
            if len(valid_data) >= 2:
                dt_vals, err_vals = zip(*valid_data)
                # Fit line in log space to get slope (convergence rate)
                log_dt = np.log(dt_vals)
                log_err = np.log(err_vals)
                slope, intercept = np.polyfit(log_dt, log_err, 1)
                rates[method] = slope
        
        # Plot convergence rates
        if rates:
            methods_list = list(rates.keys())
            rates_list = list(rates.values())
            
            bars = axes[0, 1].bar(methods_list, rates_list, 
                                 color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(rates_list)],
                                 alpha=0.7)
            axes[0, 1].set_ylabel('Convergence Rate')
            axes[0, 1].set_title('Estimated Convergence Rates')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='First Order')
            axes[0, 1].axhline(y=2, color='lightgray', linestyle='--', alpha=0.7, label='Second Order')
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates_list):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Error reduction factor
        error_reduction = {}
        for method, error_list in errors.items():
            valid_errors = [err for err in error_list if err != float('inf') and not np.isnan(err)]
            if len(valid_errors) >= 2:
                reduction_factors = [valid_errors[i]/valid_errors[i+1] for i in range(len(valid_errors)-1)]
                error_reduction[method] = reduction_factors
        
        # Plot error reduction factors
        for i, (method, reductions) in enumerate(error_reduction.items()):
            if reductions:
                x_pos = range(1, len(reductions) + 1)
                axes[1, 0].plot(x_pos, reductions, 
                               color=colors[i % len(colors)], 
                               marker=markers[i % len(markers)], 
                               label=method, linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Refinement Step')
        axes[1, 0].set_ylabel('Error Reduction Factor')
        axes[1, 0].set_title('Error Reduction with dt Refinement')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='2x Reduction')
        axes[1, 0].axhline(y=4, color='lightgray', linestyle='--', alpha=0.7, label='4x Reduction')
        
        # Summary table
        axes[1, 1].axis('off')
        
        # Create summary text
        summary_text = "CONVERGENCE STUDY SUMMARY\\n" + "="*30 + "\\n\\n"
        summary_text += f"dt values tested: {len(dt_values)}\\n"
        summary_text += f"Range: {min(dt_values):.5f} to {max(dt_values):.5f}\\n\\n"
        
        summary_text += "CONVERGENCE RATES:\\n" + "-"*20 + "\\n"
        for method, rate in rates.items():
            summary_text += f"{method:15}: {rate:.2f}\\n"
        
        summary_text += "\\nMETHOD RANKING (by final error):\\n" + "-"*30 + "\\n"
        final_errors = {method: error_list[-1] for method, error_list in errors.items() 
                       if error_list[-1] != float('inf')}
        ranked_methods = sorted(final_errors.items(), key=lambda x: x[1])
        
        for i, (method, error) in enumerate(ranked_methods):
            summary_text += f"{i+1}. {method:15}: {error:.2e}\\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_solver_comparison(self,
                             solver_results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different ODE solvers.
        
        Args:
            solver_results: Solver comparison results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'neural_lie' not in solver_results:
            print("No solver comparison data available")
            return plt.figure()
        
        methods = solver_results.get('methods', [])
        neural_lie = solver_results['neural_lie']
        neural_strang = solver_results['neural_strang']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract errors for each solver
        lie_errors = []
        strang_errors = []
        solver_names = []
        
        for method in methods:
            if method in neural_lie and neural_lie[method] is not None:
                lie_errors.append(neural_lie[method]['error']['l2_error'])
                solver_names.append(method)
            if method in neural_strang and neural_strang[method] is not None:
                strang_errors.append(neural_strang[method]['error']['l2_error'])
        
        # Bar plot comparison
        x = np.arange(len(solver_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, lie_errors, width, label='Lie Splitting', 
                      color='skyblue', alpha=0.7)
        axes[0, 0].bar(x + width/2, strang_errors, width, label='Strang Splitting', 
                      color='lightcoral', alpha=0.7)
        
        axes[0, 0].set_xlabel('ODE Solver')
        axes[0, 0].set_ylabel('L2 Error')
        axes[0, 0].set_title('Neural Operator Splitting: Solver Comparison')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(solver_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error ratio (Lie/Strang)
        error_ratios = [lie/strang for lie, strang in zip(lie_errors, strang_errors)]
        
        axes[0, 1].bar(solver_names, error_ratios, color='lightgreen', alpha=0.7)
        axes[0, 1].set_xlabel('ODE Solver')
        axes[0, 1].set_ylabel('Error Ratio (Lie/Strang)')
        axes[0, 1].set_title('Lie vs Strang Error Ratio')
        axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance ranking
        performance_data = []
        for method in solver_names:
            lie_err = neural_lie[method]['error']['l2_error']
            strang_err = neural_strang[method]['error']['l2_error']
            avg_err = (lie_err + strang_err) / 2
            performance_data.append((method, avg_err, lie_err, strang_err))
        
        # Sort by average error
        performance_data.sort(key=lambda x: x[1])
        
        methods_sorted = [x[0] for x in performance_data]
        avg_errors = [x[1] for x in performance_data]
        
        axes[1, 0].barh(methods_sorted, avg_errors, color='gold', alpha=0.7)
        axes[1, 0].set_xlabel('Average L2 Error')
        axes[1, 0].set_ylabel('ODE Solver')
        axes[1, 0].set_title('Solver Ranking (by Average Error)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        axes[1, 1].axis('off')
        
        summary_text = "SOLVER COMPARISON SUMMARY\\n" + "="*25 + "\\n\\n"
        summary_text += f"Solvers tested: {len(methods)}\\n"
        summary_text += f"Methods: {', '.join(methods)}\\n\\n"
        
        summary_text += "PERFORMANCE RANKING:\\n" + "-"*20 + "\\n"
        for i, (method, avg_err, lie_err, strang_err) in enumerate(performance_data):
            summary_text += f"{i+1}. {method:8}\\n"
            summary_text += f"   Avg: {avg_err:.2e}\\n"
            summary_text += f"   Lie: {lie_err:.2e}\\n"
            summary_text += f"   Str: {strang_err:.2e}\\n\\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self, results: Dict) -> str:
        """
        Create a comprehensive analysis report with all visualizations.
        
        Args:
            results: Complete test results dictionary
            
        Returns:
            Path to the generated report directory
        """
        print("Creating comprehensive analysis report...")
        
        report_dir = os.path.join(self.figures_dir, 'comprehensive_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Training analysis
        if 'training' in results:
            print("  Analyzing training results...")
            training_fig = self.plot_training_convergence(
                results['training']['results'],
                save_path=os.path.join(report_dir, 'training_analysis.png')
            )
            plt.close(training_fig)
        
        # 2. Method comparison
        if 'splitting_tests' in results:
            print("  Analyzing splitting methods...")
            methods_fig = self.plot_method_comparison(
                results['splitting_tests'],
                save_path=os.path.join(report_dir, 'methods_comparison.png')
            )
            plt.close(methods_fig)
        
        # 3. Convergence study
        if 'convergence_study' in results:
            print("  Analyzing convergence study...")
            conv_fig = self.plot_convergence_study(
                results['convergence_study'],
                save_path=os.path.join(report_dir, 'convergence_study.png')
            )
            plt.close(conv_fig)
        
        # 4. Solver comparison
        if 'solver_comparison' in results:
            print("  Analyzing solver comparison...")
            solver_fig = self.plot_solver_comparison(
                results['solver_comparison'],
                save_path=os.path.join(report_dir, 'solver_comparison.png')
            )
            plt.close(solver_fig)
        
        # 5. Create HTML report
        self._create_html_report(results, report_dir)
        
        print(f"Comprehensive report created: {report_dir}")
        return report_dir
    
    def _create_html_report(self, results: Dict, report_dir: str):
        """Create an HTML report with all figures and analysis."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Operator Splitting Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; }}
                h2 {{ color: #A23B72; border-bottom: 1px solid #A23B72; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .figure {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2E86AB; }}
            </style>
        </head>
        <body>
            <h1>Neural Operator Splitting Analysis Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive analysis of neural operator splitting methods 
                for advection-diffusion equations. The study compares neural network-based operator 
                splitting with classical methods across various test cases, time step sizes, and 
                ODE solvers.</p>
            </div>
        """
        
        # Add training analysis section
        if 'training' in results:
            html_content += """
            <h2>1. Training Analysis</h2>
            <p>Analysis of neural ODE training convergence for advection and diffusion operators:</p>
            <div class="figure">
                <img src="training_analysis.png" alt="Training Analysis">
            </div>
            """
        
        # Add method comparison section
        if 'splitting_tests' in results:
            html_content += """
            <h2>2. Splitting Methods Comparison</h2>
            <p>Comparison of neural and classical operator splitting methods across different test cases:</p>
            <div class="figure">
                <img src="methods_comparison.png" alt="Methods Comparison">
            </div>
            """
        
        # Add convergence study section
        if 'convergence_study' in results:
            html_content += """
            <h2>3. Convergence Study</h2>
            <p>Analysis of error convergence with respect to time step size:</p>
            <div class="figure">
                <img src="convergence_study.png" alt="Convergence Study">
            </div>
            """
        
        # Add solver comparison section
        if 'solver_comparison' in results:
            html_content += """
            <h2>4. ODE Solver Comparison</h2>
            <p>Performance comparison of different ODE solvers for neural operators:</p>
            <div class="figure">
                <img src="solver_comparison.png" alt="Solver Comparison">
            </div>
            """
        
        html_content += """
            <h2>5. Conclusions</h2>
            <div class="summary">
                <p><strong>Key Findings:</strong></p>
                <ul>
                    <li>Neural operator splitting provides a viable alternative to classical methods</li>
                    <li>Performance varies significantly across different test cases and initial conditions</li>
                    <li>Choice of ODE solver has significant impact on stability and accuracy</li>
                    <li>Time step size critically affects convergence behavior</li>
                </ul>
                
                <p><strong>Recommendations:</strong></p>
                <ul>
                    <li>Use appropriate ODE solver based on stiffness of the problem</li>
                    <li>Conduct convergence studies for specific applications</li>
                    <li>Consider hybrid approaches combining classical and neural methods</li>
                </ul>
            </div>
            
            <footer>
                <p><em>Report generated automatically by Neural Operator Splitting Analysis Suite</em></p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = os.path.join(report_dir, 'analysis_report.html')
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved: {html_file}")


def main():
    """Main function for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze neural operator splitting results')
    parser.add_argument('--results_dir', type=str, 
                       default='./paper/neural-operator-splitting/test_results',
                       help='Directory containing test results')
    parser.add_argument('--create_report', action='store_true',
                       help='Create comprehensive HTML report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NeuralSplittingAnalyzer(args.results_dir)
    
    if analyzer.results is None:
        print("No results found. Run test_neural_splitting.py first.")
        return
    
    # Create comprehensive report
    if args.create_report:
        report_dir = analyzer.create_comprehensive_report(analyzer.results)
        print(f"\\nAnalysis complete! Open {os.path.join(report_dir, 'analysis_report.html')} to view the report.")
    else:
        print("Analysis tools loaded. Use analyzer.create_comprehensive_report() to generate report.")


if __name__ == "__main__":
    main()