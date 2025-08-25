import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from both CSV files
euler_data = pd.read_csv('test_results_euler/parameter_grid/parameter_grid_results.csv')
rk4_data = pd.read_csv('test_results_rk4/parameter_grid/parameter_grid_results.csv')

# Add method identifier to distinguish datasets
euler_data['solver'] = 'Euler'
rk4_data['solver'] = 'RK4'

# Combine datasets
combined_data = pd.concat([euler_data, rk4_data])

def plot_comparison(beta_val, nu_val, title_suffix):
    """Plot L2 error vs number of steps for given beta and nu values"""
    
    # Filter data for specific beta and nu values
    filtered_data = combined_data[(combined_data['beta'] == beta_val) & 
                                 (combined_data['nu'] == nu_val)]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors and styles for better distinction
    colors = {'Euler': {'neural_lie': '#FF4444', 'classical_lie': '#4444FF'},
              'RK4': {'neural_lie': '#AA0000', 'classical_lie': '#000088'}}
    
    markers = {'Euler': {'neural_lie': 'o', 'classical_lie': 's'},
               'RK4': {'neural_lie': '^', 'classical_lie': 'D'}}
    
    linestyles = {'Euler': '-', 'RK4': ':'}
    
    # Plot for each combination of solver and method
    for solver in ['Euler', 'RK4']:
        for method in ['neural_lie', 'classical_lie']:
            subset = filtered_data[(filtered_data['solver'] == solver) & 
                                 (filtered_data['method'] == method)]
            
            if not subset.empty:
                # Sort by number of steps
                subset_sorted = subset.sort_values('num_steps')
                
                color = colors[solver][method]
                marker = markers[solver][method]
                linestyle = linestyles[solver]
                
                label = f'{solver} - {method.replace("_", " ").title()}'
                
                ax.loglog(subset_sorted['num_steps'], subset_sorted['l2_error'], 
                         marker=marker, linestyle=linestyle, color=color, 
                         label=label, markersize=10, linewidth=3, 
                         markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Number of Steps', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title(f'L2 Error vs Number of Steps\n{title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create a more informative legend
    legend = ax.legend(fontsize=10, loc='best')
    
    # Add text annotation to explain the visual encoding
    ax.text(0.02, 0.98, 'Lines: Solid (—) = Euler, Dotted (⋯) = RK4\n' +
                        'Markers: ○ = Neural Lie (Euler), ▲ = Neural Lie (RK4)\n' +
                        '         ■ = Classical Lie (Euler), ♦ = Classical Lie (RK4)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-axis to show actual step values
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'comparison_beta_{beta_val}_nu_{nu_val}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    return fig

# Create plots for two interesting cases
print("Creating comparison plots...")

# Case 1: Low advection (beta=1.0, nu=0.05)
fig1 = plot_comparison(1.0, 0.05, "Low Advection Case (β=1.0, ν=0.05)")

# Case 2: High advection (beta=4.0, nu=0.1) 
fig2 = plot_comparison(4.0, 0.1, "High Advection Case (β=4.0, ν=0.1)")

plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")

for beta_val, nu_val, case_name in [(1.0, 0.05, "Low Advection"), (4.0, 0.1, "High Advection")]:
    print(f"\n{case_name} Case (β={beta_val}, ν={nu_val}):")
    
    filtered_data = combined_data[(combined_data['beta'] == beta_val) & 
                                 (combined_data['nu'] == nu_val)]
    
    for solver in ['Euler', 'RK4']:
        print(f"\n  {solver} Method:")
        solver_data = filtered_data[filtered_data['solver'] == solver]
        
        for method in ['neural_lie', 'classical_lie']:
            method_data = solver_data[solver_data['method'] == method]
            if not method_data.empty:
                min_error = method_data['l2_error'].min()
                max_error = method_data['l2_error'].max()
                min_steps = method_data[method_data['l2_error'] == min_error]['num_steps'].iloc[0]
                print(f"    {method.replace('_', ' ').title()}: Min L2 error = {min_error:.6f} (at {min_steps} steps)")
                print(f"    {method.replace('_', ' ').title()}: Max L2 error = {max_error:.6f}")