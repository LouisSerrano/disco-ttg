import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create the dataset from your CSV data
#data = {
#    'experiment_id': list(range(50)),
#    'block_idx': [304, 495, 439, 153, 497, 131, 204, 506, 325, 247, 361, 352, 289, 84, 10, 324, 78, 30, 184, 195, 222, 209, 281, 208, 124, 250, 76, 380, 342, 323, 244, 331, 140, 9, 72, 398, 484, 225, 73, 177, 423, 68, 155, 55, 403, 298, 77, 501, 317, 473],
#    'alpha_value': [0.740500, 0.724337, 2.667403, 1.341114, 1.934487, 1.330207, 2.103760, 2.318292, 0.511983, 0.175887, 0.387794, 1.886667, 0.232015, 1.737608, 2.156922, 1.952156, 2.997282, 2.584526, 2.494434, 1.944865, 0.466314, 1.137475, 1.556844, 1.384488, 0.078867, 2.048394, 0.540497, 1.057804, 1.947568, 1.728229, 2.705113, 0.052705, 0.386540, 0.657326, 2.071924, 0.919772, 2.423581, 2.774598, 2.963842, 2.611214, 0.403097, 1.393134, 1.056206, 0.770258, 1.379328, 1.820510, 1.551667, 2.427060, 2.846142, 0.154470],
#    'final_train_loss': [0.000153, 0.000130, 0.000702, 0.000273, 0.000453, 0.000334, 0.000760, 0.000719, 0.000101, 0.000047, 0.000062, 0.001548, 0.000040, 0.000364, 0.000752, 0.000507, 0.002125, 0.001630, 0.000506, 0.000375, 0.000061, 0.000379, 0.000354, 0.000319, 0.000016, 0.000798, 0.000128, 0.000245, 0.000849, 0.000382, 0.001098, 0.000009, 0.000060, 0.000193, 0.000568, 0.000246, 0.000676, 0.001225, 0.001004, 0.000841, 0.000103, 0.000349, 0.000283, 0.000116, 0.000405, 0.000595, 0.000402, 0.000524, 0.001431, 0.000040],
#    'min_train_loss': [0.000104, 0.000071, 0.000558, 0.000202, 0.000487, 0.000210, 0.000547, 0.000627, 0.000055, 0.000026, 0.000035, 0.000411, 0.000018, 0.000349, 0.000549, 0.000458, 0.000781, 0.000594, 0.000518, 0.000339, 0.000040, 0.000262, 0.000366, 0.000280, 0.000009, 0.000512, 0.000075, 0.000179, 0.000627, 0.000314, 0.000741, 0.000006, 0.000032, 0.000162, 0.000515, 0.000183, 0.000473, 0.000607, 0.000845, 0.000692, 0.000047, 0.000307, 0.000345, 0.000067, 0.000439, 0.000503, 0.000334, 0.000441, 0.000593, 0.000022]
#}
#df = pd.DataFrame(data)

df = pd.read_csv('results_ssprk3_5.csv')
# Convert to DataFrame


# Create the plot
plt.figure(figsize=(12, 8))

# Scatter plot for both loss types
plt.scatter(df['alpha_value'], df['final_train_loss'], 
           color='red', alpha=0.7, s=60, label='Final Train Loss', edgecolors='darkred')
plt.scatter(df['alpha_value'], df['min_train_loss'], 
           color='green', alpha=0.7, s=60, label='Min Train Loss', edgecolors='darkgreen')

# Add labels and title
plt.xlabel('Alpha Value', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Training Loss vs Alpha Value', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Use log scale for y-axis
plt.yscale('log')

# Add some statistics as text
correlation_final = np.corrcoef(df['alpha_value'], df['final_train_loss'])[0, 1]
correlation_min = np.corrcoef(df['alpha_value'], df['min_train_loss'])[0, 1]

plt.text(0.02, 0.98, f'Correlation (Final Loss): {correlation_final:.3f}\nCorrelation (Min Loss): {correlation_min:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Tight layout for better appearance
plt.tight_layout()

# Save the plot as PNG
plt.savefig('burgers_alpha_vs_loss_params_ssprk3_5.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some summary statistics
print("Summary Statistics:")
print(f"Alpha value range: {df['alpha_value'].min():.6f} - {df['alpha_value'].max():.6f}")
print(f"Final loss range: {df['final_train_loss'].min():.6f} - {df['final_train_loss'].max():.6f}")
print(f"Min loss range: {df['min_train_loss'].min():.6f} - {df['min_train_loss'].max():.6f}")
print(f"Correlation between alpha and final loss: {correlation_final:.3f}")
print(f"Correlation between alpha and min loss: {correlation_min:.3f}")

# Find best performing experiments (lowest losses)
best_final_idx = df['final_train_loss'].idxmin()
best_min_idx = df['min_train_loss'].idxmin()

print(f"\nBest experiment (lowest final loss): #{best_final_idx} with alpha={df.loc[best_final_idx, 'alpha_value']:.6f}, loss={df.loc[best_final_idx, 'final_train_loss']:.6f}")
print(f"Best experiment (lowest min loss): #{best_min_idx} with alpha={df.loc[best_min_idx, 'alpha_value']:.6f}, loss={df.loc[best_min_idx, 'min_train_loss']:.6f}")
