from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os

def plot_single_variable(df, df_name, scenario, x_col, output_dir):
    df_scenario = df[df['scenario'] == scenario]
    x = df_scenario[x_col].values
    y = df_scenario["error_encoder"].values
    plt.figure(figsize=(8,6))
    plt.plot(x, y, label=df_name)
    plt.xlabel(x_col)
    plt.ylabel("error_encoder")
    plt.title(f'{scenario}: {df_name}')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{df_name}_{scenario}.png"))
    plt.close()

def plot_pair_heatmap_contours_fixed(df, df_name, scenario, x_col, y_col, error_column, title_suffix, vmin, vmax, output_dir):
    df_scenario = df[df['scenario'] == scenario]

    x = np.logspace(np.log10(df_scenario[x_col].min()+1e-12), np.log10(df_scenario[x_col].max()+1e-12), 200)
    y = np.logspace(np.log10(df_scenario[y_col].min()+1e-12), np.log10(df_scenario[y_col].max()+1e-12), 200)
    X, Y = np.meshgrid(x, y)

    # interpolate error values over the grid
    points = df_scenario[[x_col, y_col]].values
    values = df_scenario[error_column].values

    Z = griddata(points, values, (X, Y), method='linear')
    Z_nearest = griddata(points, values, (X, Y), method='nearest')
    Z = np.where(np.isnan(Z), Z_nearest, Z)

    # mask invalid or ≤0 values for log
    Z_masked = np.ma.masked_invalid(Z)
    Z_masked = np.ma.masked_less_equal(Z_masked, 0)

    # more contour levels for smoothness
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)

    plt.figure(figsize=(8,6))
    heatmap = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis',
                             norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.xscale('log')
    plt.yscale('log')

    contour = plt.contour(X, Y, Z_masked, levels=levels, colors='black', linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=7, fmt="%.1e")

    cbar = plt.colorbar(heatmap)
    cbar.set_label(error_column)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{scenario}: {title_suffix}')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    
    # Create directory and save plot
    save_dir = os.path.join(output_dir, df_name, scenario)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{error_column}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="plot")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))

    input_dir = os.path.join(cfg.results_dir, cfg.run_name)
    output_dir = os.path.join(cfg.results_dir, cfg.run_name, "plots") 

    os.makedirs(output_dir, exist_ok=True)

    #1. Interpolation vs extrapolation results
    df_interpolation = pd.read_csv(os.path.join(input_dir, "inter_test_results_10.csv"))
    df_extrapolation = pd.read_csv(os.path.join(input_dir, "extra_up_test_results_10.csv"))

    dataframes = [(df_interpolation, "interpolation", "advection", "adv1"),
                  (df_interpolation, "interpolation", "diffusion", "visc1"),
                  (df_extrapolation, "extrapolation", "advection", "adv1"),
                  (df_extrapolation, "extrapolation", "diffusion", "visc1")]

    for df, df_name, scenario, x_col in dataframes:
        plot_single_variable(df, df_name, scenario, x_col, output_dir)
    #2. Composition of operators

    df_single_finetune = pd.read_csv(os.path.join(input_dir, "single_finetune_test_results_10.csv"))
    df_double_finetune = pd.read_csv(os.path.join(input_dir, "double_finetune_test_results_10.csv"))
    df_manual_composition = pd.read_csv(os.path.join(input_dir, "composition_test_results_10.csv"))

    dataframes = [(df_single_finetune, "single_finetune", "error_finetune_composed"), (df_double_finetune, "double_finetune", "error_finetune_composed"), (df_manual_composition, "manual_composition", "error_manual_composed"), (df_manual_composition, "from_encoder", "error_encoder_composed")]
    
    for df, df_name, error_column in dataframes:
        # Get unique scenarios from the dataframe
        scenarios = df['scenario'].unique()
        
        for scenario in scenarios:
            # Determine appropriate columns and parameters based on scenario
            if 'advection+advection' in scenario:
                x_col, y_col = 'adv1', 'adv2'
            elif 'diffusion+diffusion' in scenario:
                x_col, y_col = 'visc1', 'visc2'
            elif 'advection+diffusion' in scenario:
                x_col, y_col = 'adv_sum', 'visc_sum'
            else:
                continue  # Skip unknown scenarios
            
            vmin = 1e-4
            vmax = 10
            
            plot_pair_heatmap_contours_fixed(
                df, df_name, scenario, x_col, y_col,
                error_column,
                error_column,
                vmin, vmax, output_dir
            )

if __name__=="__main__":
    main()



