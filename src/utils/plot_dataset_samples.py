import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dir = "/mnt/home/lserrano/disco-ttg/datasets"
plot_dir = "./datasets/plots"
os.makedirs(plot_dir, exist_ok=True)

def plot_random_samples(dataset_path, n_samples=5, dataset_name="dataset"):
    data = np.load(dataset_path)
    trajectories = data["trajectories"]  # shape: [N, T, H]
    velocities = data["velocities"]
    diffusivities = data["diffusivities"]
    N, T, H = trajectories.shape
    indices = random.sample(range(N), n_samples)
    for idx in indices:
        traj = trajectories[idx]  # shape: [T, H]
        velocity = velocities[idx]
        diffusivity = diffusivities[idx]
        plt.figure(figsize=(10, 6))
        # Plot as an image (time x space)
        plt.imshow(traj, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="u")
        plt.xlabel("Space")
        plt.ylabel("Time")
        plt.title(f"{dataset_name} idx={idx} | v={velocity:.3g} | D={diffusivity:.3g}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{dataset_name}_sample{idx}.png")
        plt.close()
        # Also plot as line plots for a few time steps
        plt.figure(figsize=(10, 6))
        for t in np.linspace(0, T-1, min(8, T), dtype=int):
            plt.plot(traj[t], label=f"t={t}")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.title(f"{dataset_name} idx={idx} | v={velocity:.3g} | D={diffusivity:.3g} (lines)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{dataset_name}_sample{idx}_lines.png")
        plt.close()

def plot_prediction_vs_ground_truth(pred, gt, idx=0, out_dir="./results/predictions", split="test", sample_name="sample"):
    """
    pred: [T, H] or [T, ...]
    gt: [T, H] or [T, ...]
    idx: sample index
    out_dir: directory to save plots
    split: data split name
    sample_name: identifier for the sample
    """
    pred = pred.squeeze(-2)
    gt = gt.squeeze(-2)
    os.makedirs(out_dir, exist_ok=True)
    delta = np.abs(pred - gt)
    # --- Image plots ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axs[0].imshow(gt, aspect="auto", origin="lower", cmap="viridis")
    axs[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axs[0], label="u")
    axs[0].set_xlabel("Space")
    axs[0].set_ylabel("Time")
    im1 = axs[1].imshow(pred, aspect="auto", origin="lower", cmap="viridis")
    axs[1].set_title("Prediction")
    plt.colorbar(im1, ax=axs[1], label="u")
    axs[1].set_xlabel("Space")
    axs[1].set_ylabel("Time")
    im2 = axs[2].imshow(delta, aspect="auto", origin="lower", cmap="magma")
    axs[2].set_title("Delta |Pred - GT|")
    plt.colorbar(im2, ax=axs[2], label="|u_pred - u_gt|")
    axs[2].set_xlabel("Space")
    axs[2].set_ylabel("Time")
    plt.suptitle(f"{split} {sample_name} idx={idx}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{split}_{sample_name}_idx{idx}_image.png")
    plt.close()
    # --- Line plots ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6)) # 1 row, 3 columns
    T = gt.shape[0]
    n_lines = min(5,T)
    ts = np.linspace(0, T-1, n_lines, dtype=int)
    for t in ts:
        axs[0].plot(gt[t], label=f"Time step {t}")
        axs[1].plot(pred[t], label=f"Time step {t}")
        axs[2].plot(delta[t], label=f"Time step {t}")
    # Set titles, labels, and grid for each subplot
    axs[0].set_title("Ground Truth (GT)", fontsize=14)
    axs[0].set_xlabel("Spatial Dimension", fontsize=12)
    axs[0].set_ylabel("Value (u)", fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(title="Time Steps", bbox_to_anchor=(1.05, 1), loc='upper left')

    axs[1].set_title("Prediction", fontsize=14)
    axs[1].set_xlabel("Spatial Dimension", fontsize=12)
    axs[1].set_ylabel("Value (u)", fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].set_title("Delta |Prediction - GT|", fontsize=14)
    axs[2].set_xlabel("Spatial Dimension", fontsize=12)
    axs[2].set_ylabel("|u_pred - u_gt|", fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"{split} {sample_name} idx={idx} (lines)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{split}_{sample_name}_idx{idx}_lines.png")
    plt.close()

if __name__ == "__main__":
    #for fname in ["train.npz", "val.npz", "test.npz"]:
    for fname in ["train_explicit.npz", "val_explicit.npz", "test_explicit.npz"]:
        print(f"Plotting random samples from {fname}...")
        plot_random_samples(os.path.join(data_dir, fname), n_samples=5, dataset_name=fname.split(".")[0])
    print("Done.") 