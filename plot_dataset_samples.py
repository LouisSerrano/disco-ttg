import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dir = "/mnt/home/lserrano/disco-ball/datasets"
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

if __name__ == "__main__":
    #for fname in ["train.npz", "val.npz", "test.npz"]:
    for fname in ["train_explicit.npz", "val_explicit.npz", "test_explicit.npz"]:
        print(f"Plotting random samples from {fname}...")
        plot_random_samples(os.path.join(data_dir, fname), n_samples=5, dataset_name=fname.split(".")[0])
    print("Done.") 