import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from models import DISCOHouse
from train import DISCOLitModule
from utils import RelativeL2
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import reduce
import hydra
from omegaconf import DictConfig
from plot_dataset_samples import plot_prediction_vs_ground_truth
import random

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, u, sub_x, sub_t, input_frames=16, output_frames=34):
        self.u = u 
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.slice_size = input_frames + output_frames

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        images = torch.from_numpy(self.u[idx]).unsqueeze(-2).float() # add channel dimension
        images = images[::self.sub_t, ..., ::self.sub_x]
        start_index = 0
        images = images[start_index:start_index + self.slice_size]
        input = images[:self.input_frames]
        target = images[self.input_frames:]
        return input, target

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- LOAD MODEL ---
def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    model = DISCOLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.model.to(device)
    model.eval()
    return model

# --- LOAD DATA ---
def load_split(data_dir, split, setting, sub_x=1, sub_t=1, n_input_frames=16, n_output_frames=34):
    path = os.path.join(data_dir, f"{split}_{setting}.npz")
    data = np.load(path)
    dataset = TemporalDataset(data['trajectories'], sub_x, sub_t, n_input_frames, n_output_frames)
    return dataset

# --- AUTOREGRESSIVE PREDICTION ---
def autoregressive_predict(model, initial_seq, n_pred, device):
    preds = []
    current = initial_seq.clone().to(device)
    n_input = current.shape[1]
    for t in range(n_pred):
        inp = current[:, -n_input:].to(device)
        with torch.no_grad():
            state_labels = torch.tensor([0], device=inp.device)
            next_frame, metadata = model(inp, state_labels, n_future_steps=1)
            if t == 0:
                theta = metadata['theta_latent']
        current = torch.cat([current, next_frame], axis=1)
        preds.append(next_frame)
    return torch.cat(preds, axis=1), theta

# --- PLOT THETA_LATENT ---
def plot_theta_latent(theta_dic, out_path=f"theta_latent.png"):
    plt.figure(figsize=(8, 6))
    colors = {"train": "blue", "val": "green", "test": "red"}

    for split in list(theta_dic.keys()):
        all_theta = theta_dic[split]["theta"].cpu().numpy()
        plt.scatter(all_theta[:, 0], all_theta[:, 1], c=colors[split], label=split, alpha=0.5, s=8)
    plt.xlabel("theta_latent[0]")
    plt.ylabel("theta_latent[1]")
    plt.legend()
    plt.title("Theta latent 2D projection")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    @hydra.main(config_path="configs", config_name="config")
    def main(cfg: DictConfig):
        # --- CONFIG ---
        dataset_name = cfg.test.dataset_name
        ckpt_path = cfg.test.ckpt_path
        data_dir = cfg.test.data_dir
        splits = cfg.test.splits
        setting = cfg.test.setting
        n_input = cfg.test.n_input
        n_pred = cfg.test.n_pred
        results_dir = cfg.test.results_dir
        run_name = cfg.test.get('run_name', f"{dataset_name}_{setting}")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/{dataset_name}/{run_name}/plots/{setting}", exist_ok=True)
        os.makedirs(f"{results_dir}/{dataset_name}/{run_name}/predictions/{setting}", exist_ok=True)
        os.makedirs(f"{results_dir}/{dataset_name}/{run_name}/theta/{setting}", exist_ok=True)
        os.makedirs(f"{results_dir}/{dataset_name}/{run_name}/errors/{setting}", exist_ok=True)
        relative_l2_error = RelativeL2()

        # --- MAIN TEST LOOP ---
        model = load_model(ckpt_path, device)
        
        rollout_dic = {}
        theta_dic = {}
        for split in splits:
            all_theta = []
            all_labels = []
            rollout_error = 0
            n = 0
            data = load_split(data_dir, split, setting, 
                            sub_x=cfg.test.get('sub_x', 1), 
                            sub_t=cfg.test.get('sub_t', 1), 
                            n_input_frames=cfg.test.get('n_input_frames', n_input), 
                            n_output_frames=n_pred)
            data_loader = DataLoader(data, batch_size=cfg.test.get('batch_size', 64), shuffle=False, num_workers=4, prefetch_factor=4)
            # --- Select random batch indices for saving predictions ---
            num_batches = len(data_loader)
            num_samples_to_save = 3
            random_indices = sorted(random.sample(range(num_batches), min(num_samples_to_save, num_batches)))
            batch_idx = 0
            for batch in tqdm(data_loader):
                x, target = batch
                x = x.to(device)
                target = target.to(device)
                n_sample = x.shape[0]
                pred, theta = autoregressive_predict(model, x, n_pred=target.shape[1], device=device)
                rollout_error += relative_l2_error(pred, target).item()*n_sample
                n += n_sample
                all_theta.append(theta)
                # Save predictions and ground truth for selected batches
                if batch_idx in random_indices:
                    for i in range(min(3, x.shape[0])):  # Save up to 3 samples per batch
                        pred_np = pred[i].detach().cpu().numpy()
                        target_np = target[i].detach().cpu().numpy()
                        out_dir = f"{results_dir}/{dataset_name}/{run_name}/predictions/{setting}/{split}"
                        os.makedirs(out_dir, exist_ok=True)
                        np.savez_compressed(f"{out_dir}/pred_gt_batch{batch_idx}_sample{i}.npz", pred=pred_np, gt=target_np)
                        plot_prediction_vs_ground_truth(pred_np, target_np, idx=i, out_dir=out_dir, split=split, sample_name=f"batch{batch_idx}")
                batch_idx += 1

            all_labels = [split] * len(all_theta)
            all_theta = torch.cat(all_theta, dim=0)

            # Save theta for this split
            theta_save_path = f"{results_dir}/{dataset_name}/{run_name}/theta/{setting}/{split}_theta.npy"
            np.save(theta_save_path, all_theta.cpu().numpy())

            rollout_dic[split] = {
                "rollout_error": rollout_error/n,
            }
            
            torch.save(rollout_dic, f"{results_dir}/{dataset_name}/{run_name}/errors/{setting}/rollout.pt")

            theta_dic[split] = {
                "theta": all_theta,
                "labels": all_labels
            }

        print(f"setting: {setting}", rollout_dic)
        plot_theta_latent(theta_dic, out_path=f"{results_dir}/{dataset_name}/{run_name}/theta/theta_latent.pdf")

        print("Done.")

    main() 