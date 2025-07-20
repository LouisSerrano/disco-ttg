import torch
import numpy as np
from advection_diffusion import Fractaloid
from train import DISCOLitModule, advection_diffusion_analytical
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import RelativeL2
from models import DISCOHouse
import csv
from itertools import product
import os

class TemporalDatasetFixedCI(torch.utils.data.Dataset):
    def __init__(self, n_batches, batch_size, sub_x, sub_t, split="train", input_frames=16, output_frames=2,
                 L=16.0, nx=256, nt=100, T=10.0,
                 v_range=[0.01, 0.025, 0.05, 0.1, 0.5, 1.0], D_range=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                 fractal_degree=8, fractal_power=2, seed=None):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.L = L
        self.nx = nx
        self.nt = nt
        self.T = T
        self.v_range = v_range
        self.D_range = D_range
        self.fractal_degree = fractal_degree
        self.fractal_power = fractal_power
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u0 = []
        for _ in range(self.batch_size):
            fractaloid = Fractaloid(
                degree=self.fractal_degree,
                power=self.fractal_power,
                size=self.nx,
                patch_size=self.nx
            )
            u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
            u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
            self.u0.append(torch.from_numpy(u0))
            
        self.u0 = torch.stack(self.u0)

    def __len__(self):
        return len(self.u0)

    def __getitem__(self, idx):
    
        input_frames = self.input_frames
        batch_inputs = []
        batch_targets = []
        batch_v = []
        batch_d = []
        batch_init = []
        for v in self.v_range:
            for d in self.D_range:
                u0 = self.u0[idx]
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=self.L, v=v, D=d, nt=self.nt, T=self.T
                )
                u_xt = u_xt[::self.sub_t, ::self.sub_x]
                input = u_xt[:input_frames].copy()
                target = u_xt[input_frames: input_frames + self.output_frames].copy()
                
                batch_inputs.append(torch.from_numpy(input).unsqueeze(-2).float())
                batch_targets.append(torch.from_numpy(target).unsqueeze(-2).float())
                batch_v.append(v)
                batch_d.append(d)
                batch_init.append(u0)
                
        batch = {
            'input': torch.stack(batch_inputs),
            'target': torch.stack(batch_targets),
            'velocities': batch_v,
            'diffusivities': batch_d,
            'initial_conditions': torch.stack(batch_init)
        }
        return batch
    

def get_data(
    model,
    advection_speeds = [0.6, 0.6, 1.2],
    viscosities = [0., 0, 0.],
    n_batches = 1,
    batch_size = 128,
    split="test",
    n_input_frames=16,
    n_output_frames=34,
    sub_x=1,
    sub_t=1,
    L=16.0,
    nx=256,
    nt=100,
    T=10.0,
    fractal_degree=256,
    fractal_power=3.0,
    device="cuda",
    ):

    relative_l2_error = RelativeL2()

    all_velocities = []
    all_diffusivities = []
    all_input = []
    all_target = []
    all_theta_latent = []
    all_theta = []
    all_errors = []

    train_ds = TemporalDatasetFixedCI(
            n_batches=n_batches,
            batch_size=batch_size,
            sub_x=sub_x,
            sub_t=sub_t,
            split=split,
            input_frames=n_input_frames,
            output_frames=n_output_frames,
            L=L,
            nx=nx,
            nt=nt,
            T=T,
            fractal_power=fractal_power,
            fractal_degree=fractal_degree, # nx
            v_range=[0],#(0.01, 1.0),
            D_range=[0], #(0.001, 1.0),
        )

    for advection_speed, viscosity in zip(advection_speeds, viscosities):

        train_ds.v_range=[advection_speed]
        train_ds.D_range=[viscosity]
        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=1, prefetch_factor=1, pin_memory=True, shuffle=False)
        
        for batch in tqdm(train_loader):
            inp, target = batch["input"], batch["target"]
            inp = inp.squeeze(1)
            target = target.squeeze(1)
            all_velocities += batch["velocities"]
            all_diffusivities += batch["diffusivities"]
            
        inp = inp.to(device)
        target = target.to(device)
        state_labels = torch.tensor([0], device=inp.device)
        
        x_shape = inp.shape
        B, T, C = x_shape[:3]
        spatial = x_shape[3:]
        dim = len(spatial)
        
        n_sample = inp.shape[0]
        #pred, theta = autoregressive_predict(model, inp, n_pred=target.shape[1], device=device)
        
        #predict
        with torch.no_grad():
            # encode into 2 dimensional
            theta_latent, metadata= model.encode_theta_latent(inp, state_labels)
            # decode into 100k parameters
            theta = model.decode_theta(theta_latent, dim)
            pred, metadata = model.solve_ode(inp[:, -1], theta, state_labels, dim, n_future_steps=n_output_frames, predict_normed=False, metadata=metadata)

        rollout_error = relative_l2_error(pred, target[:, :n_output_frames]).item()
        print(f"Error with encoder: {advection_speed:.3f}x{viscosity:.3f}", rollout_error)

        all_errors.append(rollout_error)
        all_theta_latent.append(theta_latent)
        all_theta.append(theta)
        all_input.append(inp)
        all_target.append(target)

    all_theta_latent = torch.stack(all_theta_latent)
    all_theta = torch.stack(all_theta)
    all_input = torch.stack(all_input)
    all_target = torch.stack(all_target)
    all_errors = np.array(all_errors)

    return all_theta_latent, all_theta, all_input, all_target, all_errors


def run_tests(model, advection_range, diffusion_range, device, n_points=20, output_csv="composition_test_results.csv"):
    results = []
    # Define parameter ranges

    # 1. Advection + Advection (viscosity=0)
    for adv1 in advection_range:
        viscosities = [0.0]
        advection_speeds = [adv1]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "advection",
            "adv1": f"{adv1:.4f}",
            "visc1": f"{0.0:.4f}", 
            "error_encoder": all_errors[0],
        })

    # 2. Diffusion + Diffusion (advection=0)
    for visc1 in diffusion_range:
        advection_speeds = [0.0]
        viscosities = [visc1]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "diffusion",
            "adv1": f"{0.0:.4f}",
            "visc1": f"{visc1:.4f}",
            "error_encoder": all_errors[0],
        })

    # 3. Advection + Diffusion
    for adv, visc in product(advection_range, diffusion_range):
        advection_speeds = [adv]
        viscosities = [visc]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "advection+diffusion",
            "adv1": f"{adv:.4f}", 
            "visc1": f"{visc1:.4f}",
            "error_encoder": all_errors[0]
        })

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "scenario", "adv1", "visc1",
            "error_encoder",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # load model
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt_time="2025-07-01/00-21-40" #"2025-06-24/15-27-28" 
    results_dir = f"/mnt/home/lserrano/disco-ball/results/{ckpt_time}"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = f"/mnt/home/lserrano/disco-ball/outputs/{ckpt_time}/model_final.ckpt"
    print(f"Loading model from {ckpt_path}...")
    model = DISCOLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.model.to(device)
    model.eval()

    # Run the composition grid tests and save to CSV
    n_points = 10
    advection_range_inter = np.exp(np.linspace(np.log(0.01), np.log(1.0), n_points)) #np.logspace(0.01, 1.0, n_points)
    diffusion_range_inter = np.exp(np.linspace(np.log(0.001), np.log(1.0), n_points)) #np.logspace(0.001, 1.0, n_points)
    run_tests(model, advection_range_inter, diffusion_range_inter, device, n_points=n_points, output_csv=f"{results_dir}/inter_test_results_{n_points}.csv")

    advection_range_extra_up = np.exp(np.linspace(np.log(1.0), np.log(5.0), n_points)) #np.logspace(0.01, 1.0, n_points)
    diffusion_range_extra_up = np.exp(np.linspace(np.log(1.0), np.log(5.0), n_points)) #np.logspace(0.001, 1.0, n_points)
    run_tests(model, advection_range_extra_up, diffusion_range_extra_up, device, n_points=n_points, output_csv=f"{results_dir}/extra_up_test_results_{n_points}.csv")

    advection_range_extra_down = np.exp(np.linspace(np.log(0.001), np.log(0.01), n_points)) #np.logspace(0.01, 1.0, n_points)
    diffusion_range_extra_down = np.exp(np.linspace(np.log(0.0001), np.log(0.001), n_points)) #np.logspace(0.001, 1.0, n_points)
    run_tests(model, advection_range_extra_down, diffusion_range_extra_down, device, n_points=n_points, output_csv=f"{results_dir}/extra_down_test_results_{n_points}.csv")