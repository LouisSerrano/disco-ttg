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
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    


def fine_tune_theta_latent(model, theta_latent, x_test, y_test, state_labels, dim, 
                           n_input_frames=16, n_output_frames=34, epochs=100, 
                           composition_type="composition", device="cuda"):
    """
    Fine-tune theta latent parameters to minimize rollout error.
    
    Args:
        model: The DISCO model
        theta_latent: Initial theta latent parameters
        x_test: Input test data
        y_test: Target test data  
        state_labels: State labels for the model
        dim: Spatial dimension
        n_input_frames: Number of input frames
        n_output_frames: Number of output frames to predict
        epochs: Number of training epochs
        composition_type: Type of composition ("sum" or "composition")
        device: Device to run on
        
    Returns:
        float: Final rollout error for the second part of the trajectory
    """
    relative_l2_error = RelativeL2()
    
    # Initialize theta latent parameters for fine-tuning
    theta_latent1 = torch.zeros_like(theta_latent.detach()).requires_grad_()
    theta_latent2 = torch.zeros_like(theta_latent.detach()).requires_grad_()
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam([theta_latent1, theta_latent2], lr=1e-1, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Starting fine-tuning...")
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning"):
        model.train()
        
        # Determine training horizon based on epoch
        #training_horizon = 1 if epoch < 400 else 10
        training_horizon = 1
        
        # Random time step for training
        t = random.randint(0, n_input_frames - training_horizon - 1)
        
        # Decode theta parameters
        theta1 = model.decode_theta(theta_latent1, dim)
        theta2 = model.decode_theta(theta_latent2, dim)
        
        # Forward pass
        if composition_type == "sum":
            pred, _ = model.solve_ode_with_2_operators(
                x_test[:, t], theta1, theta2, state_labels, dim, 
                n_future_steps=training_horizon, predict_normed=False, metadata={}
            )
        else:
            pred = []
            x_test_ = x_test[:, t].clone()
            for _ in range(training_horizon):
                pred_int, _ = model.solve_ode(
                    x_test_, theta1, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
                pred_, _ = model.solve_ode(
                    pred_int[:, -1], theta2, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
                pred.append(pred_[:, -1])
                x_test_ = pred_[:, -1]
            pred = torch.cat(pred, 1)
        
        # Calculate loss
        loss = relative_l2_error(pred, x_test[:, t+1:t+training_horizon+1])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                model.eval()
                theta1 = model.decode_theta(theta_latent1, dim)
                theta2 = model.decode_theta(theta_latent2, dim)
                
                # Evaluate on full trajectory
                pred_test = []
                x_test_ = x_test[:, -1].clone()
                for _ in range(n_output_frames):
                    if composition_type == "sum":
                        pred, _ = model.solve_ode_with_2_operators(
                            x_test_, theta1, theta2, state_labels, dim, 
                            n_future_steps=1, predict_normed=False, metadata={}
                        )
                    else:
                        pred_int, _ = model.solve_ode(
                            x_test_, theta1, state_labels, dim, 
                            n_future_steps=1, predict_normed=False, metadata={}
                        )
                        pred, _ = model.solve_ode(
                            pred_int[:, -1], theta2, state_labels, dim, 
                            n_future_steps=1, predict_normed=False, metadata={}
                        )
                    pred_test.append(pred)
                    x_test_ = pred[:, -1]
                
                pred_test = torch.cat(pred_test, 1)
                test_error = relative_l2_error(pred_test, y_test).item()
                
                print(
                    f"Epoch [{epoch+1}/{epochs}] | "
                    f"Loss: {loss.item():.6f} | "
                    f"Rollout Error: {test_error:.6f}"
                )
    
    print("Fine-tuning finished.")
    
    # Return final rollout error
    with torch.no_grad():
        model.eval()
        theta1 = model.decode_theta(theta_latent1, dim)
        theta2 = model.decode_theta(theta_latent2, dim)
        
        pred_test = []
        x_test_ = x_test[:, -1].clone()
        for _ in range(n_output_frames):
            if composition_type == "sum":
                pred, _ = model.solve_ode_with_2_operators(
                    x_test_, theta1, theta2, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
            else:
                pred_int, _ = model.solve_ode(
                    x_test_, theta1, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
                pred, _ = model.solve_ode(
                    pred_int[:, -1], theta2, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
            pred_test.append(pred)
            x_test_ = pred[:, -1]
        
        pred_test = torch.cat(pred_test, 1)
        final_error = relative_l2_error(pred_test, y_test).item()
    
    return final_error
    

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
    epochs=500,
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
    # finetune the model
    rollout_error = fine_tune_theta_latent(model, theta_latent, inp, target[:, :n_output_frames], state_labels, dim, n_input_frames=n_input_frames, n_output_frames=n_output_frames, epochs=epochs, device=device)
    print(f"Error with finetune: {advection_speed:.3f}x{viscosity:.3f}", rollout_error)

    all_errors.append(rollout_error)

    all_theta_latent = torch.stack(all_theta_latent)
    all_theta = torch.stack(all_theta)
    all_input = torch.stack(all_input)
    all_target = torch.stack(all_target)
    all_errors = np.array(all_errors)

    return all_theta_latent, all_theta, all_input, all_target, all_errors


def run_composition_tests(model, device, n_points=20, output_csv="composition_test_results.csv"):
    results = []
    # Define parameter ranges
    advection_range = np.exp(np.linspace(np.log(0.01), np.log(1.0), n_points)) #np.logspace(0.01, 1.0, n_points)
    diffusion_range = np.exp(np.linspace(np.log(0.001), np.log(1.0), n_points)) #np.logspace(0.001, 1.0, n_points)

    # 1. Advection + Advection (viscosity=0)
    for adv1, adv2 in product(advection_range, repeat=2):
        adv_sum = adv1 + adv2
        viscosities = [0.0, 0.0, 0.0]
        advection_speeds = [adv1, adv2, adv_sum]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "advection+advection",
            "adv1": f"{adv1:.3f}", "adv2": f"{adv2:.3f}", "adv_sum": f"{adv_sum:.3f}",
            "visc1": f"{0.0:.3f}", "visc2": f"{0.0:.3f}", "visc_sum": f"{0.0:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1],
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3],
        })

    # 2. Diffusion + Diffusion (advection=0)
    for visc1, visc2 in product(diffusion_range, repeat=2):
        visc_sum = visc1 + visc2
        advection_speeds = [0.0, 0.0, 0.0]
        viscosities = [visc1, visc2, visc_sum]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "diffusion+diffusion",
            "adv1": f"{0.0:.3f}", "adv2": f"{0.0:.3f}", "adv_sum": f"{0.0:.3f}",
            "visc1": f"{visc1:.3f}", "visc2": f"{visc2:.3f}", "visc_sum": f"{visc_sum:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1], 
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3]
        })

    # 3. Advection + Diffusion
    for adv, visc in product(advection_range, diffusion_range):
        advection_speeds = [adv, 0.0, adv]
        viscosities = [0.0, visc, visc]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device
        )
        results.append({
            "scenario": "advection+diffusion",
            "adv1": f"{adv:.3f}", "adv2": f"{0.0:.3f}", "adv_sum": f"{adv:.3f}",
            "visc1": f"{0.0:.3f}", "visc2": f"{visc:.3f}", "visc_sum": f"{visc:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1],
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3],
        })

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "scenario", "adv1", "adv2", "adv_sum", "visc1", "visc2", "visc_sum",
            "error_encoder1", "error_encoder2", "error_encoder_composed", "error_finetune_composed"
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
    run_composition_tests(model, device, n_points=n_points, output_csv=f"{results_dir}/finetune_test_results_{n_points}.csv")