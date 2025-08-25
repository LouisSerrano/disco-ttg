import torch
import numpy as np
from src.advection_diffusion import Fractaloid
from train.train import DISCOLitModule, advection_diffusion_analytical
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import RelativeL2
from src.operators.disco import DISCOHouse
import csv
from itertools import product, combinations_with_replacement
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

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
                           n_input_frames=16, n_output_frames=34, epochs=500, 
                           composition_type="composition", device="cuda", 
                           optimizer_type="adam", lr=1e-1, momentum=0.9, training_horizon=1, num_steps=1,
                           advection_speed=0.0, diffusion_coeff=0.0, debug_dir="debug_predictions"):
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
        num_steps: Number of sub-steps for finer dt integration
        advection_speed: Advection speed for debug output
        diffusion_coeff: Diffusion coefficient for debug output  
        debug_dir: Directory to save debug files
        
    Returns:
        float: Final rollout error for the second part of the trajectory
    """
    relative_l2_error = RelativeL2()
    
    # Initialize theta latent parameters for fine-tuning
    theta_latent1 = torch.zeros_like(theta_latent.detach()).requires_grad_()
    theta_latent2 = torch.zeros_like(theta_latent.detach()).requires_grad_()
    
    # Optimizer and scheduler
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD([theta_latent1, theta_latent2], lr=lr, weight_decay=0)
    elif optimizer_type.lower() == "sgd_momentum":
        optimizer = torch.optim.SGD([theta_latent1, theta_latent2], lr=lr, momentum=momentum, weight_decay=0)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([theta_latent1, theta_latent2], lr=lr, weight_decay=0)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Starting fine-tuning...")
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning"):
        model.train()
        
        # Use the training horizon parameter
        
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
                current_state = x_test_
                n_steps=1
                for _ in range(n_steps):
                    pred_int, _ = model.solve_ode(
                        current_state, theta1, state_labels, dim, 
                        n_future_steps=1, integration_time=1/n_steps, dt=1/n_steps, predict_normed=False, metadata={}
                    )
                    pred_, _ = model.solve_ode(
                        pred_int[:, -1], theta2, state_labels, dim, 
                        n_future_steps=1, integration_time=1/n_steps, dt=1/n_steps, predict_normed=False, metadata={}
                    )
                    current_state = pred_[:, -1]
                
                pred.append(current_state)
                x_test_ = current_state
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
                        current_state = x_test_
                        for _ in range(num_steps):
                            pred_int, _ = model.solve_ode(
                                current_state, theta1, state_labels, dim, integration_time=1/num_steps, dt=1/num_steps,
                                n_future_steps=1, predict_normed=False, metadata={}
                            )
                            pred, _ = model.solve_ode(
                                pred_int[:, -1], theta2, state_labels, dim, integration_time=1/num_steps, dt=1/num_steps,
                                n_future_steps=1, predict_normed=False, metadata={}
                            )
                            current_state = pred[:, -1]
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
                    n_future_steps=1, integration_time=1, predict_normed=False, metadata={}
                )
            else:
                current_state = x_test_
                for _ in range(num_steps):
                    pred_int, _ = model.solve_ode(
                        current_state, theta1, state_labels, dim, 
                        n_future_steps=1, integration_time=1/num_steps, dt=1/num_steps, predict_normed=False, metadata={}
                    )
                    pred, _ = model.solve_ode(
                        pred_int[:, -1], theta2, state_labels, dim, 
                        n_future_steps=1, integration_time=1/num_steps, dt=1/num_steps, predict_normed=False, metadata={}
                    )
                    current_state = pred[:, -1]
            pred_test.append(pred)
            x_test_ = pred[:, -1]
        
        pred_test = torch.cat(pred_test, 1)
        final_error = relative_l2_error(pred_test, y_test).item()
        
        
        save_predictions_and_compare(
            model=model,
            theta_latent1=theta_latent1,
            theta_latent2=theta_latent2,
            x_test=x_test,
            y_test=y_test,
            state_labels=state_labels,
            dim=dim,
            n_output_frames=n_output_frames,
            output_dir=debug_dir,
            composition_type=composition_type,
            num_steps=num_steps,
            advection_speed=advection_speed,
            diffusion_coeff=diffusion_coeff,
            optimizer_type=optimizer_type
        )
    
    return final_error


def save_predictions_and_compare(model, theta_latent1, theta_latent2, x_test, y_test, state_labels, dim, 
                                 n_output_frames=34, output_dir="debug_predictions",
                                 composition_type="composition", num_steps=1, 
                                 advection_speed=0.0, diffusion_coeff=0.0, optimizer_type="none"):
    """
    Save predictions and ground truth for debugging purposes.
    
    Args:
        model: The DISCO model
        theta_latent: Theta latent parameters  
        x_test: Input test data
        y_test: Target test data
        state_labels: State labels for the model
        dim: Spatial dimension
        n_output_frames: Number of output frames to predict
        output_dir: Directory to save debug files
        composition_type: Type of composition ("sum" or "composition")
        num_steps: Number of sub-steps for finer dt integration
        advection_speed: Advection speed for filename
        diffusion_coeff: Diffusion coefficient for filename
        optimizer_type: Type of optimizer used for filename
        
    Returns:
        dict: Dictionary containing predictions, ground truth, and error metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    relative_l2_error = RelativeL2()
    
    with torch.no_grad():
        model.eval()
        
        # Decode theta parameters for composition
        theta1 = model.decode_theta(theta_latent1, dim)
        theta2 = model.decode_theta(theta_latent2, dim)
        
        # Generate predictions using composition
        pred_test = []
        x_test_ = x_test[:, -1].clone()
        
        for step in range(n_output_frames):
            if composition_type == "sum":
                pred, _ = model.solve_ode_with_2_operators(
                    x_test_, theta1, theta2, state_labels, dim, 
                    n_future_steps=1, predict_normed=False, metadata={}
                )
            else:
                current_state = x_test_
                for _ in range(num_steps):
                    pred_int, _ = model.solve_ode(
                        current_state, theta1, state_labels, dim, 
                        n_future_steps=1, integration_time=1/num_steps, dt=1/num_steps, predict_normed=False, metadata={}
                    )
                    pred, _ = model.solve_ode(
                        pred_int[:, -1], theta2, state_labels, dim, 
                        n_future_steps=1, integration_time=1/num_steps, dt=1/num_steps, predict_normed=False, metadata={}
                    )
                    current_state = pred[:, -1]
            
            pred_test.append(pred)
            x_test_ = pred[:, -1]
        
        pred_test = torch.cat(pred_test, 1)
        
        # Calculate error metrics
        total_error = relative_l2_error(pred_test, y_test).item()
        
        # Calculate per-timestep errors
        timestep_errors = []
        for t in range(min(pred_test.shape[1], y_test.shape[1])):
            error = relative_l2_error(pred_test[:, t:t+1], y_test[:, t:t+1]).item()
            timestep_errors.append(error)
        
        # Convert to numpy for saving
        pred_np = pred_test.cpu().numpy()
        target_np = y_test.cpu().numpy()
        input_np = x_test.cpu().numpy()
        
        # Create filename suffix
        suffix = f"adv_{advection_speed:.3f}_diff_{diffusion_coeff:.3f}_opt_{optimizer_type}_steps_{num_steps}"
        
        # Save predictions and targets as .npy files
        np.save(os.path.join(output_dir, f"predictions_{suffix}.npy"), pred_np)
        np.save(os.path.join(output_dir, f"ground_truth_{suffix}.npy"), target_np)
        np.save(os.path.join(output_dir, f"input_{suffix}.npy"), input_np)
        
        # Save error metrics
        error_data = {
            'total_error': total_error,
            'timestep_errors': timestep_errors,
            'advection_speed': advection_speed,
            'diffusion_coeff': diffusion_coeff,
            'n_output_frames': n_output_frames,
            'composition_type': composition_type,
            'num_steps': num_steps,
            'optimizer_type': optimizer_type
        }
        
        # Save error data as CSV
        error_csv_path = os.path.join(output_dir, f"error_metrics_{suffix}.csv")
        with open(error_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])
            writer.writerow(['total_error', total_error])
            writer.writerow(['advection_speed', advection_speed])
            writer.writerow(['diffusion_coeff', diffusion_coeff])
            writer.writerow(['n_output_frames', n_output_frames])
            writer.writerow(['composition_type', composition_type])
            writer.writerow(['num_steps', num_steps])
            writer.writerow(['optimizer_type', optimizer_type])
            
            # Write timestep errors
            for i, error in enumerate(timestep_errors):
                writer.writerow([f'timestep_error_{i}', error])
        
        # Create visualization plots
        n_samples_to_plot = min(4, pred_np.shape[0])  # Plot first 4 samples
        n_timesteps_to_plot = min(8, pred_np.shape[1])  # Plot first 8 timesteps
        
        fig, axes = plt.subplots(n_samples_to_plot, n_timesteps_to_plot, figsize=(20, 10))
        if n_samples_to_plot == 1:
            axes = axes.reshape(1, -1)
        if n_timesteps_to_plot == 1:
            axes = axes.reshape(-1, 1)
            
        for sample_idx in range(n_samples_to_plot):
            for t_idx in range(n_timesteps_to_plot):
                ax = axes[sample_idx, t_idx]
                
                # Plot prediction vs ground truth
                pred_line = pred_np[sample_idx, t_idx, 0, :]
                true_line = target_np[sample_idx, t_idx, 0, :]
                
                x_coords = np.linspace(0, 1, len(pred_line))
                ax.plot(x_coords, pred_line, 'r-', label='Prediction', alpha=0.7)
                ax.plot(x_coords, true_line, 'b-', label='Ground Truth', alpha=0.7)
                
                ax.set_title(f'Sample {sample_idx}, t={t_idx}\nError: {timestep_errors[t_idx]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"predictions_comparison_{suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create error evolution plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(timestep_errors)), timestep_errors, 'bo-', alpha=0.7)
        plt.xlabel('Timestep')
        plt.ylabel('Relative L2 Error')
        plt.title(f'Error Evolution Over Time\nAdv: {advection_speed:.3f}, Diff: {diffusion_coeff:.3f}, Opt: {optimizer_type}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        error_plot_path = os.path.join(output_dir, f"error_evolution_{suffix}.png")
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug files saved to {output_dir}/")
        print(f"  - Predictions: predictions_{suffix}.npy")
        print(f"  - Ground truth: ground_truth_{suffix}.npy")
        print(f"  - Input: input_{suffix}.npy")
        print(f"  - Error metrics: error_metrics_{suffix}.csv")
        print(f"  - Comparison plot: predictions_comparison_{suffix}.png")
        print(f"  - Error evolution plot: error_evolution_{suffix}.png")
        print(f"Total error: {total_error:.6f}")
        
        return {
            'predictions': pred_np,
            'ground_truth': target_np,
            'input': input_np,
            'total_error': total_error,
            'timestep_errors': timestep_errors,
            'error_data': error_data
        }


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
    num_steps=1,
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
    rollout_error = fine_tune_theta_latent(model, theta_latent, inp, target[:, :n_output_frames], state_labels, dim, n_input_frames=n_input_frames, n_output_frames=n_output_frames, epochs=epochs, composition_type="composition", device=device, num_steps=num_steps)
    print(f"Error with finetune: {advection_speed:.3f}x{viscosity:.3f}", rollout_error)

    all_errors.append(rollout_error)

    all_theta_latent = torch.stack(all_theta_latent)
    all_theta = torch.stack(all_theta)
    all_input = torch.stack(all_input)
    all_target = torch.stack(all_target)
    all_errors = np.array(all_errors)

    return all_theta_latent, all_theta, all_input, all_target, all_errors


def run_composition_tests(model, device, cfg, n_points=20, output_csv="composition_test_results.csv", num_steps=1):
    results = []
    # Define parameter ranges from config
    v_min, v_max = cfg.data.v_range
    D_min, D_max = cfg.data.D_range
    advection_range = np.exp(np.linspace(np.log(v_min), np.log(v_max), n_points))
    diffusion_range = np.exp(np.linspace(np.log(D_min), np.log(D_max), n_points))

    # 1. Advection + Advection (viscosity=0)
    for adv1, adv2 in combinations_with_replacement(advection_range, 2):
        adv_sum = adv1 + adv2
        viscosities = [0.0, 0.0, 0.0]
        advection_speeds = [adv1, adv2, adv_sum]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device, num_steps=num_steps
        )
        results.append({
            "scenario": "advection+advection",
            "adv1": f"{adv1:.3f}", "adv2": f"{adv2:.3f}", "adv_sum": f"{adv_sum:.3f}",
            "visc1": f"{0.0:.3f}", "visc2": f"{0.0:.3f}", "visc_sum": f"{0.0:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1],
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3],
            "num_steps": num_steps,
        })

    # 2. Diffusion + Diffusion (advection=0)
    for visc1, visc2 in combinations_with_replacement(diffusion_range, 2):
        visc_sum = visc1 + visc2
        advection_speeds = [0.0, 0.0, 0.0]
        viscosities = [visc1, visc2, visc_sum]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device, num_steps=num_steps
        )
        results.append({
            "scenario": "diffusion+diffusion",
            "adv1": f"{0.0:.3f}", "adv2": f"{0.0:.3f}", "adv_sum": f"{0.0:.3f}",
            "visc1": f"{visc1:.3f}", "visc2": f"{visc2:.3f}", "visc_sum": f"{visc_sum:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1], 
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3],
            "num_steps": num_steps,
        })

    # 3. Advection + Diffusion
    for adv, visc in product(advection_range, diffusion_range):
        advection_speeds = [adv, 0.0, adv]
        viscosities = [0.0, visc, visc]
        all_theta_latent, all_theta, all_input, all_target, all_errors = get_data(
            model, advection_speeds=advection_speeds, viscosities=viscosities, device=device, num_steps=num_steps
        )
        results.append({
            "scenario": "advection+diffusion",
            "adv1": f"{adv:.3f}", "adv2": f"{0.0:.3f}", "adv_sum": f"{adv:.3f}",
            "visc1": f"{0.0:.3f}", "visc2": f"{visc:.3f}", "visc_sum": f"{visc:.3f}",
            "error_encoder1": all_errors[0],
            "error_encoder2": all_errors[1],
            "error_encoder_composed": all_errors[2],
            "error_finetune_composed": all_errors[3],
            "num_steps": num_steps,
        })

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "scenario", "adv1", "adv2", "adv_sum", "visc1", "visc2", "visc_sum",
            "error_encoder1", "error_encoder2", "error_encoder_composed", "error_finetune_composed", "num_steps"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {output_csv}")


def run_optimizer_comparison(model, device, epochs=500, output_csv="optimizer_comparison_results.csv", num_steps=1):
    """
    Compare different finetuning strategies for advection=0.5, diffusion=0.5
    Tests SGD, SGD with momentum, and Adam with 1, 2, and 3 output frames as targets
    """
    results = []
    
    # Fixed parameters
    advection_speed = 0.599
    diffusion_coeff = 1.0 #0.001 #1.0 #0.001
    training_horizons = [1, 5, 10]
    
    # Optimizer configurations
    optimizers = [
        {"name": "SGD", "type": "sgd", "lr": 1e-1},
        {"name": "SGD_Momentum", "type": "sgd_momentum", "lr": 1e-1, "momentum": 0.9},
        {"name": "Adam", "type": "adam", "lr": 1e-1}
    ]
    
    # Target frame counts
    
    
    # Get data once for this specific advection-diffusion pair
    train_ds = TemporalDatasetFixedCI(
        n_batches=1,
        batch_size=128,
        sub_x=1,
        sub_t=1,
        split="test",
        input_frames=16,
        output_frames=34,
        L=16.0,
        nx=256,
        nt=100,
        T=10.0,
        fractal_power=3.0,
        fractal_degree=256,
        v_range=[advection_speed],
        D_range=[diffusion_coeff],
    )
    
    train_loader = DataLoader(train_ds, batch_size=128, num_workers=1, prefetch_factor=1, pin_memory=True, shuffle=False)
    
    # Get batch data
    for batch in train_loader:
        inp, target = batch["input"], batch["target"]
        inp = inp.squeeze(1).to(device)
        target = target.squeeze(1).to(device)
        break
    
    state_labels = torch.tensor([0], device=inp.device)
    x_shape = inp.shape
    B, T, C = x_shape[:3]
    spatial = x_shape[3:]
    dim = len(spatial)
    
    # Get initial theta latent
    with torch.no_grad():
        theta_latent, metadata = model.encode_theta_latent(inp, state_labels)
    
    print(f"Running optimizer comparison for advection={advection_speed}, diffusion={diffusion_coeff}")
    
    # Test all combinations
    for opt_config in optimizers:
        for training_horizon in training_horizons:
            print(f"Testing {opt_config['name']} with {training_horizon} frames...")
            
            # Run fine-tuning
            final_error = fine_tune_theta_latent(
                model=model,
                theta_latent=theta_latent,
                x_test=inp,
                y_test=target,
                state_labels=state_labels,
                dim=dim,
                n_input_frames=16,
                n_output_frames=34,
                epochs=epochs,
                composition_type="composition",
                device=device,
                optimizer_type=opt_config["type"],
                lr=opt_config["lr"],
                momentum=opt_config.get("momentum", 0.9),
                training_horizon=training_horizon,
                num_steps=num_steps,
                advection_speed=advection_speed,
                diffusion_coeff=diffusion_coeff,
                debug_dir=f"{output_csv.replace('.csv', '')}_debug_{opt_config['name']}_h{training_horizon}"
            )
            
            results.append({
                "optimizer": opt_config["name"],
                "training_horizon": training_horizon,
                "advection": advection_speed,
                "diffusion": diffusion_coeff,
                "final_error": final_error,
                "learning_rate": opt_config["lr"],
                "momentum": opt_config.get("momentum", "N/A"),
                "num_steps": num_steps
            })
            
            print(f"{opt_config['name']} with {training_horizon} frames: {final_error:.6f}")
    
    # Save results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["optimizer", "training_horizon", "advection", "diffusion", "final_error", "learning_rate", "momentum", "num_steps"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Optimizer comparison results saved to {output_csv}")
    
    # Create and print results matrix
    print("\nResults Matrix:")
    print("Optimizer\\Target Frames", end="")
    for tf in training_horizons:
        print(f"\t{tf}", end="")
    print()
    
    for opt_config in optimizers:
        print(f"{opt_config['name']}", end="")
        for tf in training_horizons:
            error = next(r["final_error"] for r in results 
                        if r["optimizer"] == opt_config["name"] and r["training_horizon"] == tf)
            print(f"\t{error:.6f}", end="")
        print()


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # load model
    device="cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = cfg.test.dataset_name
    ckpt_time = cfg.test.ckpt_time
    run_name = cfg.test.get('run_name', None)
    if run_name is None:
        raise ValueError("run_name is required but not found in config")
    
    output_dir = f"{cfg.test.results_dir}/{dataset_name}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = cfg.test.ckpt_path
    print(f"Loading model from {ckpt_path}...")
    model = DISCOLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.model.to(device)
    model.eval()

    # Run the optimizer comparison experiment
    epochs=100
    num_steps = cfg.test.num_steps
    run_optimizer_comparison(model, device, epochs, output_csv=f"{output_dir}/optimizer_comparison_results.csv", num_steps=num_steps)

if __name__ == "__main__":
    main()
