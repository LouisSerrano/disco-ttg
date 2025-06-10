import numpy as np
import torch
import os
from advection_diffusion import (
    Fractaloid, FourierSmooth, WhiteNoise, Dirac, Rectangle, Triangle, HalfEllipse, Sine, GaussianMixtures, SmoothPlateau, AdvectionDiffusion
)

# Settings
data_dir = "/mnt/home/lserrano/disco-ball/datasets"
os.makedirs(data_dir, exist_ok=True)

total_simulation_time = 4
length_of_domain = 16.0
number_of_time_steps = 40001
number_of_snapshots = 50
spatial_size = 512
batch_size = 100  # for generation

# Initial conditions
initial_conditions = [
    (Fractaloid, dict(degree=spatial_size, power=2.0, size=spatial_size, patch_size=spatial_size)),
    #(FourierSmooth, dict(degree=spatial_size, smoothness=0.7, size=spatial_size, patch_size=spatial_size)),
    #(WhiteNoise, dict(amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
    #(Dirac, dict(amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
    #(Rectangle, dict(end=32, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
    #(Triangle, dict(center=16, end=32, amplitude=1.0, nb_triangles=2, size=spatial_size, patch_size=spatial_size)),
    #(HalfEllipse, dict(diameter=32, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
    #(Sine, dict(periods=3, amplitude=1.0, size=spatial_size, patch_size=spatial_size)),
    #(Sine, dict(periods=3, amplitude=1.0, L=length_of_domain, num_points=spatial_size)),
    #(GaussianMixtures, dict(n_gaussians=3, size=spatial_size, patch_size=spatial_size)),
    #(SmoothPlateau, dict(width_ratio=0.5, pattern_size=32, size=spatial_size, patch_size=spatial_size)),
]
initial_condition_names = [ic[0].__name__ for ic in initial_conditions]

# Parameter grids
train_velocities = [0.01, 0.05, 0.1, 0.5, 1]
train_diffusivities = [0.01, 0.5, 0.1, 0.5, 1]
test_velocities = train_velocities

test_diffusivities = train_diffusivities

# Dataset sizes
n_train = 200
n_val = 100
n_test = 100

def sample_initial_condition():
    ic_idx = np.random.randint(len(initial_conditions))
    ic_cls, ic_kwargs = initial_conditions[ic_idx]
    ic = ic_cls(**ic_kwargs)
    u0 = ic.generate(1, None)[0]  # shape: [C, H] or [H]
    return u0, ic_idx

def generate_set(n_samples, mode):
    trajectories = []
    velocities = []
    diffusivities = []
    ic_indices = []
    #t = torch.arange(time_steps)
    t = torch.linspace(0, total_simulation_time, number_of_snapshots)
    if mode == 'train':
        # Half with velocity, half with diffusivity
        n_half = n_samples // 2
        # Velocity, diffusivity=0
        for v in train_velocities:
            n = n_half // len(train_velocities)
            for _ in range(0, n, batch_size):
                b = min(batch_size, n - _)
                u0s = []
                ic_idxs = []
                for _ in range(b):
                    u0, ic_idx = sample_initial_condition()
                    u0s.append(u0)
                    ic_idxs.append(ic_idx)
                u0s = torch.stack(u0s)[:, None, :]
                op = AdvectionDiffusion(velocity=v,
                                        diffusivity=0.0,)
                                        #length_of_domain=length_of_domain,
                                        #total_simulation_time=total_simulation_time,
                                        #number_of_time_steps=number_of_time_steps,
                                        #number_of_snapshots=number_of_snapshots)
                ut = op(t, u0s).squeeze(2)
                trajectories.append(ut.cpu().numpy())
                velocities.extend([v]*b)
                diffusivities.extend([0.0]*b)
                ic_indices.extend(ic_idxs)
        # Diffusivity, velocity=0
        for d in train_diffusivities:
            n = n_half // len(train_diffusivities)
            for _ in range(0, n, batch_size):
                b = min(batch_size, n - _)
                u0s = []
                ic_idxs = []
                for _ in range(b):
                    u0, ic_idx = sample_initial_condition()
                    u0s.append(u0)
                    ic_idxs.append(ic_idx)
                u0s = torch.stack(u0s)[:, None, :]
                op = AdvectionDiffusion(velocity=0.0, diffusivity=d,)
                                        #length_of_domain=length_of_domain)
                                        #total_simulation_time=total_simulation_time,
                                        #number_of_time_steps=number_of_time_steps,
                                       # number_of_snapshots=number_of_snapshots))
                ut = op(t, u0s).squeeze(2)
                trajectories.append(ut.cpu().numpy())
                velocities.extend([0.0]*b)
                diffusivities.extend([d]*b)
                ic_indices.extend(ic_idxs)
    elif mode == 'val':
        # Use same as train, but fewer samples
        n_half = n_samples // 2
        for v in train_velocities:
            n = n_half // len(train_velocities)
            for _ in range(0, n, batch_size):
                b = min(batch_size, n - _)
                u0s = []
                ic_idxs = []
                for _ in range(b):
                    u0, ic_idx = sample_initial_condition()
                    u0s.append(u0)
                    ic_idxs.append(ic_idx)
                u0s = torch.stack(u0s)[:, None, :]
                op = AdvectionDiffusion(velocity=v,
                                        diffusivity=0.0,)
                                        #length_of_domain=length_of_domain,
                                        #total_simulation_time=total_simulation_time,
                                        #number_of_time_steps=number_of_time_steps,
                                       # number_of_snapshots=number_of_snapshots)
                ut = op(t, u0s).squeeze(2)
                trajectories.append(ut.cpu().numpy())
                velocities.extend([v]*b)
                diffusivities.extend([0.0]*b)
                ic_indices.extend(ic_idxs)
        for d in train_diffusivities:
            n = n_half // len(train_diffusivities)
            for _ in range(0, n, batch_size):
                b = min(batch_size, n - _)
                u0s = []
                ic_idxs = []
                for _ in range(b):
                    u0, ic_idx = sample_initial_condition()
                    u0s.append(u0)
                    ic_idxs.append(ic_idx)
                u0s = torch.stack(t, u0s)[:, None, :]
                op = AdvectionDiffusion(velocity=0.0,
                                        diffusivity=d,)
                                        #length_of_domain=length_of_domain,
                                        #total_simulation_time=total_simulation_time,
                                        #number_of_time_steps=number_of_time_steps,
                                        #number_of_snapshots=number_of_snapshots)
                ut = op(u0s).squeeze(2)
                trajectories.append(ut.cpu().numpy())
                velocities.extend([0.0]*b)
                diffusivities.extend([d]*b)
                ic_indices.extend(ic_idxs)
    elif mode == 'test':
        # Cardinal product
        n_per_combo = n_samples // (len(test_velocities) * len(test_diffusivities))
        for v in test_velocities:
            for d in test_diffusivities:
                for _ in range(0, n_per_combo, batch_size):
                    b = min(batch_size, n_per_combo - _)
                    u0s = []
                    ic_idxs = []
                    for _ in range(b):
                        u0, ic_idx = sample_initial_condition()
                        u0s.append(u0)
                        ic_idxs.append(ic_idx)
                    u0s = torch.stack(u0s)[:, None, :]
                    op = AdvectionDiffusion(velocity=v,
                                        diffusivity=d, )
                                        #length_of_domain=length_of_domain,
                                        #total_simulation_time=total_simulation_time,
                                        #number_of_time_steps=number_of_time_steps,
                                        #number_of_snapshots=number_of_snapshots)
                    ut = op(t, u0s).squeeze(2)
                    trajectories.append(ut.cpu().numpy())
                    velocities.extend([v]*b)
                    diffusivities.extend([d]*b)
                    ic_indices.extend(ic_idxs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Concatenate
    trajectories = np.concatenate(trajectories, axis=0)
    velocities = np.array(velocities)
    diffusivities = np.array(diffusivities)
    ic_indices = np.array(ic_indices)
    return dict(
        trajectories=trajectories,
        velocities=velocities,
        diffusivities=diffusivities,
        initial_condition_indices=ic_indices,
        initial_condition_names=initial_condition_names,
    )

if __name__ == "__main__":
    print("Generating train set...")
    train = generate_set(n_train, 'train')
    np.savez_compressed(os.path.join(data_dir, "train.npz"), **train)
    print("Generating val set...")
    val = generate_set(n_val, 'val')
    np.savez_compressed(os.path.join(data_dir, "val.npz"), **val)
    print("Generating test set...")
    test = generate_set(n_test, 'test')
    np.savez_compressed(os.path.join(data_dir, "test.npz"), **test)
    print("Done.") 