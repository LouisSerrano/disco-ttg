import numpy as np
import torch
import os
from itertools import product
from advection_diffusion import (
    Fractaloid, FourierSmooth, WhiteNoise, Dirac, Rectangle, Triangle, HalfEllipse, Sine, GaussianMixtures, SmoothPlateau, AdvectionDiffusionExplicit, AdvectionDiffusionLaxWendroff, AdvectionDiffusionImplicit, calculate_numbers
)
from advection_diffusion_ps import AdvectionDiffusionPseudospectral

# Settings
data_dir = "/mnt/home/lserrano/disco-ball/datasets"
os.makedirs(data_dir, exist_ok=True)

total_simulation_time = 10
length_of_domain = 16.0
number_of_snapshots = 100
spatial_size = 256 #8192 #1024
number_of_time_steps = total_simulation_time*10000+1 #40001
n_unique_ics = 10 #1000
n_test_ics = 10 #100  # 10% of n_unique_ics
print(os.cpu_count())
batch_size = min(50, n_test_ics)  # Use n_test_ics as the batch size for all splits

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
train_diffusivities = [0.001, 0.01, 0.05, 0.1, 0.5]
val_velocities = [0.03, 0.075, 0.3, 0.75, 1.25]
val_diffusivities = [0.005, 0.03, 0.075, 0.3, 0.75]
test_velocities = train_velocities
test_diffusivities = train_diffusivities

fixed_initial_conditions = True  # Set to False for random ICs per split
fixed_ic_seed = 12345           # Seed for reproducibility

# Pre-generate ICs if needed
def generate_fixed_initial_conditions(n_total, seed):
    rng = np.random.RandomState(seed)
    ic_list = []
    for _ in range(n_total):
        ic_idx = rng.randint(len(initial_conditions))
        ic_cls, ic_kwargs = initial_conditions[ic_idx]
        ic = ic_cls(**ic_kwargs)
        u0 = ic.generate(1, None)[0]
        ic_list.append((u0, ic_idx))
    return ic_list

if fixed_initial_conditions:
    # Generate 1000 unique ICs for train/val/test
    all_ics = generate_fixed_initial_conditions(n_unique_ics, fixed_ic_seed)
    # For test, randomly subsample 100 ICs
    rng = np.random.RandomState(fixed_ic_seed + 1)
    test_ic_indices = rng.choice(n_unique_ics, n_test_ics, replace=False)
    test_ics = [all_ics[i] for i in test_ic_indices]
    train_ics = all_ics
    val_ics = all_ics
    # Parameter settings
    train_param_grid = [(v, 0.0) for v in train_velocities] + [(0.0, d) for d in train_diffusivities]
    val_param_grid = [(v, 0.0) for v in val_velocities] + [(0.0, d) for d in val_diffusivities]
    test_param_grid = list(product(test_velocities, test_diffusivities))
    n_train = len(train_ics) * len(train_param_grid)
    n_val = len(val_ics) * len(val_param_grid)
    n_test = len(test_ics) * len(test_param_grid)
else:
    train_ics = val_ics = test_ics = None
    train_param_grid = [(v, 0.0) for v in train_velocities] + [(0.0, d) for d in train_diffusivities]
    val_param_grid = [(v, 0.0) for v in val_velocities] + [(0.0, d) for d in val_diffusivities]
    test_param_grid = list(product(test_velocities, test_diffusivities))
    n_train = 200
    n_val = 100
    n_test = 100

def generate_set(ics, param_grid):
    trajectories = []
    velocities = []
    diffusivities = []
    ic_indices = []
    t = torch.linspace(0, total_simulation_time, number_of_snapshots)
    n_ics = len(ics)
    for (v, d) in param_grid:
        # Batch over ICs
        for batch_start in range(0, n_ics, batch_size):
            batch_ics = ics[batch_start:batch_start+batch_size]
            u0s = torch.stack([u0 for (u0, _) in batch_ics])  # shape [batch, spatial_size]
            ic_ids = [ic_id for (_, ic_id) in batch_ics]
            op = AdvectionDiffusionPseudospectral( #AdvectionDiffusionExplicit(
                velocity=v,
                diffusivity=d,
                length_of_domain=length_of_domain,
                total_simulation_time=total_simulation_time,
                number_of_time_steps=number_of_time_steps,
                number_of_snapshots=number_of_snapshots
            )
            ut = op(u0s)  # shape [batch, n_snapshots, spatial_size]
            trajectories.append(ut.cpu().numpy())
            velocities.extend([v] * len(batch_ics))
            diffusivities.extend([d] * len(batch_ics))
            ic_indices.extend(ic_ids)
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

    print(f"Generating train set with {len(train_ics)} ICs and {len(train_param_grid)} param settings...")
    train_results = calculate_numbers(train_velocities, train_diffusivities, "Train", dt=total_simulation_time/number_of_time_steps, dx=length_of_domain/spatial_size)
    train = generate_set(train_ics, train_param_grid)
    np.savez_compressed(os.path.join(data_dir, "train_ps.npz"), **train)

    print(f"Generating val set with {len(val_ics)} ICs and {len(val_param_grid)} param settings...")
    val_results = calculate_numbers(val_velocities, val_diffusivities, "Validation",  dt=total_simulation_time/number_of_time_steps, dx=length_of_domain/spatial_size)
    val = generate_set(val_ics, val_param_grid)
    np.savez_compressed(os.path.join(data_dir, "val_ps.npz"), **val)

    print(f"Generating test set with {len(test_ics)} ICs and {len(test_param_grid)} param settings...")
    test_results = calculate_numbers(test_velocities, test_diffusivities, "Test",  dt=total_simulation_time/number_of_time_steps, dx=length_of_domain/spatial_size)
    test = generate_set(test_ics, test_param_grid)
    np.savez_compressed(os.path.join(data_dir, "test_ps.npz"), **test)
    print("Done.") 