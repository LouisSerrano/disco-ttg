import numpy as np
import torch
import time
from src.utils.advection_diffusion import Fractaloid

def generate_fractaloid_batch(batch_size, powers, degree, size, patch_size, rng=None, profile=False):
    """
    Vectorized fractaloid generation for multiple power values.
    
    Parameters:
        batch_size (int): Number of samples to generate
        powers (ndarray): Power values for each sample, shape (batch_size,)
        degree (int): Fourier degree
        size (int): Size of the generated signals
        patch_size (int): Patch size
        rng: Random number generator
        profile (bool): Whether to print timing information
        
    Returns:
        ndarray: Generated normalized fractaloids, shape (batch_size, size)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if profile:
        total_start = time.time()
        print(f"=== Profiling generate_fractaloid_batch (batch_size={batch_size}) ===")
    
    # Generate phases for Fourier modes
    if profile:
        start = time.time()
    phase = np.linspace(0, 2*np.pi, patch_size, endpoint=False)
    phase = np.arange(1, degree+1)[:, None] * phase[None, :]  # shape: (degree, patch_size)
    if profile:
        print(f"Phase generation: {(time.time() - start)*1000:.2f}ms")
    
    # Generate random coefficients for all samples at once
    if profile:
        start = time.time()
    proj = rng.standard_normal((batch_size, degree))  # shape: (batch_size, degree)
    if profile:
        print(f"Random coefficient generation: {(time.time() - start)*1000:.2f}ms")
    
    # Apply power spectrum for each sample
    if profile:
        start = time.time()
    # powers: (batch_size,) -> (batch_size, 1)
    # np.arange(1, degree+1): (degree,) -> (1, degree)
    power_spectrum = np.arange(1, degree+1)[None, :] ** (-powers[:, None])  # (batch_size, degree)
    proj = proj * power_spectrum  # (batch_size, degree)
    if profile:
        print(f"Power spectrum application: {(time.time() - start)*1000:.2f}ms")
    
    # Generate signals: proj @ sin(phase)
    if profile:
        start = time.time()
    # proj: (batch_size, degree), sin(phase): (degree, patch_size)
    # Result: (batch_size, patch_size)
    u0_batch = proj @ np.sin(phase)
    if profile:
        print(f"Matrix multiplication (proj @ sin): {(time.time() - start)*1000:.2f}ms")
    
    # Pad to full size if needed
    if profile:
        start = time.time()
    if size > patch_size:
        u0_full = np.zeros((batch_size, size))
        u0_full[:, :patch_size] = u0_batch
        u0_batch = u0_full
    if profile:
        print(f"Padding to full size: {(time.time() - start)*1000:.2f}ms")
    
    # Apply random shifts and signs (vectorized)
    if profile:
        start = time.time()
    # Use patch_size for shifts within patches, then multiply by number of patches for across-patch shifts
    shift1 = rng.integers(0, max(1, patch_size - patch_size), batch_size)  # within patch
    shift2 = patch_size * rng.integers(0, size // patch_size, batch_size)  # across patches
    shifts = (shift1 + shift2) % size
    signs = (rng.standard_normal(batch_size) > 0) * 2 - 1
    if profile:
        print(f"Shift and sign generation: {(time.time() - start)*1000:.2f}ms")
    
    # Apply shifts and signs using vectorized operations
    if profile:
        start = time.time()
    for i in range(batch_size):
        u0_batch[i] = np.roll(u0_batch[i], shifts[i])
        u0_batch[i] *= signs[i]
    if profile:
        print(f"Apply shifts and signs (loop): {(time.time() - start)*1000:.2f}ms")
    
    # Normalize each sample
    if profile:
        start = time.time()
    for i in range(batch_size):
        u0_batch[i] = (u0_batch[i] - u0_batch[i].mean()) / (u0_batch[i].std() + 1e-8)
    if profile:
        print(f"Normalization (loop): {(time.time() - start)*1000:.2f}ms")
        print(f"Total time: {(time.time() - total_start)*1000:.2f}ms")
        print()
    
    return u0_batch

def advection_diffusion_analytical_vectorized(u0_batch, L=16.0, v_batch=None, D_batch=None, nt=100, T=10.0, profile=False):
    """
    Vectorized analytical solution for multiple initial conditions.
    
    Parameters:
        u0_batch (ndarray): Initial conditions, shape (batch_size, nx)
        L (float): Domain length
        v_batch (ndarray): Advection speeds, shape (batch_size,)
        D_batch (ndarray): Diffusion coefficients, shape (batch_size,)
        nt (int): Number of time steps
        T (float): Final time
        profile (bool): Whether to print timing information
    
    Returns:
        u_xt_batch (ndarray): Solution array of shape (batch_size, nt, nx)
    """
    if profile:
        total_start = time.time()
        print(f"=== Profiling advection_diffusion_analytical_vectorized (batch_size={u0_batch.shape[0]}) ===")
    
    batch_size, nx = u0_batch.shape
    t = np.linspace(0, T, nt)
    
    # Fourier wavenumbers (same for all batch items)
    if profile:
        start = time.time()
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    k = 1j * k  # shape: (nx,)
    if profile:
        print(f"Wavenumber generation: {(time.time() - start)*1000:.2f}ms")
    
    # FFT of all initial conditions
    if profile:
        start = time.time()
    u0_hat_batch = np.fft.fft(u0_batch, axis=1)  # shape: (batch_size, nx)
    if profile:
        print(f"Initial FFT: {(time.time() - start)*1000:.2f}ms")
    
    # Allocate solution array
    if profile:
        start = time.time()
    u_xt_batch = np.zeros((batch_size, nt, nx))
    if profile:
        print(f"Solution array allocation: {(time.time() - start)*1000:.2f}ms")
    
    # Reshape parameters for broadcasting
    if profile:
        start = time.time()
    v_batch = v_batch.reshape(-1, 1)  # shape: (batch_size, 1)
    D_batch = D_batch.reshape(-1, 1)  # shape: (batch_size, 1)
    k_reshaped = k.reshape(1, -1)     # shape: (1, nx)
    if profile:
        print(f"Parameter reshaping: {(time.time() - start)*1000:.2f}ms")
    
    # Fully vectorized time evolution for all batch items and time steps simultaneously
    if profile:
        start = time.time()
    
    # Reshape time array for broadcasting: (nt,) -> (1, nt, 1)
    t_reshaped = t.reshape(1, -1, 1)  # shape: (1, nt, 1)
    
    # Compute decay for all times simultaneously
    # D_batch: (batch_size, 1), k_reshaped: (1, nx), t_reshaped: (1, nt, 1)
    # Broadcasting: (batch_size, 1, 1) * (1, 1, nx) * (1, nt, 1) -> (batch_size, nt, nx)
    
    decay = np.exp(D_batch[:, None, :] * (k_reshaped[None, None, :]**2) * t_reshaped)
    decay *= np.exp(-k_reshaped[None, None, :] * v_batch[:, None, :] * t_reshaped)
    
    # Apply decay to all time steps: (batch_size, 1, nx) * (batch_size, nt, nx) -> (batch_size, nt, nx)
    u_hat_t_batch = u0_hat_batch[:, None, :] * decay
    
    # IFFT for all time steps simultaneously
    u_xt_batch = np.fft.ifft(u_hat_t_batch, axis=2).real
    
    if profile:
        total_loop_time = time.time() - start
        print(f"Time evolution loop total: {total_loop_time*1000:.2f}ms")
        #print(f"  - Decay computation: {decay_time*1000:.2f}ms ({decay_time/total_loop_time*100:.1f}%)")
        #print(f"  - IFFT computation: {ifft_time*1000:.2f}ms ({ifft_time/total_loop_time*100:.1f}%)")
        print(f"Total time: {(time.time() - total_start)*1000:.2f}ms")
        print()
    
    return u_xt_batch

def generate_batch_vectorized(batch_size, nx=256, nt=100, T=10.0, L=16.0, 
                            v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                            fractal_degree=8, fractal_power_range=(1.5, 2.5), 
                            split='train', rng=None, profile=False):
    """Vectorized batch generation"""
    if rng is None:
        rng = np.random.default_rng()
    
    if profile:
        total_start = time.time()
        print(f"=== Profiling generate_batch_vectorized (batch_size={batch_size}) ===")
    
    # Generate all parameters at once
    if profile:
        start = time.time()
    if split == 'train':
        # 50% advection only, 50% diffusion only
        is_advection = rng.random(batch_size) < 0.5
        v_batch = np.where(is_advection, rng.uniform(*v_range, batch_size), 0.0)
        D_batch = np.where(~is_advection, rng.uniform(*D_range, batch_size), 0.0)
    else:
        v_batch = rng.uniform(*v_range, batch_size)
        D_batch = rng.uniform(*D_range, batch_size)
    
    # Generate fractal powers for main and context trajectories
    fractal_powers_main = rng.uniform(*fractal_power_range, batch_size)
    fractal_powers_ctx = rng.uniform(*fractal_power_range, batch_size)
    if profile:
        print(f"Parameter generation: {(time.time() - start)*1000:.2f}ms")
    
    # Generate all initial conditions using vectorized fractaloid generation
    if profile:
        start = time.time()
    u0_batch_main = generate_fractaloid_batch(
        batch_size=batch_size,
        powers=fractal_powers_main,
        degree=fractal_degree,
        size=nx,
        patch_size=nx,
        rng=rng,
        profile=profile
    )
    if profile:
        main_fractal_time = time.time() - start
        start = time.time()
    u0_batch_ctx = generate_fractaloid_batch(
        batch_size=batch_size,
        powers=fractal_powers_ctx,
        degree=fractal_degree,
        size=nx,
        patch_size=nx,
        rng=rng,
        profile=profile
    )
    if profile:
        ctx_fractal_time = time.time() - start
        print(f"Fractaloid generation total: {(main_fractal_time + ctx_fractal_time)*1000:.2f}ms")
        print(f"  - Main trajectories: {main_fractal_time*1000:.2f}ms")
        print(f"  - Context trajectories: {ctx_fractal_time*1000:.2f}ms")
    
    # Solve all analytical solutions at once
    if profile:
        start = time.time()
    u_xt_batch_main = advection_diffusion_analytical_vectorized(
        u0_batch_main, L=L, v_batch=v_batch, D_batch=D_batch, nt=nt, T=T, profile=profile
    )
    if profile:
        main_solve_time = time.time() - start
        start = time.time()
    u_xt_batch_ctx = advection_diffusion_analytical_vectorized(
        u0_batch_ctx, L=L, v_batch=v_batch, D_batch=D_batch, nt=nt, T=T, profile=profile
    )
    if profile:
        ctx_solve_time = time.time() - start
        print(f"Analytical solution total: {(main_solve_time + ctx_solve_time)*1000:.2f}ms")
        print(f"  - Main trajectories: {main_solve_time*1000:.2f}ms")
        print(f"  - Context trajectories: {ctx_solve_time*1000:.2f}ms")
        print(f"TOTAL BATCH GENERATION TIME: {(time.time() - total_start)*1000:.2f}ms")
        print("=" * 60)
    
    return u_xt_batch_main, u_xt_batch_ctx

def generate_batch_sequential(batch_size, nx=256, nt=100, T=10.0, L=16.0,
                            v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                            fractal_degree=8, fractal_power_range=(1.5, 2.5),
                            split='train', rng=None):
    """Original sequential batch generation for comparison"""
    if rng is None:
        rng = np.random.default_rng()
    
    from train.train import advection_diffusion_analytical
    
    u_xt_batch_main = []
    u_xt_batch_ctx = []
    
    for _ in range(batch_size):
        # Sample parameters
        if split == 'train':
            if rng.random() < 0.5:
                v = rng.uniform(*v_range)
                D = 0
            else:
                v = 0
                D = rng.uniform(*D_range)
        else:
            v = rng.uniform(*v_range)
            D = rng.uniform(*D_range)
        
        # Main trajectory
        fractal_power = rng.uniform(*fractal_power_range)
        fractaloid = Fractaloid(
            degree=fractal_degree,
            power=fractal_power,
            size=nx,
            patch_size=nx
        )
        u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
        u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
        u_xt, _, _ = advection_diffusion_analytical(u0, L=L, v=v, D=D, nt=nt, T=T)
        u_xt_batch_main.append(u_xt)
        
        # Context trajectory
        fractal_power_ctx = rng.uniform(*fractal_power_range)
        fractaloid_ctx = Fractaloid(
            degree=fractal_degree,
            power=fractal_power_ctx,
            size=nx,
            patch_size=nx
        )
        u0_ctx = fractaloid_ctx.generate(batch_size=1, seed=None).squeeze(0).numpy()
        u0_ctx = (u0_ctx - u0_ctx.mean()) / (u0_ctx.std() + 1e-8)
        u_xt_ctx, _, _ = advection_diffusion_analytical(u0_ctx, L=L, v=v, D=D, nt=nt, T=T)
        u_xt_batch_ctx.append(u_xt_ctx)
    
    return np.stack(u_xt_batch_main), np.stack(u_xt_batch_ctx)

def benchmark_methods():
    """Benchmark vectorized vs sequential methods"""
    import matplotlib.pyplot as plt
    
    print("Benchmarking data generation methods...")
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    n_trials = 3
    
    speedups = []
    seq_times = []
    vec_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Sequential timing
        sequential_times = []
        for trial in range(n_trials):
            start_time = time.time()
            u_main_seq, u_ctx_seq = generate_batch_sequential(
                batch_size, rng=np.random.default_rng(42 + trial)
            )
            sequential_times.append(time.time() - start_time)
        
        # Vectorized timing
        vectorized_times = []
        for trial in range(n_trials):
            start_time = time.time()
            u_main_vec, u_ctx_vec = generate_batch_vectorized(
                batch_size, rng=np.random.default_rng(42 + trial)
            )
            vectorized_times.append(time.time() - start_time)
        
        seq_mean = np.mean(sequential_times)
        vec_mean = np.mean(vectorized_times)
        speedup = seq_mean / vec_mean
        
        speedups.append(speedup)
        seq_times.append(seq_mean)
        vec_times.append(vec_mean)
        
        print(f"Sequential: {seq_mean:.3f}s ± {np.std(sequential_times):.3f}s")
        print(f"Vectorized: {vec_mean:.3f}s ± {np.std(vectorized_times):.3f}s")
        print(f"Speedup: {speedup:.2f}x")
    
    # Print summary
    print(f"\n=== SPEEDUP SUMMARY ===")
    for i, batch_size in enumerate(batch_sizes):
        print(f"Batch size {batch_size:2d}: {speedups[i]:.2f}x speedup")
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    
    # Create speedup plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, speedups, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (x)')
    plt.title('Vectorized vs Sequential Speedup')
    plt.grid(True, alpha=0.3)
    plt.xticks(batch_sizes)
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, seq_times, 'ro-', label='Sequential', linewidth=2, markersize=8)
    plt.plot(batch_sizes, vec_times, 'bo-', label='Vectorized', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(batch_sizes)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSpeedup plot saved as 'speedup_comparison.png'")
    
    # Generate and plot sample trajectories for visual comparison
    print(f"\nGenerating sample trajectories for visual verification...")
    rng_test = np.random.default_rng(999)
    u_main_seq_test, u_ctx_seq_test = generate_batch_sequential(4, rng=rng_test)
    rng_test = np.random.default_rng(999)
    u_main_vec_test, u_ctx_vec_test = generate_batch_vectorized(4, rng=rng_test)
    
    print(f"Max difference (main): {np.max(np.abs(u_main_seq_test - u_main_vec_test)):.2e}")
    print(f"Max difference (ctx): {np.max(np.abs(u_ctx_seq_test - u_ctx_vec_test)):.2e}")
    
    # Plot sample trajectories
    plt.figure(figsize=(15, 10))
    
    for i in range(4):
        # Main trajectories
        plt.subplot(4, 4, i*4 + 1)
        plt.imshow(u_main_seq_test[i], aspect='auto', cmap='RdBu_r')
        plt.title(f'Main {i+1} (Sequential)')
        plt.colorbar()
        
        plt.subplot(4, 4, i*4 + 2)
        plt.imshow(u_main_vec_test[i], aspect='auto', cmap='RdBu_r')
        plt.title(f'Main {i+1} (Vectorized)')
        plt.colorbar()
        
        # Context trajectories  
        plt.subplot(4, 4, i*4 + 3)
        plt.imshow(u_ctx_seq_test[i], aspect='auto', cmap='RdBu_r')
        plt.title(f'Context {i+1} (Sequential)')
        plt.colorbar()
        
        plt.subplot(4, 4, i*4 + 4)
        plt.imshow(u_ctx_vec_test[i], aspect='auto', cmap='RdBu_r')
        plt.title(f'Context {i+1} (Vectorized)')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Trajectory comparison saved as 'trajectory_comparison.png'")
    
    return speedups

def profile_vectorized_functions():
    """Profile individual vectorized functions to identify bottlenecks"""
    print("Profiling vectorized functions...")
    
    batch_sizes = [16, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"{'='*80}")
        
        # Test with profiling enabled
        rng = np.random.default_rng(42)
        u_main, u_ctx = generate_batch_vectorized(
            batch_size, 
            rng=rng, 
            profile=True
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile_vectorized_functions()
    else:
        benchmark_methods()
