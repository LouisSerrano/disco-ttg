"""
Generate advection-diffusion test data with a nonlinear Burgers-like perturbation.

PDE: ∂u/∂t + v ∂u/∂x + ε u ∂u/∂x = D ∂²u/∂x²

When ε=0, this recovers the standard linear advection-diffusion (paper setting).
As ε grows, the nonlinear advection term introduces steepening and potential shocks.
The diffusion term D provides regularization.

We use a pseudo-spectral method: linear terms in Fourier space, nonlinear term
(ε u ∂u/∂x) computed in physical space, integrated with ETDRK4 (exponential
time-differencing Runge-Kutta 4th order) for stability.

Addresses reviewers rJBq W2, SLt5 W2, UJ7L on imperfect decomposability.
"""
import numpy as np
import torch
import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils.advection_diffusion import FractaloidPhase


def solve_burgers_advdiff(u0, L=16.0, v=0.5, D=0.3, epsilon=0.0, nt=100, T=10.0,
                          dt_internal=None, dealias=True):
    """
    Solve ∂u/∂t + v ∂u/∂x + ε u ∂u/∂x = D ∂²u/∂x² on [0, L) periodic domain.

    Uses pseudo-spectral spatial discretization with RK4 time-stepping.

    Parameters:
        u0: Initial condition, shape (nx,)
        L: Domain length
        v: Linear advection speed
        D: Diffusion coefficient
        epsilon: Nonlinear advection strength (0 = linear)
        nt: Number of output time steps
        T: Final time
        dt_internal: Internal time step (auto-computed if None)
        dealias: Apply 2/3 dealiasing rule

    Returns:
        u_xt: Solution array, shape (nt, nx)
        x: Spatial grid, shape (nx,)
        t_out: Output time grid, shape (nt,)
        diagnostics: dict with max |u|, energy, etc.
    """
    nx = len(u0)
    x = np.linspace(0, L, nx, endpoint=False)
    t_out = np.linspace(0, T, nt)

    # Fourier wavenumbers
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi  # physical wavenumbers
    ik = 1j * k  # for derivatives: d/dx -> ik in Fourier space

    # Dealiasing mask (2/3 rule)
    if dealias:
        k_max = nx // 3
        dealias_mask = np.abs(np.fft.fftfreq(nx, d=1.0/nx)) <= k_max
    else:
        dealias_mask = np.ones(nx, dtype=bool)

    # Auto-compute internal time step (CFL-like)
    if dt_internal is None:
        dx = L / nx
        u_max = max(np.abs(u0).max(), 1.0)
        # CFL for advection + nonlinear term
        dt_adv = 0.5 * dx / (abs(v) + epsilon * u_max + 1e-10)
        # Stability for diffusion
        dt_diff = 0.25 * dx**2 / (D + 1e-10)
        dt_internal = min(dt_adv, dt_diff, T / (4 * nt))

    # Linear operator in Fourier space: L = -v*ik + D*(ik)^2 = -v*ik - D*k^2
    L_op = -v * ik - D * k**2

    def nonlinear_rhs(u_hat):
        """Compute nonlinear term -ε * ∂(u²/2)/∂x in Fourier space."""
        if epsilon == 0:
            return np.zeros_like(u_hat)
        # Transform to physical space
        u_phys = np.fft.ifft(u_hat * dealias_mask).real
        # Compute u² in physical space
        u_sq = u_phys ** 2
        # Transform back and differentiate
        u_sq_hat = np.fft.fft(u_sq)
        # -ε/2 * ∂(u²)/∂x = -ε/2 * ik * FFT(u²)
        return -epsilon * 0.5 * ik * u_sq_hat

    # Allocate output
    u_xt = np.zeros((nt, nx))
    u_xt[0] = u0.copy()

    # RK4 integration (IMEX-like: exact integration for linear part + RK4 for nonlinear)
    u_hat = np.fft.fft(u0)
    current_t = 0.0
    output_idx = 1

    diagnostics = {
        'max_u': [float(np.abs(u0).max())],
        'energy': [float(np.sum(u0**2) * L / nx)],
    }

    while output_idx < nt:
        # Time step to next output or internal step
        dt = min(dt_internal, t_out[output_idx] - current_t)

        # Exponential factor for linear part
        E = np.exp(L_op * dt)
        E2 = np.exp(L_op * dt / 2)

        # ETDRK4-like: RK4 stages with exponential integration of linear part
        # Stage 1
        N1 = nonlinear_rhs(u_hat)
        a = E2 * u_hat + (E2 - 1) / (L_op + 1e-30) * N1

        # Stage 2
        N2 = nonlinear_rhs(a)
        b = E2 * u_hat + (E2 - 1) / (L_op + 1e-30) * N2

        # Stage 3
        N3 = nonlinear_rhs(b)
        c = E2 * a + (E2 - 1) / (L_op + 1e-30) * (2 * N3 - N1)

        # Stage 4
        N4 = nonlinear_rhs(c)

        # Combine (ETDRK4 formula)
        # For stability with small L_op, use Taylor expansion
        phi1 = np.where(np.abs(L_op * dt) > 1e-6,
                        (E - 1) / (L_op * dt),
                        1.0 + L_op * dt / 2)
        phi2 = np.where(np.abs(L_op * dt) > 1e-6,
                        (E - 1 - L_op * dt) / (L_op * dt)**2,
                        1.0/2 + L_op * dt / 6)
        phi3 = np.where(np.abs(L_op * dt) > 1e-6,
                        (E - 1 - L_op * dt - (L_op * dt)**2 / 2) / (L_op * dt)**3,
                        1.0/6 + L_op * dt / 24)

        u_hat = E * u_hat + dt * (
            phi1 * (N1 + N4) / 6 +
            phi1 * (N2 + N3) / 3 +
            dt * phi2 * (N4 - N1) / 6 +
            dt * phi2 * (N2 + N3 - N1 - N4) / 3  # correction terms
        )

        # Simpler but stable: just use standard IMEX RK4
        # Actually, let's use a cleaner formulation: exact linear + RK4 nonlinear
        # Reset and use simpler scheme
        pass

        current_t += dt

        # Record output if we've reached the next output time
        if abs(current_t - t_out[output_idx]) < 1e-12 * T:
            u_phys = np.fft.ifft(u_hat).real
            u_xt[output_idx] = u_phys
            diagnostics['max_u'].append(float(np.abs(u_phys).max()))
            diagnostics['energy'].append(float(np.sum(u_phys**2) * L / nx))
            output_idx += 1

    return u_xt, x, t_out, diagnostics


def solve_burgers_advdiff_rk4(u0, L=16.0, v=0.5, D=0.3, epsilon=0.0, nt=100, T=10.0,
                               dt_internal=None, dealias=True):
    """
    Simpler RK4 pseudo-spectral solver.
    All terms (linear + nonlinear) handled together in Fourier space with RK4.
    """
    nx = len(u0)
    x = np.linspace(0, L, nx, endpoint=False)
    t_out = np.linspace(0, T, nt)

    # Fourier wavenumbers
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    ik = 1j * k
    k2 = k ** 2

    # Dealiasing mask (2/3 rule)
    if dealias:
        k_max = nx // 3
        dealias_mask = np.abs(np.fft.fftfreq(nx, d=1.0 / nx)) <= k_max
    else:
        dealias_mask = np.ones(nx, dtype=bool)

    # Auto time step
    if dt_internal is None:
        dx = L / nx
        u_max = max(np.abs(u0).max(), 1.0)
        dt_adv = 0.4 * dx / (abs(v) + epsilon * u_max + 1e-10)
        dt_diff = 0.2 * dx**2 / (D + 1e-10)
        dt_internal = min(dt_adv, dt_diff, T / (4 * nt))

    def rhs(u_hat):
        """Compute du_hat/dt = -v*ik*u_hat + D*(ik)^2*u_hat - ε/2*ik*FFT(u^2)"""
        # Linear terms
        linear = (-v * ik - D * k2) * u_hat

        # Nonlinear term
        if epsilon == 0:
            return linear

        u_phys = np.fft.ifft(u_hat * dealias_mask).real
        u_sq_hat = np.fft.fft(u_phys ** 2)
        nonlinear = -epsilon * 0.5 * ik * u_sq_hat

        return linear + nonlinear

    # Allocate output
    u_xt = np.zeros((nt, nx))
    u_xt[0] = u0.copy()

    u_hat = np.fft.fft(u0)
    current_t = 0.0
    output_idx = 1

    diagnostics = {
        'max_u': [float(np.abs(u0).max())],
        'energy': [float(np.sum(u0**2) * L / nx)],
        'dt_internal': float(dt_internal),
        'n_internal_steps': 0,
    }

    tol = dt_internal * 1e-6  # floating-point tolerance for time matching

    while output_idx < nt:
        dt = t_out[output_idx] - current_t
        if dt < tol:
            # Already at output time — record and advance
            u_phys = np.fft.ifft(u_hat).real
            u_xt[output_idx] = u_phys
            diagnostics['max_u'].append(float(np.abs(u_phys).max()))
            diagnostics['energy'].append(float(np.sum(u_phys**2) * L / nx))
            output_idx += 1
            continue

        dt = min(dt_internal, dt)

        # RK4 (using r1-r4 to avoid shadowing wavenumber k2)
        r1 = rhs(u_hat)
        r2 = rhs(u_hat + 0.5 * dt * r1)
        r3 = rhs(u_hat + 0.5 * dt * r2)
        r4 = rhs(u_hat + dt * r3)

        u_hat = u_hat + (dt / 6.0) * (r1 + 2 * r2 + 2 * r3 + r4)
        current_t += dt
        diagnostics['n_internal_steps'] += 1

        # Check for blowup
        if np.any(np.isnan(u_hat)) or np.abs(u_hat).max() > 1e10:
            print(f"  WARNING: Blowup detected at t={current_t:.4f}")
            diagnostics['blowup'] = True
            diagnostics['blowup_time'] = float(current_t)
            u_xt[output_idx:] = np.nan
            return u_xt, x, t_out, diagnostics

    diagnostics['blowup'] = False
    return u_xt, x, t_out, diagnostics


def generate_initial_conditions(n_samples, nx=256, degree=256, power=3.0, seed=42):
    """Generate fractaloid initial conditions (same as training)."""
    rng = np.random.default_rng(seed)
    u0_list = []
    for i in range(n_samples):
        fractaloid = FractaloidPhase(degree=degree, power=power, size=nx, patch_size=nx)
        u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
        u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
        u0_list.append(u0)
    return u0_list


def plot_trajectories(u_xt_dict, epsilons, x, t, output_dir, v, D, n_plot=3):
    """Plot sample trajectories for different epsilon values."""
    n_eps = len(epsilons)

    # Plot 1: Spatiotemporal evolution for each epsilon (first sample)
    fig, axes = plt.subplots(1, n_eps, figsize=(4 * n_eps, 4), squeeze=False)
    for j, eps in enumerate(epsilons):
        ax = axes[0, j]
        im = ax.imshow(u_xt_dict[f'{eps}'][0], aspect='auto', origin='lower',
                       extent=[x[0], x[-1], t[0], t[-1]], cmap='RdBu_r',
                       vmin=-1.5, vmax=1.5)
        ax.set_title(f'ε={eps}')
        ax.set_xlabel('x')
        if j == 0:
            ax.set_ylabel('t')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f'Spatiotemporal evolution (v={v}, D={D})', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatiotemporal_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Snapshots at selected times
    time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    fig, axes = plt.subplots(len(time_indices), 1, figsize=(10, 3 * len(time_indices)))
    for i, tidx in enumerate(time_indices):
        ax = axes[i]
        for eps in epsilons:
            label = f'ε={eps}'
            ax.plot(x, u_xt_dict[f'{eps}'][0][tidx], label=label, alpha=0.8)
        ax.set_title(f't = {t[tidx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Snapshots (v={v}, D={D})', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snapshots_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Diagnostics (max |u| and energy over time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for eps in epsilons:
        diag = u_xt_dict[f'{eps}_diagnostics'][0]
        ax1.plot(diag['max_u'], label=f'ε={eps}')
        ax2.plot(diag['energy'], label=f'ε={eps}')
    ax1.set_xlabel('Output time step')
    ax1.set_ylabel('max |u|')
    ax1.set_title('Maximum amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Output time step')
    ax2.set_ylabel('Energy (∫u² dx)')
    ax2.set_title('Energy evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate perturbation test data')
    parser.add_argument('--v', type=float, default=0.5, help='Fixed advection speed')
    parser.add_argument('--D', type=float, default=0.3, help='Fixed diffusion coefficient')
    parser.add_argument('--epsilons', type=float, nargs='+',
                        default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                        help='Nonlinear perturbation strengths')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of trajectories per epsilon (use 8 for preview, 128 for full)')
    parser.add_argument('--output_dir', type=str,
                        default='./test_time_compute/results/perturbation_preview')
    parser.add_argument('--L', type=float, default=16.0, help='Domain length')
    parser.add_argument('--nx', type=int, default=256, help='Spatial resolution')
    parser.add_argument('--nt', type=int, default=100, help='Number of output time steps')
    parser.add_argument('--T', type=float, default=10.0, help='Final time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot_only', action='store_true', help='Only generate plots, skip data saving')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating perturbation data: v={args.v}, D={args.D}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Samples per epsilon: {args.n_samples}")
    print(f"Domain: L={args.L}, nx={args.nx}, nt={args.nt}, T={args.T}")

    # Generate initial conditions (same for all epsilons)
    print("\nGenerating initial conditions...")
    u0_list = generate_initial_conditions(args.n_samples, nx=args.nx, seed=args.seed)

    # Solve for each epsilon
    all_data = {}
    for eps in args.epsilons:
        print(f"\nSolving for ε={eps}...")
        trajectories = []
        diagnostics_list = []

        for i, u0 in enumerate(u0_list):
            u_xt, x, t, diag = solve_burgers_advdiff_rk4(
                u0, L=args.L, v=args.v, D=args.D, epsilon=eps,
                nt=args.nt, T=args.T,
            )

            if diag['blowup']:
                print(f"  Sample {i}: BLOWUP at t={diag['blowup_time']:.4f}")
            else:
                final_max = float(np.abs(u_xt[-1]).max())
                print(f"  Sample {i}: final max|u|={final_max:.3f}, "
                      f"dt={diag['dt_internal']:.6f}, "
                      f"steps={diag['n_internal_steps']}")

            trajectories.append(u_xt)
            diagnostics_list.append(diag)

        all_data[f'{eps}'] = np.array(trajectories)
        all_data[f'{eps}_diagnostics'] = diagnostics_list

    # Plot comparison
    print("\nGenerating plots...")
    plot_trajectories(all_data, args.epsilons, x, t, args.output_dir, args.v, args.D)

    # Save data
    if not args.plot_only:
        for eps in args.epsilons:
            n_valid = sum(1 for d in all_data[f'{eps}_diagnostics'] if not d.get('blowup', False))
            save_path = os.path.join(args.output_dir, f'perturbation_eps{eps}.npz')
            np.savez(save_path,
                     trajectories=all_data[f'{eps}'],
                     x=x, t=t,
                     v=args.v, D=args.D, epsilon=eps,
                     n_valid=n_valid)
            print(f"Saved ε={eps}: {n_valid}/{args.n_samples} valid trajectories -> {save_path}")

    # Save config
    config = {
        'v': args.v, 'D': args.D,
        'epsilons': args.epsilons,
        'n_samples': args.n_samples,
        'L': args.L, 'nx': args.nx, 'nt': args.nt, 'T': args.T,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("PERTURBATION DATA SUMMARY")
    print(f"{'ε':>8} | {'Valid':>6} | {'Max |u|':>10} | {'Blowups':>8}")
    print("-" * 45)
    for eps in args.epsilons:
        diags = all_data[f'{eps}_diagnostics']
        n_blowup = sum(1 for d in diags if d.get('blowup', False))
        max_u_all = max(max(d['max_u']) for d in diags if not d.get('blowup', False)) if n_blowup < len(diags) else float('nan')
        print(f"{eps:>8.3f} | {args.n_samples - n_blowup:>6} | {max_u_all:>10.3f} | {n_blowup:>8}")


if __name__ == "__main__":
    main()
