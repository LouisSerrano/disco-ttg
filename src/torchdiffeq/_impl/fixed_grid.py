from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func, rk3_step_func
from .misc import Perturb


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0


class Heun3(FixedGridODESolver):
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)

        butcher_tableu = [
            [0.0, 0.0, 0.0, 0.0],
            [1/3, 1/3, 0.0, 0.0],
            [2/3, 0.0, 2/3, 0.0],
            [0.0, 1/4, 0.0, 3/4],
        ]

        return rk3_step_func(func, t0, dt, t1, y0, butcher_tableu=butcher_tableu, f0=f0, perturb=self.perturb), f0


class SSP_RK3(FixedGridODESolver):
    """Strong Stability Preserving Runge-Kutta 3rd order method."""
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        # Stage 1: u₁ = uⁿ + Δt L(uⁿ)
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        u1 = y0 + dt * f0

        # Stage 2: u₂ = (3/4)uⁿ + (1/4)u₁ + (1/4)Δt L(u₁)
        f1 = func(t0 + dt, u1, perturb=Perturb.NONE)
        u2 = 0.75 * y0 + 0.25 * u1 + 0.25 * dt * f1

        # Stage 3: uⁿ⁺¹ = (1/3)uⁿ + (2/3)u₂ + (2/3)Δt L(u₂)
        f2 = func(t0 + 0.5 * dt, u2, perturb=Perturb.PREV if self.perturb else Perturb.NONE)
        y_new = (1.0/3.0) * y0 + (2.0/3.0) * u2 + (2.0/3.0) * dt * f2

        return y_new - y0, f0
