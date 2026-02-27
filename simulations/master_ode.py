"""Master sparse-momentum ODE simulator utilities.

Formulas come from:
- work/ode_scaling_limits/ode_scaling_limits.tex (raw 3D ODE coefficients)
- 68cda39aec02a765ac1c823d/appendix.tex (closed-form drift/retention terms)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ScalingParams:
    d: int
    kappa: float
    sigma: float
    gamma: float
    eta: float
    p: float
    B: int
    eps: float
    beta: float


@dataclass(frozen=True)
class MasterODEModel:
    params: ScalingParams
    p_batch: float
    q_batch: float
    B1: float
    Bdiag: float
    Bcross: float
    B2: float
    bar_beta1: float
    bar_beta2: float
    delta_theta: float
    delta_g: float
    delta_theta_2: float
    delta_g_2: float
    delta_m1_theta: float
    delta_m1_g: float
    delta_theta_g: float
    A: np.ndarray


def scaling_to_params(
    d: int,
    kappa: float,
    sigma: float,
    gamma: float,
    eta: float,
) -> ScalingParams:
    """Build finite-d parameters from co-scaling exponents."""
    p = float(d ** (-kappa))
    p = min(max(p, 1e-15), 1.0)

    B = int(round(d ** sigma))
    B = max(B, 1)

    eps = float(d ** (-gamma))
    eps = min(max(eps, 1e-12), 1.0 - 1e-12)
    beta = 1.0 - eps

    return ScalingParams(
        d=int(d),
        kappa=float(kappa),
        sigma=float(sigma),
        gamma=float(gamma),
        eta=float(eta),
        p=p,
        B=B,
        eps=eps,
        beta=beta,
    )


def _active_batch_probability(p: float, B: int) -> float:
    """Compute P_batch = 1 - (1-p)^B robustly."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    log_q_pow = B * math.log1p(-p)
    q_pow = 0.0 if log_q_pow < -745.0 else math.exp(log_q_pow)
    return 1.0 - q_pow


def build_master_ode_model(params: ScalingParams) -> MasterODEModel:
    """Assemble the raw 3D master ODE d/dt (R,V,C)^T = A (R,V,C)^T."""
    d = float(params.d)
    eta = params.eta
    beta = params.beta
    eps = params.eps
    p = params.p
    B = float(params.B)

    p_batch = max(_active_batch_probability(p, int(B)), 1e-15)
    q_batch = 1.0 - p_batch

    # Batch scaling factors.
    B1 = p / p_batch
    Bdiag = p / (B * p_batch)
    Bcross = p * p * (B - 1.0) / (B * p_batch)
    B2 = (d + 2.0) * Bdiag + Bcross

    # Retention/drift closed forms from appendix.tex.
    den1 = 1.0 - q_batch * beta
    den2 = 1.0 - q_batch * beta * beta
    den1 = max(den1, 1e-15)
    den2 = max(den2, 1e-15)

    bar_beta1 = p_batch * beta / den1
    bar_beta2 = p_batch * beta * beta / den2

    delta_theta = beta / den1
    delta_g = q_batch * beta / den1

    delta_theta_2 = beta * beta * (1.0 + q_batch * beta) / (den1 * den2)
    delta_g_2 = q_batch * beta * beta * (1.0 + q_batch * beta) / (den1 * den2)

    delta_m1_theta = p_batch * beta * beta / (den1 * den2)
    delta_m1_g = p_batch * q_batch * beta * beta * beta / (den1 * den2)
    delta_theta_g = q_batch * beta * beta * (1.0 + beta) / (den1 * den2)

    # Raw (R,V,C) ODE coefficients from work/ode_scaling_limits.tex.
    a_R = -2.0 * eta * eps * B1 + eta * eta * eps * eps * B2
    a_V = (
        eta * eta * delta_theta_2
        - 2.0 * eta**3 * eps * B1 * delta_theta_g
        + eta**4 * eps * eps * B2 * delta_g_2
    )
    a_C = (
        -2.0 * eta * delta_theta
        + 2.0 * eta * eta * eps * B1 * (delta_g + delta_theta)
        - 2.0 * eta**3 * eps * eps * B2 * delta_g
    )

    b_R = eps * eps * B2
    b_V = (
        (bar_beta2 - 1.0)
        - 2.0 * eta * eps * B1 * delta_m1_g
        + eta * eta * eps * eps * B2 * delta_g_2
    )
    b_C = 2.0 * eps * bar_beta1 * B1 - 2.0 * eta * eps * eps * B2 * delta_g

    c_R = eps * B1 - eta * eps * eps * B2
    c_V = (
        -eta * delta_m1_theta
        + eta * eta * eps * B1 * (delta_theta_g + delta_m1_g)
        - eta**3 * eps * eps * B2 * delta_g_2
    )
    c_C = (
        (bar_beta1 - 1.0)
        - eta * eps * B1 * (delta_g + delta_theta + bar_beta1)
        + 2.0 * eta * eta * eps * eps * B2 * delta_g
    )

    A = np.array(
        [
            [a_R, a_V, a_C],
            [b_R, b_V, b_C],
            [c_R, c_V, c_C],
        ],
        dtype=float,
    )

    return MasterODEModel(
        params=params,
        p_batch=p_batch,
        q_batch=q_batch,
        B1=B1,
        Bdiag=Bdiag,
        Bcross=Bcross,
        B2=B2,
        bar_beta1=bar_beta1,
        bar_beta2=bar_beta2,
        delta_theta=delta_theta,
        delta_g=delta_g,
        delta_theta_2=delta_theta_2,
        delta_g_2=delta_g_2,
        delta_m1_theta=delta_m1_theta,
        delta_m1_g=delta_m1_g,
        delta_theta_g=delta_theta_g,
        A=A,
    )


def build_model_from_scaling(
    d: int,
    kappa: float,
    sigma: float,
    gamma: float,
    eta: float,
) -> MasterODEModel:
    params = scaling_to_params(d=d, kappa=kappa, sigma=sigma, gamma=gamma, eta=eta)
    return build_master_ode_model(params)


def simulate_linear_rk4(
    A: np.ndarray,
    x0: np.ndarray,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate x' = A x with fixed-step RK4."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")

    x0 = np.asarray(x0, dtype=float)
    t = np.linspace(0.0, float(t_end), int(n_steps) + 1, dtype=float)
    dt = t[1] - t[0]

    x = np.empty((t.size, x0.size), dtype=float)
    x[0] = x0

    for i in range(t.size - 1):
        xi = x[i]
        k1 = A @ xi
        k2 = A @ (xi + 0.5 * dt * k1)
        k3 = A @ (xi + 0.5 * dt * k2)
        k4 = A @ (xi + dt * k3)
        x[i + 1] = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, x

