"""Critical incoherent-boundary ODE verification with a p_* sweep.

Regime:
    kappa = sigma, 0 < gamma < 1 (below resonance)
    p = p_* d^{-sigma}, B = B_* d^{sigma}, eta = eta_* d^{sigma-1}

Prediction from
work/ode_scaling_limits/regimes/incoherent_critical_sigma_eq_kappa_below_resonance.tex:
    tau = t / d,
    dR/dtau = -c_eff^crit R,
    c_eff^crit = (eta_* p_* / P_*) (2 - eta_* / B_*),
    P_* = 1 - exp(-p_* B_*).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from master_ode import ScalingParams, build_master_ode_model, simulate_linear_rk4


@dataclass(frozen=True)
class CriticalBoundaryMetric:
    d: int
    sigma: float
    gamma: float
    p_star_target: float
    p_star_eff: float
    B_star_target: float
    B_star_eff: float
    eta_star: float
    eta: float
    p_batch: float
    P_star: float
    chi_star: float
    eta_eff: float
    c_eff_pred: float
    c_eff_pred_fd: float
    c_eff_emp: float
    rel_l2_pred: float
    rel_l2_pred_fd: float
    max_real_eig_raw: float
    tau_end: float


def _build_critical_boundary_model(
    *,
    d: int,
    sigma: float,
    gamma: float,
    p_star: float,
    B_star: float,
    eta_star: float,
):
    """Build the finite-d master model at kappa=sigma with explicit p_*, B_*, eta_*."""
    d_f = float(d)
    kappa = float(sigma)

    p = float(p_star * (d_f ** (-sigma)))
    p = min(max(p, 1e-15), 1.0)

    B = int(round(B_star * (d_f ** sigma)))
    B = max(B, 1)

    eps = float(d_f ** (-gamma))
    eps = min(max(eps, 1e-12), 1.0 - 1e-12)

    eta = float(eta_star * (d_f ** (sigma - 1.0)))
    beta = 1.0 - eps

    params = ScalingParams(
        d=int(d),
        kappa=kappa,
        sigma=float(sigma),
        gamma=float(gamma),
        eta=eta,
        p=p,
        B=B,
        eps=eps,
        beta=beta,
    )
    return build_master_ode_model(params)


def _critical_prediction_constants(
    *,
    d: int,
    sigma: float,
    model,
):
    d_f = float(d)
    p_star_eff = float(model.params.p * (d_f ** sigma))
    B_star_eff = float(model.params.B / (d_f ** sigma))
    eta_star_eff = float(model.params.eta * (d_f ** (1.0 - sigma)))

    s_star = float(p_star_eff * B_star_eff)
    P_star = float(-math.expm1(-s_star))
    chi_star = float(s_star / max(P_star, 1e-30))
    eta_eff = float(eta_star_eff / max(B_star_eff, 1e-30))

    c_eff_pred = float((eta_star_eff * p_star_eff / max(P_star, 1e-30)) * (2.0 - eta_eff))
    c_eff_pred_fd = float(
        ((model.params.eta * model.params.p * d_f) / max(model.p_batch, 1e-30))
        * (2.0 - (model.params.eta * d_f) / max(model.params.B, 1))
    )
    return p_star_eff, B_star_eff, eta_star_eff, P_star, chi_star, eta_eff, c_eff_pred, c_eff_pred_fd


def _empirical_decay_rate(tau: np.ndarray, R: np.ndarray, fit_start_frac: float) -> float:
    fit_start = float(np.clip(fit_start_frac, 0.0, 0.95) * tau[-1])
    mask = (tau >= fit_start) & (R > 1e-14)
    if int(np.sum(mask)) < 3:
        mask = (tau > 0.0) & (R > 1e-14)
    if int(np.sum(mask)) < 3:
        return float("nan")
    slope = np.polyfit(tau[mask], np.log(R[mask]), 1)[0]
    return float(-slope)


def run_single_sweep_point(
    *,
    d: int,
    sigma: float,
    gamma: float,
    p_star: float,
    B_star: float,
    eta_star: float,
    tau_max: float,
    steps_per_tau: int,
    fit_start_frac: float,
):
    model = _build_critical_boundary_model(
        d=d,
        sigma=sigma,
        gamma=gamma,
        p_star=p_star,
        B_star=B_star,
        eta_star=eta_star,
    )
    p_star_eff, B_star_eff, eta_star_eff, P_star, chi_star, eta_eff, c_eff_pred, c_eff_pred_fd = (
        _critical_prediction_constants(d=d, sigma=sigma, model=model)
    )

    A_tau = float(d) * model.A
    x0 = np.array([1.0, 0.0, 0.0], dtype=float)  # [R,V,C]
    n_steps = int(max(4000, tau_max * max(steps_per_tau, 100)))
    tau, x_raw = simulate_linear_rk4(A=A_tau, x0=x0, t_end=tau_max, n_steps=n_steps)
    R = x_raw[:, 0]

    R_pred = R[0] * np.exp(-c_eff_pred * tau)
    R_pred_fd = R[0] * np.exp(-c_eff_pred_fd * tau)
    c_eff_emp = _empirical_decay_rate(tau=tau, R=R, fit_start_frac=fit_start_frac)

    denom = max(float(np.linalg.norm(R)), 1e-30)
    rel_l2_pred = float(np.linalg.norm(R - R_pred) / denom)
    rel_l2_pred_fd = float(np.linalg.norm(R - R_pred_fd) / denom)

    eigvals = np.linalg.eigvals(model.A)
    metric = CriticalBoundaryMetric(
        d=int(d),
        sigma=float(sigma),
        gamma=float(gamma),
        p_star_target=float(p_star),
        p_star_eff=float(p_star_eff),
        B_star_target=float(B_star),
        B_star_eff=float(B_star_eff),
        eta_star=float(eta_star_eff),
        eta=float(model.params.eta),
        p_batch=float(model.p_batch),
        P_star=float(P_star),
        chi_star=float(chi_star),
        eta_eff=float(eta_eff),
        c_eff_pred=float(c_eff_pred),
        c_eff_pred_fd=float(c_eff_pred_fd),
        c_eff_emp=float(c_eff_emp),
        rel_l2_pred=float(rel_l2_pred),
        rel_l2_pred_fd=float(rel_l2_pred_fd),
        max_real_eig_raw=float(np.max(np.real(eigvals))),
        tau_end=float(tau[-1]),
    )
    return metric, tau, R, R_pred


def write_metrics_csv(path: Path, metrics: list[CriticalBoundaryMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "d",
                "sigma",
                "gamma",
                "p_star_target",
                "p_star_eff",
                "B_star_target",
                "B_star_eff",
                "eta_star",
                "eta",
                "P_batch",
                "P_star",
                "chi_star",
                "eta_eff",
                "c_eff_pred",
                "c_eff_pred_fd",
                "c_eff_emp",
                "emp_over_pred",
                "rel_l2_pred",
                "rel_l2_pred_fd",
                "max_real_eig_raw",
                "tau_end",
            ]
        )
        for m in metrics:
            ratio = m.c_eff_emp / m.c_eff_pred if m.c_eff_pred != 0.0 else np.nan
            writer.writerow(
                [
                    m.d,
                    f"{m.sigma:.6f}",
                    f"{m.gamma:.6f}",
                    f"{m.p_star_target:.12e}",
                    f"{m.p_star_eff:.12e}",
                    f"{m.B_star_target:.12e}",
                    f"{m.B_star_eff:.12e}",
                    f"{m.eta_star:.12e}",
                    f"{m.eta:.12e}",
                    f"{m.p_batch:.12e}",
                    f"{m.P_star:.12e}",
                    f"{m.chi_star:.12e}",
                    f"{m.eta_eff:.12e}",
                    f"{m.c_eff_pred:.12e}",
                    f"{m.c_eff_pred_fd:.12e}",
                    f"{m.c_eff_emp:.12e}",
                    f"{ratio:.8f}",
                    f"{m.rel_l2_pred:.8f}",
                    f"{m.rel_l2_pred_fd:.8f}",
                    f"{m.max_real_eig_raw:.12e}",
                    f"{m.tau_end:.8f}",
                ]
            )


def plot_results(
    *,
    out_png: Path,
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    metrics: list[CriticalBoundaryMetric],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax_traj, ax_slope = axes

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(traces)))
    for color, (tau, R, R_pred), metric in zip(colors, traces, metrics):
        label_core = rf"$p_*={metric.p_star_eff:.2f},\ P_*={metric.P_star:.3f}$"
        ax_traj.plot(tau, R, color=color, lw=2.2, label=rf"master {label_core}")
        ax_traj.plot(
            tau,
            R_pred,
            color=color,
            lw=1.8,
            ls="--",
            alpha=0.95,
            label=rf"pred {label_core}",
        )

    ax_traj.set_yscale("log")
    ax_traj.set_xlabel(r"$\tau=t/d$")
    ax_traj.set_ylabel(r"$R(\tau)$")
    ax_traj.set_title("Critical boundary trajectories")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="best", fontsize=8)

    P_vals = np.array([m.P_star for m in metrics], dtype=float)
    pred = np.array([m.c_eff_pred for m in metrics], dtype=float)
    pred_fd = np.array([m.c_eff_pred_fd for m in metrics], dtype=float)
    emp = np.array([m.c_eff_emp for m in metrics], dtype=float)

    ax_slope.plot(P_vals, pred, marker="o", lw=2.0, color="tab:red", label=r"predicted $c_{\mathrm{eff}}^{\mathrm{crit}}$")
    ax_slope.plot(P_vals, pred_fd, marker="^", lw=1.8, color="tab:purple", label="finite-d proxy")
    ax_slope.plot(P_vals, emp, marker="x", lw=2.0, color="tab:blue", label="empirical slope")
    ax_slope.set_xlabel(r"$P_*$")
    ax_slope.set_ylabel(r"decay constant / slope")
    ax_slope.set_title(r"Slope dependence on $P_*$")
    ax_slope.grid(True, alpha=0.25)
    ax_slope.legend(loc="best", fontsize=9)

    m0 = metrics[0]
    fig.suptitle(
        (
            r"Critical incoherent boundary check  ($\kappa=\sigma$, below resonance)"
            "\n"
            rf"$d={m0.d},\ \sigma={m0.sigma:.2f},\ \gamma={m0.gamma:.2f},\ "
            rf"\eta_*={m0.eta_star:.3f},\ B_* \approx {m0.B_star_eff:.2f}$"
            "\n"
            r"Solid: master ODE risk trajectory. Dashed: boxed prediction "
            r"$R(\tau)=R(0)e^{-c_{\mathrm{eff}}^{\mathrm{crit}}\tau}$."
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=1.2, help="Critical boundary uses kappa=sigma.")
    p.add_argument("--gamma", type=float, default=0.6, help="Must satisfy 0 < gamma < 1.")
    p.add_argument("--eta-star", type=float, default=0.2, help="Sets eta = eta_star * d^(sigma-1).")
    p.add_argument("--B-star", type=float, default=1.0, help="Sets B = round(B_star * d^sigma).")
    p.add_argument(
        "--p-star-list",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 1.0, 1.2, 1.4],
        help="Sweep values for p_* in p = p_* d^{-sigma}.",
    )
    p.add_argument("--tau-max", type=float, default=12.0, help="Horizon in tau=t/d.")
    p.add_argument("--steps-per-tau", type=int, default=1200)
    p.add_argument(
        "--fit-start-frac",
        type=float,
        default=0.35,
        help="Estimate empirical slope from tau >= fit_start_frac * tau_max.",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("simulations/outputs/critical_incoherent_boundary_verification.png"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/critical_incoherent_boundary_verification_metrics.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.d <= 1:
        raise ValueError("--d must be > 1")
    if args.sigma <= 0.0:
        raise ValueError("critical boundary setup expects sigma > 0")
    if not (0.0 < args.gamma < 1.0):
        raise ValueError("critical boundary below resonance requires 0 < gamma < 1")
    if args.B_star <= 0.0:
        raise ValueError("--B-star must be > 0")
    if args.eta_star <= 0.0:
        raise ValueError("--eta-star must be > 0")
    if args.tau_max <= 0.0:
        raise ValueError("--tau-max must be > 0")
    if args.steps_per_tau < 20:
        raise ValueError("--steps-per-tau must be >= 20")
    if not args.p_star_list:
        raise ValueError("--p-star-list cannot be empty")
    if any(v <= 0.0 for v in args.p_star_list):
        raise ValueError("all p_* values in --p-star-list must be > 0")

    metrics: list[CriticalBoundaryMetric] = []
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for p_star in args.p_star_list:
        metric, tau, R, R_pred = run_single_sweep_point(
            d=args.d,
            sigma=args.sigma,
            gamma=args.gamma,
            p_star=float(p_star),
            B_star=args.B_star,
            eta_star=args.eta_star,
            tau_max=args.tau_max,
            steps_per_tau=args.steps_per_tau,
            fit_start_frac=args.fit_start_frac,
        )
        metrics.append(metric)
        traces.append((tau, R, R_pred))

    write_metrics_csv(args.out_csv, metrics)
    plot_results(out_png=args.out_png, traces=traces, metrics=metrics)


if __name__ == "__main__":
    main()
