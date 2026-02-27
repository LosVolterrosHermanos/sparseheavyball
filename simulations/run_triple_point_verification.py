"""Triple-point ODE verification (kappa=sigma, gamma=1) with p_* sweep.

Regime:
    kappa = sigma, gamma = 1
    p = p_* d^{-sigma}, B = B_* d^{sigma},
    eps = eps_* d^{-1}, eta = eta_* d^{sigma-1}

Prediction from:
work/ode_scaling_limits/regimes/incoherent_critical_sigma_eq_kappa_on_resonance.tex

On tau=t/d and y=D_crit^{-1} x (x=(W,Z,R)):
    dy/dtau = A_inf y
with
    A_inf =
      [[-2 rho_*,   2 rho_*,         xi_*        ],
       [-rho_* b,   -rho_*,          rho_*       ],
       [0,          -2 rho_* b,      0           ]]
    rho_* = eps_* / P_*
    b     = bar_eta = eta_* p_* / eps_*
    xi_*  = eps_*^2 / (P_* p_* B_*)
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
class TriplePointMetric:
    d: int
    sigma: float
    p_star_target: float
    p_star_eff: float
    B_star_target: float
    B_star_eff: float
    eps_star: float
    eta_star: float
    p_batch: float
    P_star: float
    rho_star: float
    bar_eta: float
    xi_star: float
    max_real_eig_raw: float
    max_real_eig_tau: float
    rel_err_l2: float
    rel_err_max: float
    rel_err_end: float
    tau_end: float


def _build_triple_point_model(
    *,
    d: int,
    sigma: float,
    p_star: float,
    B_star: float,
    eps_star: float,
    eta_star: float,
):
    d_f = float(d)
    kappa = float(sigma)
    gamma = 1.0

    p = float(p_star * (d_f ** (-sigma)))
    p = min(max(p, 1e-15), 1.0)

    B = int(round(B_star * (d_f ** sigma)))
    B = max(B, 1)

    eps = float(eps_star / d_f)
    eps = min(max(eps, 1e-12), 1.0 - 1e-12)

    eta = float(eta_star * (d_f ** (sigma - 1.0)))
    beta = 1.0 - eps

    params = ScalingParams(
        d=int(d),
        kappa=kappa,
        sigma=float(sigma),
        gamma=gamma,
        eta=eta,
        p=p,
        B=B,
        eps=eps,
        beta=beta,
    )
    return build_master_ode_model(params)


def _raw_to_wzr(raw: np.ndarray, model) -> np.ndarray:
    lam_W = max(float(model.params.eps * model.params.eps * model.B2), 1e-30)
    lam_Z = max(float(model.p_batch + model.params.eps), 1e-30)
    R = raw[:, 0]
    V = raw[:, 1]
    C = raw[:, 2]
    W = V / lam_W
    Z = C / lam_Z
    return np.column_stack([W, Z, R])


def _d_crit(model) -> np.ndarray:
    d_f = float(model.params.d)
    p = max(float(model.params.p), 1e-30)
    B = max(float(model.params.B), 1.0)
    p_batch = max(float(model.p_batch), 1e-30)
    rho = float(model.params.eps / (model.p_batch + model.params.eps))

    return np.array(
        [
            1.0,
            rho * rho * (d_f / B),
            rho * rho * d_f * (p_batch / max(p * B, 1e-30)),
        ],
        dtype=float,
    )


def _initial_raw_from_balanced(y0: np.ndarray, model) -> np.ndarray:
    D = _d_crit(model)
    x0 = D * y0  # x=(W,Z,R)
    lam_W = max(float(model.params.eps * model.params.eps * model.B2), 1e-30)
    lam_Z = max(float(model.p_batch + model.params.eps), 1e-30)
    raw_R = x0[2]
    raw_V = x0[0] * lam_W
    raw_C = x0[1] * lam_Z
    return np.array([raw_R, raw_V, raw_C], dtype=float)


def _limit_matrix(rho_star: float, bar_eta: float, xi_star: float) -> np.ndarray:
    return np.array(
        [
            [-2.0 * rho_star, 2.0 * rho_star, xi_star],
            [-rho_star * bar_eta, -rho_star, rho_star],
            [0.0, -2.0 * rho_star * bar_eta, 0.0],
        ],
        dtype=float,
    )


def _stars_and_limit(model, sigma: float):
    d_f = float(model.params.d)
    p_star_eff = float(model.params.p * (d_f ** sigma))
    B_star_eff = float(model.params.B / (d_f ** sigma))
    eps_star_eff = float(model.params.eps * d_f)
    eta_star_eff = float(model.params.eta * (d_f ** (1.0 - sigma)))

    s_star = float(p_star_eff * B_star_eff)
    P_star = float(-math.expm1(-s_star))
    rho_star = float(eps_star_eff / max(P_star, 1e-30))
    bar_eta = float((eta_star_eff * p_star_eff) / max(eps_star_eff, 1e-30))
    xi_star = float((eps_star_eff * eps_star_eff) / max(P_star * p_star_eff * B_star_eff, 1e-30))

    A_inf = _limit_matrix(rho_star=rho_star, bar_eta=bar_eta, xi_star=xi_star)
    return p_star_eff, B_star_eff, eps_star_eff, eta_star_eff, P_star, rho_star, bar_eta, xi_star, A_inf


def run_single_point(
    *,
    d: int,
    sigma: float,
    p_star: float,
    B_star: float,
    eps_star: float,
    eta_star: float,
    tau_max: float,
    steps_per_tau: int,
    init_mode: str,
    init_vec: np.ndarray,
):
    model = _build_triple_point_model(
        d=d,
        sigma=sigma,
        p_star=p_star,
        B_star=B_star,
        eps_star=eps_star,
        eta_star=eta_star,
    )
    p_star_eff, B_star_eff, eps_star_eff, eta_star_eff, P_star, rho_star, bar_eta, xi_star, A_inf = (
        _stars_and_limit(model=model, sigma=sigma)
    )

    if init_mode == "raw":
        raw_x0 = np.asarray(init_vec, dtype=float)
        if raw_x0.shape != (3,):
            raise ValueError("raw init vector must have shape (3,) for [R,V,C]")
    elif init_mode == "balanced":
        raw_x0 = _initial_raw_from_balanced(y0=np.asarray(init_vec, dtype=float), model=model)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")
    n_steps = int(max(4000, tau_max * max(steps_per_tau, 120)))
    A_tau = float(d) * model.A
    tau, raw = simulate_linear_rk4(A=A_tau, x0=raw_x0, t_end=tau_max, n_steps=n_steps)

    x_wzr = _raw_to_wzr(raw=raw, model=model)
    D = _d_crit(model)
    y_master = x_wzr / D[None, :]

    _, y_limit = simulate_linear_rk4(A=A_inf, x0=y_master[0], t_end=tau_max, n_steps=n_steps)

    err = np.linalg.norm(y_master - y_limit, axis=1)
    denom = np.maximum(np.linalg.norm(y_limit, axis=1), 1e-14)
    rel_err = err / denom

    eigvals = np.linalg.eigvals(model.A)
    metric = TriplePointMetric(
        d=int(d),
        sigma=float(sigma),
        p_star_target=float(p_star),
        p_star_eff=float(p_star_eff),
        B_star_target=float(B_star),
        B_star_eff=float(B_star_eff),
        eps_star=float(eps_star_eff),
        eta_star=float(eta_star_eff),
        p_batch=float(model.p_batch),
        P_star=float(P_star),
        rho_star=float(rho_star),
        bar_eta=float(bar_eta),
        xi_star=float(xi_star),
        max_real_eig_raw=float(np.max(np.real(eigvals))),
        max_real_eig_tau=float(d * np.max(np.real(eigvals))),
        rel_err_l2=float(np.linalg.norm(y_master - y_limit) / np.linalg.norm(y_limit)),
        rel_err_max=float(np.max(rel_err)),
        rel_err_end=float(rel_err[-1]),
        tau_end=float(tau[-1]),
    )
    raw_r_master = raw[:, 0]
    raw_r_limit = D[2] * y_limit[:, 2]
    return metric, tau, y_master, y_limit, rel_err, raw_r_master, raw_r_limit


def write_metrics_csv(path: Path, metrics: list[TriplePointMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "d",
                "sigma",
                "p_star_target",
                "p_star_eff",
                "B_star_target",
                "B_star_eff",
                "eps_star",
                "eta_star",
                "P_batch",
                "P_star",
                "rho_star",
                "bar_eta",
                "xi_star",
                "max_real_eig_raw",
                "max_real_eig_tau",
                "rel_err_l2",
                "rel_err_max",
                "rel_err_end",
                "tau_end",
            ]
        )
        for m in metrics:
            w.writerow(
                [
                    m.d,
                    f"{m.sigma:.6f}",
                    f"{m.p_star_target:.12e}",
                    f"{m.p_star_eff:.12e}",
                    f"{m.B_star_target:.12e}",
                    f"{m.B_star_eff:.12e}",
                    f"{m.eps_star:.12e}",
                    f"{m.eta_star:.12e}",
                    f"{m.p_batch:.12e}",
                    f"{m.P_star:.12e}",
                    f"{m.rho_star:.12e}",
                    f"{m.bar_eta:.12e}",
                    f"{m.xi_star:.12e}",
                    f"{m.max_real_eig_raw:.12e}",
                    f"{m.max_real_eig_tau:.12e}",
                    f"{m.rel_err_l2:.8f}",
                    f"{m.rel_err_max:.8f}",
                    f"{m.rel_err_end:.8f}",
                    f"{m.tau_end:.8f}",
                ]
            )


def plot_results(
    *,
    out_png: Path,
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    metrics: list[TriplePointMetric],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 4.9))
    ax_y1, ax_y2, ax_y3, ax_err = axes

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(traces)))
    for color, (tau, y_master, y_limit, rel_err, raw_r_master, raw_r_limit), m in zip(colors, traces, metrics):
        label = rf"$p_*={m.p_star_eff:.2f},\ P_*={m.P_star:.3f}$"
        ax_y1.plot(tau, y_master[:, 0], color=color, lw=2.0, label=label)
        ax_y1.plot(tau, y_limit[:, 0], color=color, lw=1.8, ls="--", alpha=0.95)
        ax_y2.plot(tau, y_master[:, 1], color=color, lw=2.0)
        ax_y2.plot(tau, y_limit[:, 1], color=color, lw=1.8, ls="--", alpha=0.95)
        ax_y3.plot(tau, raw_r_master, color=color, lw=2.0)
        ax_y3.plot(tau, raw_r_limit, color=color, lw=1.8, ls="--", alpha=0.95)
        ax_err.plot(tau, rel_err, color=color, lw=2.0, label=label)

    ax_y1.set_title(r"$\widehat{V}$ (from $W$ channel)")
    ax_y1.set_xlabel(r"$\tau=t/d$")
    ax_y1.set_ylabel(r"$\widehat{V}$")
    ax_y1.set_yscale("symlog", linthresh=1e-7)
    ax_y1.grid(True, alpha=0.25)

    # Keep C-hat linear for readability in resonant settings.
    ax_y2.set_title(r"$\widehat{C}$ (from $Z$ channel)")
    ax_y2.set_xlabel(r"$\tau=t/d$")
    ax_y2.set_ylabel(r"$\widehat{C}$")
    ax_y2.grid(True, alpha=0.25)

    ax_y3.set_title(r"Raw risk $R$")
    ax_y3.set_xlabel(r"$\tau=t/d$")
    ax_y3.set_ylabel(r"$R$")
    ax_y3.grid(True, alpha=0.25)

    ax_y1.legend(loc="best", fontsize=8)

    ax_err.set_title("Relative error")
    ax_err.set_xlabel(r"$\tau=t/d$")
    ax_err.set_ylabel(r"$\|y_{\mathrm{master}}-y_{\mathrm{limit}}\|/\|y_{\mathrm{limit}}\|$")
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="best", fontsize=8)

    m0 = metrics[0]
    fig.suptitle(
        (
            r"Triple-point verification ($\kappa=\sigma$, $\gamma=1$): "
            r"balanced master vs limit ODE"
            "\n"
            rf"$d={m0.d},\ \sigma={m0.sigma:.2f},\ \eta_*={m0.eta_star:.3f},\ "
            rf"\epsilon_*={m0.eps_star:.3f},\ B_* \approx {m0.B_star_eff:.2f}$"
            "\n"
            r"Panels 1-2: balanced channels. Panel 3: raw risk $R$."
            r" Solid: master. Dashed: limit model."
        ),
        fontsize=12.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=1.2)
    p.add_argument("--eta-star", type=float, default=0.2)
    p.add_argument("--eps-star", type=float, default=1.0)
    p.add_argument("--B-star", type=float, default=1.0)
    p.add_argument(
        "--p-star-list",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 1.0, 1.2, 1.4],
        help="Sweep values for p_* in p=p_* d^{-sigma}.",
    )
    p.add_argument("--tau-max", type=float, default=12.0, help="Rescaled horizon in tau=t/d.")
    p.add_argument("--steps-per-tau", type=int, default=1200)
    p.add_argument(
        "--init-mode",
        type=str,
        choices=["raw", "balanced"],
        default="raw",
        help="Initialization mode: raw uses [R,V,C], balanced uses y=(y1,y2,y3).",
    )
    p.add_argument(
        "--raw-init",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("R0", "V0", "C0"),
        help="Initial condition in raw coordinates [R,V,C] (used when --init-mode raw).",
    )
    p.add_argument(
        "--y0",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("Y1_0", "Y2_0", "Y3_0"),
        help="Initial condition in balanced coordinates y=(y1,y2,y3) (used when --init-mode balanced).",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("simulations/outputs/triple_point_verification.png"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/triple_point_verification_metrics.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.d <= 1:
        raise ValueError("--d must be > 1")
    if args.sigma <= 0.0:
        raise ValueError("--sigma must be > 0")
    if args.eta_star <= 0.0:
        raise ValueError("--eta-star must be > 0")
    if args.eps_star <= 0.0:
        raise ValueError("--eps-star must be > 0")
    if args.B_star <= 0.0:
        raise ValueError("--B-star must be > 0")
    if args.tau_max <= 0.0:
        raise ValueError("--tau-max must be > 0")
    if args.steps_per_tau < 20:
        raise ValueError("--steps-per-tau must be >= 20")
    if not args.p_star_list:
        raise ValueError("--p-star-list cannot be empty")
    if any(v <= 0.0 for v in args.p_star_list):
        raise ValueError("all p_* values in --p-star-list must be > 0")

    if args.init_mode == "raw":
        init_vec = np.asarray(args.raw_init, dtype=float)
        if init_vec.shape != (3,):
            raise ValueError("--raw-init must have exactly 3 entries")
    else:
        init_vec = np.asarray(args.y0, dtype=float)
        if init_vec.shape != (3,):
            raise ValueError("--y0 must have exactly 3 entries")

    metrics: list[TriplePointMetric] = []
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for p_star in args.p_star_list:
        metric, tau, y_master, y_limit, rel_err, raw_r_master, raw_r_limit = run_single_point(
            d=args.d,
            sigma=args.sigma,
            p_star=float(p_star),
            B_star=args.B_star,
            eps_star=args.eps_star,
            eta_star=args.eta_star,
            tau_max=args.tau_max,
            steps_per_tau=args.steps_per_tau,
            init_mode=args.init_mode,
            init_vec=init_vec,
        )
        metrics.append(metric)
        traces.append((tau, y_master, y_limit, rel_err, raw_r_master, raw_r_limit))

    write_metrics_csv(args.out_csv, metrics)
    plot_results(out_png=args.out_png, traces=traces, metrics=metrics)


if __name__ == "__main__":
    main()
