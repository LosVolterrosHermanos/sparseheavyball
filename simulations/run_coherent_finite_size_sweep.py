"""Coherent-regime finite-size sweep: compare rescaled master ODE vs limit ODE across d."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from master_ode import build_model_from_scaling, simulate_linear_rk4
from run_region_panel import make_region_cases, stable_model_for_case
from run_rescaled_limit_panel import (
    build_rescaling_spec,
    initial_raw_from_spec,
    limit_matrix_3d,
    raw_to_wzr,
)


def coherent_case(sigma: float):
    for case in make_region_cases(sigma):
        if case.name.lower().startswith("coherent"):
            return case
    raise RuntimeError("Could not locate coherent case.")


def run_one(
    d: int,
    sigma: float,
    eta_prefactor: float,
    max_eta: float,
    tau_max: float,
    steps_per_tau: int,
):
    case = coherent_case(sigma)
    eta, A_raw, spectral_abscissa = stable_model_for_case(
        d=d,
        sigma=sigma,
        case=case,
        eta_prefactor=eta_prefactor,
        max_eta=max_eta,
    )
    model = build_model_from_scaling(
        d=d,
        kappa=case.kappa,
        sigma=sigma,
        gamma=case.gamma,
        eta=eta,
    )
    spec = build_rescaling_spec(case=case, model=model, d=d, sigma=sigma, eta_used=eta)
    if spec.limit_kind != "3d" or spec.diag_D is None:
        raise RuntimeError("Coherent case should map to 3d limit with a balancing transform.")

    raw_x0 = initial_raw_from_spec(spec, model)
    # Keep the rescaled coherent clock range fixed across d:
    # tau = t / d^gamma in [0, tau_max].
    t_end = float(tau_max / max(spec.time_scale, 1e-30))
    n_steps = int(max(4000, steps_per_tau * tau_max))
    t, x_raw = simulate_linear_rk4(A=A_raw, x0=raw_x0, t_end=t_end, n_steps=n_steps)
    tau = spec.time_scale * t

    x_wzr = raw_to_wzr(x_raw, model)
    y_master = x_wzr / spec.diag_D[None, :]

    A_inf = spec.limit_matrix_scale * limit_matrix_3d(theta=spec.theta)
    _, y_limit = simulate_linear_rk4(
        A=A_inf,
        x0=y_master[0],
        t_end=float(tau[-1]),
        n_steps=n_steps,
    )

    err = np.linalg.norm(y_master - y_limit, axis=1)
    denom = np.maximum(np.linalg.norm(y_limit, axis=1), 1e-14)
    rel_err = err / denom

    metrics = {
        "d": int(d),
        "eta": float(eta),
        "theta": float(spec.theta),
        "eps_star": float(model.params.eps * (d ** case.gamma)),
        "time_scale": float(spec.time_scale),
        "tau_end": float(tau[-1]),
        "max_real_eig_raw": float(spectral_abscissa),
        "rel_err_l2": float(np.linalg.norm(y_master - y_limit) / np.linalg.norm(y_limit)),
        "rel_err_max": float(np.max(rel_err)),
        "rel_err_end": float(rel_err[-1]),
    }

    return tau, y_master, y_limit, rel_err, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "d",
        "eta",
        "theta",
        "eps_star",
        "time_scale",
        "tau_end",
        "max_real_eig_raw",
        "rel_err_l2",
        "rel_err_max",
        "rel_err_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_sweep(
    out_png: Path,
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(traces)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.1 * (n + 1), 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, (tau, y_master, y_limit, rel_err, metrics) in enumerate(traces):
        ax = axes[i]
        m_colors = ["tab:blue", "tab:orange", "tab:green"]
        l_colors = ["tab:red", "tab:purple", "tab:brown"]
        labels = ["y1", "y2", "y3"]
        for j in range(3):
            ax.plot(tau, y_master[:, j], color=m_colors[j], lw=2.0, label=f"master {labels[j]}")
            ax.plot(tau, y_limit[:, j], color=l_colors[j], lw=1.9, ls="--", label=f"limit {labels[j]}")

        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("tau = t / d^gamma")
        ax.set_ylabel("balanced coherent state")
        ax.set_title(f"Coherent: d={metrics['d']}", fontsize=11)
        ax.text(
            0.02,
            0.02,
            (
                f"eta*={metrics['theta']:.3f}, eps*={metrics['eps_star']:.3f}\n"
                f"L2 rel err={metrics['rel_err_l2']:.2e}\n"
                f"end rel err={metrics['rel_err_end']:.2e}"
            ),
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    ax_err = axes[-1]
    for tau, _, _, rel_err, metrics in traces:
        ax_err.plot(tau, rel_err, lw=2.0, label=f"d={metrics['d']}")
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.25)
    ax_err.set_xlabel("tau = t / d^gamma")
    ax_err.set_ylabel("||master-limit|| / ||limit||")
    ax_err.set_title("Finite-size error vs rescaled time", fontsize=11)
    ax_err.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Coherent finite-size sweep (same setup, larger d)\nSolid: rescaled master. Dashed: coherent limit ODE.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=float, default=1.2)
    p.add_argument("--eta-prefactor", type=float, default=0.4)
    p.add_argument("--max-eta", type=float, default=0.2)
    p.add_argument(
        "--d-list",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="Dimension sweep (first value should match current baseline panel).",
    )
    p.add_argument(
        "--tau-max",
        type=float,
        default=12.0,
        help="Fixed coherent rescaled-time horizon for all d (tau=t/d^gamma).",
    )
    p.add_argument(
        "--steps-per-tau",
        type=int,
        default=1200,
        help="Integration steps per 1 unit of tau.",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("simulations/outputs/coherent_finite_size_sweep.png"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/coherent_finite_size_sweep_metrics.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traces = [
        run_one(
            d=d,
            sigma=args.sigma,
            eta_prefactor=args.eta_prefactor,
            max_eta=args.max_eta,
            tau_max=args.tau_max,
            steps_per_tau=args.steps_per_tau,
        )
        for d in args.d_list
    ]
    plot_sweep(args.out_png, traces)
    write_csv(args.out_csv, [metrics for *_, metrics in traces])


if __name__ == "__main__":
    main()
