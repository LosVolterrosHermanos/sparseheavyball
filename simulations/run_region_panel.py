"""Run six-region master ODE simulations and generate a panel plot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from repository root or importing as a module.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from master_ode import build_model_from_scaling, simulate_linear_rk4


@dataclass(frozen=True)
class RegionCase:
    name: str
    expected_dim: str
    kappa: float
    gamma: float


def make_region_cases(sigma: float) -> list[RegionCase]:
    """Pick interior points for all six regions at fixed sigma > 1."""
    if sigma <= 1.0:
        raise ValueError("This panel builder expects sigma > 1 to include all six regions.")

    k_coherent = 0.25 * (sigma - 1.0)
    k_dense = sigma - 0.35
    k_sparse = sigma + 1.0

    gamma_res_dense = 1.0 - sigma + k_dense
    gamma_mem_sparse = k_sparse - sigma
    gamma_res_sparse = 1.0 - sigma + k_sparse

    return [
        # 3D regimes first (top row in 2x3 panel).
        RegionCase(
            name="Coherent",
            expected_dim="3D",
            kappa=k_coherent,
            gamma=0.8,
        ),
        RegionCase(
            name="Incoh. dense above res.",
            expected_dim="3D",
            kappa=k_dense,
            gamma=gamma_res_dense + 0.5,
        ),
        RegionCase(
            name="Incoh. sparse above res.",
            expected_dim="3D",
            kappa=k_sparse,
            gamma=gamma_res_sparse + 0.6,
        ),
        # 1D regimes second (bottom row in 2x3 panel).
        RegionCase(
            name="Incoh. dense below res.",
            expected_dim="1D",
            kappa=k_dense,
            gamma=0.5 * gamma_res_dense,
        ),
        RegionCase(
            name="Incoh. sparse memoryless",
            expected_dim="1D",
            kappa=k_sparse,
            gamma=max(0.2, 0.4 * gamma_mem_sparse),
        ),
        RegionCase(
            name="Incoh. sparse below res.",
            expected_dim="1D",
            kappa=k_sparse,
            gamma=gamma_mem_sparse + 0.5 * (gamma_res_sparse - gamma_mem_sparse),
        ),
    ]


def eta_ceiling_scale(d: int, sigma: float, case: RegionCase) -> float:
    """Polynomial-scale heuristic from the manuscript's stability summaries."""
    if case.expected_dim == "3D":
        # Correlation-limited ceiling: eta_max ~ d^(kappa-gamma)
        return float(d ** (case.kappa - case.gamma))
    # Noise-limited ceiling: eta_max ~ d^(sigma-1) in incoherent 1D regimes.
    return float(d ** (sigma - 1.0))


def stable_model_for_case(
    d: int,
    sigma: float,
    case: RegionCase,
    eta_prefactor: float,
    max_eta: float,
) -> tuple[float, np.ndarray, float]:
    """Pick eta from scaling and back off until the linear system is stable."""
    eta = eta_prefactor * eta_ceiling_scale(d=d, sigma=sigma, case=case)
    eta = min(max(eta, 1e-8), max_eta)

    last_model = None
    last_abscissa = np.inf
    for _ in range(40):
        model = build_model_from_scaling(
            d=d,
            kappa=case.kappa,
            sigma=sigma,
            gamma=case.gamma,
            eta=eta,
        )
        eigvals = np.linalg.eigvals(model.A)
        spectral_abscissa = float(np.max(np.real(eigvals)))
        last_model = model
        last_abscissa = spectral_abscissa
        if np.isfinite(spectral_abscissa) and spectral_abscissa < -1e-4:
            return eta, model.A, spectral_abscissa
        eta *= 0.5

    assert last_model is not None
    return eta, last_model.A, last_abscissa


def integration_horizon(A: np.ndarray) -> tuple[float, int]:
    eigvals = np.linalg.eigvals(A)
    rates = -np.real(eigvals)
    pos = rates[rates > 1e-9]

    if pos.size == 0:
        t_end = 200.0
    else:
        tau_slow = 1.0 / np.min(pos)
        t_end = float(np.clip(12.0 * tau_slow, 20.0, 5000.0))

    n_steps = int(np.clip(120.0 * t_end, 4000.0, 60000.0))
    return t_end, n_steps


def plot_panel(
    d: int,
    sigma: float,
    cases: list[RegionCase],
    out_png: Path,
    out_csv: Path,
    eta_prefactor: float,
    max_eta: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    x0 = np.array([1.0, 1.0, 0.25], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    summary_rows = [
        "region,expected_dim,kappa,gamma,eta,p,B,eps,beta,P_batch,max_real_eig,lambda_min,lambda_max"
    ]

    for ax, case in zip(axes, cases):
        eta, A, spectral_abscissa = stable_model_for_case(
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

        t_end, n_steps = integration_horizon(A)
        t, x = simulate_linear_rk4(A=A, x0=x0, t_end=t_end, n_steps=n_steps)

        x_norm = x / np.maximum(np.abs(x0), 1e-12)
        ax.plot(t, x_norm[:, 0], lw=2.0, label="R")
        ax.plot(t, x_norm[:, 1], lw=2.0, label="V")
        ax.plot(t, x_norm[:, 2], lw=2.0, label="C")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("t")
        ax.set_ylabel("state / init")

        eigvals = np.linalg.eigvals(A)
        lam_min = float(np.min(np.real(eigvals)))
        lam_max = float(np.max(np.real(eigvals)))

        ax.set_title(f"{case.name} ({case.expected_dim})", fontsize=11)
        ax.text(
            0.02,
            0.02,
            (
                f"k={case.kappa:.2f}, g={case.gamma:.2f}\n"
                f"eta={eta:.2e}, B={model.params.B}\n"
                f"max Re(lambda)={lam_max:.2e}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        summary_rows.append(
            (
                f"{case.name},{case.expected_dim},{case.kappa:.6f},{case.gamma:.6f},"
                f"{eta:.10e},{model.params.p:.10e},{model.params.B},{model.params.eps:.10e},"
                f"{model.params.beta:.10e},{model.p_batch:.10e},{spectral_abscissa:.10e},"
                f"{lam_min:.10e},{lam_max:.10e}"
            )
        )

    axes[0].legend(loc="best")
    fig.suptitle(
        f"Master ODE region panel (d={d}, sigma={sigma})\nRaw system in (R,V,C), normalized by initial state",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_csv.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=100, help="Ambient dimension")
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.2,
        help="Batch exponent (use sigma > 1 to include coherent strip)",
    )
    parser.add_argument(
        "--eta-prefactor",
        type=float,
        default=0.2,
        help="Prefactor for eta relative to regime ceiling heuristic",
    )
    parser.add_argument(
        "--max-eta",
        type=float,
        default=0.2,
        help="Hard cap on eta before stability backoff",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("simulations/outputs/master_ode_region_panel.png"),
        help="Output panel image path",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/master_ode_region_summary.csv"),
        help="Output CSV summary path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = make_region_cases(sigma=args.sigma)
    plot_panel(
        d=args.d,
        sigma=args.sigma,
        cases=cases,
        out_png=args.out_png,
        out_csv=args.out_csv,
        eta_prefactor=args.eta_prefactor,
        max_eta=args.max_eta,
    )


if __name__ == "__main__":
    main()
