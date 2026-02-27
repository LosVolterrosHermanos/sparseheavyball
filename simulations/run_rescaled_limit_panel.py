"""Compare regime-rescaled master ODE trajectories against simplified limit ODEs."""

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
from run_region_panel import RegionCase, make_region_cases, stable_model_for_case


@dataclass(frozen=True)
class RescalingSpec:
    time_label: str
    time_scale: float
    # If not None, x=(W,Z,R) is mapped to y by y_i = x_i / diag_i.
    diag_D: np.ndarray | None
    # "3d" -> linear y' = A_inf y; "1d" -> scalar R' = rate*R in rescaled time.
    limit_kind: str
    limit_rate: float | None
    theta: float
    # Multiplies the canonical 3D limit matrix when limit_kind=="3d".
    limit_matrix_scale: float


def _rho(model) -> float:
    return float(model.params.eps / (model.p_batch + model.params.eps))


def _eta_scale_for_case(d: int, sigma: float, case: RegionCase) -> float:
    # Mirrors the scaling heuristics used in run_region_panel.py.
    if case.expected_dim == "3D":
        return float(d ** (case.kappa - case.gamma))
    return float(d ** (sigma - 1.0))


def _star_constants(
    *,
    case: RegionCase,
    model,
    d: int,
    sigma: float,
    eta_used: float,
) -> dict[str, float]:
    """Finite-d proxies of constant-level co-scaling parameters."""
    d_f = float(d)
    p_star = float(model.params.p * (d_f ** case.kappa))
    B_star = float(model.params.B / (d_f ** sigma))
    eps_star = float(model.params.eps * (d_f ** case.gamma))
    eta_star = float(eta_used * (d_f ** (case.gamma - case.kappa)))
    bar_eta = float(eta_star * p_star / max(eps_star, 1e-30))
    rho_star = float(eps_star / max(p_star * B_star, 1e-30))
    return {
        "p_star": p_star,
        "B_star": B_star,
        "eps_star": eps_star,
        "eta_star": eta_star,
        "bar_eta": bar_eta,
        "rho_star": rho_star,
    }


def build_rescaling_spec(
    *,
    case: RegionCase,
    model,
    d: int,
    sigma: float,
    eta_used: float,
) -> RescalingSpec:
    """Return regime-specific rescaling from work/ode_scaling_limits/regimes/*.tex."""
    rho = _rho(model)
    eta_scale = max(_eta_scale_for_case(d=d, sigma=sigma, case=case), 1e-16)
    theta = float(eta_used / eta_scale)

    name = case.name.lower()
    if "coherent" in name:
        # coherent.tex (current): tau=t/d^gamma, D=diag(1, rho^2 p, rho^2).
        # Limit matrix is eps_* times canonical_3d(bar_eta), with
        # bar_eta = eta_* p_* / eps_*.
        stars = _star_constants(case=case, model=model, d=d, sigma=sigma, eta_used=eta_used)
        eps_star = stars["eps_star"]
        bar_eta = stars["bar_eta"]
        D = np.array(
            [1.0, rho * rho * model.params.p, rho * rho],
            dtype=float,
        )
        return RescalingSpec(
            "tau = t / d^gamma",
            float(d ** (-case.gamma)),
            D,
            "3d",
            None,
            bar_eta,
            eps_star,
        )

    if "dense above" in name:
        # incoherent_dense_above_resonance.tex (updated):
        # tau=t/d^gamma, D_id=diag(1, rho^2 d/B, rho^2 d/(B p)),
        # and limit matrix eps_* * canonical_3d(bar_eta), bar_eta=eta_*p_*/eps_*.
        stars = _star_constants(case=case, model=model, d=d, sigma=sigma, eta_used=eta_used)
        eps_star = stars["eps_star"]
        bar_eta = stars["bar_eta"]
        D = np.array(
            [
                1.0,
                rho * rho * (d / max(model.params.B, 1)),
                rho * rho * (d / max(model.params.B * model.params.p, 1e-30)),
            ],
            dtype=float,
        )
        return RescalingSpec(
            "tau = t / d^gamma",
            float(d ** (-case.gamma)),
            D,
            "3d",
            None,
            bar_eta,
            eps_star,
        )

    if "sparse above" in name:
        # incoherent_sparse_above_resonance.tex (updated):
        # tau=t/d^nu, nu=gamma-(kappa-sigma),
        # D_is=diag(1, rho^2 d/B, rho^2 d),
        # and limit matrix rho_* * canonical_3d(bar_eta), bar_eta=eta_*p_*/eps_*.
        nu = float(case.gamma - (case.kappa - sigma))
        stars = _star_constants(case=case, model=model, d=d, sigma=sigma, eta_used=eta_used)
        rho_star = stars["rho_star"]
        bar_eta = stars["bar_eta"]
        D = np.array(
            [
                1.0,
                rho * rho * (d / max(model.params.B, 1)),
                rho * rho * float(d),
            ],
            dtype=float,
        )
        return RescalingSpec(
            "tau = t / d^nu",
            float(d ** (-nu)),
            D,
            "3d",
            None,
            bar_eta,
            rho_star,
        )

    # 1D regimes: τ = Λ_slow t with Λ_slow := η B1.
    tau_scale = float(model.params.eta * model.B1)
    if "dense below" in name:
        # incoherent_dense_below_resonance.tex (updated boxed law):
        # tau = t / d^(1-sigma+kappa),
        # dR/dtau = -c_eff R, c_eff = c_Lambda (2 - eta d / B), c_Lambda=eta_* p_*.
        # Finite-d proxy: c_Lambda ~= (eta B1) d^(1-sigma+kappa).
        pow_exp = 1.0 - sigma + case.kappa
        tau_scale = float(d ** (-pow_exp))
        c_lambda = float(model.params.eta * model.B1 * (d ** pow_exp))
        q = float((eta_used * d) / max(model.params.B, 1))
        c_eff = c_lambda * (2.0 - q)
        return RescalingSpec(
            "tau = t / d^(1-sigma+kappa)",
            tau_scale,
            None,
            "1d",
            -c_eff,
            theta,
            1.0,
        )
    if "memoryless" in name:
        # incoherent_sparse_memoryless.tex (updated boxed law):
        # tau = t/d, dR/dtau = -c_eff R, c_eff=(eta d / B)(2-eta d / B).
        q = float((eta_used * d) / max(model.params.B, 1))
        c_eff = q * (2.0 - q)
        return RescalingSpec("tau = t / d", 1.0 / float(d), None, "1d", -c_eff, theta, 1.0)
    if "sparse below" in name:
        # incoherent_sparse_below_resonance.tex (updated boxed law):
        # tau = t/d, dR/dtau = -c_eff R, c_eff=(eta d / B)(2-eta d / B).
        q = float((eta_used * d) / max(model.params.B, 1))
        c_eff = q * (2.0 - q)
        return RescalingSpec("tau = t / d", 1.0 / float(d), None, "1d", -c_eff, theta, 1.0)

    raise ValueError(f"Unknown regime name: {case.name}")


def limit_matrix_3d(theta: float) -> np.ndarray:
    """Strict-interior 3D limit matrix used by coherent/dense-above/sparse-above."""
    return np.array(
        [
            [-2.0, 2.0, 0.0],
            [-theta, -1.0, 1.0],
            [0.0, -2.0 * theta, 0.0],
        ],
        dtype=float,
    )


def raw_to_wzr(master_raw: np.ndarray, model) -> np.ndarray:
    """Map raw state [R,V,C] to x=[W,Z,R] using manuscript scaling definitions."""
    R = master_raw[:, 0]
    V = master_raw[:, 1]
    C = master_raw[:, 2]
    lam_W = model.params.eps * model.params.eps * model.B2
    lam_Z = model.p_batch + model.params.eps
    lam_W = max(float(lam_W), 1e-30)
    lam_Z = max(float(lam_Z), 1e-30)
    W = V / lam_W
    Z = C / lam_Z
    return np.column_stack([W, Z, R])


def initial_raw_from_spec(spec: RescalingSpec, model) -> np.ndarray:
    """Choose initial conditions in rescaled coordinates and map back to raw [R,V,C]."""
    if spec.limit_kind == "3d":
        y0 = np.array([1.0, 0.6, 1.2], dtype=float)
        assert spec.diag_D is not None
        x0 = spec.diag_D * y0  # x=(W,Z,R)
    else:
        # 1D limit is on R; start with R(0)=1 and no initial fast-state excitation.
        x0 = np.array([0.0, 0.0, 1.0], dtype=float)  # x=(W,Z,R)

    lam_W = model.params.eps * model.params.eps * model.B2
    lam_Z = model.p_batch + model.params.eps
    raw_R = x0[2]
    raw_V = x0[0] * lam_W
    raw_C = x0[1] * lam_Z
    return np.array([raw_R, raw_V, raw_C], dtype=float)


def plot_rescaled_panel(
    *,
    d: int,
    sigma: float,
    out_png: Path,
    out_csv: Path,
    eta_prefactor: float,
    max_eta: float,
    tau_max: float,
    steps_per_unit_tau: int,
) -> None:
    cases = make_region_cases(sigma=sigma)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    rows = [
        "region,expected_dim,kappa,gamma,eta,theta,time_scale,max_real_eig,limit_kind,limit_rate"
    ]

    for ax, case in zip(axes, cases):
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

        raw_x0 = initial_raw_from_spec(spec, model)
        # Force a common rescaled-time window across all regimes.
        t_end = float(tau_max / max(spec.time_scale, 1e-30))
        n_steps = int(max(4000, tau_max * max(steps_per_unit_tau, 50)))
        t, x_raw = simulate_linear_rk4(A=A_raw, x0=raw_x0, t_end=t_end, n_steps=n_steps)
        x_wzr = raw_to_wzr(x_raw, model)

        tau = spec.time_scale * t
        tau_end = float(tau[-1])

        if spec.limit_kind == "3d":
            assert spec.diag_D is not None
            y_master = x_wzr / spec.diag_D[None, :]

            A_inf = spec.limit_matrix_scale * limit_matrix_3d(theta=spec.theta)
            _, y_limit = simulate_linear_rk4(
                A=A_inf,
                x0=y_master[0],
                t_end=max(tau_end, 1e-12),
                n_steps=n_steps,
            )

            master_colors = ["tab:blue", "tab:orange", "tab:green"]
            limit_colors = ["tab:red", "tab:purple", "tab:brown"]
            labels = ["y1", "y2", "y3"]

            for j in range(3):
                ax.plot(
                    tau,
                    y_master[:, j],
                    color=master_colors[j],
                    lw=2.0,
                    alpha=0.9,
                    label=f"master {labels[j]}",
                )
                ax.plot(
                    tau,
                    y_limit[:, j],
                    color=limit_colors[j],
                    lw=2.0,
                    ls="--",
                    alpha=0.95,
                    label=f"limit {labels[j]}",
                )
            ax.set_ylabel("rescaled state")
            limit_rate = np.nan
        else:
            R_master = x_wzr[:, 2]
            R0 = float(R_master[0])
            rate = float(spec.limit_rate)
            R_limit = R0 * np.exp(rate * tau)

            ax.plot(tau, R_master, color="tab:blue", lw=2.2, label="master R")
            ax.plot(tau, R_limit, color="tab:red", lw=2.2, ls="--", label="limit R")
            ax.set_ylabel("R (slow variable)")
            limit_rate = rate

        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(spec.time_label)
        ax.set_xlim(0.0, tau_max)
        ax.set_title(f"{case.name} ({case.expected_dim})", fontsize=11)
        ax.text(
            0.02,
            0.02,
            (
                f"k={case.kappa:.2f}, g={case.gamma:.2f}\n"
                f"eta={eta:.2e}, theta={spec.theta:.2e}\n"
                f"max Re(lambda_raw)={spectral_abscissa:.2e}"
            ),
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        rows.append(
            (
                f"{case.name},{case.expected_dim},{case.kappa:.6f},{case.gamma:.6f},"
                f"{eta:.10e},{spec.theta:.10e},{spec.time_scale:.10e},{spectral_abscissa:.10e},"
                f"{spec.limit_kind},{limit_rate:.10e}"
            )
        )

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        (
            f"Rescaled Master ODE vs Regime Limit ODEs (d={d}, sigma={sigma})\n"
            "Solid: transformed master trajectories. Dashed: strict-interior limit ODE."
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--eta-prefactor", type=float, default=0.2)
    parser.add_argument("--max-eta", type=float, default=0.2)
    parser.add_argument(
        "--tau-max",
        type=float,
        default=12.0,
        help="Common rescaled-time horizon for all panels.",
    )
    parser.add_argument(
        "--steps-per-unit-tau",
        type=int,
        default=700,
        help="Integration steps per 1 unit of rescaled time.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("simulations/outputs/master_ode_rescaled_limit_panel.png"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/master_ode_rescaled_limit_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_rescaled_panel(
        d=args.d,
        sigma=args.sigma,
        out_png=args.out_png,
        out_csv=args.out_csv,
        eta_prefactor=args.eta_prefactor,
        max_eta=args.max_eta,
        tau_max=args.tau_max,
        steps_per_unit_tau=args.steps_per_unit_tau,
    )


if __name__ == "__main__":
    main()
