"""Grid heatmap for last-iterate risk and optimal LR from the master ODE.

For each (kappa, gamma) on a grid:
1) infer phase and alpha from work/ode_scaling_limits/ode_scaling_limits.tex tables,
2) set eta_0 = d^{-alpha},
3) run local multiplicative search around eta_0 and K-step binary refinement,
4) choose horizon by time rule:
   - dynamical: t = TIME * d^(phase power),
   - sample:    t = TIME * d^(1+sigma-kappa),
5) record (eta_opt, risk_last).

Outputs:
- paired heatmaps (optimal LR, last-iterate risk),
- eta-scaled heatmap (eta_opt * d^alpha),
- CSV of per-grid-point results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from master_ode import build_model_from_scaling


@dataclass(frozen=True)
class SearchConfig:
    d: int
    sigma: float
    tau: float
    time_rule: str
    prescan_points: int
    prescan_candidates: int
    eta0_local_points: int
    eta0_local_span: float
    binary_steps: int
    max_dyadic_steps: int
    search_factor: float
    eta_min: float
    eta_max: float
    x0_r: float
    x0_v: float
    x0_c: float


def classify_phase_and_scales(kappa: float, gamma: float, sigma: float) -> tuple[str, float, float]:
    """Return (phase, alpha, time_power) from the manuscript tables.

    alpha is from eta ~ d^{-alpha}.
    time_power is the exponent e in t = tau * d^e.
    """
    res_line = 1.0 - sigma + kappa
    dense_left = sigma - 1.0

    # Phase A: coherent + dense-above.
    if (kappa <= dense_left) or ((dense_left < kappa <= sigma) and (gamma > res_line)):
        phase = "A"
        alpha = gamma - kappa
        time_power = gamma
        return phase, alpha, time_power

    # Phase B: sparse-above.
    if (kappa > sigma) and (gamma > res_line):
        phase = "B"
        alpha = gamma - kappa
        nu = gamma - (kappa - sigma)
        time_power = nu
        return phase, alpha, time_power

    # Phase C: dense-below (include resonance boundary in C for a closed partition).
    if (dense_left < kappa <= sigma) and (gamma <= res_line):
        phase = "C"
        alpha = 1.0 - sigma
        time_power = 1.0 - sigma + kappa
        return phase, alpha, time_power

    # Phase D: memoryless + sparse-below.
    if kappa > sigma:
        phase = "D"
        alpha = 1.0 - sigma
        time_power = 1.0
        return phase, alpha, time_power

    # Fallback (rare boundary mismatch): use coherent-side formulas.
    phase = "A"
    alpha = gamma - kappa
    time_power = gamma
    return phase, alpha, time_power


def compute_t_end(*, cfg: SearchConfig, kappa: float, dynamical_power: float) -> tuple[float, float]:
    """Return (t_end, time_power_used) under the configured time rule.

    - dynamical: t_end = TIME * d^{dynamical_power}
    - sample:    t_end = TIME * d^{1 - (sigma-kappa)_+}
    - active-iterates: t_end = TIME
    """
    if cfg.time_rule == "sample":
        time_power_used = 1.0 - max(0.0, cfg.sigma - kappa)
    elif cfg.time_rule == "active-iterates":
        time_power_used = 0.0
    else:
        time_power_used = dynamical_power
    t_end = float(cfg.tau * (cfg.d ** time_power_used))
    return t_end, float(time_power_used)


def last_iter_risk_from_master(
    *,
    d: int,
    sigma: float,
    kappa: float,
    gamma: float,
    eta: float,
    t_end: float,
    x0: np.ndarray,
) -> float:
    """Compute |R(t_end)| for the linear master ODE using eigen propagation."""
    eta = float(max(eta, 1e-300))
    try:
        model = build_model_from_scaling(d=d, kappa=kappa, sigma=sigma, gamma=gamma, eta=eta)
        A = model.A

        eigvals, eigvecs = np.linalg.eig(A)
        spectral_abscissa = float(np.max(np.real(eigvals)))
        if spectral_abscissa * t_end > 120.0:
            return np.inf

        coeff = np.linalg.solve(eigvecs, x0.astype(np.complex128))
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            xt = eigvecs @ (np.exp(eigvals * t_end) * coeff)
        if not np.all(np.isfinite(xt)):
            return np.inf

        r_t = float(np.real(xt[0]))
        if not np.isfinite(r_t):
            return np.inf
        return abs(r_t)
    except Exception:
        return np.inf


def dyadic_binary_search_opt_eta(
    *,
    cfg: SearchConfig,
    kappa: float,
    gamma: float,
    alpha: float,
    t_end: float,
) -> tuple[float, float, int]:
    """Search eta near eta0=d^{-alpha} via multiplicative steps, then binary refine."""
    x0 = np.array([cfg.x0_r, cfg.x0_v, cfg.x0_c], dtype=float)
    eval_cache: dict[float, float] = {}

    def eval_risk(eta_raw: float) -> float:
        eta = float(np.clip(eta_raw, cfg.eta_min, cfg.eta_max))
        if eta in eval_cache:
            return eval_cache[eta]
        risk = last_iter_risk_from_master(
            d=cfg.d,
            sigma=cfg.sigma,
            kappa=kappa,
            gamma=gamma,
            eta=eta,
            t_end=t_end,
            x0=x0,
        )
        eval_cache[eta] = risk
        return risk

    eta0 = float(np.clip(cfg.d ** (-alpha), cfg.eta_min, cfg.eta_max))
    factor = float(max(1.001, cfg.search_factor))

    def better(a: float, b: float) -> bool:
        if not np.isfinite(a):
            return False
        if not np.isfinite(b):
            return True
        return a < b

    eta_mid = eta0
    f_mid = eval_risk(eta_mid)
    eta_down = float(max(cfg.eta_min, eta_mid / factor))
    eta_up = float(min(cfg.eta_max, eta_mid * factor))
    f_down = eval_risk(eta_down)
    f_up = eval_risk(eta_up)

    go_down = False
    go_up = False
    if better(f_down, f_mid) or better(f_up, f_mid):
        if better(f_down, f_up):
            go_down = True
        else:
            go_up = True

    if go_down:
        eta_hi = eta_mid
        eta_mid = eta_down
        f_mid = f_down
        eta_lo = eta_down
        for _ in range(cfg.max_dyadic_steps):
            eta_next = float(max(cfg.eta_min, eta_mid / factor))
            if eta_next >= eta_mid:
                eta_lo = eta_next
                break
            f_next = eval_risk(eta_next)
            if better(f_next, f_mid):
                eta_hi = eta_mid
                eta_mid = eta_next
                f_mid = f_next
            else:
                eta_lo = eta_next
                break
        else:
            eta_lo = float(max(cfg.eta_min, eta_mid / factor))
            eval_risk(eta_lo)
    elif go_up:
        eta_lo = eta_mid
        eta_mid = eta_up
        f_mid = f_up
        eta_hi = eta_up
        for _ in range(cfg.max_dyadic_steps):
            eta_next = float(min(cfg.eta_max, eta_mid * factor))
            if eta_next <= eta_mid:
                eta_hi = eta_next
                break
            f_next = eval_risk(eta_next)
            if better(f_next, f_mid):
                eta_lo = eta_mid
                eta_mid = eta_next
                f_mid = f_next
            else:
                eta_hi = eta_next
                break
        else:
            eta_hi = float(min(cfg.eta_max, eta_mid * factor))
            eval_risk(eta_hi)
    else:
        eta_lo = eta_down
        eta_hi = eta_up

    if not (eta_lo < eta_mid < eta_hi):
        eta_lo = float(max(cfg.eta_min, eta_mid / np.sqrt(factor)))
        eta_hi = float(min(cfg.eta_max, eta_mid * np.sqrt(factor)))
        eval_risk(eta_lo)
        eval_risk(eta_hi)

    if eta_lo < eta_mid < eta_hi:
        for _ in range(cfg.binary_steps):
            eta_lm = np.sqrt(eta_lo * eta_mid)
            eta_rm = np.sqrt(eta_mid * eta_hi)
            if not (eta_lo < eta_lm < eta_mid < eta_rm < eta_hi):
                break
            f_lm = eval_risk(eta_lm)
            f_rm = eval_risk(eta_rm)
            if better(f_lm, f_mid):
                eta_hi = eta_mid
                eta_mid = eta_lm
                f_mid = f_lm
            elif better(f_rm, f_mid):
                eta_lo = eta_mid
                eta_mid = eta_rm
                f_mid = f_rm
            else:
                eta_lo = eta_lm
                eta_hi = eta_rm

    all_items = [(e, r) for e, r in eval_cache.items() if np.isfinite(r)]
    if not all_items:
        return eta0, np.inf, len(eval_cache)
    eta_best, risk_best = min(all_items, key=lambda p: p[1])
    return eta_best, risk_best, len(eval_cache)


def _worker(task: tuple[int, int, float, float, SearchConfig]) -> tuple[int, int, dict[str, float | str]]:
    i_k, i_g, kappa, gamma, cfg = task
    phase, alpha, dynamical_time_power = classify_phase_and_scales(kappa=kappa, gamma=gamma, sigma=cfg.sigma)
    t_end, time_power = compute_t_end(cfg=cfg, kappa=kappa, dynamical_power=dynamical_time_power)

    eta_opt, risk_opt, eval_count = dyadic_binary_search_opt_eta(
        cfg=cfg,
        kappa=kappa,
        gamma=gamma,
        alpha=alpha,
        t_end=t_end,
    )

    record: dict[str, float | str] = {
        "kappa": float(kappa),
        "gamma": float(gamma),
        "phase": phase,
        "alpha": float(alpha),
        "time_rule": cfg.time_rule,
        "time_power_dynamical": float(dynamical_time_power),
        "time_power": float(time_power),
        "t_end": float(t_end),
        "eta_init": float(np.clip(cfg.d ** (-alpha), cfg.eta_min, cfg.eta_max)),
        "eta_opt": float(eta_opt),
        "risk_opt": float(risk_opt),
        "eta_scaled": float(eta_opt * (cfg.d ** alpha)),
        "eval_count": float(eval_count),
    }
    return i_k, i_g, record


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kappa",
        "gamma",
        "phase",
        "alpha",
        "time_rule",
        "time_power_dynamical",
        "time_power",
        "t_end",
        "eta_init",
        "eta_opt",
        "risk_opt",
        "eta_scaled",
        "eval_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _overlay_phase_boundaries(ax: plt.Axes, *, sigma: float, kappa_min: float, kappa_max: float) -> None:
    k1 = sigma - 1.0
    k2 = sigma
    if kappa_min <= k1 <= kappa_max:
        ax.axvline(k1, color="white", lw=1.2, ls=":")
    if kappa_min <= k2 <= kappa_max:
        ax.axvline(k2, color="white", lw=1.2, ls=":")
    xs = np.linspace(kappa_min, kappa_max, 300)
    ys = 1.0 - sigma + xs
    ax.plot(xs, ys, color="white", lw=1.2, ls="-")


def _to_log10(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0.0)
    out[mask] = np.log10(arr[mask])
    return out


def plot_pair_heatmaps(
    *,
    kappa_min: float,
    kappa_max: float,
    gamma_min: float,
    gamma_max: float,
    sigma: float,
    eta_opt_grid: np.ndarray,
    risk_opt_grid: np.ndarray,
    fig_title: str,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))

    eta_log = _to_log10(eta_opt_grid)
    risk_log = _to_log10(risk_opt_grid)

    for ax, data, title, cbar_label in [
        (axes[0], eta_log, r"Optimal LR heatmap: $\log_{10}\eta_\star$", r"$\log_{10}\eta_\star$"),
        (axes[1], risk_log, r"Last-iterate risk heatmap: $\log_{10}R_{\mathrm{last}}$", r"$\log_{10}R_{\mathrm{last}}$"),
    ]:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(
            data,
            origin="lower",
            extent=[kappa_min, kappa_max, gamma_min, gamma_max],
            aspect="auto",
            cmap=cmap,
        )
        _overlay_phase_boundaries(ax, sigma=sigma, kappa_min=kappa_min, kappa_max=kappa_max)
        ax.set_title(title)
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_eta_scaled_heatmap(
    *,
    kappa_min: float,
    kappa_max: float,
    gamma_min: float,
    gamma_max: float,
    sigma: float,
    eta_scaled_grid: np.ndarray,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.6))

    data = _to_log10(eta_scaled_grid)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(
        data,
        origin="lower",
        extent=[kappa_min, kappa_max, gamma_min, gamma_max],
        aspect="auto",
        cmap=cmap,
    )
    _overlay_phase_boundaries(ax, sigma=sigma, kappa_min=kappa_min, kappa_max=kappa_max)
    ax.set_title(r"Scaled LR check: $\log_{10}\!\left(\eta_\star d^\alpha\right)$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\gamma$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\eta_\star d^\alpha)$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=1000, help="Fixed large dimension.")
    p.add_argument("--sigma", type=float, default=1.2, help="Batch exponent.")
    p.add_argument(
        "--prescan-points",
        type=int,
        default=33,
        help="Legacy option (unused in current eta0-local search).",
    )
    p.add_argument(
        "--prescan-candidates",
        type=int,
        default=3,
        help="Legacy option (unused in current eta0-local search).",
    )
    p.add_argument(
        "--eta0-local-points",
        type=int,
        default=17,
        help="Legacy option (unused in current eta0-local search).",
    )
    p.add_argument(
        "--eta0-local-span",
        type=float,
        default=2.0,
        help="Legacy option (unused in current eta0-local search).",
    )
    p.add_argument(
        "--time",
        type=float,
        default=12.0,
        dest="tau",
        help="TIME horizon multiplier used by the selected time rule.",
    )
    p.add_argument(
        "--time-rule",
        type=str,
        choices=["dynamical", "sample", "active-iterates"],
        default="dynamical",
        help=(
            "Rule for physical horizon t_end: "
            "dynamical => TIME*d^(phase power), "
            "sample => TIME*d^(1-(sigma-kappa)_+), "
            "active-iterates => TIME."
        ),
    )
    p.add_argument("--binary-search-steps", type=int, default=8)
    p.add_argument("--max-dyadic-steps", type=int, default=20)
    p.add_argument(
        "--search-factor",
        type=float,
        default=1.1,
        help="Multiplicative factor for outward search around eta0 (1.1 means +/-10%% steps).",
    )
    p.add_argument("--grid-fineness", type=int, default=31, help="Grid points per axis.")
    p.add_argument("--kappa-min", type=float, default=0.0)
    p.add_argument("--kappa-max", type=float, default=2.0)
    p.add_argument("--gamma-min", type=float, default=0.0)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--eta-min", type=float, default=1e-12)
    p.add_argument("--eta-max", type=float, default=1e4)
    p.add_argument("--x0-r", type=float, default=1.0)
    p.add_argument("--x0-v", type=float, default=1.0)
    p.add_argument("--x0-c", type=float, default=0.25)
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Multiprocessing workers.",
    )
    p.add_argument(
        "--out-heatmaps",
        type=Path,
        default=Path("simulations/outputs/master_ode_last_iter_heatmaps.png"),
    )
    p.add_argument(
        "--out-eta-scaled",
        type=Path,
        default=Path("simulations/outputs/master_ode_last_iter_eta_scaled_heatmap.png"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("simulations/outputs/master_ode_last_iter_grid.csv"),
    )
    p.add_argument(
        "--figure-title",
        type=str,
        default="Master ODE last-iterate search over (kappa, gamma)",
        help="Suptitle for the paired heatmap figure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = int(max(2, args.grid_fineness))
    kappas = np.linspace(args.kappa_min, args.kappa_max, n)
    gammas = np.linspace(args.gamma_min, args.gamma_max, n)

    cfg = SearchConfig(
        d=int(args.d),
        sigma=float(args.sigma),
        tau=float(args.tau),
        time_rule=str(args.time_rule),
        prescan_points=int(args.prescan_points),
        prescan_candidates=int(args.prescan_candidates),
        eta0_local_points=int(args.eta0_local_points),
        eta0_local_span=float(args.eta0_local_span),
        binary_steps=int(args.binary_search_steps),
        max_dyadic_steps=int(args.max_dyadic_steps),
        search_factor=float(args.search_factor),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        x0_r=float(args.x0_r),
        x0_v=float(args.x0_v),
        x0_c=float(args.x0_c),
    )

    tasks = [
        (i_k, i_g, float(kappa), float(gamma), cfg)
        for i_g, gamma in enumerate(gammas)
        for i_k, kappa in enumerate(kappas)
    ]

    eta_opt_grid = np.full((n, n), np.nan, dtype=float)  # [gamma_idx, kappa_idx]
    risk_opt_grid = np.full((n, n), np.nan, dtype=float)
    eta_scaled_grid = np.full((n, n), np.nan, dtype=float)
    rows: list[dict[str, float | str]] = []

    total = len(tasks)
    done = 0
    if int(args.num_workers) <= 1:
        iterator = map(_worker, tasks)
    else:
        pool = Pool(processes=int(args.num_workers))
        chunksize = max(1, total // (int(args.num_workers) * 8))
        iterator = pool.imap_unordered(_worker, tasks, chunksize=chunksize)

    try:
        for i_k, i_g, rec in iterator:
            eta_opt = float(rec["eta_opt"])
            risk_opt = float(rec["risk_opt"])
            eta_scaled = float(rec["eta_scaled"])
            eta_opt_grid[i_g, i_k] = eta_opt
            risk_opt_grid[i_g, i_k] = risk_opt
            eta_scaled_grid[i_g, i_k] = eta_scaled
            rows.append(rec)

            done += 1
            if done % max(1, total // 20) == 0:
                print(f"[progress] {done}/{total} points completed")
    finally:
        if int(args.num_workers) > 1:
            pool.close()
            pool.join()

    rows.sort(key=lambda r: (float(r["gamma"]), float(r["kappa"])))
    write_csv(args.out_csv, rows)

    plot_pair_heatmaps(
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        sigma=float(args.sigma),
        eta_opt_grid=eta_opt_grid,
        risk_opt_grid=risk_opt_grid,
        fig_title=str(args.figure_title),
        out_png=args.out_heatmaps,
    )
    plot_eta_scaled_heatmap(
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        sigma=float(args.sigma),
        eta_scaled_grid=eta_scaled_grid,
        out_png=args.out_eta_scaled,
    )


if __name__ == "__main__":
    main()
