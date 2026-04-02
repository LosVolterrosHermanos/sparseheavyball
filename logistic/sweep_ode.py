"""
Sweep a hyperparameter of the logistic ODE and plot KL convergence curves.

Default: sweep epsilon = 1-beta along a geometric grid.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import ODE machinery from validate_ode
from validate_ode import (
    ode_rhs, compute_kl_pop, sigmoid, sigmoid_prime,
    compute_coefficients,
)


def solve_ode_trajectory(eta, eps, rho, p, b_star, d, batch_size, n_steps):
    """Solve the 5-variable ODE and return (t, state, kl) arrays."""
    s0 = -rho
    state0 = [s0, 0.0, 0.0, 0.0, 0.0]
    t_eval = np.linspace(0, n_steps, min(n_steps + 1, 2001))

    sol = solve_ivp(ode_rhs, (0, n_steps), state0, t_eval=t_eval,
                    method='RK45',
                    args=(eta, eps, rho, p, b_star, d, batch_size),
                    rtol=1e-8, atol=1e-10, max_step=max(1.0, n_steps / 5000))

    # Compute KL at each time point
    kl = np.array([
        compute_kl_pop(sol.y[0, i], sol.y[2, i], rho, p, b_star)
        for i in range(len(sol.t))
    ])
    return sol.t, sol.y, kl


def main():
    parser = argparse.ArgumentParser(
        description='Sweep a hyperparameter of the logistic ODE')

    # Problem parameters
    parser.add_argument('--d', type=int, default=200)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=2.0)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Default beta (used when not sweeping epsilon)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-steps', type=int, default=3000)

    # Sweep configuration
    parser.add_argument('--sweep', type=str, default='epsilon',
                        choices=['epsilon', 'p', 'eta', 'rho', 'batch_size'],
                        help='Which parameter to sweep')
    parser.add_argument('--sweep-values', type=float, nargs='+', default=None,
                        help='Explicit sweep values (overrides --sweep-min/max/n)')
    parser.add_argument('--sweep-min', type=float, default=None)
    parser.add_argument('--sweep-max', type=float, default=None)
    parser.add_argument('--sweep-n', type=int, default=5)
    parser.add_argument('--sweep-log', action='store_true', default=True,
                        help='Use geometric (log) spacing (default)')
    parser.add_argument('--sweep-linear', action='store_true',
                        help='Use linear spacing')

    # Output
    parser.add_argument('--out', type=str, default='sweep_ode.pdf')
    parser.add_argument('--plot-vars', type=str, nargs='+',
                        default=['kl'],
                        choices=['s', 'u', 'Rp', 'Vp', 'Cp', 'kl'],
                        help='Which variables to plot')
    args = parser.parse_args()

    # Resolve sweep values
    if args.sweep_values is not None:
        sweep_vals = np.array(args.sweep_values)
    else:
        # Defaults per sweep parameter
        defaults = {
            'epsilon': (0.01, 0.16),
            'p':       (0.01, 0.3),
            'eta':     (0.01, 0.2),
            'rho':     (0.5, 4.0),
            'batch_size': (1, 32),
        }
        lo = args.sweep_min if args.sweep_min is not None else defaults[args.sweep][0]
        hi = args.sweep_max if args.sweep_max is not None else defaults[args.sweep][1]
        if args.sweep_linear:
            sweep_vals = np.linspace(lo, hi, args.sweep_n)
        else:
            sweep_vals = np.geomspace(lo, hi, args.sweep_n)

    # Variable index map
    var_idx = {'s': 0, 'u': 1, 'Rp': 2, 'Vp': 3, 'Cp': 4}
    var_labels = {
        's': r'$s$', 'u': r'$u$', 'Rp': r'$R_\perp$',
        'Vp': r'$V_\perp$', 'Cp': r'$C_\perp$',
        'kl': r'$\mathrm{KL}_{\mathrm{pop}}$',
    }

    plot_vars = args.plot_vars
    n_panels = len(plot_vars)
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    # Color map
    colors = cm.viridis(np.linspace(0.1, 0.9, len(sweep_vals)))

    # Sweep label
    sweep_sym = {
        'epsilon': r'\varepsilon',
        'p': 'p',
        'eta': r'\eta',
        'rho': r'\rho',
        'batch_size': 'B',
    }[args.sweep]

    for idx, val in enumerate(sweep_vals):
        # Build parameters for this sweep point
        eta = args.eta
        eps = 1 - args.beta
        rho = args.rho
        p_val = args.p
        batch_size = args.batch_size

        if args.sweep == 'epsilon':
            eps = val
        elif args.sweep == 'p':
            p_val = val
        elif args.sweep == 'eta':
            eta = val
        elif args.sweep == 'rho':
            rho = val
        elif args.sweep == 'batch_size':
            batch_size = int(round(val))

        b_star = np.log(p_val / (1 - p_val)) - rho**2 / 2

        t, state, kl = solve_ode_trajectory(
            eta, eps, rho, p_val, b_star, args.d, batch_size, args.n_steps)

        if args.sweep == 'batch_size':
            lbl = f'${sweep_sym}={batch_size}$'
        else:
            lbl = f'${sweep_sym}={val:.4g}$'

        for pi, pv in enumerate(plot_vars):
            ax = axes_flat[pi]
            if pv == 'kl':
                ax.plot(t, kl, color=colors[idx], linewidth=1.5, label=lbl)
            else:
                ax.plot(t, state[var_idx[pv]], color=colors[idx],
                        linewidth=1.5, label=lbl)

    # Format axes
    for pi, pv in enumerate(plot_vars):
        ax = axes_flat[pi]
        ax.set_ylabel(var_labels[pv], fontsize=13)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        if pv == 'kl':
            ax.set_yscale('log')
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    for ax in axes_flat:
        ax.set_xlabel('Step $k$', fontsize=11)

    # Title showing fixed parameters
    sweep_key_map = {
        'epsilon': 'varepsilon', 'p': 'p', 'eta': 'eta',
        'rho': 'rho', 'batch_size': 'B',
    }
    skip_key = sweep_key_map[args.sweep]
    fixed_parts = []
    for name, val in [('d', args.d), ('p', args.p), (r'\rho', args.rho),
                      (r'\eta', args.eta), (r'\varepsilon', round(1 - args.beta, 6)),
                      ('B', args.batch_size)]:
        latex_key = name.replace('\\', '')
        if latex_key != skip_key:
            fixed_parts.append(f'${name}={val}$')
    fig.suptitle(
        f'ODE sweep over ${sweep_sym}$   (fixed: {", ".join(fixed_parts)})',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(args.out, bbox_inches='tight')
    print(f'Saved {args.out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
