"""
Validate the 5-variable ODE for logistic SGD with momentum.

Runs SGD-M on the rare-event logistic regression problem and compares
the per-realization trajectory of (s, u, R_perp, V_perp, C_perp)
against the ODE prediction.  Plots both on a single matplotlib axis.

The ODE is the conditional-drift system (S1)-(S2), (B1)-(B3) from
logistic_ode.tex.  Because it describes the *expected* per-step change,
a single SGD run will fluctuate around the ODE curve.  We also average
over multiple SGD runs to show convergence to the ODE.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt


# ── Gaussian-sigmoid expectations via quadrature ───────────────────

def _gauss_quad(func, mu_G, var_G, n_quad=80):
    """E[func(G)] where G ~ N(mu_G, var_G), via Gauss-Hermite."""
    if var_G < 1e-14:
        return func(mu_G)
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    # hermite_e nodes are for weight exp(-x^2/2), so G = mu + sqrt(var)*x
    g_vals = mu_G + np.sqrt(var_G) * nodes
    return np.sum(weights * func(g_vals)) / np.sqrt(2 * np.pi)


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def sigma_sq_double_prime(z):
    """d^2/dz^2 [sigma(z)^2] = 2 sigma'(z) sigma(z) (2 - 3 sigma(z))."""
    s = sigmoid(z)
    sp = s * (1 - s)
    return 2 * sp * s * (2 - 3 * s)


def one_minus_sigma_sq_double_prime(z):
    """d^2/dz^2 [(1-sigma(z))^2] = 2 sigma'(z) (1-sigma(z)) (3 sigma(z) - 1)."""
    s = sigmoid(z)
    sp = s * (1 - s)
    return 2 * sp * (1 - s) * (3 * s - 1)


# ── Coefficient functions ──────────────────────────────────────────

def compute_coefficients(m, q, rho, p, b_star):
    """Compute A, B, D0, Dtheta as functions of (m, q)."""
    var_G = max(q, 0.0)
    mu1 = b_star
    mu2 = m * rho + b_star

    A = (1 - p) * _gauss_quad(sigmoid_prime, mu1, var_G) + \
        p * _gauss_quad(sigmoid_prime, mu2, var_G)

    B = p * (_gauss_quad(sigmoid, mu2, var_G) - 1)

    D0 = (1 - p) * _gauss_quad(lambda z: sigmoid(z)**2, mu1, var_G) + \
         p * _gauss_quad(lambda z: (1 - sigmoid(z))**2, mu2, var_G)

    Dtheta = (1 - p) * _gauss_quad(sigma_sq_double_prime, mu1, var_G) + \
             p * _gauss_quad(one_minus_sigma_sq_double_prime, mu2, var_G)

    return A, B, D0, Dtheta


# ── ODE right-hand side ───────────────────────────────────────────

def ode_rhs(t, state, eta, eps, rho, p, b_star, d, batch_size=1):
    """RHS of the 5-variable ODE: [s, u, R_perp, V_perp, C_perp]."""
    s, u, Rp, Vp, Cp = state
    Rp = max(Rp, 0.0)
    Vp = max(Vp, 0.0)

    m = s + rho
    q = m**2 + Rp
    Bb = batch_size

    calA, calB, D0, Dtheta = compute_coefficients(m, q, rho, p, b_star)
    f = calA * m + calB * rho
    N_perp_single = (d - 1) * D0 + Rp * Dtheta
    # Batched noise: (1/B)*N_perp + ((B-1)/B)*A^2*R_perp
    N_perp = N_perp_single / Bb + (Bb - 1) / Bb * calA**2 * Rp
    bm = 1 - eps  # beta = 1 - epsilon

    ds = -eta * bm * u - eta * eps * f
    du = eps * (f - u)
    dRp = (-2 * eta * bm * Cp
           - 2 * eta * eps * calA * Rp
           + eta**2 * bm**2 * Vp
           + 2 * eta**2 * bm * eps * calA * Cp
           + eta**2 * eps**2 * N_perp)
    dVp = (-(2 * eps - eps**2) * Vp
           + 2 * bm * eps * calA * Cp
           + eps**2 * N_perp)
    dCp = (-eps * Cp
           + eps * calA * Rp
           - eta * bm**2 * Vp
           - 2 * eta * bm * eps * calA * Cp
           - eta * eps**2 * N_perp)

    return [ds, du, dRp, dVp, dCp]


# ── Population KL divergence ───────────────────────────────────────

def softplus(z):
    """Numerically stable log(1 + exp(z))."""
    return np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -500, 20))))


def kl_pointwise(a, c):
    """KL(P*(.|x) || P_theta(.|x)) = sigma(a)(a-c) + phi(c) - phi(a)."""
    return sigmoid(a) * (a - c) + softplus(c) - softplus(a)


def compute_kl_pop(s_val, Rp_val, rho, p, b_star, n_quad=60):
    """Population KL from ODE state (s, R_perp) via 2D Gauss-Hermite."""
    m = s_val + rho
    sqrt_Rp = np.sqrt(max(Rp_val, 0.0))

    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    w_norm = weights / np.sqrt(2 * np.pi)  # normalized weights

    # 2D grid: xi (signal noise) x zeta (bulk noise)
    xi = nodes       # shape (n_quad,)
    zeta = nodes     # shape (n_quad,)

    # Outer product weights
    W = np.outer(w_norm, w_norm)  # (n_quad, n_quad)

    XI = xi[:, None]     # (n_quad, 1)
    ZETA = zeta[None, :]  # (1, n_quad)

    # Class 1: a1 = rho*xi + b*, c1 = m*xi + sqrt(Rp)*zeta + b*
    a1 = rho * XI + b_star
    c1 = m * XI + sqrt_Rp * ZETA + b_star
    kl1 = np.sum(W * kl_pointwise(a1, c1))

    # Class 2: a2 = rho^2 + rho*xi + b*, c2 = m*rho + m*xi + sqrt(Rp)*zeta + b*
    a2 = rho**2 + rho * XI + b_star
    c2 = m * rho + m * XI + sqrt_Rp * ZETA + b_star
    kl2 = np.sum(W * kl_pointwise(a2, c2))

    return (1 - p) * kl1 + p * kl2


def compute_kl_from_theta(theta, mu, b_star, p, n_mc=5000, rng=None):
    """Population KL by Monte Carlo over X, for a given theta."""
    if rng is None:
        rng = np.random.default_rng()
    d = len(mu)
    rho = np.linalg.norm(mu)

    kl_sum = 0.0
    for _ in range(n_mc):
        Z = rng.standard_normal(d)
        y_class = 2 if rng.random() < p else 1
        x = mu + Z if y_class == 2 else Z
        a = mu @ x + b_star
        c = theta @ x + b_star
        kl_sum += kl_pointwise(a, c)
    return kl_sum / n_mc


# ── SGD-M simulation ──────────────────────────────────────────────

def run_sgd_m(theta0, M0, mu, p, b_star, eta, beta, n_steps, rng,
              batch_size=1):
    """Run SGD-M and record the 5-tuple at each step."""
    d = len(mu)
    rho = np.linalg.norm(mu)
    mu_hat = mu / rho
    eps = 1 - beta

    theta = theta0.copy()
    M = M0.copy()

    # Pre-allocate trajectory storage
    traj = np.zeros((n_steps + 1, 5))

    def extract_state(th, Mo):
        e = th - mu
        s = e @ mu_hat
        u_val = Mo @ mu_hat
        th_perp = th - (th @ mu_hat) * mu_hat
        M_perp = Mo - u_val * mu_hat
        Rp = np.dot(th_perp, th_perp)
        Vp = np.dot(M_perp, M_perp)
        Cp = np.dot(th_perp, M_perp)
        return np.array([s, u_val, Rp, Vp, Cp])

    traj[0] = extract_state(theta, M)

    for k in range(n_steps):
        # Batched gradient
        g_bar = np.zeros(d)
        for _ in range(batch_size):
            y = 2 if rng.random() < p else 1
            Z = rng.standard_normal(d)
            x = mu + Z if y == 2 else Z
            y_tilde = 1.0 if y == 2 else 0.0
            logit = theta @ x + b_star
            pred = sigmoid(logit)
            g_bar += (pred - y_tilde) * x
        g_bar /= batch_size

        # Momentum update
        M = (1 - eps) * M + eps * g_bar

        # Parameter update
        theta = theta - eta * M

        traj[k + 1] = extract_state(theta, M)

    return traj


# ── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate logistic ODE vs SGD-M')
    parser.add_argument('--d', type=int, default=200)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=2.0)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-steps', type=int, default=2000)
    parser.add_argument('--n-runs', type=int, default=50)
    parser.add_argument('--out', type=str, default='validate_ode.pdf')
    args = parser.parse_args()

    # Parameters
    d = args.d
    p = args.p
    rho = args.rho
    eta = args.eta
    beta = args.beta
    eps = 1 - beta
    batch_size = args.batch_size
    n_steps = args.n_steps
    n_runs = args.n_runs

    # Derived
    mu = np.zeros(d)
    mu[0] = rho  # signal in first coordinate
    b_star = np.log(p / (1 - p)) - rho**2 / 2

    # Initial conditions: theta=0, M=0
    theta0 = np.zeros(d)
    M0 = np.zeros(d)
    s0 = -rho  # since e = theta - mu = -mu, s = <e, mu_hat> = -rho
    state0 = [s0, 0.0, 0.0, 0.0, 0.0]

    # ── Solve ODE ──
    t_span = (0, n_steps)
    t_eval = np.arange(n_steps + 1, dtype=float)
    sol = solve_ivp(ode_rhs, t_span, state0, t_eval=t_eval, method='RK45',
                    args=(eta, eps, rho, p, b_star, d, batch_size),
                    rtol=1e-8, atol=1e-10, max_step=1.0)

    # ── Run SGD-M ──
    rng = np.random.default_rng(42)
    trajs = []
    for i in range(n_runs):
        traj = run_sgd_m(theta0, M0, mu, p, b_star, eta, beta, n_steps,
                         rng=np.random.default_rng(1000 + i),
                         batch_size=batch_size)
        trajs.append(traj)
    trajs = np.array(trajs)  # (n_runs, n_steps+1, 5)
    mean_traj = trajs.mean(axis=0)

    # ── Compute KL along ODE trajectory ──
    print('Computing KL along ODE trajectory...')
    # Subsample for speed (KL quadrature at every step is expensive)
    kl_every = 10
    kl_idx = np.arange(0, n_steps + 1, kl_every)
    kl_ode = np.array([
        compute_kl_pop(sol.y[0, k], sol.y[2, k], rho, p, b_star)
        for k in kl_idx
    ])

    # ── Compute KL along SGD trajectories (from state variables) ──
    print('Computing KL along SGD trajectories...')
    kl_sgd_all = np.zeros((n_runs, len(kl_idx)))
    for i in range(n_runs):
        for j, k in enumerate(kl_idx):
            s_k, _, Rp_k, _, _ = trajs[i, k]
            kl_sgd_all[i, j] = compute_kl_pop(s_k, Rp_k, rho, p, b_star)
    kl_sgd_mean = kl_sgd_all.mean(axis=0)

    # ── Plot ──
    labels = [r'$s$', r'$u$', r'$R_\perp$', r'$V_\perp$', r'$C_\perp$',
              r'$\mathrm{KL}_{\mathrm{pop}}$']
    t = np.arange(n_steps + 1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for i in range(5):
        ax = axes[i]
        # Individual SGD runs (light)
        for j in range(min(n_runs, 5)):
            ax.plot(t, trajs[j, :, i], color='C0', alpha=0.15, linewidth=0.5)
        # SGD mean
        ax.plot(t, mean_traj[:, i], color='C0', linewidth=1.5,
                label=f'SGD mean ({n_runs} runs)')
        # ODE
        ax.plot(sol.t, sol.y[i], color='C1', linewidth=2, linestyle='--',
                label='ODE')
        ax.set_ylabel(labels[i], fontsize=13)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # 6th panel: KL divergence
    ax = axes[5]
    for j in range(min(n_runs, 5)):
        ax.plot(kl_idx, kl_sgd_all[j], color='C0', alpha=0.15, linewidth=0.5)
    ax.plot(kl_idx, kl_sgd_mean, color='C0', linewidth=1.5,
            label=f'SGD mean ({n_runs} runs)')
    ax.plot(kl_idx, kl_ode, color='C1', linewidth=2, linestyle='--',
            label='ODE')
    ax.set_ylabel(labels[5], fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    for ax in axes[3:6]:
        ax.set_xlabel('Step $k$', fontsize=11)

    fig.suptitle(
        f'ODE vs SGD-M validation:  $d={d}$, $p={p}$, '
        rf'$\rho={rho}$, $\eta={eta}$, $\beta={beta}$, $B={batch_size}$',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = args.out
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
