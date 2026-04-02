# Logistic Regression ODE

Five-variable ODE for SGD with momentum on a rare-event binary classification problem.
See `work/logistic_ode/logistic_ode.tex` for the full derivation.

## Problem

- **Data:** Class 1 (probability `1-p`): `X = Z`. Class 2 (probability `p`): `X = mu + Z`, with `Z ~ N(0, I_d)`.
- **Model:** Logistic regression with fixed Bayes-optimal bias `b* = log(p/(1-p)) - ||mu||^2/2`.
- **Optimizer:** SGD with momentum (batch size `B`, learning rate `eta`, momentum `beta`).

## State Variables

The ODE tracks five per-realization quantities:

| Variable | Definition | Role |
|----------|-----------|------|
| `s` | `<theta - mu, mu_hat>` | Signal error (alignment with target) |
| `u` | `<M, mu_hat>` | Momentum along signal direction |
| `R_perp` | `\|theta_perp\|^2` | Bulk energy (orthogonal to signal) |
| `V_perp` | `\|M_perp\|^2` | Momentum energy (orthogonal) |
| `C_perp` | `<theta_perp, M_perp>` | Error-momentum correlation (orthogonal) |

The system closes exactly by rotational symmetry in the `(d-1)`-dimensional orthogonal subspace.

## Scripts

### `validate_ode.py`

Compares the ODE against actual SGD-M runs. Plots all 5 state variables plus
the population KL divergence.

```bash
# Basic (B=1)
python logistic/validate_ode.py

# With batching
python logistic/validate_ode.py --batch-size 4 --out logistic/validate_B4.pdf

# Full options
python logistic/validate_ode.py --d 200 --p 0.1 --rho 2.0 --eta 0.05 \
    --beta 0.9 --batch-size 1 --n-steps 2000 --n-runs 50
```

### `sweep_ode.py`

Sweeps a hyperparameter and plots ODE trajectories on a single figure.

```bash
# Sweep epsilon (= 1-beta)
python logistic/sweep_ode.py --sweep epsilon \
    --sweep-values 0.16 0.08 0.04 0.02 0.01 \
    --plot-vars kl s Rp --n-steps 3000

# Sweep class probability
python logistic/sweep_ode.py --sweep p \
    --sweep-values 0.01 0.03 0.1 0.3 \
    --plot-vars kl s --n-steps 5000

# Sweep batch size
python logistic/sweep_ode.py --sweep batch_size \
    --sweep-values 1 2 4 8 16 \
    --plot-vars kl Rp --n-steps 3000

# Available sweep parameters: epsilon, p, eta, rho, batch_size
# Available plot variables: s, u, Rp, Vp, Cp, kl
```
