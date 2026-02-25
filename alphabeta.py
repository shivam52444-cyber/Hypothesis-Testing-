import numpy as np
from scipy.stats import norm

def optimize_alpha_beta(
    mde: float,
    se: float,
    cost_fp: float,
    cost_fn: float,
    two_sided: bool = True,
    alpha_grid: np.ndarray | None = None,
    power_warn_threshold: float = 0.7,
):
    """
    Choose optimal alpha and beta by minimizing expected loss:
        Loss(alpha) = alpha * cost_fp + beta(alpha) * cost_fn

    using the normal approximation:
        MDE = (z_{1-alpha/2} + z_{1-beta}) * SE   (two-sided)
        MDE = (z_{1-alpha}   + z_{1-beta}) * SE   (one-sided)

    Parameters
    ----------
    mde : float
        Minimum Detectable Effect (in same units as the effect estimate).
    se : float
        Standard error of the estimator (from data or planning assumptions).
    cost_fp : float
        Cost of a false positive (Type I error).
    cost_fn : float
        Cost of a false negative (Type II error).
    two_sided : bool
        Whether the test is two-sided (default True). If False, uses one-sided.
    alpha_grid : np.ndarray or None
        Grid of alpha values to search. If None, uses a log-spaced grid.
    power_warn_threshold : float
        If optimal power < this, flag as underpowered.

    Returns
    -------
    result : dict
        {
          "alpha_opt": float,
          "beta_opt": float,
          "power_opt": float,
          "loss_opt": float,
          "is_underpowered": bool,
          "details": {
              "mde": mde,
              "se": se,
              "two_sided": two_sided
          }
        }
    """

    if se <= 0:
        raise ValueError("se must be > 0.")
    if mde <= 0:
        raise ValueError("mde must be > 0.")
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("Costs must be non-negative.")

    # Build a reasonable alpha grid if not provided
    if alpha_grid is None:
        # Avoid extreme 0 or 1; cover practical range finely
        alpha_grid = np.unique(
            np.concatenate([
                np.logspace(-6, -2, 200),   # very small alphas
                np.linspace(0.001, 0.2, 800) # practical range
            ])
        )

    # Precompute signal-to-noise ratio
    snr = mde / se

    best = {
        "alpha": None,
        "beta": None,
        "power": None,
        "loss": np.inf
    }

    for alpha in alpha_grid:
        # Guard against invalid alphas
        if alpha <= 0 or alpha >= 1:
            continue

        # Get z for alpha
        if two_sided:
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        # From: z_{1-beta} = snr - z_alpha
        z_1_minus_beta = snr - z_alpha

        # Compute beta = P(Z <= z_{1-beta} negative side)
        # 1 - beta = Phi(z_{1-beta})  -> beta = 1 - Phi(z_{1-beta})
        power = norm.cdf(z_1_minus_beta)
        beta = 1 - power

        # Clip for numerical safety
        beta = float(np.clip(beta, 0.0, 1.0))
        power = float(np.clip(power, 0.0, 1.0))

        loss = alpha * cost_fp + beta * cost_fn

        if loss < best["loss"]:
            best.update({
                "alpha": float(alpha),
                "beta": beta,
                "power": power,
                "loss": float(loss)
            })

    if best["alpha"] is None:
        raise RuntimeError("Failed to find optimal alpha on the provided grid.")

    result = {
        "alpha_opt": best["alpha"],
        "beta_opt": best["beta"],
        "power_opt": best["power"],
        "loss_opt": best["loss"],
        "is_underpowered": best["power"] < power_warn_threshold,
        "details": {
            "mde": float(mde),
            "se": float(se),
            "two_sided": bool(two_sided),
            "snr": float(snr),
            "power_warn_threshold": float(power_warn_threshold),
        }
    }

    return result