import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Any

def compute_se_treatment_effect(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    method: Literal["mean_diff", "proportion", "ols", "bootstrap"] = "mean_diff",
    covariates: Optional[list[str]] = None,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute the standard error (SE) of the treatment effect estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Data with outcome and group.
    outcome_col : str
        Outcome column name.
    group_col : str
        Binary group column (0/1 or two categories).
    method : {"mean_diff", "proportion", "ols", "bootstrap"}
        How to compute SE:
          - "mean_diff": Welch SE for difference in means (continuous)
          - "proportion": SE for difference in proportions (binary outcome)
          - "ols": SE of treatment coefficient from OLS (with optional covariates)
          - "bootstrap": Nonparametric bootstrap SE (generic fallback)
    covariates : list of str or None
        Covariates for OLS / RA / CUPED.
    n_boot : int
        Number of bootstrap resamples if method="bootstrap".
    random_state : int
        Random seed for bootstrap.

    Returns
    -------
    dict with:
      - se: float
      - estimate: float
      - method: str
      - details: dict
    """

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    # Drop rows with missing outcome or group
    d = df[[outcome_col, group_col] + (covariates or [])].dropna().copy()

    # Encode group to 0/1 if needed
    g = d[group_col]
    if g.nunique() != 2:
        raise ValueError("group_col must have exactly two groups for this SE calculator.")

    # Map to 0/1
    if not set(g.unique()).issubset({0, 1}):
        levels = list(g.unique())
        mapping = {levels[0]: 0, levels[1]: 1}
        d["_T"] = g.map(mapping)
    else:
        d["_T"] = g.astype(int)

    y = d[outcome_col].astype(float).values
    T = d["_T"].values

    # Split groups
    y0 = y[T == 0]
    y1 = y[T == 1]

    n0 = len(y0)
    n1 = len(y1)

    if n0 < 2 or n1 < 2:
        raise ValueError("Each group must have at least 2 observations.")

    # ----------------------------
    # 1) Difference in means (Welch)
    # ----------------------------
    if method == "mean_diff":
        m0 = np.mean(y0)
        m1 = np.mean(y1)
        v0 = np.var(y0, ddof=1)
        v1 = np.var(y1, ddof=1)

        estimate = m1 - m0
        se = np.sqrt(v0 / n0 + v1 / n1)

        return {
            "se": float(se),
            "estimate": float(estimate),
            "method": "mean_diff",
            "details": {"n0": n0, "n1": n1, "v0": float(v0), "v1": float(v1)},
        }

    # ----------------------------
    # 2) Difference in proportions (binary outcome)
    # ----------------------------
    if method == "proportion":
        # y should be 0/1
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("For method='proportion', outcome must be binary (0/1).")

        p0 = np.mean(y0)
        p1 = np.mean(y1)

        estimate = p1 - p0
        se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)

        return {
            "se": float(se),
            "estimate": float(estimate),
            "method": "proportion",
            "details": {"n0": n0, "n1": n1, "p0": float(p0), "p1": float(p1)},
        }

    # ----------------------------
    # 3) OLS / Regression Adjustment / CUPED
    # ----------------------------
    if method == "ols":
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError("statsmodels is required for method='ols'.")

        X_cols = ["_T"]
        if covariates:
            X_cols += covariates

        X = d[X_cols].copy()
        X = sm.add_constant(X)

        model = sm.OLS(d[outcome_col].astype(float).values, X).fit()

        estimate = model.params["_T"]
        se = model.bse["_T"]

        return {
            "se": float(se),
            "estimate": float(estimate),
            "method": "ols",
            "details": {
                "n": int(len(d)),
                "covariates": covariates or [],
                "r2": float(model.rsquared),
            },
        }

    # ----------------------------
    # 4) Bootstrap SE (generic fallback)
    # ----------------------------
    if method == "bootstrap":
        rng = np.random.default_rng(random_state)

        ests = []
        n = len(d)

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            db = d.iloc[idx]

            yb = db[outcome_col].astype(float).values
            Tb = db["_T"].values

            y0b = yb[Tb == 0]
            y1b = yb[Tb == 1]

            if len(y0b) == 0 or len(y1b) == 0:
                continue

            ests.append(np.mean(y1b) - np.mean(y0b))

        if len(ests) < 10:
            raise RuntimeError("Too few valid bootstrap samples to estimate SE.")

        ests = np.array(ests)
        estimate = np.mean(ests)
        se = np.std(ests, ddof=1)

        return {
            "se": float(se),
            "estimate": float(estimate),
            "method": "bootstrap",
            "details": {"n_boot": n_boot, "n_used": int(len(ests))},
        }

    raise ValueError(f"Unknown method '{method}'.")