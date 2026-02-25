import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

def detect_outliers_and_heavy_tails(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: Optional[str] = None,
    z_thresh: float = 3.5,          # robust z threshold (MAD-based)
    max_outlier_frac: float = 0.01, # 1% default
    max_sd_iqr_ratio: float = 1.5,  # heuristic: SD much larger than IQR-based scale
    min_n: int = 20,                # below this, be conservative
) -> Dict[str, Any]:
    """
    Detect whether outcome data likely violates parametric assumptions due to
    outliers / heavy tails, and recommend switching to non-parametric tests.

    Returns a dict with:
      - use_nonparametric: bool
      - reason: str
      - diagnostics: dict
    """

    if outcome_col not in df.columns:
        raise ValueError(f"Column '{outcome_col}' not found in DataFrame.")

    x = df[outcome_col].dropna().astype(float)

    n = len(x)
    if n == 0:
        raise ValueError("Outcome column has no non-missing values.")

    # --- Robust location and scale ---
    median = np.median(x)
    mad = np.median(np.abs(x - median))

    # Consistent MAD scale for normal data
    if mad == 0:
        # All values identical or nearly so
        mad_scale = 0.0
    else:
        mad_scale = 1.4826 * mad

    # Robust z-scores
    if mad_scale > 0:
        z_robust = (x - median) / mad_scale
        outlier_mask = np.abs(z_robust) > z_thresh
        outlier_frac = float(np.mean(outlier_mask))
    else:
        z_robust = np.zeros_like(x)
        outlier_mask = np.zeros_like(x, dtype=bool)
        outlier_frac = 0.0

    # --- Classical vs robust spread comparison ---
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    # For normal data, SD ≈ IQR / 1.349
    if iqr > 0:
        sd_iqr_ratio = sd / (iqr / 1.349)
    else:
        sd_iqr_ratio = np.inf

    # --- Robust skew proxy ---
    # Compare median to mean in units of robust scale
    mean = float(np.mean(x))
    if mad_scale > 0:
        robust_skew = (mean - median) / mad_scale
    else:
        robust_skew = 0.0

    # --- Per-group outlier rates (if group_col provided) ---
    per_group = {}
    if group_col is not None and group_col in df.columns:
        for g, sub in df[[group_col, outcome_col]].dropna().groupby(group_col):
            xs = sub[outcome_col].astype(float).values
            if len(xs) < 5:
                continue
            med_g = np.median(xs)
            mad_g = np.median(np.abs(xs - med_g))
            if mad_g > 0:
                mad_scale_g = 1.4826 * mad_g
                z_g = (xs - med_g) / mad_scale_g
                frac_g = float(np.mean(np.abs(z_g) > z_thresh))
            else:
                frac_g = 0.0
            per_group[g] = frac_g

    # --- Decision logic ---
    reasons = []

    if n < min_n:
        reasons.append(f"Small sample size (n={n}) → be conservative")

    if outlier_frac > max_outlier_frac:
        reasons.append(
            f"Outlier fraction {outlier_frac:.2%} > threshold {max_outlier_frac:.2%}"
        )

    if sd_iqr_ratio > max_sd_iqr_ratio:
        reasons.append(
            f"SD/IQR-based-scale ratio {sd_iqr_ratio:.2f} > {max_sd_iqr_ratio} (heavy tails likely)"
        )

    if abs(robust_skew) > 1.0:
        reasons.append(
            f"Strong skew detected (robust skew ≈ {robust_skew:.2f})"
        )

    # If any strong reason exists, recommend non-parametric
    use_nonparametric = len(reasons) > 0

    if use_nonparametric:
        reason_text = " ; ".join(reasons)
    else:
        reason_text = "No strong evidence of problematic outliers/heavy tails."

    diagnostics = {
        "n": n,
        "median": float(median),
        "mean": float(mean),
        "sd": float(sd),
        "mad_scale": float(mad_scale),
        "iqr": float(iqr),
        "sd_iqr_ratio": float(sd_iqr_ratio),
        "robust_skew": float(robust_skew),
        "outlier_fraction": float(outlier_frac),
        "per_group_outlier_fraction": per_group,
        "z_threshold": z_thresh,
        "thresholds": {
            "max_outlier_frac": max_outlier_frac,
            "max_sd_iqr_ratio": max_sd_iqr_ratio,
            "min_n": min_n,
        },
    }

    return {
        "use_nonparametric": use_nonparametric,
        "reason": reason_text,
        "diagnostics": diagnostics,
    }