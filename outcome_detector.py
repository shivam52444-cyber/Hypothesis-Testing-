import numpy as np
import pandas as pd


def detect_outcome_type(series: pd.Series):
    s = series.dropna()
    reasons = []

    if len(s) == 0:
        return {
            "suggested_type": "unknown",
            "confidence": 0.0,
            "reasons": ["Column has only missing values"],
        }

    unique_vals = s.unique()
    n_unique = len(unique_vals)

    is_numeric = pd.api.types.is_numeric_dtype(s)

    is_integer_like = False
    if is_numeric:
        is_integer_like = np.all(np.isclose(s, np.round(s)))

    min_val = s.min() if is_numeric else None
    max_val = s.max() if is_numeric else None

    # 1) Binary
    if is_numeric and set(np.unique(s.astype(float))) <= {0.0, 1.0}:
        reasons.append("Only values are {0,1} -> classic binary outcome")
        return {
            "suggested_type": "binary",
            "confidence": 0.95,
            "reasons": reasons,
        }

    # 2) Count should be checked before "few unique numeric levels"
    if is_numeric and is_integer_like and min_val >= 0:
        reasons.append("Numeric, integer-like, non-negative -> looks like count data")
        reasons.append(f"Range: [{min_val}, {max_val}]")
        return {
            "suggested_type": "count",
            "confidence": 0.8,
            "reasons": reasons,
        }

    # 3) Categorical
    if (not is_numeric) or n_unique <= 10:
        if not is_numeric:
            reasons.append("Non-numeric values -> categorical by nature")
            return {
                "suggested_type": "categorical",
                "confidence": 0.9,
                "reasons": reasons,
            }
        reasons.append(f"Only {n_unique} unique numeric values -> likely encoded categories")
        return {
            "suggested_type": "categorical",
            "confidence": 0.7,
            "reasons": reasons,
        }

    # 4) Continuous
    if is_numeric:
        reasons.append("Numeric with decimals or wide range -> looks continuous")
        reasons.append(f"Range: [{min_val}, {max_val}]")
        return {
            "suggested_type": "continuous",
            "confidence": 0.8,
            "reasons": reasons,
        }

    return {
        "suggested_type": "unknown",
        "confidence": 0.3,
        "reasons": ["Could not confidently infer the outcome type"],
    }
