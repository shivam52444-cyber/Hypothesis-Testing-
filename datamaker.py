import numpy as np
import pandas as pd

def prepare_test_input(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    test_type: str,
):
    """
    Prepare data in the right shape for different statistical tests.

    test_type options:
      - "two_group_continuous"
      - "k_group_continuous"
      - "proportion"
      - "categorical"
      - "count"
      - "regression"
    """

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    # Drop rows with missing outcome or group (you already handle missing upstream,
    # but this is a safety net)
    d = df[[outcome_col, group_col]].dropna().copy()

    groups = d[group_col].unique()
    k = len(groups)

    if k < 2:
        raise ValueError("Need at least 2 groups.")

    # Map groups to labels 0,1,2,...
    group_map = {g: i for i, g in enumerate(groups)}
    d["_g"] = d[group_col].map(group_map)

    y = d[outcome_col].values

    # -----------------------------
    # 1) Two-group continuous tests
    # -----------------------------
    if test_type == "two_group_continuous":
        if k != 2:
            raise ValueError("two_group_continuous requires exactly 2 groups.")

        y0 = y[d["_g"] == 0].astype(float)
        y1 = y[d["_g"] == 1].astype(float)

        return {
            "y0": y0,
            "y1": y1,
            "groups": groups,
            "n0": len(y0),
            "n1": len(y1),
        }

    # -----------------------------
    # 2) K-group continuous tests
    # -----------------------------
    if test_type == "k_group_continuous":
        arrays = []
        for gi in range(k):
            arrays.append(y[d["_g"] == gi].astype(float))

        return {
            "arrays": arrays,
            "groups": groups,
            "ns": [len(a) for a in arrays],
        }

    # -----------------------------
    # 3) Proportion / binary outcome
    # -----------------------------
    if test_type == "proportion":
        # Expect y to be 0/1
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("For proportion test, outcome must be binary (0/1).")

        if k != 2:
            raise ValueError("Proportion test here assumes 2 groups.")

        y0 = y[d["_g"] == 0]
        y1 = y[d["_g"] == 1]

        return {
            "success0": int(np.sum(y0)),
            "n0": len(y0),
            "success1": int(np.sum(y1)),
            "n1": len(y1),
            "p0": float(np.mean(y0)),
            "p1": float(np.mean(y1)),
        }

    # -----------------------------
    # 4) Categorical outcome (contingency table)
    # -----------------------------
    if test_type == "categorical":
        table = pd.crosstab(d[outcome_col], d[group_col])
        return {
            "table": table,
            "observed": table.values,
            "groups": table.columns.tolist(),
            "categories": table.index.tolist(),
        }

    # -----------------------------
    # 5) Count outcome
    # -----------------------------
    if test_type == "count":
        if k != 2:
            raise ValueError("Count test here assumes 2 groups (extendable).")

        c0 = y[d["_g"] == 0].astype(float)
        c1 = y[d["_g"] == 1].astype(float)

        return {
            "c0": c0,
            "c1": c1,
            "sum0": float(np.sum(c0)),
            "sum1": float(np.sum(c1)),
            "n0": len(c0),
            "n1": len(c1),
            "mean0": float(np.mean(c0)),
            "mean1": float(np.mean(c1)),
        }

    # -----------------------------
    # 6) Regression / CUPED / RA
    # -----------------------------
    if test_type == "regression":
        # Here we just return y and treatment indicator; covariates handled elsewhere
        if k != 2:
            raise ValueError("Regression A/B test here assumes 2 groups.")

        T = d["_g"].values  # 0/1 indicator
        y = d[outcome_col].astype(float).values

        return {
            "y": y,
            "T": T,
            "n": len(y),
        }

    raise ValueError(f"Unknown test_type '{test_type}'.")
