# bayesiantest.py
import numpy as np
import pandas as pd

import pymc as pm
import arviz as az


def bayesian_ab_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    prior: dict | None = None,
):
    """
    Bayesian A/B test for binary outcomes using Beta-Binomial model.

    Assumptions:
      - outcome_col is binary: 0/1
      - group_col has exactly 2 groups: control and treatment (any labels)

    Returns:
      - trace: PyMC InferenceData
      - summary: ArviZ summary DataFrame
    """

    # -----------------------------
    # 1) Validate input
    # -----------------------------
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in dataframe")
    if outcome_col not in df.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not in dataframe")

    d = df[[group_col, outcome_col]].dropna().copy()

    # Ensure binary
    unique_vals = set(d[outcome_col].unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError("Bayesian A/B test here expects binary outcome (0/1).")

    groups = d[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Bayesian A/B test requires exactly 2 groups.")

    g0, g1 = groups[0], groups[1]

    y0 = d.loc[d[group_col] == g0, outcome_col].astype(int).values
    y1 = d.loc[d[group_col] == g1, outcome_col].astype(int).values

    n0 = len(y0)
    n1 = len(y1)

    s0 = int(y0.sum())
    s1 = int(y1.sum())

    # -----------------------------
    # 2) Priors
    # -----------------------------
    # Default: weakly-informative Beta(1,1)
    if prior is None:
        a0 = b0 = 1.0
        a1 = b1 = 1.0
    else:
        # Expect something like:
        # prior = {"control": (a0, b0), "treatment": (a1, b1)}
        a0, b0 = prior.get("control", (1.0, 1.0))
        a1, b1 = prior.get("treatment", (1.0, 1.0))

    # -----------------------------
    # 3) Build model
    # -----------------------------
    with pm.Model() as model:
        # Conversion rates
        p0 = pm.Beta("p0", alpha=a0, beta=b0)
        p1 = pm.Beta("p1", alpha=a1, beta=b1)

        # Likelihoods (Binomial)
        obs0 = pm.Binomial("obs0", n=n0, p=p0, observed=s0)
        obs1 = pm.Binomial("obs1", n=n1, p=p1, observed=s1)

        # Derived quantity: lift / delta
        delta = pm.Deterministic("delta", p1 - p0)

        # -----------------------------
        # 4) Sample (IMPORTANT: cores=1 for Streamlit/Windows)
        # -----------------------------
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=1,            # 🔴 CRITICAL: avoid multiprocessing hang on Windows/Streamlit
            progressbar=False,
            target_accept=0.9,
            random_seed=42,
        )

    # -----------------------------
    # 5) Summarize
    # -----------------------------
    summary = az.summary(trace, var_names=["p0", "p1", "delta"])

    return trace, summary