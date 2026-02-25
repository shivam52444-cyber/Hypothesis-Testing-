import pandas as pd
import numpy as np
import lifelines
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter

def compute_rmst(time, event, tau):
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)
    return kmf.restricted_mean_survival_time(t=tau)

def survival_test_auto(
    dataset: pd.DataFrame,
    time_col: str,
    is_censored_col: str,   # True = censored, False = event observed
    group_col: str,         # treatment / control (0/1 or binary)
    covariate_cols: list | None = None,
    rmst_tau: float | None = None,
    alpha: float = 0.05,
):
    """
    Automatically chooses between:
      - Cox PH (with robust SE) if PH assumption is OK
      - RMST difference if PH assumption is violated

    This is analogous to:
      - Student t-test if variances OK
      - Welch t-test if variances not OK
    but for censored time-to-event data.

    Returns a dict with method used and results.
    """

    df = dataset.copy()

    # lifelines expects: event_col = 1 if event happened, 0 if censored
    df["_event"] = (~df[is_censored_col]).astype(int)

    if covariate_cols is None:
        covariate_cols = []

    # Encode group to a numeric treatment indicator for Cox models.
    group_levels = list(df[group_col].dropna().unique())
    if len(group_levels) != 2:
        raise ValueError("group_col must contain exactly 2 groups for survival comparison.")
    group_mapping = {group_levels[0]: 0, group_levels[1]: 1}
    df["_group_numeric"] = df[group_col].map(group_mapping)

    cols = [time_col, "_event", "_group_numeric"] + covariate_cols
    df_model = df[cols].dropna()

    # ---------------------------
    # 1) Fit Cox with robust SE
    # ---------------------------
    cph = CoxPHFitter()
    cph.fit(
        df_model,
        duration_col=time_col,
        event_col="_event",
        robust=True,   # <-- THIS is the "Welch-like" part: heteroskedasticity-robust SE
    )

    # ---------------------------
    # 2) Check PH assumption
    # ---------------------------
    # Schoenfeld residual-based test
    ph_test = proportional_hazard_test(cph, df_model, time_transform="rank")
    ph_pvalues = ph_test.summary["p"]

    # Focus especially on the group column
    ph_p_group = ph_pvalues.get("_group_numeric", np.nan)

    ph_ok = (not np.isnan(ph_p_group)) and (ph_p_group > alpha)

    # ---------------------------
    # 3) If PH is OK -> use Cox
    # ---------------------------
    if ph_ok:
        summary = cph.summary

        row = summary.loc["_group_numeric"]

        result = {
            "method": "cox_ph_robust",
            "ph_test_p_value": float(ph_p_group),
            "group_labels": {"control_like": str(group_levels[0]), "treatment_like": str(group_levels[1])},
            "hazard_ratio": float(row["exp(coef)"]),
            "coef": float(row["coef"]),
            "p_value": float(row["p"]),
            "ci_lower": float(row["exp(coef) lower 95%"]),
            "ci_upper": float(row["exp(coef) upper 95%"]),
            "model_summary": summary,
            "cox_model": cph,
            "interpretation": "PH assumption looks OK. Using Cox PH with robust (sandwich) standard errors."
        }

        return result

    # ---------------------------
    # 4) If PH is violated -> use RMST
    # ---------------------------
    # Choose tau if not provided: e.g., 90th percentile of observed times
    if rmst_tau is None:
        rmst_tau = np.percentile(df_model[time_col], 90)

    # Split groups without assuming numeric labels (0/1)
    g0_label, g1_label = group_levels[0], group_levels[1]
    g0 = df[df[group_col] == g0_label][[time_col, "_event"]].dropna()
    g1 = df[df[group_col] == g1_label][[time_col, "_event"]].dropna()

    rmst_0 = compute_rmst(g0[time_col], g0["_event"], tau=rmst_tau)
    rmst_1 = compute_rmst(g1[time_col], g1["_event"], tau=rmst_tau)

    diff = rmst_1 - rmst_0

    result = {
        "method": "rmst",
        "ph_test_p_value": float(ph_p_group),
        "group_labels": {"control_like": str(g0_label), "treatment_like": str(g1_label)},
        "rmst_tau": float(rmst_tau),
        "rmst_control": float(rmst_0),
        "rmst_treatment": float(rmst_1),
        "rmst_difference": float(diff),
        "interpretation": (
            "PH assumption violated. Switched to RMST, which is more stable under "
            "non-proportional hazards and heavy heterogeneity."
        ),
    }

    return result
