# inference.py
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import chi2_contingency, combine_pvalues, kruskal, mannwhitneyu, ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# ---- Frequentist pipeline imports (your existing stack) ----
from groupmissing import handle_missing_group
from outcomemissing import handle_missing_outcome_with_mi
from covariatemissing import handle_missing_covariates_with_indicator
from outcome_detector import detect_outcome_type
from outlierdetector import detect_outliers_and_heavy_tails
from se import compute_se_treatment_effect
from alphabeta import optimize_alpha_beta
from datamaker import prepare_test_input
from powerutils import required_sample_size_for_mde
from validation import validate_run_inputs

# ---- Bayesian imports (your files) ----
from bayesiantest import bayesian_ab_test
from bayesian_decision import decision_from_posterior

# ---- Censored / Survival imports (your file) ----
from censoring import survival_test_auto


# ---------------------------
# Helpers
# ---------------------------
def _normalize_outcome_type(raw_outcome_type):
    # Accept dict or string
    if isinstance(raw_outcome_type, dict):
        ot_raw = raw_outcome_type.get("suggested_type", "unknown")
    else:
        ot_raw = raw_outcome_type

    ot = str(ot_raw).lower()

    if ot in ["continuous", "float", "numeric", "real", "double"]:
        return "continuous"
    if ot in ["binary", "bool", "boolean", "0/1"]:
        return "binary"
    if ot in ["categorical", "category", "nominal", "factor", "object", "string"]:
        return "categorical"
    if ot in ["count", "counts", "integer", "int", "discrete"]:
        return "count"
    if ot in ["unknown"]:
        raise ValueError("Outcome type is 'unknown'. Please specify explicitly.")

    raise ValueError(f"Unknown outcome type from detector: {raw_outcome_type}")


# ---------------------------
# Main Orchestrator
# ---------------------------
def run_inference(
    df: pd.DataFrame,
    mode: str,  # "frequentist" | "bayesian" | "survival"
    outcome_col: str | None = None,
    group_col: str | None = None,
    covariate_cols: list[str] | None = None,
    # Bayesian decision params
    want_decision: bool = False,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
    c_continue: float = 0.0,
    mde: float | None = None,
    # Survival params
    time_col: str | None = None,
    is_censored_col: str | None = None,
    # Frequentist business params
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    two_sided: bool = True,
):
    """
    mode:
      - "frequentist"
      - "bayesian"
      - "survival"
    """

    validate_run_inputs(
        df=df,
        mode=mode,
        outcome_col=outcome_col,
        group_col=group_col,
        covariate_cols=covariate_cols,
        time_col=time_col,
        is_censored_col=is_censored_col,
    )

    if mode == "bayesian":
        # ---------------------------
        # Bayesian A/B test (binary)
        # ---------------------------
        trace, summary = bayesian_ab_test(
            df=df,
            group_col=group_col,
            outcome_col=outcome_col,
            prior=None,
        )

        # Extract posterior samples of delta
        delta_samples = trace.posterior["delta"].values.flatten()

        result = {
            "mode": "bayesian",
            "trace": trace,
            "summary": summary,
            "delta_samples": delta_samples,
        }

        # Optional decision layer
        if want_decision:
            if mde is None:
                raise ValueError("MDE is required for Bayesian decision making.")

            decision = decision_from_posterior(
                delta_samples=delta_samples,
                c_fp=c_fp,
                c_fn=c_fn,
                c_continue=c_continue,
                mde=mde,
            )
            result["decision"] = decision

        return result

    elif mode == "survival":
        # ---------------------------
        # Survival / Censored data
        # ---------------------------
        if time_col is None or is_censored_col is None or group_col is None:
            raise ValueError("time_col, is_censored_col, and group_col are required for survival analysis.")

        res = survival_test_auto(
            dataset=df,
            time_col=time_col,
            is_censored_col=is_censored_col,
            group_col=group_col,
            covariate_cols=covariate_cols,
            alpha=0.05,
        )

        return {
            "mode": "survival",
            "result": res,
        }

    elif mode == "frequentist":
        # ---------------------------
        # Your existing Frequentist pipeline
        # ---------------------------
        diagnostics = {}

        # 1) Handle missing group
        df_clean = handle_missing_group(
            df=df,
            group_col=group_col,
            missing_tolerance=0.05,
            equality_tol=0.05,
            interactive=False,
        )

        # 2) Handle missing covariates
        if covariate_cols:
            df_clean, cov_info = handle_missing_covariates_with_indicator(
                df_clean, covariate_cols, fill_value=0.0
            )
            diagnostics["covariate_missing"] = cov_info

        # 3) Handle missing outcome (MI)
        imputed_dfs, outcome_miss_info = handle_missing_outcome_with_mi(
            df_clean, outcome_col=outcome_col, m=5
        )
        diagnostics["outcome_missing"] = outcome_miss_info

        df_use = imputed_dfs[0]

        # 4) Detect outcome type
        raw_outcome_type = detect_outcome_type(df_use[outcome_col])
        outcome_type = _normalize_outcome_type(raw_outcome_type)
        diagnostics["outcome_type_raw"] = raw_outcome_type
        diagnostics["outcome_type_norm"] = outcome_type

        # 5) Outlier check
        use_nonparametric = False
        if outcome_type == "continuous":
            outlier_info = detect_outliers_and_heavy_tails(
                df_use, outcome_col=outcome_col, group_col=group_col
            )
            diagnostics["outlier_check"] = outlier_info
            use_nonparametric = bool(outlier_info.get("use_nonparametric", False))

        # 6) Choose test
        groups = df_use[group_col].dropna().unique()
        k = len(groups)

        if outcome_type == "continuous":
            if k == 2:
                test_name = "mann_whitney" if use_nonparametric else "welch_t"
            else:
                test_name = "kruskal" if use_nonparametric else "welch_anova"
        elif outcome_type == "binary":
            test_name = "proportion_z"
        elif outcome_type == "categorical":
            test_name = "chi_square"
        elif outcome_type == "count":
            test_name = "count_compare"
        else:
            raise ValueError(f"Unhandled outcome type: {outcome_type}")

        diagnostics["test_chosen"] = test_name

        # 7) SE
        se_info = None
        if k == 2 and outcome_type in ["continuous", "binary"]:
            if outcome_type == "continuous":
                se_info = compute_se_treatment_effect(df_use, outcome_col, group_col, method="mean_diff")
            else:
                se_info = compute_se_treatment_effect(df_use, outcome_col, group_col, method="proportion")
        diagnostics["se_info"] = se_info

        # 9) Alpha/Beta optimization
        alpha_beta = None
        if mde is not None and se_info is not None and mde > 0:
            alpha_beta = optimize_alpha_beta(
                mde=mde,
                se=se_info["se"],
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                two_sided=two_sided,
            )
        diagnostics["alpha_beta"] = alpha_beta

        # 9.5) Power warning + required N
        power_warning = None
        required_n = None
        TARGET_POWER = 0.80

        if alpha_beta is not None and se_info is not None and mde is not None and mde > 0:
            if "power_opt" in alpha_beta:
                power_opt = float(alpha_beta["power_opt"])
            elif "beta_opt" in alpha_beta:
                power_opt = 1.0 - float(alpha_beta["beta_opt"])
            else:
                power_opt = np.nan

            if not np.isnan(power_opt) and power_opt < TARGET_POWER:
                power_warning = (
                    f"⚠️ Underpowered for MDE with current data (power ≈ {power_opt:.2f} < {TARGET_POWER:.2f})."
                )
                n_current = len(df_use.dropna(subset=[outcome_col, group_col]))
                se_per_unit = se_info["se"] * (n_current ** 0.5)
                alpha_used = alpha_beta["alpha_opt"]

                required_n = required_sample_size_for_mde(
                    mde=mde,
                    se_per_unit=se_per_unit,
                    alpha=alpha_used,
                    power_target=TARGET_POWER,
                    two_sided=two_sided,
                )

        # 10) Run test on each imputed dataset and combine p-values if needed
        def _run_single_test(df_candidate: pd.DataFrame):
            if test_name in ["welch_t", "mann_whitney"]:
                prep_local = prepare_test_input(df_candidate, outcome_col, group_col, "two_group_continuous")
            elif test_name in ["welch_anova", "kruskal"]:
                prep_local = prepare_test_input(df_candidate, outcome_col, group_col, "k_group_continuous")
            elif test_name == "proportion_z":
                prep_local = prepare_test_input(df_candidate, outcome_col, group_col, "proportion")
            elif test_name == "chi_square":
                prep_local = prepare_test_input(df_candidate, outcome_col, group_col, "categorical")
            elif test_name == "count_compare":
                prep_local = prepare_test_input(df_candidate, outcome_col, group_col, "count")
            else:
                raise ValueError("No data prep path.")

            if test_name == "welch_t":
                stat_local, pval_local = ttest_ind(prep_local["y0"], prep_local["y1"], equal_var=False)
            elif test_name == "mann_whitney":
                stat_local, pval_local = mannwhitneyu(prep_local["y0"], prep_local["y1"], alternative="two-sided")
            elif test_name == "welch_anova":
                res_local = pg.welch_anova(data=df_candidate, dv=outcome_col, between=group_col)
                stat_local = res_local
                pval_local = float(res_local["p-unc"].iloc[0])
            elif test_name == "kruskal":
                stat_local, pval_local = kruskal(*prep_local["arrays"])
            elif test_name == "proportion_z":
                stat_local, pval_local = proportions_ztest(
                    count=[prep_local["success1"], prep_local["success0"]],
                    nobs=[prep_local["n1"], prep_local["n0"]],
                )
            elif test_name == "chi_square":
                chi2_local, pval_local, _, _ = chi2_contingency(prep_local["observed"])
                stat_local = chi2_local
            elif test_name == "count_compare":
                stat_local, pval_local = mannwhitneyu(prep_local["c0"], prep_local["c1"], alternative="two-sided")
            else:
                raise ValueError("Test not implemented.")

            return stat_local, float(pval_local)

        stats_list = []
        pvals = []
        for df_imp in imputed_dfs:
            stat_i, pval_i = _run_single_test(df_imp)
            stats_list.append(stat_i)
            pvals.append(pval_i)

        if len(pvals) == 1:
            stat = stats_list[0]
            pval = pvals[0]
        else:
            _, pval = combine_pvalues(pvals, method="fisher")
            stat = stats_list[0]
            diagnostics["mi_pooling"] = {
                "n_imputations": len(pvals),
                "method": "fisher_combined_pvalue",
                "pvalues_per_imputation": pvals,
            }

        return {
            "mode": "frequentist",
            "test_used": test_name,
            "statistic": stat,
            "p_value": float(pval),
            "alpha_beta": alpha_beta,
            "se_info": se_info,
            "power_warning": power_warning,
            "required_sample_size": required_n,
            "diagnostics": diagnostics,
        }

    else:
        raise ValueError("mode must be one of: 'frequentist', 'bayesian', 'survival'")
