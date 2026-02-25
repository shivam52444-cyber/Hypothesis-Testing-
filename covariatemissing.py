import pandas as pd
import numpy as np

def handle_missing_covariates_with_indicator(
    df: pd.DataFrame,
    covariate_cols,
    fill_value: float = 0.0,
    suffix: str = "_missing",
    verbose: bool = True,
):
    """
    Handle missing covariates using the Missing-Indicator method.

    For each covariate X:
      - Create indicator I_X = 1 if X is missing else 0
      - Fill missing X with `fill_value`
      - Keep both X_filled and I_X for downstream regression / CUPED / RA

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    covariate_cols : list[str]
        List of covariate column names to process
    fill_value : float
        Value to fill missing entries (default 0.0)
    suffix : str
        Suffix for indicator columns (default "_missing")
    verbose : bool
        If True, print diagnostics

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with indicator columns added and covariates filled
    info : dict
        Diagnostics info about missingness handled
    """

    df_out = df.copy()
    info = {
        "processed_covariates": [],
        "missing_fractions": {},
        "fill_value": fill_value,
        "indicator_suffix": suffix,
    }

    for col in covariate_cols:
        if col not in df_out.columns:
            raise ValueError(f"Covariate column '{col}' not found in DataFrame.")

        missing_mask = df_out[col].isna()
        missing_frac = missing_mask.mean()

        info["missing_fractions"][col] = float(missing_frac)

        if verbose:
            print(f"[Info] Covariate '{col}': missing = {missing_frac:.2%}")

        if missing_frac == 0:
            if verbose:
                print(f"[OK] No missing in '{col}'. No indicator created.")
            continue

        # Create indicator
        ind_col = f"{col}{suffix}"
        if ind_col in df_out.columns:
            raise ValueError(f"Indicator column '{ind_col}' already exists. Name collision.")

        df_out[ind_col] = missing_mask.astype(int)

        # Fill missing values
        df_out[col] = df_out[col].fillna(fill_value)

        info["processed_covariates"].append(col)

        if verbose:
            print(f"[Action] Created indicator '{ind_col}' and filled missing in '{col}' with {fill_value}.")

    if verbose and len(info["processed_covariates"]) == 0:
        print("[OK] No covariates had missing values. No changes made.")

    return df_out, info