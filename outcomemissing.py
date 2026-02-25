import pandas as pd
import numpy as np

def handle_missing_outcome_with_mi(
    df: pd.DataFrame,
    outcome_col: str,
    m: int = 5,
    missing_tolerance: float = 0.2,
    random_state: int = 42,
):
    """
    Handle missing outcome using Multiple Imputation (MI).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    outcome_col : str
        Name of the outcome column.
    m : int
        Number of multiple imputations (datasets) to generate.
    missing_tolerance : float
        If fraction of missing outcome exceeds this, raise a warning (still proceed).
    random_state : int
        Base random seed for reproducibility.

    Returns
    -------
    imputed_datasets : list[pd.DataFrame]
        List of m DataFrames with imputed outcome values.
        If no missing in outcome, returns [df.copy()].
    info : dict
        Diagnostics info (missing fraction, method used).
    """

    if outcome_col not in df.columns:
        raise ValueError(f"Column '{outcome_col}' not found in DataFrame.")

    n = len(df)
    missing_mask = df[outcome_col].isna()
    missing_frac = missing_mask.mean()

    info = {
        "n_rows": n,
        "missing_fraction_outcome": float(missing_frac),
        "method": None,
    }

    print(f"[Info] Total rows: {n}")
    print(f"[Info] Missing in outcome '{outcome_col}': {missing_frac:.2%}")

    # If no missing outcome, just return the original dataset
    if missing_frac == 0:
        print("[OK] No missing outcome. No imputation needed.")
        info["method"] = "none"
        return [df.copy()], info

    # Warn if too much missing
    if missing_frac > missing_tolerance:
        print(
            f"[Warn] Missing outcome fraction ({missing_frac:.2%}) exceeds tolerance "
            f"({missing_tolerance:.2%}). Inference may be sensitive to missingness assumptions."
        )

    # --- Multiple Imputation using IterativeImputer (MICE-like) ---
    # We will:
    # 1) Use all numeric columns (including outcome) as the imputation model matrix
    # 2) Generate m different imputations by changing the random seed
    # 3) Replace only the outcome column in each returned dataset

    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for Multiple Imputation. "
            "Please install it with `pip install scikit-learn`."
        ) from e

    # Select numeric columns for imputation model
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if outcome_col not in numeric_cols:
        raise ValueError(
            f"Outcome column '{outcome_col}' is not numeric. "
            "This simple MI implementation expects a numeric outcome."
        )

    # Prepare matrix for imputation
    X = df[numeric_cols].copy()

    imputed_datasets = []

    print(f"[Info] Running Multiple Imputation with m = {m} ...")

    for k in range(m):
        seed = random_state + k

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            sample_posterior=True,   # IMPORTANT: makes it stochastic for MI
            random_state=seed,
        )

        X_imp = imputer.fit_transform(X)

        X_imp_df = pd.DataFrame(X_imp, columns=numeric_cols, index=df.index)

        # Build the completed dataset:
        df_imp = df.copy()
        df_imp[outcome_col] = X_imp_df[outcome_col]

        imputed_datasets.append(df_imp)

    info["method"] = "multiple_imputation"
    info["m"] = m

    print("[OK] Multiple Imputation completed.")
    print("[Note] You should run your analysis on each dataset and pool results (Rubin's rules).")

    return imputed_datasets, info