import pandas as pd
import numpy as np

def handle_missing_group(
    df: pd.DataFrame,
    group_col: str,
    missing_tolerance: float = 0.05,
    equality_tol: float = 0.05,
    interactive: bool = True,
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    group_col : str
        Column name for group (treatment/control)
    missing_tolerance : float
        Max allowed fraction of missing in group_col (e.g., 0.05 = 5%)
    equality_tol : float
        Tolerance for saying per-group missingness is 'approximately equal'
        (e.g., 0.05 = 5 percentage points)
    interactive : bool
        If True, ask user whether to drop rows. If False, just stop or proceed.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame if dropped, otherwise original DataFrame.

    Raises
    ------
    ValueError
        If conditions fail and function must stop.
    """

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    n = len(df)
    missing_mask = df[group_col].isna()
    missing_frac = missing_mask.mean()

    print(f"[Info] Total rows: {n}")
    print(f"[Info] Missing in '{group_col}': {missing_frac:.2%}")

    # 1) If too much missing -> stop
    if missing_frac > missing_tolerance:
        raise ValueError(
            f"[Stop] Missingness in '{group_col}' is {missing_frac:.2%}, "
            f"which is greater than tolerance {missing_tolerance:.2%}. "
            "This can seriously bias inference. Please fix data upstream."
        )

    # If no missing at all, just return df
    if missing_frac == 0:
        print("[OK] No missing in group column. Proceeding without changes.")
        return df

    # 2) Check per-group missingness pattern
    # We look at missingness of group_col vs other groups is tricky because group is missing.
    # Instead, we check whether missingness correlates with other observed groups by:
    # - Create a temporary indicator: is_group_missing
    # - Compare missing rate across observed groups (excluding NaN group itself)

    temp = df.copy()
    temp["_is_group_missing"] = temp[group_col].isna()

    observed = temp[~temp[group_col].isna()]

    # If only one group level exists, we can't compare
    if observed[group_col].nunique() < 2:
        print("[Warn] Only one observed group level found. Cannot assess MCAR across groups.")
        print("[Warn] Dropping may still be risky.")
        if interactive:
            ans = input("Do you still want to drop rows with missing group? (yes/no): ").strip().lower()
            if ans == "yes":
                print("[Action] Dropping rows with missing group.")
                return df.loc[~missing_mask].copy()
            else:
                raise ValueError("[Stop] User chose not to drop missing group rows.")
        else:
            raise ValueError("[Stop] Non-interactive mode: cannot safely decide to drop.")
    
    # Compute missing rate per observed group (proxy check)
    rates = (
        temp
        .assign(_group_obs=temp[group_col].fillna("__MISSING__"))
        .groupby("_group_obs")["_is_group_missing"]
        .mean()
    )

    print("[Info] Missingness rate by group label (including __MISSING__):")
    print(rates)

    # Exclude the "__MISSING__" pseudo-group for comparison
    rates_no_missing = rates.drop("__MISSING__", errors="ignore")

    max_rate = rates_no_missing.max()
    min_rate = rates_no_missing.min()
    diff = max_rate - min_rate

    print(f"[Info] Max-min missingness difference across groups: {diff:.2%}")

    # 3) Decide if approximately equal -> MCAR-ish
    if diff <= equality_tol:
        print(
            "[OK] Missingness looks approximately equal across groups.\n"
            "This is consistent with MCAR-ish behavior. Dropping rows may introduce minimal bias."
        )
        if interactive:
            ans = input("Do you want to drop rows with missing group? (yes/no): ").strip().lower()
            if ans == "yes":
                print("[Action] Dropping rows with missing group.")
                return df.loc[~missing_mask].copy()
            else:
                raise ValueError("[Stop] User chose not to drop missing group rows.")
        else:
            print("[Non-interactive] Proceeding without dropping.")
            return df
    else:
        # 4) Not equal -> warn and stop
        raise ValueError(
            "[Stop] Missingness is not approximately equal across groups.\n"
            "This suggests missingness may bias your inference.\n"
            "Please review data collection or use a model-based approach."
        )