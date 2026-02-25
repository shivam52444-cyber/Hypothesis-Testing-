from __future__ import annotations

import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_run_inputs(
    df: pd.DataFrame,
    mode: str,
    outcome_col: str | None,
    group_col: str | None,
    covariate_cols: list[str] | None,
    time_col: str | None,
    is_censored_col: str | None,
) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input df must be a non-empty pandas DataFrame.")

    valid_modes = {"frequentist", "bayesian", "survival"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    if mode in {"frequentist", "bayesian"}:
        if outcome_col is None or group_col is None:
            raise ValueError("outcome_col and group_col are required for frequentist/bayesian modes.")
        _require_columns(df, [outcome_col, group_col])

    if mode == "survival":
        if time_col is None or is_censored_col is None or group_col is None:
            raise ValueError("time_col, is_censored_col, and group_col are required for survival mode.")
        _require_columns(df, [time_col, is_censored_col, group_col])

    if covariate_cols:
        _require_columns(df, covariate_cols)
