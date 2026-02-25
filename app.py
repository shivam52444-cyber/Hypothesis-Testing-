import numpy as np
import pandas as pd
import streamlit as st

from inference import run_inference

st.set_page_config(page_title="Inference Lab", layout="wide")
st.title("Inference Lab")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to start.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    st.success("Dataset loaded.")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

cols = df.columns.tolist()
st.markdown("---")

mode = st.selectbox("Choose analysis type", ["frequentist", "bayesian", "survival"])
st.markdown("---")

group_col = None
outcome_col = None
covariate_cols = None
time_col = None
is_censored_col = None

if mode in ["frequentist", "bayesian"]:
    group_col = st.selectbox("Group column (Treatment / Control)", cols)
    outcome_col = st.selectbox("Outcome column", cols)
    covariate_cols = st.multiselect(
        "Covariates (optional)",
        [c for c in cols if c not in [group_col, outcome_col]],
    )

if mode == "survival":
    time_col = st.selectbox("Time column", cols)
    is_censored_col = st.selectbox("Censoring indicator column (True = censored)", cols)
    group_col = st.selectbox("Group column", cols)
    covariate_cols = st.multiselect(
        "Covariates (optional)",
        [c for c in cols if c not in [time_col, is_censored_col, group_col]],
    )

st.markdown("---")

mde = st.number_input(
    "MDE (Minimum Detectable Effect) - set 0 to skip power/alpha-beta optimization",
    min_value=0.0,
    value=0.0,
    step=0.01,
)

cost_fp = st.number_input("Cost of False Positive (Type I error)", min_value=0.0, value=1.0)
cost_fn = st.number_input("Cost of False Negative (Type II error)", min_value=0.0, value=1.0)
two_sided = st.radio("Test sidedness", ["Two-sided", "One-sided"]) == "Two-sided"

want_decision = False
c_fp = 1.0
c_fn = 1.0
c_continue = 0.0

if mode == "bayesian":
    want_decision = st.checkbox("Also perform Bayesian decision-making?")
    if want_decision:
        st.markdown("### Decision Costs")
        c_fp = st.number_input("Cost FP (ship but bad)", min_value=0.0, value=1.0)
        c_fn = st.number_input("Cost FN (don't ship but good)", min_value=0.0, value=1.0)
        c_continue = st.number_input("Cost Continue", min_value=0.0, value=0.0)

st.markdown("---")

if st.button("Run Analysis"):
    try:
        with st.spinner("Running analysis..."):
            res = run_inference(
                df=df,
                mode=mode,
                outcome_col=outcome_col,
                group_col=group_col,
                covariate_cols=covariate_cols if covariate_cols else None,
                want_decision=want_decision,
                c_fp=c_fp,
                c_fn=c_fn,
                c_continue=c_continue,
                mde=mde if mde > 0 else None,
                time_col=time_col,
                is_censored_col=is_censored_col,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                two_sided=two_sided,
            )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

    st.success("Analysis completed!")
    st.markdown("---")

    if res["mode"] == "frequentist":
        st.subheader("Frequentist Result")
        st.write("**Test used:**", res["test_used"])
        st.write("**p-value:**", res["p_value"])

        if res.get("power_warning"):
            st.warning(res["power_warning"])
            if res.get("required_sample_size") is not None:
                st.info(
                    f"Estimated required total sample size ~= "
                    f"**{int(np.ceil(res['required_sample_size']))}**"
                )

        st.subheader("Diagnostics")
        st.json(res["diagnostics"])

    elif res["mode"] == "bayesian":
        st.subheader("Bayesian Result")
        st.write(res["summary"])

        if "decision" in res:
            st.subheader("Decision")
            st.json(res["decision"])

    elif res["mode"] == "survival":
        st.subheader("Survival / Censored Data Result")
        st.json(res["result"])
