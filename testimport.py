# =========================
# Core scientific stack
# =========================
import numpy as np
import pandas as pd

# =========================
# SciPy: classical tests
# =========================
from scipy import stats
from scipy.stats import (
    ttest_ind,          # Welch t-test (use equal_var=False)
    mannwhitneyu,       # Mann–Whitney U
    wilcoxon,           # Wilcoxon signed-rank
    kruskal,            # Kruskal–Wallis
    chi2_contingency,   # Chi-square test
    fisher_exact,       # Fisher exact test (2x2)
    permutation_test,  # Permutation tests (new SciPy)
)

# =========================
# Statsmodels: proportions, regression, ANOVA
# =========================
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.proportion import proportions_ztest   # Proportion z-test
from statsmodels.stats.anova import anova_lm                  # ANOVA tables
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind  # alt t-test

# =========================
# Pingouin: Welch ANOVA, pairwise tests, effect sizes
# =========================
import pingouin as pg
# pg.welch_anova(...)
# pg.pairwise_gameshowell(...)
# pg.pairwise_ttests(...)

# =========================
# scikit-posthocs: Post-hoc tests
# =========================
import scikit_posthocs as sp
# sp.posthoc_dunn(...)
# sp.posthoc_conover(...)
# sp.posthoc_gameshowell(...)

# =========================
# scikit-learn: Multiple Imputation, utilities
# =========================
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# =========================
# Convenience aliases (optional)
# =========================

# Welch t-test shortcut
def welch_ttest(x, y):
    return ttest_ind(x, y, equal_var=False)

# Welch ANOVA shortcut (via pingouin)
def welch_anova(df, dv, between):
    return pg.welch_anova(data=df, dv=dv, between=between)

# Games-Howell post-hoc (for Welch ANOVA)
def posthoc_games_howell(df, dv, between):
    return pg.pairwise_gameshowell(data=df, dv=dv, between=between)

# Dunn post-hoc (for Kruskal–Wallis)
def posthoc_dunn(df, val_col, group_col, p_adjust="bonferroni"):
    return sp.posthoc_dunn(df, val_col=val_col, group_col=group_col, p_adjust=p_adjust)

# =========================
# End of imports
# =========================