# power_utils.py
import numpy as np
from scipy.stats import norm

def required_sample_size_for_mde(
    mde: float,
    se_per_unit: float,
    alpha: float,
    power_target: float = 0.8,
    two_sided: bool = True,
):
    """
    Compute required total sample size given:
      - mde: minimum detectable effect
      - se_per_unit: standard error per sqrt(n) unit (i.e., SE scales as se_per_unit / sqrt(n))
      - alpha: significance level
      - power_target: desired power (e.g., 0.8)
      - two_sided: test sidedness

    Returns:
      n_required (float): required total sample size (for two groups combined, assuming equal split)
    """

    if two_sided:
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_power = norm.ppf(power_target)

    # From: MDE = (z_alpha + z_power) * SE
    # And SE = se_per_unit / sqrt(n)
    # => mde = (z_alpha + z_power) * se_per_unit / sqrt(n)
    # => sqrt(n) = (z_alpha + z_power) * se_per_unit / mde
    # => n = ((z_alpha + z_power) * se_per_unit / mde)^2

    n_required = ((z_alpha + z_power) * se_per_unit / mde) ** 2

    return float(n_required)