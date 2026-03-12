"""
Statistical tests for CAM lead analysis.

- Sign test: proportion of lead > 0 significantly > 0.5?
- Permutation test: under null (no lead), distribution of mean(lead); p-value = proportion of permuted mean >= observed mean.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats


def sign_test_lead(
    lead_series: pd.Series,
    alternative: str = "greater",
) -> Dict[str, float]:
    """
    Sign test: test if proportion of positive leads is > 0.5 (or two-sided).

    Args:
        lead_series: lead values (numeric; NaN excluded)
        alternative: 'greater' (mean lead > 0), 'two-sided', or 'less'

    Returns:
        stat, p_value, n_positive, n_total, proportion_positive
    """
    leads = lead_series.dropna()
    if len(leads) == 0:
        return {
            "stat": np.nan,
            "p_value": np.nan,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "n_total": 0,
            "proportion_positive": np.nan,
        }
    n_positive = int((leads > 0).sum())
    n_negative = int((leads < 0).sum())
    n_zero = int((leads == 0).sum())
    n_total = len(leads)
    proportion_positive = n_positive / n_total if n_total > 0 else np.nan
    # Binomial test: under H0 p=0.5, number of successes = n_positive, n = n_total
    stat = n_positive
    try:
        res = stats.binomtest(n_positive, n_total, p=0.5, alternative=alternative)
        p_value = float(res.pvalue)
    except AttributeError:
        p_value = float(stats.binom_test(n_positive, n_total, p=0.5, alternative=alternative))
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "n_total": n_total,
        "proportion_positive": float(proportion_positive),
    }


def permutation_test_lead(
    lead_series: pd.Series,
    n_permutations: int = 10000,
    alternative: str = "greater",
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Permutation test: under H0 (lead is symmetric around 0), shuffle signs of lead.
    Observed stat = mean(lead). Permuted: multiply lead by random ±1, then mean.
    p_value = proportion of permuted means >= observed mean (for alternative='greater').

    Args:
        lead_series: lead values
        n_permutations: number of permutation samples
        alternative: 'greater' (mean lead > 0), 'two-sided', or 'less'
        random_state: for reproducibility

    Returns:
        observed_mean, p_value, permuted_means (optional), ci_low, ci_high (empirical from perm dist)
    """
    rng = np.random.default_rng(random_state)
    leads = lead_series.dropna().values
    if len(leads) == 0:
        return {
            "observed_mean": np.nan,
            "p_value": np.nan,
            "permuted_mean_std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_total": 0,
        }
    observed_mean = float(np.mean(leads))
    n = len(leads)
    permuted_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted_means[i] = np.mean(leads * signs)
    if alternative == "greater":
        p_value = float(np.mean(permuted_means >= observed_mean))
    elif alternative == "less":
        p_value = float(np.mean(permuted_means <= observed_mean))
    else:
        p_value = float(np.mean(np.abs(permuted_means) >= np.abs(observed_mean)))
    ci_low = float(np.percentile(permuted_means, 2.5))
    ci_high = float(np.percentile(permuted_means, 97.5))
    return {
        "observed_mean": observed_mean,
        "p_value": p_value,
        "permuted_mean_std": float(np.std(permuted_means)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_total": n,
    }


def aggregate_lead_stats(
    lead_df: pd.DataFrame,
    lead_col: str = "lead",
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, any]:
    """
    Aggregate: mean lead, std, sign test, permutation test, counts (lead/coincident/lag).
    """
    if lead_df is None or len(lead_df) == 0 or lead_col not in lead_df.columns:
        return {
            "mean_lead": np.nan,
            "std_lead": np.nan,
            "n_lead": 0,
            "n_coincident": 0,
            "n_lag": 0,
            "n_cam_missing": 0,
            "n_total": 0,
            "sign_test": {},
            "permutation_test": {},
        }
    leads = lead_df[lead_col].dropna()
    n_lead = int((leads > 0).sum())
    n_coincident = int((leads == 0).sum())
    n_lag = int((leads < 0).sum())
    n_cam_missing = int(lead_df["lead"].isna().sum())  # alignment == 'cam_missing'
    n_total = len(lead_df)

    mean_lead = float(leads.mean()) if len(leads) > 0 else np.nan
    std_lead = float(leads.std()) if len(leads) > 1 else np.nan

    sign = sign_test_lead(leads, alternative="greater") if len(leads) > 0 else {}
    perm = permutation_test_lead(leads, n_permutations=n_permutations, random_state=random_state, alternative="greater") if len(leads) > 0 else {}

    return {
        "mean_lead": mean_lead,
        "std_lead": std_lead,
        "n_lead": n_lead,
        "n_coincident": n_coincident,
        "n_lag": n_lag,
        "n_cam_missing": n_cam_missing,
        "n_total": n_total,
        "sign_test": sign,
        "permutation_test": perm,
    }
