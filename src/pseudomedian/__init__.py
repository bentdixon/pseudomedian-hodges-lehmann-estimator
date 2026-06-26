"""Pseudomedian (Hodges-Lehmann estimator) with bootstrap confidence intervals.

Replicates the approach in Frank Harrell's post:
https://www.fharrell.com/post/aci/
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap

__version__ = "0.1.0"

__all__ = ["PseudomedianResult", "pseudomedian", "pseudomedian_ci"]


@dataclass
class PseudomedianResult:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    n: int


def pseudomedian(x, axis=-1, max_exact_n=20000):
    """Median of all Walsh averages (x_i + x_j) / 2 for i <= j.

    The i == j pairs are included, matching the blog's definition
    ("including pairing an observation with itself"). For a sample of
    size n this is the median of n(n + 1) / 2 values.

    Supports the ND input scipy's bootstrap passes to a vectorized
    statistic: the pseudomedian is computed along `axis`.

    This is an exact O(n^2) computation. For n > max_exact_n it raises,
    since the Walsh set no longer fits comfortably in memory; subsample
    or use a dedicated large-n algorithm in that regime.
    """
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    if n > max_exact_n:
        raise ValueError(
            f"n={n} exceeds max_exact_n={max_exact_n}; the exact Walsh-average "
            "computation is O(n^2) in memory. Subsample first or raise the cap."
        )

    # Build the Walsh averages row by row (sorted x_i + x_j for j >= i).
    # O(n^2) values total but only O(n) memory per row, avoiding an
    # explicit O(n^2) index allocation.
    xs = np.sort(x, axis=-1)
    rows = [0.5 * (xs[..., i:] + xs[..., i, np.newaxis]) for i in range(n)]
    walsh = np.concatenate(rows, axis=-1)
    return np.median(walsh, axis=-1)


def _resolve_method(method, n):
    if method != "auto":
        return method
    # Harrell's pMedian default: BCa for n < 150, percentile otherwise.
    return "bca" if n < 150 else "percentile"


def pseudomedian_ci(
    x,
    confidence_level=0.95,
    method="auto",
    n_resamples=9999,
    rng=None,
):
    """Pseudomedian point estimate with a bootstrap confidence interval.

    method:
        "auto"       -> BCa if n < 150 else percentile (pMedian default)
        "bca"        -> bias-corrected and accelerated
        "percentile" -> nonparametric percentile
        "basic"      -> reverse-percentile
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 non-NaN observations")

    resolved = _resolve_method(method, n)
    estimate = float(pseudomedian(x))

    res = bootstrap(
        (x,),
        pseudomedian,
        vectorized=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method=resolved,
        rng=rng,
    )

    return PseudomedianResult(
        estimate=estimate,
        lower=float(res.confidence_interval.low),
        upper=float(res.confidence_interval.high),
        confidence_level=confidence_level,
        method=resolved,
        n=n,
    )
