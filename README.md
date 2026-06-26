# Pseudomedian with bootstrap confidence intervals

A small wrapper computing the pseudomedian (Hodges-Lehmann location
estimator) and a bootstrap confidence interval for it, following Frank
Harrell's post [*Measures of Central Tendency for an Asymmetric
Distribution, and Confidence Intervals*](https://www.fharrell.com/post/aci/).

The pseudomedian sits between the mean and median, more robust than the
mean and more efficient than the median
under normality. Its sampling
distribution is close to symmetric even when the population is heavily
skewed, so accurate confidence intervals are available where the usual
central-limit-theorem interval for the mean fails. The blog finds the BCa
and percentile bootstrap intervals to be the most reliable.


## Method

**Point estimate.** The pseudomedian is the median of all Walsh averages:

    (x_i + x_j) / 2   for   1 <= i <= j <= n

i.e. the midpoint of every pair of observations, including each
observation paired with itself. For *n* observations that is n(n+1)/2
values.

**Confidence interval.** Bootstrap via `scipy.stats.bootstrap`. The
`method="auto"` default uses Harrell's `Hmisc::pMedian`: BCa for
n < 150, nonparametric percentile for n >= 150. You can force `"bca"`,
`"percentile"`, or `"basic"`.

## Installation

Install directly from the repository:

```bash
uv add git+https://github.com/bentdixon/pseudomedian-hodges-lehmann-estimator
# or
pip install git+https://github.com/bentdixon/pseudomedian-hodges-lehmann-estimator
```

## Usage

```python
import numpy as np
from pseudomedian import pseudomedian, pseudomedian_ci

x = np.random.default_rng(1).lognormal(0.0, 1.65, size=200)

pseudomedian(x)          # point estimate only

r = pseudomedian_ci(x)   # estimate + CI
r.estimate, r.lower, r.upper, r.method
```

A small demo is available via:

```bash
python -m pseudomedian
```

## Notes and limits

- The exact computation is O(n²) in the Walsh-average set. `pseudomedian`
  raises above `max_exact_n` (default 20000); subsample or raise the cap
  for larger n. Bootstrap cost scales with `n_resamples` × that O(n²),
  so large n with the default 9999 resamples is slow.
- The classic *exact* Hodges-Lehmann interval (inverting the Wilcoxon
  signed-rank test) assumes population symmetry. It is intentionally not
  used here; the bootstrap intervals do not require that assumption, which
  is the point of the blog in the asymmetric case.
