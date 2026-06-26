"""Small demo: run with `python -m pseudomedian`."""

from __future__ import annotations

import numpy as np

from pseudomedian import pseudomedian_ci


def main() -> None:
    rng = np.random.default_rng(1)
    sample = rng.lognormal(mean=0.0, sigma=1.65, size=200)

    result = pseudomedian_ci(sample, n_resamples=2000, rng=rng)
    print(f"n          = {result.n}")
    print(f"method     = {result.method}")
    print(f"pseudomedian = {result.estimate:.4f}")
    print(
        f"{int(result.confidence_level * 100)}% CI   = "
        f"[{result.lower:.4f}, {result.upper:.4f}]"
    )


if __name__ == "__main__":
    main()
