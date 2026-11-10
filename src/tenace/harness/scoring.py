from __future__ import annotations

from bisect import bisect_left
import math

__all__ = ["score_diff_to_imps"]

# Official WBF IMP upper bounds for 0..23 IMPs (>= 4000 maps to 24 IMPs).
_WBF_IMP_UPPER_BOUNDS: tuple[int, ...] = (
    10,
    40,
    80,
    120,
    160,
    210,
    260,
    310,
    360,
    420,
    490,
    590,
    740,
    890,
    1090,
    1290,
    1490,
    1740,
    1990,
    2240,
    2490,
    2990,
    3490,
    3990,
)


def score_diff_to_imps(diff: int | float) -> int:
    """Convert raw duplicate score difference (Open NS - Closed NS) to signed WBF IMPs."""
    if (
        isinstance(diff, bool)
        or not isinstance(diff, (int, float))
        or not math.isfinite(diff)
    ):
        raise ValueError(f"Score difference must be a finite number, got {diff!r}")

    if diff == 0:
        return 0

    sign = 1 if diff > 0 else -1
    magnitude = abs(float(diff))
    imps = bisect_left(_WBF_IMP_UPPER_BOUNDS, magnitude)
    return sign * imps
