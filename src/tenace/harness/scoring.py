from __future__ import annotations

from bisect import bisect_left
import math

__all__ = ["score_diff_to_imps", "wbf_continuous_vp"]

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


def wbf_continuous_vp(imp_diff: int | float, num_boards: int) -> tuple[float, float]:
    """Official WBF 2013 continuous 20–0 Victory Point (VP) scale.

    Returns ``(team_a_vp, team_b_vp)`` rounded to 2 decimal places summing to ``20.00``.
    Positive ``imp_diff`` favors Team A; zero yields ``(10.0, 10.0)``.
    """
    if (
        isinstance(imp_diff, bool)
        or not isinstance(imp_diff, (int, float))
        or not math.isfinite(imp_diff)
    ):
        raise ValueError(f"imp_diff must be a finite number, got {imp_diff!r}")
    if (
        isinstance(num_boards, bool)
        or not isinstance(num_boards, int)
        or num_boards < 1
    ):
        raise ValueError(f"num_boards must be an integer >= 1, got {num_boards!r}")

    if imp_diff == 0:
        return (10.0, 10.0)

    tau = (math.sqrt(5.0) - 1.0) / 2.0
    cap = 15.0 * math.sqrt(num_boards)
    x = abs(float(imp_diff))
    if x >= cap:
        win_vp = 20.0
    else:
        win_vp = 10.0 + 10.0 * (1.0 - tau ** (3.0 * x / cap)) / (1.0 - tau**3)
    win_vp = round(win_vp, 2)
    lose_vp = round(20.0 - win_vp, 2)
    return (win_vp, lose_vp) if imp_diff > 0 else (lose_vp, win_vp)
