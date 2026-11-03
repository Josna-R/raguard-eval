"""Direction-aware regression gating against a saved baseline aggregate.

A single eval run's numbers don't tell you whether a change made things
worse -- you need something to compare against. This module saves/loads
a baseline aggregate scorecard and compares a new run's aggregate to it,
flagging any metric that moved the WRONG direction by more than a
tolerance. This is the actual point of RAGuard as a CI gate, not just a
scoring library: `raguard check` exits non-zero when a regression is
found, so it can fail a pull request before a change ships.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

# Metrics where a LOWER value is better (rates of bad outcomes). Every
# other metric in the aggregate dict is assumed higher-is-better.
_LOWER_IS_BETTER = {"hallucination_flag_rate", "pii_fabrication_rate"}

# Default allowed drift before a metric's movement counts as a regression.
DEFAULT_TOLERANCE = 0.05


@dataclass
class Regression:
    """One metric that regressed between baseline and current run."""

    metric: str
    baseline_value: float
    current_value: float
    delta: float  # current - baseline, raw (sign depends on direction)
    direction: str  # "higher_is_better" or "lower_is_better"


def save_baseline(path: Union[str, Path], aggregate_dict: Dict) -> None:
    """Persist an aggregate dict (from harness.aggregate) as a JSON baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(aggregate_dict, f, indent=2, sort_keys=True)


def load_baseline(path: Union[str, Path]) -> Dict:
    """Load a previously-saved baseline aggregate dict."""
    with open(path, "r") as f:
        return json.load(f)


def compare_to_baseline(
    current: Dict,
    baseline: Dict,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Regression]:
    """Compare a current aggregate dict to a baseline, direction-aware.

    For each numeric metric present in both dicts:
      - higher-is-better metrics (everything not in _LOWER_IS_BETTER)
        regress when current < baseline - tolerance.
      - lower-is-better metrics (_LOWER_IS_BETTER) regress when
        current > baseline + tolerance.

    This means an improvement on a lower-is-better metric (e.g.
    hallucination_flag_rate dropping) never triggers a false regression
    even though its raw delta is negative, and a regression on a
    lower-is-better metric (that rate going up) is always caught even
    though its raw delta is positive.

    `n_cases` is ignored -- it's metadata, not a quality metric.

    Returns the list of Regressions found, most-affected order follows
    dict iteration order of `current`.
    """
    regressions: List[Regression] = []

    for metric, current_value in current.items():
        if metric == "n_cases":
            continue
        if metric not in baseline:
            continue
        baseline_value = baseline[metric]
        if not isinstance(current_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            continue

        delta = current_value - baseline_value

        if metric in _LOWER_IS_BETTER:
            direction = "lower_is_better"
            regressed = current_value > baseline_value + tolerance
        else:
            direction = "higher_is_better"
            regressed = current_value < baseline_value - tolerance

        if regressed:
            regressions.append(
                Regression(
                    metric=metric,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    delta=delta,
                    direction=direction,
                )
            )

    return regressions
