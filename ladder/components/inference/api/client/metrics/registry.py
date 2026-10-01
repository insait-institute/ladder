from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ladder.components.inference.api.client.episode import Episode

# Every metric returns (numerator, denominator), so aggregation is always sum/sum and the reported
# value has the same "k/n" shape the success rate has always had — one reader handles them all.
# Returning None means the metric does not apply to this episode (no composition goals, no
# hierarchical controller); it is then skipped, NOT counted as zero, so an unrelated benchmark
# never drags an average down.
Fraction = Tuple[float, float]
Metric = Callable[[Episode], Optional[Fraction]]

METRICS: Dict[str, Metric] = {}


def metric(function: Metric) -> Metric:
    """Register ``function`` under its own name. Adding a metric is this decorator and nothing else."""
    METRICS[function.__name__] = function
    return function


def _format(numerator: float, denominator: float) -> str:
    return f"{numerator:g}/{denominator:g}"


def aggregate(episodes: Iterable[Episode]) -> Dict[str, str]:
    """Every applicable metric over ``episodes`` as "numerator/denominator" strings.

    Reads whatever is in :data:`METRICS`, which the modules in this package fill at import. Import
    the package rather than this module, or the registry is empty and every episode scores nothing.
    """
    totals: Dict[str, List[float]] = {}
    for episode in episodes:
        for name, function in METRICS.items():
            value = function(episode)
            if value is None:
                continue
            total = totals.setdefault(name, [0.0, 0.0])
            total[0] += value[0]
            total[1] += value[1]
    return {name: _format(*total) for name, total in totals.items()}


def merge(*maps: Dict[str, str]) -> Dict[str, str]:
    """Sum aggregate maps — parallel lanes appending to one results.yaml, or a re-run of a cell."""
    totals: Dict[str, List[float]] = {}
    for values in maps:
        for name, value in values.items():
            numerator, denominator = (float(part) for part in str(value).split("/"))
            total = totals.setdefault(name, [0.0, 0.0])
            total[0] += numerator
            total[1] += denominator
    return {name: _format(*total) for name, total in totals.items()}
