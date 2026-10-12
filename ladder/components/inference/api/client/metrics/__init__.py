from ladder.components.inference.api.client.metrics import (  # noqa: F401
    control,
    failure,
    objects,
    outcome,
    plan,
    timing,
)
from ladder.components.inference.api.client.metrics.registry import (
    METRICS,
    Fraction,
    Metric,
    aggregate,
    merge,
    metric,
)

# The metric modules are imported for their side effect: each `@metric` registers into METRICS at
# import, so importing this package is what makes the registry complete. They are split by what they
# need to know rather than by benchmark -- `outcome` reads only the env's own success and per-goal
# state, `failure` reads that plus proprioception to say why a failed episode failed, `control`
# reads the controller's events, `objects` reads the env's ground-truth object trace to say what
# became of each grasp, `plan` parses the subtask language and is the only one whose answers are
# heuristic (see its module comment), `timing` says where the step budget went.

__all__ = ["METRICS", "Fraction", "Metric", "aggregate", "merge", "metric"]
