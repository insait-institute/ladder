"""Binary task success conditions for RobotArena scenes.

Provides deterministic, geometric success evaluation at each simulation step,
complementing the post-hoc VLM-based GVL evaluator. The pipeline:

1. Parse the scene's text instruction into a structured TaskSpec (task type + object roles)
2. Match object references against scene entities using fuzzy name matching
3. Evaluate geometric predicates on live Genesis entity positions each step

Follows the same ``metadata["done"]`` convention as SimplerEnv so the existing
RecorderWrapper and SimplerEpisodeRunner pipelines work without changes.
"""

from ladder.components.inference.robots.robot_arena.success_checker.checker import BatchedSuccessChecker
from ladder.components.inference.robots.robot_arena.success_checker.matching import (
    match_instances,
    match_objects,
)
from ladder.components.inference.robots.robot_arena.success_checker.multi_goal import (
    MultiGoalSuccessChecker,
    subtask_instruction,
)
from ladder.components.inference.robots.robot_arena.success_checker.parsing import (
    TASK_SPEC_OVERRIDES,
    parse_instruction,
)
from ladder.components.inference.robots.robot_arena.success_checker.predicates import (
    is_between,
    is_in_table_region,
    is_inside,
    is_near,
    is_on_top_of,
    is_picked_up,
    is_relative_to,
)
from ladder.components.inference.robots.robot_arena.success_checker.types import (
    CheckResult,
    ResolvedTask,
    SimEntity,
    SuccessThresholds,
    TaskSpec,
    TaskType,
)

__all__ = [
    "match_instances",
    "CheckResult",
    "SimEntity",
    "BatchedSuccessChecker",
    "MultiGoalSuccessChecker",
    "subtask_instruction",
    "SuccessThresholds",
    "TaskSpec",
    "TaskType",
    "ResolvedTask",
    "TASK_SPEC_OVERRIDES",
    "parse_instruction",
    "match_objects",
    "is_on_top_of",
    "is_inside",
    "is_picked_up",
    "is_relative_to",
    "is_in_table_region",
    "is_between",
    "is_near",
]
