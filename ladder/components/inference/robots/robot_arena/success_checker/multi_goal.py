import collections
import dataclasses
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.core.instructions import ON, Placement, subtask
from ladder.components.inference.robots.robot_arena.success_checker.checker import (
    BatchedSuccessChecker,
    privileged_payloads,
)
from ladder.components.inference.robots.robot_arena.success_checker.matching import match_instances
from ladder.components.inference.robots.robot_arena.success_checker.types import TaskSpec


def subtask_instruction(goal: TaskSpec) -> str:
    """The single-goal instruction a hierarchical system commands; the plan cache uses this wording."""
    return subtask(Placement(goal.target_ref, goal.spatial_relation or ON, goal.destination_ref))


class MultiGoalSuccessChecker:
    """Scores a multi-goal task: one ``BatchedSuccessChecker`` per goal, combined per env.

    Emits per env ``done``, ``success_status`` (``"<achieved>/<n>"``), and the ``oracle`` block the
    privileged HLC reads -- ``subtasks`` for its decomposer, ``achieved`` for its advancer.

    ``ordered`` keeps the conjunction but additionally requires the goals' FIRST satisfactions to
    happen in list order. ``milestones`` drops it, for a task that moves one object more than once:
    its earlier legs are false by the time the last is true, so a leg counts only once every leg
    before it has been seen satisfied -- which is also what stops a policy that skips the
    intermediate from being credited for it. Goals naming the same object and destination are
    otherwise a duplicate group, bound to that many distinct instances.

    The env rebuilds this checker on reset, so all tracking state is per-episode.
    """

    def __init__(
        self,
        goals: List[TaskSpec],
        ordered: bool = False,
        milestones: bool = False,
        **checker_kwargs,
    ):
        if ordered and milestones:
            raise ValueError("ordered and milestones are two ways to score a sequence; pick one")
        self._goals = goals
        self._ordered = ordered
        self._milestones = milestones
        self._subtasks = [subtask_instruction(goal) for goal in goals]
        # A duplicate group holds one checker per instance, so `_checkers` can outnumber `goals`.
        self._checkers, self._groups = self._build(goals, checker_kwargs, milestones)
        # first_seen[env][goal] = check index of the goal's first satisfaction (None = never).
        self._first_seen: List[List[Optional[int]]] = []
        self._check_index = 0

    @staticmethod
    def _build(goals, checker_kwargs, milestones: bool):
        """One checker per goal, except that a duplicate group binds its goals to distinct instances
        -- under ``milestones`` there are none, since a repeated pair is a revisit, not a copy."""
        objects = [cfg for _, cfg in checker_kwargs["object_entities"]]
        counts = collections.Counter((goal.target_ref, goal.destination_ref) for goal in goals)
        instances = (
            {}
            if milestones
            else {key: match_instances(key[0], objects, count) for key, count in counts.items() if count > 1}
        )
        used: Dict[Tuple[str, Optional[str]], int] = collections.Counter()
        checkers, groups = [], collections.defaultdict(list)
        for index, goal in enumerate(goals):
            key = (goal.target_ref, goal.destination_ref)
            spec = goal
            if key in instances:
                # By the instance's real name, so both goals of a two-spoon group cannot resolve to
                # whichever spoon the matcher likes best.
                spec = dataclasses.replace(goal, target_ref=objects[instances[key][used[key]]].name)
                used[key] += 1
            groups[key].append(index)
            checkers.append(
                BatchedSuccessChecker(task_instruction="", task_spec_override=spec, **checker_kwargs)
            )
        return checkers, dict(groups)

    def _order_respected(self, env_i: int) -> bool:
        firsts = self._first_seen[env_i]
        satisfied = [step for step in firsts if step is not None]
        return firsts == sorted(satisfied) + [None] * (len(firsts) - len(satisfied))

    def _advance_milestones(self, env_i: int, holding: List[bool]) -> int:
        """How many leading legs have been satisfied IN ORDER; never decreases. Only the leg the task
        is on can be recorded, or reaching the final destination first would score as a relay."""
        firsts = self._first_seen[env_i]
        progress = next((g for g, seen in enumerate(firsts) if seen is None), len(firsts))
        while progress < len(firsts) and holding[progress]:
            firsts[progress] = self._check_index
            progress += 1
        return progress

    def _credited(self, holding: List[bool]) -> List[bool]:
        """Per-goal credit, by COUNT within a duplicate group: which instance landed is not something
        the instruction distinguishes."""
        credited = list(holding)
        for indices in self._groups.values():
            if len(indices) == 1:
                continue
            landed = sum(holding[i] for i in indices)
            for rank, index in enumerate(indices):
                credited[index] = rank < landed
        return credited

    def privileged_state(
        self, world_to_base, achieved: List[List[bool]], held_indices=None
    ) -> List[Dict[str, object]]:
        """Ground-truth payload for the scripted policy: one condition per goal, base frame.

        Each goal states itself through its own single-goal checker, so a composition task and an
        ordinary scene task publish the same contract. ``achieved`` is the per-env, per-goal credit
        this checker just published, so the policy advances on exactly the goals the score counts
        rather than forming a second opinion.
        """
        goals = [
            checker.privileged_goal(world_to_base, [achieved[env][index] for env in range(len(achieved))])
            for index, checker in enumerate(self._checkers[: len(self._goals)])
        ]
        return privileged_payloads(self._checkers[0], world_to_base, goals, len(achieved), held_indices)

    def check_all(self) -> List[Dict[str, object]]:
        per_goal = [checker.check_all() for checker in self._checkers]  # [goal][env]
        n_goals = len(self._goals)
        n_envs = len(per_goal[0])
        if not self._first_seen:
            self._first_seen = [[None] * n_goals for _ in range(n_envs)]
        results: List[Dict[str, object]] = []
        for env_i in range(n_envs):
            holding = [bool(per_goal[g][env_i]["done"]) for g in range(n_goals)]
            if self._milestones:
                done_count = self._advance_milestones(env_i, holding)
                achieved = [g < done_count for g in range(n_goals)]
                all_done = done_count == n_goals
                task_type = "milestones"
            else:
                achieved = self._credited(holding)
                for g, is_done in enumerate(achieved):
                    if is_done and self._first_seen[env_i][g] is None:
                        self._first_seen[env_i][g] = self._check_index
                done_count = sum(achieved)
                all_done = done_count == n_goals and (not self._ordered or self._order_respected(env_i))
                task_type = "ordered" if self._ordered else "conjunction"
            results.append(
                {
                    "done": all_done,
                    "success_status": f"{done_count}/{n_goals}",
                    "task_type": task_type,
                    # A sequence-scored task says so, and the oracle arm then walks the legs in order.
                    "oracle": {
                        "subtasks": self._subtasks,
                        "achieved": achieved,
                        "ordered": bool(self._ordered or self._milestones),
                        # A milestones task repeats a leg with the SAME object, never a copy.
                        "revisits": bool(self._milestones),
                    },
                }
            )
        self._check_index += 1
        return results
