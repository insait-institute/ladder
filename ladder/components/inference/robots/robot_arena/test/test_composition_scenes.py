import collections
import unittest
from typing import Dict, List, Set

from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
from ladder.components.inference.robots.robot_arena.composition.tasks import (
    CompositionTask,
    compose_scene_data,
)
from ladder.components.inference.robots.robot_arena.scene_loader import ObjectConfig, load_scene
from ladder.components.inference.robots.robot_arena.success_checker import (
    TaskSpec,
    TaskType,
    match_instances,
    match_objects,
)

_DATA_ROOT = "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data"


def _named(task: CompositionTask, objects: List[ObjectConfig]) -> Set[int]:
    named: Set[int] = set()
    counts = collections.Counter((goal.target_ref, goal.destination_ref) for goal in task.goals)
    for goal in task.goals:
        key = (goal.target_ref, goal.destination_ref)
        if counts[key] > 1 and not task.milestones:
            named.update(match_instances(goal.target_ref, objects, counts[key]))
        resolved = match_objects(goal, objects)
        assert resolved is not None, f"{task.scene}: goal {goal} matches no object"
        named.add(resolved.target_index)
        if resolved.destination_index is not None:
            named.add(resolved.destination_index)
    if task.start_on is not None:
        named.update(match_instances(task.start_on, objects, 1))
    return named


class TestNoUndeclaredDistractors(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenes: Dict[str, List[ObjectConfig]] = {
            name: compose_scene_data(load_scene(_DATA_ROOT, task.scene), task, _DATA_ROOT).objects
            for name, task in COMPOSITION_TASKS.items()
        }

    def test_every_object_on_the_table_is_a_goal_or_a_declared_distractor(self):
        for name, task in COMPOSITION_TASKS.items():
            objects = self.scenes[name]
            named = _named(task, objects)
            for ref in task.excluded:
                resolved = match_objects(TaskSpec(TaskType.PICK_AND_PLACE, ref), objects)
                self.assertIsNotNone(resolved, f"{name}: excluded {ref!r} matches no object")
                named.add(resolved.target_index)
            leftover = [obj.name for i, obj in enumerate(objects) if i not in named]
            with self.subTest(task=name):
                self.assertEqual(
                    leftover,
                    [],
                    f"{name} leaves {leftover} on the table unnamed: make each a goal, name it in "
                    f"`excluded`, or list it in `remove_objects`",
                )

    def test_only_the_exclusion_families_declare_a_distractor(self):
        for name, task in COMPOSITION_TASKS.items():
            with self.subTest(task=name):
                self.assertEqual(
                    bool(task.excluded),
                    "except" in name,
                    f"{name}: excluded={task.excluded} does not match its family",
                )


if __name__ == "__main__":
    unittest.main()
