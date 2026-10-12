"""CPU-only validation of success checker parse + match for ALL available scenes.

Verifies that every scene's instruction can be parsed into a TaskSpec and that
object references resolve correctly against the actual scene object list — without
requiring Genesis or a GPU.

Also validates structural invariants on TASK_SPEC_OVERRIDES.

Run via: bazel test //ladder/components/inference/robots/robot_arena:test_scene_manifest
"""

import dataclasses
import unittest
from typing import Dict, List, Optional

import numpy as np

from ladder.components.inference.robots.robot_arena.scene_loader import (
    ObjectConfig,
    list_available_scenes,
    load_scene,
)
from ladder.components.inference.robots.robot_arena.success_checker import (
    TASK_SPEC_OVERRIDES,
    SuccessThresholds,
    TaskType,
    match_objects,
    parse_instruction,
)

_DATA_ROOT = "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data"


class SceneInfo:
    """Lightweight container for scene data needed by tests."""

    def __init__(
        self,
        instruction: str,
        objects: List[ObjectConfig],
        load_error: Optional[str] = None,
    ) -> None:
        self.instruction = instruction
        self.objects = objects
        self.load_error = load_error


class TestSceneManifest(unittest.TestCase):
    """Validate parse_instruction + match_objects for every available scene."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenes: Dict[str, SceneInfo] = {}
        scene_names = list_available_scenes(_DATA_ROOT)
        for scene_name in scene_names:
            try:
                scene_data = load_scene(_DATA_ROOT, scene_name)
                cls.scenes[scene_name] = SceneInfo(
                    instruction=scene_data.task_instruction,
                    objects=scene_data.objects,
                )
            except Exception as exc:  # noqa: BLE001
                # Record the error but don't fail setup — individual tests will skip.
                cls.scenes[scene_name] = SceneInfo(
                    instruction="",
                    objects=[],
                    load_error=str(exc),
                )

    def test_all_scenes_load_successfully(self) -> None:
        """Every scene on disk must load without errors."""
        errors: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                errors.append(f"  {scene_name}: {info.load_error}")
        if errors:
            self.fail("Scene load failures:\n" + "\n".join(errors))

    def test_all_scenes_parse_instruction(self) -> None:
        """Every scene's instruction must parse into a non-None TaskSpec."""
        failures: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                continue
            task_spec = parse_instruction(info.instruction, scene_name)
            if task_spec is None:
                failures.append(f"  {scene_name}: parse_instruction returned None for '{info.instruction}'")
        if failures:
            self.fail("Instruction parse failures:\n" + "\n".join(failures))

    def test_all_scenes_match_target(self) -> None:
        """Every scene's parsed TaskSpec must match a target object."""
        failures: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                continue
            task_spec = parse_instruction(info.instruction, scene_name)
            if task_spec is None:
                continue  # Already caught by test_all_scenes_parse_instruction.

            resolved = match_objects(task_spec, info.objects)
            if resolved is None:
                obj_names = [o.name for o in info.objects]
                failures.append(
                    f"  {scene_name}: target '{task_spec.target_ref}' not matched. " f"Objects: {obj_names}"
                )
        if failures:
            self.fail("Target match failures:\n" + "\n".join(failures))

    def test_task_type_constraints(self) -> None:
        """Validate per-task-type structural constraints on resolved tasks."""
        failures: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                continue
            task_spec = parse_instruction(info.instruction, scene_name)
            if task_spec is None:
                continue

            resolved = match_objects(task_spec, info.objects)
            if resolved is None:
                continue  # Already caught above.

            task_type = resolved.task_spec.task_type

            # PICK_AND_PLACE / STACK require a destination.
            if task_type in (TaskType.PICK_AND_PLACE, TaskType.STACK):
                if resolved.destination_index is None:
                    failures.append(
                        f"  {scene_name}: {task_type.value} has no destination. "
                        f"dest_ref='{task_spec.destination_ref}', "
                        f"objects={[o.name for o in info.objects]}"
                    )

            # BETWEEN requires two reference objects.
            if task_type == TaskType.BETWEEN:
                if resolved.destination_index is None:
                    failures.append(f"  {scene_name}: BETWEEN missing reference 1")
                if resolved.between_anchor_index is None:
                    failures.append(f"  {scene_name}: BETWEEN missing reference 2")

            # MOVE_TO_REGION / PUSH require a table region.
            if task_type in (TaskType.MOVE_TO_REGION, TaskType.PUSH):
                if task_spec.table_region is None:
                    failures.append(f"  {scene_name}: {task_type.value} has no table_region")

            # MOVE_RELATIVE requires a spatial direction.
            if task_type == TaskType.MOVE_RELATIVE:
                if task_spec.spatial_relation is None:
                    failures.append(f"  {scene_name}: MOVE_RELATIVE has no spatial_relation")
                # Needs either a reference object or a table region.
                if resolved.destination_index is None and task_spec.table_region is None:
                    failures.append(f"  {scene_name}: MOVE_RELATIVE has no reference object or table region")

        if failures:
            self.fail("Task constraint failures:\n" + "\n".join(failures))

    def test_target_is_not_fixed(self) -> None:
        """The matched target should not be a fixed (non-movable) object."""
        warnings: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                continue
            task_spec = parse_instruction(info.instruction, scene_name)
            if task_spec is None:
                continue
            resolved = match_objects(task_spec, info.objects)
            if resolved is None:
                continue

            target_obj = info.objects[resolved.target_index]
            if target_obj.fixed:
                warnings.append(f"  {scene_name}: target '{target_obj.name}' is classified as fixed")
        if warnings:
            self.fail("Target is fixed (non-movable):\n" + "\n".join(warnings))

    def test_no_self_referencing(self) -> None:
        """Target and destination must not resolve to the same object."""
        failures: List[str] = []
        for scene_name, info in self.scenes.items():
            if info.load_error is not None:
                continue
            task_spec = parse_instruction(info.instruction, scene_name)
            if task_spec is None:
                continue
            resolved = match_objects(task_spec, info.objects)
            if resolved is None:
                continue

            if resolved.destination_index is not None:
                if resolved.target_index == resolved.destination_index:
                    failures.append(
                        f"  {scene_name}: target and destination resolve to same object "
                        f"(index={resolved.target_index}, name='{info.objects[resolved.target_index].name}')"
                    )
        if failures:
            self.fail("Self-referencing failures:\n" + "\n".join(failures))


class TestTaskSpecOverridesInvariants(unittest.TestCase):
    """Validate structural properties of TASK_SPEC_OVERRIDES."""

    def test_overrides_have_valid_task_types(self) -> None:
        """All overrides must have valid TaskType values."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            self.assertIsInstance(spec.task_type, TaskType, f"{scene_name}: task_type is not a TaskType enum")

    def test_overrides_have_target_ref(self) -> None:
        """All overrides must have a non-empty target_ref."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            self.assertTrue(spec.target_ref and spec.target_ref.strip(), f"{scene_name}: target_ref is empty")

    def test_overrides_pick_place_have_destination(self) -> None:
        """PICK_AND_PLACE and STACK overrides must have destination_ref."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            if spec.task_type in (TaskType.PICK_AND_PLACE, TaskType.STACK):
                self.assertTrue(
                    spec.destination_ref and spec.destination_ref.strip(),
                    f"{scene_name}: {spec.task_type.value} override has no destination_ref",
                )

    def test_overrides_between_has_two_refs(self) -> None:
        """BETWEEN overrides must have both destination_ref and between_anchor_ref."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            if spec.task_type == TaskType.BETWEEN:
                self.assertTrue(
                    spec.destination_ref, f"{scene_name}: BETWEEN override missing destination_ref"
                )
                self.assertTrue(
                    spec.between_anchor_ref, f"{scene_name}: BETWEEN override missing between_anchor_ref"
                )

    def test_overrides_region_tasks_have_region(self) -> None:
        """MOVE_TO_REGION and PUSH overrides must have table_region."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            if spec.task_type in (TaskType.MOVE_TO_REGION, TaskType.PUSH):
                self.assertTrue(
                    spec.table_region, f"{scene_name}: {spec.task_type.value} override has no table_region"
                )

    def test_overrides_relative_has_direction(self) -> None:
        """MOVE_RELATIVE overrides must have spatial_relation."""
        for scene_name, spec in TASK_SPEC_OVERRIDES.items():
            if spec.task_type == TaskType.MOVE_RELATIVE:
                self.assertTrue(
                    spec.spatial_relation, f"{scene_name}: MOVE_RELATIVE override has no spatial_relation"
                )


class TestSuccessThresholdsInvariants(unittest.TestCase):
    """Validate that default thresholds are physically reasonable."""

    def setUp(self) -> None:
        self.thresholds = SuccessThresholds()

    def test_all_thresholds_are_finite(self) -> None:
        for field in dataclasses.fields(self.thresholds):
            value = getattr(self.thresholds, field.name)
            self.assertTrue(np.isfinite(value), f"{field.name}={value} is not finite")

    def test_height_range_is_valid(self) -> None:
        self.assertLess(
            self.thresholds.on_top_height_min_m,
            self.thresholds.on_top_height_max_m,
            "on_top_height_min_m must be < on_top_height_max_m",
        )

    def test_thresholds_are_physically_reasonable(self) -> None:
        """No threshold should exceed 0.5m (half a meter) for tabletop tasks."""
        for field in dataclasses.fields(self.thresholds):
            value = getattr(self.thresholds, field.name)
            self.assertLess(abs(value), 0.5, f"{field.name}={value} exceeds 0.5m — likely a bug")

    def test_positive_distance_thresholds(self) -> None:
        """Distance thresholds must be positive."""
        positive_fields = [
            "on_top_xy_margin_m",
            "inside_z_tolerance_m",
            "pickup_height_m",
            "relative_margin_m",
            "displacement_min_m",
            "between_tolerance_m",
            "reach_distance_m",
            "region_padding_m",
            "relative_max_distance_m",
            "region_max_distance_m",
        ]
        for field in positive_fields:
            value = getattr(self.thresholds, field)
            self.assertGreater(value, 0.0, f"{field}={value} must be > 0")


if __name__ == "__main__":
    unittest.main()
