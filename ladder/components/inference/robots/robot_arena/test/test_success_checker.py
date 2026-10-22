"""Tests for the binary task success checker.

All tests are CPU-only. Genesis entities are mocked with simple position/AABB stubs.
The grasp predicate is stubbed at the module level so importing the checker does not
pull in Genesis (whose native core aborts at import in CPU-only contexts).
"""

import sys
import types
import unittest
from typing import List, Optional, Tuple

import numpy as np
import torch

from ladder.components.inference.robots.robot_arena.scene_loader import ObjectConfig
from ladder.components.inference.robots.robot_arena.success_checker import (
    BatchedSuccessChecker,
    TaskSpec,
    TaskType,
    match_objects,
    parse_instruction,
)
from ladder.components.inference.robots.robot_arena.success_checker.matching import (
    _match_single_object,
    _normalize_name,
    _strip_role_suffix,
    _tokenize,
)

# Stub the grasp predicate so the lazy import in BatchedSuccessChecker._grasped does
# not pull in Genesis (its native core aborts at import on CPU). Installed before any
# test runs; the import inside _grasped resolves to this stub.
_GRASPING_MODULE = "ladder.components.inference.robots.robot_arena.grasping"


def _fake_is_grasping_two_finger(
    robot_entity, obj_entity, left_finger_idx, right_finger_idx, n_envs, **kw  # noqa: ARG001
):
    grasped = np.asarray(getattr(obj_entity, "grasped", False)).reshape(()).astype(bool)
    return torch.as_tensor(np.full(n_envs, bool(grasped), dtype=bool))


_fake_grasping = types.ModuleType(_GRASPING_MODULE)
_fake_grasping.is_grasping_two_finger = _fake_is_grasping_two_finger
sys.modules.setdefault(_GRASPING_MODULE, _fake_grasping)


def _make_object_config(name: str, fixed: bool = False) -> ObjectConfig:
    """Create a minimal ObjectConfig for testing."""
    return ObjectConfig(
        name=name,
        mesh_path="/fake/path.glb",
        position=[0.0, 0.0, 0.0],
        rotation_quat=[1.0, 0.0, 0.0, 0.0],
        scale=1.0,
        mass=0.1,
        friction=0.5,
        surface="Default",
        fixed=fixed,
    )


class MockEntity:
    """Mock Genesis entity with configurable position and AABB."""

    def __init__(
        self,
        pos: Tuple[float, float, float],
        aabb_min: Optional[Tuple[float, float, float]] = None,
        aabb_max: Optional[Tuple[float, float, float]] = None,
        grasped: bool = False,
    ) -> None:
        self._pos = np.array(pos, dtype=np.float64)
        # Default AABB: 10cm cube centered on position.
        half = 0.05
        self._aabb_min = np.array(aabb_min or (pos[0] - half, pos[1] - half, pos[2] - half))
        self._aabb_max = np.array(aabb_max or (pos[0] + half, pos[1] + half, pos[2] + half))
        self.grasped = grasped

    def move_to(self, x: float, y: float, z: float = 0.02, half: float = 0.05) -> None:
        self._pos = np.array([x, y, z], dtype=np.float64)
        self._aabb_min = np.array([x - half, y - half, z - half])
        self._aabb_max = np.array([x + half, y + half, z + half])

    def get_pos(self) -> np.ndarray:
        return self._pos.copy()

    def get_AABB(self) -> Tuple[np.ndarray, np.ndarray]:  # noqa: N802
        return self._aabb_min.copy(), self._aabb_max.copy()


class BatchedMockEntity:
    """Mock entity returning per-env (K,3) state, to exercise the K>=2 batched path."""

    def __init__(self, positions: np.ndarray, half: float = 0.05) -> None:
        self._pos = np.asarray(positions, dtype=np.float64)  # (K, 3)
        self._aabb_min = self._pos - half
        self._aabb_max = self._pos + half
        self.grasped = False

    def get_pos(self) -> torch.Tensor:
        return torch.as_tensor(self._pos)

    def get_AABB(self) -> torch.Tensor:  # noqa: N802
        return torch.stack([torch.as_tensor(self._aabb_min), torch.as_tensor(self._aabb_max)], dim=1)


def _checker(
    task_spec: TaskSpec,
    entities: List[Tuple[object, ObjectConfig]],
    n_envs: int = 1,
    robot_entity: Optional[object] = None,
    fingers: Optional[Tuple[int, int]] = None,
) -> BatchedSuccessChecker:
    left, right = fingers if fingers is not None else (None, None)
    return BatchedSuccessChecker(
        task_instruction="",
        scene_name="test_scene",
        object_entities=entities,
        n_envs=n_envs,
        task_spec_override=task_spec,
        robot_entity=robot_entity,
        left_finger_idx=left,
        right_finger_idx=right,
    )


def _done(checker: BatchedSuccessChecker) -> List[bool]:
    return [r["done"] for r in checker.check_all()]


class TestParseInstruction(unittest.TestCase):
    def test_pick_and_place_put_on(self) -> None:
        result = parse_instruction("put spoon on towel", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICK_AND_PLACE)
        self.assertIn("spoon", result.target_ref)
        self.assertIn("towel", result.destination_ref or "")
        self.assertEqual(result.spatial_relation, "on")

    def test_pick_and_place_put_in(self) -> None:
        result = parse_instruction("put carrot in cup", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICK_AND_PLACE)
        self.assertIn("carrot", result.target_ref)
        self.assertIn("cup", result.destination_ref or "")
        self.assertEqual(result.spatial_relation, "in")

    def test_pick_and_place_place_into(self) -> None:
        result = parse_instruction("Place the blue object into the pot.", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICK_AND_PLACE)
        self.assertIn("blue object", result.target_ref)
        self.assertIn("pot", result.destination_ref or "")
        self.assertEqual(result.spatial_relation, "in")

    def test_pick_and_place_place_on(self) -> None:
        result = parse_instruction("place the silver pot on top of the blue cloth", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.STACK)
        self.assertIn("silver pot", result.target_ref)
        self.assertIn("blue cloth", result.destination_ref or "")

    def test_stack_explicit(self) -> None:
        result = parse_instruction("stack green cube on yellow cube", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.STACK)
        self.assertIn("green cube", result.target_ref)
        self.assertIn("yellow cube", result.destination_ref or "")

    def test_pickup(self) -> None:
        result = parse_instruction("pickup eggplant", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICKUP)
        self.assertIn("eggplant", result.target_ref)

    def test_pick_up_with_space(self) -> None:
        result = parse_instruction("Pick up milk bottle", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICKUP)
        self.assertIn("milk bottle", result.target_ref)

    def test_move_relative_left(self) -> None:
        result = parse_instruction("move the white towel to the left of the orange can", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.MOVE_RELATIVE)
        self.assertIn("white towel", result.target_ref)
        self.assertIn("orange can", result.destination_ref or "")
        self.assertEqual(result.spatial_relation, "left")

    def test_move_relative_right(self) -> None:
        result = parse_instruction("Move the can to the right of the pot", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.MOVE_RELATIVE)
        self.assertIn("can", result.target_ref)
        self.assertEqual(result.spatial_relation, "right")

    def test_move_to_region_corner(self) -> None:
        result = parse_instruction("put blue cube to left corner of the table", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.MOVE_TO_REGION)
        self.assertIn("blue cube", result.target_ref)
        self.assertEqual(result.table_region, "left")

    def test_move_to_region_center(self) -> None:
        result = parse_instruction("move the rectangular block to the center of the table", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.MOVE_TO_REGION)
        self.assertIn("rectangular block", result.target_ref)
        self.assertEqual(result.table_region, "center")

    def test_move_to_bottom_edge(self) -> None:
        result = parse_instruction("move salt shaker to the bottom edge of the table", "test_scene")
        self.assertIsNotNone(result)
        self.assertIn(result.task_type, [TaskType.MOVE_TO_REGION, TaskType.MOVE_RELATIVE])
        self.assertIn("salt shaker", result.target_ref)

    def test_push(self) -> None:
        result = parse_instruction("push milk bottle to the back of the table", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PUSH)
        self.assertIn("milk bottle", result.target_ref)
        self.assertEqual(result.table_region, "upper")

    def test_reach(self) -> None:
        result = parse_instruction("end effector reaching pot or pan", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.REACH)

    def test_between(self) -> None:
        result = parse_instruction(
            "take the blue object and place between two silvered pots and the can", "test_scene"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.BETWEEN)

    def test_override_takes_precedence(self) -> None:
        result = parse_instruction("upright basil bottle cardboard fence", "scene44")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICKUP)
        self.assertIn("red capped bottle", result.target_ref)

    def test_empty_instruction_returns_none(self) -> None:
        result = parse_instruction("", "test_scene")
        self.assertIsNone(result)

    def test_unparseable_instruction_returns_none(self) -> None:
        result = parse_instruction("xyz 123 nonsense", "test_scene")
        self.assertIsNone(result)

    def test_put_on_cutting_board(self) -> None:
        result = parse_instruction("put spatula on cutting board", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICK_AND_PLACE)
        self.assertIn("spatula", result.target_ref)
        self.assertIn("cutting board", result.destination_ref or "")

    def test_put_lid_on_pot(self) -> None:
        result = parse_instruction("put lid on pot", "test_scene")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, TaskType.PICK_AND_PLACE)
        self.assertIn("lid", result.target_ref)
        self.assertIn("pot", result.destination_ref or "")


class TestNormalization(unittest.TestCase):
    def test_normalize_strips_article(self) -> None:
        self.assertEqual(_normalize_name("a red toy bell pepper"), "red toy bell pepper")
        self.assertEqual(_normalize_name("the spoon"), "spoon")
        self.assertEqual(_normalize_name("an orange"), "orange")

    def test_strip_role_suffix(self) -> None:
        self.assertEqual(_strip_role_suffix("spoon_target"), "spoon")
        self.assertEqual(_strip_role_suffix("cloth_destination"), "cloth")
        self.assertEqual(_strip_role_suffix("spoon"), "spoon")

    def test_tokenize(self) -> None:
        tokens = _tokenize("red toy bell pepper")
        self.assertEqual(tokens, {"red", "toy", "bell", "pepper"})


class TestMatchSingleObject(unittest.TestCase):
    def setUp(self) -> None:
        self.objects = [
            _make_object_config("a toy knife with a yellow handle and white blade"),
            _make_object_config("a red toy bell pepper"),
            _make_object_config("a blue cloth_destination", fixed=True),
            _make_object_config("a metal bowl_target"),
        ]

    def test_exact_match(self) -> None:
        idx = _match_single_object("a red toy bell pepper", self.objects, "target")
        self.assertEqual(idx, 1)

    def test_suffix_stripped_match(self) -> None:
        idx = _match_single_object("blue cloth", self.objects, "destination")
        self.assertEqual(idx, 2)

    def test_substring_match(self) -> None:
        idx = _match_single_object("knife", self.objects, "target")
        self.assertEqual(idx, 0)

    def test_substring_match_bowl(self) -> None:
        idx = _match_single_object("metal bowl", self.objects, "target")
        self.assertEqual(idx, 3)

    def test_token_overlap_match(self) -> None:
        idx = _match_single_object("yellow knife", self.objects, "target")
        self.assertEqual(idx, 0)

    def test_no_name_match_falls_back_to_role(self) -> None:
        self.assertEqual(_match_single_object("banana", self.objects, "target"), 3)
        self.assertEqual(_match_single_object("banana", self.objects, "destination"), 2)

    def test_no_match_no_role_returns_none(self) -> None:
        plain = [_make_object_config("apple"), _make_object_config("orange")]
        self.assertIsNone(_match_single_object("banana", plain, "target"))

    def test_exclude_indices(self) -> None:
        idx = _match_single_object("cloth", self.objects, "destination", exclude_indices={2})
        self.assertIsNone(idx)

    def test_disambiguation_prefers_target_suffix(self) -> None:
        objects = [
            _make_object_config("spoon_target"),
            _make_object_config("spoon_destination", fixed=True),
        ]
        idx = _match_single_object("spoon", objects, "target")
        self.assertEqual(idx, 0)

    def test_disambiguation_prefers_destination_suffix(self) -> None:
        objects = [
            _make_object_config("cloth_target"),
            _make_object_config("cloth_destination", fixed=True),
        ]
        idx = _match_single_object("cloth", objects, "destination")
        self.assertEqual(idx, 1)


class TestMatchObjects(unittest.TestCase):
    def test_pick_and_place_resolution(self) -> None:
        objects = [
            _make_object_config("spoon_target"),
            _make_object_config("cloth_destination", fixed=True),
        ]
        task_spec = TaskSpec(
            task_type=TaskType.PICK_AND_PLACE,
            target_ref="spoon",
            destination_ref="cloth",
            spatial_relation="on",
        )
        resolved = match_objects(task_spec, objects)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.target_index, 0)
        self.assertEqual(resolved.destination_index, 1)

    def test_pickup_resolution(self) -> None:
        objects = [_make_object_config("eggplant_target")]
        task_spec = TaskSpec(task_type=TaskType.PICKUP, target_ref="eggplant")
        resolved = match_objects(task_spec, objects)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.target_index, 0)
        self.assertIsNone(resolved.destination_index)

    def test_unmatched_target_returns_none(self) -> None:
        objects = [_make_object_config("spoon")]
        task_spec = TaskSpec(task_type=TaskType.PICKUP, target_ref="banana")
        resolved = match_objects(task_spec, objects)
        self.assertIsNone(resolved)


# A dest block: AABB x[0.1,0.3] y[0.0,0.2] z[0.0,0.10] (top at 0.10).
def _dest() -> MockEntity:
    return MockEntity(pos=(0.2, 0.1, 0.05), aabb_min=(0.1, 0.0, 0.0), aabb_max=(0.3, 0.2, 0.10))


def _pp_spec(relation: str = "on") -> TaskSpec:
    return TaskSpec(
        task_type=TaskType.PICK_AND_PLACE,
        target_ref="obj",
        destination_ref="dest",
        spatial_relation=relation,
    )


class TestPickAndPlace(unittest.TestCase):
    def _entities(self, target: MockEntity) -> List[Tuple[object, ObjectConfig]]:
        return [
            (target, _make_object_config("obj_target")),
            (_dest(), _make_object_config("dest_destination", fixed=True)),
        ]

    def test_placed_on_top(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.15))
        self.assertEqual(_done(_checker(_pp_spec("on"), self._entities(target))), [True])

    def test_too_high_above(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.30))
        self.assertEqual(_done(_checker(_pp_spec("on"), self._entities(target))), [False])

    def test_off_to_side(self) -> None:
        target = MockEntity(pos=(0.5, 0.1, 0.15))
        self.assertEqual(_done(_checker(_pp_spec("on"), self._entities(target))), [False])

    def test_inside_container(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.05))
        self.assertEqual(_done(_checker(_pp_spec("in"), self._entities(target))), [True])

    def test_above_container_not_inside(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.20))
        self.assertEqual(_done(_checker(_pp_spec("in"), self._entities(target))), [False])

    def test_place_gate_when_still_grasped(self) -> None:
        # With grasp wiring present, a placed-but-still-held target is gated to False.
        target = MockEntity(pos=(0.2, 0.1, 0.15), grasped=True)
        checker = _checker(_pp_spec("on"), self._entities(target), robot_entity=object(), fingers=(1, 2))
        self.assertEqual(_done(checker), [False])


class TestStack(unittest.TestCase):
    def test_stacked(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.15))
        spec = TaskSpec(task_type=TaskType.STACK, target_ref="obj", destination_ref="dest")
        entities = [
            (target, _make_object_config("obj_target")),
            (_dest(), _make_object_config("dest_destination", fixed=True)),
        ]
        self.assertEqual(_done(_checker(spec, entities)), [True])


class TestPickup(unittest.TestCase):
    def _checker_with(self, target: MockEntity) -> BatchedSuccessChecker:
        spec = TaskSpec(task_type=TaskType.PICKUP, target_ref="obj")
        return _checker(spec, [(target, _make_object_config("obj_target"))])

    def test_lifted(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02))
        checker = self._checker_with(target)  # init z = 0.02
        target.move_to(0.2, 0.1, 0.10)  # +8cm > pickup_lift_m (3cm)
        self.assertEqual(_done(checker), [True])

    def test_not_lifted(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02))
        checker = self._checker_with(target)
        target.move_to(0.2, 0.1, 0.04)  # +2cm < 3cm
        self.assertEqual(_done(checker), [False])

    def test_latches_after_lift(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02))
        checker = self._checker_with(target)
        target.move_to(0.2, 0.1, 0.10)
        self.assertEqual(_done(checker), [True])
        target.move_to(0.2, 0.1, 0.02)  # dropped back, latch holds
        self.assertEqual(_done(checker), [True])


class TestGrasp(unittest.TestCase):
    def test_grasped(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02), grasped=True)
        spec = TaskSpec(task_type=TaskType.GRASP, target_ref="obj")
        checker = _checker(
            spec, [(target, _make_object_config("obj_target"))], robot_entity=object(), fingers=(1, 2)
        )
        self.assertEqual(_done(checker), [True])

    def test_not_grasped(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02), grasped=False)
        spec = TaskSpec(task_type=TaskType.GRASP, target_ref="obj")
        checker = _checker(
            spec, [(target, _make_object_config("obj_target"))], robot_entity=object(), fingers=(1, 2)
        )
        self.assertEqual(_done(checker), [False])


class TestMoveToRegion(unittest.TestCase):
    def _spec(self) -> TaskSpec:
        return TaskSpec(task_type=TaskType.MOVE_TO_REGION, target_ref="obj", table_region="center")

    def test_moved_into_region(self) -> None:
        target = MockEntity(pos=(0.10, 0.30, 0.02))  # init outside center
        checker = _checker(self._spec(), [(target, _make_object_config("obj_target"))])
        target.move_to(0.265, 0.055, 0.02)  # table center
        self.assertEqual(_done(checker), [True])

    def test_must_enter_gate(self) -> None:
        # Starts already in the region -> degenerate, gated to False even after moving within it.
        target = MockEntity(pos=(0.265, 0.055, 0.02))
        checker = _checker(self._spec(), [(target, _make_object_config("obj_target"))])
        target.move_to(0.30, 0.055, 0.02)  # displaced but still center
        self.assertEqual(_done(checker), [False])

    def test_airborne_gate(self) -> None:
        target = MockEntity(pos=(0.10, 0.30, 0.02))
        checker = _checker(self._spec(), [(target, _make_object_config("obj_target"))])
        target.move_to(0.265, 0.055, 0.20)  # in region XY but lifted high -> not resting
        self.assertEqual(_done(checker), [False])


class TestMoveRelative(unittest.TestCase):
    def test_left_of_reference(self) -> None:
        # Default camera axes: image-right = +x. "left" => target x well below ref x.
        # Init RIGHT of ref so the must-enter gate (started-in-relation) does not fire.
        target = MockEntity(pos=(0.40, 0.10, 0.02))
        ref = MockEntity(pos=(0.30, 0.10, 0.02))
        spec = TaskSpec(
            task_type=TaskType.MOVE_RELATIVE, target_ref="obj", destination_ref="ref", spatial_relation="left"
        )
        entities = [
            (target, _make_object_config("obj_target")),
            (ref, _make_object_config("ref_destination", fixed=True)),
        ]
        checker = _checker(spec, entities)
        target.move_to(0.10, 0.10, 0.02)  # move left of ref, displaced from init
        self.assertEqual(_done(checker), [True])


class TestBetween(unittest.TestCase):
    def test_between(self) -> None:
        target = MockEntity(pos=(0.2, 0.1, 0.02))
        ref1 = MockEntity(pos=(0.15, 0.1, 0.02))
        ref2 = MockEntity(pos=(0.25, 0.1, 0.02))
        spec = TaskSpec(
            task_type=TaskType.BETWEEN, target_ref="obj", destination_ref="ref1", between_anchor_ref="ref2"
        )
        entities = [
            (target, _make_object_config("obj_target")),
            (ref1, _make_object_config("ref1_destination", fixed=True)),
            (ref2, _make_object_config("ref2_anchor", fixed=True)),
        ]
        self.assertEqual(_done(_checker(spec, entities)), [True])

    def test_not_between(self) -> None:
        target = MockEntity(pos=(0.5, 0.1, 0.02))
        ref1 = MockEntity(pos=(0.15, 0.1, 0.02))
        ref2 = MockEntity(pos=(0.25, 0.1, 0.02))
        spec = TaskSpec(
            task_type=TaskType.BETWEEN, target_ref="obj", destination_ref="ref1", between_anchor_ref="ref2"
        )
        entities = [
            (target, _make_object_config("obj_target")),
            (ref1, _make_object_config("ref1_destination", fixed=True)),
            (ref2, _make_object_config("ref2_anchor", fixed=True)),
        ]
        self.assertEqual(_done(_checker(spec, entities)), [False])


class TestReach(unittest.TestCase):
    def test_near_reference(self) -> None:
        target = MockEntity(pos=(0.20, 0.10, 0.02))
        ref = MockEntity(pos=(0.22, 0.10, 0.02))  # within reach_distance_m (8cm)
        spec = TaskSpec(task_type=TaskType.REACH, target_ref="obj", destination_ref="ref")
        entities = [
            (target, _make_object_config("obj_target")),
            (ref, _make_object_config("ref_destination", fixed=True)),
        ]
        self.assertEqual(_done(_checker(spec, entities)), [True])

    def test_far_from_reference(self) -> None:
        target = MockEntity(pos=(0.20, 0.10, 0.02))
        ref = MockEntity(pos=(0.50, 0.10, 0.02))
        spec = TaskSpec(task_type=TaskType.REACH, target_ref="obj", destination_ref="ref")
        entities = [
            (target, _make_object_config("obj_target")),
            (ref, _make_object_config("ref_destination", fixed=True)),
        ]
        self.assertEqual(_done(_checker(spec, entities)), [False])


class TestBatchedPerEnv(unittest.TestCase):
    def test_two_envs_distinct_verdicts(self) -> None:
        # env 0 placed on dest; env 1 off to the side.
        target = BatchedMockEntity(np.array([[0.2, 0.1, 0.15], [0.5, 0.1, 0.15]]))
        dest = BatchedMockEntity(np.array([[0.2, 0.1, 0.05], [0.2, 0.1, 0.05]]))
        dest._aabb_min = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])  # noqa: SLF001
        dest._aabb_max = np.array([[0.3, 0.2, 0.10], [0.3, 0.2, 0.10]])  # noqa: SLF001
        entities = [
            (target, _make_object_config("obj_target")),
            (dest, _make_object_config("dest_destination", fixed=True)),
        ]
        self.assertEqual(_done(_checker(_pp_spec("on"), entities, n_envs=2)), [True, False])

    def test_env_subset_indices(self) -> None:
        target = BatchedMockEntity(np.array([[0.2, 0.1, 0.15], [0.5, 0.1, 0.15], [0.2, 0.1, 0.15]]))
        dest = BatchedMockEntity(np.tile(np.array([0.2, 0.1, 0.05]), (3, 1)))
        dest._aabb_min = np.tile(np.array([0.1, 0.0, 0.0]), (3, 1))  # noqa: SLF001
        dest._aabb_max = np.tile(np.array([0.3, 0.2, 0.10]), (3, 1))  # noqa: SLF001
        entities = [
            (target, _make_object_config("obj_target")),
            (dest, _make_object_config("dest_destination", fixed=True)),
        ]
        checker = BatchedSuccessChecker(
            task_instruction="",
            scene_name="test_scene",
            object_entities=entities,
            n_envs=3,
            env_indices=[0, 2],  # score only envs 0 and 2 (both placed)
            task_spec_override=_pp_spec("on"),
        )
        self.assertEqual(_done(checker), [True, True])


if __name__ == "__main__":
    unittest.main()
