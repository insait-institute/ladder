from typing import Dict, List, Tuple

import numpy as np

from ladder.components.inference.api.client.episode import Episode, record_motion
from ladder.components.inference.api.core.observation import DualManipulatorState, ManipulatorState


def _arm(x: float, openness: float) -> ManipulatorState:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    return ManipulatorState(eef_pose=pose, gripper_state=np.array([openness], dtype=np.float32))


def _trace(steps) -> Dict[str, List[Tuple[int, List[float]]]]:
    """Record a dual-arm episode from [(step, (right_x, right_openness), (left_x, left_openness))]."""
    trace: Dict[str, List[Tuple[int, List[float]]]] = {}
    for step, right, left in steps:
        record_motion(trace, step, DualManipulatorState(right=_arm(*right), left=_arm(*left)))
    return trace


def test_a_single_arm_state_traces_the_right_arm():
    trace: Dict[str, List[Tuple[int, List[float]]]] = {}
    record_motion(trace, 1, _arm(0.0, 1.0))
    record_motion(trace, 2, _arm(0.5, 1.0))
    assert list(trace) == ["right"]
    assert [step for step, _ in trace["right"]] == [1, 2]


def test_each_arm_writes_its_own_change_points():
    # The right arm moves every step; the left one is parked after the first.
    trace = _trace([(step, (0.1 * step, 1.0), (0.0, 1.0)) for step in range(1, 5)])
    assert [step for step, _ in trace["right"]] == [1, 2, 3, 4]
    assert [step for step, _ in trace["left"]] == [1]


def test_grasp_cycles_are_read_against_each_arms_own_openness():
    # Only the left arm actuates; the right holds a constant grip and must not borrow the left's range.
    openness = [1.0, 1.0, 0.0, 0.0, 1.0]
    episode = Episode(
        env_name="e",
        robot_name="r",
        instruction="i",
        batch=0,
        num_steps=len(openness),
        motion_trace=_trace(
            [(step, (0.1 * step, 1.0), (0.1 * step, value)) for step, value in enumerate(openness, 1)]
        ),
    )
    assert episode.arms == ["right", "left"]
    assert episode.grasp_cycles("right") == []
    assert episode.grasp_cycles("left") == [(3, 5)]
    assert episode.grasps == [(3, 5)]
    assert episode.holding_at_end is False


def test_an_arm_still_holding_at_the_end_leaves_an_open_cycle():
    episode = Episode(
        env_name="e",
        robot_name="r",
        instruction="i",
        batch=0,
        num_steps=3,
        motion_trace=_trace(
            [(1, (0.0, 1.0), (0.0, 1.0)), (2, (0.1, 0.0), (0.1, 1.0)), (3, (0.2, 0.0), (0.2, 1.0))]
        ),
    )
    assert episode.grasps == [(2, None)]
    assert episode.holding_at_end is True


def test_longest_stall_is_measured_over_both_arms_together():
    # The arms take turns moving, so the robot is never idle for more than a step at a time.
    episode = Episode(
        env_name="e",
        robot_name="r",
        instruction="i",
        batch=0,
        num_steps=6,
        motion_trace={
            "right": [(1, [0.0, 0.0, 0.0, 1.0]), (3, [0.5, 0.0, 0.0, 1.0])],
            "left": [(2, [0.0, 0.0, 0.0, 1.0]), (4, [0.5, 0.0, 0.0, 1.0])],
        },
    )
    assert episode.longest_stall == 2  # the tail, steps 4 -> 6; every gap between arms is 1


def test_a_per_arm_trace_round_trips_through_the_log():
    episode = Episode(
        env_name="e",
        robot_name="r",
        instruction="i",
        batch=0,
        num_steps=2,
        motion_trace=_trace([(1, (0.0, 1.0), (0.0, 0.0)), (2, (0.5, 1.0), (0.5, 0.0))]),
    )
    restored = Episode.from_log_entry(episode.to_log_entry())
    assert restored.motion_trace == episode.motion_trace


def test_a_log_written_before_the_split_loads_as_the_right_arm():
    entry = {
        "env_name": "e",
        "robot_name": "r",
        "instruction": "i",
        "batch": 0,
        "num_steps": 2,
        "motion_trace": [[1, [0.0, 0.0, 0.0, 1.0]], [2, [0.5, 0.0, 0.0, 0.0]]],
    }
    restored = Episode.from_log_entry(entry)
    assert restored.arms == ["right"]
    assert restored.motion_trace["right"] == [(1, [0.0, 0.0, 0.0, 1.0]), (2, [0.5, 0.0, 0.0, 0.0])]
