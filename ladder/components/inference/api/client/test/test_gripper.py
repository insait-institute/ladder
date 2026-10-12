import numpy as np

from ladder.components.inference.api.client.hlc.advancers.gripper import (
    GripperAdvancer,
    GripperAdvancerConfig,
)
from ladder.components.inference.api.client.hlc.base import Decomposition
from ladder.components.inference.api.client.hlc.gripper import GripperTracker
from ladder.components.inference.api.core.observation import (
    DualManipulatorState,
    ManipulatorState,
    RobotObservation,
)


def _obs(openness, z=0.0):
    gs = None if openness is None else np.array([openness], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = z
    return RobotObservation(
        images=None,
        state=ManipulatorState(eef_pose=pose, eef_pose_from_target=None, gripper_state=gs, joints=None),
        timestamps=None,
        metadata={},
    )


def _dual_obs(right_openness, left_openness, right_z=0.0, left_z=0.0):
    return RobotObservation(
        images=None,
        state=DualManipulatorState(
            right=_obs(right_openness, right_z).state, left=_obs(left_openness, left_z).state
        ),
        timestamps=None,
        metadata={},
    )


def test_each_arm_latches_on_its_own_gripper_and_the_env_answer_is_either_arm():
    t = GripperTracker(open_threshold=0.5)
    t.reset(0)
    t.update(_dual_obs(0.9, 0.9), 0)  # both open
    assert t.is_closed(0) is False and t.toggled(0) is False
    t.update(_dual_obs(0.9, 0.1), 0)  # the left arm grasps
    assert t.is_closed(0, arm="right") is False
    assert t.is_closed(0, arm="left") is True
    assert t.is_closed(0) is True  # either arm holding means the robot is holding
    assert t.toggled(0) is True and t.toggled(0, arm="right") is False
    assert t.closed_streak(0) == 1 and t.closed_streak(0, arm="right") == 0


def test_a_sustained_release_fires_for_the_arm_that_held():
    t = GripperTracker(open_threshold=0.5)
    t.reset(0)
    fires = []
    for left in [0.9, 0.1, 0.1, 0.1, 0.1, 0.9]:  # only the left arm grasps and releases
        t.update(_dual_obs(0.9, left), 0)
        fires.append(t.released_after_hold(0, min_hold_steps=3))
    assert fires == [False, False, False, False, False, True]
    assert t.released_after_hold(0, min_hold_steps=3, arm="right") is False
    assert t.ever_closed(0, arm="left") is True and t.ever_closed(0, arm="right") is False


def test_sustained_grasp_release_fires_once_on_release():
    t = GripperTracker(open_threshold=0.5)
    t.reset(0)
    fires = []
    for openness in [0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]:  # open, grasp-held x4, release, open
        t.update(_obs(openness), 0)
        fires.append(t.released_after_hold(0, min_hold_steps=3))
    assert fires == [False, False, False, False, False, True, False]
    assert t.ever_closed(0) is True


def test_hysteresis_holds_latch_in_band_vs_plain_threshold_chatters():
    band = [0.9, 0.4, 0.6, 0.5, 0.1]  # open, three in-band values, then below-band -> closed
    hyst = GripperTracker(open_threshold=0.5, hysteresis=0.2)  # band [0.3, 0.7]
    plain = GripperTracker(open_threshold=0.5, hysteresis=0.0)
    hyst.reset(0)
    plain.reset(0)
    tog_hyst, tog_plain = [], []
    for openness in band:
        hyst.update(_obs(openness), 0)
        plain.update(_obs(openness), 0)
        tog_hyst.append(hyst.toggled(0))
        tog_plain.append(plain.toggled(0))
    assert tog_hyst == [False, False, False, False, True]  # only the below-band value flips
    assert tog_plain == [False, True, True, True, False]  # plain threshold flips inside the band


def test_no_gripper_signal_leaves_state_untouched():
    t = GripperTracker(open_threshold=0.5)
    t.reset(0)
    t.update(_obs(0.1), 0)  # closed, streak 1
    assert t.closed_streak(0) == 1 and t.is_closed(0) is True
    t.update(_obs(None), 0)  # no signal -> no change, no toggle
    assert t.closed_streak(0) == 1 and t.is_closed(0) is True and t.toggled(0) is False
    t.update(_obs(0.1), 0)  # closed again -> streak resumes
    assert t.closed_streak(0) == 2


def test_lift_gate_rejects_no_lift_and_passes_real_lift():
    no_lift = GripperTracker(open_threshold=0.5)
    no_lift.reset(0)
    for openness, z in [(0.9, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0), (0.9, 0.0)]:
        no_lift.update(_obs(openness, z), 0)
    assert no_lift.released_after_hold(0, min_hold_steps=3, lift_threshold_m=0.04) is False

    lifted = GripperTracker(open_threshold=0.5)
    lifted.reset(0)
    for openness, z in [(0.9, 0.0), (0.1, 0.0), (0.1, 0.06), (0.1, 0.06), (0.9, 0.06)]:
        lifted.update(_obs(openness, z), 0)
    assert lifted.released_after_hold(0, min_hold_steps=3, lift_threshold_m=0.04) is True


def _advancer(**kwargs):
    return GripperAdvancer(GripperAdvancerConfig(**kwargs))


def _plan(n):
    return Decomposition(subtasks=[f"subtask {i}" for i in range(n)], ordered=False)


def _hold_open(advancer, plan, steps, idx=0):
    """Feed `steps` open-gripper observations, returning the index the advancer settles on."""
    for _ in range(steps):
        idx = advancer.next_index(_obs(0.9), 0, idx, plan)
    return idx


def test_stall_backstop_advances_a_subtask_the_policy_never_grasps():
    # The defect this exists for: with a gate strict enough to reject failed clenches, a subtask the
    # policy never completes strands the arm for the whole episode.
    plan = _plan(3)
    stuck = _advancer(min_hold_steps=21, lift_threshold_m=0.03)
    stuck.reset(0)
    assert _hold_open(stuck, plan, 500) == 0

    escaping = _advancer(min_hold_steps=21, lift_threshold_m=0.03, attempt_stall_steps=100)
    escaping.reset(0)
    assert _hold_open(escaping, plan, 500) == 2  # walks the plan instead of freezing on subtask 0


def test_stall_backstop_does_not_preempt_a_real_grasp():
    # A qualifying release must still be what advances, so the backstop cannot be what is measured.
    plan = _plan(3)
    a = _advancer(min_hold_steps=3, attempt_stall_steps=100)
    a.reset(0)
    idx = 0
    for openness in [0.9, 0.1, 0.1, 0.1, 0.1, 0.9]:  # open, sustained grasp, release
        idx = a.next_index(_obs(openness), 0, idx, plan)
    assert idx == 1  # advanced on the release at step 6, far short of the 100-step backstop


def test_stall_counter_restarts_when_the_subtask_changes():
    # The count is time ON THE COMMANDED SUBTASK; a caller-driven change must not carry it over.
    plan = _plan(3)
    a = _advancer(min_hold_steps=21, attempt_stall_steps=100)
    a.reset(0)
    _hold_open(a, plan, 99)
    assert a.next_index(_obs(0.9), 0, 1, plan) == 1  # index moved: counter restarts, no advance
    assert _hold_open(a, plan, 98, idx=1) == 1
    assert a.next_index(_obs(0.9), 0, 1, plan) == 2  # 100 steps on subtask 1 -> now it fires
