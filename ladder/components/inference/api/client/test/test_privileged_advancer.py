import unittest
from typing import Sequence

import numpy as np

from ladder.components.inference.api.client.hlc.advancers.privileged import (
    PrivilegedAdvancer,
    PrivilegedAdvancerConfig,
)
from ladder.components.inference.api.client.hlc.base import Decomposition
from ladder.components.inference.api.core.observation import ManipulatorState, RobotObservation


def _observation(achieved: Sequence[bool], ordered: bool = False, openness: float = 0.9) -> RobotObservation:
    return RobotObservation(
        images=None,
        state=ManipulatorState(
            eef_pose=np.eye(4, dtype=np.float32),
            eef_pose_from_target=None,
            gripper_state=np.array([openness], dtype=np.float32),
            joints=None,
        ),
        timestamps=None,
        metadata={"oracle": {"achieved": list(achieved), "ordered": ordered}},
    )


class TestSurfaceLast(unittest.TestCase):
    """Placements onto a flat surface must be scheduled after placements into a container.

    A surface placement stays exposed for every subtask that follows it and the arm knocks it off
    crossing back; a container placement cannot be undone that way.
    """

    PLAN = Decomposition(
        subtasks=[
            "put the lemon on the plate",  # 0 surface
            "put the lime on the plate",  # 1 surface
            "put the orange in the bin",  # 2 container
            "put the avocado in the bin",  # 3 container
        ],
        ordered=False,
    )

    def _advancer(self, **kwargs) -> PrivilegedAdvancer:
        advancer = PrivilegedAdvancer(PrivilegedAdvancerConfig(**kwargs))
        advancer.reset(0)
        return advancer

    def test_container_goals_are_commanded_before_surface_goals(self):
        advancer = self._advancer(surface_last=True, advance_settle_steps=1)
        self.assertEqual(advancer.next_index(_observation([False] * 4), 0, 0, self.PLAN), 2)

    def test_ordering_is_off_by_default(self):
        advancer = self._advancer(advance_settle_steps=1)
        self.assertEqual(advancer.next_index(_observation([False] * 4), 0, 0, self.PLAN), 0)

    def test_an_ordered_task_keeps_its_declared_order(self):
        # A relay ("put everything on the plate, THEN move it to the bin") prescribes the surface leg
        # first, so hoisting the containers would invert the very sequence success is scored on.
        advancer = self._advancer(surface_last=True, advance_settle_steps=1)
        observation = _observation([False] * 4, ordered=True)
        self.assertEqual(advancer.next_index(observation, 0, 0, self.PLAN), 0)

    def test_surface_goals_are_still_reached_once_containers_are_done(self):
        advancer = self._advancer(surface_last=True, advance_settle_steps=1)
        observation = _observation([False, False, True, True])
        self.assertEqual(advancer.next_index(observation, 0, 2, self.PLAN), 0)

    ONE_DESTINATION = Decomposition(
        subtasks=[
            "put the lemon in the bin",
            "put the lime in the bin",
            "put the orange in the bin",
        ],
        ordered=False,
    )

    def test_a_single_destination_task_is_left_alone(self):
        """With no surface goal there is nothing to schedule late, so the flag must not change what
        is commanded OR when. comp_bin_N, comp_identical_N and comp_transfer_N are all of this shape
        -- 25 of the suite's 42 units -- and the redirect below used to fire on them the instant a
        goal flipped achieved, jumping to the next goal with the settle streak reset and so
        bypassing advance_settle_steps entirely. Measured cost over those 25 units, 300 episodes an
        arm: -7.3 episode success against the same oracle with the flag off."""
        for achieved, current in (
            ([False, False, False], 0),
            ([True, False, False], 0),
            ([True, True, False], 1),
        ):
            with self.subTest(achieved=achieved, current=current):
                plain = self._advancer(advance_settle_steps=120, stuck_patience_steps=600)
                flagged = self._advancer(
                    surface_last=True, advance_settle_steps=120, stuck_patience_steps=600
                )
                observation = _observation(achieved)
                self.assertEqual(
                    flagged.next_index(observation, 0, current, self.ONE_DESTINATION),
                    plain.next_index(observation, 0, current, self.ONE_DESTINATION),
                )

    def test_an_achieved_surface_goal_does_not_short_circuit_the_settle(self):
        """The redirect exists to steer the arm OFF an unachieved surface goal while containers
        remain. Once that goal is achieved the settle clock owns the decision, so redirecting on it
        would hand back the 8 s hold that advance_settle_steps was raised to 120 to buy."""
        advancer = self._advancer(surface_last=True, advance_settle_steps=120)
        observation = _observation([True, False, False, False])
        self.assertEqual(advancer.next_index(observation, 0, 0, self.PLAN), 0)

    def test_rotation_walks_the_prioritised_order_not_the_index_order(self):
        # Leaving the first container goal under patience must land on the SECOND container goal.
        # The old index-based rotation would have returned 3 here too, so the discriminating case is
        # the wrap: from the last container it must go to a surface, not back to index 0 blindly.
        advancer = self._advancer(surface_last=True, stuck_patience_steps=1, advance_settle_steps=99)
        observation = _observation([False] * 4)
        advancer.next_index(observation, 0, 3, self.PLAN)
        self.assertEqual(advancer.next_index(observation, 0, 3, self.PLAN), 0)


if __name__ == "__main__":
    unittest.main()
