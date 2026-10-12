from typing import Dict, List, Optional, Sequence, Tuple

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer
from ladder.components.inference.api.client.hlc.gripper import gripper_is_open
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config


class PrivilegedAdvancerConfig(Config):
    advance_settle_steps: int = 3  # consecutive steps achieved[idx] must HOLD before advancing
    gripper_open_threshold: float = 0.5  # only advance once the gripper isn't still clamping
    # Skip-and-return patience (0 = retry-in-place). >0: after this many steps with the current goal
    # unachieved, defer it at the next moment the gripper is open and rotate to the next unachieved
    # goal (revisited on wrap-around), so one ungraspable object can't freeze the episode;
    # completion also advances round-robin rather than snapping back to unachieved[0]. The count is
    # TIME ON THE GOAL, not consecutive open-gripper steps, so a policy that keeps re-grasping cannot
    # reset it forever.
    stuck_patience_steps: int = 0
    # Schedule placements onto a flat SURFACE after placements into a CONTAINER. A goal already
    # satisfied on an open surface stays exposed for every subtask that follows it, and the arm
    # crossing back over the table knocks it off; a container goal cannot be undone that way.
    # Reads the destination off the oracle's own command vocabulary ("in the" = container,
    # "on the" = surface, per _TEMPLATES in robots/robolab/oracle.py). Off by default: it changes
    # WHICH goal is commanded, so it is a scheduling policy, not a correction. Ignored on tasks the
    # env flags ``ordered``, whose leg order is prescribed and would be inverted by it.
    surface_last: bool = False


class PrivilegedAdvancer(SubtaskAdvancer[PrivilegedAdvancerConfig]):
    """Advance on the env's privileged ``metadata['oracle']['achieved']``, debounced and decoupled
    from the gripper-release instant (the flag flips True several steps AFTER release). Retargets the
    first still-unachieved goal (greedy repair). With ``stuck_patience_steps`` > 0 a goal that won't
    complete is deferred (skip-and-return) so it can't starve the rest."""

    def __init__(self, config: PrivilegedAdvancerConfig) -> None:
        super().__init__(config)
        self._achieved_streak: Dict[int, int] = {}
        self._stuck_steps: Dict[int, int] = {}

    def reset(self, env_idx: int) -> None:
        self._achieved_streak[env_idx] = 0
        self._stuck_steps[env_idx] = 0

    @staticmethod
    def _split_by_destination(unachieved: List[int], subtasks: Sequence[str]) -> Tuple[List[int], List[int]]:
        """``unachieved`` split into (container goals, surface goals), each keeping its order.

        The destination is read off the oracle's own command vocabulary: "put X in the <container>"
        against "put X on the <surface>" (``_TEMPLATES`` in robots/robolab/oracle.py).
        """
        surface = [i for i in unachieved if i < len(subtasks) and " on the " in subtasks[i]]
        return [i for i in unachieved if i not in surface], surface

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        achieved: Optional[List[bool]] = ((observation.metadata or {}).get("oracle") or {}).get("achieved")
        # Absent is not "nothing achieved yet": with no list every goal reads as done, the advancer
        # takes the all-complete branch below and never moves again. There is no step budget to
        # recover on, and the episode would score zero without erroring.
        if not achieved:
            raise RuntimeError(
                "PrivilegedAdvancer needs a non-empty metadata['oracle']['achieved']; this env "
                "publishes none, so it could never advance"
            )
        ordered = bool(((observation.metadata or {}).get("oracle") or {}).get("ordered"))
        current_done = current_idx < len(achieved) and achieved[current_idx]
        self._achieved_streak[env_idx] = self._achieved_streak.get(env_idx, 0) + 1 if current_done else 0
        not_clamping = gripper_is_open(observation, self.config.gripper_open_threshold) is not False
        unachieved = [i for i, done in enumerate(achieved) if not done]
        if not unachieved:
            return current_idx
        # Skip-and-return is meaningless on a SEQUENCE: success needs the goals' FIRST satisfactions
        # in order, so rotating past an unfinished goal does not rescue the episode, it forfeits it.
        patience = 0 if ordered else self.config.stuck_patience_steps

        # Reordering is as meaningless on a SEQUENCE as the rotation above, and for the same reason:
        # an ordered task's success needs the goals' first satisfactions IN ORDER, so hoisting the
        # containers ahead of the surfaces forfeits it. Some ordered tasks prescribe the surface leg
        # FIRST ("put everything on the plate, THEN move it to the bin"), the exact order this inverts.
        containers: List[int] = []
        surfaces: List[int] = []
        if self.config.surface_last and not ordered and decomposition is not None:
            containers, surfaces = self._split_by_destination(unachieved, decomposition.subtasks)
            unachieved = containers + surfaces

        def _next_after(idx: int) -> int:
            # Walk `unachieved` POSITIONALLY, so surface_last's ordering survives the rotation.
            # With the flag off the list is ascending and this is the old "next higher index,
            # wrapping" behaviour exactly; the numeric fallback covers a current goal that has
            # since been achieved and so is no longer in the list.
            if idx in unachieved:
                return unachieved[(unachieved.index(idx) + 1) % len(unachieved)]
            later = [i for i in unachieved if i > idx]
            return later[0] if later else unachieved[0]

        # Reordering only takes effect where an index is CHOSEN, and the arm opens on whatever
        # index the controller starts with (goal 0, often a surface goal). So redirect off a surface
        # goal while containers remain, with the gripper OPEN: retargeting mid-carry would deliver
        # the held object to the new goal's receptacle.
        #
        # The test is that the current goal IS an unachieved surface goal, not merely that it is
        # absent from `containers`. A goal that has just been achieved is in neither list, so the
        # looser test would fire on every single-destination task the moment a goal completed and
        # jump to the next goal with the settle streak reset, bypassing advance_settle_steps.
        if containers and current_idx in surfaces and not_clamping:
            self._achieved_streak[env_idx] = 0
            self._stuck_steps[env_idx] = 0
            return containers[0]

        if self._achieved_streak[env_idx] >= self.config.advance_settle_steps and not_clamping:
            self._achieved_streak[env_idx] = 0
            self._stuck_steps[env_idx] = 0
            return _next_after(current_idx) if patience > 0 else unachieved[0]

        # Count time ON THE SUBTASK, not consecutive gripper-open steps. Gating the COUNTER on
        # not_clamping would reset it on every grasp, so a policy that runs several grasp cycles per
        # goal would never accumulate `patience` and freeze on one subtask. The gate belongs on the
        # DEFERRAL instead: do not retarget mid-grasp.
        if patience > 0 and not current_done:
            self._stuck_steps[env_idx] = self._stuck_steps.get(env_idx, 0) + 1
            if self._stuck_steps[env_idx] >= patience and not_clamping:
                self._stuck_steps[env_idx] = 0
                return _next_after(current_idx)
        else:
            self._stuck_steps[env_idx] = 0
        return current_idx
