import unittest
from typing import List

from ladder.components.inference.robots.robolab.composition.specs import (
    SPECS,
    advance_legs,
    chain_complete,
    decomp_plan,
)


def _run(n_legs: int, observations: List[List[bool]], finals: List[bool]) -> List[bool]:
    """Feed a chain one observation at a time; return whether it was complete at each step."""
    latched = [False] * n_legs
    out = []
    for holding, final in zip(observations, finals, strict=True):
        latched = advance_legs(latched, holding)
        out.append(chain_complete(latched, final))
    return out


class TestLegChainOrdering(unittest.TestCase):
    """The ordering rule is the whole of a relay: without it the intermediate stops are decoration."""

    def test_legs_reached_in_order_complete_the_chain(self):
        # leg0 holds, then leg1 holds while leg0 no longer does, then the object rests in dest.
        done = _run(2, [[True, False], [False, True], [False, False]], [False, False, True])
        self.assertEqual(done, [False, False, True])

    def test_going_straight_to_the_destination_completes_nothing(self):
        done = _run(2, [[False, False], [False, False]], [True, True])
        self.assertEqual(done, [False, False])

    def test_legs_reached_backwards_do_not_complete_the_chain(self):
        """leg1 before leg0 is not the task: an object that visits the stops in the wrong order has
        not passed through them."""
        done = _run(2, [[False, True], [True, False], [False, False]], [False, False, True])
        self.assertEqual(done, [False, False, False])

    def test_a_latched_leg_stays_latched(self):
        """Starting the next leg necessarily undoes the previous one, so latches never clear."""
        latched = advance_legs([False, False], [True, False])
        self.assertEqual(advance_legs(latched, [False, False]), [True, False])

    def test_a_chain_needs_the_final_placement_to_hold_now(self):
        """An object knocked back out of its last receptacle has not finished, however far it got."""
        done = _run(1, [[True], [False], [False]], [False, True, False])
        self.assertEqual(done, [False, True, False])

    def test_the_frontier_advances_more_than_one_leg_in_a_step(self):
        self.assertEqual(advance_legs([False, False, False], [True, True, False]), [True, True, False])
        self.assertEqual(advance_legs([False, False, False], [True, False, True]), [True, False, False])


class TestRevisitFamily(unittest.TestCase):
    """A revisit returns each object to a receptacle it already left, so its final receptacle
    appears among its legs and only the order tells the two visits apart."""

    def _revisits(self):
        return [spec for spec in SPECS.values() if spec.family == "revisit"]

    def test_the_suite_registers_revisit_tasks(self):
        self.assertTrue(self._revisits())

    def test_a_revisit_returns_each_object_to_its_final_receptacle(self):
        for spec in self._revisits():
            with self.subTest(spec=spec.name):
                finals = {g.receptacle for g in spec.goals}
                self.assertEqual(len(finals), 1)
                self.assertIn(finals.pop(), spec.via_legs, "no leg returns to the final receptacle")

    def test_a_revisit_moves_each_object_three_times(self):
        for spec in self._revisits():
            with self.subTest(spec=spec.name):
                self.assertEqual(spec.n_ops, 3 * spec.n)

    def test_the_plan_lists_every_leg_in_order(self):
        for spec in self._revisits():
            with self.subTest(spec=spec.name):
                plan = decomp_plan(spec)
                self.assertTrue(plan["ordered"])
                self.assertEqual(len(plan["subtasks"]), spec.n_ops)

    def test_the_instruction_names_the_return(self):
        for spec in self._revisits():
            with self.subTest(spec=spec.name):
                self.assertIn("then", spec.instruction.lower())


class TestRelayIsUnchanged(unittest.TestCase):
    """Relay stays a two-leg chain, with the expected op count and plan."""

    def _relays(self):
        return [spec for spec in SPECS.values() if spec.family == "relay"]

    def test_relay_still_moves_each_object_twice(self):
        for spec in self._relays():
            with self.subTest(spec=spec.name):
                self.assertEqual(len(spec.via_legs), 1)
                self.assertEqual(spec.n_ops, 2 * spec.n)
                self.assertEqual(len(decomp_plan(spec)["subtasks"]), 2 * spec.n)


class TestPlacementFamiliesUnaffected(unittest.TestCase):
    def test_a_single_placement_family_declares_no_legs(self):
        for spec in SPECS.values():
            if spec.family in ("relay", "revisit"):
                continue
            with self.subTest(spec=spec.name):
                self.assertEqual(spec.via_legs, ())
                self.assertEqual(spec.n_ops, spec.n)


if __name__ == "__main__":
    unittest.main()
