import collections
import unittest
from typing import Dict, List

import torch

from ladder.components.inference.robots.simpler.composition.placement import (
    CARROT,
    FOOTPRINT_SLACK_M,
    INSTANCE_SEP,
    PLATE,
    SPOON,
    TOWEL,
    build_layouts,
    strip_instance_id,
)
from ladder.components.inference.robots.simpler.composition.specs import SPECS, Spec, scene_layout

_LAYOUTS = 24  # the per-episode grid every registered scene is evaluated over


def _by_name() -> Dict[str, Spec]:
    return {spec.name: spec for spec in SPECS}


class TestEverySpecPacks(unittest.TestCase):
    """The packer is the only gate a new rung passes before it costs GPU hours. A spec whose
    receptacles leave no room for its movers raises 'cannot pack' here rather than during an
    eval run."""

    def test_every_spec_packs_over_the_full_layout_grid(self):
        for spec in SPECS:
            layout = scene_layout(spec)
            with self.subTest(spec=spec.name):
                xyz, _ = build_layouts(layout.obj_names, layout.bbox, layout.goals, _LAYOUTS, spec.start_on)
                self.assertEqual(xyz.shape[0], _LAYOUTS)
                self.assertEqual(xyz.shape[1], len(layout.obj_names))


class TestNoEpisodeStartsScored(unittest.TestCase):
    """``BaseEnv.reset`` calls ``get_info``, so ``composition.envs.evaluate`` runs before the policy acts:
    a mover spawned inside a goal target's scoring region is credited at step 0, which inflates
    num_subgoals_complete and stamps _first_done for a goal nobody achieved. Only the footprint is checked
    here -- the height test compares centres, and every mover sits above every receptacle, so it never
    discriminates."""

    def test_no_mover_spawns_inside_a_goal_targets_footprint(self):
        for spec in SPECS:
            layout = scene_layout(spec)
            xyz, _ = build_layouts(layout.obj_names, layout.bbox, layout.goals, _LAYOUTS, spec.start_on)
            index = {name: i for i, name in enumerate(layout.obj_names)}
            for target in sorted({goal.target for goal in layout.goals}):
                half = torch.tensor(layout.bbox[target]) / 2 + FOOTPRINT_SLACK_M
                for mover in sorted({goal.source for goal in layout.goals}):
                    offset = xyz[:, index[mover], :2] - xyz[:, index[target], :2]
                    scored = (offset.abs() <= half).all(dim=1)
                    with self.subTest(spec=spec.name, mover=mover, target=target):
                        self.assertFalse(
                            bool(scored.any()), f"layouts {scored.nonzero().flatten().tolist()} start scored"
                        )


class TestNamesAreUnique(unittest.TestCase):
    def test_no_two_specs_share_a_name(self):
        """Two specs of one name silently register one env, so the other is never evaluated."""
        duplicated = [name for name, n in collections.Counter(s.name for s in SPECS).items() if n > 1]
        self.assertEqual(duplicated, [])


class TestNoUndeclaredDistractors(unittest.TestCase):
    """``envs.evaluate`` fails an episode whose distractor moved, so only a family whose wording
    states that constraint may place one."""

    def test_only_the_exclusion_family_places_a_distractor(self):
        for spec in SPECS:
            with self.subTest(spec=spec.name):
                self.assertEqual(
                    bool(spec.distractors),
                    spec.name.startswith("except_"),
                    f"{spec.name}: distractors={spec.distractors} does not match its family",
                )

    def test_every_object_in_a_scene_is_a_goal_a_distractor_or_the_start_receptacle(self):
        for spec in SPECS:
            layout = scene_layout(spec)
            scored = {goal.source for goal in layout.goals} | {goal.target for goal in layout.goals}
            declared = scored | set(spec.distractors) | ({spec.start_on} if spec.start_on else set())
            with self.subTest(spec=spec.name):
                self.assertEqual([name for name in layout.obj_names if name not in declared], [])


class TestTheLengthAxisStartsAtOne(unittest.TestCase):
    """Rung 1 is the canonical bridge pair at native receptacle size, not a scaled rung of this
    ladder."""

    def test_the_single_destination_family_has_a_one_goal_rung_per_receptacle(self):
        by_name = _by_name()
        for name, pair in (("all1_plate", (CARROT, PLATE)), ("all1_towel", (SPOON, TOWEL))):
            with self.subTest(spec=name):
                self.assertIn(name, by_name)
                self.assertEqual(by_name[name].goals, [pair])
                self.assertEqual(by_name[name].target_scales, {})


class TestRevisitFamilies(unittest.TestCase):
    """A revisit chain moves the SAME object twice. Instancing it instead would silently turn a
    3-leg relay into a 3-object scene, which packs differently and scores differently."""

    def test_a_revisit_spec_reuses_its_movers_rather_than_duplicating_them(self):
        for spec in SPECS:
            if not spec.revisits:
                continue
            layout = scene_layout(spec)
            with self.subTest(spec=spec.name):
                self.assertFalse(
                    any(INSTANCE_SEP in name for name in layout.obj_names),
                    "a revisit leg was given its own object instance",
                )
                movers = {strip_instance_id(s) for s, _ in spec.goals}
                self.assertLessEqual(len(movers), 3, "only 3 movers exist in the scene")

    def test_a_relay_moves_each_mover_more_than_once(self):
        """Without a repeated mover the flag means nothing: the spec would be an ordinary
        single-destination scene registered under a family that claims otherwise."""
        for spec in SPECS:
            if not spec.revisits:
                continue
            with self.subTest(spec=spec.name):
                self.assertGreater(len(spec.goals), len({source for source, _ in spec.goals}))

    def test_a_shuttle_returns_a_mover_to_a_receptacle_it_already_left(self):
        """What separates revisit from relay: the same (mover, receptacle) pair occurs twice, which
        is the leg no end-state conjunction can witness. shuttle7 is the family floor: three movers make
        one round, so the third round is the first to return anything."""
        for spec in SPECS:
            if not spec.name.startswith("shuttle"):
                continue
            with self.subTest(spec=spec.name):
                self.assertLess(len(set(spec.goals)), len(spec.goals), "no leg is revisited by this shuttle")


class TestOrderedFamilyStaysStateable(unittest.TestCase):
    def test_no_ordered_spec_repeats_a_mover(self):
        """'first the carrot, then the carrot' cannot be followed. Only 3 movers exist, so the
        ordered family caps at 3 goals; this is the assertion that keeps it there."""
        for spec in SPECS:
            if not spec.ordered or spec.revisits:
                continue
            sources: List[str] = [source for source, _ in spec.goals]
            with self.subTest(spec=spec.name):
                self.assertEqual(len(sources), len(set(sources)))


class TestTypeCoverageAcrossLength(unittest.TestCase):
    """The suite exists to vary task type and length independently, so each type has to appear at
    more than one length. A family pinned to one rung cannot separate the two axes."""

    _FAMILIES = {
        "all": "single destination",
        "move_": "transfer",
        "split": "split destination",
        "relay": "relay",
        "shuttle": "revisit",
        "except": "exclusion",
        "ordered": "ordered",
        "spoons": "duplicates",
    }

    def test_every_family_spans_at_least_two_lengths(self):
        lengths: Dict[str, set] = collections.defaultdict(set)
        for spec in SPECS:
            for prefix, family in self._FAMILIES.items():
                if spec.name.startswith(prefix):
                    lengths[family].add(len(spec.goals))
                    break
        for family, seen in lengths.items():
            with self.subTest(family=family):
                self.assertGreaterEqual(len(seen), 2, f"{family} exists at one length only: {seen}")

    def test_every_family_reaches_the_lengths_the_type_comparison_pools(self):
        """Section 5.2 of the paper pools types over 2-5 goals. Revisit starts at 4 because a
        shuttle needs a 4th leg before anything is revisited; ordered stops at 3 for want of a
        4th distinct mover. Every other family has to cover the window."""
        window = {2, 3, 4, 5}
        # Two families cannot cover the window and the reason is structural, not a missing rung.
        # A shuttle revisits nothing before leg 7 (3 movers make a round), and an ordered task
        # cannot repeat a mover, so with 3 movers it stops at 3.
        floors = {"revisit": {7, 8, 9, 10}, "ordered": {2, 3}}
        lengths: Dict[str, set] = collections.defaultdict(set)
        for spec in SPECS:
            for prefix, family in self._FAMILIES.items():
                if spec.name.startswith(prefix):
                    lengths[family].add(len(spec.goals))
                    break
        for family, seen in lengths.items():
            with self.subTest(family=family):
                self.assertTrue(floors.get(family, window) <= seen, f"{family} covers {sorted(seen)}")


if __name__ == "__main__":
    unittest.main()
