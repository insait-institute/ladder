import unittest
from typing import Set

from ladder.components.inference.robots.robolab.composition.specs import SPECS


class TestCompositionBenchmark(unittest.TestCase):
    def test_the_benchmark_is_exactly_these_tasks(self):
        """SPECS is the benchmark -- there is no second list to select from it, because there used
        to be and the two drifted. Campaigns were run on retired variants (bowl-receptacle twins,
        intermediate ladder rungs) that scored as if they were benchmark results and cost days of
        GPU time. Adding or removing a task must be a deliberate edit here, so that a change to the
        benchmark cannot happen as a side effect of touching the generators."""
        expected: Set[str] = {
            "comp_bin_1",
            "comp_plate_1",
            "comp_bin_2",
            "comp_bin_6",
            "comp_bin_10",
            "comp_bin_2_distractor",
            "comp_bin_4_distractor",
            "comp_ordered_bin_2",
            "comp_ordered_bin_3",
            "comp_ordered_bin_4",
            "comp_ordered_bin_5",
            "comp_ordered_bin_6",
            "comp_ordered_bin_7",
            "comp_ordered_bin_8",
            "comp_ordered_bin_9",
            "comp_ordered_bin_10",
            "comp_transfer_2",
            "comp_transfer_3",
            "comp_relay_2",
            "comp_relay_3",
            "comp_relay_4",
            "comp_relay_5",
            "comp_revisit_1",
            "comp_revisit_2",
            "comp_route_2",
            "comp_route_3",
            "comp_route_4",
            "comp_route_5",
            "comp_route_6",
            "comp_route_7",
            "comp_route_8",
            "comp_route_9",
            "comp_route_10",
            "comp_route_2_binfirst",
            "comp_revisit_3",
            "comp_split_plate_2",
            "comp_split_plate_3",
            "comp_split_plate_4",
            "comp_split_plate_5",
            "comp_split_plate_6",
            "comp_split_plate_7",
            "comp_split_plate_8",
            "comp_identical_2",
            "comp_identical_4",
        }
        self.assertEqual(set(SPECS), expected)


class TestNoUndeclaredDistractors(unittest.TestCase):
    """An object with no receptacle is present, scored nothing and required to stay put, so only a
    family whose wording states that constraint may place one."""

    def test_only_the_distractor_family_places_a_distractor(self):
        for name, spec in SPECS.items():
            with self.subTest(spec=name):
                self.assertEqual(
                    bool(spec.distractor_objects),
                    spec.family == "distractor",
                    f"{name}: {[o.type_id for o in spec.distractor_objects]} in family {spec.family!r}",
                )

    def test_every_receptacle_is_a_destination_or_the_source_of_a_transfer(self):
        for name, spec in SPECS.items():
            used = {obj.receptacle for obj in spec.objects if obj.receptacle}
            used |= {obj.start_on for obj in spec.objects if obj.start_on}
            used |= set(spec.via_legs)
            with self.subTest(spec=name):
                self.assertEqual([r.prim for r in spec.receptacles if r.prim not in used], [])
