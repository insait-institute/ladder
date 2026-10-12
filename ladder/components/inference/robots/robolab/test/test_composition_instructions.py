import unittest

from ladder.components.inference.robots.robolab.composition.specs import SPECS


class TestCompositionInstructions(unittest.TestCase):
    """A distractor task must NAME what to leave behind, the way SimplerEnv ("put everything except
    the eggplant on the plate") and RobotArena ("put everything except the pot on the cloth") do, so
    the three composition benchmarks state the same task type the same way."""

    def test_one_distractor_is_named(self):
        self.assertEqual(
            SPECS["comp_bin_2_distractor"].instruction,
            "put everything except the red block in the bin",
        )

    def test_two_distractors_are_both_named(self):
        self.assertEqual(
            SPECS["comp_bin_4_distractor"].instruction,
            "put everything except the red block and the blue block in the bin",
        )

    def test_a_task_without_distractors_names_its_objects(self):
        """Without an exclusion to resolve, "put everything in the bin" is the same string at two
        objects and at ten and tests no referring expression, so the bin rungs enumerate."""
        self.assertEqual(
            SPECS["comp_bin_2"].instruction,
            "put the banana and the orange in the bin",
        )
        self.assertEqual(SPECS["comp_identical_4"].instruction, "put the four limes in the bin")


if __name__ == "__main__":
    unittest.main()
