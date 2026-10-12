import unittest
from dataclasses import asdict

import numpy as np

from ladder.components.inference.api.core.action import (
    DualManipulatorAction,
    EndEffectorPoseAction,
    EndEffectorPoseDeltaMode,
    ManipulatorAction,
    RobotAction,
)

ROT_Z_45_DEG = np.array(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

ROT_Z_90_DEG = np.array(
    [
        [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
        [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


class TestEndEffectorPoseAction(unittest.TestCase):
    def test_apply_delta(self):
        rb_eef_translation = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rb_eef_rotation = np.array(ROT_Z_45_DEG)

        current_pose_action = EndEffectorPoseAction(
            translation=rb_eef_translation,
            rotation=rb_eef_rotation,
        )

        sub_tests = [
            (
                "trans_robot_base",
                (
                    np.array(ROT_Z_45_DEG),
                    np.array([0.5, 0, 0], dtype=np.float32),
                    EndEffectorPoseDeltaMode.ROT_EEF_TRANS_BASE,
                ),
                EndEffectorPoseAction(
                    translation=np.array([1.5, 1.0, 1.0], dtype=np.float32),
                    rotation=np.array(ROT_Z_90_DEG),
                ),
            ),
            (
                "trans_end_effector",
                (
                    np.array(ROT_Z_45_DEG),
                    np.array([0.5, 0, 0], dtype=np.float32),
                    EndEffectorPoseDeltaMode.ROT_EEF_TRANS_EEF,
                ),
                EndEffectorPoseAction(
                    translation=np.array(
                        [1.0 + 0.5 * ROT_Z_45_DEG[0, 0], 1.0 + 0.5 * ROT_Z_45_DEG[1, 0], 1.0],
                        dtype=np.float32,
                    ),
                    rotation=np.array(ROT_Z_90_DEG),
                ),
            ),
        ]

        for name, delta_pose, expected_absolute_pose in sub_tests:
            with self.subTest(name=name):
                delta_rotation, delta_translation, delta_mode = delta_pose
                absolute_pose = current_pose_action.apply_delta(
                    delta_translation=delta_translation,
                    delta_rotation=delta_rotation,
                    delta_mode=delta_mode,
                )

                np.testing.assert_array_almost_equal(
                    absolute_pose.translation, expected_absolute_pose.translation
                )
                np.testing.assert_array_almost_equal(absolute_pose.rotation, expected_absolute_pose.rotation)


class TestRobotAction(unittest.TestCase):
    def _make_manipulator(self, translation, gripper) -> ManipulatorAction:
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=np.array(translation, dtype=np.float32),
                rotation=np.array(ROT_Z_45_DEG),
            ),
            gripper_action=gripper,
        )

    def test_manipulator_action_roundtrip(self):
        action = self._make_manipulator([1.0, 2.0, 3.0], gripper=0.5)

        restored = RobotAction.from_dict(asdict(action))

        self.assertIsInstance(restored, ManipulatorAction)
        np.testing.assert_array_almost_equal(restored.arm_action.translation, action.arm_action.translation)
        np.testing.assert_array_almost_equal(restored.arm_action.rotation, action.arm_action.rotation)
        self.assertEqual(restored.gripper_action, action.gripper_action)

    def test_dual_manipulator_action_roundtrip(self):
        action = DualManipulatorAction(
            right=self._make_manipulator([1.0, 2.0, 3.0], gripper=0.5),
            left=self._make_manipulator([-1.0, -2.0, -3.0], gripper=0.1),
        )

        restored = RobotAction.from_dict(asdict(action))

        self.assertIsInstance(restored, DualManipulatorAction)
        for side in ("right", "left"):
            original_arm = getattr(action, side).arm_action
            restored_arm = getattr(restored, side).arm_action
            np.testing.assert_array_almost_equal(restored_arm.translation, original_arm.translation)
            np.testing.assert_array_almost_equal(restored_arm.rotation, original_arm.rotation)
            self.assertEqual(getattr(restored, side).gripper_action, getattr(action, side).gripper_action)

    def test_from_dict_dispatches_on_type(self):
        single = self._make_manipulator([1.0, 2.0, 3.0], gripper=0.5)
        dual = DualManipulatorAction(
            right=self._make_manipulator([1.0, 2.0, 3.0], gripper=0.5),
            left=self._make_manipulator([-1.0, -2.0, -3.0], gripper=0.1),
        )

        self.assertIsInstance(RobotAction.from_dict(asdict(single)), ManipulatorAction)
        self.assertIsInstance(RobotAction.from_dict(asdict(dual)), DualManipulatorAction)


if __name__ == "__main__":
    unittest.main()
