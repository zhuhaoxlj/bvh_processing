"""Focused tests for the offline BVH parser and G1 kinematic model."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PROJECT_ROOT / "scripts/bvh_to_g1_csv.py"
BVH_PATH = PROJECT_ROOT / "motions/Take_007_053_Skeleton7.bvh"
URDF_PATH = (
    PROJECT_ROOT
    / "source/whole_body_tracking/whole_body_tracking/assets"
    / "unitree_description/urdf/g1/main.urdf"
)

module_specification = importlib.util.spec_from_file_location(
    "bvh_to_g1_csv", CONVERTER_PATH
)
assert module_specification is not None
assert module_specification.loader is not None
converter = importlib.util.module_from_spec(module_specification)
sys.modules[module_specification.name] = converter
module_specification.loader.exec_module(converter)


class BvhToG1CsvTest(unittest.TestCase):
    def test_parses_nokov_skeleton_and_motion_metadata(self) -> None:
        motion = converter.load_bvh(BVH_PATH)

        self.assertEqual(motion.frame_values.shape, (579, 72))
        self.assertEqual(len(motion.joints), 23)
        self.assertAlmostEqual(motion.frames_per_second, 120.0, places=3)
        root_joint = motion.joints[0]
        self.assertEqual(root_joint.name, "Hips")
        self.assertEqual(
            root_joint.channels,
            (
                "Xposition",
                "Yposition",
                "Zposition",
                "Yrotation",
                "Xrotation",
                "Zrotation",
            ),
        )

    def test_composes_rotations_in_declared_order(self) -> None:
        actual_rotation = converter.compose_channel_rotation(
            ("Yrotation", "Xrotation", "Zrotation"),
            np.array([30.0, 20.0, 10.0]),
        )
        expected_rotation = (
            converter.Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
            @ converter.Rotation.from_euler("x", 20.0, degrees=True).as_matrix()
            @ converter.Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
        )

        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1.0e-12)

    def test_loads_all_controlled_g1_joints_and_soft_limits(self) -> None:
        model = converter.UrdfKinematicModel(URDF_PATH, converter.G1_JOINT_NAMES)

        self.assertEqual(model.lower_limits.shape, (29,))
        self.assertEqual(model.upper_limits.shape, (29,))
        self.assertTrue(np.all(model.soft_lower_limits > model.lower_limits))
        self.assertTrue(np.all(model.soft_upper_limits < model.upper_limits))

        transforms = model.forward_kinematics(
            converter.G1_DEFAULT_JOINT_POSITIONS,
            converter.DEFAULT_G1_ROOT_POSITION,
            np.eye(3),
        )
        for required_link_name in (
            "pelvis",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ):
            self.assertIn(required_link_name, transforms)


if __name__ == "__main__":
    unittest.main()
