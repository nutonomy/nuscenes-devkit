# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import unittest

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import scale_iou, quaternion_yaw, yaw_diff
from nuscenes.eval.detection.data_classes import EvalBox


class TestEval(unittest.TestCase):
    def test_scale_iou(self):
        """Test valid and invalid inputs for scale_iou()."""

        # Identical boxes.
        sa = EvalBox(size=[4, 4, 4])
        sr = EvalBox(size=[4, 4, 4])
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1)

        # SA is bigger.
        sa = EvalBox(size=[2, 2, 2])
        sr = EvalBox(size=[1, 1, 1])
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1/8)

        # SR is bigger.
        sa = EvalBox(size=[1, 1, 1])
        sr = EvalBox(size=[2, 2, 2])
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1/8)

        # Arbitrary values.
        sa = EvalBox(size=[0.96, 0.37, 0.69])
        sr = EvalBox(size=[0.32, 0.01, 0.39])
        res = scale_iou(sa, sr)
        self.assertAlmostEqual(res, 0.00509204)

        # One empty box.
        sa = EvalBox(size=[0, 4, 4])
        sr = EvalBox(size=[4, 4, 4])
        self.assertRaises(AssertionError, scale_iou, sa, sr)

        # Two empty boxes.
        sa = EvalBox(size=[0, 4, 4])
        sr = EvalBox(size=[4, 0, 4])
        self.assertRaises(AssertionError, scale_iou, sa, sr)

        # Negative sizes.
        sa = EvalBox(size=[4, 4, 4])
        sr = EvalBox(size=[4, -5, 4])
        self.assertRaises(AssertionError, scale_iou, sa, sr)

    def test_quaternion_yaw(self):
        """Test valid and invalid inputs for quaternion_yaw()."""

        # Misc yaws.
        for yaw_in in np.linspace(-10, 10, 100):
            q = Quaternion(axis=(0, 0, 1), angle=yaw_in)
            yaw_true = yaw_in % (2 * np.pi)
            if yaw_true > np.pi:
                yaw_true -= 2 * np.pi
            yaw_test = quaternion_yaw(q)
            self.assertAlmostEqual(yaw_true, yaw_test)

        # Non unit axis vector.
        yaw_in = np.pi/4
        q = Quaternion(axis=(0, 0, 0.5), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Inverted axis vector.
        yaw_in = np.pi/4
        q = Quaternion(axis=(0, 0, -1), angle=yaw_in)
        yaw_test = -quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Rotate around another axis.
        yaw_in = np.pi/4
        q = Quaternion(axis=(0, 1, 0), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(0, yaw_test)

        # Rotate around two axes jointly.
        yaw_in = np.pi/2
        q = Quaternion(axis=(0, 1, 1), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Rotate around two axes separately.
        yaw_in = np.pi/2
        q = Quaternion(axis=(0, 0, 1), angle=yaw_in) * Quaternion(axis=(0, 1, 0), angle=0.5821)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

    def test_yaw_diff(self):
        """Test valid and invalid inputs for yaw_diff()."""

        # Identical rotation.
        sa = EvalBox(rotation= Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        sr = EvalBox(rotation= Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, 0)

        # Rotation around another axis.
        sa = EvalBox(rotation= Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        sr = EvalBox(rotation= Quaternion(axis=(0, 1, 0), angle=np.pi/8).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, np.pi/8)

        # Misc sr yaws for fixed sa yaw.
        q0 = Quaternion(axis=(0, 0, 1), angle=0)
        sa = EvalBox(rotation= q0.elements)
        for yaw_in in np.linspace(-10, 10, 100):
            q1 = Quaternion(axis=(0, 0, 1), angle=yaw_in)
            sr = EvalBox(rotation= q1.elements)
            diff = yaw_diff(sa, sr)
            yaw_true = yaw_in % (2 * np.pi)
            if yaw_true > np.pi:
                yaw_true = 2 * np.pi - yaw_true
            self.assertAlmostEqual(diff, yaw_true)

        # Rotation beyond pi.
        sa = EvalBox(rotation= Quaternion(axis=(0, 0, 1), angle=1.1 * np.pi).elements)
        sr = EvalBox(rotation= Quaternion(axis=(0, 0, 1), angle=0.9 * np.pi).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, 0.2 * np.pi)


if __name__ == '__main__':
    unittest.main()
