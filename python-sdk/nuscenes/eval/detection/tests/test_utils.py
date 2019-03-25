# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pyquaternion import Quaternion

from nuscenes.eval.detection.data_classes import EvalBox
from nuscenes.eval.detection.utils import attr_acc, scale_iou, yaw_diff, angle_diff, center_distance, velocity_l2, \
    cummean


class TestEval(unittest.TestCase):
    def test_scale_iou(self):
        """Test valid and invalid inputs for scale_iou()."""

        # Identical boxes.
        sa = EvalBox(size=(4, 4, 4))
        sr = EvalBox(size=(4, 4, 4))
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1)

        # SA is bigger.
        sa = EvalBox(size=(2, 2, 2))
        sr = EvalBox(size=(1, 1, 1))
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1/8)

        # SR is bigger.
        sa = EvalBox(size=(1, 1, 1))
        sr = EvalBox(size=(2, 2, 2))
        res = scale_iou(sa, sr)
        self.assertEqual(res, 1/8)

        # Arbitrary values.
        sa = EvalBox(size=(0.96, 0.37, 0.69))
        sr = EvalBox(size=(0.32, 0.01, 0.39))
        res = scale_iou(sa, sr)
        self.assertAlmostEqual(res, 0.00509204)

        # One empty box.
        sa = EvalBox(size=(0, 4, 4))
        sr = EvalBox(size=(4, 4, 4))
        self.assertRaises(AssertionError, scale_iou, sa, sr)

        # Two empty boxes.
        sa = EvalBox(size=(0, 4, 4))
        sr = EvalBox(size=(4, 0, 4))
        self.assertRaises(AssertionError, scale_iou, sa, sr)

        # Negative sizes.
        sa = EvalBox(size=(4, 4, 4))
        sr = EvalBox(size=(4, -5, 4))
        self.assertRaises(AssertionError, scale_iou, sa, sr)

    def test_yaw_diff(self):
        """Test valid and invalid inputs for yaw_diff()."""

        # Identical rotation.
        sa = EvalBox(rotation=Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        sr = EvalBox(rotation=Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, 0)

        # Rotation around another axis.
        sa = EvalBox(rotation=Quaternion(axis=(0, 0, 1), angle=np.pi/8).elements)
        sr = EvalBox(rotation=Quaternion(axis=(0, 1, 0), angle=np.pi/8).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, np.pi/8)

        # Misc sr yaws for fixed sa yaw.
        q0 = Quaternion(axis=(0, 0, 1), angle=0)
        sa = EvalBox(rotation=q0.elements)
        for yaw_in in np.linspace(-10, 10, 100):
            q1 = Quaternion(axis=(0, 0, 1), angle=yaw_in)
            sr = EvalBox(rotation=q1.elements)
            diff = yaw_diff(sa, sr)
            yaw_true = yaw_in % (2 * np.pi)
            if yaw_true > np.pi:
                yaw_true = 2 * np.pi - yaw_true
            self.assertAlmostEqual(diff, yaw_true)

        # Rotation beyond pi.
        sa = EvalBox(rotation=Quaternion(axis=(0, 0, 1), angle=1.1 * np.pi).elements)
        sr = EvalBox(rotation=Quaternion(axis=(0, 0, 1), angle=0.9 * np.pi).elements)
        diff = yaw_diff(sa, sr)
        self.assertAlmostEqual(diff, 0.2 * np.pi)

    def test_angle_diff(self):
        """Test valid and invalid inputs for angle_diff()."""
        def rad(x):
            return x/180*np.pi

        a = 90.0
        b = 0.0
        period = 360
        self.assertAlmostEqual(rad(90), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 90.0
        b = 0.0
        period = 180
        self.assertAlmostEqual(rad(90), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 90.0
        b = 0.0
        period = 90
        self.assertAlmostEqual(rad(0), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 0.0
        b = 90.0
        period = 90
        self.assertAlmostEqual(rad(0), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 0.0
        b = 180.0
        period = 180
        self.assertAlmostEqual(rad(0), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 0.0
        b = 180.0
        period = 360
        self.assertAlmostEqual(rad(180), abs(angle_diff(rad(a), rad(b), rad(period))))

        a = 0.0
        b = 180.0 + 360*200
        period = 360
        self.assertAlmostEqual(rad(180), abs(angle_diff(rad(a), rad(b), rad(period))))

    def test_center_distance(self):
        """Test for center_distance()."""

        # Same boxes.
        sa = EvalBox(translation=(4, 4, 5))
        sr = EvalBox(translation=(4, 4, 5))
        self.assertAlmostEqual(center_distance(sa, sr), 0)

        # When no translation given
        sa = EvalBox(size=(4, 4, 4))
        sr = EvalBox(size=(3, 3, 3))
        self.assertAlmostEqual(center_distance(sa, sr), 0)

        # Different z translation (z should be ignored).
        sa = EvalBox(translation=(4, 4, 4))
        sr = EvalBox(translation=(3, 3, 3))
        self.assertAlmostEqual(center_distance(sa, sr), np.sqrt((3 - 4) ** 2 + (3 - 4) ** 2))

        # Negative values.
        sa = EvalBox(translation=(-1, -1, -1))
        sr = EvalBox(translation=(1, 1, 1))
        self.assertAlmostEqual(center_distance(sa, sr), np.sqrt((1 + 1) ** 2 + (1 + 1) ** 2))

        # Arbitrary values.
        sa = EvalBox(translation=(4.2, 2.8, 4.2))
        sr = EvalBox(translation=(-1.45, 3.5, 3.9))
        self.assertAlmostEqual(center_distance(sa, sr), np.sqrt((-1.45 - 4.2) ** 2 + (3.5 - 2.8) ** 2))

    def test_velocity_l2(self):
        """Test for velocity_l2()."""

        # Same velocity.
        sa = EvalBox(velocity=(4, 4))
        sr = EvalBox(velocity=(4, 4))
        self.assertAlmostEqual(velocity_l2(sa, sr), 0)

        # Negative values.
        sa = EvalBox(velocity=(-1, -1))
        sr = EvalBox(velocity=(1, 1))
        self.assertAlmostEqual(velocity_l2(sa, sr), np.sqrt((1 + 1) ** 2 + (1 + 1) ** 2))

        # Arbitrary values.
        sa = EvalBox(velocity=(8.2, 1.4))
        sr = EvalBox(velocity=(6.4, -9.4))
        self.assertAlmostEqual(velocity_l2(sa, sr), np.sqrt((6.4 - 8.2) ** 2 + (-9.4 - 1.4) ** 2))

    def test_cummean(self):
        """Test for cummean()."""

        # Single NaN.
        x = np.array((np.nan, 5))
        assert_array_almost_equal(cummean(x), np.array((0, 5)))

        x = np.array((5, 2, np.nan))
        assert_array_almost_equal(cummean(x), np.array((5, 3.5, 3.5)))

        # Two NaN values.
        x = np.array((np.nan, 4.5, np.nan))
        assert_array_almost_equal(cummean(x), np.array((0, 4.5, 4.5)))

        # All NaN values.
        x = np.array((np.nan, np.nan, np.nan, np.nan))
        assert_array_almost_equal(cummean(x), np.array((1, 1, 1, 1)))

        # Single value array.
        x = np.array([np.nan])
        assert_array_almost_equal(cummean(x), np.array([1]))
        x = np.array([4])
        assert_array_almost_equal(cummean(x), np.array([4.0]))

        # Arbitrary values.
        x = np.array((np.nan, 3.58, 2.14, np.nan, 9, 1.48, np.nan))
        assert_array_almost_equal(cummean(x), np.array((0, 3.58, 2.86, 2.86, 4.906666, 4.05, 4.05)))

    def test_attr_acc(self):
        """Test for attr_acc()."""

        # Same attributes.
        sa = EvalBox(attribute_name='vehicle.parked')
        sr = EvalBox(attribute_name='vehicle.parked')
        self.assertAlmostEqual(attr_acc(sa, sr), 1.0)

        # Different attributes.
        sa = EvalBox(attribute_name='vehicle.parked')
        sr = EvalBox(attribute_name='vehicle.moving')
        self.assertAlmostEqual(attr_acc(sa, sr), 0.0)

        # No attribute in one.
        sa = EvalBox(attribute_name='')
        sr = EvalBox(attribute_name='vehicle.parked')
        self.assertIs(attr_acc(sa, sr), np.nan)


if __name__ == '__main__':
    unittest.main()
