# nuScenes dev-kit.
# Code written by Holger Caesar and Alex Lang, 2018.

import unittest

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box


class TestGeometryUtils(unittest.TestCase):

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

    def test_points_in_box(self):
        """ Test the box.in_box method. """

        vel = (np.nan, np.nan, np.nan)

        def qyaw(yaw):
            return Quaternion(axis=(0, 0, 1), angle=yaw)

        # Check points inside box
        box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)

        # Check points outside box
        box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.1, 0.0, 0.0], [0.5, -1.1, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), False)

        # Check corner cases
        box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], qyaw(0.0), 1, 2.0, vel)
        points = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)

        # Check rotation (45 degs) and translation (by [1,1])
        rot = 45
        trans = [1.0, 1.0]
        box = Box([0.0+trans[0], 0.0+trans[1], 0.0], [2.0, 2.0, 0.0], qyaw(rot / 180.0 * np.pi), 1, 2.0, vel)
        points = np.array([[0.70+trans[0], 0.70+trans[1], 0.0], [0.71+1.0, 0.71+1.0, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], False)

        # Check 3d box
        box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)

        # Check wlh factor
        for wlh_factor in [0.5, 1.0, 1.5, 10.0]:
            box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], qyaw(0.0), 1, 2.0, vel)
            points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
            mask = points_in_box(box, points, wlh_factor=wlh_factor)
            self.assertEqual(mask.all(), True)

        for wlh_factor in [0.1, 0.49]:
            box = Box([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], qyaw(0.0), 1, 2.0, vel)
            points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
            mask = points_in_box(box, points, wlh_factor=wlh_factor)
            self.assertEqual(mask[0], True)
            self.assertEqual(mask[1], False)


if __name__ == '__main__':
    unittest.main()
