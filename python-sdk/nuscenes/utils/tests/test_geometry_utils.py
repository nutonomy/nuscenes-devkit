# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import unittest

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import quaternion_yaw


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


if __name__ == '__main__':
    unittest.main()
