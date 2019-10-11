# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import os
import unittest

import cv2
import numpy as np

from nuscenes.utils.map_mask import MapMask


class TestLoad(unittest.TestCase):

    fixture = 'testmap.png'
    foreground = 255
    native_res = 0.1  # Maps defined on a 0.1 meter resolution grid.
    small_number = 0.00001  # Just a small numbers to avoid edge effects.
    half_gt = native_res / 2 + small_number  # Just larger than half a cell.
    half_lt = native_res / 2 - small_number  # Just smaller than half a cell.

    def setUp(self):

        # Build a test map. 5 x 4 meters. All background except one pixel.
        mask = np.zeros((50, 40))

        # Native resolution is 0.1
        # Transformation in y is defined as y_pixel = nrows - y_meters /  resolution
        # Transformation in x is defined as x_pixel = x_meters /  resolution
        # The global map location x=2, y=2 becomes row 30, column 20 in image map coords.
        mask[30, 20] = self.foreground
        cv2.imwrite(filename=self.fixture, img=mask)

    def tearDown(self):
        os.remove(self.fixture)

    def test_native_resolution(self):

        # Load mask and assert that the
        map_mask = MapMask(self.fixture, resolution=0.1)

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assertTrue(map_mask.is_on_mask(2, 2))

        # Each pixel is 10 x 10 cm, so if we step less than 5 cm in either direction we are still on foreground.
        # Note that we add / subtract a "small number" to break numerical ambiguities along the edges.
        self.assertTrue(map_mask.is_on_mask(2 + self.half_lt, 2))
        self.assertTrue(map_mask.is_on_mask(2 - self.half_lt, 2))
        self.assertTrue(map_mask.is_on_mask(2, 2 + self.half_lt))
        self.assertTrue(map_mask.is_on_mask(2, 2 - self.half_lt))

        # But if we step outside this range, we should get false
        self.assertFalse(map_mask.is_on_mask(2 + self.half_gt, 2))
        self.assertFalse(map_mask.is_on_mask(2 + self.half_gt, 2))
        self.assertFalse(map_mask.is_on_mask(2, 2 + self.half_gt))
        self.assertFalse(map_mask.is_on_mask(2, 2 + self.half_gt))

    def test_edges(self):

        # Add foreground pixels in the corners for this test.
        mask = np.ones((50, 40)) * self.foreground

        # Just over-write the fixture
        cv2.imwrite(filename=self.fixture, img=mask)

        map_mask = MapMask(self.fixture, resolution=0.1)

        # Asssert that corners are indeed drivable as encoded in map.
        self.assertTrue(map_mask.is_on_mask(0, 0.1))
        self.assertTrue(map_mask.is_on_mask(0, 5))
        self.assertTrue(map_mask.is_on_mask(3.9, 0.1))
        self.assertTrue(map_mask.is_on_mask(3.9, 5))

        # Not go juuuust outside the map. This should no longer be drivable.
        self.assertFalse(map_mask.is_on_mask(3.9 + self.half_gt, 0.1))
        self.assertFalse(map_mask.is_on_mask(3.9 + self.half_gt, 5))
        self.assertFalse(map_mask.is_on_mask(0 - self.half_gt, 0.1))
        self.assertFalse(map_mask.is_on_mask(0 - self.half_gt, 5))

    def test_dilation(self):

        map_mask = MapMask(self.fixture, resolution=0.1)

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assertTrue(map_mask.is_on_mask(2, 2))

        # Go 1 meter to the right. Obviously not on the mask.
        self.assertFalse(map_mask.is_on_mask(2, 3))

        # But if we dilate by 1 meters, we are on the dilated mask.
        self.assertTrue(map_mask.is_on_mask(2, 3, dilation=1))  # x direction
        self.assertTrue(map_mask.is_on_mask(3, 2, dilation=1))  # y direction
        self.assertTrue(map_mask.is_on_mask(2 + np.sqrt(1/2), 2 + np.sqrt(1/2), dilation=1))  # diagonal

        # If we dilate by 0.9 meter, it is not enough.
        self.assertFalse(map_mask.is_on_mask(2, 3, dilation=0.9))

    def test_coarse_resolution(self):

        # Due to resize that happens on load we need to inflate the fixture.
        mask = np.zeros((50, 40))
        mask[30, 20] = self.foreground
        mask[31, 20] = self.foreground
        mask[30, 21] = self.foreground
        mask[31, 21] = self.foreground

        # Just over-write the fixture
        cv2.imwrite(filename=self.fixture, img=mask)

        map_mask = MapMask(self.fixture, resolution=0.2)

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assertTrue(map_mask.is_on_mask(2, 2))

        # Go two meters to the right. Obviously not on the mask.
        self.assertFalse(map_mask.is_on_mask(2, 4))

        # But if we dilate by two meters, we are on the dilated mask.
        self.assertTrue(map_mask.is_on_mask(2, 4, dilation=2))

        # And if we dilate by 1.9 meter, we are off the dilated mask.
        self.assertFalse(map_mask.is_on_mask(2, 4, dilation=1.9))


if __name__ == '__main__':
    unittest.main()
