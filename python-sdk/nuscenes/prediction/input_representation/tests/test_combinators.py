# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import unittest

import cv2
import numpy as np

from nuscenes.prediction.input_representation.combinators import Rasterizer


class TestRasterizer(unittest.TestCase):

    def test(self):

        layer_1 = np.zeros((100, 100, 3))
        box_1 = cv2.boxPoints(((50, 50), (20, 20), 0))
        layer_1 = cv2.fillPoly(layer_1, pts=[np.int0(box_1)], color=(255, 255, 255))

        layer_2 = np.zeros((100, 100, 3))
        box_2 = cv2.boxPoints(((70, 30), (10, 10), 0))
        layer_2 = cv2.fillPoly(layer_2, pts=[np.int0(box_2)], color=(0, 0, 255))

        rasterizer = Rasterizer()
        image = rasterizer.combine([layer_1.astype('uint8'), layer_2.astype('uint8')])

        answer = np.zeros((100, 100, 3))
        answer = cv2.fillPoly(answer, pts=[np.int0(box_1)], color=(255, 255, 255))
        answer = cv2.fillPoly(answer, pts=[np.int0(box_2)], color=(0, 0, 255))
        answer = answer.astype('uint8')

        np.testing.assert_allclose(answer, image)
