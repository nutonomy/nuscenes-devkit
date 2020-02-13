# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import cv2

from nuscenes.predict.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.predict import PredictHelper

class TestStaticLayerRasterizer(unittest.TestCase):

    PATH = 'nuscenes.predict.input_representation.static_layers.{}'

    def get_layer_mocks(self):

        layer_1 = np.zeros((100, 100, 3))
        box = cv2.boxPoints(((50, 50), (20, 10), -90))
        layer_1 = cv2.fillPoly(layer_1, pts=[np.int0(box)], color=(1, 1, 1))
        layer_1 = layer_1[::-1, :, 0]

        layer_2 = np.zeros((100, 100, 3))
        layer_2 = cv2.line(layer_2, (50, 50), (50, 40), color=(1, 0, 0), thickness=2)
        layer_2 = layer_2[::-1, :, 0]

        return [layer_1, layer_2]

    @patch(PATH.format('load_all_maps'))
    def test_make_rasterization(self, mock_load_maps):
        """
        Mainly a smoke test since most of the logic is handled under-the-hood
        by get_map_mask method of nuScenes map API.
        """

        layers = self.get_layer_mocks()
        mock_map_api = MagicMock()
        mock_map_api.get_map_mask.return_value = layers

        mock_maps = {'mock_map_version': mock_map_api}

        mock_load_maps.return_value = mock_maps

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_map_name_from_sample_token.return_value = 'mock_map_version'
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0],
                                                          'rotation': [-np.pi/8, 0, 0, -np.pi/8]}

        static_layers = StaticLayerRasterizer(mock_helper, ['layer_1', 'layer_2'],
                                              [(255, 255, 255), (255, 0, 0)],
                                              resolution=0.1, meters_ahead=5, meters_behind=5,
                                              meters_left=5, meters_right=5)

        image = static_layers.make_representation('foo_instance', 'foo_sample')

        answer = np.zeros((100, 100, 3))
        box = cv2.boxPoints(((50, 50), (20, 10), -90))
        answer = cv2.fillPoly(answer, pts=[np.int0(box)], color=(255, 255, 255))
        answer = cv2.line(answer, (50, 50), (50, 40), color=(255, 0, 0), thickness=2)

        np.testing.assert_allclose(answer, image)
