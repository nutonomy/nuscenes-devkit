# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import copy
import unittest
from typing import Dict, List, Any

import numpy as np

from nuscenes.prediction import PredictHelper, convert_global_coords_to_local, convert_local_coords_to_global


class MockNuScenes:
    """Mocks the NuScenes API needed to test PredictHelper"""

    def __init__(self, sample_annotations: List[Dict[str, Any]],
                 samples: List[Dict[str, Any]]):

        self._sample_annotation = {r['token']: r for r in sample_annotations}
        self._sample = {r['token']: r for r in samples}

    @property
    def sample_annotation(self,) -> List[Dict[str, Any]]:
        return list(self._sample_annotation.values())

    def get(self, table_name: str, token: str) -> Dict[str, Any]:
        assert table_name in {'sample_annotation', 'sample'}
        return getattr(self, "_" + table_name)[token]

class Test_convert_coords(unittest.TestCase):

    def setUp(self):
        along_pos_x = np.zeros((5, 2))
        along_pos_y = np.zeros((5, 2))
        along_neg_x = np.zeros((5, 2))
        along_neg_y = np.zeros((5, 2))

        along_pos_x[:, 0] = np.arange(1, 6)
        along_pos_y[:, 1] = np.arange(1, 6)
        along_neg_x[:, 0] = -np.arange(1, 6)
        along_neg_y[:, 1] = -np.arange(1, 6)
        self.along_pos_x, self.along_pos_y = along_pos_x, along_pos_y
        self.along_neg_x, self.along_neg_y = along_neg_x, along_neg_y

        y_equals_x = np.zeros((5, 2))
        y_equals_x[:, 0] = np.arange(1, 6)
        y_equals_x[:, 1] = np.arange(1, 6)
        self.y_equals_x = y_equals_x

    def test_heading_0(self):
        rotation = (1, 0, 0, 0)
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)

    def test_heading_pi_over_4(self):
        rotation = (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * np.sqrt(2) / 2, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * np.sqrt(2) / 2, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer,  self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer,  self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer,  self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer,  self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)

    def test_heading_pi_over_2(self):
        rotation = (np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x,  atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)

    def test_heading_3pi_over_4(self):
        rotation = (np.cos(3 * np.pi / 8), 0, 0, np.sin(3 * np.pi / 8))
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]],  atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)

    def test_heading_pi(self):
        rotation = (np.cos(np.pi / 2), 0, 0, np.sin(np.pi / 2))
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_y,  atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_pos_y, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.along_neg_x, atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)

    def test_heading_neg_pi_over_4(self):
        rotation = (np.cos(-np.pi / 8), 0, 0, np.sin(-np.pi / 8))
        origin = (0, 0, 0)
        offset = (50, 25, 0)

        # Testing path along pos x direction
        answer = convert_global_coords_to_local(self.along_pos_x, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, np.sqrt(2) / 2]],  atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_x + [[50, 25]], atol=1e-4)

        # Testing path along pos y direction
        answer = convert_global_coords_to_local(self.along_pos_y, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_pos_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_pos_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[-np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_pos_y + [[50, 25]], atol=1e-4)

        # Testing path along neg x direction
        answer = convert_global_coords_to_local(self.along_neg_x, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_x, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_x + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, -np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_x + [[50, 25]], atol=1e-4)

        # Testing path along neg y direction
        answer = convert_global_coords_to_local(self.along_neg_y, origin, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, origin, rotation),
                                   self.along_neg_y, atol=1e-4)

        answer = convert_global_coords_to_local(self.along_neg_y + [[50, 25]], offset, rotation)
        np.testing.assert_allclose(answer, self.y_equals_x * [[np.sqrt(2) / 2, np.sqrt(2) / 2]], atol=1e-4)
        np.testing.assert_allclose(convert_local_coords_to_global(answer, offset, rotation),
                                   self.along_neg_y + [[50, 25]], atol=1e-4)


class TestPredictHelper(unittest.TestCase):

    def setUp(self):

        self.mock_annotations = [{'token': '1', 'instance_token': '1', 'sample_token': '1', 'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0],
                                  'prev': '', 'next': '2'},
                                 {'token': '2', 'instance_token': '1', 'sample_token': '2', 'translation': [1, 1, 1], 'rotation': [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                                  'prev': '1', 'next': '3'},
                                 {'token': '3', 'instance_token': '1', 'sample_token': '3', 'translation': [2, 2, 2], 'prev': '2', 'next': '4'},
                                 {'token': '4', 'instance_token': '1', 'sample_token': '4', 'translation': [3, 3, 3], 'prev': '3', 'next': '5'},
                                 {'token': '5', 'instance_token': '1', 'sample_token': '5', 'translation': [4, 4, 4], 'rotation': [1, 0, 0, 0],
                                  'prev': '4', 'next': '6'},
                                 {'token': '6', 'instance_token': '1', 'sample_token': '6', 'translation': [5, 5, 5], 'prev': '5', 'next': ''}]

        self.multiagent_mock_annotations = [{'token': '1', 'instance_token': '1', 'sample_token': '1', 'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0],
                                             'prev': '', 'next': '2'},
                                            {'token': '2', 'instance_token': '1', 'sample_token': '2', 'translation': [1, 1, 1], 'prev': '1', 'next': '3'},
                                            {'token': '3', 'instance_token': '1', 'sample_token': '3', 'translation': [2, 2, 2], 'prev': '2', 'next': '4'},
                                            {'token': '4', 'instance_token': '1', 'sample_token': '4', 'translation': [3, 3, 3], 'prev': '3', 'next': '5'},
                                            {'token': '5', 'instance_token': '1', 'sample_token': '5', 'translation': [4, 4, 4], 'rotation': [1, 0, 0, 0],
                                             'prev': '4', 'next': '6'},
                                            {'token': '6', 'instance_token': '1', 'sample_token': '6', 'translation': [5, 5, 5], 'prev': '5', 'next': ''},
                                            {'token': '1b', 'instance_token': '2', 'sample_token': '1', 'translation': [6, 6, 6], 'rotation': [1, 0, 0, 0],
                                             'prev': '', 'next': '2b'},
                                            {'token': '2b', 'instance_token': '2', 'sample_token': '2', 'translation': [7, 7, 7], 'prev': '1b', 'next': '3b'},
                                            {'token': '3b', 'instance_token': '2', 'sample_token': '3', 'translation': [8, 8, 8], 'prev': '2b', 'next': '4b'},
                                            {'token': '4b', 'instance_token': '2', 'sample_token': '4', 'translation': [9, 9, 9], 'prev': '3b', 'next': '5b'},
                                            {'token': '5b', 'instance_token': '2', 'sample_token': '5', 'translation': [10, 10, 10], 'rotation': [1, 0, 0, 0],
                                            'prev': '4b', 'next': '6b'},
                                            {'token': '6b', 'instance_token': '2', 'sample_token': '6', 'translation': [11, 11, 11], 'prev': '5b', 'next': ''}]


    def test_get_sample_annotation(self,):

        mock_annotation = {'token': '1', 'instance_token': 'instance_1',
                           'sample_token': 'sample_1'}
        mock_sample = {'token': 'sample_1', 'timestamp': 0}

        nusc = MockNuScenes([mock_annotation], [mock_sample])

        helper = PredictHelper(nusc)
        self.assertDictEqual(mock_annotation, helper.get_sample_annotation('instance_1', 'sample_1'))


    def test_get_future_for_agent_exact_amount(self,):

        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 1e6},
                        {'token': '3', 'timestamp': 2e6},
                        {'token': '4', 'timestamp': 3e6},
                        {'token': '5', 'timestamp': 4e6}]

        # Testing we can get the exact amount of future seconds available
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '1', 3, False)
        np.testing.assert_equal(future, np.array([[1, 1], [2, 2], [3, 3]]))

    def test_get_future_for_agent_in_agent_frame(self):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 1e6},
                        {'token': '3', 'timestamp': 2e6},
                        {'token': '4', 'timestamp': 3e6},
                        {'token': '5', 'timestamp': 4e6}]

        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '1', 3, True)
        np.testing.assert_allclose(future, np.array([[-1, 1], [-2, 2], [-3, 3]]))

    def test_get_future_for_agent_less_amount(self,):

        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 1e6},
                        {'token': '3', 'timestamp': 2.6e6},
                        {'token': '4', 'timestamp': 4e6},
                        {'token': '5', 'timestamp': 5.5e6}]

        # Testing we do not include data after the future seconds
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '1', 3, False)
        np.testing.assert_equal(future, np.array([[1, 1], [2, 2]]))

    def test_get_future_for_agent_within_buffer(self,):

        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 1e6},
                        {'token': '3', 'timestamp': 2.6e6},
                        {'token': '4', 'timestamp': 3.05e6},
                        {'token': '5', 'timestamp': 3.5e6}]

        # Testing we get data if it is after future seconds but within buffer
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '1', 3, False)
        np.testing.assert_equal(future, np.array([[1, 1], [2, 2], [3, 3]]))

    def test_get_future_for_agent_no_data_to_get(self,):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 3.5e6}]

        # Testing we get nothing if the first sample annotation is past our threshold
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '1', 3, False)
        np.testing.assert_equal(future, np.array([]))

    def test_get_future_for_last_returns_nothing(self):
        mock_samples = [{'token': '6', 'timestamp': 0}]

        # Testing we get nothing if we're at the last annotation
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_agent('1', '6', 3, False)
        np.testing.assert_equal(future, np.array([]))

    def test_get_past_for_agent_exact_amount(self,):

        mock_samples = [{'token': '5', 'timestamp': 0},
                        {'token': '4', 'timestamp': -1e6},
                        {'token': '3', 'timestamp': -2e6},
                        {'token': '2', 'timestamp': -3e6},
                        {'token': '1', 'timestamp': -4e6}]

        # Testing we can get the exact amount of past seconds available
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '5', 3, False)
        np.testing.assert_equal(past, np.array([[3, 3], [2, 2], [1, 1]]))

    def test_get_past_for_agent_in_frame(self,):

        mock_samples = [{'token': '5', 'timestamp': 0},
                        {'token': '4', 'timestamp': -1e6},
                        {'token': '3', 'timestamp': -2e6},
                        {'token': '2', 'timestamp': -3e6},
                        {'token': '1', 'timestamp': -4e6}]

        # Testing we can get the exact amount of past seconds available
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '5', 3, True)
        np.testing.assert_allclose(past, np.array([[1., -1.], [2., -2.], [3., -3.]]))

    def test_get_past_for_agent_less_amount(self,):

        mock_samples = [{'token': '5', 'timestamp': 0},
                        {'token': '4', 'timestamp': -1e6},
                        {'token': '3', 'timestamp': -2.6e6},
                        {'token': '2', 'timestamp': -4e6},
                        {'token': '1', 'timestamp': -5.5e6}]

        # Testing we do not include data after the past seconds
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '5', 3, False)
        np.testing.assert_equal(past, np.array([[3, 3], [2, 2]]))

    def test_get_past_for_agent_within_buffer(self,):

        mock_samples = [{'token': '5', 'timestamp': 0},
                        {'token': '4', 'timestamp': -1e6},
                        {'token': '3', 'timestamp': -3.05e6},
                        {'token': '2', 'timestamp': -3.2e6}]

        # Testing we get data if it is after future seconds but within buffer
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '5', 3, False)
        np.testing.assert_equal(past, np.array([[3, 3], [2, 2]]))

    def test_get_past_for_agent_no_data_to_get(self,):
        mock_samples = [{'token': '5', 'timestamp': 0},
                        {'token': '4', 'timestamp': -3.5e6}]

        # Testing we get nothing if the first sample annotation is past our threshold
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '5', 3, False)
        np.testing.assert_equal(past, np.array([]))

    def test_get_past_for_last_returns_nothing(self):
        mock_samples = [{'token': '1', 'timestamp': 0}]

        # Testing we get nothing if we're at the last annotation
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_agent('1', '1', 3, False)
        np.testing.assert_equal(past, np.array([]))

    def test_get_future_for_sample(self):

        mock_samples = [{'token': '1', 'timestamp': 0, 'anns': ['1', '1b']},
                        {'token': '2', 'timestamp': 1e6},
                        {'token': '3', 'timestamp': 2e6},
                        {'token': '4', 'timestamp': 3e6},
                        {'token': '5', 'timestamp': 4e6}]

        nusc = MockNuScenes(self.multiagent_mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        future = helper.get_future_for_sample("1", 3, False)

        answer = {'1': np.array([[1, 1], [2, 2], [3, 3]]),
                  '2': np.array([[7, 7], [8, 8], [9, 9]])}

        for k in answer:
            np.testing.assert_equal(answer[k], future[k])

        future_in_sample = helper.get_future_for_sample("1", 3, True)

        answer_in_sample = {'1': np.array([[-1, 1], [-2, 2], [-3, 3]]),
                            '2': np.array([[-1, 1], [-2, 2], [-3, 3]])}

        for k in answer_in_sample:
            np.testing.assert_allclose(answer_in_sample[k], future_in_sample[k])

    def test_get_past_for_sample(self):

        mock_samples = [{'token': '5', 'timestamp': 0, 'anns': ['5', '5b']},
                        {'token': '4', 'timestamp': -1e6},
                        {'token': '3', 'timestamp': -2e6},
                        {'token': '2', 'timestamp': -3e6},
                        {'token': '1', 'timestamp': -4e6}]

        nusc = MockNuScenes(self.multiagent_mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        past = helper.get_past_for_sample('5', 3, True)

        answer = {'1': np.array([[-1, -1], [-2, -2], [-3, -3]]),
                  '2': np.array([[-1, -1], [-2, -2], [-3, -3]])}

        for k in answer:
            np.testing.assert_equal(answer[k], answer[k])

    def test_velocity(self):

        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 0.5e6}]

        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)

        self.assertEqual(helper.get_velocity_for_agent("1", "2"), np.sqrt(8))

    def test_velocity_return_nan_one_obs(self):

        mock_samples = [{'token': '1', 'timestamp': 0}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)

        self.assertTrue(np.isnan(helper.get_velocity_for_agent('1', '1')))

    def test_velocity_return_nan_big_diff(self):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 2.5e6}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertTrue(np.isnan(helper.get_velocity_for_agent('1', '2')))

    def test_heading_change_rate(self):
        mock_samples = [{'token': '1', 'timestamp': 0}, {'token': '2', 'timestamp': 0.5e6}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertEqual(helper.get_heading_change_rate_for_agent('1', '2'), np.pi)

    def test_heading_change_rate_near_pi(self):
        mock_samples = [{'token': '1', 'timestamp': 0}, {'token': '2', 'timestamp': 0.5e6}]
        mock_annotations = copy.copy(self.mock_annotations)
        mock_annotations[0]['rotation'] = [np.cos((np.pi - 0.05)/2), 0, 0, np.sin((np.pi - 0.05) / 2)]
        mock_annotations[1]['rotation'] = [np.cos((-np.pi + 0.05)/2), 0, 0, np.sin((-np.pi + 0.05) / 2)]
        nusc = MockNuScenes(mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertAlmostEqual(helper.get_heading_change_rate_for_agent('1', '2'), 0.2)

    def test_acceleration_zero(self):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 0.5e6},
                        {'token': '3', 'timestamp': 1e6}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertEqual(helper.get_acceleration_for_agent('1', '3'), 0)

    def test_acceleration_nonzero(self):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 0.5e6},
                        {'token': '3', 'timestamp': 1e6}]
        mock_annotations = copy.copy(self.mock_annotations)
        mock_annotations[2]['translation'] = [3, 3, 3]
        nusc = MockNuScenes(mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertAlmostEqual(helper.get_acceleration_for_agent('1', '3'), 2 * (np.sqrt(32) - np.sqrt(8)))

    def test_acceleration_nan_not_enough_data(self):
        mock_samples = [{'token': '1', 'timestamp': 0},
                        {'token': '2', 'timestamp': 0.5e6},
                        {'token': '3', 'timestamp': 1e6}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        self.assertTrue(np.isnan(helper.get_acceleration_for_agent('1', '2')))

    def test_get_no_data_when_seconds_0(self):
        mock_samples = [{'token': '1', 'timestamp': 0, 'anns': ['1']}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)

        np.testing.assert_equal(helper.get_future_for_agent('1', '1', 0, False), np.array([]))
        np.testing.assert_equal(helper.get_past_for_agent('1', '1', 0, False), np.array([]))
        np.testing.assert_equal(helper.get_future_for_sample('1', 0, False), np.array([]))
        np.testing.assert_equal(helper.get_past_for_sample('1', 0, False), np.array([]))

    def test_raises_error_when_seconds_negative(self):
        mock_samples = [{'token': '1', 'timestamp': 0, 'anns': ['1', '1b']}]
        nusc = MockNuScenes(self.mock_annotations, mock_samples)
        helper = PredictHelper(nusc)
        with self.assertRaises(ValueError):
            helper.get_future_for_agent('1', '1', -1, False)

        with self.assertRaises(ValueError):
            helper.get_past_for_agent('1', '1', -1, False)

        with self.assertRaises(ValueError):
            helper.get_past_for_sample('1', -1, False)

        with self.assertRaises(ValueError):
            helper.get_future_for_sample('1', -1, False)