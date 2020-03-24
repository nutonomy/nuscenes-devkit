# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import make_2d_rotation_matrix
from nuscenes.prediction.input_representation import agents


class Test_get_track_box(unittest.TestCase):


    def test_heading_positive_30(self):

        annotation = {'translation': [0, 0, 0],
                      'rotation': [np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)],
                      'size': [4, 2]}

        ego_center = (0, 0)
        ego_pixels = (50, 50)

        pi_over_six = np.pi / 6

        box = agents.get_track_box(annotation, ego_center, ego_pixels, resolution=1.)

        mat = make_2d_rotation_matrix(pi_over_six)
        coordinates = np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
        answer = mat.dot(coordinates.T).T + ego_pixels
        answer = answer[:, [1, 0]]

        np.testing.assert_allclose(np.sort(answer, axis=0), np.sort(box, axis=0))


    def test_heading_neg_30(self):

        annotation = {'translation': [0, 0, 0],
                      'rotation': [np.cos(-np.pi / 12), 0, 0, np.sin(-np.pi / 12)],
                      'size': [4, 2]}

        ego_center = (0, 0)
        ego_pixels = (50, 50)

        pi_over_six = -np.pi / 6

        box = agents.get_track_box(annotation, ego_center, ego_pixels, resolution=1.)

        mat = make_2d_rotation_matrix(pi_over_six)
        coordinates = np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
        answer = mat.dot(coordinates.T).T + ego_pixels
        answer = answer[:, [1, 0]]

        np.testing.assert_allclose(np.sort(answer, axis=0), np.sort(box, axis=0))


class Test_reverse_history(unittest.TestCase):

    def test(self):

        history = {'instance_1': [{'time': 0}, {'time': -1}, {'time': -2}],
                   'instance_2': [{'time': -1}, {'time': -2}],
                   'instance_3': [{'time': 0}]}

        agent_history = agents.reverse_history(history)

        answer = {'instance_1': [{'time': -2}, {'time': -1}, {'time': 0}],
                  'instance_2': [{'time': -2}, {'time': -1}],
                  'instance_3': [{'time': 0}]}

        self.assertDictEqual(answer, agent_history)


class Test_add_present_time_to_history(unittest.TestCase):

    def test(self):

        current_time = [{'instance_token': 0, 'time': 3},
                        {'instance_token': 1, 'time': 3},
                        {'instance_token': 2, 'time': 3}]

        history = {0: [{'instance_token': 0, 'time': 1},
                       {'instance_token': 0, 'time': 2}],
                   1: [{'instance_token': 1, 'time': 2}]}

        history = agents.add_present_time_to_history(current_time, history)

        answer = {0: [{'instance_token': 0, 'time': 1},
                      {'instance_token': 0, 'time': 2},
                      {'instance_token': 0, 'time': 3}],
                  1: [{'instance_token': 1, 'time': 2},
                      {'instance_token': 1, 'time': 3}],
                  2: [{'instance_token': 2, 'time': 3}]}

        self.assertDictEqual(answer, history)


class Test_fade_color(unittest.TestCase):

    def test_dont_fade_last(self):

        color = agents.fade_color((200, 0, 0), 10, 10)
        self.assertTupleEqual(color, (200, 0, 0))

    def test_first_is_darkest(self):

        color = agents.fade_color((200, 200, 0), 0, 10)
        self.assertTupleEqual(color, (102, 102, 0))


class TestAgentBoxesWithFadedHistory(unittest.TestCase):

    def test_make_representation(self):

        mock_helper = MagicMock(spec=PredictHelper)

        mock_helper.get_past_for_sample.return_value = {0: [{'rotation': [1, 0, 0, 0], 'translation': [-5, 0, 0],
                                                             'size': [2, 4, 0], 'instance_token': 0,
                                                             'category_name': 'vehicle'}],
                                                        1: [{'rotation': [1, 0, 0, 0], 'translation': [5, -5, 0],
                                                             'size': [3, 3, 0], 'instance_token': 1,
                                                             'category_name': 'human'}]}
        mock_helper.get_annotations_for_sample.return_value = [{'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0],
                                                   'size': [2, 4, 0], 'instance_token': 0,
                                                   'category_name': 'vehicle'},
                                                   {'rotation': [1, 0, 0, 0], 'translation': [10, -5, 0],
                                                    'size': [3, 3, 0], 'instance_token': 1,
                                                    'category_name': 'human'}]

        mock_helper.get_sample_annotation.return_value = {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0],
                                                   'size': [2, 4, 0], 'instance_token': 0,
                                                   'category_name': 'vehicle'}

        def get_colors(name):
            if 'vehicle' in name:
                return (255, 0, 0)
            else:
                return (255, 255, 0)

        agent_rasterizer = agents.AgentBoxesWithFadedHistory(mock_helper,
                                                             color_mapping=get_colors)

        img = agent_rasterizer.make_representation(0, 'foo_sample')

        answer = np.zeros((500, 500, 3))

        agent_0_ts_0 = cv2.boxPoints(((250, 450), (40, 20), -90))
        angent_0_ts_1 = cv2.boxPoints(((250, 400), (40, 20), -90))

        agent_1_ts_0 = cv2.boxPoints(((300, 350), (30, 30), -90))
        agent_1_ts_1 = cv2.boxPoints(((300, 300), (30, 30), -90))

        answer = cv2.fillPoly(answer, pts=[np.int0(agent_0_ts_0)], color=(102, 0, 0))
        answer = cv2.fillPoly(answer, pts=[np.int0(angent_0_ts_1)], color=(255, 0, 0))
        answer = cv2.fillPoly(answer, pts=[np.int0(agent_1_ts_0)], color=(102, 102, 0))
        answer = cv2.fillPoly(answer, pts=[np.int0(agent_1_ts_1)], color=(255, 255, 0))

        np.testing.assert_allclose(answer, img)
