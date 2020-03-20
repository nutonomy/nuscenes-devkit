import unittest

import math
import numpy as np

from nuscenes.map_expansion import arcline_path_utils


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.straight_path = {'start_pose': [421.2419602954602, 1087.9127960414617, 2.739593514975998],
                              'end_pose': [391.7142849867393, 1100.464077182952, 2.7365754617298705],
                              'shape': 'LSR',
                              'radius': 999.999,
                              'segment_length': [0.23651121617864976, 28.593481378991886, 3.254561444252876]}
        self.left_path = {'start_pose': [391.7142849867393, 1100.464077182952, 2.7365754617298705],
                          'end_pose': [372.7733659833846, 1093.0160135871615, -2.000208580915862],
                          'shape': 'LSL',
                          'radius': 14.473414516079979,
                          'segment_length': [22.380622583127813, 0.18854612175175053, 0.0010839266609007578]}
        self.right_path = {'start_pose': [367.53376358458553, 1097.5300417399676, 1.1738120532326812],
                           'end_pose': [392.24904359636037, 1112.5206834496375, -0.4033046016493182],
                           'shape': 'RSR',
                           'radius': 16.890467008945414,
                           'segment_length': [4.423187697943063e-05, 6.490596454713637, 26.63819259666578]}

        self.straight_lane = [self.straight_path]
        self.curved_lane = [self.straight_path, self.left_path]
        self.right_lane = [self.right_path]

    def test_discretize_straight_path(self):

        discrete_path = arcline_path_utils.discretize(self.straight_path, 10)
        answer = np.array([(421.2419602954602, 1087.9127960414617, 2.739593514975998),
                           (413.85953060356087, 1091.049417600379, 2.739830026428688),
                           (406.4770899726762, 1094.1860134184205, 2.739830026428688),
                           (399.0946493417915, 1097.322609236462, 2.739830026428688),
                           (391.71428498673856, 1100.4640771829522, 2.7365754617298705)])

        np.testing.assert_allclose(answer, discrete_path)

    def test_discretize_curved_path(self):

        discrete_path = arcline_path_utils.discretize(self.left_path, 2)
        answer = np.array([(391.7142849867393, 1100.464077182952, 2.7365754617298705),
                           (389.94237388555354, 1101.0909492468568, 2.8665278225823894),
                           (388.10416900705434, 1101.4829190922167, 2.996480183434908),
                           (386.23066958739906, 1101.633376593063, 3.126432544287426),
                           (384.3534700650694, 1101.539784454639, -3.026800402039642),
                           (382.50422727657343, 1101.2037210019917, -2.8968480411871234),
                           (380.714126599876, 1100.630853563314, -2.7668956803346045),
                           (379.01335604844144, 1099.830842896896, -2.6369433194820857),
                           (377.4305971846951, 1098.8171802734153, -2.506990958629568),
                           (375.99254143806974, 1097.6069599609898, -2.377038597777049),
                           (374.7234399843828, 1096.220590949774, -2.24708623692453),
                           (373.64469477731785, 1094.6814527775348, -2.117133876072012),
                           (372.7733659833847, 1093.0160135871613, -2.0002085809158623)])

        np.testing.assert_allclose(answer, discrete_path)

    def test_discretize_curved_lane(self):

        discrete_path = arcline_path_utils.discretize_lane(self.curved_lane, 5)
        answer = np.array([(421.2419602954602, 1087.9127960414617, 2.739593514975998),
                           (417.0234337310829, 1089.7051622497897, 2.739830026428688),
                           (412.80489622772023, 1091.497502717242, 2.739830026428688),
                           (408.5863587243576, 1093.2898431846943, 2.739830026428688),
                           (404.3678212209949, 1095.0821836521468, 2.739830026428688),
                           (400.1492837176322, 1096.874524119599, 2.739830026428688),
                           (395.93074621426956, 1098.6668645870514, 2.739830026428688),
                           (391.71428498673856, 1100.4640771829522, 2.7365754617298705),
                           (391.7142849867393, 1100.464077182952, 2.7365754617298705),
                           (387.35724292592613, 1101.5723176767192, 3.048461127775915),
                           (382.87033132963325, 1101.2901176788932, -2.922838513357627),
                           (378.6864775951582, 1099.6447057425564, -2.610952847311582),
                           (375.20936805976606, 1096.7948422737907, -2.2990671812655377),
                           (372.7733659833847, 1093.0160135871613, -2.0002085809158623)])
        np.testing.assert_allclose(answer, discrete_path)

    def test_length_of_lane(self):

        self.assertEqual(arcline_path_utils.length_of_lane(self.straight_lane),
                         sum(self.straight_path['segment_length']))

        self.assertEqual(arcline_path_utils.length_of_lane(self.right_lane),
                         sum(self.right_path['segment_length']))

        self.assertEqual(arcline_path_utils.length_of_lane(self.curved_lane),
                         sum(self.straight_path['segment_length']) + sum(self.left_path['segment_length']))

    def test_project_pose_to_straight_lane(self):

        theta = 2.739593514975998
        end_pose = 421.2419602954602 + 10 * math.cos(theta), 1087.9127960414617 + 10 * math.sin(theta), theta

        pose, s = arcline_path_utils.project_pose_to_lane(end_pose, self.straight_lane)

        np.testing.assert_allclose(np.array(pose).astype('int'), np.array(end_pose).astype('int'))
        self.assertTrue(abs(s - 10) <= 0.5)

    def test_project_pose_not_close_to_lane(self):

        pose = 362, 1092, 1.15

        pose_on_lane, s = arcline_path_utils.project_pose_to_lane(pose, self.right_lane)
        self.assertListEqual(list(pose_on_lane), self.right_path['start_pose'])
        self.assertEqual(s, 0)

    def test_project_pose_to_curved_lane(self):

        theta = 2.739593514975998
        end_pose_1 = 421.2419602954602 + 10 * math.cos(theta), 1087.9127960414617 + 10 * math.sin(theta), theta

        end_pose_2 = 381, 1100, -2.76

        pose, s = arcline_path_utils.project_pose_to_lane(end_pose_1, self.curved_lane)
        np.testing.assert_allclose(np.array(pose).astype('int'), np.array(end_pose_1).astype('int'))
        self.assertTrue(abs(s - 10) <= 0.5)

        pose_2, s_2 = arcline_path_utils.project_pose_to_lane(end_pose_2, self.curved_lane)
        np.testing.assert_allclose(np.array(pose_2[:2]).astype('int'), np.array([380, 1100]))
        self.assertTrue(abs(s_2 - 44) <= 0.5)

    def test_get_curvature_straight_lane(self):

        curvature = arcline_path_utils.get_curvature_at_distance_along_lane(15, self.straight_lane)
        self.assertEqual(curvature, 0)

    def test_curvature_curved_lane(self):

        curvature = arcline_path_utils.get_curvature_at_distance_along_lane(53, self.curved_lane)
        self.assertEqual(curvature, 1 / self.left_path['radius'])
