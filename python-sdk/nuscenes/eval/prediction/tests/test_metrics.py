import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.prediction import metrics
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction import PredictHelper


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.x_one_mode = np.ones((1, 5, 2))
        self.y_one_mode = np.expand_dims(np.arange(10).reshape(5, 2), 0)
        self.p_one_mode = np.array([[1.]])

        x_many_modes = np.repeat(self.x_one_mode, 3, axis=0)
        x_many_modes[0, :] = 0
        x_many_modes[2, :] = 2
        self.x_many_modes = x_many_modes
        self.y_many_modes = np.repeat(self.y_one_mode, 3, axis=0)
        self.p_many_modes = np.array([[0.2, 0.5, 0.3]])

        self.x_many_batches_and_modes = np.repeat(np.expand_dims(self.x_many_modes, 0), 5, axis=0)
        self.y_many_batches_and_modes = np.repeat(np.expand_dims(self.y_many_modes, 0), 5, axis=0)
        self.p_many_batches_and_modes = np.array([[0.2, 0.5, 0.3],
                                                  [0.5, 0.3, 0.2],
                                                  [0.2, 0.3, 0.5],
                                                  [0.3, 0.2, 0.5],
                                                  [0.3, 0.5, 0.2]])

    def test_returns_2d_array_float(self):

        func = lambda x: 2
        value = metrics.returns_2d_array(func)(2)
        np.testing.assert_equal(value, np.array([[2]]))

        func = lambda x: 3.
        value = metrics.returns_2d_array(func)(np.ones((10, 1)))
        np.testing.assert_equal(value, np.array([[3]]))

    def test_returns_2d_array_one_dim(self):

        func = lambda x: np.ones(10)
        value = metrics.returns_2d_array(func)(1)
        np.testing.assert_equal(value, np.ones((1, 10)))

    def test_mean_distances_one_mode(self):

        value = metrics.mean_distances(self.x_one_mode, self.y_one_mode)
        np.testing.assert_allclose(value, np.array([[5.33529]]), atol=1e-4, rtol=1e-4)

    def test_mean_distances_many_modes(self):
        value = metrics.mean_distances(self.x_many_modes, self.y_many_modes)
        np.testing.assert_allclose(value, np.array([[6.45396, 5.33529, 4.49286]]), atol=1e-4, rtol=1e-4)

    def test_mean_distances_many_batches_and_modes(self):
        value = metrics.mean_distances(self.x_many_batches_and_modes, self.y_many_batches_and_modes)
        np.testing.assert_allclose(value, np.array(5*[[6.45396, 5.33529, 4.49286]]), atol=1e-4, rtol=1e-4)

    def test_max_distances_one_mode(self):
        value = metrics.max_distances(self.x_one_mode, self.y_one_mode)
        np.testing.assert_allclose(value, np.array([[10.63014]]), atol=1e-4, rtol=1e-4)

    def test_max_distances_many_modes(self):
        value = metrics.max_distances(self.x_many_modes, self.y_many_modes)
        np.testing.assert_allclose(value, np.array([[12.04159, 10.63014, 9.21954]]), atol=1e-4, rtol=1e-4)

    def test_max_distances_many_batches_and_modes(self):
        value = metrics.max_distances(self.x_many_batches_and_modes, self.y_many_batches_and_modes)
        np.testing.assert_allclose(value, np.array(5*[[12.04159, 10.63014, 9.21954]]), atol=1e-4, rtol=1e-4)

    def test_final_distances_one_mode(self):
        value = metrics.max_distances(self.x_one_mode, self.y_one_mode)
        np.testing.assert_allclose(value, np.array([[10.63014]]), atol=1e-4, rtol=1e-4)

    def test_final_distances_many_modes(self):
        value = metrics.max_distances(self.x_many_modes, self.y_many_modes)
        np.testing.assert_allclose(value, np.array([[12.04159, 10.63014, 9.21954]]), atol=1e-4, rtol=1e-4)

    def test_final_distances_many_batches_and_modes(self):
        value = metrics.max_distances(self.x_many_batches_and_modes, self.y_many_batches_and_modes)
        np.testing.assert_allclose(value, np.array(5*[[12.04159, 10.63014, 9.21954]]), atol=1e-4, rtol=1e-4)

    def test_miss_max_distance_one_mode(self):
        value = metrics.miss_max_distances(self.x_one_mode, self.y_one_mode, 1)
        np.testing.assert_equal(value, np.array([[True]]))

        value = metrics.miss_max_distances(self.x_one_mode, self.y_one_mode, 15)
        np.testing.assert_equal(value, np.array([[False]]))

    def test_miss_max_distances_many_modes(self):
        value = metrics.miss_max_distances(self.x_many_modes, self.y_many_modes, 10)
        np.testing.assert_equal(value, np.array([[True, True, False]]))

    def test_miss_max_distances_many_batches_and_modes(self):
        value = metrics.miss_max_distances(self.x_many_batches_and_modes, self.y_many_batches_and_modes, 10)
        np.testing.assert_equal(value, np.array(5*[[True, True, False]]))

    def test_miss_rate_top_k_one_mode(self):
        value = metrics.miss_rate_top_k(self.x_one_mode, self.y_one_mode, self.p_one_mode, 2)
        np.testing.assert_equal(value, np.array([[True]]))

    def test_miss_rate_top_k_many_modes(self):
        value = metrics.miss_rate_top_k(self.x_many_modes, self.y_many_modes, self.p_many_modes, 10)
        np.testing.assert_equal(value, np.array([[True, False, False]]))

    def test_miss_rate_top_k_many_batches_and_modes(self):
        value = metrics.miss_rate_top_k(self.x_many_batches_and_modes,
                                        self.y_many_batches_and_modes, self.p_many_batches_and_modes, 10)
        np.testing.assert_equal(value, np.array([[True, False, False],
                                                 [True, True, False],
                                                 [False, False, False],
                                                 [False, False, False],
                                                 [True, True, False]]))

    def test_min_ade_k_one_mode(self):
        value = metrics.min_ade_k(self.x_one_mode, self.y_one_mode, self.p_one_mode)
        np.testing.assert_allclose(value, np.array([[5.33529]]), atol=1e-4, rtol=1e-4)

    def test_min_ade_k_many_modes(self):
        value = metrics.min_ade_k(self.x_many_modes, self.y_many_modes, self.p_many_modes)
        np.testing.assert_allclose(value, np.array([[5.33529, 4.49286, 4.49286]]), atol=1e-4, rtol=1e-4)

    def test_min_ade_k_many_batches_and_modes(self):
        value = metrics.min_ade_k(self.x_many_batches_and_modes, self.y_many_batches_and_modes,
                                  self.p_many_batches_and_modes)
        np.testing.assert_allclose(value, np.array([[5.33529, 4.49286, 4.49286],
                                                    [6.45396, 5.33529, 4.49286],
                                                    [4.49286, 4.49286, 4.49286],
                                                    [4.49286, 4.49286, 4.49286],
                                                    [5.33529, 5.33529, 4.49286]
                                                    ]), atol=1e-4, rtol=1e-4)

    def test_min_fde_k_one_mode(self):
        value = metrics.min_fde_k(self.x_one_mode, self.y_one_mode, self.p_one_mode)
        np.testing.assert_allclose(value, np.array([[10.63014]]), atol=1e-4, rtol=1e-4)

    def test_min_fde_k_many_modes(self):
        value = metrics.min_fde_k(self.x_many_modes, self.y_many_modes, self.p_many_modes)
        np.testing.assert_allclose(value, np.array([[10.63014, 9.21954, 9.21954]]), atol=1e-4, rtol=1e-4)

    def test_min_fde_k_many_batches_and_modes(self):
        value = metrics.min_fde_k(self.x_many_batches_and_modes, self.y_many_batches_and_modes,
                                  self.p_many_batches_and_modes)
        np.testing.assert_allclose(value, np.array([[10.63014, 9.21954, 9.21954],
                                                    [12.04159, 10.63014, 9.21954],
                                                    [9.21954, 9.21954, 9.21954],
                                                    [9.21954, 9.21954, 9.21954],
                                                    [10.63014, 10.64014, 9.21954]]), atol=1e-3, rtol=1e-3)

    def test_stack_ground_truth(self):
        value = metrics.stack_ground_truth(np.ones((5, 2)), 10)
        np.testing.assert_equal(value, np.ones((10, 5, 2)))

    def test_desired_number_of_modes_one_mode(self):
        results = np.ones((10, 1))
        value = metrics.desired_number_of_modes(results, [1, 5, 15, 25])
        np.testing.assert_equal(value, np.ones((10, 4)))

    def test_desired_number_of_modes_enough_data(self):
        results = np.arange(75).reshape(3, 25)
        value = metrics.desired_number_of_modes(results, [1, 5, 15, 25])
        np.testing.assert_equal(value, np.array([[0, 4, 14, 24],
                                                 [25, 29, 39, 49],
                                                 [50, 54, 64, 74]]))

    def test_desired_number_of_modes_not_enough(self):
        results = np.arange(30).reshape(2, 15)
        value = metrics.desired_number_of_modes(results, [1, 5, 15, 25])
        np.testing.assert_equal(value, np.array([[0, 4, 14, 14],
                                                 [15, 19, 29, 29]]))


class TestAggregators(unittest.TestCase):

    def test_RowMean(self):
        rm = metrics.RowMean()
        value = rm(np.arange(20).reshape(2, 10))
        self.assertListEqual(list(value), [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        self.assertDictEqual(rm.serialize(), {'name': 'RowMean'})


class TestMetrics(unittest.TestCase):

    def test_MinADEK(self):
        min_ade = metrics.MinADEK([1, 5, 10], [metrics.RowMean()])
        self.assertDictEqual(min_ade.serialize(), {'name': 'MinADEK',
                                                   'k_to_report': [1, 5, 10],
                                                   'aggregators': [{'name': 'RowMean'}]})

    def test_MinFDEK(self):
        min_fde = metrics.MinFDEK([1, 5, 10], [metrics.RowMean()])
        self.assertDictEqual(min_fde.serialize(), {'name': 'MinFDEK',
                                                   'k_to_report': [1, 5, 10],
                                                   'aggregators': [{'name': 'RowMean'}]})

    def test_MissRateTopK(self):
        hit_rate = metrics.MissRateTopK([1, 5, 10], [metrics.RowMean()], 2)
        self.assertDictEqual(hit_rate.serialize(), {'k_to_report': [1, 5, 10],
                                                    'name': 'MissRateTopK',
                                                    'aggregators': [{'name': 'RowMean'}],
                                                    'tolerance': 2})

    def test_OffRoadRate(self):
        with patch.object(metrics.OffRoadRate, 'load_drivable_area_masks'):
            helper = MagicMock(spec=PredictHelper)
            off_road_rate = metrics.OffRoadRate(helper, [metrics.RowMean()])
            self.assertDictEqual(off_road_rate.serialize(), {'name': 'OffRoadRate',
                                                             'aggregators': [{'name': 'RowMean'}]})

    def test_deserialize_metric(self):

        config = {'name': 'MinADEK',
                  'k_to_report': [1, 5, 10],
                  'aggregators': [{'name': 'RowMean'}]}

        helper = MagicMock(spec=PredictHelper)
        m = metrics.deserialize_metric(config, helper)
        self.assertEqual(m.name, 'MinADEK')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')

        config = {'name': 'MinFDEK',
                  'k_to_report': [1, 5, 10],
                  'aggregators': [{'name': 'RowMean'}]}

        m = metrics.deserialize_metric(config, helper)
        self.assertEqual(m.name, 'MinFDEK')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')

        config = {'name': 'MissRateTopK',
                  'k_to_report': [1, 5, 10],
                  'tolerance': 2,
                  'aggregators': [{'name': 'RowMean'}]}

        m = metrics.deserialize_metric(config, helper)
        self.assertEqual(m.name, 'MissRateTopK_2')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')

        with patch.object(metrics.OffRoadRate, 'load_drivable_area_masks'):
            config = {'name': 'OffRoadRate',
                      'aggregators': [{'name': 'RowMean'}]}

            m = metrics.deserialize_metric(config, helper)
            self.assertEqual(m.name, 'OffRoadRate')
            self.assertEqual(m.aggregators[0].name, 'RowMean')

    def test_flatten_metrics(self):
        results = {"MinFDEK": {"RowMean": [5.92, 6.1, 7.2]},
                   "MinADEK": {"RowMean": [2.48, 3.29, 3.79]},
                   "MissRateTopK_2": {"RowMean": [0.37, 0.45, 0.55]}}

        metric_functions = [metrics.MinFDEK([1, 5, 10], aggregators=[metrics.RowMean()]),
                            metrics.MinADEK([1, 5, 10], aggregators=[metrics.RowMean()]),
                            metrics.MissRateTopK([1, 5, 10], tolerance=2, aggregators=[metrics.RowMean()])]

        flattened = metrics.flatten_metrics(results, metric_functions)

        answer = {'MinFDEK_1': 5.92, 'MinFDEK_5': 6.1, 'MinFDEK_10': 7.2,
                  'MinADEK_1': 2.48, 'MinADEK_5': 3.29, 'MinADEK_10': 3.79,
                  'MissRateTopK_2_1': 0.37, 'MissRateTopK_2_5': 0.45, 'MissRateTopK_2_10': 0.55}

        self.assertDictEqual(flattened, answer)


class TestOffRoadRate(unittest.TestCase):

    def _do_test(self, map_name, predictions, answer):
        with patch.object(PredictHelper, 'get_map_name_from_sample_token') as get_map_name:
            get_map_name.return_value = map_name
            nusc = NuScenes('v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)
            helper = PredictHelper(nusc)

            off_road_rate = metrics.OffRoadRate(helper, [metrics.RowMean()])

            probabilities = np.array([1/3] * predictions.shape[0])
            prediction = Prediction('foo-instance', 'foo-sample', predictions, probabilities)

            # Two violations out of three trajectories
            np.testing.assert_allclose(off_road_rate(np.array([]), prediction), np.array([answer]))

    def test_boston(self):
        predictions = np.array([[(486.91778944573264, 812.8782745377198),
                                 (487.3648565923963, 813.7269620253566),
                                 (487.811923719944, 814.5756495230632),
                                 (488.2589908474917, 815.4243370207698)],
                                [(486.91778944573264, 812.8782745377198),
                                 (487.3648565923963, 813.7269620253566),
                                 (487.811923719944, 814.5756495230632),
                                 (0, 0)],
                                [(0, 0), (0, 1), (0, 2), (0, 3)]])
        self._do_test('boston-seaport', predictions, 2/3)


    def test_one_north(self):
        predictions = np.array([[[965.8515334916171, 535.711518726687],
                                 [963.6475430050381, 532.9713854167148],
                                 [961.4435525191437, 530.231252106192],
                                 [959.239560587773, 527.4911199583674]],
                                [[508.8742570078554, 875.3458194583762],
                                 [505.2029816111618, 877.7929160023881],
                                 [501.5317062144682, 880.2400125464],
                                 [497.86043081777467, 882.6871090904118]],
                                [[0, 0], [0, 1], [0, 2], [0, 3]]])
        self._do_test('singapore-onenorth', predictions, 1/3)

    def test_queenstown(self):
        predictions = np.array([[[744.8769428947988, 2508.398411382534],
                                 [747.7808552527478, 2507.131371270205],
                                 [750.7893530020073, 2506.1385301483474],
                                 [751, 2506]],
                                [[-100, 0], [-10, 100], [0, 2], [-20, 70]]])
        self._do_test('singapore-queenstown', predictions, 1/2)

    def test_hollandvillage(self):
        predictions = np.array([[(1150.811356677105, 1598.0397224872172),
                                (1158.783061670897, 1595.5210995059333),
                                (1166.7543904812692, 1593.0012894706226),
                                (1174.6895821186222, 1590.3704726754975)],
                               [(1263.841977478558, 943.4546342496925),
                                (1262.3235250519404, 944.6782247770625),
                                (1260.8163412684773, 945.9156425437817),
                                (1259.3272449205788, 947.1747683330505)]])
        self._do_test('singapore-hollandvillage', predictions, 0)
