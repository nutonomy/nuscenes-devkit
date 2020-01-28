import unittest

import numpy as np

from nuscenes.eval.predict import metrics


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

    def test_hit_max_distance_one_mode(self):
        value = metrics.hit_max_distances(self.x_one_mode, self.y_one_mode, 1)
        np.testing.assert_equal(value, np.array([[False]]))

        value = metrics.hit_max_distances(self.x_one_mode, self.y_one_mode, 15)
        np.testing.assert_equal(value, np.array([[True]]))

    def test_hit_max_distances_many_modes(self):
        value = metrics.hit_max_distances(self.x_many_modes, self.y_many_modes, 10)
        np.testing.assert_equal(value, np.array([[False, False, True]]))

    def test_hit_max_distances_many_batches_and_modes(self):
        value = metrics.hit_max_distances(self.x_many_batches_and_modes, self.y_many_batches_and_modes, 10)
        np.testing.assert_equal(value, np.array(5*[[False, False, True]]))

    def test_hit_rate_top_k_one_mode(self):
        value = metrics.hit_rate_top_k(self.x_one_mode, self.y_one_mode, self.p_one_mode, 2)
        np.testing.assert_equal(value, np.array([[False]]))

    def test_hit_rate_top_k_many_modes(self):
        value = metrics.hit_rate_top_k(self.x_many_modes, self.y_many_modes, self.p_many_modes, 10)
        np.testing.assert_equal(value, np.array([[False, True, True]]))

    def test_hit_rate_top_k_many_batches_and_modes(self):
        value = metrics.hit_rate_top_k(self.x_many_batches_and_modes,
                                       self.y_many_batches_and_modes, self.p_many_batches_and_modes, 10)
        np.testing.assert_equal(value, np.array([[False, True, True],
                                                 [False, False, True],
                                                 [True, True, True],
                                                 [True, True, True],
                                                 [False, False, True]]))

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
        value = list(np.arange(20).reshape(2, 10))
        self.assertListEqual(value, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

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

    def test_HitRateTopK(self):
        hit_rate = metrics.HitRateTopK([1, 5, 10], [metrics.RowMean()], 2)
        self.assertDictEqual(hit_rate.serialize(), {'k_to_report': [1, 5, 10],
                                                    'name': 'HitRateTopK',
                                                    'aggregators': [{'name': 'RowMean'}],
                                                    'tolerance': 2})

    def test_DeserializeMetric(self):

        config = {'name': 'MinADEK',
                  'k_to_report': [1, 5, 10],
                  'aggregators': [{'name': 'RowMean'}]}

        m = metrics.DeserializeMetric(config)
        self.assertEqual(m.name, 'MinADEK')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')

        config = {'name': 'MinFDEK',
                  'k_to_report': [1, 5, 10],
                  'aggregators': [{'name': 'RowMean'}]}

        m = metrics.DeserializeMetric(config)
        self.assertEqual(m.name, 'MinFDEK')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')

        config = {'name': 'HitRateTopK',
                  'k_to_report': [1, 5, 10],
                  'tolerance': 2,
                  'aggregators': [{'name': 'RowMean'}]}

        m = metrics.DeserializeMetric(config)
        self.assertEqual(m.name, 'HitRateTopK_2')
        self.assertListEqual(m.k_to_report, [1, 5, 10])
        self.assertEqual(m.aggregators[0].name, 'RowMean')


