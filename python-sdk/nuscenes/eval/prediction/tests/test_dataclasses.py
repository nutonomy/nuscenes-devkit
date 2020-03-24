import unittest

import numpy as np

from nuscenes.eval.prediction.data_classes import Prediction


class TestPrediction(unittest.TestCase):

    def test(self):
        prediction = Prediction('instance', 'sample', np.ones((2, 2, 2)), np.zeros(2))

        self.assertEqual(prediction.number_of_modes, 2)
        self.assertDictEqual(prediction.serialize(), {'instance': 'instance',
                                                      'sample': 'sample',
                                                      'prediction': [[[1, 1], [1, 1]],
                                                                     [[1, 1], [1, 1]]],
                                                      'probabilities': [0, 0]})
