import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle


class TestPhysicsBaselines(unittest.TestCase):

    def test_Baselines_raise_error_when_sec_from_now_bad(self):

        with self.assertRaises(AssertionError):
            ConstantVelocityHeading(2.23, None)

        with self.assertRaises(AssertionError):
            PhysicsOracle(2.25, None)

        PhysicsOracle(5.5, None)
        ConstantVelocityHeading(3, None)

    @patch('nuscenes.prediction.models.physics._kinematics_from_tokens')
    def test_ConstantVelocityHeading(self, mock_kinematics):

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
        mock_kinematics.return_value = 0, 0, 1, 0, 2, 0, 1, 0, 2, 0

        cv_model = ConstantVelocityHeading(6, mock_helper)
        prediction = cv_model('foo-instance_bar-sample')

        answer = np.array([[[0.5, 0], [1, 0], [1.5, 0], [2.0, 0], [2.5, 0], [3.0, 0],
                           [3.5, 0.0], [4.0, 0], [4.5, 0], [5.0, 0], [5.5, 0], [6.0, 0]]])

        np.testing.assert_allclose(answer, np.round(prediction.prediction, 3))

    @patch('nuscenes.prediction.models.physics._kinematics_from_tokens')
    def test_PhysicsOracle(self, mock_kinematics):

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # Made to look like constant acceleration and heading
        mock_helper.get_future_for_agent.return_value = np.array([[0, 1.3], [0, 2.9], [0, 5.2], [0, 8.3], [0, 11.3],
                                                                  [0, 14.6], [0, 19.29], [0, 23.7], [0, 29.19],
                                                                  [0, 33.], [0, 41.3], [0, 48.2]])

        # x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
        mock_kinematics.return_value = 0, 0, 0, 2, 0, 2, 2, 0.05, 2, 0

        oracle = PhysicsOracle(6, mock_helper)
        prediction = oracle('foo-instance_bar-sample')

        answer = np.array([[[0., 1.25], [0., 3.], [0., 5.25], [0., 8.], [0., 11.25], [0., 15.],
                           [0., 19.25], [0., 24.], [0., 29.25], [0., 35.], [0., 41.25], [0., 48.]]])

        np.testing.assert_allclose(answer, np.round(prediction.prediction, 3))

    @patch('nuscenes.prediction.models.physics._kinematics_from_tokens')
    def test_PhysicsOracle_raises_error_when_not_enough_gt(self, mock_kinematics):

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # Made to look like constant acceleration and heading
        mock_helper.get_future_for_agent.return_value = np.array([[0, 1.3], [0, 2.9], [0, 5.2], [0, 8.3], [0, 11.3],
                                                                  [0, 14.6], [0, 19.29], [0, 23.7], [0, 29.19],
                                                                  [0, 33.]])

        # x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
        mock_kinematics.return_value = 0, 0, 0, 2, 0, 2, 2, 0.05, 2, 0

        oracle = PhysicsOracle(6, mock_helper)
        with self.assertRaises(AssertionError):
            oracle('foo-instance_bar-sample')
