import unittest
from unittest.mock import MagicMock, patch
from nuscenes.predict import PredictHelper
from nuscenes.predict.models.physics import ConstantVelocityHeading, PhysicsOracle
import numpy as np

class TestPhysicsBaselines(unittest.TestCase):


    @patch('nuscenes.predict.models.physics._kinematics_from_tokens')
    def test_ConstantVelocityHeading(self, mock_kinematics):

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
        mock_kinematics.return_value = 0, 0, 1, 0, 2, 0, 1, 0, 2, 0

        cv_model = ConstantVelocityHeading(6, mock_helper)
        prediction = cv_model('foo-instance_bar-sample')

        answer = np.array([[[0, 0.5], [0, 1], [0, 1.5], [0, 2.0], [0, 2.5], [0, 3.0],
                           [0, 3.5], [0, 4.0], [0, 4.5], [0, 5.0], [0, 5.5], [0, 6.0]]])

        np.testing.assert_allclose(answer, np.round(prediction.prediction, 3))

    @patch('nuscenes.predict.models.physics._kinematics_from_tokens')
    def test_PhysicsOracle(self, mock_kinematics):

        mock_helper = MagicMock(spec=PredictHelper)
        mock_helper.get_sample_annotation.return_value = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # Made to look like constant acceleration and heading
        mock_helper.get_future_for_agent.return_value = np.array([[0, 1.3], [0, 2.9], [0, 5.2], [0, 8.3], [0, 11.3],
                                                                  [0, 14.6], [0, 19.29], [0, 23.7], [0, 29.19],
                                                                  [0, 33.], [0, 41.3], [0, 48.2]])

        # x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
        mock_kinematics.return_value = 0, 0, 2, 0, 2, 0, 2, 0.05, 2, 0

        cv_model = PhysicsOracle(6, mock_helper)
        prediction = cv_model('foo-instance_bar-sample')

        answer = np.array([[[0., 1.25], [0., 3.], [0., 5.25], [0., 8.], [0., 11.25], [0., 15.],
                           [0., 19.25], [0., 24.], [0., 29.25], [0., 35.], [0., 41.25], [0., 48.]]])

        np.testing.assert_allclose(answer, np.round(prediction.prediction, 3))
