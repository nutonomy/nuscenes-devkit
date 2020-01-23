import abc
from typing import Tuple
import numpy as np
from pyquaternion import Quaternion
from nuscenes.predict import PredictHelper, convert_global_coords_to_local
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.predict.data_classes import Prediction

KinematicsData = Tuple[float, float, float, float, float, float, float, float, float, float]

def _kinematics_from_tokens(helper: PredictHelper, instance: str, sample: str) -> KinematicsData:
    """
    Returns the 2D position, velocity, and acceleration vectors from the given track
    records, along with the speed, yaw rate, (scalar) acceleration (magnitude), and heading.
    """

    annotation = helper.get_sample_annotation(instance, sample)
    x, y, _ = annotation['translation']
    yaw = quaternion_yaw(Quaternion(annotation['rotation']))

    velocity = helper.get_velocity_for_agent(instance, sample)
    acceleration = helper.get_acceleration_for_agent(instance, sample)
    yaw_rate = helper.get_heading_change_rate_for_agent(instance, sample)

    if np.isnan(velocity):
        velocity = 0.0
    if np.isnan(acceleration):
        acceleration = 0.0
    if np.isnan(yaw_rate):
        yaw_rate = 0.0

    hx, hy = np.cos(yaw), np.sin(yaw)
    vx, vy = velocity * hx, velocity * hy
    ax, ay = acceleration * hx, acceleration * hy

    return x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw


def _constant_velocity_heading_from_kinematics(kinematics_data: KinematicsData,
                                               sec_from_now: int,
                                               sampled_at: int) -> np.ndarray:
    """
    Computes a constant velocity baseline for given kinematics data, time window
    and frequency.
    """
    x, y, vx, vy, _, _, _, _, _, _ = kinematics_data
    preds = []
    time_step = 1.0 / sampled_at
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        preds.append((x + time * vx, y + time * vy))
    return np.array(preds)


def _constant_acceleration_and_heading(kinematics_data: KinematicsData,
                                       sec_from_now: int, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the acceleration and heading are constant.
    """
    # From freshman physics, integrating x'' = a yields x = x_0 + v_0 * t + (1/2) a * t^2.
    x, y, vx, vy, ax, ay, _, _, _, _ = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        half_time_squared = 0.5 * time * time
        preds.append((x + time * vx + half_time_squared * ax,
                      y + time * vy + half_time_squared * ay))
    return np.array(preds)


def _constant_speed_and_yaw_rate(kinematics_data: KinematicsData,
                                 sec_from_now: int, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the (scalar) speed and yaw rate are constant.
    """
    x, y, vx, vy, _, _, speed, yaw_rate, _, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    distance_step = time_step * speed
    yaw_step = time_step * yaw_rate
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        preds.append((x, y))
        yaw += yaw_step
    return np.array(preds)


def _constant_magnitude_accel_and_yaw_rate(kinematics_data: KinematicsData,
                                           sec_from_now: int, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the rates of change of speed and yaw are constant.
    """
    x, y, vx, vy, _, _, speed, yaw_rate, accel, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    speed_step = time_step * accel
    yaw_step = time_step * yaw_rate
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        distance_step = time_step * speed
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        preds.append((x, y))
        speed += speed_step
        yaw += yaw_step
    return np.array(preds)


class Baseline(abc.ABC):

    def __init__(self, helper: PredictHelper):
        self.helper = helper
        self.sec_from_now = 6
        self.sampled_at = 2 # 2 Hz between annotations

    @abc.abstractmethod
    def __call__(self, token: str):
        pass


class ConstantVelocityHeading(Baseline):

    def __call__(self, token: str):
        instance, sample = token.split("_")
        annotation = self.helper.get_sample_annotation(instance, sample)
        kinematics = _kinematics_from_tokens(self.helper, instance, sample)
        cv_heading = _constant_acceleration_and_heading(kinematics, self.sec_from_now, self.sampled_at)
        prediction = convert_global_coords_to_local(cv_heading, annotation['translation'], annotation['rotation'])
        return Prediction(instance, sample, prediction, np.array([1]))


class PhysicsOracle(Baseline):

    def __call__(self, token):
        instance, sample = token.split("_")
        annotation = self.helper.get_sample_annotation(instance, sample)
        kinematics = _kinematics_from_tokens(self.helper, instance, sample)
        ground_truth = self.helper.get_future_for_agent(instance, sample, self.sec_from_now, in_agent_frame=True)

        path_funs = [
        _constant_acceleration_and_heading,
        _constant_magnitude_accel_and_yaw_rate,
        _constant_speed_and_yaw_rate,
        _constant_velocity_heading_from_kinematics]

        paths = [path_fun(kinematics, self.sec_from_now, self.sampled_at) for path_fun in path_funs]
        paths_ego = [convert_global_coords_to_local(path, annotation['translation'], annotation['rotation']) for path in paths]

        # Select the one with the least l2 error, averaged (or equivalently, summed) over all
        # points of the path.  This is (proportional to) the Frobenius norm of the difference
        # between the path (as an n x 2 matrix) and the ground truth.

        oracle = sorted(paths_ego,
                        key=lambda path: np.linalg.norm(np.array(path) - ground_truth, ord="fro"))[0]
        return Prediction(instance, sample, oracle, np.array([1]))