
from typing import Dict, Tuple, Any, List
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
import numpy as np

MICROSECONDS_PER_SECOND = 1e6
BUFFER = 0.15 # seconds

def angle_of_rotation(yaw: float) -> float:
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame."""

    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]

class PredictHelper:
    """Wrapper class around NuScenes to help retrieve data for the prediction task."""

    def __init__(self, nusc: NuScenes):
        """Inits PredictHelper
        :param nusc: Instance of NuScenes class
        """
        self.data = nusc
        self.inst_sample_to_ann = self._map_sample_and_instance_to_annotation()

    def _map_sample_and_instance_to_annotation(self) -> Dict[Tuple[str, str], str]:
        """Creates mapping to look up an annotation given a sample and instance in constant time."""
        mapping = {}

        for record in self.data.sample_annotation:
            mapping[(record['sample_token'], record['instance_token'])] = record['token']

        return mapping

    def _timestamp_for_sample(self, sample: str):
        """"""
        return self.data.get('sample', sample)['timestamp']

    def _absolute_time_diff(self, time1: int, time2: int) -> int:
        """Helper to compute how much time has elapsed in _iterate method."""
        return abs(time1 - time2) / MICROSECONDS_PER_SECOND

    def _iterate(self, starting_annotation: Dict[str, Any], seconds: float, direction: str) -> List[Dict[str, Any]]:
        """Iterates forwards or backwards in time through the annotations for a given amount of seconds."""

        seconds_with_buffer = seconds + BUFFER
        starting_time = self._timestamp_for_sample(starting_annotation['sample_token'])

        next_annotation = starting_annotation

        time_elapsed = 0

        annotations = []

        while time_elapsed <= seconds_with_buffer:

            if next_annotation[direction] == '':
                break

            next_annotation = self.data.get('sample_annotation', next_annotation[direction])
            current_time = self._timestamp_for_sample(next_annotation['sample_token'])

            time_elapsed = self._absolute_time_diff(current_time, starting_time)

            if time_elapsed < seconds_with_buffer:
                annotations.append(next_annotation)

        return annotations

    def get_sample_annotation(self, instance: str, sample: str) -> Dict[str, Any]:
        """Retrives an annotation given an instance token and its sample.
        :param instance: Instance token
        :param sample: Sample token
        """
        return self.data.get('sample_annotation', self.inst_sample_to_ann[(sample, instance)])

    def get_future_for_agent(self, instance: str, sample: str,
                             seconds: float, in_agent_frame: bool) -> np.array:
        """Retrieves the agent's future x,y locations.
        :param instance: Instance token
        :param sample: Sample token
        :param seconds: How much future data to retrieve
        :in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows increase with time, i.e the last row occurs the farthest from
            the current time.
        """
        starting_annotation = self.get_sample_annotation(instance, sample)
        future = self._iterate(starting_annotation, seconds, 'next')
        coords = np.array([r['translation'][:2] for r in future])

        if in_agent_frame:
            coords = convert_global_coords_to_local(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

    def get_past_for_agent(self, instance: str, sample: str,
                           seconds: float, in_agent_frame: bool) -> np.array:
        """Retrieves the agent's past x,y locations
        :param instance: Instance token
        :param sample: Sample token
        :param seconds: How much past data to retrieve
        :in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows decreate with time, i.e the last row happened the farthest from
            the current time.
        """
        starting_annotation = self.get_sample_annotation(instance, sample)
        past = self._iterate(starting_annotation, seconds, 'prev')
        coords = np.array([r['translation'][:2] for r in past])

        if in_agent_frame:
            coords = convert_global_coords_to_local(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

    def get_future_for_sample(self, sample: str, seconds: float, in_agent_frame: bool) -> Dict[str, np.ndarray]:
        """Retrieves the the future x,y locations of all agents in the sample.
        :param instance: Instance token
        :param sample: Sample token
        :param seconds: How much future data to retrieve
        :in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows increase with time, i.e the last row occurs the farthest from
            the current time.
        """
        sample_record = self.data.get('sample', sample)
        futures = {}
        for annotation in sample_record['anns']:
            annotation_record = self.data.get('sample_annotation', annotation)
            future = self.get_future_for_agent(annotation_record['instance_token'],
                                               annotation_record['sample_token'],
                                               seconds, in_agent_frame)

            futures[annotation_record['instance_token']] = future

        return futures

    def get_past_for_sample(self, sample: str, seconds: float, in_agent_frame: bool) -> Dict[str, np.ndarray]:
        """Retrieves the the past x,y locations of all agents in the sample.
        :param instance: Instance token
        :param sample: Sample token
        :param seconds: How much past data to retrieve
        :in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows increase with time, i.e the last row occurs the farthest from
            the current time.
        """
        sample_record = self.data.get('sample', sample)
        pasts = {}
        for annotation in sample_record['anns']:
            annotation_record = self.data.get('sample_annotation', annotation)
            past = self.get_past_for_agent(annotation_record['instance_token'],
                                               annotation_record['sample_token'],
                                               seconds, in_agent_frame)

            pasts[annotation_record['instance_token']] = past

        return pasts

    def _compute_diff_between_sample_annotations(self, instance: str, sample: str, max_time_diff: float,
                                                 with_function, **kwargs) -> float:
        annotation = self.get_sample_annotation(instance, sample)

        if annotation['prev'] == '':
            return np.nan

        prev = self.data.get('sample_annotation', annotation['prev'])

        current_time = 1e-6 * self.data.get('sample', sample)['timestamp']
        prev_time = 1e-6 * self.data.get('sample', prev['sample_token'])['timestamp']
        time_diff = current_time - prev_time

        if time_diff <= max_time_diff:

            return with_function(annotation, prev, time_diff, **kwargs)

        else:
            return np.nan

    def get_velocity_for_agent(self, instance: str, sample: str, max_time_diff=1.5) -> float:
        """Computes velocity based on the difference between the current and previous annotation."""
        return self._compute_diff_between_sample_annotations(instance, sample, max_time_diff, with_function=velocity)

    def get_heading_change_rate_for_agent(self, instance: str, sample: str, max_time_diff=1.5) -> float:
        return self._compute_diff_between_sample_annotations(instance, sample, max_time_diff, with_function=heading_change_rate)

    def get_acceleration_for_agent(self, instance: str, sample: str, max_time_diff=1.5) -> float:
        return self._compute_diff_between_sample_annotations(instance, sample, max_time_diff, with_function=acceleration,
                                                            instance_token=instance, helper=self)

def velocity(current: Dict[str, Any], prev: Dict[str, Any], time_diff: float) -> float:
    diff = (np.array(current['translation']) - np.array(prev['translation'])) / time_diff
    return np.linalg.norm(diff[:2])

def heading_change_rate(current: Dict[str, Any], prev: Dict[str, Any], time_diff: float) -> float:
    current_yaw = quaternion_yaw(Quaternion(current['rotation']))
    prev_yaw = quaternion_yaw(Quaternion(prev['rotation']))

    return angle_diff(current_yaw, prev_yaw, period=2*np.pi) / time_diff

def acceleration(current: Dict[str, Any], prev: Dict[str, Any], time_diff: float, instance_token: str, helper: PredictHelper) -> float:
    current_velocity = helper.get_velocity_for_agent(instance_token, current['sample_token'])
    prev_velocity = helper.get_velocity_for_agent(instance_token, prev['sample_token'])

    return (current_velocity - prev_velocity) / time_diff




