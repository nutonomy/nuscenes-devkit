# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

from typing import Dict, Tuple, Any, List, Callable

import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw

MICROSECONDS_PER_SECOND = 1e6
BUFFER = 0.15  # seconds


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


class PredictHelper:
    """ Wrapper class around NuScenes to help retrieve data for the prediction task. """

    def __init__(self, nusc: NuScenes):
        """
        Inits PredictHelper
        :param nusc: Instance of NuScenes class.
        """
        self.data = nusc
        self.inst_sample_to_ann = self._map_sample_and_instance_to_annotation()

    def _map_sample_and_instance_to_annotation(self) -> Dict[Tuple[str, str], str]:
        """
        Creates mapping to look up an annotation given a sample and instance in constant time.
        :return: Mappig from (sample_token, instance_token) -> sample_annotation_token.
        """
        mapping = {}

        for record in self.data.sample_annotation:
            mapping[(record['sample_token'], record['instance_token'])] = record['token']

        return mapping

    def _timestamp_for_sample(self, sample_token: str) -> float:
        """
        Gets timestamp from sample token.
        :param sample_token: Get the timestamp for this sample.
        :return: Timestamp (microseconds).
        """
        return self.data.get('sample', sample_token)['timestamp']

    def _absolute_time_diff(self, time1: float, time2: float) -> float:
        """
        Helper to compute how much time has elapsed in _iterate method.
        :param time1: First timestamp (microseconds since unix epoch).
        :param time2: Second timestamp (microseconds since unix epoch).
        :return: Absolute Time difference in floats.
        """
        return abs(time1 - time2) / MICROSECONDS_PER_SECOND

    def _iterate(self, starting_annotation: Dict[str, Any], seconds: float, direction: str) -> List[Dict[str, Any]]:
        """
        Iterates forwards or backwards in time through the annotations for a given amount of seconds.
        :param starting_annotation: Sample annotation record to start from.
        :param seconds: Number of seconds to iterate.
        :param direction: 'prev' for past and 'next' for future.
        :return: List of annotations ordered by time.
        """
        if seconds < 0:
            raise ValueError(f"Parameter seconds must be non-negative. Recevied {seconds}.")

        # Need to exit early because we technically _could_ return data in this case if
        # the first observation is within the BUFFER.
        if seconds == 0:
            return []

        seconds_with_buffer = seconds + BUFFER
        starting_time = self._timestamp_for_sample(starting_annotation['sample_token'])

        next_annotation = starting_annotation

        time_elapsed = 0.

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

    def get_sample_annotation(self, instance_token: str, sample_token: str) -> Dict[str, Any]:
        """
        Retrieves an annotation given an instance token and its sample.
        :param instance_token: Instance token.
        :param sample_token: Sample token for instance.
        :return: Sample annotation record.
        """
        return self.data.get('sample_annotation', self.inst_sample_to_ann[(sample_token, instance_token)])

    def _get_past_or_future_for_agent(self, instance_token: str, sample_token: str,
                                      seconds: float, in_agent_frame: bool,
                                      direction: str) -> np.ndarray:
        """
        Helper function to reduce code duplication between get_future and get_past for agent.
        :param instance_token: Instance of token.
        :param sample_token: Sample token for instance.
        :param seconds: How many seconds of data to retrieve.
        :param in_agent_frame: Whether to rotate the coordinates so the
            heading is aligned with the y-axis.
        :param direction: 'next' for future or 'prev' for past.
        :return: array of shape [n_timesteps, 2].
        """
        starting_annotation = self.get_sample_annotation(instance_token, sample_token)
        sequence = self._iterate(starting_annotation, seconds, direction)
        coords = np.array([r['translation'][:2] for r in sequence])

        if coords.size == 0:
            return coords

        if in_agent_frame:
            coords = convert_global_coords_to_local(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

    def get_future_for_agent(self, instance_token: str, sample_token: str,
                             seconds: float, in_agent_frame: bool) -> np.ndarray:
        """
        Retrieves the agent's future x,y locations.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param seconds: How much future data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows increase with time, i.e the last row occurs the farthest in the future.
        """
        return self._get_past_or_future_for_agent(instance_token, sample_token, seconds,
                                                  in_agent_frame, direction='next')

    def get_past_for_agent(self, instance_token: str, sample_token: str,
                           seconds: float, in_agent_frame: bool) -> np.ndarray:
        """
        Retrieves the agent's past x,y locations.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_agent(instance_token, sample_token, seconds,
                                                  in_agent_frame, direction='prev')

    def _get_past_or_future_for_sample(self, sample_token: str, seconds: float, in_agent_frame: bool,
                                       function: Callable[[str, str, float, bool], np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Helper function to reduce code duplication between get_future and get_past for sample.
        :param sample_token: Sample token.
        :param seconds: How much past or future data to retrieve.
        :param in_agent_frame: Whether to rotate each agent future for .
        :param function: Either get_past or get_future for agent.
        :return: array of shapes [n_timesteps, 2].
        """
        sample_record = self.data.get('sample', sample_token)
        sequences = {}
        for annotation in sample_record['anns']:
            annotation_record = self.data.get('sample_annotation', annotation)
            sequence = function(annotation_record['instance_token'],
                                annotation_record['sample_token'],
                                seconds, in_agent_frame)

            sequences[annotation_record['instance_token']] = sequence

        return sequences

    def get_future_for_sample(self, sample_token: str, seconds: float, in_agent_frame: bool) -> Dict[str, np.ndarray]:
        """Retrieves the the future x,y locations of all agents in the sample.
        :param sample_token: Sample token.
        :param seconds: How much future data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows increase with time, i.e the last row occurs the farthest in the future.
        """
        return self._get_past_or_future_for_sample(sample_token, seconds, in_agent_frame,
                                                   function=self.get_future_for_agent)

    def get_past_for_sample(self, sample_token: str, seconds: float, in_agent_frame: bool) -> Dict[str, np.ndarray]:
        """Retrieves the the past x,y locations of all agents in the sample.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
        :return: np.ndarray. First column is the x coordinate, second is the y.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_sample(sample_token, seconds, in_agent_frame,
                                                   function=self.get_past_for_agent)
