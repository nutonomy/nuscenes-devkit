# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

from typing import Dict, Tuple, Any, List, Callable
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
import numpy as np

MICROSECONDS_PER_SECOND = 1e6
BUFFER = 0.15 # seconds

def angle_of_rotation(yaw: float) -> float:
    """Given a yaw (measured from x axis) find the angle needed to rotate by so that
    the yaw points upwards (pi / 2)."""
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2)."""

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

    def _get_past_or_future_for_agent(self, instance: str, sample: str,
                                       seconds: float, in_agent_frame: bool,
                                       direction: str) -> np.ndarray:
        """Helper function to reduce code duplication between get_future and get_past for agent."""
        starting_annotation = self.get_sample_annotation(instance, sample)
        sequence = self._iterate(starting_annotation, seconds, direction)
        coords = np.array([r['translation'][:2] for r in sequence])

        if in_agent_frame:
            coords = convert_global_coords_to_local(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

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
        return self._get_past_or_future_for_agent(instance, sample, seconds,
                                                  in_agent_frame, direction='next')

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
        return self._get_past_or_future_for_agent(instance, sample, seconds,
                                                  in_agent_frame, direction='prev')

    def _get_past_or_future_for_sample(self, sample: str, seconds: float, in_agent_frame: bool,
                                      function: Callable[[str, str, float, bool], np.ndarray]):
        """Helper function to reduce code duplication between get_future and get_past for sample."""
        sample_record = self.data.get('sample', sample)
        sequences = {}
        for annotation in sample_record['anns']:
            annotation_record = self.data.get('sample_annotation', annotation)
            sequence = function(annotation_record['instance_token'],
                                annotation_record['sample_token'],
                                seconds, in_agent_frame)

            sequences[annotation_record['instance_token']] = sequence

        return sequences


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
        return self._get_past_or_future_for_sample(sample, seconds, in_agent_frame,
                                                   function=self.get_future_for_agent)


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
        return self._get_past_or_future_for_sample(sample, seconds, in_agent_frame,
                                                   function=self.get_past_for_agent)


