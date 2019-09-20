# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

from collections import defaultdict
from typing import List, Tuple
import abc

import numpy as np


class EvalBox(abc.ABC):
    """ Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_dist: float = 0.0,  # Distance to ego vehicle in meters.
                 num_pts: int = -1):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.

        # Assert data for shape and NaNs.
        assert type(sample_token) == str, 'Error: sample_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 4, 'Error: Rotation must have 4 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Velocity must have 2 elements!'

        assert type(ego_dist) == float, 'Error: ego_dist must be a float!'
        assert not np.any(np.isnan(ego_dist)), 'Error: ego_dist may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_dist = ego_dist
        self.num_pts = num_pts

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        pass


class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item) -> List[EvalBox]:
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    @property
    def all(self) -> List[EvalBox]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[EvalBox]) -> None:
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb


class MetricData(abc.ABC):
    """ Abstract base class for the *MetricData classes specific to each task. """

    @abc.abstractmethod
    def serialize(self):
        """ Serialize instance into json-friendly format. """
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        pass


class MetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[MetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[MetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: MetricData): #TODO: not match_distance for tracking
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict, metric_data_cls):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), metric_data_cls.deserialize(md))
        return mdl
