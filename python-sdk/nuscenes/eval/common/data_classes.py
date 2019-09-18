# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

from collections import defaultdict
from typing import List, Tuple
import abc

import numpy as np

from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES


class EvalBox:
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""
    # TODO: Add tracking specific fields

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 detection_name: str = "car",
                 attribute_name: str = "",  # Box attribute. Each box can have at most 1 attribute.
                 ego_dist: float = 0.0,  # Distance to ego vehicle in meters.
                 detection_score: float = -1.0,  # Only applies to predictions.
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

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(ego_dist) == float, 'Error: ego_dist must be a float!'
        assert not np.any(np.isnan(ego_dist)), 'Error: ego_dist may not be NaN!'

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.detection_name = detection_name
        self.attribute_name = attribute_name
        self.ego_dist = ego_dist
        self.detection_score = detection_score
        self.num_pts = num_pts

    def __repr__(self):
        return str(self.serialize())

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.detection_name == other.detection_name and
                self.attribute_name == other.attribute_name and
                self.ego_dist == other.ego_dist and
                self.detection_score == other.detection_score and
                self.num_pts == other.num_pts)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'detection_name': self.detection_name,
            'attribute_name': self.attribute_name,
            'ego_dist': self.ego_dist,
            'detection_score': self.detection_score,
            'num_pts': self.num_pts
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   detection_name=content['detection_name'],
                   attribute_name=content['attribute_name'],
                   ego_dist=0.0 if 'ego_dist' not in content else float(content['ego_dist']),
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']))


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
        ok = True
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                ok = ok and box1 == box2
        return ok

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
    def deserialize(cls, content):
        """ Initialize from serialized content. """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [EvalBox.deserialize(box) for box in boxes])
        return eb


class MetricData(abc.ABC):
    """ Abstract base class for the *MetricData classes specific to each task. """

    @abc.abstractmethod
    def serialize(self):
        """ Serialize instance into json-friendly format. """
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content):
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
    def deserialize(cls, content, metric_data_cls):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), metric_data_cls.deserialize(md))
        return mdl
