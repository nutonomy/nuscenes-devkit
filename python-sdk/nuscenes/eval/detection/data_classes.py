# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: str,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: float,
                 mean_ap_weight: int
                 ):

        assert set(class_range.keys()) == set(DETECTION_NAMES)

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized dictionary """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])


class EvalBox:
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

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
        assert type(sample_token) == str

        assert len(translation) == 3
        assert not np.any(np.isnan(translation))

        assert len(size) == 3
        assert not np.any(np.isnan(size))

        assert len(rotation) == 4
        assert not np.any(np.isnan(rotation))

        assert len(velocity) == 2
        assert not np.any(np.isnan(velocity))

        assert detection_name in DETECTION_NAMES

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == ""

        assert type(ego_dist) == float
        assert not np.any(np.isnan(ego_dist))

        assert type(detection_score) == float
        assert not np.any(np.isnan(detection_score))

        assert type(num_pts) == int

        assert not np.any(np.isnan(num_pts))

        # Assign
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
        """ Serialize instance into json-friendly format """
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
        """ Initialize from serialized content """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   detection_name=content['detection_name'],
                   attribute_name=content['attribute_name'],
                   ego_dist=content['ego_dist'],
                   detection_score=content['detection_score'],
                   num_pts=int(content['num_pts']))


class EvalBoxes:
    """ Data class that groups EvalBox instances by sample """

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
            if not len(self[token]) == len(self[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                ok = ok and box1 == box2
        return ok

    @property
    def all(self) -> List[EvalBox]:
        """ Returns all EvalBoxes in a list """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[float]:
        """ Returns a list of all keys """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token, boxes) -> None:
        """ Adds a list of boxes """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized content """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [EvalBox.deserialize(box) for box in boxes])
        return eb


class MetricData:
    """ This class holds accumulated and interpolated data required to calculate the metrics """

    nelem = 101

    def __init__(self,
                 recall: np.array = np.empty(0),
                 precision: np.array = np.empty(0),
                 confidence: np.array = np.empty(0),
                 trans_err: np.array = np.empty(0),
                 vel_err: np.array = np.empty(0),
                 scale_err: np.array = np.empty(0),
                 orient_err: np.array = np.empty(0),
                 attr_err: np.array = np.empty(0),
                 ):

        # Assert lenths
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        # Assert ordering
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be decending.
        assert all(recall == sorted(recall))  # Recalls should be ascending

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self):
        """ Serialize instance into json-friendly format """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized content """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']))

    @classmethod
    def random_md(cls):
        return cls(recall=np.sort(np.random.random(cls.nelem)),
                   precision=np.random.random(cls.nelem),
                   confidence=np.sort(np.random.random(cls.nelem))[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem))


class MetricDataList:
    """ This stores a set of MetricData in a dict indexed by (detection-name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def set(self, detection_name: str, match_distance: float, data: MetricData):
        """ Sets the MetricData entry for a certain detectdion_name and match_distance """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0]+':'+str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), MetricData.deserialize(md))
        return mdl


class DetectionMetrics:
    """ Stores average precision and true positive metrics. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self.label_aps = defaultdict(list)
        self.label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, ap: float):
        self.label_aps[detection_name].append(ap)

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self.label_tp_errors[detection_name][metric_name] = tp

    def add_runtime(self, eval_time: float):
        self.eval_time = eval_time

    @property
    def mean_ap(self) -> float:
        return float(np.mean([np.mean(aps) for aps in self.label_aps.values()]))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:

                class_errors.append(self.label_tp_errors[detection_name][metric_name])

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error
            score = 1.0 - self.tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def weighted_sum(self) -> float:

        # Summarize
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {'label_aps': self.label_aps,
                'label_tp_errors': self.label_tp_errors,
                'mean_ap': self.mean_ap,
                'tp_errors': self.tp_errors,
                'tp_scores': self.tp_scores,
                'weighted_sum': self.weighted_sum,
                'eval_time': self.eval_time}
