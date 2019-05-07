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
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: float,
                 mean_ap_weight: int
                 ):

        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized dictionary. """
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
        assert type(sample_token) == str, 'Error: sample_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 4, 'Error: Rotation must have 4 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Rotation must have 2 elements!'

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


class MetricData:
    """ This class holds accumulated and interpolated data required to calculate the metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array,
                 ):

        # Assert lengths
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        # Assert ordering
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

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

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
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
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
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

    def get_class_data(self, detection_name: str) -> List[Tuple[MetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[MetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: MetricData):
        """ Sets the MetricData entry for a certain detectdion_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

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
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float):
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float):
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:

                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuTonomy detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """

        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {'label_aps': self._label_aps,
                'mean_dist_aps': self.mean_dist_aps,
                'mean_ap': self.mean_ap,
                'label_tp_errors': self._label_tp_errors,
                'tp_errors': self.tp_errors,
                'tp_scores': self.tp_scores,
                'nd_score': self.nd_score,
                'eval_time': self.eval_time}
