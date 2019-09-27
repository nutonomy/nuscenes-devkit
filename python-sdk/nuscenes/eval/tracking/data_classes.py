# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

from typing import List, Dict, Tuple

import numpy as np

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.tracking.constants import TRACKING_NAMES


class TrackingConfig:
    """ Data class that specifies the tracking evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: float,
                 num_sample_pts: int):

        assert set(class_range.keys()) == set(TRACKING_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.num_sample_pts = num_sample_pts

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

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
            'num_sample_pts': self.num_sample_pts
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['num_sample_pts'])


class TrackingMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the tracking metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 confidence: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence

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
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']))

    @classmethod
    def no_predictions(cls):
        """ Returns an md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   confidence=np.zeros(cls.nelem))

    @classmethod
    def random_md(cls):
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1])


class TrackingMetrics:
    """ Stores tracking metric results. Provides properties to summarize. """

    def __init__(self, cfg: TrackingConfig):

        self.cfg = cfg
        self.eval_time = None

    @property
    def amota(self) -> float:
        raise NotImplementedError

    @property
    def amotp(self) -> float:
        raise NotImplementedError

    @property
    def mota(self) -> float:
        raise NotImplementedError

    @property
    def motp(self) -> float:
        raise NotImplementedError

    @property
    def idf1(self) -> float:
        raise NotImplementedError

    @property
    def faf(self) -> float:
        raise NotImplementedError

    @property
    def mt(self) -> float:
        raise NotImplementedError

    @property
    def ml(self) -> float:
        raise NotImplementedError

    @property
    def fp(self) -> float:
        raise NotImplementedError

    @property
    def fn(self) -> float:
        raise NotImplementedError

    @property
    def ids(self) -> float:
        raise NotImplementedError

    @property
    def frag(self) -> float:
        raise NotImplementedError

    @property
    def trans_err(self) -> float:
        raise NotImplementedError

    @property
    def scale_err(self) -> float:
        raise NotImplementedError

    @property
    def orient_err(self) -> float:
        raise NotImplementedError

    @property
    def vel_err(self) -> float:
        raise NotImplementedError

    def add_runtime(self, eval_time: float):
        self.eval_time = eval_time

    def serialize(self):
        return {
            'amota': self.amota,
            'amotp': self.amotp,
            'mota': self.mota,
            'motp': self.motp,
            'idf1': self.idf1,
            'faf': self.faf,
            'mt': self.mt,
            'ml': self.ml,
            'fp': self.fp,
            'fn': self.fn,
            'ids': self.ids,
            'frag': self.frag,
            'trans_err': self.trans_err,
            'scale_err': self.scale_err,
            'orient_err': self.orient_err,
            'vel_err': self.vel_err,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = TrackingConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        # TODO: add stuff

        return metrics

    def __eq__(self, other):

        eq = True
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class TrackingBox(EvalBox):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_dist: float = 0.0,  # Distance to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: int = -1,  # Instance id of this object.
                 tracking_name: str = "car",  # TODO: set to ''
                 tracking_score: float = -1.0):

        super().__init__(sample_token, translation, size, rotation, velocity, ego_dist, num_pts)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'
        assert tracking_name in TRACKING_NAMES, 'Error: Unknown tracking_name %s' % tracking_name

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_dist == other.ego_dist and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_dist': self.ego_dist,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_dist=0.0 if 'ego_dist' not in content else float(content['ego_dist']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))
