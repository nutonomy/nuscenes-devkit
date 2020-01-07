# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

from typing import Dict, Tuple, Any

import numpy as np

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.tracking.constants import TRACKING_NAMES, TRACKING_METRICS, AMOT_METRICS


class TrackingConfig:
    """ Data class that specifies the tracking evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_th_tp: float,
                 min_recall: float,
                 max_boxes_per_sample: float,
                 metric_worst: Dict[str, float],
                 num_thresholds: int):

        assert set(class_range.keys()) == set(TRACKING_NAMES), "Class count mismatch."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.max_boxes_per_sample = max_boxes_per_sample
        self.metric_worst = metric_worst
        self.num_thresholds = num_thresholds

        TrackingMetricData.set_nelem(num_thresholds)

        self.class_names = sorted(self.class_range.keys())

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
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'metric_worst': self.metric_worst,
            'num_thresholds': self.num_thresholds
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['max_boxes_per_sample'],
                   content['metric_worst'],
                   content['num_thresholds'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class TrackingMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the tracking metrics. """

    nelem = None
    metrics = [m for m in list(set(TRACKING_METRICS) - set(AMOT_METRICS))]

    def __init__(self):
        # Set attributes explicitly to help IDEs figure out what is going on.
        assert TrackingMetricData.nelem is not None
        init = np.full(TrackingMetricData.nelem, np.nan)
        self.confidence = init
        self.recall_hypo = init
        self.recall = init
        self.motar = init
        self.mota = init
        self.motp = init
        self.faf = init
        self.gt = init
        self.tp = init
        self.mt = init
        self.ml = init
        self.fp = init
        self.fn = init
        self.ids = init
        self.frag = init
        self.tid = init
        self.lgd = init

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def __setattr__(self, *args, **kwargs):
        assert len(args) == 2
        name = args[0]
        values = np.array(args[1])
        assert values is None or len(values) == TrackingMetricData.nelem
        super(TrackingMetricData, self).__setattr__(name, values)

    def set_metric(self, metric_name: str, values: np.ndarray) -> None:
        """ Sets the specified metric. """
        self.__setattr__(metric_name, values)

    def get_metric(self, metric_name: str) -> np.ndarray:
        """ Returns the specified metric. """
        return self.__getattribute__(metric_name)

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
        ret_dict = dict()
        for metric_name in ['confidence', 'recall_hypo'] + TrackingMetricData.metrics:
            ret_dict[metric_name] = self.get_metric(metric_name).tolist()
        return ret_dict

    @classmethod
    def set_nelem(cls, nelem: int) -> None:
        cls.nelem = nelem

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        md = cls()
        for metric in ['confidence', 'recall_hypo'] + TrackingMetricData.metrics:
            md.set_metric(metric, content[metric])
        return md

    @classmethod
    def no_predictions(cls):
        """ Returns an md instance corresponding to having no predictions. """
        md = cls()
        md.confidence = np.zeros(cls.nelem)
        for metric in TrackingMetricData.metrics:
            md.set_metric(metric, np.zeros(cls.nelem))
        md.recall = np.linspace(0, 1, cls.nelem)
        return md

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        md = cls()
        md.confidence = np.linspace(0, 1, cls.nelem)[::-1]
        for metric in TrackingMetricData.metrics:
            md.set_metric(metric, np.random.random(cls.nelem))
        md.recall = np.linspace(0, 1, cls.nelem)
        return md


class TrackingMetrics:
    """ Stores tracking metric results. Provides properties to summarize. """

    def __init__(self, cfg: TrackingConfig):

        self.cfg = cfg
        self.eval_time = None
        self.label_metrics: Dict[str, Dict[str, float]] = {}
        self.class_names = self.cfg.class_names
        self.metric_names = [l for l in TRACKING_METRICS]

        # Init every class.
        for metric_name in self.metric_names:
            self.label_metrics[metric_name] = {}
            for class_name in self.class_names:
                self.label_metrics[metric_name][class_name] = np.nan

    def add_label_metric(self, metric_name: str, tracking_name: str, value: float) -> None:
        assert metric_name in self.label_metrics
        self.label_metrics[metric_name][tracking_name] = float(value)

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    def compute_metric(self, metric_name: str, class_name: str = 'all') -> float:
        if class_name == 'all':
            data = list(self.label_metrics[metric_name].values())
            if len(data) > 0:
                # Some metrics need to be summed, not averaged.
                # Nan entries are ignored.
                if metric_name in ['mt', 'ml', 'tp', 'fp', 'fn', 'ids', 'frag']:
                    return float(np.nansum(data))
                else:
                    return float(np.nanmean(data))
            else:
                return np.nan
        else:
            return float(self.label_metrics[metric_name][class_name])

    def serialize(self) -> Dict[str, Any]:
        metrics = dict()
        metrics['label_metrics'] = self.label_metrics
        metrics['eval_time'] = self.eval_time
        metrics['cfg'] = self.cfg.serialize()
        for metric_name in self.label_metrics.keys():
            metrics[metric_name] = self.compute_metric(metric_name)

        return metrics

    @classmethod
    def deserialize(cls, content: dict) -> 'TrackingMetrics':
        """ Initialize from serialized dictionary. """
        cfg = TrackingConfig.deserialize(content['cfg'])
        tm = cls(cfg=cfg)
        tm.add_runtime(content['eval_time'])
        tm.label_metrics = content['label_metrics']

        return tm

    def __eq__(self, other):
        eq = True
        eq = eq and self.label_metrics == other.label_metrics
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
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0):  # Does not apply to GT.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

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
                self.ego_translation == other.ego_translation and
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
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))


class TrackingMetricDataList:
    """ This stores a set of MetricData in a dict indexed by name. """

    def __init__(self):
        self.md: Dict[str, TrackingMetricData] = {}

    def __getitem__(self, key) -> TrackingMetricData:
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def set(self, tracking_name: str, data: TrackingMetricData):
        """ Sets the MetricData entry for a certain tracking_name. """
        self.md[tracking_name] = data

    def serialize(self) -> dict:
        return {key: value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict, metric_data_cls):
        mdl = cls()
        for name, md in content.items():
            mdl.set(name, metric_data_cls.deserialize(md))
        return mdl
