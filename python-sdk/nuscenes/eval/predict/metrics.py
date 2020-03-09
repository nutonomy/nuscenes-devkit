# nuScenes dev-kit.
# Code written by Freddy Boulton, Eric Wolff 2020.
""" Implementation of metrics used in the nuScenes prediction challenge. """
import abc
from typing import List, Dict, Any

import numpy as np
from shapely.geometry import MultiPolygon, LineString

from nuscenes.predict import PredictHelper
from nuscenes.predict.input_representation.static_layers import load_all_maps
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.predict.data_classes import Prediction


def returns_2d_array(function):
    """Makes sure that the metric returns an array of
    shape [batch_size, num_modes]"""

    def _returns_array(*args, **kwargs):
        result = function(*args, **kwargs)

        if isinstance(result, (int, float)):
            result = np.array([[result]])

        elif result.ndim == 1:
            result = np.expand_dims(result, 0)

        return result

    return _returns_array


@returns_2d_array
def mean_distances(stacked_trajs: np.ndarray,
                   stacked_ground_truth: np.ndarray) -> np.ndarray:
    """Efficiently compute mean L2 norm between trajectories and ground truths
        (pairwise over states). Batch dimension is optional.
    :param stacked_trajs: [batch_size, num_modes, horizon_length, state_dim]
    :param stacked_ground_truth: [batch_size, num_modes, horizon_length, state_dim]
    :return: mean L2 norms as [batch_size, num_modes]
    """
    return np.mean(np.linalg.norm(stacked_trajs - stacked_ground_truth, axis=-1), axis=-1)


@returns_2d_array
def max_distances(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray) -> np.ndarray:
    """Efficiently compute max L2 norm between trajectories and ground truths
        (pairwise over states).
    :pram stacked_trajs: [num_modes, horizon_length, state_dim]
    :pram stacked_ground_truth: [num_modes, horizon_length, state_dim]
    :return: max L2 norms as [num_modes]
    """
    return np.max(np.linalg.norm(stacked_trajs - stacked_ground_truth, axis=-1), axis=-1)


@returns_2d_array
def final_distances(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray) -> np.ndarray:
    """Efficiently compute the L2 norm between the last points in the trajectory
    :param stacked_trajs: [num_modes, horizon_length, state_dim]
    :param stacked_ground_truth: [num_modes, horizon_length, state_dim]
    :return: mean L2 norms as [num_modes]
    """
    # We use take to index the elements in the last dimension so that we can also
    # apply this function for a batch
    diff_of_last = np.take(stacked_trajs, [-1], -2).squeeze() - np.take(stacked_ground_truth, [-1], -2).squeeze()
    return np.linalg.norm(diff_of_last, axis=-1)


@returns_2d_array
def hit_max_distances(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray,
                      tolerance: float) -> np.array:
    """Efficiently compute 'hit' metric between trajectories and ground truths.
    :param stacked_trajs: [num_modes, horizon_length, state_dim]
    :param stacked_ground_truth: [num_modes, horizon_length, state_dim]
    :param tolerance: max distance (m) for a 'hit' to be True
    :return: True iff there was a 'hit.' Size [num_modes]
    """
    return max_distances(stacked_trajs, stacked_ground_truth) < tolerance


@returns_2d_array
def rank_metric_over_top_k_modes(metric_results: np.ndarray,
                                 mode_probabilities: np.ndarray,
                                 ranking_func: str) -> np.ndarray:
    """Compute a metric over all trajectories ranked by probability of each trajectory.
    :param metric_results: 1-dimensional array of shape [batch_size, num_modes]
    :param mode_probabilities: 1-dimensional array of shape [batch_size, num_modes]
    :param ranking_func: Either 'min' or 'max'. How you want to metrics ranked over the top
            k modes.
    :return: Array of shape [num_modes]
    """

    if ranking_func == "min":
        func = np.minimum.accumulate
    elif ranking_func == "max":
        func = np.maximum.accumulate
    else:
        raise ValueError(f"Parameter ranking_func must be one of min or max. Received {ranking_func}")

    p_sorted = np.flip(mode_probabilities.argsort(axis=-1), axis=-1)
    indices = np.indices(metric_results.shape)

    sorted_metrics = metric_results[indices[0], p_sorted]

    return func(sorted_metrics, axis=-1)


def hit_rate_top_k(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray,
                   mode_probabilities: np.ndarray,
                   tolerance: float) -> np.ndarray:
    """Compute the hit rate over the top k modes."""

    hit_rate = hit_max_distances(stacked_trajs, stacked_ground_truth, tolerance)
    return rank_metric_over_top_k_modes(hit_rate, mode_probabilities, "max")


def min_ade_k(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray,
              mode_probabilities: np.ndarray) -> np.ndarray:
    """Compute the min ade over the top k modes."""

    ade = mean_distances(stacked_trajs, stacked_ground_truth)
    return rank_metric_over_top_k_modes(ade, mode_probabilities, "min")


def min_fde_k(stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray,
              mode_probabilities: np.ndarray) -> np.ndarray:
    """Compute the min fde over the top k modes."""

    fde = final_distances(stacked_trajs, stacked_ground_truth)
    return rank_metric_over_top_k_modes(fde, mode_probabilities, "min")


def stack_ground_truth(ground_truth: np.ndarray, num_modes: int) -> np.ndarray:
    """Make k identical copies of the ground truth to make computing the metrics across modes
    easier.
    :param ground_truth: shape [horizon_length, state_dim]
    :param num_modes: number of modes in prediction
    :return: shape [num_modes, horizon_length, state_dim]
    """
    return np.repeat(np.expand_dims(ground_truth, 0), num_modes, axis=0)


class SerializableFunction(abc.ABC):
    """Function that can be serialized/deserialized to/from json."""

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def name(self,) -> str:
        pass


class Aggregator(SerializableFunction):
    """Function that can aggregate many metrics across predictions."""

    @abc.abstractmethod
    def __call__(self, array: np.ndarray, **kwargs) -> List[float]:
        pass


class RowMean(Aggregator):

    def __call__(self, array: np.ndarray, **kwargs) -> np.ndarray:
        return array.mean(axis=0).tolist()

    def serialize(self) -> Dict[str, Any]:
        return {'name': self.name}

    @property
    def name(self,) -> str:
        return 'RowMean'


class Metric(SerializableFunction):

    @abc.abstractmethod
    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def aggregators(self,) -> List[Aggregator]:
        pass

    @property
    @abc.abstractmethod
    def shape(self,) -> str:
        pass


def desired_number_of_modes(results: np.ndarray,
                            k_to_report: List[int]) -> np.ndarray:
    """Ensures we return len(k_to_report) values even when results
    has less modes than what we want."""
    return results[:, [min(k, results.shape[1]) - 1 for k in k_to_report]]


class MinADEK(Metric):

    def __init__(self, k_to_report: List[int], aggregators: List[Aggregator]):
        super().__init__()
        self.k_to_report = k_to_report
        self._aggregators = aggregators

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = min_ade_k(prediction.prediction, ground_truth, prediction.probabilities)
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {'k_to_report': self.k_to_report,
                'name': self.name,
                'aggregators': [agg.serialize() for agg in self.aggregators]}

    @property
    def aggregators(self,) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return 'MinADEK'

    @property
    def shape(self):
        return len(self.k_to_report)


class MinFDEK(Metric):

    def __init__(self, k_to_report, aggregators: List[Aggregator]):
        super().__init__()
        self.k_to_report = k_to_report
        self._aggregators = aggregators

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = min_fde_k(prediction.prediction, ground_truth, prediction.probabilities)
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {'k_to_report': self.k_to_report,
                'name': self.name,
                'aggregators': [agg.serialize() for agg in self.aggregators]}

    @property
    def aggregators(self,) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "MinFDEK"

    @property
    def shape(self):
        return len(self.k_to_report)


class HitRateTopK(Metric):

    def __init__(self, k_to_report: List[int], aggregators: List[Aggregator],
                 tolerance: float = 2.):
        self.k_to_report = k_to_report
        self._aggregators = aggregators
        self.tolerance = tolerance

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = hit_rate_top_k(prediction.prediction, ground_truth,
                                 prediction.probabilities, self.tolerance)
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {'k_to_report': self.k_to_report,
                'name': 'HitRateTopK',
                'aggregators': [agg.serialize() for agg in self.aggregators],
                'tolerance': self.tolerance}

    @property
    def aggregators(self,) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return f"HitRateTopK_{self.tolerance}"

    @property
    def shape(self):
        return len(self.k_to_report)


class OffRoadRate(Metric):

    def __init__(self, helper: PredictHelper, aggregators: List[Aggregator]):
        self._aggregators = aggregators
        self.helper = helper
        self.drivable_area_polygons = self.load_drivable_area_polygons(helper)

    @staticmethod
    def load_drivable_area_polygons(helper: PredictHelper) -> Dict[str, MultiPolygon]:
        """
        Loads the polygon representation of the drivable area for each map
        :param helper: Instance of PredictHelper
        :return: Mapping from map_name to drivable area polygon
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        polygons = {}
        for map_name, map_api in maps.items():
            drivable_area_polygons = [map_api.extract_polygon(token) for record in map_api.drivable_area
                                      for token in record['polygon_tokens']]

            # In order to query the drivable area polygon, it must be a valid polygon
            # To ensure the polygon is valid, we use the buffer method, which returns a valid new
            # polygon constructed from the points withing a given distance of the input polygon
            polygons[map_name] = MultiPolygon(drivable_area_polygons).buffer(0)

        return polygons

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
        drivable_area = self.drivable_area_polygons[map_name]

        n_violations = 0
        for mode in prediction.prediction:

            trajectory = LineString(mode)
            if not drivable_area.contains(trajectory):
                n_violations += 1

        return np.array([n_violations / prediction.prediction.shape[0]])

    def serialize(self) -> Dict[str, Any]:
        return {'name': self.name,
                'aggregators': [agg.serialize() for agg in self.aggregators]}

    @property
    def aggregators(self,) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return 'OffRoadRate'

    @property
    def shape(self):
        return 1


def DeserializeAggregator(config: Dict[str, Any]) -> Aggregator:
    """ Helper for deserializing Aggregators. """
    if config['name'] == 'RowMean':
        return RowMean()
    else:
        raise ValueError(f"Cannot deserialize Aggregator {config['name']}.")


def DeserializeMetric(config: Dict[str, Any], helper: PredictHelper) -> Metric:
    """ Helper for deserializing Metrics. """
    if config['name'] == 'MinADEK':
        return MinADEK(config['k_to_report'], [DeserializeAggregator(agg) for agg in config['aggregators']])
    elif config['name'] == 'MinFDEK':
        return MinFDEK(config['k_to_report'], [DeserializeAggregator(agg) for agg in config['aggregators']])
    elif config['name'] == 'HitRateTopK':
        return HitRateTopK(config['k_to_report'], [DeserializeAggregator(agg) for agg in config['aggregators']],
                           tolerance=config['tolerance'])
    elif config['name'] == 'OffRoadRate':
        return OffRoadRate(helper, [DeserializeAggregator(agg) for agg in config['aggregators']])
    else:
        raise ValueError(f"Cannot deserialize function {config['name']}.")


def flatten_metrics(results: Dict[str, Any], metrics: List[Metric]):
    """

    """

    metric_names = {metric.name: metric for metric in metrics}

    flattened_metrics = {}

    for metric_name, values in results.items():

        metric_class = metric_names[metric_name]

        if hasattr(metric_class, 'k_to_report'):
            for value, k in zip(values['RowMean'], metric_class.k_to_report):
                flattened_metrics[f"{metric_name}_{k}"] = value
        else:
            flattened_metrics[metric_name] = values['RowMean']

    return flattened_metrics
