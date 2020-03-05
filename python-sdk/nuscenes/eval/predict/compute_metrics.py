# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for computing metrics for a submission to the nuscenes prediction challenge. """

import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.predict.config import PredictionConfig, load_prediction_config
from nuscenes.eval.predict.data_classes import Prediction
from nuscenes.predict import PredictHelper


def compute_metrics(predictions: List[Dict[str, Any]],
                    helper: PredictHelper, config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: Unserialized predictions in json file.
    :param helper: Instance of PredictHelper that wraps the nuScenes test set.
    :param config: Config file.
    """
    # TODO: Add check that n_preds is same size as test set once that is finalized
    n_preds = len(predictions)

    containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}

    for i, prediction_str in enumerate(predictions):

        prediction = Prediction.deserialize(prediction_str)
        ground_truth = helper.get_future_for_agent(prediction.instance, prediction.sample,
                                                   config.seconds, in_agent_frame=False)

        for metric in config.metrics:
            containers[metric.name][i] = metric(ground_truth, prediction)

    aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for metric in config.metrics:
        for agg in metric.aggregators:
            aggregations[metric.name][agg.name] = agg(containers[metric.name])

    return aggregations


def main(version: str, data_root: str, submission_path: str, submission_name: str,
         config_name: str = 'predict_2020_icra.json') -> None:
    """
    Computes metrics for a submission stored in submission_path with a given submission_name with the metrics
    specified by the config_name.
    :param version: NuScenes dataset version.
    :param data_root: Directory storing NuScenes data.
    :param submission_path: Directory storing submission.
    :param submission_name: Name of json file in submission_path directory that stores
        predictions.
    :param config_name: Name of config file.
    """

    predictions = json.load(open(os.path.join(submission_path, f"{submission_name}_inference.json"), "r"))
    nusc = NuScenes(version=version, dataroot=data_root)
    helper = PredictHelper(nusc)
    config = load_prediction_config(helper, config_name)

    results = compute_metrics(predictions, helper, config)
    json.dump(results, open(os.path.join(submission_path, f"{submission_name}_metrics.json"), "w"))
