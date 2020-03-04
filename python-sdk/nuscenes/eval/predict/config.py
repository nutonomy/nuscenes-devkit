# nuScenes dev-kit.
# Code written by Freddy Boulton, Eric Wolff 2020.
from typing import List, Dict, Any
import os
import json

from nuscenes.eval.predict.metrics import Metric, DeserializeMetric
from nuscenes.predict import PredictHelper


class PredictionConfig:
    """
    Data class that specifies the prediction evaluation settings.
    Intialized with:
    metrics: List of nuscenes.eval.predict.metric.Metric objects.
    seconds: Number of seconds to predict for each agent.
    frequency: Rate at which prediction is made, in Hz.
    """

    def __init__(self,
                 metrics: List[Metric],
                 seconds: int = 6,
                 frequency: int = 2):
        self.metrics = metrics
        self.seconds = seconds
        self.frequency = frequency  # Hz

    def serialize(self) -> Dict[str, Any]:
        """ Serialize instance into json-friendly format. """

        return {'metrics': [metric.serialize() for metric in self.metrics],
                'seconds': self.seconds}

    @classmethod
    def deserialize(cls, content: Dict[str, Any], helper: PredictHelper):
        """ Initialize from serialized dictionary. """
        return cls([DeserializeMetric(metric, helper) for metric in content['metrics']],
                   seconds=content['seconds'])


def load_prediction_config(helper: PredictHelper, config_name: str = 'predict_2020_icra.json') -> PredictionConfig:
    """
    Loads a PredictionConfig from json file stored in eval/predict/configs
    :param helper: Instance of PredictHelper. Needed for OffRoadRate metric.
    :param config_name: Name of json cofig file
    :return: PredictionConfig
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, "configs", config_name)
    assert os.path.exists(cfg_path), f'Requested unknown configuration {cfg_path}'

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        config = json.load(f)

    return PredictionConfig.deserialize(config, helper)


