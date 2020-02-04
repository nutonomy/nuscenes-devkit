# nuScenes dev-kit.
# Code written by Freddy Boulton, Eric Wolff 2020.
from typing import List, Dict, Any

from nuscenes.eval.predict.metrics import Metric, DeserializeMetric


class PredictionConfig:
    """ Data class that specifies the prediction evaluation settings. """

    def __init__(self,
                 metrics: List[Metric],
                 seconds: int = 6,
                 frequency: int = 2):
        self.metrics = metrics
        self.seconds = seconds
        self.frequency = 2

    def serialize(self) -> Dict[str, Any]:
        """ Serialize instance into json-friendly format. """

        return {'metrics': [metric.serialize() for metric in self.metrics],
                'seconds': self.seconds}

    @classmethod
    def deserialize(cls, content: Dict[str, Any]):
        """ Initialize from serialized dictionary. """
        return cls([DeserializeMetric(metric) for metric in content['metrics']],
                   seconds=content['seconds'])
