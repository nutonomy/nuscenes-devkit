# nuScenes dev-kit.
# Code written by Freddy Boulton 2020.
from typing import Dict, Any

import numpy as np

from nuscenes.eval.common.data_classes import MetricData

MAX_NUMBER_OF_MODES = 25


class Prediction(MetricData):
    """
    Stores predictions of Models.
    Metrics are calculated from Predictions.

    Attributes:
        instance: Instance token for prediction.
        sample: Sample token for prediction.
        prediction: Prediction of model [num_modes, n_timesteps, state_dim].
        probabilities: Probabilities of each mode [num_modes].
    """
    def __init__(self, instance: str, sample: str, prediction: np.ndarray,
                 probabilities: np.ndarray):
        self.is_valid(instance, sample, prediction, probabilities)

        self.instance = instance
        self.sample = sample
        self.prediction = prediction
        self.probabilities = probabilities

    @property
    def number_of_modes(self) -> int:
        return self.prediction.shape[0]

    def serialize(self):
        """ Serialize to json. """
        return {'instance': self.instance,
                'sample': self.sample,
                'prediction': self.prediction.tolist(),
                'probabilities': self.probabilities.tolist()}

    @classmethod
    def deserialize(cls, content: Dict[str, Any]):
        """ Initialize from serialized content. """
        return cls(instance=content['instance'],
                   sample=content['sample'],
                   prediction=np.array(content['prediction']),
                   probabilities=np.array(content['probabilities']))

    @staticmethod
    def is_valid(instance, sample, prediction, probabilities):
        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"Error: prediction must be of type np.ndarray. Received {str(type(prediction))}.")
        if not isinstance(probabilities, np.ndarray):
            raise ValueError(f"Error: probabilities must be of type np.ndarray. Received {type(probabilities)}.")
        if not isinstance(instance, str):
            raise ValueError(f"Error: instance token must be of type string. Received {type(instance)}")
        if not isinstance(sample, str):
            raise ValueError(f"Error: sample token must be of type string. Received {type(sample)}.")
        if prediction.ndim != 3:
            raise ValueError("Error: prediction must have three dimensions (number of modes, number of timesteps, 2).\n"
                             f"Received {prediction.ndim}")
        if probabilities.ndim != 1:
            raise ValueError(f"Error: probabilities must be a single dimension. Received {probabilities.ndim}.")
        if len(probabilities) != prediction.shape[0]:
            raise ValueError("Error: there must be the same number of probabilities as predicted modes.\n"
                             f"Received {len(probabilities)} probabilities and {prediction.shape[0]} modes.")
        if prediction.shape[0] > MAX_NUMBER_OF_MODES:
            raise ValueError(f"Error: prediction contains more than {MAX_NUMBER_OF_MODES} modes.")

    def __repr__(self):
        return f"Prediction(instance={self.instance}, sample={self.sample},"\
               f" prediction={self.prediction}, probabilities={self.probabilities})"

