from typing import List, Callable, Dict, Any
import dataclasses
import numpy as np
from nuscenes.eval.common.data_classes import MetricData

@dataclasses.dataclass
class Prediction(MetricData):
    """Stores predictions of Models.
    Metrics are calculated from Predictions.
    :param instance: Instance token for prediction.
    :param sample: Sample token for prediction.
    :param prediction: Predicion of model [num_modes, n_timesteps, state_dim]
    :param probabilities: Probabilities of each mode [num_modes]
    """
    instance: str
    sample: str
    prediction: np.ndarray
    probabilities: np.ndarray

    @property
    def number_of_modes(self) -> int:
        return self.prediction.shape[0]

    def serialize(self):
        """Serialize to json."""
        return {'instance': self.instance,
                'sample': self.sample,
                'prediction': self.prediction.tolist(),
                'probabilities': self.probabilities.tolist()}

    @classmethod
    def deserialize(cls, content: Dict[str, Any]):
        """ Initialize from serialized content."""
        return cls(instance=content['instance'],
                   sample=content['sample'],
                   prediction=np.array(content['prediction']),
                   probabilities=np.array(content['probabilities']))

