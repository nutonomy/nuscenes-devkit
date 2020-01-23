import dataclasses
import numpy as np

@dataclasses.dataclass
class Prediction:
    instance: str
    sample: str
    prediction: np.ndarray
    probabilities: np.ndarray

    def serialize(self):
        return {'instance': self.instance,
                'sample': self.sample,
                'prediction': self.prediction.tolist(),
                'probabilities': self.probabilities.tolist()}


