from typing import List, Dict, Any


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self, content: Dict[str, Any]):

        self.range = content['range']
        self.dist_fcn = content['dist_fcn']
        self.dist_ths = content['dist_ths']
        self.dist_th_tp = content['dist_th_tp']
        self.metric_bounds = content['metric_bounds']
        self.attributes = content['attributes']
        self.recall_range = content['recall_range']
        self.weighted_sum_tp_metrics = content['weighted_sum_tp_metrics']
        self.max_boxes_per_sample = content['max_boxes_per_sample']
        self.mean_ap_weight = content['mean_ap_weight']
        self.class_names = content['class_names']

        self.metric_names = self.metric_bounds.keys()


class EvalBox:
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: List[float] = None,
                 size: List[float] = None,
                 rotation: List[float] = None,
                 velocity: List[float] = None,
                 detection_name: str = None,
                 detection_score: float = None,
                 attribute_scores: List[float] = None,
                 attribute_labels: List[float] = None,
                 ego_dist: float = None):

        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_scores = attribute_scores
        self.attribute_labels = attribute_labels
        self.ego_dist = ego_dist

    def __repr__(self):
        return self.detection_name

    def serialize(self):
        pass  # TODO: write

    @classmethod
    def deserialize(cls, content):
        # TODO type-checking
        return cls(content['sample_token'],
                   content['translation'],
                   content['size'],
                   content['rotation'],
                   content['velocity'],
                   content['detection_name'],
                   content['detection_score'],
                   content['attribute_scores'])


class EvalBoxes:
    """ Data class that groups EvalBox instances by sample """

    def __init__(self):
        self.boxes = {}

    def __repr__(self):
        return "{} boxes across {} samples".format(123, len(self.boxes))

    def __getitem__(self, item):
        return self.boxes[item]

    @property
    def all(self) -> List[EvalBox]:
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self):
        return list(self.boxes.keys())

    def add_boxes(self, sample_token, boxes):
        self.boxes[sample_token] = boxes

    def serialize(self):
        pass  # TODO: write

    @classmethod
    def deserialize(cls, content):
        # TODO type-checking
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [EvalBox.deserialize(box) for box in boxes])
        return eb
