from nuscenes.eval.detection.data_classes import DetectionConfig


class TrackingConfig(DetectionConfig):
    """
    Dummy tracking config, analog to detection. May need to be modified later. TODO
    """


class TrackingMetrics:
    pass # TODO

    def __init__(self, cfg: TrackingConfig):

        self.cfg = cfg