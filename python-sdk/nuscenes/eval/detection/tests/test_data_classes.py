# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2019.

import json
import os
import unittest

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList


class TestDetectionConfig(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """

        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_name = 'detection_cvpr_2019'
        config_path = os.path.join(this_dir, '..', 'configs', cfg_name + '.json')

        with open(config_path) as f:
            cfg = json.load(f)

        detect_cfg = DetectionConfig.deserialize(cfg)

        self.assertEqual(cfg, detect_cfg.serialize())

        recovered = DetectionConfig.deserialize(json.loads(json.dumps(detect_cfg.serialize())))
        self.assertEqual(detect_cfg, recovered)


class TestDetectionBox(unittest.TestCase):

    def test_serialization(self):
        """ Test that instance serialization protocol works with json encoding. """
        box = DetectionBox()
        recovered = DetectionBox.deserialize(json.loads(json.dumps(box.serialize())))
        self.assertEqual(box, recovered)


class TestEvalBoxes(unittest.TestCase):

    def test_serialization(self):
        """ Test that instance serialization protocol works with json encoding. """
        boxes = EvalBoxes()
        for i in range(10):
            boxes.add_boxes(str(i), [DetectionBox(), DetectionBox(), DetectionBox()])

        recovered = EvalBoxes.deserialize(json.loads(json.dumps(boxes.serialize())), DetectionBox)
        self.assertEqual(boxes, recovered)


class TestMetricData(unittest.TestCase):

    def test_serialization(self):
        """ Test that instance serialization protocol works with json encoding. """
        md = DetectionMetricData.random_md()
        recovered = DetectionMetricData.deserialize(json.loads(json.dumps(md.serialize())))
        self.assertEqual(md, recovered)


class TestDetectionMetricDataList(unittest.TestCase):

    def test_serialization(self):
        """ Test that instance serialization protocol works with json encoding. """
        mdl = DetectionMetricDataList()
        for i in range(10):
            mdl.set('name', 0.1, DetectionMetricData.random_md())
        recovered = DetectionMetricDataList.deserialize(json.loads(json.dumps(mdl.serialize())))
        self.assertEqual(mdl, recovered)


class TestDetectionMetrics(unittest.TestCase):

    def test_serialization(self):
        """ Test that instance serialization protocol works with json encoding. """

        cfg = {
            'class_range': {
                'car': 1.0,
                'truck': 1.0,
                'bus': 1.0,
                'trailer': 1.0,
                'construction_vehicle': 1.0,
                'pedestrian': 1.0,
                'motorcycle': 1.0,
                'bicycle': 1.0,
                'traffic_cone': 1.0,
                'barrier': 1.0
            },
            'dist_fcn': 'distance',
            'dist_ths': [0.0, 1.0],
            'dist_th_tp': 1.0,
            'min_recall': 0.0,
            'min_precision': 0.0,
            'max_boxes_per_sample': 1,
            'mean_ap_weight': 1.0
        }
        detect_config = DetectionConfig.deserialize(cfg)

        metrics = DetectionMetrics(cfg=detect_config)

        for i, name in enumerate(cfg['class_range'].keys()):
            metrics.add_label_ap(name, 1.0, float(i))
            for j, tp_name in enumerate(TP_METRICS):
                metrics.add_label_tp(name, tp_name, float(j))

        serialized = json.dumps(metrics.serialize())
        deserialized = DetectionMetrics.deserialize(json.loads(serialized))

        self.assertEqual(metrics, deserialized)


if __name__ == '__main__':
    unittest.main()
