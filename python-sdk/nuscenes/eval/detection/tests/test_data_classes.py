# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json
import unittest

from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import MetricData, EvalBox, EvalBoxes, MetricDataList, DetectionConfig, \
    DetectionMetrics


class TestEvalBox(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """
        box = EvalBox()
        recovered = EvalBox.deserialize(json.loads(json.dumps(box.serialize())))
        self.assertEqual(box, recovered)


class TestEvalBoxes(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """
        boxes = EvalBoxes()
        for i in range(10):
            boxes.add_boxes(str(i), [EvalBox(), EvalBox(), EvalBox()])

        recovered = EvalBoxes.deserialize(json.loads(json.dumps(boxes.serialize())))
        self.assertEqual(boxes, recovered)


class TestMetricData(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """
        md = MetricData.random_md()
        recovered = MetricData.deserialize(json.loads(json.dumps(md.serialize())))
        self.assertEqual(md, recovered)


class TestMetricDataList(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """
        mdl = MetricDataList()
        for i in range(10):
            mdl.set('name', 0.1, MetricData.random_md())
        recovered = MetricDataList.deserialize(json.loads(json.dumps(mdl.serialize())))
        self.assertEqual(mdl, recovered)


class TestDetectionMetrics(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """

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
