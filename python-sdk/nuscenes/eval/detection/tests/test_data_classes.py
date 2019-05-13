# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json
import os
import unittest

from nuscenes.eval.detection.data_classes import MetricData, EvalBox, EvalBoxes, MetricDataList, DetectionConfig


class TestDetectionConfig(unittest.TestCase):

    def test_serialization(self):
        """ test that instance serialization protocol works with json encoding """

        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_name = 'cvpr_2019.json'
        config_path = os.path.join(this_dir, '../configs/{}'.format(cfg_name))

        cfg = json.load(open(config_path))

        detect_cfg = DetectionConfig.deserialize(cfg)

        self.assertEqual(cfg, detect_cfg.serialize())

        recovered = DetectionConfig.deserialize(json.loads(json.dumps(detect_cfg.serialize())))
        self.assertEqual(detect_cfg, recovered)


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


if __name__ == '__main__':
    unittest.main()
