# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json
import unittest

from nuscenes.eval.detection.data_classes import MetricData, EvalBox, EvalBoxes, MetricDataList


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
