# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest

import json
from nuscenes.eval.detection.data_classes import MetricData, MetricDataList


class TestMetricData(unittest.TestCase):

    def test_serialization(self):
        md1 = MetricData.random_md()
        md2 = MetricData.deserialize(json.loads(json.dumps(md1.serialize())))
        self.assertEqual(md1, md2)


class TestMetricDataList(unittest.TestCase):

    def test_serialization(self):
        mdl1 = MetricDataList()
        for i in range(10):
            mdl1.add('name', 0.1, MetricData.random_md())
        mdl2 = MetricDataList.deserialize(json.loads(json.dumps(mdl1.serialize())))
        self.assertEqual(mdl1, mdl2)


if __name__ == '__main__':
    unittest.main()
