# nuScenes dev-kit.
# Code written by Sourabh Vora, 2019.
# Licensed under the Creative Commons [see licence.txt]

import os
import unittest

from nuscenes import NuScenes
from nuscenes.eval.detection.loaders import filter_eval_boxes
from nuscenes.eval.detection.data_classes import EvalBox, EvalBoxes


class TestLoader(unittest.TestCase):
    def test_filter_eval_boxes(self):
        """
        This tests runs the evaluation for an arbitrary random set of predictions.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        sample_token = '0af0feb5b1394b928dd13d648de898f5'
        # This sample has a bike rack instance 'bfe685042aa34ab7b2b2f24ee0f1645f' with these parameters
        # 'translation': [683.681, 1592.002, 0.809],
        # 'size': [1.641, 14.465, 1.4],
        # 'rotation': [0.3473693995546558, 0.0, 0.0, 0.9377283723195315]

        max_dist = {'car': 50,
                    'truck': 50,
                    'bus': 50,
                    'trailer': 50,
                    'construction_vehicle': 50,
                    'pedestrian': 40,
                    'motorcycle': 40,
                    'bicycle': 40,
                    'traffic_cone': 30,
                    'barrier': 30}

        # Test bicycle filtering by creating a box at the same position as the bike rack.
        box1 = EvalBox(sample_token=sample_token,
                       translation=(683.681, 1592.002, 0.809),
                       size=(1, 1, 1),
                       detection_name='bicycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes('0af0feb5b1394b928dd13d648de898f5', [box1])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 0)    # box1 should be filtered.

        # Test motorcycle filtering by creating a box at the same position as the bike rack.
        box2 = EvalBox(sample_token=sample_token,
                       translation=(683.681, 1592.002, 0.809),
                       size=(1, 1, 1),
                       detection_name='motorcycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes('0af0feb5b1394b928dd13d648de898f5', [box1, box2])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 0)    # both box1 and box2 should be filtered.

        # Now create a car at the same position as the bike rack.
        box3 = EvalBox(sample_token=sample_token,
                       translation=(683.681, 1592.002, 0.809),
                       size=(1, 1, 1),
                       detection_name='car')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes('0af0feb5b1394b928dd13d648de898f5', [box1, box2, box3])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 1)  # box1 and box2 to be filtered. box3 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')

        # Now add a bike outside the bike rack.

        box4 = EvalBox(sample_token=sample_token,
                       translation=(68.681, 1592.002, 0.809),
                       size=(1, 1, 1),
                       detection_name='bicycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes('0af0feb5b1394b928dd13d648de898f5', [box1, box2, box3, box4])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 2)  # box1, box2 to be filtered. box3, box4 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].detection_name, 'bicycle')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].translation[0], 68.681)

        # Finally add another bike on the bike rack center but set the ego_dist higher than what's defined in max_dist
        box5 = EvalBox(sample_token=sample_token,
                       translation=(683.681, 1592.002, 0.809),
                       size=(1, 1, 1),
                       detection_name='bicycle',
                       ego_dist=100.0)

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes('0af0feb5b1394b928dd13d648de898f5', [box1, box2, box3, box4, box5])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)
        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 2)  # box1, box2 to be filtered. box3, box4 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].detection_name, 'bicycle')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].translation[0], 68.681)


if __name__ == '__main__':
    unittest.main()
