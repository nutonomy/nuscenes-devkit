# nuScenes dev-kit.
# Code written by Sourabh Vora, 2019.

import os
import unittest

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import _get_box_class_field


class TestLoader(unittest.TestCase):
    def test_filter_eval_boxes(self):
        """
        This tests runs the evaluation for an arbitrary random set of predictions.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """
        # Get the maximum distance from the config
        cfg = config_factory('detection_cvpr_2019')
        max_dist = cfg.class_range

        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        sample_token = '0af0feb5b1394b928dd13d648de898f5'
        # This sample has a bike rack instance 'bfe685042aa34ab7b2b2f24ee0f1645f' with these parameters
        # 'translation': [683.681, 1592.002, 0.809],
        # 'size': [1.641, 14.465, 1.4],
        # 'rotation': [0.3473693995546558, 0.0, 0.0, 0.9377283723195315]

        # Test bicycle filtering by creating a box at the same position as the bike rack.
        box1 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 0)    # box1 should be filtered.

        # Test motorcycle filtering by creating a box at the same position as the bike rack.
        box2 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='motorcycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 0)    # both box1 and box2 should be filtered.

        # Now create a car at the same position as the bike rack.
        box3 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='car')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2, box3])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 1)  # box1 and box2 to be filtered. box3 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')

        # Now add a bike outside the bike rack.
        box4 = DetectionBox(sample_token=sample_token,
                            translation=(68.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle')

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2, box3, box4])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)

        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 2)  # box1, box2 to be filtered. box3, box4 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].detection_name, 'bicycle')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].translation[0], 68.681)

        # Add another bike on the bike rack center,
        # but set the ego_dist (derived from ego_translation) higher than what's defined in max_dist
        box5 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            ego_translation=(100.0, 0.0, 0.0))

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2, box3, box4, box5])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)
        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 2)  # box1, box2, box5 filtered. box3, box4 to stay.
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].detection_name, 'bicycle')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].translation[0], 68.681)

        # Add another bike on the bike rack center but set the num_pts to be zero so that it gets filtered.
        box6 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            num_pts=0)

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2, box3, box4, box5, box6])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)
        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 2)  # box1, box2, box5, box6 filtered. box3, box4 stay
        self.assertEqual(filtered_boxes.boxes[sample_token][0].detection_name, 'car')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].detection_name, 'bicycle')
        self.assertEqual(filtered_boxes.boxes[sample_token][1].translation[0], 68.681)

        # Check for a sample where there are no bike racks. Everything should be filtered correctly.
        sample_token = 'ca9a282c9e77460f8360f564131a8af5'   # This sample has no bike-racks.

        box1 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            ego_translation=(25.0, 0.0, 0.0))

        box2 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='motorcycle',
                            ego_translation=(45.0, 0.0, 0.0))

        box3 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='car',
                            ego_translation=(45.0, 0.0, 0.0))

        box4 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='car',
                            ego_translation=(55.0, 0.0, 0.0))

        box5 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            num_pts=1)

        box6 = DetectionBox(sample_token=sample_token,
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            num_pts=0)

        eval_boxes = EvalBoxes()
        eval_boxes.add_boxes(sample_token, [box1, box2, box3, box4, box5, box6])

        filtered_boxes = filter_eval_boxes(nusc, eval_boxes, max_dist)
        self.assertEqual(len(filtered_boxes.boxes[sample_token]), 3)  # box2, box4, box6 filtered. box1, box3, box5 stay
        self.assertEqual(filtered_boxes.boxes[sample_token][0].ego_dist, 25.0)
        self.assertEqual(filtered_boxes.boxes[sample_token][1].ego_dist, 45.0)
        self.assertEqual(filtered_boxes.boxes[sample_token][2].num_pts, 1)

    def test_get_box_class_field(self):
        eval_boxes = EvalBoxes()
        box1 = DetectionBox(sample_token='box1',
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='bicycle',
                            ego_translation=(25.0, 0.0, 0.0))

        box2 = DetectionBox(sample_token='box2',
                            translation=(683.681, 1592.002, 0.809),
                            size=(1, 1, 1),
                            detection_name='motorcycle',
                            ego_translation=(45.0, 0.0, 0.0))
        eval_boxes.add_boxes('sample1', [])
        eval_boxes.add_boxes('sample2', [box1, box2])

        class_field = _get_box_class_field(eval_boxes)
        self.assertEqual(class_field, 'detection_name')


if __name__ == '__main__':
    unittest.main()
