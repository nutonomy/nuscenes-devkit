# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import os
import random
import unittest
from typing import Dict, List

import numpy as np

from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import EvalBoxes, EvalBox, MetricDataList, DetectionMetrics, MetricData
from nuscenes.eval.detection.utils import detection_name_to_rel_attributes


class TestAlgo(unittest.TestCase):

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = config_factory('cvpr_2019')

    @staticmethod
    def _mock_results(nsamples, ngt, npred, detection_name):

        def random_attr():
            """
            This is the most straight-forward way to generate a random attribute.
            Not currently used b/c we want the test fixture to be back-wards compatible.
            """
            # Get relevant attributes.
            rel_attributes = detection_name_to_rel_attributes(detection_name)

            if len(rel_attributes) == 0:
                # Empty string for classes without attributes.
                return ''
            else:
                # Pick a random attribute otherwise.
                return rel_attributes[np.random.randint(0, len(rel_attributes))]

        pred = EvalBoxes()
        gt = EvalBoxes()

        for sample_itt in range(nsamples):

            this_gt = []

            for box_itt in range(ngt):

                this_gt.append(EvalBox(
                    sample_token=str(sample_itt),
                    translation=tuple(list(np.random.rand(2)*15) + [0.0]),
                    size=tuple(np.random.rand(3)*4),
                    rotation=tuple(np.random.rand(4)),
                    velocity=tuple(np.random.rand(3)[:2]*4),
                    detection_name=detection_name,
                    detection_score=random.random(),
                    attribute_name=random_attr(),
                    ego_dist=random.random()*10,
                ))
            gt.add_boxes(str(sample_itt), this_gt)

        for sample_itt in range(nsamples):
            this_pred = []

            for box_itt in range(npred):

                this_pred.append(EvalBox(
                    sample_token=str(sample_itt),
                    translation=tuple(list(np.random.rand(2) * 10) + [0.0]),
                    size=tuple(np.random.rand(3) * 4),
                    rotation=tuple(np.random.rand(4)),
                    velocity=tuple(np.random.rand(3)[:2] * 4),
                    detection_name=detection_name,
                    detection_score=random.random(),
                    attribute_name=random_attr(),
                    ego_dist=random.random() * 10,
                ))

            pred.add_boxes(str(sample_itt), this_pred)

        return gt, pred

    def test_weighted_sum(self):
        """
        This tests runs the full evaluation for an arbitrary random set of predictions.
        """

        random.seed(42)
        np.random.seed(42)

        mdl = MetricDataList()
        for class_name in self.cfg.class_names:
            gt, pred = self._mock_results(30, 3, 25, class_name)
            for dist_th in self.cfg.dist_ths:
                mdl.set(class_name, dist_th, accumulate(gt, pred, class_name, 'center_distance', 2))

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                ap = calc_ap(mdl[(class_name, dist_th)], self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = mdl[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        self.assertEqual(0.0862032595184608, metrics.weighted_sum)

    def test_calc_tp(self):
        """Test for calc_tp()."""

        random.seed(42)
        np.random.seed(42)

        md = MetricData.random_md()

        # min_recall greater than 1.
        self.assertEqual(1.0, calc_tp(md, min_recall=1, metric_name='trans_err'))

    def test_calc_ap(self):
        """Test for calc_ap()."""

        random.seed(42)
        np.random.seed(42)

        md = MetricData.random_md()

        # Negative min_recall and min_precision
        self.assertRaises(AssertionError, calc_ap, md, -0.5, 0.4)
        self.assertRaises(AssertionError, calc_ap, md, 0.5, -0.8)

        # More than 1 min_precision/min_recall
        self.assertRaises(AssertionError, calc_ap, md, 0.7, 1)
        self.assertRaises(AssertionError, calc_ap, md, 1.2, 0)


class TestAPSimple(unittest.TestCase):
    """ Tests the correctness of AP calculation for simple cases. """

    def setUp(self):
        self.car1 = {'trans': (1, 1, 1), 'size': (2, 4, 2), 'rot': (0, 0, 0, 0), 'name': 'car', 'score': 1.0}
        self.car2 = {'trans': (3, 3, 1), 'size': (2, 4, 2), 'rot': (0, 0, 0, 0), 'name': 'car', 'score': 0.7}
        self.bicycle1 = {'trans': (5, 5, 1), 'size': (2, 2, 2), 'rot': (0, 0, 0, 0), 'name': 'bicycle', 'score': 1.0}
        self.bicycle2 = {'trans': (7, 7, 1), 'size': (2, 2, 2), 'rot': (0, 0, 0, 0), 'name': 'bicycle', 'score': 0.7}

    def check_ap(self, gts: Dict[str, List[Dict]],
                 preds: Dict[str, List[Dict]],
                 target_ap: float,
                 detection_name: str = 'car',
                 dist_th: float = 2.0,
                 min_precision: float = 0.1,
                 min_recall: float = 0.1) -> None:
        """
        Calculate and check the AP value.
        :param gts: Ground truth data.
        :param preds: Predictions.
        :param target_ap: Expected Average Precision value.
        :param detection_name: Name of the class we are interested in.
        :param dist_th: Distance threshold for matching.
        :param min_precision: Minimum precision value.
        :param min_recall: Minimum recall value.
        """
        # Create GT EvalBoxes instance.
        gt_eval_boxes = EvalBoxes()
        for sample_token, data in gts.items():
            gt_boxes = []
            for gt in data:
                eb = EvalBox(sample_token=sample_token, translation=gt['trans'], size=gt['size'], rotation=gt['rot'],
                             detection_name=gt['name'])
                gt_boxes.append(eb)

            gt_eval_boxes.add_boxes(sample_token, gt_boxes)

        # Create Predictions EvalBoxes instance.
        pred_eval_boxes = EvalBoxes()
        for sample_token, data in preds.items():
            pred_boxes = []
            for pred in data:
                eb = EvalBox(sample_token=sample_token, translation=pred['trans'], size=pred['size'],
                             rotation=pred['rot'], detection_name=pred['name'], detection_score=pred['score'])
                pred_boxes.append(eb)
            pred_eval_boxes.add_boxes(sample_token, pred_boxes)

        metric_data = accumulate(gt_eval_boxes, pred_eval_boxes, class_name=detection_name,
                                 dist_fcn_name='center_distance', dist_th=dist_th)

        ap = calc_ap(metric_data, min_precision=min_precision, min_recall=min_recall)

        # We quantize the curve into 100 bins to calculate integral so the AP is accurate up to 1%.
        self.assertGreaterEqual(0.01, abs(ap - target_ap), msg='Incorrect AP')

    def test_no_data(self):
        """ Test empty ground truth and/or predictions. """

        gts = {'sample1': [self.car1]}
        preds = {'sample1': [self.car1]}
        empty = {'sample1': []}

        # No ground truth objects (all False positives)
        self.check_ap(empty, preds, target_ap=0.0)

        # No predictions (all False negatives)
        self.check_ap(gts, empty, target_ap=0.0)

        # No predictions and no ground truth objects.
        self.check_ap(empty, empty, target_ap=0.0)

    def test_one_img(self):
        """ Test perfect detection. """
        # Perfect detection.
        self.check_ap({'sample1': [self.car1]},
                      {'sample1': [self.car1]},
                      target_ap=1.0, detection_name='car')

        # Detect one of the two objects
        self.check_ap({'sample1': [self.car1, self.car2]},
                      {'sample1': [self.car1]},
                      target_ap=0.4/0.9, detection_name='car')

        # One detection and one FP. FP score is less than TP score.
        self.check_ap({'sample1': [self.car1]},
                      {'sample1': [self.car1, self.car2]},
                      target_ap=1.0, detection_name='car')

        # One detection and one FP. FP score is more than TP score.
        self.check_ap({'sample1': [self.car2]},
                      {'sample1': [self.car1, self.car2]},
                      target_ap=((0.8*0.4)/2)/(0.9*0.9), detection_name='car')

        # FP but different class.
        self.check_ap({'sample1': [self.car1]},
                      {'sample1': [self.car1, self.bicycle1]},
                      target_ap=1.0, detection_name='car')

    def test_two_imgs(self):
        # Objects in both samples are detected.
        self.check_ap({'sample1': [self.car1], 'sample2': [self.car2]},
                      {'sample1': [self.car1], 'sample2': [self.car2]},
                      target_ap=1.0, detection_name='car')

        # Object in first sample is detected, second sample is empty.
        self.check_ap({'sample1': [self.car1], 'sample2': []},
                      {'sample1': [self.car1], 'sample2': []},
                      target_ap=1.0, detection_name='car')

        # Perfect detection in one image, FN in other.
        self.check_ap({'sample1': [self.car1], 'sample2': [self.car2]},
                      {'sample1': [self.car1], 'sample2': []},
                      target_ap=0.4/0.9, detection_name='car')


if __name__ == '__main__':
    unittest.main()