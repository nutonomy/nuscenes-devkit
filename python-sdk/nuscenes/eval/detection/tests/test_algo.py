# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import os
import random
import unittest

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

    def test_ap(self):
        """
        This tests runs the ap calculation for an arbitrary random set of predictions.
        """
        random.seed(42)
        np.random.seed(42)

        detection_name = 'barrier'
        gt, pred = self._mock_results(100, 3, 250, detection_name)
        metrics = accumulate(gt, pred, detection_name, 'center_distance', 2)
        ap = calc_ap(metrics, self.cfg.min_recall, self.cfg.min_precision)
        self.assertEqual(ap, 7.794866035607614e-06)

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
                metrics.add_label_ap(class_name, ap)

            for metric_name in TP_METRICS:
                metric_data = mdl[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        self.assertEqual(0.10063518713627559, metrics.weighted_sum)

    def test_calc_tp(self):
        """Test for calc_tp()."""

        random.seed(42)
        np.random.seed(42)

        md = MetricData.random_md()

        self.assertEqual(0.5389758924116305, calc_tp(md, min_recall=0.4, metric_name='orient_err'))

        # min_recall greater than 1.
        self.assertEqual(1.0, calc_tp(md, min_recall=1, metric_name='trans_err'))

    def test_calc_ap(self):
        """Test for calc_ap()."""

        random.seed(42)
        np.random.seed(42)

        md = MetricData.random_md()
        self.assertAlmostEqual(0.026738322081734534, calc_ap(md, min_recall=0.7, min_precision=0.8))
        self.assertAlmostEqual(0, calc_ap(md, min_recall=1.0, min_precision=0.8))

        # Negative min_recall and min_precision
        self.assertRaises(AssertionError, calc_ap, md, -0.5, 0.4)
        self.assertRaises(AssertionError, calc_ap, md, 0.5, -0.8)

        # More than 1 min_precision/min_recall
        self.assertRaises(AssertionError, calc_ap, md, 0.7, 1)
        self.assertRaises(AssertionError, calc_ap, md, 1.2, 0)

    if __name__ == '__main__':
        unittest.main()
