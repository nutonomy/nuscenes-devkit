# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest
import random
import numpy as np

from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.utils import detection_name_to_rel_attributes
from nuscenes.eval.detection.data_classes import DetectionConfig, EvalBoxes, EvalBox, MetricDataList, DetectionMetrics


class TestEndToEnd(unittest.TestCase):

    cfg = DetectionConfig({
        "range": 40,
        "dist_fcn": "center_distance",
        "dist_ths": [2.0, 4.0],
        "dist_th_tp": 2.0,
        "metric_bounds": {
            "trans_err": 0.5,
            "vel_err": 1.5,
            "scale_err": 0.5,
            "orient_err": 1.570796,
            "attr_err": 1
        },
        "attributes": ["cycle.with_rider", "cycle.without_rider", "pedestrian.moving",
                       "pedestrian.sitting_lying_down", "pedestrian.standing", "vehicle.moving",
                       "vehicle.parked", "vehicle.stopped"],
        "min_recall": 0.1,
        "min_precision": 0.0,
        "weighted_sum_tp_metrics": ["trans_err", "scale_err", "orient_err"],
        "max_boxes_per_sample": 500,
        "mean_ap_weight": 5,
        "class_names": ["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
                        "pedestrian", "traffic_cone", "trailer", "truck"]
    })

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
                    translation=list(np.random.rand(2)*15) + [0],
                    size=list(np.random.rand(3)*4),
                    rotation=list(np.random.rand(4)),
                    velocity=list(np.random.rand(3)*4),
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
                    translation = list(np.random.rand(2)*10) + [0],
                    size=list(np.random.rand(3) * 4),
                    rotation=list(np.random.rand(4)),
                    velocity=list(np.random.rand(3) * 4),
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
        self.assertEqual(ap, 0.03790748135861538)

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
                mdl.add(class_name, dist_th, accumulate(gt, pred, class_name, 'center_distance', 2))

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                ap = calc_ap(mdl[(class_name, dist_th)], self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, ap)

            for metric_name in self.cfg.metric_names:
                tp = calc_tp(mdl[(class_name, self.cfg.dist_th_tp)], self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        self.assertEqual(metrics.weighted_sum, 0.10616416924100319)


if __name__ == '__main__':
    unittest.main()
