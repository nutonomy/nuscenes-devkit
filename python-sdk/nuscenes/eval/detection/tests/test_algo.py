# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.
# Licensed under the Creative Commons [see licence.txt]

import os
import random
import unittest
from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

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

    def test_nd_score(self):
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

        self.assertEqual(0.08606662159639042, metrics.nd_score)

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


def get_metric_data(gts: Dict[str, List[Dict]],
                    preds: Dict[str, List[Dict]],
                    detection_name: str,
                    dist_th: float) -> MetricData:
        """
        Calculate and check the AP value.
        :param gts: Ground truth data.
        :param preds: Predictions.
        :param detection_name: Name of the class we are interested in.
        :param dist_th: Distance threshold for matching.
        """

        # Some or all of the defaults will be replaced by if given.
        defaults = {'trans': (0, 0, 0), 'size': (1, 1, 1), 'rot': (0, 0, 0, 0),
                    'vel': (0, 0), 'attr': 'vehicle.parked', 'score': -1.0, 'name': 'car'}
        # Create GT EvalBoxes instance.
        gt_eval_boxes = EvalBoxes()
        for sample_token, data in gts.items():
            gt_boxes = []
            for gt in data:
                gt = {**defaults, **gt}  # The defaults will be replaced by gt if given.
                eb = EvalBox(sample_token=sample_token, translation=gt['trans'], size=gt['size'], rotation=gt['rot'],
                             detection_name=gt['name'], attribute_name=gt['attr'], velocity=gt['vel'])
                gt_boxes.append(eb)

            gt_eval_boxes.add_boxes(sample_token, gt_boxes)

        # Create Predictions EvalBoxes instance.
        pred_eval_boxes = EvalBoxes()
        for sample_token, data in preds.items():
            pred_boxes = []
            for pred in data:
                pred = {**defaults, **pred}
                eb = EvalBox(sample_token=sample_token, translation=pred['trans'], size=pred['size'],
                             rotation=pred['rot'], detection_name=pred['name'], detection_score=pred['score'],
                             velocity=pred['vel'], attribute_name=pred['attr'])
                pred_boxes.append(eb)
            pred_eval_boxes.add_boxes(sample_token, pred_boxes)

        metric_data = accumulate(gt_eval_boxes, pred_eval_boxes, class_name=detection_name,
                                 dist_fcn_name='center_distance', dist_th=dist_th)

        return metric_data


class TestAPSimple(unittest.TestCase):
    """ Tests the correctness of AP calculation for simple cases. """

    def setUp(self):
        self.car1 = {'trans': (1, 1, 1), 'name': 'car', 'score': 1.0, }
        self.car2 = {'trans': (3, 3, 1), 'name': 'car', 'score': 0.7}
        self.bicycle1 = {'trans': (5, 5, 1), 'name': 'bicycle', 'score': 1.0}
        self.bicycle2 = {'trans': (7, 7, 1), 'name': 'bicycle', 'score': 0.7}

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
        metric_data = get_metric_data(gts, preds, detection_name, dist_th)
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

    def test_one_sample(self):
        """ Test the single sample case. """
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

    def test_two_samples(self):
        """ Test more than one sample case. """
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


class TestTPSimple(unittest.TestCase):
    """ Tests the correctness of true positives metrics calculation for simple cases. """

    def setUp(self):

        self.car3 = {'trans': (3, 3, 1), 'size': (2, 4, 2), 'rot': Quaternion(axis=(0, 0, 1), angle=0), 'score': 1.0}
        self.car4 = {'trans': (3, 3, 1), 'size': (2, 4, 2), 'rot': Quaternion(axis=(0, 0, 1), angle=0), 'score': 1.0}

    def check_tp(self, gts: Dict[str, List[Dict]],
                 preds: Dict[str, List[Dict]],
                 target_error: float,
                 metric_name: str,
                 detection_name: str = 'car',
                 min_recall: float = 0.1):
        """
        Calculate and check the AP value.
        :param gts: Ground truth data.
        :param preds: Predictions.
        :param target_error: Expected error value.
        :param metric_name: Name of the TP metric.
        :param detection_name: Name of the class we are interested in.
        :param min_recall: Minimum recall value.
        """

        metric_data = get_metric_data(gts, preds, detection_name, 2.0)  # Distance threshold for TP metrics is 2.0
        tp_error = calc_tp(metric_data, min_recall=min_recall, metric_name=metric_name)
        # We quantize the error curve into 100 bins to calculate the metric so it is only accurate up to 1%.
        self.assertGreaterEqual(0.01, abs(tp_error - target_error), msg='Incorrect {} value'.format(metric_name))

    def test_no_positives(self):
        """ Tests the error if there are no matches. The expected behaviour is to return error of 1.0. """

        # Same type of objects but are more than 2m away.
        car1 = {'trans': (1, 1, 1), 'score': 1.0}
        car2 = {'trans': (3, 3, 1), 'score': 1.0}
        bike1 = {'trans': (1, 1, 1), 'score': 1.0, 'name': 'bicycle', 'attr': 'cycle.with_rider'}
        for metric_name in TP_METRICS:
            self.check_tp({'sample1': [car1]}, {'sample1': [car2]}, target_error=1.0, metric_name=metric_name)

        # Within distance threshold away but different classes.
        for metric_name in TP_METRICS:
            self.check_tp({'sample1': [car1]}, {'sample1': [bike1]}, target_error=1.0, metric_name=metric_name)

    def test_perfect(self):
        """ Tests when everything is estimated perfectly. """

        car1 = {'trans': (1, 1, 1), 'score': 1.0}
        car2 = {'trans': (1, 1, 1), 'score': 0.3}
        for metric_name in TP_METRICS:
            # Detected with perfect score.
            self.check_tp({'sample1': [car1]}, {'sample1': [car1]}, target_error=0.0, metric_name=metric_name)

            # Detected with low score.
            self.check_tp({'sample1': [car1]}, {'sample1': [car2]}, target_error=0.0, metric_name=metric_name)

    def test_one_img(self):
        """ Test single sample case. """

        # Note all the following unit tests can be repeated to other metrics, but they are not needed.
        # The intention of these tests is to measure the calc_tp function which is common for all metrics.

        gt1 = {'trans': (1, 1, 1)}
        gt2 = {'trans': (10, 10, 1), 'size': (2, 2, 2)}
        gt3 = {'trans': (20, 20, 1), 'size': (2, 4, 2)}

        pred1 = {'trans': (1, 1, 1), 'score': 1.0}
        pred2 = {'trans': (11, 10, 1), 'size': (2, 2, 2), 'score': 0.9}
        pred3 = {'trans': (100, 10, 1), 'size': (2, 2, 2), 'score': 0.8}
        pred4 = {'trans': (20, 20, 1), 'size': (2, 4, 2), 'score': 0.7}
        pred5 = {'trans': (21, 20, 1), 'size': (2, 4, 2), 'score': 0.7}

        # one GT and one matching prediction. Object location is off by 1 meter, so error 1.
        self.check_tp({'sample1': [gt2]}, {'sample1': [pred2]}, target_error=1, metric_name='trans_err')

        # Two GT's and two detections.
        # The target is the average value of the recall vs. Error curve.
        # In this case there will three points on the curve. (0.1, 0), (0.5, 0), (1.0, 0.5).
        # (0.1, 0): Minimum recall we start from.
        # (0.5, 0): Detection with highest score has no translation error, and one of out of two objects recalled.
        # (1.0, 0.5): The last object is recalled but with 1m translation error, the cumulative mean gets to 0.5m error.
        # Error value of first segment of curve starts at 0 and ends at 0, so the average of this segment is 0.
        # Next segment of the curve starts at 0 and ends at 0.5, so the average is 0.25.
        # Then we take average of all segments and normalize it with the recall values we averaged over.
        target_error = ((0 + 0) / 2 + (0 + 0.5) / 2) / (2 * 0.9)
        self.check_tp({'sample1': [gt1, gt2]}, {'sample1': [pred1, pred2]}, target_error=target_error,
                      metric_name='trans_err')

        # Adding a false positive with smaller detection score should not affect the true positive metric.
        self.check_tp({'sample1': [gt1, gt2]}, {'sample1': [pred1, pred2, pred3]}, target_error=target_error,
                      metric_name='trans_err')

        # In this case there will four points on the curve. (0.1, 0), (0.33, 0), (0.66, 0.5) (1.0, 0.33).
        # (0.1, 0): Minimum recall we start from.
        # (0.33, 0): One of out of three objects recalled with no error.
        # (0.66, 0.5): Second object is recalled but with 1m error. Cumulative error becomes 0.5m.
        # (1.0, 0.33): Third object recalled with no error. Cumulative error becomes 0.33m.
        # First segment starts at 0 and ends at 0: average error 0.
        # Next segment starts at 0 and ends at 0.5: average error is 0.25.
        # Next segment starts at 0.5 and ends at 0.33: average error is 0.416
        # Then we take average of all segments and normalize it with the recall values we averaged over.
        target_error = ((0+0)/2 + (0+0.5)/2 + (0.5 + 0.33)/2) / (3 * 0.9)  # It is a piecewise linear with 3 segments
        self.check_tp({'sample1': [gt1, gt2, gt3]}, {'sample1': [pred1, pred2, pred4]}, target_error=target_error,
                      metric_name='trans_err')

        # Both matches have same translational error (1 meter), so the overall error is also 1 meter
        self.check_tp({'sample1': [gt2, gt3]}, {'sample1': [pred2, pred5]}, target_error=1.0,
                      metric_name='trans_err')

    def test_two_imgs(self):
        """ Test the more than one sample case. """

        # Note all the following unit tests can be repeated to other metrics, but they are not needed.
        # The intention of these tests is to measure the calc_tp function which is common for all metrics.

        gt1 = {'trans': (1, 1, 1)}
        gt2 = {'trans': (10, 10, 1), 'size': (2, 2, 2)}
        gt3 = {'trans': (20, 20, 1), 'size': (2, 4, 2)}

        pred1 = {'trans': (1, 1, 1), 'score': 1.0}
        pred2 = {'trans': (11, 10, 1), 'size': (2, 2, 2), 'score': 0.9}
        pred3 = {'trans': (100, 10, 1), 'size': (2, 2, 2), 'score': 0.8}
        pred4 = {'trans': (21, 20, 1), 'size': (2, 4, 2), 'score': 0.7}

        # One GT and one detection
        self.check_tp({'sample1': [gt2]}, {'sample1': [pred2]}, target_error=1, metric_name='trans_err')

        # Two GT's and two detections.
        # The target is the average value of the recall vs. Error curve.
        target_error = ((0 + 0) / 2 + (0 + 0.5) / 2) / (2 * 0.9)  # It is a piecewise linear with 2 segments.
        self.check_tp({'sample1': [gt1], 'sample2': [gt2]}, {'sample1': [pred1], 'sample2': [pred2]},
                      target_error=target_error, metric_name='trans_err')

        # Adding a false positive and/or an empty sample should not affect the score
        self.check_tp({'sample1': [gt1], 'sample2': [gt2], 'sample3': []},
                      {'sample1': [pred1], 'sample2': [pred2, pred3], 'sample3': []},
                      target_error=target_error, metric_name='trans_err')

        # All the detections does have same error, so the overall error is also same.
        self.check_tp({'sample1': [gt2, gt3], 'sample2': [gt3]}, {'sample1': [pred2], 'sample2': [pred4]},
                      target_error=1.0, metric_name='trans_err')


if __name__ == '__main__':
    unittest.main()
