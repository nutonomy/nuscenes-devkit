import unittest

import numpy as np

from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingBox
from nuscenes.eval.common.config import config_factory


class TestAlgo(unittest.TestCase):

    @staticmethod
    def single_scene():
        class_name = 'car'
        timestamp_boxes_gt = {
            0: [TrackingBox(sample_token='a', translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name, tracking_score=0.5)],
            1: [TrackingBox(sample_token='b', translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name, tracking_score=0.5)],
            2: [TrackingBox(sample_token='c', translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name, tracking_score=0.5)],
            3: [TrackingBox(sample_token='d', translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name, tracking_score=0.5)],
        }
        tracks_gt = {'scene-1': timestamp_boxes_gt}

        return class_name, tracks_gt

    def test_perfect_gt(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one prediction.
        timestamp_boxes_pred = tracks_gt['scene-1'].copy()
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.ml == 0)
        assert np.all(md.fn == 0)
        assert np.all(md.fp == 0)

    def test_drop_prediction(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = tracks_gt['scene-1'].copy()
        timestamp_boxes_pred[1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # Recall values above 0.75 (3/4 correct) are not achieved and therefore nan.
        assert np.all(np.isnan(md.confidence[md.recall_hypo > 0.75]))
        assert np.all(md.tp[3] == 3)
        assert np.all(md.fp[3] == 0)
        assert np.all(md.fn[3] == 1)

    def test_drop_gt(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one GT box.
        timestamp_boxes_pred = tracks_gt['scene-1'].copy()
        tracks_gt['scene-1'][1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # We only look at the 100% recall value.
        assert np.all(md.tp[0] == 3)
        assert np.all(md.fp[0] == 1)
        assert np.all(md.fn[0] == 0)


if __name__ == '__main__':
    TestAlgo().test_perfect_gt()
