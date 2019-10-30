import unittest
from typing import Tuple, Dict, List
import copy
from collections import defaultdict

import numpy as np

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingBox
from nuscenes.eval.tracking.loaders import interpolate_tracks


class TestAlgo(unittest.TestCase):

    @staticmethod
    def single_scene() -> Tuple[str, Dict[str, Dict[int, List[TrackingBox]]]]:
        class_name = 'car'
        box = TrackingBox(translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name,
                          tracking_score=0.5)
        timestamp_boxes_gt = {
            0: [copy.deepcopy(box)],
            1: [copy.deepcopy(box)],
            2: [copy.deepcopy(box)],
            3: [copy.deepcopy(box)]
        }
        timestamp_boxes_gt[0][0].sample_token = 'a'
        timestamp_boxes_gt[1][0].sample_token = 'b'
        timestamp_boxes_gt[2][0].sample_token = 'c'
        timestamp_boxes_gt[3][0].sample_token = 'd'
        tracks_gt = {'scene-1': timestamp_boxes_gt}

        return class_name, tracks_gt

    def test_perfect_gt(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one prediction.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fn == 0)
        assert np.all(md.fp == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_drop_prediction(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
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
        assert md.tp[3] == 3
        assert md.fp[3] == 0
        assert md.fn[3] == 1
        assert md.lgd[3] == 0.5
        assert md.tid[3] == 0
        assert md.frag[3] == 1
        assert md.ids[3] == 0

    def test_identity_switch(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        timestamp_boxes_pred[2][0].tracking_id = 'tb'
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert md.tp[5] == 2
        assert md.fp[5] == 0
        assert md.fn[5] == 0
        assert md.lgd[5] == 0  # TODO: Fix this. IDS should count as misses in LGD and TID.
        assert md.tid[5] == 0
        assert md.frag[5] == 0
        assert md.ids[5] == 2  # One wrong id leads to 2 identity switches.

    def test_drop_gt(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one GT box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_gt['scene-1'][1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 3)
        assert np.all(md.fp == 1)
        assert np.all(md.fn == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_drop_gt_interpolate(self):

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one GT box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_gt['scene-1'][1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Interpolate to "restore" dropped GT.
        tracks_gt['scene-1'] = interpolate_tracks(defaultdict(list, tracks_gt['scene-1']))

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fp == 0)
        assert np.all(md.fn == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.ids == 0)


if __name__ == '__main__':
    unittest.main()
