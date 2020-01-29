import copy
import unittest
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingBox
from nuscenes.eval.tracking.loaders import interpolate_tracks
from nuscenes.eval.tracking.tests.scenarios import get_scenarios


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

    def test_gt_submission(self):
        """ Test with GT submission. """

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
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fn == 0)
        assert np.all(md.fp == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_empty_submission(self):
        """ Test a submission with no predictions. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove all predictions.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        for timestamp, box in timestamp_boxes_pred.items():
            timestamp_boxes_pred[timestamp] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.mota == 0)
        assert np.all(md.motar == 0)
        assert np.all(np.isnan(md.recall_hypo))
        assert np.all(md.tp == 0)
        assert np.all(md.fn == 4)
        assert np.all(np.isnan(md.fp))  # FP/Frag/IDS are nan as we there were no predictions.
        assert np.all(md.lgd == 20)
        assert np.all(md.tid == 20)
        assert np.all(np.isnan(md.frag))
        assert np.all(np.isnan(md.ids))

    def test_drop_prediction(self):
        """ Drop one prediction from the GT submission. """

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
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # Recall values above 0.75 (3/4 correct) are not achieved and therefore nan.
        first_achieved = np.where(md.recall_hypo <= 0.75)[0][0]
        assert np.all(np.isnan(md.confidence[:first_achieved]))
        assert md.tp[first_achieved] == 3
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 1
        assert md.lgd[first_achieved] == 0.5
        assert md.tid[first_achieved] == 0
        assert md.frag[first_achieved] == 1
        assert md.ids[first_achieved] == 0

    def test_drop_prediction_multiple(self):
        """  Drop the first three predictions from the GT submission. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        timestamp_boxes_pred[0] = []
        timestamp_boxes_pred[1] = []
        timestamp_boxes_pred[2] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # Recall values above 0.75 (3/4 correct) are not achieved and therefore nan.
        first_achieved = np.where(md.recall_hypo <= 0.25)[0][0]
        assert np.all(np.isnan(md.confidence[:first_achieved]))
        assert md.tp[first_achieved] == 1
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 3
        assert md.lgd[first_achieved] == 3 * 0.5
        assert md.tid[first_achieved] == 3 * 0.5
        assert md.frag[first_achieved] == 0
        assert md.ids[first_achieved] == 0

    def test_identity_switch(self):
        """ Change the tracking_id of one frame from the GT submission. """

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
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        first_achieved = np.where(md.recall_hypo <= 0.5)[0][0]
        assert md.tp[first_achieved] == 2
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 0
        assert md.lgd[first_achieved] == 0
        assert md.tid[first_achieved] == 0
        assert md.frag[first_achieved] == 0
        assert md.ids[first_achieved] == 2  # One wrong id leads to 2 identity switches.

    def test_drop_gt(self):
        """ Drop one box from the GT. """

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
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 3)
        assert np.all(md.fp == 1)
        assert np.all(md.fn == 0)
        assert np.all(md.lgd == 0.5)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_drop_gt_interpolate(self):
        """ Drop one box from the GT and interpolate the results to fill in that box. """

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
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fp == 0)
        assert np.all(md.fn == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_scenarios(self):
        """ More flexible scenario test structure. """

        def create_tracks(_scenario, tag=None):
            tracks = {}
            for entry_id, entry in enumerate(_scenario['input']['pos_'+tag]):
                tracking_id = 'tag_{}'.format(entry_id)
                for timestamp, pos in enumerate(entry):
                    if timestamp not in tracks.keys():
                        tracks[timestamp] = []
                    box = TrackingBox(translation=(pos[0], pos[1], 0.0), tracking_id=tracking_id, tracking_name='car',
                                      tracking_score=0.5)
                    tracks[timestamp].append(box)

            return tracks

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        for scenario in get_scenarios():
            tracks_gt = {'scene-1': create_tracks(scenario, tag='gt')}
            tracks_pred = {'scene-1': create_tracks(scenario, tag='pred')}

            # Accumulate metrics.
            ev = TrackingEvaluation(tracks_gt, tracks_pred, 'car', cfg.dist_fcn_callable,
                                    cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                    metric_worst=cfg.metric_worst, verbose=False)
            md = ev.accumulate()

            for key, value in scenario['output'].items():
                metric_values = getattr(md, key)
                metric_values = metric_values[np.logical_not(np.isnan(metric_values))]
                assert np.all(metric_values == value)


if __name__ == '__main__':
    unittest.main()
