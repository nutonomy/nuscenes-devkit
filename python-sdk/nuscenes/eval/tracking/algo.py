"""
This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
from typing import List, Dict, Callable, Any

import motmetrics
import numpy as np

from nuscenes.eval.tracking.data_classes import TrackingBox


class TrackingEvaluation(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 num_thresholds: int = 11,
                 output_dir: str = '.'):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt:
        :param tracks_pred:
        :param class_name:
        :param dist_fcn:
        :param dist_th_tp:
        :param num_thresholds:
        :param output_dir:

        Tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
         MOTA	        - Multi-object tracking accuracy in [0,100].
         MOTP	        - Multi-object tracking precision in [0,100] (3D) / [td,100] (2D).
         id-switches    - number of id switches.
         fragments      - number of fragmentations.
         MT, ML	        - number of mostly tracked and mostly lost trajectories.
         recall	        - recall = percentage of detected targets.
         precision	    - precision = percentage of correctly detected targets.
         FAR		    - number of false alarms per frame.
         falsepositives - number of false positives (FP).
         missed         - number of missed targets (FN).

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.num_thresholds = num_thresholds
        self.output_dir = output_dir

        self.n_scenes = len(self.tracks_gt)

    def compute_all_metrics(self) -> None:
        """
        Compute all relevant metrics for the current class.
        """
        # Init.
        accumulators = []
        names = []
        mh = motmetrics.metrics.create()

        # Register custom metrics.
        mh.register(TrackingEvaluation.track_initialization_duration, ['obj_frequencies'],
                    formatter='{:.2%}'.format, name='tid')

        # Get thresholds.
        thresholds = self.get_thresholds()

        for threshold in thresholds:
            # Compute CLEARMOT/MT/ML metrics for current threshold.
            acc = self.accumulate(threshold)
            accumulators.append(acc)
            names.append('threshold {0:f}'.format(threshold))

            summary = mh.compute(acc,
                                 metrics=['num_frames', 'mota', 'motp', 'tid'],
                                 name='threshold {0:f}'.format(threshold))
            print(summary)

        # Compute overall metrics: AMOTA, AMOTP, mAP.
        summary = mh.compute_many(
            accumulators,
            metrics=['num_frames', 'mota', 'motp', 'tid'],  # TODO: implement AMOTA
            names=names,
            generate_overall=True)
        print(summary)

    def accumulate(self, threshold: float = None) -> motmetrics.MOTAccumulator:
        """
        Aggregate the raw data for the traditional CLEARMOT/MT/ML metrics.
        :param threshold: score threshold used to determine positives and negatives.
        """
        # Init.
        acc = motmetrics.MOTAccumulator()

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in self.tracks_gt.keys():
            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred_unfiltered = self.tracks_pred[scene_id]

            # Create map from timestamp to frame_id.
            timestamp_map = {t: i for i, t in enumerate(scene_tracks_gt.keys())}

            # Threshold predicted tracks using the specified threshold.
            if threshold is None:
                scene_tracks_pred = scene_tracks_pred_unfiltered
            else:
                scene_tracks_pred = TrackingEvaluation._threshold_tracks(scene_tracks_pred_unfiltered, threshold)

            for timestamp in scene_tracks_gt.keys():
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]

                # Calculate distances.
                distances: np.ndarray = np.ones((len(frame_gt), len(frame_pred)))
                for y, gg in enumerate(frame_gt):
                    for x, tt in enumerate(frame_pred):
                        distances[y, x] = float(self.dist_fcn(gg, tt))

                # Distances that are larger than the threshold won't be associated.
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                acc.update(gt_ids, pred_ids, distances, frameid=timestamp_map[timestamp])

        return acc

    @staticmethod
    def _threshold_tracks(scene_tracks_pred_unfiltered: Dict[int, List[TrackingBox]],
                          threshold: float) \
            -> Dict[int, List[TrackingBox]]:
        """
        For the current threshold, remove the tracks with low confidence for each frame.
        Note that the average of the per-frame scores forms the track-level score.
        :param scene_tracks_pred_unfiltered: The predicted tracks for this scene.
        :param threshold: The score threshold.
        :return: A subset of the predicted tracks with scores above the threshold.
        """
        assert threshold is not None, 'Error: threshold must be specified!'
        scene_tracks_pred = {}
        for track_id, track in scene_tracks_pred_unfiltered.items():
            # Compute average score for current track.
            track_scores = []
            for box in track:
                track_scores.append(box.tracking_score)
            avg_score = np.mean(track_scores)

            # Decide whether to keep track by thresholding.
            if avg_score >= threshold:
                scene_tracks_pred[track_id] = track
            else:
                scene_tracks_pred[track_id] = []

        return scene_tracks_pred

    def get_thresholds(self) -> List[float]:
        """
        Specify recall thresholds.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        TODO: We need determine the thresholds from recall, not scores.
        :return: The list of thresholds.
        """
        thresholds = list(np.linspace(0, 1, self.num_thresholds))

        return thresholds

    # Custom metrics.
    @staticmethod
    def track_initialization_duration(df: Any, obj_frequencies: Any) -> float:
        """
        Computes the track initialization duration, which is the duration from the first occurrance of an object to
        it's first correct detection (TP).
        :param df:
        :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
        :return: The track initialization time.
        """
        tid = 0
        for gt_tracking_id in obj_frequencies.index:
            # Get matches.
            dfo = df.noraw[df.noraw.OId == gt_tracking_id]
            match = dfo[dfo.Type == 'MATCH']

            if len(match) == 0:
                # For missed objects return the length of the track.
                diff = dfo.index[-1][0] - dfo.index[0][0]
            else:
                # Find the first time the object was detected and compute the difference to first time the object
                # entered the scene.
                diff = match.index[0][0] - dfo.index[0][0]
            assert diff >= 0, 'Time difference should be larger than or equal to zero'
            # Multiply number of sample differences with sample period (0.5 sec)
            tid += float(diff) * 0.5
        return tid
