"""
This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
from typing import List, Dict, Callable, Any, Tuple

import motmetrics
import numpy as np

from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetrics


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

    def compute_all_metrics(self, metrics: TrackingMetrics) -> TrackingMetrics:
        """
        Compute all relevant metrics for the current class.
        :param metrics: The TrackingMetrics to be augmented with the metric values.
        :returns: Augmented TrackingMetrics instance.
        """
        # Init.
        print('Computing metrics for class %s...' % self.class_name)
        accumulators = []
        thresh_metrics = []
        names = []
        mh = motmetrics.metrics.create()

        # Skip missing classes.
        gt_count = 0
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_count += 1
        if gt_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return metrics

        # Specify metrics and threshold naming pattern.
        metric_names = ['num_frames', 'mota', 'motp', 'tid', 'lgd']

        def name_gen(_threshold):
            return 'threshold_%.3f' % _threshold

        # Register custom metrics.
        mh.register(TrackingEvaluation.track_initialization_duration, ['obj_frequencies'],
                    formatter='{:.2%}'.format, name='tid')
        mh.register(TrackingEvaluation.longest_gap_duration, ['obj_frequencies'],
                    formatter='{:.2%}'.format, name='lgd')

        # Get thresholds.
        thresholds = self.get_thresholds(gt_count)

        for threshold in thresholds:
            # Compute CLEARMOT/MT/ML metrics for current threshold.
            acc, _ = self.accumulate(threshold)
            accumulators.append(acc)
            names.append(name_gen(threshold))

            thresh_summary = mh.compute(acc, metrics=metric_names, name=name_gen(threshold))
            print(thresh_summary)
            thresh_metrics.append(thresh_summary)

        # Aggregate metrics. We only do this for more convenient access.
        summary = mh.compute_many(accumulators, metrics=metric_names, names=names, generate_overall=False)

        # Find best mota to determine threshold to pick for traditional metrics.
        best_thresh_idx = int(np.argmax(summary.mota.values))
        best_thresh = thresholds[best_thresh_idx]
        best_name = name_gen(best_thresh)

        # Create a dictionary of all metrics.
        for metric_name in metric_names:
            value = summary.get(metric_name)[best_name]
            metrics.add_raw_metric(metric_name, self.class_name, value)

        return metrics

    def accumulate(self, threshold: float = None) -> Tuple[motmetrics.MOTAccumulator, List[float]]:
        """
        Aggregate the raw data for the traditional CLEARMOT/MT/ML metrics.
        :param threshold: score threshold used to determine positives and negatives.
        :returns: The MOTAccumulator that stores all the hits/misses/etc.
        """
        # Init.
        acc = motmetrics.MOTAccumulator()
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in self.tracks_gt.keys():
            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred_unfiltered = self.tracks_pred[scene_id]

            # Threshold predicted tracks using the specified threshold.
            if threshold is None:
                scene_tracks_pred = scene_tracks_pred_unfiltered
            else:
                scene_tracks_pred = TrackingEvaluation._threshold_tracks(scene_tracks_pred_unfiltered, threshold)

            frame_id = 0
            for timestamp in scene_tracks_gt.keys():
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]

                # Select only the current class.
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]

                # Abort if there are neither GT nor pred boxes.
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                distances: np.ndarray = np.ones((len(frame_gt), len(frame_pred)))
                for y, gg in enumerate(frame_gt):
                    for x, tt in enumerate(frame_pred):
                        distances[y, x] = float(self.dist_fcn(gg, tt))

                # Distances that are larger than the threshold won't be associated.
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                # Store scores of matches, which are used to determine recall thresholds.
                events = acc.events.ix[frame_id]  # TODO: ix is deprecated
                matches = events[events.Type == 'MATCH']
                match_ids = matches.HId.values
                match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                scores.extend(match_scores)

                # Increment the frame_id, unless there were no boxes (equivalent to what motmetrics does).
                frame_id += 1

        return acc, scores

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

    def get_thresholds(self, gt_count: int) -> List[float]:
        """
        Specify recall thresholds.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_count: The number of GT boxes for this class.
        :return: The list of thresholds.
        """
        # Run accumulate to get the scores of TPs.
        acc, scores = self.accumulate(0.0)
        assert len(scores) > 0

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / gt_count
        assert len(rec) > 0 and np.max(rec) <= 1

        # Determine thresholds.
        rec_interp = np.linspace(0, 1, self.num_thresholds)  # 11 steps, from 0% to 100% recall.
        rec_interp = rec_interp[rec_interp >= 0.1]  # Remove recall values < 10%.
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Sort thresholds to be more intuitive.
        thresholds.sort()

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
            notmiss = dfo[dfo.Type != 'MISS']

            if len(notmiss) == 0:
                # For missed objects return the length of the track.
                diff = dfo.index[-1][0] - dfo.index[0][0]
            else:
                # Find the first time the object was detected and compute the difference to first time the object
                # entered the scene.
                diff = notmiss.index[0][0] - dfo.index[0][0]
            assert diff >= 0, 'Time difference should be larger than or equal to zero'
            # Multiply number of sample differences with sample period (0.5 sec)
            tid += float(diff) * 0.5
        return tid / len(obj_frequencies)

    @staticmethod
    def longest_gap_duration(df, obj_frequencies):
        gap = 0
        for gt_tracking_id in obj_frequencies.index:
            # Find the frame_ids object is tracked and compute the gaps between those. Take the maximum one for longest
            # gap.
            dfo = df.noraw[df.noraw.OId == gt_tracking_id]
            notmiss = dfo[dfo.Type != 'MISS']
            if len(notmiss) == 0:
                # For missed objects return the length of the track.
                diff = dfo.index[-1][0] - dfo.index[0][0]
            else:
                diff = notmiss.index.get_level_values(0).to_series().diff().max() - 1
            if np.isnan(diff):
                diff = 0
            assert diff >= 0, 'Time difference should be larger than or equal to zero {0:f}'.format(diff)
            gap += diff * 0.5
        return gap / len(obj_frequencies)

