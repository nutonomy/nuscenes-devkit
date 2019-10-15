"""
This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
from typing import List, Dict, Callable, Tuple

import numpy as np
import motmetrics
import pandas
import sklearn

from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetrics
from nuscenes.eval.tracking.utils import print_threshold_metrics, create_motmetrics
from nuscenes.eval.tracking.metrics import motap, motp_custom, faf_custom, track_initialization_duration, \
    longest_gap_duration


class TrackingEvaluation(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int = 11,
                 output_dir: str = '.',
                 verbose: bool = True):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.

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
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.output_dir = output_dir
        self.verbose = verbose

        self.n_scenes = len(self.tracks_gt)

    def compute_all_metrics(self, metrics: TrackingMetrics) -> TrackingMetrics:
        """
        Compute all relevant metrics for the current class.
        :param metrics: The TrackingMetrics to be augmented with the metric values.
        :returns: Augmented TrackingMetrics instance.
        """
        # Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)
        accumulators = []
        thresh_metrics = []
        thresh_names = []

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

        # Define mapping for metrics that use motmetrics library.
        mot_metric_map = {  # Mapping from motmetrics names to metric names used here.
            'num_frames': '',  # Used in FAF.
            'num_objects': '',  # Used in MOTAP computation.
            'num_predictions': '',  # Only printed out.
            'num_matches': '',  # Used in MOTAP computation and printed out.
            'motap': '',  # Only used in AMOTA.
            'mota': 'mota',  # Traditional MOTA.
            'motp_custom': 'motp',  # Traditional MOTP.
            'faf_custom': 'faf',
            'mostly_tracked': 'mt',
            'mostly_lost': 'ml',
            'num_false_positives': 'fp',
            'num_misses': 'fn',
            'num_switches': 'ids',
            'num_fragmentations': 'frag',
            'tid': 'tid',
            'lgd': 'lgd'
        }

        # Define mapping for metrics averaged over classes.
        avg_metric_map = {  # Mapping from average metric name to individual per-threshold metric name.
            'amota': 'motap',
            'amotp': 'motp_custom'
        }
        avg_metric_worst = {  # Mapping to the worst possible value.
            'amota': 0.0,
            'amotp': self.dist_th_tp
        }

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'threshold_%.4f' % _threshold

        # Register default mot metrics.
        mh = create_motmetrics()

        # Register custom metrics.
        mh.register(motap,
                    ['num_matches', 'num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                    formatter='{:.2%}'.format, name='motap')
        mh.register(motp_custom,
                    formatter='{:.2%}'.format, name='motp_custom')
        mh.register(faf_custom,
                    formatter='{:.2%}'.format, name='faf_custom')
        mh.register(track_initialization_duration, ['obj_frequencies'],
                    formatter='{:.2%}'.format, name='tid')
        mh.register(longest_gap_duration, ['obj_frequencies'],
                    formatter='{:.2%}'.format, name='lgd')

        # Get thresholds.
        # Note: The recall values are the "desired" recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        thresholds, recalls = self.get_thresholds(gt_count)

        for threshold, recall in zip(thresholds, recalls):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Compute CLEARMOT/MT/ML metrics for current threshold.
            acc, _ = self.accumulate(threshold)
            accumulators.append(acc)
            thresh_names.append(name_gen(threshold))
            thresh_summary = mh.compute(acc, metrics=mot_metric_map.keys(), name=name_gen(threshold))
            thresh_metrics.append(thresh_summary)

            # Print metrics to stdout.
            print_threshold_metrics(thresh_summary.to_dict())

        # Concatenate all metrics. We only do this for more convenient access.
        assert len(thresh_names) == len(set(thresh_names))
        summary = pandas.concat(thresh_metrics)

        # Find best MOTA to determine threshold to pick for traditional metrics.
        best_name = summary.mota.idxmax()

        # Compute AMOTA / AMOTP.
        for metric_name in avg_metric_map.keys():
            values = summary.get(avg_metric_map[metric_name]).values.tolist()
            values.extend([avg_metric_worst[metric_name]] * np.sum(np.isnan(thresholds)))
            assert len(values) == len(thresholds)
            if np.all(np.isnan(values)):
                value = np.nan
            else:
                value = float(np.nanmean(values))
            metrics.add_raw_metric(metric_name, self.class_name, value)

        # Store all traditional metrics.
        for (mot_name, metric_name) in mot_metric_map.items():
            # Skip metrics which we don't output.
            if metric_name == '':
                continue

            # Clip all metrics to be >= 0, in particular MOTA.
            value = np.maximum(float(summary.loc[best_name][mot_name]), 0.0)

            metrics.add_raw_metric(metric_name, self.class_name, value)

        return metrics

    def accumulate(self, threshold: float = None) -> Tuple[motmetrics.MOTAccumulator, List[float]]:
        """
        Aggregate the raw data for the traditional CLEARMOT/MT/ML metrics.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :returns: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        # Init.
        acc = motmetrics.MOTAccumulator()
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.
        frame_id = 0  # Frame ids must be unique across all scenes

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for i, scene_id in enumerate(self.tracks_gt.keys()):
            if self.verbose:
                print('Processing scene %d of %d: %s...' % (i, len(self.tracks_gt), scene_id))

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred_unfiltered = self.tracks_pred[scene_id]

            # Threshold predicted tracks using the specified threshold.
            if threshold is None:
                scene_tracks_pred = scene_tracks_pred_unfiltered
            else:
                scene_tracks_pred = TrackingEvaluation._threshold_tracks(scene_tracks_pred_unfiltered, threshold)

            for timestamp in scene_tracks_gt.keys():
                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]

                # Abort if there are neither GT nor pred boxes.
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                # Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    events = acc.events.loc[frame_id]
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
            if len(track) == 0:
                scene_tracks_pred[track_id] = []
            else:
                # Compute average score for current track.
                track_scores = [box.tracking_score for box in track]
                avg_score = np.mean(track_scores)

                # Decide whether to keep track by thresholding.
                if avg_score >= threshold:
                    scene_tracks_pred[track_id] = track
                else:
                    scene_tracks_pred[track_id] = []

        return scene_tracks_pred

    def get_thresholds(self, gt_count: int) -> Tuple[List[float], List[float]]:
        """
        Specify recall thresholds.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_count: The number of GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # Run accumulate to get the scores of TPs.
        acc, scores = self.accumulate(threshold=None)
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
        rec_interp = rec_interp[rec_interp >= self.min_recall]  # Remove recall values < 10%.
        max_recall_achieved = np.max(rec)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        return thresholds, rec_interp
