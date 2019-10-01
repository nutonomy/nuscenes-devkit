"""
This code is based on Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py
"""
import os
from typing import List, Dict, Callable, Tuple
import motmetrics as mm
import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt

from nuscenes.eval.tracking.data_classes import TrackingBox


# Custom metrics
def track_initialization_duration(df, obj_frequencies):
    tid = 0
    for o in obj_frequencies.index:
        # Find the first time object was detected and compute the difference to first time
        # object entered the scene. For non detected objects that is the length of the track
        dfo = df.noraw[df.noraw.OId == o]
        match = dfo[dfo.Type == 'MATCH']
        if len(match) == 0:
            diff = dfo.index[-1][0] - dfo.index[0][0]
        else:
            diff = match.index[0][0] - dfo.index[0][0]
        assert diff >= 0, 'Time difference should be larger than or equal to zero'
        tid += float(diff)/1e+6
    return tid


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

        # Mot metrics stuff
        self.acc = mm.MOTAccumulator()

        # Internal statistics.
        self.recall = 0
        self.precision = 0
        self.total_cost = 0
        self.tp = 0  # Number of true positives including ignored true positives!
        self.n_tr = 0  # Number of tracker detections minus ignored tracker detections
        self.n_trs = []  # Number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_gt = 0  # Number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # Number of ignored ground truth detections
        self.n_gts = []  # Number of ground truth detections minus ignored FNs and TPs PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_trajectories = 0
        self.gt_trajectories = dict()
        self.ign_trajectories = dict()

        # Challenge metrics.
        self.MOTA = 0
        self.MOTP = 0
        self.fp = 0  # Number of false positives
        self.fn = 0  # Number of false negatives WITHOUT ignored false negatives
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.ML = 0

    def compute_all_metrics(self, class_name: str, suffix: str) -> None:
        """
        Compute all relevant metrics for the current class.
        :param class_name:
        :param suffix:
        """
        # Initialize stats logger.
        filename = os.path.join(self.output_dir, 'summary_%s_average_%s.txt' % (class_name, suffix))
        dump = open(filename, "w+")
        stat_meter = Stat(class_name=class_name, suffix=suffix, dump=dump, num_thresholds=self.num_thresholds)

        # Get thresholds.
        thresholds = self.get_thresholds()

        # Evaluate the mean average metrics.
        best_mota, best_threshold = -1, -1
        accumulators = []
        names = []
        mh = mm.metrics.create()
        # Register custom metric
        mh.register(track_initialization_duration, ['obj_frequencies',], formatter='{:.2%}'.format,
                    name='tid')
        for threshold in thresholds:
            print('threshold {0:f}'.format(threshold))
            acc = self.compute_third_party_metrics(threshold)
            accumulators.append(acc)
            names.append('threshold {0:f}'.format(threshold))

            # # Compute metrics for current threshold.
            # self.reset()
            # self.compute_third_party_metrics(threshold)
            # self.save_to_stats(dump, threshold)
            #
            # # Update counters for average metrics.
            # data_tmp = dict()
            # data_tmp['mota'], data_tmp['motp'], data_tmp['precision'], data_tmp['fp'], data_tmp['fn'], \
            #     data_tmp['recall'] = \
            #     self.MOTA, self.MOTP, self.precision, self.fp, self.fn, self.recall
            # stat_meter.update(data_tmp)
            #
            # # Store best MOTA threshold used for CLEARMOT metrics.
            # if self.MOTA > best_mota:
            #     best_mota = self.MOTA
            #     best_threshold = threshold

            summary = mh.compute(acc,
                                 metrics=['num_frames', 'mota', 'motp', 'tid'],
                                 name='threshold {0:f}'.format(threshold))
            print(summary)

        summary = mh.compute_many(
            accumulators,
            metrics=['num_frames', 'mota', 'motp', 'tid'],
            names=names,
            generate_overall=True)
        print(summary)

        # # Use best threshold for CLEARMOT metrics.
        # self.reset()
        # self.compute_third_party_metrics(best_threshold)
        # self.save_to_stats(dump, best_threshold)
        #
        # # Compute average metrics and print summary.
        # stat_meter.output()
        # summary = stat_meter.print_summary()
        # print(summary)
        #
        # stat_meter.plot(save_dir=self.output_dir)
        # dump.close()

    def get_thresholds(self) -> List[float]:
        """
        Specify recall thresholds.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :return: The list of thresholds.
        """
        thresholds = list(np.linspace(0, 1, self.num_thresholds))

        return thresholds

    def reset(self) -> None:
        self.n_gt = 0  # Number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # Number of ignored ground truth detections
        self.n_tr = 0  # Number of tracker detections minus ignored tracker detections

        self.MOTA = 0
        self.MOTP = 0

        self.recall = 0
        self.precision = 0

        self.total_cost = 0
        self.tp = 0
        self.fn = 0
        self.fp = 0

        self.n_gts = []  # Number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_trs = []  # Number of tracker detections minus ignored tracker detections PER SEQUENCE

        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.ML = 0

        self.gt_trajectories = dict()
        self.ign_trajectories = dict()

    def compute_third_party_metrics(self, threshold: float = None) -> None:
        """
        Computes the traditional CLEARMOT/MT/ML metrics.
        """
        # Init.
        self.gt_trajectories = dict()
        self.ign_trajectories = dict()

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in self.tracks_gt.keys():
            acc = mm.MOTAccumulator()
            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred_unfiltered = self.tracks_pred[scene_id]

            # Threshold predicted tracks using the specified threshold.
            if threshold is None:
                scene_tracks_pred = scene_tracks_pred_unfiltered
            else:
                scene_tracks_pred = TrackingEvaluation._threshold_tracks(scene_tracks_pred_unfiltered, threshold)

            # Statistics over the current sequence.
            # The *_trajectories fields both map from GT track_id and timestamp to pred track_id.
            scene_gt_trajectories: Dict[str, Dict[int, str]] = dict()
            scene_ign_trajectories: Dict[str, Dict[int, bool]] = dict()
            n_gts = 0
            n_trs = 0

            for timestamp in scene_tracks_gt.keys():
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]

                # Calculate distances
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                distances: np.ndarray = np.ones((len(frame_gt), len(frame_pred)))
                for y, gg in enumerate(frame_gt):
                    for x, tt in enumerate(frame_pred):
                        distances[y, x] = float(self.dist_fcn(gg, tt))

                # Distances that are larger than the threshold won't be associated
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results
                frameid = acc.update(gt_ids, pred_ids, distances, frameid=timestamp)

        return acc


    def create_summary_details(self) -> str:
        """
        Generate and log a summary of the results.
        """
        summary = ''
        summary += 'Evaluation: Best results with single threshold'.center(80, '=') + '\n'
        summary += self.print_entry('Multiple Object Tracking Accuracy (MOTA)', self.MOTA) + '\n'
        summary += self.print_entry('Multiple Object Tracking Precision (MOTP)', float(self.MOTP)) + '\n'
        summary += '\n'
        summary += self.print_entry('Recall', self.recall) + '\n'
        summary += self.print_entry('Precision', self.precision) + '\n'
        summary += '\n'
        summary += self.print_entry('Mostly Tracked', self.MT) + '\n'
        summary += self.print_entry('Mostly Lost', self.ML) + '\n'
        summary += '\n'
        summary += self.print_entry('True Positives', self.tp) + '\n'
        summary += self.print_entry('False Positives', self.fp) + '\n'
        summary += self.print_entry('False Negatives', self.fn) + '\n'
        summary += self.print_entry('ID-switches', self.id_switches) + '\n'
        summary += self.print_entry('Fragmentations', self.fragments) + '\n'
        summary += '\n'
        summary += self.print_entry('Ground Truth Objects (Total)', self.n_gt + self.n_igt) + '\n'
        summary += self.print_entry('Ground Truth Trajectories', self.n_gt_trajectories) + '\n'
        summary += '\n'
        summary += self.print_entry('Tracker Objects (Total)', self.n_tr) + '\n'
        summary += '=' * 80

        return summary

    def create_summary_simple(self, threshold: float, is_best: bool) -> str:
        """
        Generate and mail a summary of the results.
        :param threshold: Recall threshold used for evaluation.
        :param is_best: Whether this is the best confidence threshold for this class.
        """
        summary = ''

        if is_best:
            summary += ('Evaluation with best confidence threshold %f' % threshold).center(80, '=') + '\n'
        else:
            summary += ('Evaluation with confidence threshold %f' % threshold).center(79, '=') + '\n'

        summary += ' MOTA   MOTP   MT     ML     IDS  FRAG    F1   Prec  Recall      TP    FP    FN\n'
        summary += '{:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:.4f} {:.4f} {:5d} {:5d} {:5d}\n'.format(
            self.MOTA, self.MOTP, self.MT, self.ML, self.id_switches, self.fragments,
            self.precision, self.recall, self.tp, self.fp, self.fn)
        summary += '=' * 80

        return summary

    def print_entry(self, key, val, width=(70, 10)) -> None:
        """
            Pretty print an entry in a table fashion.
        """
        s_out = key.ljust(width[0])
        if type(val) == int:
            s = '%%%dd' % width[1]
            s_out += s % val
        elif type(val) == float:
            s = '%%%d.4f' % (width[1])
            s_out += s % val
        else:
            s_out += ('%s' % val).rjust(width[1])
        return s_out

    def save_to_stats(self, dump, threshold: float = None, is_best: bool = False) -> None:
        """
        Save the statistics in a whitespace separate file.
        """
        if threshold is None:
            summary = self.create_summary_details()
        else:
            summary = self.create_summary_simple(threshold, is_best)

        print(summary)
        print(summary, file=dump)

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

    @staticmethod
    def _hungarian_method(frame_gt, frame_pred, dist_fcn: Callable, dist_th_tp: float) \
            -> Tuple[List[Tuple[int, int]], List[List[float]]]:
        """
        Use Hungarian method to associate, using center distance 0..1 as cost build cost matrix.
        Row are gt, columns are predictions.
        :param frame_gt: The GT boxes of the current frame.
        :param frame_pred: The predicted boxes of the current frame.
        :param dist_fcn: Distance function used for matching.
        :param dist_th_tp: Distance function threshold.
        :return: Association matrix and cost matrix.
        """
        cost_matrix: np.ndarray = dist_th_tp * np.ones((len(frame_gt), len(frame_pred)))
        for y, gg in enumerate(frame_gt):
            # Save current ids.
            gg.tracker = -1
            gg.id_switch = 0
            gg.fragmentation = 0

            for x, tt in enumerate(frame_pred):
                cost_matrix[y, x] = float(np.minimum(dist_fcn(gg, tt), dist_th_tp))

        # Convert to list.
        cost_matrix: List[List[float]] = cost_matrix.tolist()

        # Associate using Hungarian aka. Munkres algorithm.
        hm = Munkres()
        association_matrix: List[Tuple[int, int]] = hm.compute(cost_matrix)

        return association_matrix, cost_matrix

    def _compute_metrics(self) -> None:
        """
        Compute MT/PT/ML, fragments and idswitches for all GT trajectories.
        """
        assert self.total_cost >= 0

        n_ignored_tr_total = 0
        for scene_trajectories, scene_ignored in zip(self.gt_trajectories.values(), self.ign_trajectories.values()):

            assert len(scene_trajectories) != 0

            n_ignored_tr = 0
            for g, ign_g in zip(scene_trajectories.values(), scene_ignored.values()):
                assert len(g) == len(ign_g)

                # All frames of this GT trajectory are ignored.
                if all(ign_g.values()):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue

                # All frames of this GT trajectory are not assigned to any detections.
                if all([gg == '' for gg in g.values()]):
                    self.ML += 1
                    continue

                # Compute tracked frames in trajectory.
                # First detection (necessary to be in gt_trajectories) is always tracked.
                first_timestamp = list(g.keys())[0]
                last_id = g[first_timestamp]
                tracked = 1 if g[first_timestamp] != '' else 0
                for i in range(1, len(g)):  # Skip first timestamp.
                    prev_timestamp = list(g.keys())[i - 1]
                    timestamp = list(g.keys())[i]
                    if i < len(g) - 1:
                        next_timestamp = list(g.keys())[i + 1]

                    if ign_g[timestamp]:
                        last_id = ''
                        continue

                    # Count number of identity switches.
                    if last_id != g[timestamp] \
                            and last_id != '' \
                            and g[timestamp] != '' \
                            and g[prev_timestamp] != '':
                        self.id_switches += 1

                    # Count number of fragmentations.
                    if timestamp < len(g) - 1 \
                            and g[prev_timestamp] != g[timestamp] \
                            and last_id != '' \
                            and g[timestamp] != '' \
                            and g[next_timestamp] != '':
                        self.fragments += 1

                    # Updated tracked counter.
                    if g[timestamp] != '':
                        tracked += 1
                        last_id = g[timestamp]

                # Handle last frame; tracked state is handled in for loop (g[f] != -1).
                # Count number of fragmentations.
                if len(g) > 1 \
                        and g[prev_timestamp] != g[timestamp] \
                        and last_id != '' \
                        and g[timestamp] != '' \
                        and not ign_g[timestamp]:
                    self.fragments += 1

                # Compute MT/PT/ML.
                tracking_ratio = tracked / float(len(g) - sum(ign_g.values()))
                if tracking_ratio > 0.8:
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    self.ML += 1

        if self.n_gt_trajectories - n_ignored_tr_total == 0:
            self.MT = 0
            self.ML = 0
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # Precision, recall and F1 metrics.
        if self.fp + self.tp == 0 or self.tp + self.fn == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)

        # Compute CLEARMOT metrics.
        if self.n_gt == 0:
            self.MOTA = -float('inf')
        else:
            self.MOTA = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
        if self.tp == 0:
            self.MOTP = 0
        else:
            self.MOTP = self.total_cost / float(self.tp)

        self.num_gt = self.tp + self.fn


class Stat:
    """
    Utility class to load data.
    """

    def __init__(self,
                 class_name: str,
                 suffix: str,
                 dump,
                 num_thresholds):
        """
        Constructor, initializes the object given the parameters.
        """
        # init object data
        self.mota = 0
        self.motp = 0
        self.precision = 0
        self.fp = 0
        self.fn = 0

        self.mota_list = list()
        self.motp_list = list()
        self.precision_list = list()
        self.fp_list = list()
        self.fn_list = list()
        self.recall_list = list()

        self.class_name = class_name
        self.suffix = suffix
        self.dump = dump
        self.num_thresholds = num_thresholds

    def update(self, data: Dict) -> None:
        self.mota += data['mota']
        self.motp += data['motp']
        self.precision += data['precision']
        self.fp += data['fp']
        self.fn += data['fn']

        self.mota_list.append(data['mota'])
        self.motp_list.append(data['motp'])
        self.precision_list.append(data['precision'])
        self.fp_list.append(data['fp'])
        self.fn_list.append(data['fn'])
        self.recall_list.append(data['recall'])

    def output(self) -> None:
        self.ave_mota = self.mota / self.num_thresholds
        self.ave_motp = self.motp / self.num_thresholds
        self.ave_precision = self.precision / self.num_thresholds

    def print_summary(self) -> str:
        summary = ''

        summary += 'Evaluation: Average at all thresholds'.center(80, '=') + '\n'
        summary += ' AMOTA  AMOTP  APrec\n'

        summary += '{:.4f} {:.4f} {:.4f}\n'.format(
            self.ave_mota, self.ave_motp, self.ave_precision)
        summary += '=' * 80
        summary += '\n'
        print(summary, file=self.dump)

        return summary

    def plot_over_recall(self, data_list, title, y_name, save_path) -> None:
        """
        TODO
        :param data_list:
        :param title:
        :param y_name:
        :param save_path:
        """
        # Abort if there is no data to plot.
        if len(data_list) == 0:
            return

        # add extra zero at the end
        largest_recall = self.recall_list[-1]
        extra_zero = np.arange(largest_recall, 1, 0.01).tolist()
        len_extra = len(extra_zero)
        y_zero = [0] * len_extra

        plt.clf()
        fig, (ax) = plt.subplots(1, 1)
        plt.title(title)
        ax.plot(np.array(self.recall_list + extra_zero), np.array(data_list + y_zero))
        ax.set_ylabel(y_name, fontsize=20)
        ax.set_xlabel('Recall', fontsize=20)
        ax.set_xlim(0.0, 1.0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        if y_name in ['MOTA', 'MOTP', 'F1', 'Precision'] or max(data_list) == 0:
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(0.0, max(data_list))

        if y_name in ['MOTA', 'F1']:
            max_ind = np.argmax(np.array(data_list))
            plt.axvline(self.recall_list[max_ind], ymax=data_list[max_ind], color='r')
            plt.plot(self.recall_list[max_ind], data_list[max_ind], 'or', markersize=12)
            plt.text(self.recall_list[max_ind] - 0.05, data_list[max_ind] + 0.03, '%.2f' % (data_list[max_ind]),
                     fontsize=20)
        fig.savefig(save_path)
        plt.close(fig)

    def plot(self, save_dir: str) -> None:
        self.plot_over_recall(self.mota_list, 'MOTA - Recall Curve', 'MOTA',
                              os.path.join(save_dir, 'MOTA_recall_curve_%s_%s.pdf' % (self.class_name, self.suffix)))
        self.plot_over_recall(self.motp_list, 'MOTP - Recall Curve', 'MOTP',
                              os.path.join(save_dir, 'MOTP_recall_curve_%s_%s.pdf' % (self.class_name, self.suffix)))
        self.plot_over_recall(self.fp_list, 'False Positive - Recall Curve', 'False Positive',
                              os.path.join(save_dir, 'FP_recall_curve_%s_%s.pdf' % (self.class_name, self.suffix)))
        self.plot_over_recall(self.fn_list, 'False Negative - Recall Curve', 'False Negative',
                              os.path.join(save_dir, 'FN_recall_curve_%s_%s.pdf' % (self.class_name, self.suffix)))
        self.plot_over_recall(self.precision_list, 'Precision - Recall Curve', 'Precision',
                              os.path.join(save_dir, 'precision_recall_curve_%s_%s.pdf' % (self.class_name, self.suffix)))
