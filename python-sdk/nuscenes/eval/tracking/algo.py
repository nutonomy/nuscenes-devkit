"""
This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
from typing import List, Dict, Callable
import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt

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

    def compute_all_metrics(self, class_name: str, suffix: str) -> None:
        """
        Compute all relevant metrics for the current class.
        :param class_name:
        :param suffix:
        """
        # Init.
        # filename = os.path.join(self.output_dir, 'summary_%s_average_%s.txt' % (class_name, suffix))
        # dump = open(filename, "w+")
        accumulators = []
        names = []
        mh = mm.metrics.create()

        # Get thresholds.
        thresholds = self.get_thresholds()

        # Evaluate the mean average metrics.

        # Register custom metrics.
        mh.register(TrackingEvaluation.track_initialization_duration,
                    ['obj_frequencies',], formatter='{:.2%}'.format, name='tid')

        for threshold in thresholds:
            # Compute metrics for current threshold.
            acc = self.accumulate(threshold)
            accumulators.append(acc)
            names.append('threshold {0:f}'.format(threshold))
            # self.save_to_stats(dump, threshold)

            # Update counters for average metrics.
            # stat_meter.update(data_tmp)

            # Store best MOTA threshold used for CLEARMOT metrics.
            # if self.MOTA > best_mota:
            #     best_mota = self.MOTA
            #     best_threshold = threshold

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

        # # Compute average metrics and print summary.
        # stat_meter.output()
        # summary = stat_meter.print_summary()
        # print(summary)

    def get_thresholds(self) -> List[float]:
        """
        Specify recall thresholds.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :return: The list of thresholds.
        """
        thresholds = list(np.linspace(0, 1, self.num_thresholds))

        return thresholds

    def accumulate(self, threshold: float = None) -> mm.MOTAccumulator:
        """
        Aggregate the raw data for the traditional CLEARMOT/MT/ML metrics.
        """
        # Init.
        acc = mm.MOTAccumulator(auto_id=True)

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

                # Accumulate results.
                # TODO: Cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances)

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

    # Custom metrics
    @staticmethod
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
            tid += float(diff) / 1e+6
        return tid


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
