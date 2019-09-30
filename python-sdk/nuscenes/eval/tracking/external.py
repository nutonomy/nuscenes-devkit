"""
This code is based on Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py
"""
import os
from typing import List, Dict, Callable, Tuple
from collections import defaultdict

import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt

from nuscenes.eval.tracking.data_classes import TrackingBox


class TrackingEvaluation(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	        - Multi-object tracking accuracy in [0,100]
             MOTP	        - Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	        - Multi-object tracking accuracy in [0,100] with log10(id-switches)
             id-switches    - number of id switches
             fragments      - number of fragmentations
             MT, PT, ML	    - number of mostly tracked, partially tracked and mostly lost trajectories
             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """

    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 mail,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 num_sample_pts: int = 11):
        """
        TODO
        :param tracks_gt:
        :param tracks_pred:
        :param class_name:
        :param mail:
        :param dist_fcn:
        :param dist_th_tp:
        :param num_sample_pts:
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.cls = class_name
        self.mail = mail
        self.num_sample_pts = num_sample_pts
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp

        self.n_scenes = len(self.tracks_gt)
        self.gt_trajectories = dict()
        self.ign_trajectories = dict()
        self.scores = list()  # True positive scores are used to compute the 11 recall thresholds.

        # Statistics and numbers for evaluation.
        self.n_gt = 0  # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # number of ignored ground truth detections
        self.n_gts = []  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = []  # number of ground ignored truth detections PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_seq = []
        self.n_tr = 0  # number of tracker detections minus ignored tracker detections
        self.n_trs = []  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itr = 0  # number of ignored tracker detections
        self.n_itrs = []  # number of ignored tracker detections PER SEQUENCE
        self.n_tr_trajectories = 0
        self.n_tr_seq = []
        self.MOTA = 0
        self.MOTP = 0
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0
        self.total_cost = 0
        self.itp = 0  # number of ignored true positives
        self.itps = []  # number of ignored true positives PER SEQUENCE
        self.tp = 0  # number of true positives including ignored true positives!
        self.tps = []  # number of true positives including ignored true positives PER SEQUENCE
        self.fn = 0  # number of false negatives WITHOUT ignored false negatives
        self.fns = []  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifn = 0  # number of ignored false negatives
        self.ifns = []  # number of ignored false negatives PER SEQUENCE
        self.fp = 0  # number of false positives
        self.fps = []  # above PER SEQUENCE
        self.mme = 0
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.n_sample_points = 500

    def compute_all_metrics(self, class_name: str, suffix: str):

        filename = os.path.join("summary_%s_average_%s.txt" % (class_name, suffix))
        dump = open(filename, "w+")
        stat_meter = Stat(cls=class_name, suffix=suffix, dump=dump, num_sample_pts=self.n_sample_points)
        self.compute_third_party_metrics()

        # Evaluate the mean average metrics.
        best_mota, best_threshold = 0, -10000
        threshold_list = self.get_thresholds(self.scores, self.num_gt)
        for threshold_tmp in threshold_list:
            data_tmp = dict()
            self.reset()
            self.compute_third_party_metrics(threshold_tmp)
            data_tmp['mota'], data_tmp['motp'], data_tmp['precision'], data_tmp['F1'], data_tmp['fp'], data_tmp['fn'],\
                data_tmp['recall'] = \
                self.MOTA, self.MOTP, self.precision, self.F1, self.fp, self.fn, self.recall
            stat_meter.update(data_tmp)
            mota_tmp = self.MOTA
            if mota_tmp > best_mota:
                best_threshold = threshold_tmp
                best_mota = mota_tmp
                self.save_to_stats(dump, threshold_tmp)

                self.reset()
                self.compute_third_party_metrics(best_threshold)
                self.save_to_stats(dump)

        stat_meter.output()
        summary = stat_meter.print_summary()
        print(summary)

        stat_meter.plot()
        dump.close()

    def get_thresholds(self, scores, num_gt):
        # based on score of true positive to discretize the recall
        # may not be 11 due to not fully recall the results, all the results point has zero precision
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / float(num_gt)
            if i < (len(scores) - 1):
                r_recall = (i + 2) / float(num_gt)
            else:
                r_recall = l_recall
            if (r_recall - current_recall) < (current_recall - l_recall) and i < (len(scores) - 1):
                continue

            thresholds.append(score)
            current_recall += 1 / (self.num_sample_pts - 1.0)

        return thresholds

    def reset(self):
        self.n_gt = 0  # Number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # Number of ignored ground truth detections
        self.n_tr = 0  # Number of tracker detections minus ignored tracker detections
        self.n_itr = 0  # Number of ignored tracker detections

        self.MOTA = 0
        self.MOTP = 0

        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0

        self.total_cost = 0
        self.itp = 0
        self.tp = 0
        self.fn = 0
        self.ifn = 0
        self.fp = 0

        self.n_gts = []  # Number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = []  # Number of ground ignored truth detections PER SEQUENCE
        self.n_trs = []  # Number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itrs = []  # Number of ignored tracker detections PER SEQUENCE

        self.itps = []  # Number of ignored true positives PER SEQUENCE
        self.tps = []  # Number of true positives including ignored true positives PER SEQUENCE
        self.fns = []  # Number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifns = []  # Number of ignored false negatives PER SEQUENCE
        self.fps = []  # Above PER SEQUENCE

        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.gt_trajectories = dict()
        self.ign_trajectories = dict()
        self.scores = list()

    def compute_third_party_metrics(self, threshold: float = None):
        """
            Computes the metrics defined in:
            - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
              MOTA, MOTP
            - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
              MT/PT/ML
        """
        # Init.
        self.gt_trajectories = dict()
        self.ign_trajectories = dict()
        self.scores = list()

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
                print('Tracks before filtering: %d, after filtering: %d' %
                      (len(scene_tracks_pred_unfiltered), len(scene_tracks_pred)))

            # Statistics over the current sequence.
            # Check the corresponding variable comments in __init__ to get their meaning.
            # The *_trajectories fields both map from GT track_id and timestamp to pred track_id.
            scene_gt_trajectories: Dict[str, Dict[str, Dict[int, str]]] = dict()
            scene_ign_trajectories: Dict[str, Dict[str, Dict[int, str]]] = dict()
            seqtp = 0
            seqitp = 0
            seqfn = 0
            seqifn = 0
            seqfp = 0
            seqigt = 0
            seqitr = 0
            n_gts = 0
            n_trs = 0

            for timestamp in scene_tracks_gt.keys():
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]

                # Counting total number of ground truth and predicted objects.
                self.n_gt += len(frame_gt)
                self.n_tr += len(frame_pred)
                n_gts += len(frame_gt)
                n_trs += len(frame_pred)

                # Use Hungarian method to compute association between predicted and GT tracks.
                association_matrix, cost_matrix = TrackingEvaluation._hungarian_method(
                    frame_gt, frame_pred, self.dist_fcn, self.dist_th_tp
                )

                # Initially set all GT trajectories as not associated.
                for gg in frame_gt:
                    scene_gt_trajectories[gg.tracking_id] = dict()
                    scene_gt_trajectories[gg.tracking_id][gg.timestamp] = ''
                    scene_ign_trajectories[gg.tracking_id] = dict()
                    scene_ign_trajectories[gg.tracking_id][gg.timestamp] = False

                # Update tp/fn stats.
                for row, col in association_matrix:
                    c = cost_matrix[row][col]
                    if c < self.dist_th_tp:
                        # Update stats
                        self.total_cost += 1 - c
                        self.tp += 1

                        gg = frame_gt[row]
                        tt = frame_pred[col]
                        scene_gt_trajectories[gg.tracking_id][gg.timestamp] = tt.tracking_id
                        self.scores.append(tt.tracking_score)
                    else:
                        self.fn += 1

            # Remove empty lists for current gt trajectories.
            self.gt_trajectories[scene_id] = scene_gt_trajectories
            self.ign_trajectories[scene_id] = scene_ign_trajectories

            # Gather statistics for "per sequence" statistics.
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)

        # Compute the relevant metrics.
        self._compute_metrics()

    def create_summary_details(self):
        """
            Generate and mail a summary of the results.
            If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += "evaluation: best results with single threshold".center(80, "=") + "\n"
        summary += self.print_entry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA) + "\n"
        summary += self.print_entry("Multiple Object Tracking Precision (MOTP)", float(self.MOTP)) + "\n"
        summary += "\n"
        summary += self.print_entry("Recall", self.recall) + "\n"
        summary += self.print_entry("Precision", self.precision) + "\n"
        summary += self.print_entry("F1", self.F1) + "\n"
        summary += "\n"
        summary += self.print_entry("Mostly Tracked", self.MT) + "\n"
        summary += self.print_entry("Partly Tracked", self.PT) + "\n"
        summary += self.print_entry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.print_entry("True Positives", self.tp) + "\n"
        summary += self.print_entry("False Positives", self.fp) + "\n"
        summary += self.print_entry("False Negatives", self.fn) + "\n"
        summary += self.print_entry("ID-switches", self.id_switches) + "\n"
        summary += self.print_entry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += self.print_entry("Ground Truth Objects (Total)", self.n_gt + self.n_igt) + "\n"
        summary += self.print_entry("Ground Truth Trajectories", self.n_gt_trajectories) + "\n"
        summary += "\n"
        summary += self.print_entry("Tracker Objects (Total)", self.n_tr) + "\n"
        summary += self.print_entry("Tracker Trajectories", self.n_tr_trajectories) + "\n"
        summary += "=" * 80

        return summary

    def create_summary_simple(self, threshold):
        """
            Generate and mail a summary of the results.
            If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += ("evaluation with confidence threshold %f" % threshold).center(80, "=") + "\n"
        summary += ' MOTA   MOTP   MODA   MODP    MT     ML     IDS  FRAG    F1   Prec  Recall      TP    FP    FN\n'

        summary += '{:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:5d}\n'.format(
            self.MOTA, self.MOTP, self.MT, self.ML, self.id_switches, self.fragments,
            self.F1, self.precision, self.recall, self.tp, self.fp, self.fn)
        summary += "=" * 80

        return summary

    def print_entry(self, key, val, width=(70, 10)):
        """
            Pretty print an entry in a table fashion.
        """
        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%d.4f" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def save_to_stats(self, dump, threshold=None):
        """
            Save the statistics in a whitespace separate file.
        """
        # write summary to file summary_cls.txt
        if threshold is None:
            summary = self.create_summary_details()
        else:
            summary = self.create_summary_simple(threshold)
        self.mail.msg(summary)  # mail or print the summary.
        print(summary, file=dump)

    @staticmethod
    def _threshold_tracks(scene_tracks_pred_unfiltered: Dict[int, List[TrackingBox]],
                          threshold: float) \
            -> Dict[int, List[TrackingBox]]:
        """
        For the current threshold, remove the tracks with low confidence for each frame.
        Note that the average of ther per-frame scores forms the track-level score.
        :param scene_tracks_pred_unfiltered: The predicted tracks for this scene.
        :param threshold: The score threshold.
        :return: A subset of the predicted tracks with scores above the thresold.
        """
        assert threshold is not None, 'Error: threshold must be specified!'
        scene_tracks_pred = {}
        for track_id, track in scene_tracks_pred_unfiltered.items():
            # Compute average score for current track.
            track_scores = []
            for box in track:
                track_scores.append(box.tracking_score)
            avg_score = np.mean(track_scores)

            # Decide whether to keep track by tresholding.
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

    def _compute_metrics(self):
        """
        TODO
        """

        # Compute MT/PT/ML, fragments and idswitches for all GT trajectories.
        n_ignored_tr_total = 0
        for scene_trajectories, scene_ignored in zip(self.gt_trajectories.values(), self.ign_trajectories.values()):

            if len(scene_trajectories) == 0:
                continue

            n_ignored_tr = 0
            for g, ign_g in zip(scene_trajectories.values(), scene_ignored.values()):
                assert len(g) == len(ign_g)

                # All frames of this GT trajectory are ignored.
                if all(ign_g):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue

                # All frames of this GT trajectory are not assigned to any detections.
                if all([this == -1 for this in g]):
                    self.ML += 1
                    continue

                # Compute tracked frames in trajectory.
                last_id = g[0]

                # First detection (necessary to be in gt_trajectories) is always tracked.
                tracked = 1 if g[0] >= 0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt += 1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f - 1] != -1:
                        self.id_switches += 1
                    if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]

                # Handle last frame; tracked state is handled in for loop (g[f]!=-1).
                if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and not ign_g[f]:
                    self.fragments += 1

                # Compute MT/PT/ML.
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    self.ML += 1
                else:  # 0.2 <= tracking_ratio <= 0.8
                    self.PT += 1

        if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # Precision, recall and F1 metrics.
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        if (self.recall + self.precision) == 0:
            self.F1 = 0.
        else:
            self.F1 = 2. * (self.precision * self.recall) / (self.precision + self.recall)

        # Compute CLEARMOT metrics.
        if self.n_gt == 0:
            self.MOTA = -float("inf")
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
                 cls,
                 suffix,
                 dump,
                 num_sample_pts: int = 11):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.mota = 0
        self.motp = 0
        self.F1 = 0
        self.precision = 0
        self.fp = 0
        self.fn = 0

        self.mota_list = list()
        self.motp_list = list()
        self.f1_list = list()
        self.precision_list = list()
        self.fp_list = list()
        self.fn_list = list()
        self.recall_list = list()

        self.cls = cls
        self.suffix = suffix
        self.dump = dump
        self.num_sample_pts = num_sample_pts

    def update(self, data):
        self.mota += data['mota']
        self.motp += data['motp']
        self.F1 += data['F1']
        self.precision += data['precision']
        self.fp += data['fp']
        self.fn += data['fn']

        self.mota_list.append(data['mota'])
        self.motp_list.append(data['motp'])
        self.f1_list.append(data['F1'])
        self.precision_list.append(data['precision'])
        self.fp_list.append(data['fp'])
        self.fn_list.append(data['fn'])
        self.recall_list.append(data['recall'])

    def output(self):
        self.ave_mota = self.mota / self.num_sample_pts
        self.ave_motp = self.motp / self.num_sample_pts
        self.ave_f1 = self.F1 / self.num_sample_pts
        self.ave_precision = self.precision / self.num_sample_pts

    def print_summary(self):
        summary = ""

        summary += "evaluation: average at all thresholds".center(80, "=") + "\n"
        summary += ' AMOTA  AMOTP  APrec\n'

        summary += '{:.4f} {:.4f} {:.4f}\n'.format(
            self.ave_mota, self.ave_motp, self.ave_precision)
        summary += "=" * 80
        print(summary, file=self.dump)

        return summary

    def plot_over_recall(self, data_list, title, y_name, save_path):
        # add extra zero at the end
        largest_recall = self.recall_list[-1]
        extra_zero = np.arange(largest_recall, 1, 0.01).tolist()
        len_extra = len(extra_zero)
        y_zero = [0] * len_extra

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(self.recall_list + extra_zero), np.array(data_list + y_zero))
        # ax.set_title(title, fontsize=20)
        ax.set_ylabel(y_name, fontsize=20)
        ax.set_xlabel('Recall', fontsize=20)
        ax.set_xlim(0.0, 1.0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        if y_name in ['MOTA', 'MOTP', 'F1', 'Precision']:
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(0.0, max(data_list))

        if y_name in ['MOTA', 'F1']:
            max_ind = np.argmax(np.array(data_list))
            # print(max_ind)
            plt.axvline(self.recall_list[max_ind], ymax=data_list[max_ind], color='r')
            plt.plot(self.recall_list[max_ind], data_list[max_ind], 'or', markersize=12)
            plt.text(self.recall_list[max_ind] - 0.05, data_list[max_ind] + 0.03, '%.2f' % (data_list[max_ind]),
                     fontsize=20)
        fig.savefig(save_path)
        # zxc

    def plot(self):
        save_dir = os.path.join("./results", 'plot')

        self.plot_over_recall(self.mota_list, 'MOTA - Recall Curve', 'MOTA',
                              os.path.join(save_dir, 'MOTA_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.motp_list, 'MOTP - Recall Curve', 'MOTP',
                              os.path.join(save_dir, 'MOTP_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.f1_list, 'F1 - Recall Curve', 'F1',
                              os.path.join(save_dir, 'F1_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.fp_list, 'False Positive - Recall Curve', 'False Positive',
                              os.path.join(save_dir, 'FP_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.fn_list, 'False Negative - Recall Curve', 'False Negative',
                              os.path.join(save_dir, 'FN_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.precision_list, 'Precision - Recall Curve', 'Precision',
                              os.path.join(save_dir, 'precision_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))

class Mail:
    """ Dummy class to print messages without sending e-mails"""
    def __init__(self,mailaddress):
        pass
    def msg(self,msg):
        print(msg)
    def finalize(self,success,benchmark,sha_key,mailaddress=None):
        if success:
            print("Results for %s (benchmark: %s) sucessfully created" % (benchmark,sha_key))
        else:
            print("Creating results for %s (benchmark: %s) failed" % (benchmark,sha_key))

