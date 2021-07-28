"""
Multi-object Panoptic Tracking evaluation.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""

from typing import Dict, List, Tuple

import numpy as np
from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval


class PanopticTrackingEval(PanopticEval):
    """ Multi-object panoptic tracking evaluator"""

    def __init__(self, n_classes: int, device: str = None, ignore: List[int] = None, offset: int = 2 ** 32,
                 min_points: int = 30):
        """
        :param n_classes: Number of classes.
        :param ignore: List of ignored class index.
        :param offset: Largest instance number in a frame.
        :param min_points: minimal number of points to consider instances in GT.
        """
        super().__init__(n_classes=n_classes, device=device, ignore=ignore, offset=offset, min_points=min_points)
        self.pan_ids = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_soft_ids = np.zeros(self.n_classes, dtype=np.int64)

    def get_panoptic_track_stats(self, x_inst_in_cl: np.ndarray, y_inst_in_cl: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int],
                     Dict[int, int], np.ndarray]:
        """
        Calculate class-specific panoptic tracking stats given predicted instances and target instances.
        :param x_inst_in_cl: <np.int64: num_points>, instance IDs of each point for predicted instances.
        :param y_inst_in_cl: <np.int64: num_points>, instance IDs of each point for target instances.
        :return: A tuple of MOPT stats:
            {
              counts_pred, # <np.int64, num_instances>, point counts of each predicted instance.
              counts_gt,  # <np.int64, num_instances>, point counts of each ground truth instance.
              gt_labels,  # <np.int64, num_instances>, instance ID of each ground truth instance.
              pred_labels, # <np.int64, num_instances>, instance ID of each predicted instance.
              matched_pred, # <np.bool, num_instances>, whether each predicted instance has matched ground truth.
              matched_gt, # <np.bool, num_instances>, whether each predicted instance has matched prediction.
              id2idx_gt, # {instance ID: array index}, instance ID to array index mapping for ground truth instances.
              id2idx_pred, # {instance ID: array index}, instance ID to array index mapping for predicted instances.
              ious, # <np.float32, num_instances>, IoU scores between prediction and ground truth instance pair.
            }
        """
        # generate the areas for each unique instance in prediction.
        unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
        id2idx_pred = {inst_id: idx for idx, inst_id in enumerate(unique_pred)}
        matched_pred = np.array([False] * unique_pred.shape[0])

        # generate the areas for each unique instance in ground truth.
        unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
        id2idx_gt = {inst_id: idx for idx, inst_id in enumerate(unique_gt)}
        matched_gt = np.array([False] * unique_gt.shape[0])

        # generate intersection using offset
        valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
        offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
        unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

        # generate an intersection map, count the intersections with over 0.5 IoU as TP
        gt_labels = unique_combo // self.offset
        pred_labels = unique_combo % self.offset
        gt_areas = np.array([counts_gt[id2idx_gt[g_id]] for g_id in gt_labels])
        pred_areas = np.array([counts_pred[id2idx_pred[p_id]] for p_id in pred_labels])
        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.astype(np.float32) / unions.astype(np.float32)

        return counts_pred, counts_gt, gt_labels, pred_labels, matched_pred, matched_gt, id2idx_gt, id2idx_pred, ious

    def add_batch_panoptic(self,
                           x_sem_row: List[np.ndarray],
                           x_inst_row: List[np.ndarray],
                           y_sem_row: List[np.ndarray],
                           y_inst_row: List[np.ndarray]) -> None:
        """
        Add panoptic tracking metrics for one frame/batch.
        :param x_sem_row: [None, <np.int64: num_points>], predicted semantics.
        :param x_inst_row: [None, <np.uint64: num_points>], predicted instances.
        :param y_sem_row: [None, <np.int64: num_points>], target semantics.
        :param y_inst_row: [None, <np.uint64: num_points>], target instances.
        """
        # make sure instances are not zeros (it messes with my approach)
        x_inst_row[1] = x_inst_row[1] + 1
        y_inst_row[1] = y_inst_row[1] + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore:
            # Current Frame:
            # make a mask for this class
            gt_not_in_excl_mask = y_sem_row[1] != cl
            # remove all other points
            x_sem_row[1] = x_sem_row[1][gt_not_in_excl_mask]
            y_sem_row[1] = y_sem_row[1][gt_not_in_excl_mask]
            x_inst_row[1] = x_inst_row[1][gt_not_in_excl_mask]
            y_inst_row[1] = y_inst_row[1][gt_not_in_excl_mask]

            # Previous Frame
            if x_sem_row[0] is not None:  # First frame
                gt_not_in_excl_mask = y_sem_row[0] != cl
                # remove all other points
                x_sem_row[0] = x_sem_row[0][gt_not_in_excl_mask]
                y_sem_row[0] = y_sem_row[0][gt_not_in_excl_mask]
                x_inst_row[0] = x_inst_row[0][gt_not_in_excl_mask]
                y_inst_row[0] = y_inst_row[0][gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in self.include:
            # Previous Frame
            if x_sem_row[0] is not None:  # First frame
                x_inst_in_cl_mask = x_sem_row[0] == cl
                y_inst_in_cl_mask = y_sem_row[0] == cl

                # get instance points in class (makes outside stuff 0)
                x_inst_in_cl = x_inst_row[0] * x_inst_in_cl_mask.astype(np.int64)
                y_inst_in_cl = y_inst_row[0] * y_inst_in_cl_mask.astype(np.int64)
                _, _, gt_labels_prev, inst_prev, _, _, _, _, ious = \
                    self.get_panoptic_track_stats(x_inst_in_cl, y_inst_in_cl)
                tp_indexes = ious > 0.5
                not_tp_indexes = np.logical_not(tp_indexes)
                inst_prev[not_tp_indexes] = -1

            # Current Frame:
            # get a class mask
            x_inst_in_cl_mask = x_sem_row[1] == cl
            y_inst_in_cl_mask = y_sem_row[1] == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row[1] * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row[1] * y_inst_in_cl_mask.astype(np.int64)

            counts_pred, counts_gt, gt_labels, pred_labels, matched_pred, matched_gt, id2idx_gt, id2idx_pred, ious = \
                self.get_panoptic_track_stats(x_inst_in_cl, y_inst_in_cl)
            inst_cur = pred_labels
            tp_indexes = ious > 0.5
            not_tp_indexes = np.logical_not(tp_indexes)
            inst_cur[not_tp_indexes] = -1

            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[g_id] for g_id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[p_id] for p_id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))
            # count the FP
            self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

            # compute ids
            if x_sem_row[0] is not None:  # skip first frame
                track_id = np.intersect1d(gt_labels_prev, gt_labels)
                trac_id_prev = np.intersect1d(track_id, inst_prev)
                trac_id_cur = np.intersect1d(track_id, inst_cur)
                ids = list(set(trac_id_prev).symmetric_difference(trac_id_cur))

                self.pan_ids[cl] += len(ids)
                ssid_ind = np.where(np.in1d(ids, inst_cur))[0]
                self.pan_soft_ids[cl] += np.sum(ious[ssid_ind])

    def get_ptq(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate PTQ metrics.
        :return: (mean_PTQ, all_class_PTQ, mean_sPTQ, all_class_sPTQ).
            mean_PTQ: <float64, 1>, mean PTQ score over all classes.
            all_class_PTQ: <float64, num_classes,>, PTQ scores for all classes.
            mean_sPTQ: <float64, 1>, mean soft-PTQ score over all classes.
            all_class_sPTQ: <float64, num_classes,>, soft-PTQ scores for all classes.
        """
        iou = self.pan_iou.astype(np.double)
        ids, soft_ids = self.pan_ids.astype(np.double), self.pan_soft_ids.astype(np.double)
        tp, fp = self.pan_tp.astype(np.double), self.pan_fp.astype(np.double)
        tp_eps, fn = np.maximum(tp, self.eps), self.pan_fn.astype(np.double)
        tp_half_fp_half_fn_eps = np.maximum(tp + 0.5 * fp + 0.5 * fn, self.eps)

        ptq_all = ((iou - ids) / tp_eps) * (tp / tp_half_fp_half_fn_eps)  # calculate PTQ of all classes.
        soft_ptq_all = ((iou - soft_ids) / tp_eps) * (tp / tp_half_fp_half_fn_eps)  # calculate soft-PTQ of all classes.
        mean_ptq = ptq_all[self.include].mean()  # mean PTQ over all classes except ignored classes.
        mean_soft_ptq = soft_ptq_all[self.include].mean()  # mean soft-PTQ over all classes except ignored classes.

        return mean_ptq, ptq_all, mean_soft_ptq, soft_ptq_all

    def add_batch(self, x_sem: List[np.ndarray], x_inst: List[np.ndarray], y_sem: List[np.ndarray],
                  y_inst: List[np.ndarray]) -> None:
        """
        Add semantic IoU and panoptic tracking metrics for one frame/batch.
        :param x_sem: [None, <np.int64: num_points>], predicted semantics.
        :param x_inst: [None, <np.uint64: num_points>], predicted instances.
        :param y_sem: [None, <np.int64: num_points>], target semantics.
        :param y_inst: [None, <np.uint64: num_points>], target instances.
        """
        self.addBatchSemIoU(x_sem[1], y_sem[1])  # add to IoU calculation for checking purpose.
        self.add_batch_panoptic(x_sem, x_inst, y_sem, y_inst)  # do panoptic tracking stuff.
