"""
Multi-object Panoptic Tracking evaluation.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""

from typing import Dict, List, Tuple

import numpy as np
from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval


class PanopticTrackingEval(PanopticEval):
    """ Multi-object panoptic tracking evaluator"""

    def __init__(self,
                 n_classes: int,
                 min_stuff_cls_id: int,
                 ignore: List[int] = None,
                 offset: int = 2 ** 32,
                 min_points: int = 30,
                 iou_thr: float = 0.5):
        """
        :param n_classes: Number of classes.
        :param min_stuff_cls_id: Minimum stuff class index, 11 for nuScenes-panoptic challenge classes.
        :param ignore: List of ignored class index.
        :param offset: Largest instance number in a frame.
        :param min_points: minimal number of points to consider instances in GT.
        :param iou_thr: IoU threshold to consider as a true positive. Note "iou_thr > 0.5" is required for Panoptic
            Quality metric and its variants.
        """
        super().__init__(n_classes=n_classes, ignore=ignore, offset=offset, min_points=min_points)
        self.iou_thr = iou_thr
        assert self.iou_thr >= 0.5, f'IoU threshold mush be >= 0.5, but {self.iou_thr} is given.'

        self.min_stuff_cls_id = min_stuff_cls_id

        # IoU stuff.
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

        # Panoptic stuff.
        self.pan_ids = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_soft_ids = np.zeros(self.n_classes, dtype=np.double)
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        # Tracking stuff.
        self.sequences = []
        self.preds = {}
        self.gts = {}
        self.intersects = {}
        self.intersects_ovr = {}

        # Per-class association quality stuff.
        self.pan_aq = np.zeros(self.n_classes, dtype=np.double)
        self.pan_aq_ovr = 0.0

    @staticmethod
    def update_dict_stat(stat_dict: Dict[int, int], unique_ids: np.ndarray, unique_cnts: np.ndarray) -> None:
        """
        Update stats dict with new combo of ids and counts.
        :param stat_dict: {class_id: counts}, a dict of stats for the counts of each class.
        :param unique_ids: <np.int64, <k,>>, an array of class IDs.
        :param unique_cnts: <np.int64, <k,>>, an array of counts for corresponding class IDs.
        """
        for uniqueid, counts in zip(unique_ids, unique_cnts):
            if uniqueid in stat_dict:
                stat_dict[uniqueid] += counts
            else:
                stat_dict[uniqueid] = counts

    def get_panoptic_track_stats(self,
                                 x_inst_in_cl: np.ndarray,
                                 y_inst_in_cl: np.ndarray,
                                 x_inst_row: np.ndarray = None,
                                 scene: str = None,
                                 cl: int = None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int],
                     Dict[int, int], np.ndarray]:
        """
        Calculate class-specific panoptic tracking stats given predicted instances and target instances.
        :param x_inst_in_cl: <np.int64: num_points>, instance IDs of each point for predicted instances.
        :param y_inst_in_cl: <np.int64: num_points>, instance IDs of each point for target instances.
        :param x_inst_row: <np.int64: num_points>, class-agnostic instance IDs of each point for predicted instances.
        :param scene: str, name of scene.
        :param cl: int, semantic class id.
        :return: A tuple of MOPT stats:
            {
              counts_pred, # <np.int64, num_instances>, point counts of each predicted instance.
              counts_gt,  # <np.int64, num_instances>, point counts of each ground truth instance.
              gt_labels,  # <np.int64, num_instances>, instance ID of each ground truth instance.
              pred_labels, # <np.int64, num_instances>, instance ID of each predicted instance.
              id2idx_gt, # {instance ID: array index}, instance ID to array index mapping for ground truth instances.
              id2idx_pred, # {instance ID: array index}, instance ID to array index mapping for predicted instances.
              ious, # <np.float32, num_instances>, IoU scores between prediction and ground truth instance pair.
            }
        """
        # Generate the areas for each unique instance in prediction.
        unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
        id2idx_pred = {inst_id: idx for idx, inst_id in enumerate(unique_pred)}

        # Generate the areas for each unique instance in ground truth.
        unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
        id2idx_gt = {inst_id: idx for idx, inst_id in enumerate(unique_gt)}

        # Generate intersection using offset.
        valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
        offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
        unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

        # Per-class accumulated stats.
        if scene is not None and cl < self.min_stuff_cls_id:
            cl_preds = self.preds[scene]
            cl_gts = self.gts[scene][cl]
            cl_intersects = self.intersects[scene][cl]
            self.update_dict_stat(cl_gts,
                                  unique_gt[counts_gt > self.min_points],
                                  counts_gt[counts_gt > self.min_points])
            self.update_dict_stat(cl_preds,
                                  unique_pred[counts_pred > self.min_points],
                                  counts_pred[counts_pred > self.min_points])
            valid_combos_min_point = np.zeros_like(y_inst_in_cl)  # instances which have more than self.min points
            for valid_id in unique_gt[counts_gt > self.min_points]:
                valid_combos_min_point = np.logical_or(valid_combos_min_point, y_inst_in_cl == valid_id)
            y_inst_in_cl = y_inst_in_cl * valid_combos_min_point
            valid_combos_ = np.logical_and(x_inst_row > 0, y_inst_in_cl > 0)
            offset_combo_ = x_inst_row[valid_combos_] + self.offset * y_inst_in_cl[valid_combos_]
            unique_combo_, counts_combo_ = np.unique(offset_combo_, return_counts=True)
            self.update_dict_stat(cl_intersects, unique_combo_, counts_combo_)

        # Generate an intersection map, count the intersections with over 0.5 IoU as TP.
        gt_labels = unique_combo // self.offset
        pred_labels = unique_combo % self.offset
        gt_areas = np.array([counts_gt[id2idx_gt[g_id]] for g_id in gt_labels])
        pred_areas = np.array([counts_pred[id2idx_pred[p_id]] for p_id in pred_labels])
        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.astype(np.float32) / unions.astype(np.float32)

        return counts_pred, counts_gt, gt_labels, pred_labels, id2idx_gt, id2idx_pred, ious

    def add_batch_panoptic(self,
                           scene: str,
                           x_sem_row: List[np.ndarray],
                           x_inst_row: List[np.ndarray],
                           y_sem_row: List[np.ndarray],
                           y_inst_row: List[np.ndarray]) -> None:
        """
        Add panoptic tracking metrics for one frame/batch.
        :param scene: str, name of scene.
        :param x_sem_row: [None, <np.int64: num_points>], predicted semantics.
        :param x_inst_row: [None, <np.uint64: num_points>], predicted instances.
        :param y_sem_row: [None, <np.int64: num_points>], target semantics.
        :param y_inst_row: [None, <np.uint64: num_points>], target instances.
        """
        if scene not in self.sequences:
            self.sequences.append(scene)
            self.preds[scene] = {}
            self.gts[scene] = [{} for _ in range(self.n_classes)]
            self.intersects[scene] = [{} for _ in range(self.n_classes)]
            self.intersects_ovr[scene] = [{} for _ in range(self.n_classes)]
        # Make sure instance IDs are non-zeros. Otherwise, they will be ignored. Note in nuScenes-panoptic,
        # instance IDs start from 1 already, so the following 2 lines of code are actually not necessary, but to be
        # consistent with the PanopticEval class in panoptic_seg_evaluator.py from 3rd party. We keep these 2 lines. It
        # means the actual instance IDs will start from 2 during metrics evaluation.
        x_inst_row[1] = x_inst_row[1] + 1
        y_inst_row[1] = y_inst_row[1] + 1

        # Only interested in points that are outside the void area (not in excluded classes).
        for cl in self.ignore:
            # Current Frame.
            gt_not_in_excl_mask = y_sem_row[1] != cl  # make a mask for class cl.
            # Remove all other points.
            x_sem_row[1] = x_sem_row[1][gt_not_in_excl_mask]
            y_sem_row[1] = y_sem_row[1][gt_not_in_excl_mask]
            x_inst_row[1] = x_inst_row[1][gt_not_in_excl_mask]
            y_inst_row[1] = y_inst_row[1][gt_not_in_excl_mask]

            # Previous Frame.
            if x_sem_row[0] is not None:  # First frame.
                gt_not_in_excl_mask = y_sem_row[0] != cl
                # Remove all other points.
                x_sem_row[0] = x_sem_row[0][gt_not_in_excl_mask]
                y_sem_row[0] = y_sem_row[0][gt_not_in_excl_mask]
                x_inst_row[0] = x_inst_row[0][gt_not_in_excl_mask]
                y_inst_row[0] = y_inst_row[0][gt_not_in_excl_mask]

        # First step is to count intersections > 0.5 IoU for each class (except the ignored ones).
        for cl in self.include:
            # Previous Frame.
            inst_prev, gt_labels_prev, tp_indexes_prev = None, None, None
            if x_sem_row[0] is not None:  # First frame.
                x_inst_in_cl_mask = x_sem_row[0] == cl
                y_inst_in_cl_mask = y_sem_row[0] == cl

                # Get instance points in class (makes outside stuff 0).
                x_inst_in_cl = x_inst_row[0] * x_inst_in_cl_mask.astype(np.int64)
                y_inst_in_cl = y_inst_row[0] * y_inst_in_cl_mask.astype(np.int64)
                _, _, gt_labels_prev, inst_prev, _, _, ious = self.get_panoptic_track_stats(x_inst_in_cl, y_inst_in_cl)
                tp_indexes_prev = ious > self.iou_thr

            # Current Frame: get a class mask.
            x_inst_in_cl_mask = x_sem_row[1] == cl
            y_inst_in_cl_mask = y_sem_row[1] == cl

            # Get instance points in class (makes outside stuff 0).
            x_inst_in_cl = x_inst_row[1] * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row[1] * y_inst_in_cl_mask.astype(np.int64)

            counts_pred, counts_gt, gt_labels, pred_labels, id2idx_gt, id2idx_pred, ious =\
                self.get_panoptic_track_stats(x_inst_in_cl, y_inst_in_cl, x_inst_row[1], scene, cl)
            inst_cur = pred_labels
            tp_indexes = ious > 0.5

            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt = np.array([False] * len(id2idx_gt))
            matched_gt[[id2idx_gt[g_id] for g_id in gt_labels[tp_indexes]]] = True
            matched_pred = np.array([False] * len(id2idx_pred))
            matched_pred[[id2idx_pred[p_id] for p_id in pred_labels[tp_indexes]]] = True

            # Count the FN.
            self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, np.logical_not(matched_gt)))
            # Count the FP.
            self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, np.logical_not(matched_pred)))

            # Compute ID switches (IDS).
            if x_sem_row[0] is not None and cl < self.min_stuff_cls_id:  # Skip first frame.
                gt_labels_prev, gt_labels = gt_labels_prev[tp_indexes_prev], gt_labels[tp_indexes]
                inst_prev, inst_cur = inst_prev[tp_indexes_prev], inst_cur[tp_indexes]
                ious = ious[tp_indexes]
                _, prev_ind, cur_ind = np.intersect1d(gt_labels_prev, gt_labels, return_indices=True)

                ids, soft_ids = 0, 0.0
                for prev_i, cur_i in zip(prev_ind, cur_ind):
                    if inst_prev[prev_i] != inst_cur[cur_i]:
                        ids += 1
                        soft_ids += ious[cur_i]

                self.pan_ids[cl] += ids
                self.pan_soft_ids[cl] += soft_ids

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

        ptq_all = ((iou - ids) / tp_eps) * (tp / tp_half_fp_half_fn_eps)  # Calculate PTQ of all classes.
        soft_ptq_all = ((iou - soft_ids) / tp_eps) * (tp / tp_half_fp_half_fn_eps)  # Calculate soft-PTQ of all classes.
        mean_ptq = ptq_all[self.include].mean()  # Mean PTQ over all classes except ignored classes.
        mean_soft_ptq = soft_ptq_all[self.include].mean()  # Mean soft-PTQ over all classes except ignored classes.

        return mean_ptq, ptq_all, mean_soft_ptq, soft_ptq_all

    def get_lstq(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Lidar Segmentation and Tracking Quality (LSTQ) metric. https://arxiv.org/pdf/2102.12472.pdf
        :return: (LSTQ, S_assoc). LSTQ: <float64, 1>, LSTQ score over all classes. S_assoc: <float64, 1,>, S_assoc for
            all classes.
        """
        num_tubes = [0] * self.n_classes
        for seq in self.sequences:
            for cl in self.include:
                cl_preds = self.preds[seq]
                cl_gts = self.gts[seq][cl]
                cl_intersects = self.intersects[seq][cl]
                outer_sum_iou = 0.0
                num_tubes[cl] += len(cl_gts)
                for gt_id, gt_size in cl_gts.items():
                    inner_sum_iou = 0.0
                    for pr_id, pr_size in cl_preds.items():
                        tpa_key = pr_id + self.offset * gt_id
                        if tpa_key in cl_intersects:
                            tpa_ovr = cl_intersects[tpa_key]
                            inner_sum_iou += tpa_ovr * (tpa_ovr / (gt_size + pr_size - tpa_ovr))
                    outer_sum_iou += inner_sum_iou / float(gt_size)
                self.pan_aq[cl] += outer_sum_iou
                self.pan_aq_ovr += outer_sum_iou
        s_assoc = np.sum(self.pan_aq) / np.sum(num_tubes[1:self.min_stuff_cls_id])  # num_things 1:11.
        s_cls, iou = self.getSemIoU()
        lstq = np.sqrt(s_assoc * s_cls)
        return lstq, s_assoc

    def add_batch(self, scene: str, x_sem: List[np.ndarray], x_inst: List[np.ndarray], y_sem: List[np.ndarray],
                  y_inst: List[np.ndarray]) -> None:
        """
        Add semantic IoU and panoptic tracking metrics for one frame/batch.
        :param scene: str, name of scene.
        :param x_sem: [None, <np.int64: num_points>], predicted semantics.
        :param x_inst: [None, <np.uint64: num_points>], predicted instances.
        :param y_sem: [None, <np.int64: num_points>], target semantics.
        :param y_inst: [None, <np.uint64: num_points>], target instances.
        """
        self.addBatchSemIoU(x_sem[1], y_sem[1])  # Add to IoU calculation for checking purpose.
        self.add_batch_panoptic(scene, x_sem, x_inst, y_sem, y_inst)  # Do panoptic tracking stuff.
