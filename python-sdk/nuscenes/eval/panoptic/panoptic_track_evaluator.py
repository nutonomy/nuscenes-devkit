"""
Multi-object Panoptic Tracking evaluation.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""

from typing import Dict, List, Tuple

import numpy as np
from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval


class PanopticTrackingEval(PanopticEval):
    """ Panoptic tracking evaluator"""

    def __init__(self,
                 n_classes: int,
                 min_stuff_cls_id: int,
                 ignore: List[int] = None,
                 offset: int = 2 ** 32,
                 min_points: int = 30,
                 iou_thr: float = 0.5):
        """
        :param n_classes: Number of classes.
        :param min_stuff_cls_id: Minimum stuff class index, 11 for Panoptic nuScenes challenge classes.
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

        # PAT Tracking stuff.
        self.instance_preds = {}
        self.instance_gts = {}

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
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], np.ndarray]:
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

            # Computation for PAT score
            # Computes unique gt instances and its number of points > self.min_points
            unique_gt_, counts_gt_ = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
            id2idx_gt_ = {inst_id: idx for idx, inst_id in enumerate(unique_gt_)}
            # Computes unique pred instances (class-agnotstic) and its number of points
            unique_pred_, counts_pred_ = np.unique(x_inst_row[x_inst_row > 0], return_counts=True)
            id2idx_pred_ = {inst_id: idx for idx, inst_id in enumerate(unique_pred_)}
            # Actually unique_combo_ = pred_labels_ + self.offset * gt_labels_
            gt_labels_ = unique_combo_ // self.offset
            pred_labels_ = unique_combo_ % self.offset
            gt_areas_ = np.array([counts_gt_[id2idx_gt_[g_id]] for g_id in gt_labels_])
            pred_areas_ = np.array([counts_pred_[id2idx_pred_[p_id]] for p_id in pred_labels_])
            # Here counts_combo_ : TP (point-level)
            intersections_ = counts_combo_
            # Here gt_areas_ : TP + FN, pred_areas_ : TP + FP (point-level)
            # Overall unions_ : TP + FP + FN (point-level)
            unions_ = gt_areas_ + pred_areas_ - intersections_
            # IoU : TP / (TP + FP + FN)
            ious_agnostic = intersections_.astype(np.float32) / unions_.astype(np.float32)
            # tp_indexes_agnostic : TP (instance-level, IoU > 0.5)
            tp_indexes_agnostic = ious_agnostic > 0.5
            matched_gt_ = np.array([False] * len(id2idx_gt_))
            matched_gt_[[id2idx_gt_[g_id] for g_id in gt_labels_[tp_indexes_agnostic]]] = True

            # Stores matched tracks (the corresponding class-agnostic predicted instance) for the unique gt instances:
            for idx, value in enumerate(tp_indexes_agnostic):
                if value:
                    g_label = gt_labels_[idx]
                    p_label = pred_labels_[idx]
                    if g_label not in self.instance_gts[scene][cl]:
                        self.instance_gts[scene][cl][g_label] = [p_label,]
                    else:
                        self.instance_gts[scene][cl][g_label].append(p_label)

            # Stores unmatched tracks for the unique gt instances: assigns 1 for no match 
            for g_label in unique_gt_:
                if not matched_gt_[id2idx_gt_[g_label]]:
                    if g_label not in self.instance_gts[scene][cl]:
                        self.instance_gts[scene][cl][g_label] = [1,]
                    else:
                        self.instance_gts[scene][cl][g_label].append(1)

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
            self.instance_preds[scene] = {}
            self.instance_gts[scene] = [{} for _ in range(self.n_classes)]
        # Make sure instance IDs are non-zeros. Otherwise, they will be ignored. Note in Panoptic nuScenes,
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

        # Accumulate class-agnostic predictions
        unique_pred_, counts_pred_ = np.unique(x_inst_row[1][x_inst_row[1] > 0], return_counts=True) 
        for p_id in unique_pred_[counts_pred_ > self.min_points]:
            if p_id not in self.instance_preds[scene]:
                self.instance_preds[scene][p_id] = 1
            else:
                self.instance_preds[scene][p_id] += 1

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

        ground_truths = tp + fn
        # Get classes that at least has 1 ground truth instance (use threshold 0.5), and is included in self.include.
        valid_classes = ground_truths > 0.5
        for i in range(valid_classes.shape[0]):
            if i not in self.include:
                valid_classes[i] = False

        # Mean PTQ and sPTQ over all classes except invalid (ignored or classes has zero ground truth) classes.
        mean_ptq = ptq_all[valid_classes].mean()
        mean_soft_ptq = soft_ptq_all[valid_classes].mean()

        return mean_ptq, ptq_all, mean_soft_ptq, soft_ptq_all

    def get_motsa(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MOTSA metrics.
        :return: (mean_MOTSA, mean_sMOTSA, mean_MOTSP).
            mean_MOTSA: <float64, 1>, mean MOTSA score over all thing classes.
            mean_sMOTSA: <float64, 1>, mean soft-MOTSA score over all thing classes.
            mean_sMOTSP: <float64, 1>, mean soft-MOTSP score over all thing classes.
        """
        iou = self.pan_iou[1:self.min_stuff_cls_id].astype(np.double)
        ids = self.pan_ids[1:self.min_stuff_cls_id].astype(np.double)

        # Get tp, fp, and fn for all things: class 1:min_stuff_cls_id.
        tp = self.pan_tp[1:self.min_stuff_cls_id].astype(np.double)
        fp = self.pan_fp[1:self.min_stuff_cls_id].astype(np.double)
        tp_eps, fn = np.maximum(tp, self.eps), self.pan_fn[1:self.min_stuff_cls_id].astype(np.double)

        ground_truths = tp + fn
        # Get classes that at least has 1 ground truth instance (use threshold 0.5), and is included in self.include.
        valid_classes = ground_truths > 0.5
        for i in range(valid_classes.shape[0]):
            if i + 1 not in self.include:  # i + 1 as valid_clssses covers class IDs of 1:self.min_stuff_cls_id.
                valid_classes[i] = False

        # Calculate MOTSA of all valid thing classes.
        motsa = (tp - fp - ids)[valid_classes] / (tp_eps + fn)[valid_classes]
        # Calculate sMOTSA of all valid thing classes.
        s_motsa = (iou - fp - ids)[valid_classes] / (tp_eps + fn)[valid_classes]
        motsp = iou[valid_classes] / tp_eps[valid_classes]
        mean_motsa = motsa.mean()  # Mean MOTSA over all thing classes.
        mean_s_motsa = s_motsa.mean()  # Mean sMOTSA over all thing classes.
        mean_motsp = motsp.mean()

        return mean_motsa, mean_s_motsa, mean_motsp

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

    def get_pat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Panoptic Tracking (PAT) metric. https://arxiv.org/pdf/2109.03805.pdf
       :return: (PAT, mean_PQ, mean_TQ).
            PAT: <float64, 1>, PAT score over all classes.
            mean_PQ: <float64, 1>, mean PQ scores over all classes.
            mean_TQ: <float64, 1>, mean TQ score over all classes.
        """
        # First calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
            self.eps)
        pq_all = sq_all * rq_all

        # Then do the REAL mean (no ignored classes)
        pq = pq_all[self.include].mean()

        accumulate_tq = 0.0
        accumulate_norm = 0

        for seq in self.sequences:
            preds = self.instance_preds[seq]
            for cl in self.include:
                cls_gts = self.instance_gts[seq][cl]
                for gt_id, pr_ids in cls_gts.items():
                    unique_pr_id, counts_pr_id = np.unique(pr_ids, return_counts=True)

                    track_length = len(pr_ids)
                    # void/stuff have instance value 1 due to the +1 in ln205 as well as unmatched gt is denoted by 1 
                    # Thus we remove 1 from the prediction id list  
                    unique_pr_id, counts_pr_id = unique_pr_id[unique_pr_id != 1], counts_pr_id[unique_pr_id != 1] 
                    fp_pr_id = []

                    # Computes the total false positive for each prediction id:
                    #     preds[uid]: TPA + FPA (class-agnostic)
                    #     counts_pr_id[idx]: TPA (class-agnostic)
                    # If prediction id is not in preds it means it has number of points < self.min_points.
                    # Similar to PQ computation we consider pred with number of points < self.min_points 
                    # with IoU overlap greater than 0.5 over gt as TPA but not for FPA (the else part).
                    for idx, uid in enumerate(unique_pr_id):
                        if uid in preds:
                            fp_pr_id.append(preds[uid] - counts_pr_id[idx])
                        else:
                            fp_pr_id.append(0)

                    fp_pr_id = np.array(fp_pr_id)
                    # AQ component of TQ where counts_pr_id = TPA, track_length = TPA + FNA, fp_pr_id = FPA.
                    gt_id_aq = np.sum(counts_pr_id ** 2 / np.double(track_length + fp_pr_id)) / np.double(track_length)
                    # Assigns ID switch component of TQ as 1.0 if the gt instance occurs only once.  
                    gt_id_is = 1.0

                    if track_length > 1:
                        # Compute the ID switch component 
                        s_id = -1
                        ids = 0
                        # Total possible id switches
                        total_ids = track_length - 1
                        # Gt tracks with no corresponding prediction match are assigned 1.
                        # We consider an id switch occurs if previous predicted id and the current one doesn't match
                        # for the given gt tracker or if there is no matching prediction for the given gt track
                        for pr_id in pr_ids:
                            if s_id != -1:
                                if pr_id != s_id or s_id == 1: 
                                    ids += 1
                            s_id = pr_id
                        gt_id_is = 1-(ids/np.double(total_ids))     
                    # Accumulate TQ over all the possible unique gt instances
                    accumulate_tq += np.sqrt(gt_id_aq * gt_id_is)
                    # Count the total number of unique gt instances
                    accumulate_norm += 1 
        # Normalization
        tq = np.array(accumulate_tq/accumulate_norm)
        pat = (2 * pq * tq) / (pq + tq)
        return pat, pq, tq

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
