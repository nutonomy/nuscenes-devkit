from typing import Tuple, Dict, Callable

import numpy as np

from nuscenes.eval.detection.data_classes import DetectionConfig, EvalBoxes
from nuscenes.eval.detection.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc


def average_precision(gt_boxes: EvalBoxes,
                      pred_boxes: EvalBoxes,
                      class_name: str,
                      cfg: DetectionConfig,
                      dist_fcn: Callable = center_distance,
                      dist_th: float = 2.0,
                      score_range: Tuple[float, float] = (0.1, 1.0),
                      verbose: bool = False)\
        -> Tuple[float, dict]:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param cfg: Config.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param score_range: Lower and upper score bound between which we compute the metrics.
    :param verbose: whether to print messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    
    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

    if verbose:
        print('Class %s, dist_th: %.1fm, npos: %d' % (class_name, dist_th, npos))

    # For missing classes in the GT, return nan mAP.
    if npos == 0:
        return np.nan, dict()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp, fp = [], []
    metrics = {key: [] for key in cfg.metric_names}
    metrics.update({'conf': [], 'ego_dist': [], 'vel_magn': []})
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, sample_annotation in enumerate(gt_boxes[pred_box.sample_token]):
            if sample_annotation.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(sample_annotation, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # Update TP / FP and raw metrics and mark matched GT boxes.
        if min_dist < dist_th:
            assert match_gt_idx is not None
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)

            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            trans_err = center_distance(gt_box_match, pred_box)
            vel_err = velocity_l2(gt_box_match, pred_box)
            scale_err = 1 - scale_iou(gt_box_match, pred_box)
            orient_err = yaw_diff(gt_box_match, pred_box)
            attr_err = 1 - attr_acc(gt_box_match, pred_box, cfg.attributes)

            ego_dist = gt_box_match.ego_dist
            vel_magn = np.sqrt(np.sum(np.array(gt_box_match.velocity) ** 2))
        else:
            tp.append(0)
            fp.append(1)

            trans_err = np.nan
            vel_err = np.nan
            scale_err = np.nan
            orient_err = np.nan
            attr_err = np.nan

            ego_dist = np.nan
            vel_magn = np.nan

        metrics['trans_err'].append(trans_err)
        metrics['vel_err'].append(vel_err)
        metrics['scale_err'].append(scale_err)
        metrics['orient_err'].append(orient_err)
        metrics['attr_err'].append(attr_err)
        metrics['conf'].append(pred_confs[ind])

        # For debugging only.
        metrics['ego_dist'].append(ego_dist)
        metrics['vel_magn'].append(vel_magn)

    # Accumulate.
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    tp, fp = tp.astype(np.float), fp.astype(np.float)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    if npos > 0:
        rec = tp / float(npos)
    else:
        rec = 0 * tp

    # IF there are no data points, add a point at (rec, prec) of (0.01, 0) such that the AP equals 0.
    if len(prec) == 0:
        rec = np.array([0.01])
        prec = np.array([0])

    # If there is no precision value for recall == 0, we add a conservative estimate.
    if rec[0] != 0:
        rec = np.append(0.0, rec)
        prec = np.append(prec[0], prec)
        metrics['trans_err'].insert(0, np.nan)
        metrics['vel_err'].insert(0, np.nan)
        metrics['scale_err'].insert(0, np.nan)
        metrics['orient_err'].insert(0, np.nan)
        metrics['attr_err'].insert(0, np.nan)
        metrics['conf'].insert(0, 1)

        # For debugging only.
        metrics['ego_dist'].insert(0, np.nan)
        metrics['vel_magn'].insert(0, np.nan)

    # Store modified rec and prec values.
    metrics['rec'] = rec
    metrics['prec'] = prec

    # If the max recall is below the minimum recall range, return the maximum error
    if max(rec) < min(score_range):
        return np.nan, dict()

    # Find indices of rec that are close to the interpolated recall thresholds.
    assert all(rec == sorted(rec))  # np.searchsorted requires sorted inputs.
    thresh_count = int((score_range[1] - score_range[0]) * 100 + 1)
    rec_interp = np.linspace(score_range[0], score_range[1], thresh_count)
    threshold_inds = np.searchsorted(rec, rec_interp, side='left').astype(np.float32)
    threshold_inds[threshold_inds == len(rec)] = np.nan  # Mark unachieved recall values as such.
    assert np.nanmax(threshold_inds) < len(rec)  # Check that threshold indices are not out of bounds.
    metrics['threshold_inds'] = threshold_inds

    # Interpolation of precisions to the nearest lower recall threshold.
    # For unachieved high recall values the precision is set to 0.
    prec_interp = np.interp(rec_interp, rec, prec, right=0)
    metrics['rec_interp'] = rec_interp
    metrics['prec_interp'] = prec_interp

    # Compute average precision over predefined thresholds.
    average_prec = prec_interp.mean()

    # Plot PR curve.
    # plt.clf()
    # plt.plot(metrics['rec'], metrics['prec'])
    # plt.plot(metrics['rec'], metrics['conf'])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision (blue), Conf (orange)')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall curve: class %s, dist_th=%.1fm, AP=%.4f' % (class_name, dist_th, average_prec))
    # save_path = os.path.join(self.plot_dir, '%s-recall-precision-%.1f.png' % (class_name, dist_th))
    # plt.savefig(save_path)

    # Print stats.
    if verbose:
        tp_count = tp[-1] if len(tp) > 0 else 0
        fp_count = fp[-1] if len(fp) > 0 else 0
        print('AP is %.4f, tp: %d, fp: %d' % (average_prec, tp_count, fp_count))

    return average_prec, metrics


def calc_tp_metrics(raw_metrics: Dict, cfg: DetectionConfig, class_name: str, verbose: bool) -> Dict:
    """
    True Positive metrics for a single class and distance threshold.
    For each metric and recall threshold we compute the mean for all matches with a lower recall.
    The raw_metrics are computed in average_precision() to avoid redundant computation.
    :param raw_metrics: Raw data for a number of metrics.
    :param cfg: Config.
    :param class_name: Class to compute AP on.
    :param verbose: Whether to print outputs to console.
    :return: Maps each metric name to its average metric error.
    """

    def cummean(x):
        """ Computes the cumulative mean up to each position. """
        sum_vals = np.nancumsum(x)  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

    # Init each metric as nan.
    tp_metrics = {key: np.nan for key in cfg.metric_names}

    # If raw_metrics are empty, this means that no GT samples exist for this class.
    # Then we set the metrics to nan and ignore their contribution later on.
    if len(raw_metrics) == 0:
        return tp_metrics

    for metric_name in {key: [] for key in cfg.metric_names}:
        # If no box was predicted for this class, no raw metrics exist and we set secondary metrics to 1.
        # Likewise if all predicted boxes are false positives.
        metric_vals = raw_metrics[metric_name]
        if len(metric_vals) == 0 or all(np.isnan(metric_vals)):
            tp_metrics[metric_name] = 1
            continue

        # Certain classes do not have attributes. In this case keep nan and continue.
        if metric_name == 'attr_err' and class_name in ['barrier', 'traffic_cone']:
            continue

        # Normalize and clip metric errors.
        metric_bound = cfg.metric_bounds[metric_name]
        metric_vals = np.array(metric_vals) / metric_bound  # Normalize.
        metric_vals = np.minimum(1, metric_vals)  # Clip.

        # Compute mean metric error for every sample (sorted by conf).
        metric_cummeans = cummean(metric_vals)

        # Average over predefined recall thresholds.
        # Note: For unachieved recall values this assigns the maximum possible error (1). This punishes methods that
        # do not achieve these recall values and users that intentionally submit less boxes.
        metric_errs = np.ones((len(raw_metrics['threshold_inds']),))
        valid = np.where(np.logical_not(np.isnan(raw_metrics['threshold_inds'])))[0]
        valid_thresholds = raw_metrics['threshold_inds'][valid].astype(np.int32)
        metric_errs[valid] = metric_cummeans[valid_thresholds]
        tp_metrics[metric_name] = metric_errs.mean()

        # Write plot to disk.
        # if self.plot_dir is not None:
        #     plt.clf()
        #     plt.plot(raw_metrics['rec_interp'], metric_errs)
        #     plt.xlabel('Recall r')
        #     plt.ylabel('%s of matches with recall <= r' % metric_name)
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.title('%s curve: class %s, avg=%.4f' % (metric_name, class_name, tp_metrics[metric_name]))
        #     save_path = os.path.join(self.plot_dir, '%s-recall-%s.png' % (class_name, metric_name))
        #     plt.savefig(save_path)

        if verbose:
            clip_ratio: float = np.mean(metric_vals == 1)
            print('%s: %.4f, %.1f%% values clipped' % (metric_name, tp_metrics[metric_name], clip_ratio * 100))

    return tp_metrics
