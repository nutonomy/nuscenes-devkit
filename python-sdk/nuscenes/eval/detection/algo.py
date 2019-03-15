# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np

from nuscenes.eval.detection.data_classes import EvalBoxes, MetricData
from nuscenes.eval.detection.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn_name: str,
               dist_th: float):
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn_name: Name of distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """

    dist_fcn_map = {
        'center_distance': center_distance
    }
    dist_fcn = dist_fcn_map[dist_fcn_name]

    # ---------------------------------------------
    # Organize input and inititialize accumulators
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return MetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives
    fp = []  # Accumulator of false positives
    conf = []  # Accumulator of confidences

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': [],
                  'ego_dist': [],
                  'vel_magn': []}

    # ---------------------------------------------
    # Match and accumulate match data
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, sample_annotation in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if sample_annotation.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(sample_annotation, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

            # For debugging only.
            match_data['ego_dist'].append(gt_box_match.ego_dist)
            match_data['vel_magn'].append(np.sqrt(np.sum(np.array(gt_box_match.velocity) ** 2)))

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return MetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, MetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(match_data[key])

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp)

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return MetricData(recall=rec,
                      precision=prec,
                      confidence=conf,
                      trans_err=match_data['trans_err'],
                      vel_err=match_data['vel_err'],
                      scale_err=match_data['scale_err'],
                      orient_err=match_data['orient_err'],
                      attr_err=match_data['attr_err'])


def calc_ap(md: MetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    prec = md.precision
    prec = prec[round(100 * min_recall):]  # Clip low recalls.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: MetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall)
    last_ind = np.nonzero(md.confidence)[0][-1]  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind]))
