# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np

from nuscenes.eval.detection.data_classes import EvalBoxes, MetricData
from nuscenes.eval.detection.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc


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
    # Step0: Organize input and inititialize accumulators
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

    # For missing classes in the GT, return nan mAP.
    if npos == 0:
        return MetricData()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives
    fp = []  # Accumulator of false positives
    confs = []  # Accumulator of confidences

    # match_data holds the extra metrics we calcualte for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': [],
                  'ego_dist': [],
                  'vel_magn': []}

    # ---------------------------------------------
    # Step1: Match and accumulate match data
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
            confs.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

            # For debugging only.
            match_data['ego_dist'].append(gt_box_match.ego_dist)
            match_data['vel_magn'].append(np.sqrt(np.sum(np.array(gt_box_match.velocity) ** 2)))

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            confs.append(pred_box.detection_score)

    # Now that the data has been accumulated we will apply three post-processing step

    # ---------------------------------------------
    # Step2: accumulate and catch corner cases for precision / recall curve
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    confs = np.array(confs)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    if npos > 0:
        rec = tp / float(npos)
    else:
        rec = 0 * tp

    # If there are no data points, add a point at (rec, prec) of (0.01, 0) such that the AP equals 0.
    if len(prec) == 0:
        rec = np.array([0.01])
        prec = np.array([0])
        confs = np.array([0.5])

    # If there is no precision value for recall == 0, we add a conservative estimate.
    if rec[0] != 0:
        rec = np.append(0.0, rec)
        prec = np.append(prec[0], prec)
        confs = np.append(1, confs)

    # ---------------------------------------------
    # Step3: Re-sample recall, precision and confidences such that we have one data point for each
    # recall percentage between 0 and 1.
    # ---------------------------------------------

    rec_interp = np.linspace(0, 1,  MetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, confs, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Step 4: Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    def cummean(x):
        """ Computes the cumulative mean up to each position. """
        if len(np.isnan(x)) == len(x):
            # Is all numbers in array are NaN's.
            return np.empty(len(x))
        else:
            # Accumulate in a nan-aware manner.
            sum_vals = np.nancumsum(x)  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
            return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to aligh with fp and tp. So skip in this step.

        # For each match_data, we first calculate the accumulated mean.
        tmp = cummean(match_data[key])

        # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
        match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp)

    # ---------------------------------------------
    # Done: Instantiate MetricData and return
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
    """ Calculates true positive metrics """

    first_ind = round(100 * min_recall)
    last_ind = np.nonzero(md.confidence)[0][-1]  # First instance of confidence = 0 is index of max achived recall.
    if last_ind < first_ind:
        return 1  # Assign 1 here. If this happends for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind]))
