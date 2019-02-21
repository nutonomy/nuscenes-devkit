# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os
import time
import json
from typing import List, Tuple, Dict, Callable
import random
import argparse

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.eval_utils import center_distance, category_to_detection_name, filter_boxes, \
    visualize_sample, scale_iou, yaw_diff, velocity_l2, attr_acc
from nuscenes.eval.create_splits_logs import create_splits_logs


class NuScenesEval:
    """
    This is the official nuScenes detection evaluation code.

    nuScenes uses the following metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - Weighted sum: The weighted sum of the above.

    Here is an overview of the functions in this method:
    - load_boxes(): Loads GT annotations an predictions stored in JSON format.
    - run_eval(): Performs evaluation and returns the above metrics.
    - average_precision(): Computes AP for a single distance threshold.
    - tp_metrics(): Computes the TP metrics.

    We assume that:
    - Every sample_token is given in the results, although there may be no predictions for that sample.

    Please see https://github.com/nutonomy/nuscenes-devkit for more details.
    """
    def __init__(self, nusc: NuScenes, result_path: str, eval_set: str, class_names: List[str]=None,
                 output_dir: str=None, verbose: bool=True, eval_limit: int=-1):
        """
        Initialize a NuScenesEval object.
        :param nusc: A NuScenes object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. teaser/train/val/test.
        :param class_names: List of classes to evaluate. If None, all detection classes are used.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        :param eval_limit: Number of images to evaluate or -1 to evaluate all images.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.class_names = class_names
        self.output_dir = output_dir
        self.verbose = verbose
        self.eval_limit = eval_limit

        # Define evaluation classes and criterion.
        if self.class_names is None:
            self.class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                                'traffic_cone', 'trailer', 'truck']
        self.eval_range = 40  # Range in meters beyond which boxes are ignored.
        self.dist_fcn = center_distance
        self.dist_ths = [0.5, 1.0, 2.0, 4.0]  # mAP distance thresholds.
        self.dist_th_tp = 2.0  # TP metric distance threshold.
        assert self.dist_th_tp in self.dist_ths
        self.metric_bounds = {  # Arbitrary upper bounds for each metric.
            'trans_err': 0.5,
            'vel_err': 1.5,
            'scale_err': 0.5,
            'orient_err': np.pi / 2,
            'attr_err': 1
        }
        self.attributes = ['cycle.with_rider', 'cycle.without_rider',
                           'pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing',
                           'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
        self.score_range = (0.1, 1.0)  # Do not include low recall thresholds as they are very noisy.
        self.weighted_sum_tp_metrics = ['trans_err', 'scale_err', 'orient_err']  # Which metrics to include by default.
        self.max_boxes_per_sample = 500  # Abort if there are more boxes in any sample.
        self.mean_ap_weight = 5  # The relative weight of mAP in the weighted sum metric.

        self.metric_names = self.metric_bounds.keys()

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Dummy init.
        self.sample_tokens = None
        self.all_annotations = None
        self.all_results = None

        # Load and store GT and predictions.
        self.sample_tokens, self.all_annotations, self.all_results = self.load_boxes()

    def load_boxes(self) -> Tuple[List[str], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Loads the GT and EST boxes used in this class.
        :return: (sample_tokens, all_annotations, all_results). The sample tokens and the mapping from token to GT and
            predictions.
        """

        # Init.
        all_annotations = dict()
        attribute_map = {a['name']: a['token'] for a in self.nusc.attribute}

        if self.verbose:
            print('## Loading annotations and results...')

        # Load results.
        with open(self.result_path) as f:
            start_time = time.time()
            all_results = json.load(f)
            end_time = time.time()

            if self.verbose:
                print('Loading results for %d samples in %.1fs.' % (len(all_results), end_time - start_time))
                print('Loading annotations...')

        # Read sample_tokens.
        splits = create_splits_logs(self.nusc)
        sample_tokens_all = [s['token'] for s in self.nusc.sample]
        assert len(sample_tokens_all) > 0, 'Error: Results file is empty!'

        # Only keep samples from this split.
        sample_tokens = []
        for sample_token in sample_tokens_all:
            scene_token = self.nusc.get('sample', sample_token)['scene_token']
            log_token = self.nusc.get('scene', scene_token)['log_token']
            logfile = self.nusc.get('log', log_token)['logfile']
            if logfile in splits[self.eval_set]:
                sample_tokens.append(sample_token)

        # Limit number of images for debugging.
        if self.eval_limit != -1:
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:self.eval_limit]
            sample_tokens = sorted(sample_tokens)

        # Check completeness of the results.
        result_sample_tokens = list(all_results.keys())
        missing_bool = [x not in result_sample_tokens for x in sample_tokens]
        if any(missing_bool):
            missing_tokens = np.array(sample_tokens)[missing_bool]
            raise Exception('Error: GT sample(s) missing in results: %s' % missing_tokens)

        # Check that each sample has no more than x predicted boxes.
        for sample_token in sample_tokens:
            assert len(all_results[sample_token]) <= self.max_boxes_per_sample, \
                "Error: Only <= %d boxes per sample allowed!" % self.max_boxes_per_sample

        # Check that each result has the right format.
        field_formats = {
            'sample_token': (str, -1), 'translation': (list, 3), 'size': (list, 3), 'rotation': (list, 4),
            'velocity': (list, 3), 'detection_name': (str, -1), 'detection_score': (float, -1),
            'attribute_scores': (list, 8)
        }
        for sample_token in sample_tokens:
            sample_results = all_results[sample_token]
            for sample_result in sample_results:
                for field_name, (field_type, field_len) in field_formats.items():
                    cur_type = type(sample_result[field_name])
                    assert cur_type == field_type, 'Error: Expected %s, got %s for field %s!' \
                                                   % (field_type, cur_type, field_name)
                    if field_len != -1:  # Ignore the length of fields with -1 entries.
                        cur_len = len(sample_result[field_name])
                        assert cur_len == field_len, 'Error: Expected %d values, got %d values for field %s!' \
                                                     % (field_len, cur_len, field_name)

        # Load annotations and filter predictions and annotations.
        for sample_token in tqdm.tqdm(sample_tokens):
            # Load GT.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']

            # Augment with detection name and velocity and filter unused labels.
            sample_annotations = []
            for sample_annotation_token in sample_annotation_tokens:
                # Get label name in detection task and filter unused labels.
                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue
                sample_annotation['detection_name'] = detection_name

                # Get attribute_labels.
                attribute_labels = np.zeros((len(self.attributes),))
                for i, attribute in enumerate(self.attributes):
                    if attribute_map[attribute] in sample_annotation['attribute_tokens']:
                        attribute_labels[i] = 1.0
                sample_annotation['attribute_labels'] = attribute_labels.tolist()

                # Compute object velocity.
                sample_annotation['velocity'] = self.nusc.box_velocity(sample_annotation['token'])

                sample_annotations.append(sample_annotation)

            # Filter boxes > x meters away.
            sample_rec = self.nusc.get('sample', sample_token)
            sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            sample_results = all_results[sample_token]
            sample_annotations, dists_annotations = filter_boxes(sample_annotations, pose_record, cs_record, self.eval_range)
            sample_results, dists_results = filter_boxes(sample_results, pose_record, cs_record, self.eval_range)

            # Store the distances to the ego vehicle.
            for sample_annotation, dist in zip(sample_annotations, dists_annotations):
                sample_annotation['ego_dist'] = dist

            for sample_result, dist in zip(sample_results, dists_results):
                sample_result['ego_dist'] = dist

            # Aggregate.
            all_annotations[sample_token] = sample_annotations
            all_results[sample_token] = sample_results

        return sample_tokens, all_annotations, all_results

    def run_eval(self) -> Dict:
        """
        Perform evaluation given the predictions and annotations stored in self.
        :return: All nuScenes detection metrics. These are also written to disk.
        """

        start_time = time.time()

        # Check that data has been loaded.
        assert self.all_annotations is not None

        # Compute metrics.
        raw_metrics = {label: [] for label in self.class_names}
        label_aps = {label: [] for label in self.class_names}
        label_tp_metrics = {label: [] for label in self.class_names}
        for class_name in self.class_names:
            if self.verbose:
                print('\n# Computing stats for class %s' % class_name)

            # Compute AP and to get the confidence thresholds used for TP metrics.
            dist_th_count = len(self.dist_ths)
            label_aps[class_name] = np.zeros((dist_th_count,))
            raw_metrics[class_name] = [None] * dist_th_count
            for d, dist_th in enumerate(self.dist_ths):
                label_aps[class_name][d], raw_metrics[class_name][d] = \
                    self.average_precision(self.all_annotations, self.all_results, class_name, dist_fcn=self.dist_fcn,
                                           dist_th=dist_th, score_range=self.score_range)

            # Given the raw metrics, compute the TP metrics.
            tp_ind = [i for i, dist_th in enumerate(self.dist_ths) if dist_th == self.dist_th_tp][0]
            label_tp_metrics[class_name] = self.tp_metrics(raw_metrics[class_name][tp_ind], class_name)

        # Compute stats.
        mean_ap = np.nanmean(np.vstack(list(label_aps.values())))  # Nan APs are ignored.
        tp_metrics = dict()
        for metric in self.metric_names:
            tp_metrics[metric] = np.nanmean([label_tp_metrics[label][metric] for label in self.class_names])
        tp_metrics_neg = [1 - tp_metrics[m] for m in self.weighted_sum_tp_metrics]
        weighted_sum = np.sum([self.mean_ap_weight * mean_ap] + tp_metrics_neg)  # Sum with special weight for mAP.
        weighted_sum /= (self.mean_ap_weight + len(tp_metrics_neg))  # Normalize by total weight.
        end_time = time.time()
        eval_time = end_time - start_time

        # Print stats.
        if self.verbose:
            print('\n# Results')
            print('mAP: %.4f' % mean_ap)
            print('TP metrics:\n%s' % '\n'.join(['  %s: %.4f' % (k, v) for (k, v) in tp_metrics.items()]))
            print('Weighted sum: %.4f' % weighted_sum)
            print('Evaluation time: %.1fs.' % eval_time)

        # Write metrics to disk.
        all_metrics = {
            'label_aps': {k: v.tolist() for (k, v) in label_aps.items()},
            'label_tp_metrics': label_tp_metrics,
            'mean_ap': mean_ap,
            'tp_metrics': tp_metrics,
            'weighted_sum': weighted_sum,
            'eval_time': eval_time
        }
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics

    def average_precision(self, all_annotations: dict, all_results: dict, class_name: str,
                          dist_fcn: Callable=center_distance, dist_th: float=2.0,
                          score_range: Tuple[float, float]=(0.1, 1.0)) -> Tuple[float, dict]:
        """
        Average Precision over predefined different recall thresholds for a single distance threshold.
        The recall/conf thresholds and other raw metrics will be used in secondary metrics.
        :param all_annotations: Maps every sample_token to a list of its sample_annotations.
        :param all_results: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param dist_fcn: Distance function used to match detections and ground truths.
        :param dist_th: Distance threshold for a match.
        :param score_range: Lower and upper score bound between which we compute the metrics.
        :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
        """

        # Count the positives.
        npos = 0
        for sample_token in self.sample_tokens:
            for sa_idx, sample_annotation in enumerate(all_annotations[sample_token]):
                if sample_annotation['detection_name'] == class_name:
                    npos += 1
        if self.verbose:
            print('Class %s, dist_th: %.1fm, npos: %d' % (class_name, dist_th, npos))

        # For missing classes in the GT, return nan mAP.
        if npos == 0:
            return np.nan, dict()

        # Organize the estimated bbs in a single list.
        all_results_list, all_confs_list = [], []
        for sample_token in self.sample_tokens:
            for sample_result in all_results[sample_token]:
                if sample_result['detection_name'] == class_name:
                    all_results_list.append(sample_result)
                    all_confs_list.append(sample_result['detection_score'])
        assert len(all_results_list) == len(all_confs_list)

        # Sort by confidence.
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(all_confs_list))][::-1]

        # Do the actual matching.
        tp, fp = [], []
        metrics = {key: [] for key in self.metric_names}
        metrics.update({'conf': [], 'ego_dist': [], 'vel_magn': []})
        taken = set()  # Initially no gt bounding box is matched.
        for ind in sortind:
            sample_result = all_results_list[ind]
            sample_token = sample_result['sample_token']
            min_dist = np.inf
            min_sa_idx = None

            for sa_idx, sample_annotation in enumerate(all_annotations[sample_token]):
                if sample_annotation['detection_name'] == class_name and not (sample_token, sa_idx) in taken:
                    this_distance = dist_fcn(sample_annotation, sample_result)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        min_sa_idx = sa_idx

            # Update TP / FP and raw metrics and mark matched GT boxes.
            if min_dist < dist_th:
                assert min_sa_idx is not None
                taken.add((sample_token, min_sa_idx))
                tp.append(1)
                fp.append(0)

                min_sa = all_annotations[sample_token][min_sa_idx]
                trans_err = center_distance(min_sa, sample_result)
                vel_err = velocity_l2(min_sa, sample_result)
                scale_err = 1 - scale_iou(min_sa, sample_result)
                orient_err = yaw_diff(min_sa, sample_result)
                attr_err = 1 - attr_acc(min_sa, sample_result, self.attributes)

                ego_dist = min_sa['ego_dist']
                vel_magn = np.sqrt(np.sum(np.array(min_sa['velocity']) ** 2))
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
            metrics['conf'].append(all_confs_list[ind])

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

        # Store original values.
        metrics['rec'] = rec
        metrics['prec'] = prec

        # IF there are no data points, add a point at (rec, prec) of (0.01, 0) such that the AP equals 0.
        if len(prec) == 0:
            rec = np.array([0.01])
            prec = np.array([0])

        # If there is no precision value for recall == 0, we add a conservative estimate.
        if rec[0] != 0:
            rec = np.append(0.0, rec)
            prec = np.append(prec[0], prec)

        # Find indices of rec that are close to the interpolated recall thresholds.
        assert all(rec == sorted(rec))  # np.searchsorted requires sorted inputs.
        thresh_count = int((score_range[1] - score_range[0]) * 100 + 1)
        rec_interp = np.linspace(score_range[0], score_range[1], thresh_count)
        threshold_inds = np.searchsorted(rec, rec_interp, side='left').astype(np.float32)
        threshold_inds[threshold_inds == len(rec)] = np.nan  # Mark unachieved recall values as such.
        metrics['threshold_inds'] = threshold_inds

        # Interpolation of precisions to the nearest lower recall threshold.
        # For unachieved high recall values the precision is set to 0.
        prec_interp = np.interp(rec_interp, rec, prec, right=0)
        metrics['rec_interp'] = rec_interp
        metrics['prec_interp'] = prec_interp

        # Compute average precision over predefined thresholds.
        average_prec = prec_interp.mean()

        # Plot PR curve.
        if self.plot_dir is not None:
            plt.clf()
            plt.plot(metrics['rec'], metrics['prec'])
            plt.plot(metrics['rec'], metrics['conf'])
            plt.xlabel('Recall')
            plt.ylabel('Precision (blue), Conf (orange)')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve: class %s, dist_th=%.1fm, AP=%.4f' % (class_name, dist_th, average_prec))
            save_path = os.path.join(self.plot_dir, '%s-recall-precision-%.1f.png' % (class_name, dist_th))
            plt.savefig(save_path)

        # Print stats.
        if self.verbose:
            tp_count = tp[-1] if len(tp) > 0 else 0
            fp_count = fp[-1] if len(fp) > 0 else 0
            print('AP is %.4f, tp: %d, fp: %d' % (average_prec, tp_count, fp_count))

        return average_prec, metrics

    def tp_metrics(self, raw_metrics: Dict, class_name: str) -> Dict:
        """
        True Positive metrics for a single class and distance threshold.
        For each metric and recall threshold we compute the mean for all matches with a lower recall.
        The raw_metrics are computed in average_precision() to avoid redundant computation.
        :param raw_metrics: Raw data for a number of metrics.
        :param class_name: Class to compute AP on.
        :return: Maps each metric name to its average metric error.
        """

        def cummean(x):
            """ Computes the cumulative mean up to each position. """
            sum_vals = np.nancumsum(x)  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
            return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

        # Init each metric as nan.
        tp_metrics = {key: np.nan for key in self.metric_names}

        # If raw_metrics are empty, this means that no GT samples exist for this class.
        # Then we set the metrics to nan and ignore their contribution later on.
        if len(raw_metrics) == 0:
            return tp_metrics

        for metric_name in {key: [] for key in self.metric_names}:
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
            metric_bound = self.metric_bounds[metric_name]
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
            if self.plot_dir is not None:
                plt.clf()
                plt.plot(raw_metrics['rec_interp'], metric_errs)
                plt.xlabel('Recall r')
                plt.ylabel('%s of matches with recall <= r' % metric_name)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.title('%s curve: class %s, avg=%.4f' % (metric_name, class_name, tp_metrics[metric_name]))
                save_path = os.path.join(self.plot_dir, '%s-recall-%s.png' % (class_name, metric_name))
                plt.savefig(save_path)

            # Print stats.
            if self.verbose:
                clip_ratio: float = np.mean(metric_vals == 1)
                print('%s: %.4f, %.1f%% values clipped' % (metric_name, tp_metrics[metric_name], clip_ratio * 100))

        return tp_metrics


if __name__ == "__main__":
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes result submission.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str, default='~/nuscenes-metrics/results.json',
                        help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='teaser',
                        help='Which dataset split to evaluate on, e.g. teaser, train, val, test.')
    parser.add_argument('--eval_limit', type=int, default=-1,
                        help='Number of images to evaluate or -1 to evaluate all images in the split.')
    parser.add_argument('--plot_examples', type=int, default=0,
                        help='Whether to plot example visualizations to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()
    result_path = os.path.expanduser(args.result_path)
    output_dir = os.path.expanduser(args.output_dir)
    eval_set = args.eval_set
    eval_limit = args.eval_limit
    plot_examples = bool(args.plot_examples)
    verbose = bool(args.verbose)

    # Init.
    random.seed(43)
    nusc = NuScenes(verbose=verbose)
    nusc_eval = NuScenesEval(nusc, result_path, eval_set=eval_set, output_dir=output_dir, verbose=verbose,
                             eval_limit=eval_limit)

    # Visualize samples.
    if plot_examples:
        sample_tokens = list(nusc_eval.all_annotations.keys())
        random.shuffle(sample_tokens)
        for sample_token in sample_tokens:
            visualize_sample(nusc, sample_token, nusc_eval.all_annotations, nusc_eval.all_results,
                             eval_range=nusc_eval.eval_range)

    # Run evaluation.
    nusc_eval.run_eval()
