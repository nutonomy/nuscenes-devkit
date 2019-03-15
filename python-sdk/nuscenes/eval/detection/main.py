# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os
import time
import json
from typing import Dict

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.detection.utils import dist_fcn_map
from nuscenes.eval.detection.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.algo import average_precision, calc_tp_metrics


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
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a NuScenesEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train or val.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Make dirs
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data
        self.pred_boxes = load_prediction(self.result_path, self.cfg.max_boxes_per_sample)
        self.gt_boxes = load_gt(self.nusc, self.eval_set)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.)
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.range)
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.range)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def run(self) -> Dict:
        """
        Perform evaluation given the predictions and annotations stored in self.
        :return: All nuScenes detection metrics. These are also written to disk.
        """

        start_time = time.time()

        # Compute metrics.
        raw_metrics = {label: [] for label in self.cfg.class_names}
        label_aps = {label: [] for label in self.cfg.class_names}
        label_tp_metrics = {label: [] for label in self.cfg.class_names}

        for class_name in self.cfg.class_names:
            if self.verbose:
                print('\n# Computing stats for class %s' % class_name)

            # Compute AP and to get the confidence thresholds used for TP metrics.
            dist_th_count = len(self.cfg.dist_ths)
            label_aps[class_name] = np.zeros((dist_th_count,))
            raw_metrics[class_name] = [None] * dist_th_count
            for d, dist_th in enumerate(self.cfg.dist_ths):
                label_aps[class_name][d], raw_metrics[class_name][d] = \
                    average_precision(self.gt_boxes, self.pred_boxes, class_name, self.cfg,
                                      dist_fcn=dist_fcn_map[self.cfg.dist_fcn], dist_th=dist_th,
                                      score_range=self.cfg.recall_range, verbose=self.verbose)

            # Given the raw metrics, compute the TP metrics.
            tp_ind = [i for i, dist_th in enumerate(self.cfg.dist_ths) if dist_th == self.cfg.dist_th_tp][0]
            label_tp_metrics[class_name] = calc_tp_metrics(raw_metrics[class_name][tp_ind], self.cfg, class_name,
                                                           self.verbose)

        # Compute stats.
        mean_ap = np.nanmean(np.vstack(list(label_aps.values())))  # Nan APs are ignored.
        tp_metrics = dict()
        for metric in self.cfg.metric_names:
            tp_metrics[metric] = np.nanmean([label_tp_metrics[label][metric] for label in self.cfg.class_names])
        tp_metrics_neg = [1 - tp_metrics[m] for m in self.cfg.weighted_sum_tp_metrics]
        weighted_sum = np.sum([self.cfg.mean_ap_weight * mean_ap] + tp_metrics_neg)  # Sum with special weight for mAP.
        weighted_sum /= (self.cfg.mean_ap_weight + len(tp_metrics_neg))  # Normalize by total weight.
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
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics


if __name__ == "__main__":
    pass
    # Settings.
    # parser = argparse.ArgumentParser(description='Evaluate nuScenes result submission.',
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--result_path', type=str, default='~/nuscenes-metrics/results.json',
    #                     help='The submission as a JSON file.')
    # parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
    #                     help='Folder to store result metrics, graphs and example visualizations.')
    # parser.add_argument('--eval_set', type=str, default='val',
    #                     help='Which dataset split to evaluate on, e.g. train or val.')
    # parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
    #                     help='Default nuScenes data directory.')
    # parser.add_argument('--version', type=str, default='v0.5',
    #                     help='Which version of the nuScenes dataset to evaluate on, e.g. v0.5.')
    # parser.add_argument('--plot_examples', type=int, default=0,
    #                     help='Whether to plot example visualizations to disk.')
    # parser.add_argument('--verbose', type=int, default=1,
    #                     help='Whether to print to stdout.')
    # args = parser.parse_args()
    # result_path = os.path.expanduser(args.result_path)
    # output_dir = os.path.expanduser(args.output_dir)
    # eval_set = args.eval_set
    # dataroot = args.dataroot
    # version = args.version
    # eval_limit = args.eval_limit
    # plot_examples = bool(args.plot_examples)
    # verbose = bool(args.verbose)
    #
    # Init.
    # random.seed(43)
    # nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)
    # nusc_eval = NuScenesEval(nusc_, result_path, eval_set=eval_set, output_dir=output_dir, verbose=verbose)
    #
    # Visualize samples.
    # if plot_examples:
    #     sample_tokens_ = list(nusc_eval.gt_boxes.keys())
    #     random.shuffle(sample_tokens_)
    #     for sample_token_ in sample_tokens_:
    #         visualize_sample(nusc, sample_token_, nusc_eval.gt_boxes, nusc_eval.pred_boxes,
    #                          eval_range=nusc_eval.cfg.eval_range)
    #
    # Run evaluation.
    # nusc_eval.run()
