# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]

import argparse
import json
import os
import random
import time
from typing import Tuple

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, MetricDataList, DetectionMetrics
from nuscenes.eval.detection.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample


class NuScenesEval:
    """
    This is the official nuScenes detection evaluation code.

    nuScenes uses the following metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - Weighted sum: The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations an predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

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
        self.pred_boxes = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.)
        if verbose:
            print('=> Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('=> Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def run(self) -> Tuple[DetectionMetrics, MetricDataList]:

        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds
        # -----------------------------------

        metric_data_list = MetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data
        # -----------------------------------

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)

        # -----------------------------------
        # Step 3: Dump the metric data and metrics to disk
        # -----------------------------------

        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics.serialize(), f, indent=2)

        with open(os.path.join(self.output_dir, 'metric_data_list.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        return metrics, metric_data_list

    def render(self, md_list: MetricDataList, metrics: DetectionMetrics):

        def savepath(name):
            return os.path.join(self.plot_dir, name+'.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name+'_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name+'_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_'+str(dist_th)))


def main(result_path, output_dir, eval_set, dataroot, version, verbose, config_name, plot_examples):

    # Init.
    cfg = config_factory(config_name)
    nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)
    nusc_eval = NuScenesEval(nusc_, config=cfg, result_path=result_path, eval_set=eval_set, output_dir=output_dir,
                             verbose=verbose)

    # Visualize samples.
    random.seed(43)
    if plot_examples:
        sample_tokens_ = list(nusc_eval.sample_tokens)
        random.shuffle(sample_tokens_)
        for sample_token_ in sample_tokens_:
            visualize_sample(nusc_, sample_token_, nusc_eval.gt_boxes, nusc_eval.pred_boxes,
                             eval_range=max(nusc_eval.cfg.class_range.values()),
                             savepath=os.path.join(output_dir, '{}.png'.format(sample_token_)))

    # Run evaluation.
    metrics, md_list = nusc_eval.run()
    nusc_eval.render(md_list, metrics)


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes result submission.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, e.g. train or val.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_name', type=str, default='cvpr_2019',
                        help='Name of the configuration to use for evaluation, e.g. cvpr_2019.')
    parser.add_argument('--plot_examples', type=int, default=0,
                        help='Whether to plot example visualizations to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = bool(args.verbose)
    config_name_ = args.config_name
    plot_examples_ = bool(args.plot_examples)

    main(result_path_, output_dir_, eval_set_, dataroot_, version_, verbose_, config_name_, plot_examples_)
