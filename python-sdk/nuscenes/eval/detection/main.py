# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os
import time
import json

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, MetricDataList, DetectionMetrics
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS


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
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range)
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def run(self) -> DetectionMetrics:

        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds
        # -----------------------------------

        metric_data_list = MetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn, dist_th)
                metric_data_list.add(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data
        # -----------------------------------

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)

        # TODO: call rendering methods here

        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics.serialize(), f, indent=2)

        with open(os.path.join(self.output_dir, 'metric_data_list.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        return metrics


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
