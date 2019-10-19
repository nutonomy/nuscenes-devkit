# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import argparse
import json
import os
import time
from typing import Dict, Any, Tuple

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox,\
    TrackingMetricData
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.utils import print_final_metrics
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.render import recall_metric_curve


class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a TrackingEval object.
        :param nusc: A NuScenes object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(self.gt_boxes, self.nusc, self.eval_set)
        self.tracks_pred = create_tracks(self.pred_boxes, self.nusc, self.eval_set)

    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()
        for class_name in self.cfg.class_names:
            ev = TrackingEvaluation(self.tracks_gt, self.tracks_pred, class_name, self.cfg.dist_fcn_callable,
                                    self.cfg.dist_th_tp, self.cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                    verbose=self.verbose)
            md = ev.accumulate()
            metric_data_list.set(class_name, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional metrics.
            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT/pred exist, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.avg_metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: TrackingMetrics, md_list: TrackingMetricDataList) -> None:
        """
        Renders a plot for each class and each metric.
        :param metrics: TrackingMetrics instance.
        :param md_list: TrackingMetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # For each metric, plot all the classes in one diagram.
        for metric_name in LEGACY_METRICS:
            recall_metric_curve(md_list, metrics, metric_name, self.cfg.min_precision,
                                self.cfg.min_recall, savepath=savepath('%s' % metric_name))

    def main(self, render_curves: bool = True) -> TrackingMetrics:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print metrics to stdout.
        print_final_metrics(metrics)

        # Render curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        return metrics


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = TrackingEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                             output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(render_curves=render_curves_)
