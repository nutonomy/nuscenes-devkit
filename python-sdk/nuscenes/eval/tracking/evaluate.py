# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import MetricDataList
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingConfig, TrackingBox
from nuscenes.eval.tracking.external import TrackingEvaluation, Stat, Mail
from nuscenes.eval.tracking.render import visualize_sample


class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following tracking metrics:
    TODO

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
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

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
            "Samples in split doesn't match samples in predicted tracks."

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

    def evaluate(self) -> Tuple[TrackingMetrics, MetricDataList]:
        """
        Performs the actual evaluation.
        This code is based on Xinshuo Weng's AB3DMOT code at:
        https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)
        metric_data_list = MetricDataList() # TODO: Do we still need this?

        suffix = self.cfg.dist_fcn.title().lower()
        mail = Mail("")
        num_sample_pts = self.cfg.num_sample_pts
        for class_name in self.cfg.class_names:
            ev = TrackingEvaluation(self.nusc, self.gt_boxes, self.pred_boxes, class_name, mail=mail, num_sample_pts=num_sample_pts)
            filename = os.path.join("summary_%s_average_%s.txt" % (class_name, suffix))
            dump = open(filename, "w+")
            stat_meter = Stat(cls=class_name, suffix=suffix, dump=dump, num_sample_pts=num_sample_pts)
            ev.compute_third_party_metrics()

            # evaluate the mean average metrics
            best_mota, best_threshold = 0, -10000
            threshold_list = ev.get_thresholds(ev.scores, ev.num_gt)
            for threshold_tmp in threshold_list:
                data_tmp = dict()
                ev.reset()
                ev.compute_third_party_metrics(threshold_tmp)
                data_tmp['mota'], data_tmp['motp'], data_tmp['precision'], \
                data_tmp['F1'], data_tmp['fp'], data_tmp['fn'], data_tmp['recall'] = \
                    ev.MOTA, ev.MOTP, ev.precision, ev.F1, ev.fp, ev.fn, ev.recall
                stat_meter.update(data_tmp)
                mota_tmp = ev.MOTA
                if mota_tmp > best_mota:
                    best_threshold = threshold_tmp
                    best_mota = mota_tmp
                ev.save_to_stats(dump, threshold_tmp)

            ev.reset()
            ev.compute_third_party_metrics(best_threshold)
            ev.save_to_stats(dump)

            stat_meter.output()
            summary = stat_meter.print_summary()
            print(summary)

            stat_meter.plot()
            dump.close()

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: TrackingMetrics, md_list: MetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: TrackingMetrics instance.
        :param md_list: MetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # TODO

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        # TODO
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        # TODO

        return metrics_summary


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
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('nips_tracking_2019')
    else:
        with open(config_path, 'r') as f:
            cfg_ = TrackingConfig.deserialize(json.load(f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = TrackingEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                             output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
