# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from nuscenes.eval.detection.utils import boxes_to_sensor
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS
from nuscenes.eval.detection.data_classes import MetricDataList, DetectionMetrics, EvalBoxes


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     eval_range: Dict[str, float],
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     verbose=True) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    """
    max_eval_range = max(eval_range.values())
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / max_eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = max_eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show plot.
    if verbose:
        print('Showing sample token %s' % sample_token)
    plt.title(sample_token)
    plt.show()


def setup_axis(xlabel: str = None,
               ylabel: str = None,
               xlim: int = None,
               ylim: int = None,
               title: str = None,
               min_precision: float = None,
               min_recall: float = None,
               ax=None):

    if ax is None:
        ax = plt.subplot()

    ax.get_xaxis().tick_bottom()
    ax.tick_params(labelsize=16)
    ax.get_yaxis().tick_left()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if title is not None:
        ax.set_title(title, size=24)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=16)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    if min_recall is not None:
        ax.axvline(x=min_recall, linestyle='--', color=(0, 0, 0, 0.3))
    if min_precision is not None:
        ax.axhline(y=min_precision, linestyle='--', color=(0, 0, 0, 0.3))

    return ax


def class_pr_curve(md_list: MetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax=None):
    if ax is None:
        ax = setup_axis(title=pretty_name(detection_name), xlabel='Recall', ylabel='Precision', xlim=1, ylim=1,
                        min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='dist_th: {}, ap: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def class_tp_curve(md_list: MetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax=None):

    if ax is None:
        ax = setup_axis(title=pretty_name(detection_name), xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)

    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan or min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(metric)
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(metric)
        else:
            label = '{}: {:.2f} ({})'.format(metric, tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)

    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def dist_pr_curve(md_list: MetricDataList,
                  metrics: DetectionMetrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None,
                  ax=None) -> None:

    if ax is None:
        ax = setup_axis(title='Distance threshold = {}'.format(dist_th), xlabel='Recall', ylabel='Precision',
                        xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{} ap: {:.1f}'.format(detection_name, ap * 100),
                color=DETECTION_COLORS[detection_name])

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def summary_plot(md_list: MetricDataList,
                 metrics: DetectionMetrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: str = None) -> None:

    n_classes = len(DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(pretty_name(detection_name)), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,  min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def pretty_name(name: str):
    """ Get a pretty name for a detection class. """
    return ' '.join(name.title().split('_'))
