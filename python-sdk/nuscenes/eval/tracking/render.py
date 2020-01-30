# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle, Varun Bankiti, and Alex Lang, 2019.

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion

from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricDataList
from nuscenes.utils.data_classes import Box

Axis = Any


def summary_plot(md_list: TrackingMetricDataList,
                 min_recall: float,
                 ncols: int = 2,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with which includes all traditional metrics for each class.
    :param md_list: TrackingMetricDataList instance.
    :param min_recall: Minimum recall value.
    :param ncols: How many columns the resulting plot should have.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Select metrics and setup plot.
    rel_metrics = ['motar', 'motp', 'mota', 'recall', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
    n_metrics = len(rel_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 5 * nrows))

    # For each metric, plot all the classes in one diagram.
    for ind, metric_name in enumerate(rel_metrics):
        row = ind // ncols
        col = np.mod(ind, ncols)
        recall_metric_curve(md_list, metric_name, min_recall, ax=axes[row, col])

    # Set layout with little white space and save to disk.
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def recall_metric_curve(md_list: TrackingMetricDataList,
                        metric_name: str,
                        min_recall: float,
                        savepath: str = None,
                        ax: Axis = None) -> None:
    """
    Plot the recall versus metric curve for the given metric.
    :param md_list: TrackingMetricDataList instance.
    :param metric_name: The name of the metric to plot.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render or None to create a new axis.
    """
    # Setup plot.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel=metric_name.upper(),
                    xlim=1, ylim=None, min_recall=min_recall, ax=ax, show_spines='bottomleft')

    # Plot the recall vs. precision curve for each detection class.
    for tracking_name, md in md_list.md.items():
        # Get values.
        confidence = md.confidence
        recalls = md.recall_hypo
        values = md.get_metric(metric_name)

        # Filter unachieved recall thresholds.
        valid = np.where(np.logical_not(np.isnan(confidence)))[0]
        if len(valid) == 0:
            continue
        first_valid = valid[0]
        assert not np.isnan(confidence[-1])
        recalls = recalls[first_valid:]
        values = values[first_valid:]

        ax.plot(recalls,
                values,
                label='%s' % PRETTY_TRACKING_NAMES[tracking_name],
                color=TRACKING_COLORS[tracking_name])

    # Scale count statistics and FAF logarithmically.
    if metric_name in ['mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag']:
        ax.set_yscale('symlog')

    if metric_name in ['amota', 'motar', 'recall', 'mota']:
        # Some metrics have an upper bound of 1.
        ax.set_ylim(0, 1)
    elif metric_name != 'motp':
        # For all other metrics except MOTP we set a lower bound of 0.
        ax.set_ylim(bottom=0)

    ax.legend(loc='upper right', borderaxespad=0)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class TrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.

    def render(self, events: DataFrame, timestamp: int, frame_gt: List[TrackingBox], frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        print('Rendering {}'.format(timestamp))
        switches = events[events.Type == 'SWITCH']
        switch_ids = switches.HId.values
        fig, ax = plt.subplots()

        # Plot GT boxes.
        for b in frame_gt:
            color = 'k'
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

        # Plot predicted boxes.
        for b in frame_pred:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

            # Determine color for this tracking id.
            if b.tracking_id not in self.id2color.keys():
                self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
                                                float(hash(b.tracking_id + 'g') % 256) / 255,
                                                float(hash(b.tracking_id + 'b') % 256) / 255)

            # Render box. Highlight identity switches in red.
            if b.tracking_id in switch_ids:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=('r', 'r', color))
            else:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=(color, color, color))

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)))
        plt.close(fig)
