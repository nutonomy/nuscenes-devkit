# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.tracking.data_classes import TrackingMetricDataList
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES, LEGACY_METRICS

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
    rel_metrics = LEGACY_METRICS
    rel_metrics.insert(2, 'motap')
    rel_metrics.insert(3, 'recall')
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
        values = md.get_metric(metric_name)
        nans = np.where(np.logical_not(np.isnan(values)))[0]
        if len(nans) == 0:
            continue
        first_valid = nans[0]
        last_valid = nans[-1]
        recalls = md.recall_hypo[first_valid:last_valid + 1]
        values = values[first_valid:last_valid + 1]

        ax.plot(recalls,
                values,
                label='%s' % PRETTY_TRACKING_NAMES[tracking_name],
                color=TRACKING_COLORS[tracking_name])

    # Scale count statistics and FAF logarithmically.
    if metric_name in ['mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag']:
        ax.set_yscale('symlog')

    if metric_name in ['amota', 'motap', 'recall', 'mota']:
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
