# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import matplotlib.pyplot as plt
import numpy as np

from nuscenes.eval.detection.render import setup_axis  # TODO: move to common
from nuscenes.eval.tracking.data_classes import TrackingMetricDataList
from nuscenes.eval.tracking.constants import TRACKING_NAMES, TRACKING_COLORS, PRETTY_TRACKING_NAMES, LEGACY_METRICS


def summary_plot(md_list: TrackingMetricDataList,
                 min_recall: float,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with which includes all traditional metrics for each class.
    :param md_list: TrackingMetricDataList instance.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(TRACKING_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))

    # For each metric, plot all the classes in one diagram.
    rows = round(len(LEGACY_METRICS) / 2)
    for ind, metric_name in enumerate(LEGACY_METRICS):
        ax = setup_axis(xlim=1, ylim=1, title=metric_name.upper(),
                        min_recall=min_recall, ax=axes[np.mod(ind, rows), ind // rows])
        recall_metric_curve(md_list, metric_name, min_recall, ax=ax)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def recall_metric_curve(md_list: TrackingMetricDataList,
                        metric_name: str,
                        min_recall: float,
                        savepath: str = None,
                        ax=None) -> None:
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
                        xlim=1, ylim=None, min_precision=None, min_recall=min_recall, ax=ax)

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

    if metric_name in ['amota', 'motap', 'recall', 'mota']:
        ax.set_ylim(0, 1)
    ax.legend(loc='best', borderaxespad=0)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
