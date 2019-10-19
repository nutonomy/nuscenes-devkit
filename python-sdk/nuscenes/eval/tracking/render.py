# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import matplotlib.pyplot as plt
import numpy as np

from nuscenes.eval.detection.render import setup_axis  # TODO: move to common
from nuscenes.eval.tracking.data_classes import TrackingMetricDataList, TrackingMetrics
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES


def recall_metric_curve(md_list: TrackingMetricDataList,
                        metrics: TrackingMetrics,
                        metric_name: str,
                        min_precision: float,
                        min_recall: float,
                        savepath: str = None) -> None:
    # Setup plot.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    if metric_name in ['amota', 'motap', 'recall', 'mota']:
        ylim = 1
    else:
        ylim = None
    ax = setup_axis(xlabel='Recall', ylabel=metric_name.upper(),
                    xlim=1, ylim=ylim, min_precision=None, min_recall=min_recall, ax=ax)

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

        print(metric_name)
        print(tracking_name)
        print(recalls)
        print(values)  # TODO: remove

        ax.plot(recalls,
                values,
                label='%s' % PRETTY_TRACKING_NAMES[tracking_name],
                color=TRACKING_COLORS[tracking_name])

    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, loc='best', borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
