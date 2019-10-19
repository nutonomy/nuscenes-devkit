# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

from typing import Any

import matplotlib.pyplot as plt

Axis = Any


def setup_axis(xlabel: str = None,
               ylabel: str = None,
               xlim: int = None,
               ylim: int = None,
               title: str = None,
               min_precision: float = None,
               min_recall: float = None,
               ax: Axis = None,
               show_spines: str = 'none'):
    """
    Helper method that sets up the axis for a plot.
    :param xlabel: x label text.
    :param ylabel: y label text.
    :param xlim: Upper limit for x axis.
    :param ylim: Upper limit for y axis.
    :param title: Axis title.
    :param min_precision: Visualize minimum precision as horizontal line.
    :param min_recall: Visualize minimum recall as vertical line.
    :param ax: (optional) an existing axis to be modified.
    :param show_spines: Whether to show axes spines, set to 'none' by default.
    :return: The axes object.
    """
    if ax is None:
        ax = plt.subplot()

    ax.get_xaxis().tick_bottom()
    ax.tick_params(labelsize=16)
    ax.get_yaxis().tick_left()

    # Hide the selected axes spines.
    if show_spines in ['bottomleft', 'none']:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_spines == 'none':
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    elif show_spines in ['all']:
        pass
    else:
        raise NotImplementedError

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
