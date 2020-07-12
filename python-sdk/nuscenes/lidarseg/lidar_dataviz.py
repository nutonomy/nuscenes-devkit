import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import seaborn as sns

from nuscenes import NuScenes
from nuscenes.utils.color_map import get_colormap


def truncate_class_name(class_name) -> str:
    """
    Truncate a given class name according to a pre-defined map.
    :param class_name: The long form (i.e. original form) of the class name.
    :return: The truncated form of the class name.
    """

    string_mapper = {
        "noise": 'noise',
        "human.pedestrian.adult": 'adult',
        "human.pedestrian.child": 'child',
        "human.pedestrian.wheelchair": 'wheelchair',
        "human.pedestrian.stroller": 'stroller',
        "human.pedestrian.personal_mobility": 'p.mobility',
        "human.pedestrian.police_officer": 'police',
        "human.pedestrian.construction_worker": 'worker',
        "animal": 'animal',
        "vehicle.car": 'car',
        "vehicle.motorcycle": 'motorcycle',
        "vehicle.bicycle": 'bicycle',
        "vehicle.bus.bendy": 'bus.bendy',
        "vehicle.bus.rigid": 'bus.rigid',
        "vehicle.truck": 'truck',
        "vehicle.construction": 'constr. veh',
        "vehicle.emergency.ambulance": 'ambulance',
        "vehicle.emergency.police": 'police car',
        "vehicle.trailer": 'trailer',
        "movable_object.barrier": 'barrier',
        "movable_object.trafficcone": 'trafficcone',
        "movable_object.pushable_pullable": 'push/pullable',
        "movable_object.debris": 'debris',
        "static_object.bicycle_rack": 'bicycle racks',
        "flat.driveable_surface": 'driveable',
        "flat.sidewalk": 'sidewalk',
        "flat.terrain": 'terrain',
        "flat.other": 'flat.other',
        "static.manmade": 'manmade',
        "static.vegetation": 'vegetation',
        "static.other": 'static.other',
        "vehicle.ego": "ego"
    }

    return string_mapper[class_name]


def render_lidarseg_histogram(nusc: NuScenes,
                              sort_by: str = 'count',
                              chart_title: str = None,
                              x_label: str = None,
                              y_label: str = "Lidar points (logarithmic)",
                              y_log_scale: bool = True,
                              verbose: bool = True,
                              font_size: int = 20,
                              save_as_img_name: str = None) -> None:
    """
    Render a histogram for the given nuScenes split.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes:
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param chart_title: Title to display on the histogram.
    :param x_label: Title to display on the x-axis of the histogram.
    :param y_label: Title to display on the y-axis of the histogram.
    :param y_log_scale: Whether to use log scale on the y-axis.
    :param verbose: Whether to display plot in a window after rendering.
    :param font_size: Size of the font to use for the histogram.
    :param save_as_img_name: Path (including image name and extension) to save the histogram as.
    """

    print('Calculating stats for nuScenes-lidarseg...')
    start_time = time.time()

    # Get the statistics for the given nuScenes split.
    class_names, counts = get_lidarseg_stats(nusc, sort_by=sort_by)

    print('Calculated stats for {} point clouds in {:.1f} seconds.\n====='.format(
        len(nusc.lidarseg), time.time() - start_time))

    # Place the class names and counts into a dataframe for use with seaborn later.
    df = pd.DataFrame(list(zip(class_names, counts)), columns=['Class', 'Count'])

    # Create an array with the colors to use.
    cmap = get_colormap()
    colors = ['#%02x%02x%02x' % tuple(cmap[row['Class']]) for index, row in df.iterrows()]  # Convert from RGB to hex.

    # Make the class names shorter so that they do not take up much space in the plot.
    df['Class'] = df['Class'].apply(lambda x: truncate_class_name(x))

    # Start a plot.
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.set_style(style="darkgrid")

    # Plot the histogram.
    chart = sns.barplot(x="Class", y="Count", data=df, label="Total", palette=colors, ci=None)
    assert len(df) == len(chart.get_xticks()), \
        'There are {} classes, but {} are shown on the x-axis'.format(len(df), len(chart.get_xticks()))

    # Format the x-axis.
    chart.set_xlabel(x_label, fontsize=font_size)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right',
                          fontweight='light', fontsize=font_size)

    # Shift the class names on the x-axis slightly to the right for aesthetics reasons.
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    # Format the y-axis.
    chart.set_ylabel(y_label, fontsize=font_size)
    chart.set_yticklabels(chart.get_yticks(), size=font_size)

    # Transform the y-axis to log scale.
    if y_log_scale:
        chart.set_yscale("log")

    # Display the y-axis using nice scientific notation.
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    chart.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "${}$".format(formatter._formatSciNotation('%1.10e' % x))))

    if chart_title:
        chart.set_title(chart_title, fontsize=font_size)

    if save_as_img_name:
        fig = chart.get_figure()
        plt.tight_layout()
        fig.savefig(save_as_img_name)

    if verbose:
        plt.show()


def get_lidarseg_stats(nusc: NuScenes, sort_by: str = 'count_desc') -> Tuple[List[str], List[int]]:
    """
    Get the number of points belonging to each class for the given nusc split.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes:
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :return: A list of class names and a list of the corresponding number of points for each class.
    """

    # Initialize an array of zeroes, one for each class name.
    lidarseg_counts = [0] * len(nusc.lidarseg_idx2name_mapping)

    for record_lidarseg in nusc.lidarseg:
        lidarseg_labels_filename = os.path.join(nusc.dataroot, record_lidarseg['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        indices = np.bincount(points_label)
        ii = np.nonzero(indices)[0]
        for class_idx, class_count in zip(ii, indices[ii]):
            lidarseg_counts[class_idx] += class_count

    lidarseg_counts_dict = dict()
    for i in range(len(lidarseg_counts)):
        lidarseg_counts_dict[nusc.lidarseg_idx2name_mapping[i]] = lidarseg_counts[i]

    if sort_by == 'count_desc':
        out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1], reverse=True)
    elif sort_by == 'count_asc':
        out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1])
    elif sort_by == 'name':
        out = sorted(lidarseg_counts_dict.items())
    elif sort_by == 'index':
        out = lidarseg_counts_dict.items()
    else:
        raise Exception('Error: Invalid sorting mode {}. '
                        'Only `count_desc`, `count_asc`, `name` or `index` are valid.'.format(sort_by))

    # Get frequency counts of each class in the lidarseg dataset.
    class_names = []
    counts = []
    for class_name, count in out:
        class_names.append(class_name)
        counts.append(count)

    return class_names, counts
