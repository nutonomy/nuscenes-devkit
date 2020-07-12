import os
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import seaborn as sns

from nuscenes import NuScenes
from nuscenes.utils.color_map import get_colormap


def string_formatter(string):
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

    return string_mapper[string]


def render_lidarseg_histogram(nusc: NuScenes, sort_by: str = 'count',
                              chart_title: str = None,
                              verbose: bool = True,
                              save_as_img_name: str = None) -> None:

    print('Calculating stats for nuScenes-lidarseg...')
    start_time = time.time()

    class_names, counts = get_lidarseg_stats(nusc, sort_by=sort_by)

    print('Calculated stats for {} point clouds in {:.1f} seconds.\n====='.format(
        len(nusc.lidarseg), time.time() - start_time))

    # Place the class names and counts into a dataframe for use with seaborn later.
    df = pd.DataFrame(list(zip(class_names, counts)), columns=['Class', 'Count'])
    df = df[~df['Class'].str.match('unwanted')]  # TODO: remove

    # Create an array with the colors to use.
    cmap = get_colormap()
    colors = ['#%02x%02x%02x' % tuple(cmap[row['Class']]) for index, row in df.iterrows()]  # Convert from RGB to hex.

    # Make the class names shorter so that they do not take up much space in the plot.
    df['Class'] = df['Class'].apply(lambda x: string_formatter(x))

    # Start a plot.
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.set(style="darkgrid")

    # Plot the histogram.
    chart = sns.barplot(x="Class", y="Count", data=df, label="Total", palette=colors, ci=None)

    chart.set_yscale("log")  # Transform the y-axis to log scale.

    assert len(df) == len(chart.get_xticks()), \
        'There are {} classes, but only {} are shown on the x-axis'.format(len(df), len(chart.get_xticks()))

    # ------------------------------ Format x and y labels --------------------------------------------------
    # chart.set_xlabel("Class",fontsize=15)
    chart.set_xlabel(None ,fontsize=20)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right',
                          fontweight='light', fontsize=20)

    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    chart.set_ylabel("Lidar points (logarithmic)", fontsize=20)
    chart.set_yticklabels(chart.get_yticks(), size=20)
    # chart.set_yticklabels(chart.get_yticklabels()) #, fontweight='light',fontsize=15)
    chart.tick_params(which="both")
    # ------------------------------ Format x and y labels --------------------------------------------------

    # chart.yaxis.set_major_formatter(ScalarFormatter())  # useOffset=False, useMathText=True))
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    # ax.yaxis.set_major_formatter(formatter)

    f = ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    chart.yaxis.set_major_formatter(FuncFormatter(g))

    if chart_title:
        chart.set_title('nuScenes-lidarseg', fontsize=20)

    if save_as_img_name:
        fig = chart.get_figure()
        plt.tight_layout()
        fig.savefig(save_as_img_name)

    if verbose:
        plt.show()


def get_lidarseg_stats(nusc: NuScenes, sort_by: str = 'count'):
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

    if sort_by == 'count':
        out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1], reverse=True)
    elif sort_by == 'name':
        out = sorted(lidarseg_counts_dict.items())
    else:
        out = lidarseg_counts_dict.items()

    # Print frequency counts of each class in the lidarseg dataset.
    class_names = []
    counts = []
    for class_name, count in out:
        class_names.append(class_name)
        counts.append(count)

    return class_names, counts


import sys
sys.path.insert(0, os.path.expanduser('~/Desktop/nuscenes-devkit/python-sdk'))

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
render_lidarseg_histogram(nusc, save_as_img_name=os.path.expanduser('~/Desktop/histo_test.png'))
