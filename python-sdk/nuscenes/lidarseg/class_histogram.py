import os
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np

from nuscenes import NuScenes
from nuscenes.panoptic.panoptic_utils import get_frame_panoptic_instances, get_panoptic_instances_stats
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


def truncate_class_name(class_name: str) -> str:
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


def render_histogram(nusc: NuScenes,
                     sort_by: str = 'count_desc',
                     verbose: bool = True,
                     font_size: int = 20,
                     save_as_img_name: str = None) -> None:
    """
    Render two histograms for the given nuScenes split. The top histogram depicts the number of scan-wise instances
    for each class, while the bottom histogram depicts the number of points for each class.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes to display in the plot (note that the x-axis, where the class names will be
        displayed on, is shared by the two histograms):
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param verbose: Whether to display the plot in a window after rendering.
    :param font_size: Size of the font to use for the plot.
    :param save_as_img_name: Path (including image name and extension) to save the plot as.
    """

    # Get the statistics for the given nuScenes split.
    lidarseg_num_points_per_class = get_lidarseg_num_points_per_class(nusc, sort_by=sort_by)
    panoptic_num_instances_per_class = get_panoptic_num_instances_per_class(nusc, sort_by=sort_by)

    # Align the two dictionaries by adding entries for the stuff classes to panoptic_num_instances_per_class; the
    # instance count for each of these stuff classes is 0.
    panoptic_num_instances_per_class_tmp = dict()
    for class_name in lidarseg_num_points_per_class.keys():
        num_instances_for_class = panoptic_num_instances_per_class.get(class_name, 0)
        panoptic_num_instances_per_class_tmp[class_name] = num_instances_for_class
    panoptic_num_instances_per_class = panoptic_num_instances_per_class_tmp

    # Define some settings for each histogram.
    histograms_config = dict({
        'panoptic': {
            'y_values': list(panoptic_num_instances_per_class.values()),
            'y_label': 'No. of instances',
            'y_scale': 'log'
        },
        'lidarseg': {
            'y_values': list(lidarseg_num_points_per_class.values()),
            'y_label': 'No. of lidar points',
            'y_scale': 'log'
        }
    })

    # Ensure the same set of class names are used for all histograms.
    assert lidarseg_num_points_per_class.keys() == panoptic_num_instances_per_class.keys(), \
        'Error: There are {} classes for lidarseg, but {} classes for panoptic.'.format(
            len(lidarseg_num_points_per_class.keys()), len(panoptic_num_instances_per_class.keys()))
    class_names = list(lidarseg_num_points_per_class.keys())

    # Create an array with the colors to use.
    cmap = get_colormap()
    colors = ['#%02x%02x%02x' % tuple(cmap[cn]) for cn in class_names]  # Convert from RGB to hex.

    # Make the class names shorter so that they do not take up much space in the plot.
    class_names = [truncate_class_name(cn) for cn in class_names]

    # Start a plot.
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    for ax in axes:
        ax.margins(x=0.005)  # Add some padding to the left and right limits of the x-axis for aesthetics.
        ax.set_axisbelow(True)  # Ensure that axis ticks and gridlines will be below all other ploy elements.
        ax.yaxis.grid(color='white', linewidth=2)  # Show horizontal gridlines.
        ax.set_facecolor('#eaeaf2')  # Set background of plot.
        ax.spines['top'].set_visible(False)  # Remove top border of plot.
        ax.spines['right'].set_visible(False)  # Remove right border of plot.
        ax.spines['bottom'].set_visible(False)  # Remove bottom border of plot.
        ax.spines['left'].set_visible(False)  # Remove left border of plot.

    # Plot the histograms.
    for i, (histogram, config) in enumerate(histograms_config.items()):
        axes[i].bar(class_names, config['y_values'], color=colors)
        assert len(class_names) == len(axes[i].get_xticks()), \
            'There are {} classes, but {} are shown on the x-axis'.format(len(class_names), len(axes[i].get_xticks()))

        # Format the x-axis.
        axes[i].set_xticklabels(class_names, rotation=45, horizontalalignment='right',
                                fontweight='light', fontsize=font_size)

        # Shift the class names on the x-axis slightly to the right for aesthetics.
        trans = mtrans.Affine2D().translate(10, 0)
        for t in axes[i].get_xticklabels():
            t.set_transform(t.get_transform() + trans)

        # Format the y-axis.
        axes[i].set_ylabel(config['y_label'], fontsize=font_size)
        axes[i].set_yticklabels(config['y_values'], size=font_size)
        axes[i].set_yscale(config['y_scale'])

        if config['y_scale'] == 'linear':
            axes[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))

    if save_as_img_name:
        plt.tight_layout()
        fig.savefig(save_as_img_name)

    if verbose:
        plt.show()


def get_lidarseg_num_points_per_class(nusc: NuScenes, sort_by: str = 'count_desc') -> Dict[str, int]:
    """
    Get the number of points belonging to each class for the given nuScenes split.
    :param nusc: A NuScenes object.
    :param sort_by: How to sort the classes:
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :return: A dictionary whose keys are the class names and values are the corresponding number of points for each
        class.
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

    num_points_per_class = dict()
    for i in range(len(lidarseg_counts)):
        num_points_per_class[nusc.lidarseg_idx2name_mapping[i]] = lidarseg_counts[i]

    if sort_by == 'count_desc':
        num_points_per_class = dict(sorted(num_points_per_class.items(), key=lambda item: item[1], reverse=True))
    elif sort_by == 'count_asc':
        num_points_per_class = dict(sorted(num_points_per_class.items(), key=lambda item: item[1]))
    elif sort_by == 'name':
        num_points_per_class = dict(sorted(num_points_per_class.items()))
    elif sort_by == 'index':
        num_points_per_class = dict(num_points_per_class.items())
    else:
        raise Exception('Error: Invalid sorting mode {}. '
                        'Only `count_desc`, `count_asc`, `name` or `index` are valid.'.format(sort_by))

    return num_points_per_class


def get_panoptic_num_instances_per_class(nusc: NuScenes, sort_by: str = 'count_desc') -> Dict[str, int]:
    """
    Get the number of scan-wise instances belonging to each class for the given nuScenes split.
    :param nusc: A NuScenes object.
    :param sort_by: How to sort the classes:
        - count_desc: Sort the classes by the number of instances belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of instances belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :return: A dictionary whose keys are the class names and values are the corresponding number of scan-wise instances
        for each class.
    """
    sequence_wise_instances_per_class = dict()
    for instance in nusc.instance:
        instance_class = nusc.get('category', instance['category_token'])['name']
        if instance_class not in sequence_wise_instances_per_class.keys():
            sequence_wise_instances_per_class[instance_class] = 0
        sequence_wise_instances_per_class[instance_class] += instance['nbr_annotations']

    if sort_by == 'count_desc':
        sequence_wise_instances_per_class = dict(
            sorted(sequence_wise_instances_per_class.items(), key=lambda item: item[1], reverse=True))
    elif sort_by == 'count_asc':
        sequence_wise_instances_per_class = dict(
            sorted(sequence_wise_instances_per_class.items(), key=lambda item: item[1]))
    elif sort_by == 'name':
        sequence_wise_instances_per_class = dict(sorted(sequence_wise_instances_per_class.items()))
    elif sort_by == 'index':
        sequence_wise_instances_per_class = dict(sequence_wise_instances_per_class.items())
    else:
        raise Exception('Error: Invalid sorting mode {}. '
                        'Only `count_desc`, `count_asc`, `name` or `index` are valid.'.format(sort_by))

    return sequence_wise_instances_per_class
