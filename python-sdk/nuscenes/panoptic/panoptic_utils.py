"""
Panoptic nuScenes utils.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


STUFF_START_CLASS_ID = 24


def stuff_cat_ids(num_categories: int) -> List[int]:
    """
    :param num_categories: total number of classes.
    :return: List of stuff class indices.
    """
    return list(range(STUFF_START_CLASS_ID, num_categories, 1))


def generate_panoptic_colors(colormap: Dict[str, Iterable[int]],
                             name2idx: Dict[str, int],
                             max_instances: int = 32000) -> np.ndarray:
    """
    Create an array of RGB values from a colormap for stuff categories, and random colors for thing instances. Note
    that the RGB values are normalized between 0 and 1, not 0 and 255.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :param name2idx: A dictionary containing the mapping form class names to class index.
    :param max_instances: maximal number of instances.
    :return: An array of colors.
    """
    np.random.seed(0)
    colors_for_categories = colormap_to_colors(colormap=colormap, name2idx=name2idx)
    # randomly generate colors for stuff and thing instances
    colors = np.random.uniform(low=0.0, high=1.0, size=(max_instances, 3))
    # Use constant colors for stuff points, category ranges in [24, 31]
    for id_i in range(STUFF_START_CLASS_ID, len(colors_for_categories)):
        colors[id_i * 1000] = colors_for_categories[id_i]
    colors[0] = [c / 255.0 for c in get_colormap()['noise']]

    return colors


def paint_panop_points_label(panoptic_labels_filename: str,
                             filter_panoptic_labels: List[int],
                             name2idx: Dict[str, int],
                             colormap: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Paint each label in a pointcloud with the corresponding RGB value, and set the opacity of the labels to
    be shown to 1 (the opacity of the rest will be set to 0); e.g.:
        [30, 5, 12, 34, ...] ------> [[R30, G30, B30, 0], [R5, G5, B5, 1], [R34, G34, B34, 1], ...]
    :param panoptic_labels_filename: Path to the .bin file containing the labels.
    :param filter_panoptic_labels: The labels for which to set opacity to zero; this is to hide those points
                                   thereby preventing them from being displayed.
    :param name2idx: A dictionary containing the mapping from class names to class indices.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
    """
    # Load labels from .npz file.
    panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')  # [num_points]
    # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
    # get an array of RGB values where each color sits at the index in the array corresponding
    # to the class index.
    colors = generate_panoptic_colors(colormap, name2idx)  # Shape: [num_instances, 3]

    if filter_panoptic_labels is not None:
        # Ensure that filter_panoptic_labels is an iterable.
        assert isinstance(filter_panoptic_labels, (list, np.ndarray)), \
            'Error: filter_panoptic_labels should be a list of class indices, eg. [9], [10, 21].'

        # Check that class indices in filter_panoptic_labels are valid.
        assert all([0 <= x < len(name2idx) for x in filter_panoptic_labels]), \
            f'All class indices in filter_panoptic_labels should be between 0 and {len(name2idx) - 1}'

        # Filter to get only the colors of the desired classes; this is done by setting the
        # alpha channel of the classes to be viewed to 1, and the rest to 0.
        colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), 1)
        for id_i in np.unique(panoptic_labels):  # Shape: [num_class, 4]
            if id_i // 1000 not in filter_panoptic_labels:
                colors[id_i, -1] = 0.0

    coloring = colors[panoptic_labels]  # Shape: [num_points, 4]

    return coloring


def get_frame_panoptic_instances(panoptic_label: np.ndarray, frame_id: int = None) -> np.ndarray:
    """
    Get frequency of each label in a point cloud.
    :param panoptic_label: np.array((rows, cols), np.uint16), a numPy array which contains the panoptic labels of the
    point cloud; e.g. 1000 * cat_idx + instance_id. The instance_id starts from 1.
    :param frame_id: frame index.
    :return: np.array((num_instances, k), np.int), k = 3. Each true flag of frame_id will add extra 1 column.
    An full array contains one row (frame_index, category_id, instance_id, num_points) for each instance. frame_index
    column will be skipped if the flag is False.
    """
    inst_count = np.array(np.unique(panoptic_label, return_counts=True)).T
    cat_inst_count = np.concatenate([inst_count[:, 0:1] // 1000, inst_count[:, 0:1] % 1000, inst_count[:, 1:]],
                                    axis=1).astype(np.int32)

    if frame_id is not None:
        frame_id_col = np.full((cat_inst_count.shape[0], 1), fill_value=frame_id, dtype=np.int32)
        cat_inst_count = np.concatenate([frame_id_col, cat_inst_count], axis=1)

    return cat_inst_count


def get_panoptic_instances_stats(scene_inst_stats: Dict[str, np.ndarray],
                                 cat_idx2name: Dict[int, str],
                                 get_hist: bool = False) -> Dict[str, Any]:
    """
    Get panoptic instance stats on a database level.
    :param scene_inst_stats:  {scene_token : np.array((n, 4), np.int32)), each row is (frame_id, category_id,
        instance_id, num_points) for each instance within the same scene.
    :param cat_idx2name: {int: str}, category index to name mapping.
    :param get_hist: True to return per frame instance counts and per category counts (number of spanned frames
        per instance and num of points per instance). These could be used for more deep histogram analysis.
    :return: A dict of panoptic data stats.
        {
            'num_instances': int,
            'num_instance_states': int,
            'per_frame_panoptic_stats': {
                'per_frame_num_instances': (mean, std),
                'per_frame_num_instance_hist': np.array((num_frames,), np.int32)
            },
            'per_category_panoptic_stats': {
                'human.pedestrian.adult': {
                    'num_instances': int,
                    'num_frames_per_instance': (mean, std),
                    'num_points_per_instance': (mean, std),
                    'num_frames_per_instance_count': np.ndarray((num_instances,), np.int32)  # optional
                    'num_points_per_instance_count': np.ndarray((num_boxes,), np.int32)      # optional
                }
            },
        }
    """
    assert len(scene_inst_stats) > 0, "Empty input data !"
    ncols = scene_inst_stats[list(scene_inst_stats.keys())[0]].shape[1]
    # Create database mat, add scene_id column, each row: (scene_id, frame_id, category_id, instance_id, num_points).
    data = np.empty((0, ncols + 1), dtype=np.int32)
    for scene_id, scene_data in enumerate(scene_inst_stats.values()):
        scene_id_col = np.full((scene_data.shape[0], 1), fill_value=scene_id, dtype=np.int32)
        scene_data = np.concatenate([scene_id_col, scene_data], axis=1)
        data = np.concatenate([data, scene_data], axis=0)

    thing_row_mask = np.logical_and(data[:, 2] > 0, data[:, 2] < STUFF_START_CLASS_ID)  # thing instances only.
    data_thing = data[thing_row_mask]
    # 1. Per-frame instance stats.
    # Need to make unique frame index: 1000 * scene_id * frame_id, support max 1000 frames per scene. All instances
    # in the same frame will share the same unique frame index. We can count the occurrences for each frame.
    inst_num_each_frame = np.array(np.unique(1000 * data_thing[:, 0] + data_thing[:, 1], return_counts=True))[1]
    mean_inst_num_per_frame, std_inst_num_per_frame = np.mean(inst_num_each_frame), np.std(inst_num_each_frame)
    total_num_sample_annotations = np.sum(inst_num_each_frame)

    per_frame_panoptic_stats = {'per_frame_num_instances': (mean_inst_num_per_frame, std_inst_num_per_frame)}
    if get_hist:
        inst_num_per_frame_hist = np.array(np.unique(inst_num_each_frame, return_counts=True)).T
        per_frame_panoptic_stats.update({'per_frame_num_instances_hist': inst_num_per_frame_hist})

    # 2. Per-category instance stats.
    per_category_panoptic_stats = dict()
    unique_cat_ids = np.array(np.unique(data_thing[:, 2]))
    # Need to make unique instance ID across scene, inst_id = 1000 * scene_id + inst_id
    unique_inst_ids = 1000 * data_thing[:, 0] + data_thing[:, 3]
    for cat_id in unique_cat_ids:
        per_cat_inst_mask = data_thing[:, 2] == cat_id
        per_cat_unique_inst_ids = unique_inst_ids[per_cat_inst_mask]
        per_cat_inst_frame_count = np.array(np.unique(per_cat_unique_inst_ids, return_counts=True))[1]
        num_instances = len(per_cat_inst_frame_count)
        mean_num_frames, std_num_frames = np.mean(per_cat_inst_frame_count), np.std(per_cat_inst_frame_count)
        per_cat_inst_num_pts = data_thing[per_cat_inst_mask, 4]
        mean_num_pts, std_num_pts = np.mean(per_cat_inst_num_pts), np.std(per_cat_inst_num_pts)
        per_category_panoptic_stats[cat_idx2name[cat_id]] = {
            'num_instances': num_instances,
            'num_frames_per_instance': (mean_num_frames, std_num_frames),
            'num_points_per_instance': (mean_num_pts, std_num_pts),
        }
        if get_hist:
            per_category_panoptic_stats[cat_idx2name[cat_id]].update({
                'num_frames_per_instance_count': per_cat_inst_frame_count,
                'num_points_per_instance_count': per_cat_inst_num_pts,
            })

    total_num_instances = np.sum([v['num_instances'] for _, v in per_category_panoptic_stats.items()])
    # for completeness of all categories, fill with empty values for categories with zero instances.
    other_cats = [name for i, name in cat_idx2name.items() if 0 < i < STUFF_START_CLASS_ID and i not in unique_cat_ids]
    for cat_name in other_cats:
        per_category_panoptic_stats[cat_name] = {
            'num_instances': 0,
            'num_frames_per_instance': (0, 0),
            'num_points_per_instance': (0, 0),
            'num_frames_per_instance_count': np.array([], dtype=np.int32),
            'num_points_per_instance_count': np.array([], dtype=np.int32)
        }

    return dict(num_instances=total_num_instances,
                num_sample_annotations=total_num_sample_annotations,
                per_frame_panoptic_stats=per_frame_panoptic_stats,
                per_category_panoptic_stats=per_category_panoptic_stats)
