from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import cv2
from nuscenes.utils.data_classes import Box
from pyquaternion.quaternion import Quaternion
from collections import OrderedDict
import os
from tqdm import tqdm
import json
import argparse


def get_color(category_name):
    """
    Provides the default colors based on the category names.
    :param category_name: Name of the category (must be in the relevant classes)
    """
    if category_name in ['vehicle.bicycle', 'vehicle.motorcycle']:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name:
        return 255, 158, 0  # Orange
    elif 'human.pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


def box_in_image(box, intrinsic: np.ndarray, imsize: tuple, vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def get_box(sample_annotation_token: str):
    """
    Instantiates a Box class from a sample annotation record.
    :param sample_annotation_token: Unique sample_annotation identifier.
    """
    record = nusc.get('sample_annotation', sample_annotation_token)
    return Box(record['translation'], record['size'], Quaternion(record['rotation']),
               name=record['category_name'], token=record['token'])


def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """
    Generate one 2d annotation record .
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data tolk
    :param filename:
    :return:
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    ignored_keys = ['rotation', 'sample_token', 'size', 'token', 'translation']

    for key, value in ann_rec.items():
        if key in ignored_keys:
            pass
        else:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(sample_data_token):
    """
    Get the 2d annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a keyframe.
    :return: List of 2d annotation record that belongs to the input `sample_data_token`
    """
    # Get the sample data, and the sample corresponding to that sample data
    sd_rec = nusc.get('sample_data', sample_data_token)
    if not (sd_rec['is_key_frame']):
        raise ValueError("Selected sample data token is not a keyframe")
    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with above a visibility threshold
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in args.visibilities)]

    # Load the image data
    data_path = os.path.join(nusc.dataroot, sd_rec['filename'])
    img = cv2.imread(data_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    repro_recs = []
    for ann_rec in ann_recs:

        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        if not box_in_image(box, camera_intrinsic, (1600, 900), 1):
            continue

        c = get_color(box.name)

        corners = view_points(box.corners(), camera_intrinsic, True)
        max_x = max(corners[0])
        min_x = min(corners[0])
        max_y = max(corners[1])
        min_y = min(corners[1])

        img = cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), c, 2)

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def main():
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if
                                 (s['sensor_modality'] == 'camera') and s['is_key_frame']]

    print("Generating 2d reprojections of the nuScenes dataset")
    reprojections = []
    for token in tqdm(sample_data_camera_tokens):
        reprojection_records = get_2d_boxes(token)
        reprojections.extend(reprojection_records)

    with open('test.json', 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)

    return reprojections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--dest_path', type=str, default='v1.0_2d_bbox.json')
    parser.add_argument('--visibilities', type=str, default=['2', '3', '4'])
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main()

