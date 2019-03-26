# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# Licensed under the Creative Commons [see license.txt]

"""
Export 2D annotations (xmin, ymin,xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import box_in_image

import numpy as np
import json
import argparse
import os

from typing import List
from pyquaternion.quaternion import Quaternion
from collections import OrderedDict
from tqdm import tqdm


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
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


def get_2d_boxes(sample_data_token: str) -> List[OrderedDict]:
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

    repro_recs = []
    for ann_rec in ann_recs:

        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = nusc.get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        if not box_in_image(box, camera_intrinsic, (1600, 900), 1):
            continue

        corners = view_points(box.corners(), camera_intrinsic, True)
        max_x = max(corners[0])
        min_x = min(corners[0])
        max_y = max(corners[1])
        min_y = min(corners[1])

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

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    with open('{}.json'.format(os.path.join(args.dest_path, args.version)), 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--dest_path', type=str, default='./2D_annotations/')
    parser.add_argument('--visibilities', type=str, default=['2', '3', '4'])
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main()
