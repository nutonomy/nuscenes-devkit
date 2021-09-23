"""
Script to generate Panoptic nuScenes ground truth data from nuScenes database and nuScene-lidarseg.
Code written by Motional and the Robot Learning Lab, University of Freiburg.

Example usage:
python python-sdk/nuscenes/panoptic/generate_panoptic_labels.py --version v1.0-mini --out_dir /data/sets/nuscenes \
--verbose True
"""
import argparse
import json
import os
from shutil import copyfile

import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.panoptic.panoptic_utils import STUFF_START_CLASS_ID
from nuscenes.utils.data_classes import LidarSegPointCloud
from nuscenes.utils.geometry_utils import points_in_box


def generate_panoptic_labels(nusc: NuScenes, out_dir: str, verbose: bool = False) -> None:
    """
    Generate Panoptic nuScenes ground truth labels.
    :param nusc: NuScenes instance.
    :param out_dir: output directory.
    :param verbose: True to print verbose.
    """
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    cat_name_to_idx = nusc.lidarseg_name2idx_mapping
    num_samples = len(nusc.sample)
    if verbose:
        print(f'There are {num_samples} samples.')

    # Make output directory.
    panoptic_subdir = os.path.join('panoptic', nusc.version)
    panoptic_dir = os.path.join(out_dir, panoptic_subdir)
    os.makedirs(panoptic_dir, exist_ok=True)

    panoptic_json = []
    inst_tok2id = {}  # instance token to ID mapping
    for sample_idx in tqdm(range(num_samples), disable=not verbose):
        curr_sample = nusc.sample[sample_idx]
        scene_token = curr_sample['scene_token']
        if scene_token not in inst_tok2id:
            inst_tok2id[scene_token] = {}
        lidar_token = curr_sample['data']['LIDAR_TOP']
        point_path = os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'])
        label_path = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
        lidar_seg = LidarSegPointCloud(point_path, label_path)
        # panoptic labels will be set as 1000 * category_index + instance_id, instance_id will be [1, 2, 3, ...] within
        # each thing category, these points fall within more than 1 bounding boxes will have instance_id = 0.
        panop_labels = lidar_seg.labels.astype(np.int32) * 1000
        overlap_box_count = np.zeros(lidar_seg.labels.shape, dtype=np.uint16)

        for ann_token in curr_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            if not inst_tok2id[scene_token]:
                inst_tok2id[scene_token][instance_token] = 1
            elif instance_token not in inst_tok2id[scene_token]:
                inst_tok2id[scene_token][instance_token] = len(inst_tok2id[scene_token]) + 1
            instance_id = inst_tok2id[scene_token][instance_token]
            _, boxes, _ = nusc.get_sample_data(lidar_token, selected_anntokens=[ann_token])
            indices = np.where(points_in_box(boxes[0], lidar_seg.points[:, :3].T))[0]

            for index in range(len(indices)):
                if lidar_seg.labels[indices[index]] == cat_name_to_idx[ann['category_name']]:
                    panop_labels[indices[index]] += instance_id
                    overlap_box_count[indices[index]] += 1

        panop_labels[overlap_box_count > 1] = 0  # Set pixels overlapped by > 1 boxes to 0.

        # Thing pixels that are not inside any box have instance id == 0, reset them to 0.
        # For these pixels that have thing semantic classes, but do not fall inside any annotation box, set their
        # panoptic label to 0.
        semantic_labels = panop_labels // 1000
        thing_mask = np.logical_and(semantic_labels > 0, semantic_labels < STUFF_START_CLASS_ID)
        pixels_wo_box = np.logical_and(thing_mask, panop_labels % 1000 == 0)
        panop_labels[pixels_wo_box] = 0

        panoptic_file = nusc.get('lidarseg', lidar_token)['filename'].split('/')[-1]
        panoptic_file = panoptic_file.replace('lidarseg', 'panoptic')
        panoptic_file = panoptic_file.replace('.bin', '.npz')
        panoptic_json.append({
            "token": lidar_token,
            "sample_data_token": lidar_token,
            "filename": os.path.join(panoptic_subdir, panoptic_file),
        })
        np.savez_compressed(os.path.join(panoptic_dir, panoptic_file), data=panop_labels.astype(np.uint16))

    # Write the panoptic table.
    meta_dir = os.path.join(out_dir, nusc.version)
    os.makedirs(meta_dir, exist_ok=True)
    json_file = os.path.join(meta_dir, 'panoptic.json')
    with open(json_file, 'w') as f:
        json.dump(panoptic_json, f, indent=2)

    # Copy category.json from lidarseg.
    category_dst_path = os.path.join(meta_dir, 'category.json')
    if not os.path.isfile(category_dst_path):
        category_src_path = os.path.join(nusc.dataroot, nusc.version, 'category.json')
        copyfile(category_src_path, category_dst_path)


def main():
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    parser.add_argument('--out_dir', type=str, default=None, help='Folder to write the panoptic labels to.')
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else f'Panoptic-nuScenes-{args.version}'
    print(f'Start panoptic ground truths generation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    generate_panoptic_labels(nusc=nusc, out_dir=out_dir, verbose=args.verbose)
    print(f'Panoptic ground truths saved at {args.out_dir}. \nFinished panoptic ground truth generation.')


if __name__ == '__main__':
    main()
