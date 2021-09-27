"""
Script to generate Panoptic nuScenes predictions from nuScene-lidarseg predictions and nuScenes-tracking or
nuScenes-detection predictions.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""
import argparse
import os
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from nuscenes.eval.common.loaders import load_prediction, add_center_dist
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.lidarseg.utils import get_samples_in_eval_set
from nuscenes.eval.panoptic.utils import PanopticClassMapper
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarSegPointCloud
from nuscenes.utils.geometry_utils import points_in_box


OVERLAP_THRESHOLD = 0.5  # Amount by which an instance can overlap with other instances, before it is discarded.
STUFF_START_COARSE_CLASS_ID = 11


def generate_panoptic_labels(nusc: NuScenes,
                             lidarseg_preds_folder: str,
                             preds_json: str,
                             eval_set: str,
                             task: str = 'segmentation',
                             out_dir: str = None,
                             verbose: bool = False):
    """
    Generate NuScenes lidar panoptic predictions.
    :param nusc: A instance of NuScenes.
    :param lidarseg_preds_folder: Path to the directory where the lidarseg predictions are stored.
    :param preds_json: Path of the json where the tracking / detection predictions are stored.
    :param eval_set: Which dataset split to evaluate on, train, val or test.
    :param task: The task to create the panoptic predictions for (either tracking or segmentation).
    :param out_dir: Path to save any output to.
    :param verbose: Whether to print any output.
    """
    assert task in ['tracking', 'segmentation'], \
        'Error: Task can only be either `tracking` or `segmentation, not {}'.format(task)

    sample_tokens = get_samples_in_eval_set(nusc, eval_set)
    num_samples = len(sample_tokens)
    mapper = PanopticClassMapper(nusc)
    coarse2idx = mapper.get_coarse2idx()
    if verbose:
        print(f'There are {num_samples} samples.')

    panoptic_subdir = os.path.join('panoptic', eval_set)
    panoptic_dir = os.path.join(out_dir, panoptic_subdir)
    os.makedirs(panoptic_dir, exist_ok=True)

    box_type = TrackingBox if task == 'tracking' else DetectionBox

    # Load the predictions.
    pred_boxes_all, meta = load_prediction(preds_json, 1000, box_type, verbose=verbose)
    pred_boxes_all = add_center_dist(nusc, pred_boxes_all)

    inst_tok2id = {}  # Only used if task == 'tracking'.
    for sample_token in tqdm(sample_tokens, disable=not verbose):
        sample = nusc.get('sample', sample_token)

        scene_token = sample['scene_token']  # Only used if task == 'tracking'.
        if task == 'tracking' and scene_token not in inst_tok2id:
            inst_tok2id[scene_token] = {}

        # Get the sample data token of the point cloud.
        sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Load the predictions for the point cloud.
        lidar_path = os.path.join(nusc.dataroot, sd_record['filename'])
        lidarseg_pred_filename = os.path.join(lidarseg_preds_folder, 'lidarseg', nusc.version.split('-')[-1],
                                              sd_record['token'] + '_lidarseg.bin')
        lidar_seg = LidarSegPointCloud(lidar_path, lidarseg_pred_filename)

        panop_labels = np.zeros(lidar_seg.labels.shape, dtype=np.uint16)
        overlaps = np.zeros(lidar_seg.labels.shape, dtype=np.uint8)

        pred_boxes = pred_boxes_all[sample_token]

        # Tracking IDs will be None if pred_boxes is of DetectionBox type.
        sorted_pred_boxes, pred_cls, tracking_ids = sort_confidence(pred_boxes)
        sorted_pred_boxes = boxes_to_sensor(sorted_pred_boxes, pose_record, cs_record)

        # Go through each box (a.k.a instance) and obtain the panoptic label for each.
        for instance_id, (pred_box, cl, tracking_id) in enumerate(zip(sorted_pred_boxes, pred_cls, tracking_ids)):
            cl_id = coarse2idx[cl]

            if task == 'tracking':
                if not inst_tok2id[scene_token]:
                    inst_tok2id[scene_token][tracking_id] = 1
                elif tracking_id not in inst_tok2id[scene_token]:
                    inst_tok2id[scene_token][tracking_id] = len(inst_tok2id[scene_token]) + 1

            msk = np.zeros(lidar_seg.labels.shape, dtype=np.uint8)
            indices = np.where(points_in_box(pred_box, lidar_seg.points[:, :3].T))[0]

            msk[indices] = 1
            msk[np.logical_and(lidar_seg.labels != cl_id, msk == 1)] = 0
            intersection = np.logical_and(overlaps, msk)

            # If the current instance overlaps with previous instances by a certain threshold, then ignore it (note
            # that the instances are processed in order of decreasing confidence).
            if np.sum(intersection) / (np.float32(np.sum(msk)) + 1e-32) > OVERLAP_THRESHOLD:
                continue
            # Add non-overlapping part to output.
            msk = msk - intersection
            if task == 'tracking':
                panop_labels[msk != 0] = (cl_id * 1000 + inst_tok2id[scene_token][tracking_id])
            else:
                panop_labels[msk != 0] = (cl_id * 1000 + instance_id + 1)
            overlaps += msk

        stuff_msk = np.logical_and(panop_labels == 0, lidar_seg.labels >= STUFF_START_COARSE_CLASS_ID)
        panop_labels[stuff_msk] = lidar_seg.labels[stuff_msk] * 1000
        panoptic_file = sd_record['token'] + '_panoptic.npz'
        np.savez_compressed(os.path.join(panoptic_dir, panoptic_file), data=panop_labels.astype(np.uint16))


def sort_confidence(boxes: List[Union[DetectionBox, TrackingBox]]) \
        -> Tuple[List[Union[DetectionBox, TrackingBox]], List[str], List[Union[str, None]]]:
    """
    Sort a list of boxes by confidence.
    :param boxes: A list of boxes.
    :return: A list of boxes sorted by confidence and a list of classes and a list of tracking IDs (if available)
        corresponding to each box.
    """
    scores = [box.tracking_score if isinstance(box, TrackingBox) else box.detection_score
              for box in boxes]
    inds = np.argsort(scores)[::-1]

    sorted_bboxes = [boxes[ind] for ind in inds]

    sorted_cls = [box.tracking_name if isinstance(box, TrackingBox) else box.detection_name
                  for box in sorted_bboxes]

    tracking_ids = [box.tracking_id if isinstance(box, TrackingBox) else None
                    for box in sorted_bboxes]

    return sorted_bboxes, sorted_cls, tracking_ids


def main():
    """
    The main method for this script.
    """
    parser = argparse.ArgumentParser(
        description='Generate panoptic from nuScenes-LiDAR segmentation and tracking results.')
    parser.add_argument('--seg_path', type=str, help='The path to the segmentation results folder.')
    parser.add_argument('--track_path', type=str, help='The path to the track json file.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--task', type=str, default='segmentation',
                        help='The task to create the panoptic predictions for (either tracking or segmentation).')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print to stdout.')
    parser.add_argument('--out_dir', type=str, default=None, help='Folder to write the panoptic labels to.')
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else f'nuScenes-panoptictrack-merged-prediction-{args.version}'

    print(f'Start Generation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)

    generate_panoptic_labels(nusc=nusc,
                             lidarseg_preds_folder=args.seg_path,
                             preds_json=args.track_path,
                             eval_set=args.eval_set,
                             task=args.task,
                             out_dir=out_dir,
                             verbose=args.verbose)
    print(f'Generated results saved at {out_dir}. \nFinished.')


if __name__ == '__main__':
    main()
