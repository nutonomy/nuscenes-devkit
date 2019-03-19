# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json

import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes.eval.detection.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes


def load_prediction(result_path: str, max_boxes_per_sample: int) -> EvalBoxes:
    """ Loads object predictions from file. """
    with open(result_path) as f:
        all_results = EvalBoxes.deserialize(json.load(f))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results


def load_gt(nusc, eval_split: str) -> EvalBoxes:
    """ Loads ground truth boxes from DB. """

    # Init.
    attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()
    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm.tqdm(sample_tokens):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            # Get label name in detection task and filter unused labels.
            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            # Get attribute_name.
            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise Exception('Error: GT annotations must not have more than one attribute!')

            sample_boxes.append(
                EvalBox(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    detection_name=detection_name,
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attribute_name,
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_lidar_pts']
                )
            )
        all_annotations.add_boxes(sample_token, sample_boxes)

    return all_annotations


def add_center_dist(nusc, eval_boxes: EvalBoxes):
    """ Adds the cylindrical (xy) center distance from ego vehicle to each box. """

    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            diff = np.array(pose_record['translation'][:2]) - np.array(box.translation[:2])
            box.ego_dist = np.sqrt(np.sum(diff ** 2))

    return eval_boxes


def filter_eval_boxes(nusc, eval_boxes: EvalBoxes, max_dist: dict):
    """ Applies filtering to boxes. Distance, bike-racks and points per box. """

    for sample_token in eval_boxes.sample_tokens:

        # Filter on distance first
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.detection_name]]

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]

        # Perform bike-rack filtering
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']

        filtered_boxes = []

        for rec in bikerack_recs:
            bikerack_box = Box(rec['translation'], rec['size'], Quaternion(rec['rotation']))
            for box in eval_boxes[sample_token]:
                if box.detection_name in ['bicycle', 'motorcycle'] and \
                        np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                    continue
                else:
                    filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes

    return eval_boxes
