import json
from typing import List, Dict

import numpy as np
import tqdm
from nuscenes.eval.detection.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name, boxes_to_sensor
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


def load_gt(nusc, eval_split, cfg) -> EvalBoxes:
    """ Loads ground truth boxes from DB. """

    # Init.
    attribute_map = {a['name']: a['token'] for a in nusc.attribute}

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes(nusc)
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

            # Get attribute_labels.
            attribute_labels = np.zeros((len(cfg.attributes),), dtype=bool)
            for i, attribute in enumerate(cfg.attributes):
                if attribute_map[attribute] in sample_annotation['attribute_tokens']:
                    attribute_labels[i] = True

            sample_boxes.append(
                EvalBox(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=list(nusc.box_velocity(sample_annotation['token'])),
                    detection_name=detection_name,
                    detection_score=0,
                    attribute_scores=list(attribute_labels.tolist()),
                    attribute_labels=list(attribute_labels)
                )
            )
        all_annotations.add_boxes(sample_token, sample_boxes)

    return all_annotations


def add_center_dist(nusc, eval_boxes: EvalBoxes):
    """ Appends the center distance from ego vehicle to each box. """

    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        eval_boxes.boxes[sample_token] = append_ego_dist_tmp(eval_boxes.boxes[sample_token], pose_record, cs_record)

    return eval_boxes


def append_ego_dist_tmp(sample_boxes: List[EvalBox], pose_record: Dict, cs_record: Dict) -> List[EvalBox]:
    """
    Removes all boxes that are not within the valid eval_range of the LIDAR.
    :param sample_boxes: A list of sample_annotation OR sample_result entries.
    :param pose_record: An ego_pose entry stored as a dict.
    :param cs_record: A calibrated_sensor entry stored as a dict.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :return: The filtered sample_boxes and their distances to the sensor.
    """
    # TODO: remove this whole method. Once we move to the distance to ego vehicle frame, this becomes trivial.
    # Moved boxes to lidar coordinate frame
    sample_boxes_sensor = boxes_to_sensor(sample_boxes, pose_record, cs_record)

    # Filter boxes outside the relevant area.
    result = []
    for box_sensor, box_global in zip(sample_boxes_sensor, sample_boxes):
        box_global.ego_dist = np.sqrt(np.sum(box_sensor.center[:2] ** 2))
        result.append(box_global)  # Add the sample_box, not the box.

    return result


def filter_eval_boxes(nusc, eval_boxes: EvalBoxes, max_dist: float):
    """ Applies filtering to boxes. Distance, bike-racks and point per box. """

    # TODO: add the other filtering here
    for sample_token in eval_boxes.sample_tokens:
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if box.ego_dist < max_dist]

    return eval_boxes