# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

from typing import List, Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.data_classes import EvalBox

# Define constant
IGNORE = -1


def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection classes.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


def center_distance(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(sample_result.translation[:2]) - np.array(sample_annotation.translation[:2]))


def velocity_l2(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: L2 distance.
    """
    if any(np.isnan(sample_result.velocity[:2])):
        return np.inf
    else:
        return np.linalg.norm(np.array(sample_result.velocity[:2]) - np.array(sample_annotation.velocity[:2]))


def yaw_diff(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_annotation = quaternion_yaw(Quaternion(sample_annotation.rotation))
    yaw_result = quaternion_yaw(Quaternion(sample_result.rotation))

    # Compute smallest angle between two yaw values.
    angle_diff = abs(yaw_annotation - yaw_result)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff  # Shift (pi, 2*pi] to (0, pi].
    return angle_diff


def attr_acc(sample_annotation: EvalBox, sample_result: EvalBox, attributes: List[str]) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes, we assign an accuracy of nan, which is ignored later on.
    If any attribute_scores are set to ignore, we assign an accuracy of 0.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :param attributes: Names of attributes in the same order as attribute_scores below.
    :return: Attribute classification accuracy or nan if no GT class does not have any attributes.
    """
    # Specify the relevant attributes for the current GT class.
    gt_attr_vec = np.array(sample_annotation.attribute_labels)
    res_scores = np.array(sample_result.attribute_scores)
    gt_class = sample_annotation.detection_name
    if gt_class in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    elif gt_class in ['bicycle', 'motorcycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif gt_class in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    else:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []

    # Map labels to indices and compute accuracy; nan if no attributes are relevant.
    if len(rel_attributes) == 0:
        # If a class has no attributes we return nan, which is ignored later.
        acc = np.nan
    elif any(np.isnan(res_scores)):
        # Catch errors and abort early if any score is nan.
        raise Exception('Error: attribute_score is nan. Set to -1 to ignore!')
    elif not(any(gt_attr_vec)):
        # About 0.4% of the sample_annotations have no attributes, although they should.
        # We return nan, which is ignored later.
        acc = np.nan
    elif any(res_scores == IGNORE):
        # If attributes scores are set to ignore, we return an accuracy of 0.
        acc = 0
    else:
        # Otherwise compute accuracy.
        attr_inds = np.array([i for (i, a) in enumerate(attributes) if a in rel_attributes])
        ann_label = attr_inds[gt_attr_vec[attr_inds] == 1]
        res_label = attr_inds[np.argmax(res_scores[attr_inds])]
        acc = float(ann_label == res_label)

    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection: float = np.prod(min_wlh)
    union: float = volume_annotation + volume_result - intersection
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


dist_fcn_map = {
    'center_distance': center_distance
}
