# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes

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


def visualize_sample(nusc: NuScenes, sample_token: str, all_annotations: Dict, all_results: Dict, nsweeps: int=1,
                     conf_th: float=0.15, eval_range: float=40, verbose=True) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param all_annotations: Maps each sample token to its annotations.
    :param all_results: Maps each sample token to its results.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    """

    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = all_annotations[sample_token]
    boxes_est_global = all_results[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show plot.
    if verbose:
        print('Showing sample token %s' % sample_token)
    plt.title(sample_token)
    plt.show()


def filter_boxes(sample_boxes: List[Dict], pose_record: Dict, cs_record: Dict, eval_range: float) \
        -> Tuple[List[Dict], List[float]]:
    """
    Removes all boxes that are not within the valid eval_range of the LIDAR.
    :param sample_boxes: A list of sample_annotation OR sample_result entries.
    :param pose_record: An ego_pose entry stored as a dict.
    :param cs_record: A calibrated_sensor entry stored as a dict.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :return: The filtered sample_boxes and their distances to the sensor.
    """
    # Moved boxes to lidar coordinate frame
    sample_boxes_sensor = boxes_to_sensor(sample_boxes, pose_record, cs_record)

    # Filter boxes outside the relevant area.
    result = []
    ego_dists = []
    for box_sensor, box_global in zip(sample_boxes_sensor, sample_boxes):
        dist = np.sqrt(np.sum(box_sensor.center ** 2))
        if dist <= eval_range:
            result.append(box_global)  # Add the sample_box, not the box.
            ego_dists.append(dist)

    return result, ego_dists


def center_distance(sample_annotation: Dict, sample_result: Dict) -> float:
    """
    L2 distance between the box centers (xy only).
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(sample_result['translation'][:2]) - np.array(sample_annotation['translation'][:2]))


def velocity_l2(sample_annotation: Dict, sample_result: Dict) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: L2 distance.
    """
    if any(np.isnan(sample_result['velocity'][:2])):
        return np.inf
    else:
        return np.linalg.norm(np.array(sample_result['velocity'][:2]) - np.array(sample_annotation['velocity'][:2]))


def yaw_diff(sample_annotation: Dict, sample_result: Dict) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_annotation = quaternion_yaw(Quaternion(sample_annotation['rotation']))
    yaw_result = quaternion_yaw(Quaternion(sample_result['rotation']))

    # Compute smallest angle between two yaw values.
    angle_diff = np.maximum(yaw_annotation - yaw_result, yaw_result - yaw_annotation)
    if angle_diff > np.pi:
        angle_diff = angle_diff - np.pi  # Shift (pi, 2*pi] to (0, pi].
    return angle_diff


def attr_acc(sample_annotation: Dict, sample_result: Dict, attributes: List[str]) -> float:
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
    gt_attr_vec = np.array(sample_annotation['attribute_labels'])
    res_scores = np.array(sample_result['attribute_scores'])
    gt_class = sample_annotation['detection_name']
    if gt_class in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    elif gt_class in ['bicycle', 'motorcyle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif gt_class in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    else:
        # Classes without attributes: barrier, traffic_cone
        rel_attributes = []

    # Map labels to indices and compute accuracy; nan if no attributes are relevant.
    if len(rel_attributes) == 0:
        # If a class has no attributes, we return nan.
        acc = np.nan
    elif any(np.isnan(res_scores)):
        # Catch errors and abort early if any score is nan.
        raise Exception('Error: attribute_score is nan. Set to -1 to ignore!')
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


def scale_iou(sample_annotation: Dict, sample_result: Dict) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation['size'])
    sr_size = np.array(sample_result['size'])
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


def boxes_to_sensor(boxes: List[Dict], pose_record: Dict, cs_record: Dict):
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
        box = Box(box['translation'], box['size'], Quaternion(box['rotation']))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out
