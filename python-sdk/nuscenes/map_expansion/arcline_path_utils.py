# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import math
from typing import Dict, Any, Tuple, List

import numpy as np

# (x, y, yaw) in global frame
Pose = Tuple[float, float, float]

ArcLinePath = Dict[str, Any]


def principal_value(angle_in_radians: float) -> float:
    """
    Ensures the angle is within [-pi, pi).
    :param angle_in_radians: Angle in radians.
    :return: Scaled angle in radians.
    """

    interval_min = -math.pi
    two_pi = 2 * math.pi
    scaled_angle = (angle_in_radians - interval_min) % two_pi + interval_min
    return scaled_angle


def compute_segment_sign(arcline_path: ArcLinePath) -> Tuple[int, int, int]:
    """
    Compute the sign of an arcline path based on its shape.
    :param arcline_path: arcline path record.
    :return: Tuple of signs for all three parts of the path. 0 if straight, -1 if right,
        1 if left.
    """
    shape = arcline_path['shape']
    segment_sign = [0, 0, 0]

    if shape in ("LRL", "LSL", "LSR"):
        segment_sign[0] = 1
    else:
        segment_sign[0] = -1

    if shape == "RLR":
        segment_sign[1] = 1
    elif shape == "LRL":
        segment_sign[1] = -1
    else:
        segment_sign[1] = 0

    if shape in ("LRL", "LSL", "RSL"):
        segment_sign[2] = 1
    else:
        segment_sign[2] = -1

    return segment_sign[0], segment_sign[1], segment_sign[2]


def get_transformation_at_step(pose: Pose,
                               step: float) -> Pose:
    """
    Get the affine transformation at s meters along the path.
    :param pose: Pose represented as tuple (x, y, yaw).
    :param step: Length along the arcline path in range (0, length_of_arcline_path].
    :return: Transformation represented as pose tuple.
    """

    theta = pose[2] * step
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    if abs(pose[2]) < 1e-6:
        return pose[0] * step, pose[1] * step, theta
    else:
        new_x = (pose[1] * (ctheta - 1.0) + pose[0] * stheta) / pose[2]
        new_y = (pose[0] * (1.0 - ctheta) + pose[1] * stheta) / pose[2]
        return new_x, new_y, theta


def apply_affine_transformation(pose: Pose,
                                transformation: Pose) -> Pose:
    """
    Apply affine transformation to pose.
    :param pose: Starting pose.
    :param transformation: Affine transformation represented as a pose tuple.
    :return: Pose tuple - the result of applying the transformation to the starting pose.
    """

    new_x = math.cos(pose[2]) * transformation[0] - math.sin(pose[2]) * transformation[1] + pose[0]
    new_y = math.sin(pose[2]) * transformation[0] + math.cos(pose[2]) * transformation[1] + pose[1]
    new_yaw = principal_value(pose[2] + transformation[2])

    return new_x, new_y, new_yaw


def _get_lie_algebra(segment_sign: Tuple[int, int, int],
                     radius: float) -> List[Tuple[float, float, float]]:
    """
    Gets the Lie algebra for an arcline path.
    :param segment_sign: Tuple of signs for each segment in the arcline path.
    :param radius: Radius of curvature of the arcline path.
    :return: List of lie algebra poses.
    """

    return [(1.0, 0.0, segment_sign[0] / radius),
            (1.0, 0.0, segment_sign[1] / radius),
            (1.0, 0.0, segment_sign[2] / radius)]


def pose_at_length(arcline_path: ArcLinePath,
                   pos: float) -> Tuple[float, float, float]:
    """
    Retrieves pose at l meters along the arcline path.
    :param arcline_path: Arcline path object.
    :param pos: Get the pose this many meters along the path.
    :return: Pose tuple.
    """

    path_length = sum(arcline_path['segment_length'])

    assert 1e-6 <= pos

    pos = max(0.0, min(pos, path_length))

    result = arcline_path['start_pose']
    segment_sign = compute_segment_sign(arcline_path)

    break_points = _get_lie_algebra(segment_sign, arcline_path['radius'])

    for i in range(len(break_points)):

        length = arcline_path['segment_length'][i]

        if pos <= length:
            transformation = get_transformation_at_step(break_points[i], pos)
            result = apply_affine_transformation(result, transformation)
            break

        transformation = get_transformation_at_step(break_points[i], length)
        result = apply_affine_transformation(result, transformation)
        pos -= length

    return result


def discretize(arcline_path: ArcLinePath,
               resolution_meters: float) -> List[Pose]:
    """
    Discretize an arcline path.
    :param arcline_path: Arcline path record.
    :param resolution_meters: How finely to discretize the path.
    :return: List of pose tuples.
    """

    path_length = sum(arcline_path['segment_length'])
    radius = arcline_path['radius']

    n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))

    resolution_meters = path_length / (n_points - 1)

    discretization = []

    cumulative_length = [arcline_path['segment_length'][0],
                         arcline_path['segment_length'][0] + arcline_path['segment_length'][1],
                         path_length + resolution_meters]

    segment_sign = compute_segment_sign(arcline_path)

    poses = _get_lie_algebra(segment_sign, radius)

    temp_pose = arcline_path['start_pose']

    g_i = 0
    g_s = 0.0

    for step in range(n_points):

        step_along_path = step * resolution_meters

        if step_along_path > cumulative_length[g_i]:
            temp_pose = pose_at_length(arcline_path, step_along_path)
            g_s = step_along_path
            g_i += 1

        transformation = get_transformation_at_step(poses[g_i], step_along_path - g_s)
        new_pose = apply_affine_transformation(temp_pose, transformation)
        discretization.append(new_pose)

    return discretization


def discretize_lane(lane: List[ArcLinePath],
                    resolution_meters: float) -> List[Pose]:
    """
    Discretizes a lane and returns list of all the poses alone the lane.
    :param lane: Lanes are represented as a list of arcline paths.
    :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
        lanes are properly represented.
    :return: List of pose tuples along the lane.
    """

    pose_list = []
    for path in lane:
        poses = discretize(path, resolution_meters)
        for pose in poses:
            pose_list.append(pose)
    return pose_list


def length_of_lane(lane: List[ArcLinePath]) -> float:
    """
    Calculates the length of a lane in meters.
    :param lane: Lane.
    :return: Length of lane in meters.
    """

    # Meters
    return sum(sum(path['segment_length']) for path in lane)


def project_pose_to_lane(pose: Pose, lane: List[ArcLinePath], resolution_meters: float = 0.5) -> Tuple[Pose, float]:
    """
    Find the closest pose on a lane to a query pose and additionally return the
    distance along the lane for this pose. Note that this function does
    not take the heading of the query pose into account.
    :param pose: Query pose.
    :param lane: Will find the closest pose on this lane.
    :param resolution_meters: How finely to discretize the lane.
    :return: Tuple of the closest pose and the distance along the lane
    """

    discretized_lane = discretize_lane(lane, resolution_meters=resolution_meters)

    xy_points = np.array(discretized_lane)[:, :2]
    closest_pose_index = np.linalg.norm(xy_points - pose[:2], axis=1).argmin()

    closest_pose = discretized_lane[closest_pose_index]
    distance_along_lane = closest_pose_index * resolution_meters
    return closest_pose, distance_along_lane


def _find_index(distance_along_lane: float, lengths: List[float]) -> int:
    """
    Helper function for finding of path along lane corresponding to the distance_along_lane.
    :param distance_along_lane: Distance along the lane (in meters).
    :param lengths: Cumulative distance at each end point along the paths in the lane.
    :return: Index of path.
    """

    if len(lengths) == 1:
        return 0
    else:
        return min(index for index, length in enumerate(lengths) if distance_along_lane <= length)


def get_curvature_at_distance_along_lane(distance_along_lane: float, lane: List[ArcLinePath]) -> float:
    """
    Computes the unsigned curvature (1 / meters) at a distance along a lane.
    :param distance_along_lane: Distance along the lane to calculate the curvature at.
    :param lane: Lane to query.
    :return: Curvature, always non negative.
    """

    total_length_at_segments = np.cumsum([sum(path['segment_length']) for path in lane])
    segment_index = _find_index(distance_along_lane, total_length_at_segments)

    path = lane[segment_index]
    path_length = path['segment_length']

    if segment_index > 0:
        distance_along_path = distance_along_lane - total_length_at_segments[segment_index - 1]
    else:
        distance_along_path = distance_along_lane

    segment_index = _find_index(distance_along_path, np.cumsum(path_length))

    segment_shape = path['shape'][segment_index]

    # Straight lanes have no curvature
    if segment_shape == 'S':
        return 0
    else:
        return 1 / path['radius']
