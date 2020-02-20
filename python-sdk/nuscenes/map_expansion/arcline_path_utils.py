# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import math
from typing import Dict, Any, List, Tuple


Pose = Tuple[float, float, float]


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


def compute_segment_sign(arcline_path_3: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Compute the sign of an arcline_path_3 based on its shape.
    :param arcline_path_3: arcline_patch_3 record.
    :return: Tuple of signs for all three parts of the path. 0 if straight, -1 if right,
        1 if left.
    """
    shape = arcline_path_3['shape']
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


def get_transformation_at_s(pose: Pose,
                            s: float) -> Pose:
    """
    Get the affine transformation at s meters along the path.
    :param pose: Pose represented as tuple (x, y, yaw).
    :param s: Length along the arcline path in range (0, length_of_arcline_path].
    :return: Transformation represented as pose tuple.
    """

    theta = pose[2] * s
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    if abs(pose[2]) < 1e-6:
        return pose[0] * s, pose[1] * s, theta
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



def pose_at_length(arcline_path_3: Dict[str, Any],
                   l: float) -> Tuple[float, float, float]:
    """
    Retrieves pose at step meters along the arcline_path_3.
    :param arcline_path_3: Arcline_path_3 object.
    :param l: Get the pose this many meters along the path.
    :return: Pose tuple.
    """

    path_length = sum(arcline_path_3['segment_length'])

    assert -1e-6 <= l <= path_length

    l = max(0.0, min(l, path_length))

    result = arcline_path_3['start_pose']
    segment_sign = compute_segment_sign(arcline_path_3)

    break_points = _get_lie_algebra(segment_sign, arcline_path_3['radius'])

    for i in range(len(break_points)):

        length = arcline_path_3['segment_length'][i]

        if l <= length:
            transformation = get_transformation_at_s(break_points[i], l)
            result = apply_affine_transformation(result, transformation)
            break

        transformation = get_transformation_at_s(break_points[i], length)
        result = apply_affine_transformation(result, transformation)
        l -= length

    return result


def discretize(arcline_path_3: Dict[str, Any],
               resolution_meters: float) -> List[Tuple[float, float, float]]:
    """
    Discretize an arcline_path_3.
    :param arcline_path_3: Arcline_Path_3 object.
    :param resolution_meters: How finely to discretize the path.
    :return: List of pose tuples.
    """

    path_length = sum(arcline_path_3['segment_length'])
    radius = arcline_path_3['radius']

    n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))

    resolution_meters = path_length / (n_points - 1)

    discretization = []

    cumulative_length = [arcline_path_3['segment_length'][0],
                         arcline_path_3['segment_length'][0] + arcline_path_3['segment_length'][1],
                         path_length + resolution_meters]

    segment_sign = compute_segment_sign(arcline_path_3)

    poses = _get_lie_algebra(segment_sign, radius)

    temp_pose = arcline_path_3['start_pose']

    g_i = 0
    g_s = 0.0

    for step in range(n_points):

        step_along_path = step * resolution_meters

        if step_along_path > cumulative_length[g_i]:
            temp_pose = pose_at_length(arcline_path_3, step_along_path)
            g_s = step_along_path
            g_i += 1

        transformation = get_transformation_at_s(poses[g_i], step_along_path - g_s)
        new_pose = apply_affine_transformation(temp_pose, transformation)
        discretization.append(new_pose)

    return discretization


def discretize_lane(arcline_list: List[Dict[str, Any]],
                    resolution_meters: float) -> List[Tuple[float, float, float]]:
    """
    Discretizes a lane and returns list of all the poses alone the lane.
    :param arcline_list: Lanes are represented as a list of arcline_3_paths.
    :param resolution_meters: How finely to discretize the lane. Smaller values ensure turning
        lanes are properly represented.
    :return: List of pose tuples along the lane.
    """

    pose_list = []
    for arcline_path in arcline_list:
        poses = discretize(arcline_path, resolution_meters)
        for pose in poses:
            pose_list.append(pose)
    return pose_list
