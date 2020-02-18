from typing import Dict, Any, List, Tuple
import math

def principal_value(angle_in_radians: float) -> float:
    """
    Ensures the angle is within -pi, pi.
    :param angle_in_radians: Angle in radians.
    :return: Scaled angle in radians.
    """

    interval_min = -math.pi
    two_pi = 2 * math.pi
    scaled_angle = (angle_in_radians - interval_min) % two_pi + interval_min
    return scaled_angle

def compute_segment_sign(arcline_path_3: Dict[str, Any]) -> Tuple[float, float, float]:
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

    return segment_sign

def group_exponential(pose: Tuple[float, float, float],
                      t: float) -> Tuple[float, float, float]:
    """
    Computes group exponential for a pose.
    :param pose: Pose represented as tuple (x, y, yaw).
    :param t: Exponent to raise the pose to.
    """

    theta = pose[2] * t
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    if abs(pose[2]) < 1e-6:
        return [pose[0] * t, pose[1] * t, theta]
    else:
        new_x = (pose[1] * (ctheta - 1.0) + pose[0] * stheta) / pose[2]
        new_y = (pose[0] * (1.0 - ctheta) + pose[1] * stheta) / pose[2]
        return [new_x, new_y, theta]

def compose_right(left: Tuple[float, float, float],
                  pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Composes two poses.
    :param left: Tuple representing a pose.
    :param pose: Tuple representing a pose.
    :return: Pose tuple.
    """

    new_x = math.cos(left[2]) * pose[0] - math.sin(left[2]) * pose[1] + left[0]
    new_y = math.sin(left[2]) * pose[0] + math.cos(left[2]) * pose[1] + left[1]
    new_yaw = principal_value(left[2] + pose[2])

    return [new_x, new_y, new_yaw]


def get_breakpoint_poses(segment_sign: Tuple[float, float, float],
                        radius: float) -> List[Tuple[float, float, float]]:

    return [[1.0, 0.0, segment_sign[0] / radius],
            [1.0, 0.0, segment_sign[1] / radius],
            [1.0, 0.0, segment_sign[2] / radius]]

def pose_at_step(arcline_path_3: Dict[str, Any],
                 step: float) -> Tuple[float, float, float]:
    """
    Retrieves pose at step meters along the arcline_path_3.
    :param arcline_path_3: Arcline_path_3 object.
    :param step: Get the pose this many meters along the path.
    :return: Pose tuple.
    """

    path_length = sum(arcline_path_3['segment_length'])

    assert -1e-6 <= step <= path_length

    step = max(0, min(step, path_length))

    result = arcline_path_3['start_pose']
    segment_sign = compute_segment_sign(arcline_path_3)

    break_points = get_breakpoint_poses(segment_sign, arcline_path_3['radius'])

    for i in range(len(break_points)):

        length = arcline_path_3['segment_length'][i]

        if step <= length:
            temp = group_exponential(break_points[i], step)
            result = compose_right(result, temp)
            break

        temp = group_exponential(break_points[i], length)
        result = compose_right(result, temp)
        step -= length

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

    cummulative_length = [arcline_path_3['segment_length'][0],
                          arcline_path_3['segment_length'][0] + arcline_path_3['segment_length'][1],
                          path_length + resolution_meters]

    segment_sign = compute_segment_sign(arcline_path_3)

    poses = get_breakpoint_poses(segment_sign, radius)

    temp_pose = arcline_path_3['start_pose']

    g_i = 0
    g_s = 0.0
    for step in range(n_points):

        step_along_path = step * resolution_meters

        if (step_along_path > cummulative_length[g_i]):
            temp_pose = pose_at_step(arcline_path_3, step_along_path)
            g_s = step_along_path
            g_i += 1

        frame_increment = group_exponential(poses[g_i], step_along_path - g_s)
        new_pose = compose_right(temp_pose, frame_increment)
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










