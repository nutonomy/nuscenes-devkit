from typing import List, Dict, Set

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes


def get_egoposes_on_drivable_ratio(nusc: NuScenes, nusc_map: NuScenesMap, scene_token: str) -> float:
    """
    Get the ratio of ego poses on the drivable area.
    :param nusc: A NuScenes instance.
    :param nusc_map: The NuScenesMap instance of a particular map location.
    :param scene_token: The token of the current scene.
    :return: The ratio of poses that fall on the driveable area.
    """

    # Go through each sample in the scene.
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
    poses_all = 0
    poses_valid = 0
    for sample_token in sample_tokens:

        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_record = nusc.get('sample', sample_token)
        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

        # Check if the ego pose is on the driveable area.
        ego_pose = pose_record['translation'][:2]
        record = nusc_map.record_on_point(ego_pose[0], ego_pose[1], 'drivable_area')
        if len(record) > 0:
            poses_valid += 1
        poses_all += 1
    ratio_valid = poses_valid / poses_all

    return ratio_valid


def get_disconnected_subtrees(connectivity: Dict[str, dict]) -> Set[str]:
    """
    Compute lanes or lane_connectors that are part of disconnected subtrees.
    :param connectivity: The connectivity of the current NuScenesMap.
    :return: The lane_tokens for lanes that are part of a disconnected subtree.
    """
    # Init.
    connected = set()
    pending = set()

    # Add first lane.
    all_keys = list(connectivity.keys())
    first_key = all_keys[0]
    all_keys = set(all_keys)
    pending.add(first_key)

    while len(pending) > 0:
        # Get next lane.
        lane_token = pending.pop()
        connected.add(lane_token)

        # Add lanes connected to this lane.
        if lane_token in connectivity:
            incoming = connectivity[lane_token]['incoming']
            outgoing = connectivity[lane_token]['outgoing']
            inout_lanes = set(incoming + outgoing)
            for other_lane_token in inout_lanes:
                if other_lane_token not in connected:
                    pending.add(other_lane_token)

    disconnected = all_keys - connected
    assert len(disconnected) < len(connected), 'Error: Bad initialization chosen!'
    return disconnected


def drop_disconnected_lanes(nusc_map: NuScenesMap) -> NuScenesMap:
    """
    Remove any disconnected lanes.
    Note: This function is currently not used and we do not recommend using it. Some lanes that we do not drive on are
    disconnected from the other lanes. Removing them would create a single connected graph. It also removes
    meaningful information, which is why we do not drop these.
    :param nusc_map: The NuScenesMap instance of a particular map location.
    :return: The cleaned NuScenesMap instance.
    """

    # Get disconnected lanes.
    disconnected = get_disconnected_lanes(nusc_map)

    # Remove lane.
    nusc_map.lane = [lane for lane in nusc_map.lane if lane['token'] not in disconnected]

    # Remove lane_connector.
    nusc_map.lane_connector = [lane for lane in nusc_map.lane_connector if lane['token'] not in disconnected]

    # Remove connectivity entries.
    for lane_token in disconnected:
        if lane_token in nusc_map.connectivity:
            del nusc_map.connectivity[lane_token]

    # Remove arcline_path_3.
    for lane_token in disconnected:
        if lane_token in nusc_map.arcline_path_3:
            del nusc_map.arcline_path_3[lane_token]

    # Remove connectivity references.
    empty_connectivity = []
    for lane_token, connectivity in nusc_map.connectivity.items():
        connectivity['incoming'] = [i for i in connectivity['incoming'] if i not in disconnected]
        connectivity['outgoing'] = [o for o in connectivity['outgoing'] if o not in disconnected]
        if len(connectivity['incoming']) + len(connectivity['outgoing']) == 0:
            empty_connectivity.append(lane_token)
    for lane_token in empty_connectivity:
        del nusc_map.connectivity[lane_token]

    # To fix the map class, we need to update some indices.
    nusc_map._make_token2ind()

    return nusc_map


def get_disconnected_lanes(nusc_map: NuScenesMap) -> List[str]:
    """
    Get a list of all disconnected lanes and lane_connectors.
    :param nusc_map: The NuScenesMap instance of a particular map location.
    :return: A list of lane or lane_connector tokens.
    """
    disconnected = set()
    for lane_token, connectivity in nusc_map.connectivity.items():
        # Lanes which are disconnected.
        inout_lanes = connectivity['incoming'] + connectivity['outgoing']
        if len(inout_lanes) == 0:
            disconnected.add(lane_token)
            continue

        # Lanes that only exist in connectivity (not currently an issue).
        for inout_lane_token in inout_lanes:
            if inout_lane_token not in nusc_map._token2ind['lane'] and \
                    inout_lane_token not in nusc_map._token2ind['lane_connector']:
                disconnected.add(inout_lane_token)

    # Lanes that are part of disconnected subtrees.
    subtrees = get_disconnected_subtrees(nusc_map.connectivity)
    disconnected = disconnected.union(subtrees)

    return sorted(list(disconnected))
