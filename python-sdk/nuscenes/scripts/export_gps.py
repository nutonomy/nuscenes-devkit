# nuScenes dev-kit.

"""
Exports an GPS coordinates for each scene into a JSON formatted file.
"""


import argparse
import json
import math
import os
from typing import List, Tuple

from tqdm import tqdm

from nuscenes.nuscenes import NuScenes


EARTH_RADIUS_METERS = 6.371e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_poses(scene_token: str):
    pose_list = []
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    
    ego_pose = nusc.get('ego_pose', sd_rec['token'])
    pose_list.append(ego_pose)

    while sd_rec['next'] != '':
        sd_rec = nusc.get('sample_data', sd_rec['next'])
        ego_pose = nusc.get('ego_pose', sd_rec['token'])
        pose_list.append(ego_pose)

    return pose_list


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: the angle the target point makes with the reference coordinate in radians, clockwise from north.
    :param dist: the distance, in meters, from the reference point to the target point.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS
    
    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance) + 
        math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat)
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_gps(location: str, poses: List[dict]) -> Tuple[dict]:
    """
    For each pose value, extract its respective GPS coordinate and timestamp.
    
    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner
        2. The origin of the global poses is also in the south-western corner

    :param location: the name of the map the poses correspond to, ie: 'boston-seaport'.
    :prame poses: all pose dictionaries of a scene.
    """
    
    assert location in REFERENCE_COORDINATES.keys(), f'The given location: {location}, has no available reference.'
    
    coordinates = []
    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    for p in poses:
        ts = p['timestamp']
        x, y = p['translation'][:2]
        bearing = math.atan(x / y)
        distance = math.sqrt(x**2 + y**2)
        lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
        coordinates.append({'timestamp': ts, 'latitude': lat, 'longitude': lon})
    return coordinates


def main(args):
    """
    Extract the GPS coordinates for each available pose and write the results to a JSON file organized by scene name.
    """

    coordinates_per_scene = {}
    for scene in tqdm(nusc.scene):
        scene_name = scene['name']
        scene_token = scene['token']
        location = nusc.get('log', scene['log_token'])['location']  # Needed to extract the reference coordinate
        poses = get_poses(scene_token)  # for each pose, we will extract the corresponding coordinate
        
        print(f'Extracting GPS coordinates for {scene_name} in {location}.')
        coordinates = derive_gps(location, poses)
        coordinates_per_scene[scene_name] = coordinates


    dest_dir = os.path.join(args.dataroot, args.version)
    dest_fpath = os.path.join(dest_dir, args.filename)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    with open(dest_fpath, 'w') as fh:
        json.dump(coordinates_per_scene, fh, sort_keys=True, indent=4)

    print(f"Saved the GPS coordinates in {dest_fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export GPS coordinates from a scene to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='gps_coordinates.json', help='Output filename.')
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main(args)
