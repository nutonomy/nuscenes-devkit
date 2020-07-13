# nuScenes dev-kit.
# Code contributed by jean-lucas, 2020.

"""
Exports the nuScenes ego poses as "GPS" coordinates (lat/lon) for each scene into JSON or KML formatted files.
"""


import argparse
import json
import math
import os
from typing import List, Tuple, Dict

from tqdm import tqdm

from nuscenes.nuscenes import NuScenes


EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_poses(nusc: NuScenes, scene_token: str) -> List[dict]:
    """
    Return all ego poses for the current scene.
    :param nusc: The NuScenes instance to load the ego poses from.
    :param scene_token: The token of the scene.
    :return: A list of the ego pose dicts.
    """
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
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
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


def derive_latlon(location: str, poses: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.
    
    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).

    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), \
        f'Error: The given location: {location}, has no available reference.'
    
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


def export_kml(coordinates_per_location: Dict[str, Dict[str, List[Dict[str, float]]]], output_path: str) -> None:
    """
    Export the coordinates of a scene to .kml file.
    :param coordinates_per_location: A dict of lat/lon coordinate dicts for each scene.
    :param output_path: Path of the kml file to write to disk.
    """
    # Opening lines.
    result = \
        f'<?xml version="1.0" encoding="UTF-8"?>\n' \
        f'<kml xmlns="http://www.opengis.net/kml/2.2">\n' \
        f'  <Document>\n' \
        f'    <name>nuScenes ego poses</name>\n'

    # Export each scene as a separate placemark to be able to view them independently.
    for location, coordinates_per_scene in coordinates_per_location.items():
        result += \
            f'    <Folder>\n' \
            f'    <name>{location}</name>\n'

        for scene_name, coordinates in coordinates_per_scene.items():
            result += \
                f'        <Placemark>\n' \
                f'          <name>{scene_name}</name>\n' \
                f'          <LineString>\n' \
                f'            <tessellate>1</tessellate>\n' \
                f'            <coordinates>\n'

            for coordinate in coordinates:
                coordinates_str = '%.10f,%.10f,%d' % (coordinate['longitude'], coordinate['latitude'], 0)
                result += f'              {coordinates_str}\n'

            result += \
                f'            </coordinates>\n' \
                f'          </LineString>\n' \
                f'        </Placemark>\n'

        result += \
            f'    </Folder>\n'

    # Closing lines.
    result += \
        f'  </Document>\n' \
        f'</kml>'

    # Write to disk.
    with open(output_path, 'w') as f:
        f.write(result)


def main(dataroot: str, version: str, output_prefix: str, output_format: str = 'kml') -> None:
    """
    Extract the latlon coordinates for each available pose and write the results to a file.
    The file is organized by location and scene_name.
    :param dataroot: Path of the nuScenes dataset.
    :param version: NuScenes version.
    :param output_format: The output file format, kml or json.
    :param output_prefix: Where to save the output file (without the file extension).
    """
    # Init nuScenes.
    nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)

    coordinates_per_location = {}
    print(f'Extracting coordinates...')
    for scene in tqdm(nusc.scene):
        # Retrieve nuScenes poses.
        scene_name = scene['name']
        scene_token = scene['token']
        location = nusc.get('log', scene['log_token'])['location']  # Needed to extract the reference coordinate.
        poses = get_poses(nusc, scene_token)  # For each pose, we will extract the corresponding coordinate.

        # Compute and store coordinates.
        coordinates = derive_latlon(location, poses)
        if location not in coordinates_per_location:
            coordinates_per_location[location] = {}
        coordinates_per_location[location][scene_name] = coordinates

    # Create output directory if necessary.
    dest_dir = os.path.dirname(output_prefix)
    if dest_dir != '' and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Write to json.
    output_path = f'{output_prefix}_{version}.{output_format}'
    if output_format == 'json':
        with open(output_path, 'w') as fh:
            json.dump(coordinates_per_location, fh, sort_keys=True, indent=4)
    elif output_format == 'kml':
        # Write to kml.
        export_kml(coordinates_per_location, output_path)
    else:
        raise Exception('Error: Invalid output format: %s' % output_format)

    print(f"Saved the coordinates in {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ego pose coordinates from a scene to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    parser.add_argument('--output_prefix', type=str, default='latlon',
                        help='Output file path without file extension.')
    parser.add_argument('--output_format', type=str, default='kml', help='Output format (kml or json).')
    args = parser.parse_args()

    main(args.dataroot, args.version, args.output_prefix, args.output_format)
