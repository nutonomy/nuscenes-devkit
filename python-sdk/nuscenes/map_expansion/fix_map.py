import argparse
import json
import os

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.utils import drop_disconnected_lanes


def fix_map(dataroot: str, map_name: str, out_suffix: str) -> None:
    """
    Fix a map file by dropping disconnected lanes.
    :param dataroot: Path of the nuScenes dataset.
    :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`,
    :param out_suffix: A suffix to the output file name, e.g. "fixed" results in "<map>_fixed.json".
    """
    # Load map.
    nusc_map = NuScenesMap(map_name=map_name, dataroot=dataroot)

    # Drop disconnected lanes.
    nusc_map = drop_disconnected_lanes(nusc_map)

    # Load a clean version of the map file.
    # Some fields have been modified and therefore we need to use the originals.
    in_path = os.path.join(dataroot, 'maps', map_name + '.json')
    with open(in_path, 'r') as f:
        nusc_dict = json.load(f)

    # Save to disk.
    for field in ['arcline_path_3', 'connectivity', 'lane', 'lane_connector']:
        nusc_dict[field] = nusc_map.json_obj[field]
    out_path = os.path.join(dataroot, 'maps', '%s_%s.json' % (map_name, out_suffix))
    with open(out_path, 'w') as f:
        json.dump(nusc_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fixes known issues in a map file.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--map_name', type=str, default='singapore-onenorth', help='Path to the map json file.')
    parser.add_argument('--out_suffix', type=str, default='fixed', help='Output file name suffix.')
    args = parser.parse_args()

    fix_map(dataroot=args.dataroot, map_name=args.map_name, out_suffix=args.out_suffix)