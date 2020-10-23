import os
import unittest
from collections import defaultdict
from typing import List, Dict, Set

import matplotlib.pyplot as plt
import tqdm

from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.nuscenes import NuScenes


class TestAllMaps(unittest.TestCase):
    version = 'v1.0-trainval'
    use_new_map = False
    drop_lanes = False
    render = False

    def setUp(self):
        """ Initialize the map for each location. """

        self.nusc_maps = dict()
        for map_name in locations:
            # Load map.
            if self.use_new_map and map_name == 'singapore-onenorth':
                nusc_map = NuScenesMap(map_name=map_name, dataroot=os.path.expanduser('~'))
                nusc_map.connectivity = dict()
            else:
                nusc_map = NuScenesMap(map_name=map_name, dataroot=os.environ['NUSCENES'])

            # Postprocess lanes until they are fixed.
            if self.drop_lanes:
                nusc_map = self.drop_disconnected_lanes(nusc_map)

            # Render for debugging.
            if self.render:
                nusc_map.render_layers(['lane'], figsize=1)
                plt.show()

            self.nusc_maps[map_name] = nusc_map

    def test_layer_stats(self):
        """ Test if each layer has the right number of instances. This is useful to compare between map versions. """
        layer_counts = defaultdict(lambda: [])
        ref_counts = {
            'singapore-onenorth': [1, 783, 645, 936, 120, 838, 451, 39, 152, 357, 127],
            'singapore-hollandvillage': [426, 167, 387, 602, 28, 498, 300, 0, 107, 220, 119],
            'singapore-queenstown': [219, 260, 676, 910, 75, 457, 437, 40, 172, 257, 81],
            'boston-seaport': [2, 928, 969, 1215, 340, 301, 775, 275, 377, 671, 307]
        }

        for map_name in locations:
            nusc_map = self.nusc_maps[map_name]
            for layer_name in nusc_map.non_geometric_layers:
                layer_objs = nusc_map.json_obj[layer_name]
                layer_counts[map_name].append(len(layer_objs))

            assert ref_counts[map_name] == layer_counts[map_name], \
                'Error: Map %s has a different number of layers: \n%s vs. \n%s' % \
                (map_name, ref_counts[map_name], layer_counts[map_name])

    def test_disconnected_lanes(self):
        """ Check if any lanes are disconnected. """
        found_error = False
        for map_name in locations:
            nusc_map = self.nusc_maps[map_name]
            disconnected = self.get_disconnected_lanes(nusc_map)
            if len(disconnected) > 0:
                print('Error: Missing connectivity in map %s for %d lanes: \n%s'
                      % (map_name, len(disconnected), disconnected))
                found_error = True
        self.assertFalse(found_error, 'Error: Found missing connectivity. See messages above!')

    @classmethod
    def get_disconnected_lanes(cls, nusc_map: NuScenesMap) -> List[str]:
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
        subtrees = cls.get_disconnected_subtrees(nusc_map.connectivity)
        disconnected = disconnected.union(subtrees)

        return sorted(list(disconnected))

    @classmethod
    def drop_disconnected_lanes(cls, nusc_map: NuScenesMap) -> NuScenesMap:
        """
        Remove any disconnected lanes.
        :param nusc_map: The NuScenesMap instance of a particular map location.
        :return: The cleaned NuScenesMap instance.
        """

        # Get disconnected lanes.
        disconnected = cls.get_disconnected_lanes(nusc_map)

        # Remove lane.
        nusc_map.lane = [lane for lane in nusc_map.lane if lane['token'] not in disconnected]

        # Remove lane_connector.
        nusc_map.lane_connector = [lane for lane in nusc_map.lane_connector if lane['token'] not in disconnected]

        # Remove connectivity entries.
        for lane_token in disconnected:
            if lane_token in nusc_map.connectivity:
                del nusc_map.connectivity[lane_token]

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

    def test_egoposes_on_map(self):
        nusc = NuScenes(version=self.version, dataroot=os.environ['NUSCENES'], verbose=False)
        whitelist = ['scene-0499', 'scene-0501', 'scene-0502', 'scene-0515', 'scene-0517']

        invalid_scenes = []
        for scene in tqdm.tqdm(nusc.scene):
            if scene['name'] in whitelist:
                continue

            log = nusc.get('log', scene['log_token'])
            map_name = log['location']
            nusc_map = self.nusc_maps[map_name]
            ratio_valid = self.get_egoposes_on_map_ratio(nusc, nusc_map, scene['token'])
            if ratio_valid != 1.0:
                print('Error: Scene %s has a ratio of %f ego poses on the driveable area!'
                      % (scene['name'], ratio_valid))
                invalid_scenes.append(scene['name'])

        self.assertEqual(len(invalid_scenes), 0)

    @classmethod
    def get_egoposes_on_map_ratio(cls, nusc: NuScenes, nusc_map: NuScenesMap, scene_token: str) -> float:
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

    @classmethod
    def get_disconnected_subtrees(cls, connectivity: Dict[str, dict]) -> Set[str]:
        """
        Compute lanes or lane_connectors that are part of disconnected subtrees.
        :param connectivity: The connectivity of the current NuScenesMap.
        :return: The lane_tokens for lanes that are part of a disconnected subtree.
        """
        # Init.
        connected = set()
        todo = set()

        # Add first lane.
        all_keys = list(connectivity.keys())
        first_key = all_keys[0]
        all_keys = set(all_keys)
        todo.add(first_key)

        while len(todo) > 0:
            # Get next lane.
            lane_token = todo.pop()
            connected.add(lane_token)

            # Add lanes connected to this lane.
            if lane_token in connectivity:
                incoming = connectivity[lane_token]['incoming']
                outgoing = connectivity[lane_token]['outgoing']
                inout_lanes = set(incoming + outgoing)
                for other_lane_token in inout_lanes:
                    if other_lane_token not in connected:
                        todo.add(other_lane_token)

        disconnected = all_keys - connected
        assert len(disconnected) < len(connected), 'Error: Bad initialization chosen!'
        return disconnected


if __name__ == '__main__':
    unittest.main()
