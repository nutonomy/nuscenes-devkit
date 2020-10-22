import os
import unittest
from collections import defaultdict
from typing import List
import tqdm

from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.nuscenes import NuScenes


class TestAllMaps(unittest.TestCase):
    version = 'v1.0-trainval'  # TODO

    def setUp(self):
        """ Initialize the map for each location. """
        self.nusc_maps = dict()
        for map_name in locations:
            if False:  # TODO map_name == 'singapore-onenorth':
                nusc_map = NuScenesMap(map_name=map_name, dataroot=os.path.expanduser('~'))  # TODO
            else:
                nusc_map = NuScenesMap(map_name=map_name, dataroot=os.environ['NUSCENES'])

            # Postprocess lanes until they are fixed.
            nusc_map = self.drop_disconnected_lanes(nusc_map)

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
        disconnected = []
        for lane_token, connectivity in nusc_map.json_obj['connectivity'].items():
            # Lanes which are disconnected.
            inout_lanes = connectivity['incoming'] + connectivity['outgoing']
            if len(inout_lanes) == 0:
                disconnected.append(lane_token)
                continue

            # Lanes that only exist in connectivity (not currently an issue).
            for inout_lane_token in inout_lanes:
                if inout_lane_token not in nusc_map._token2ind['lane'] and \
                        inout_lane_token not in nusc_map._token2ind['lane_connector']:
                    disconnected.append(inout_lane_token)

        return disconnected

    @classmethod
    def drop_disconnected_lanes(cls, nusc_map: NuScenesMap) -> NuScenesMap:
        """ Remove any disconnected lanes. """

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
        for lane_token, connectivity in nusc_map.json_obj['connectivity'].items():
            connectivity['incoming'] = [i for i in connectivity['incoming'] if i not in disconnected]
            connectivity['outgoing'] = [o for o in connectivity['outgoing'] if o not in disconnected]

        # To fix the map class, we need to update some indices.
        nusc_map._make_token2ind()

        return nusc_map

    def test_egoposes_on_map(self):
        nusc = NuScenes(version=self.version, dataroot=os.environ['NUSCENES'], verbose=False)

        invalid_scenes = []
        for scene in tqdm.tqdm(nusc.scene):
            log = nusc.get('log', scene['log_token'])
            map_name = log['location']
            nusc_map = self.nusc_maps[map_name]
            ratio_valid = self.get_egoposes_on_map_ratio(nusc, nusc_map, scene['token'])
            if ratio_valid != 1.0:
                print('Error: Scene %s has a valid ratio of %f!' % (scene['name'], ratio_valid))
                invalid_scenes.append(scene['name'])

        self.assertEqual(len(invalid_scenes), 0)

    @classmethod
    def get_egoposes_on_map_ratio(cls, nusc: NuScenes, nusc_map: NuScenesMap, scene_token: str) -> float:
        """
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


if __name__ == '__main__':
    unittest.main()
