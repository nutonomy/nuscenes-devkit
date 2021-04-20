import os
import unittest
from collections import defaultdict

import matplotlib.pyplot as plt
import tqdm

from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.map_expansion.utils import get_egoposes_on_drivable_ratio, get_disconnected_lanes
from nuscenes.nuscenes import NuScenes


class TestAllMaps(unittest.TestCase):
    version = 'v1.0-mini'
    render = False

    def setUp(self):
        """ Initialize the map for each location. """

        self.nusc_maps = dict()
        for map_name in locations:
            # Load map.
            nusc_map = NuScenesMap(map_name=map_name, dataroot=os.environ['NUSCENES'])

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
            'singapore-hollandvillage': [426, 167, 387, 601, 28, 498, 300, 0, 107, 220, 119],
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

    @unittest.skip  # This test is known to fail on dozens of disconnected lanes.
    def test_disconnected_lanes(self):
        """ Check if any lanes are disconnected. """
        found_error = False
        for map_name in locations:
            nusc_map = self.nusc_maps[map_name]
            disconnected = get_disconnected_lanes(nusc_map)
            if len(disconnected) > 0:
                print('Error: Missing connectivity in map %s for %d lanes: \n%s'
                      % (map_name, len(disconnected), disconnected))
                found_error = True
        self.assertFalse(found_error, 'Error: Found missing connectivity. See messages above!')

    def test_egoposes_on_map(self):
        """ Test that all ego poses land on """
        nusc = NuScenes(version=self.version, dataroot=os.environ['NUSCENES'], verbose=False)
        whitelist = ['scene-0499', 'scene-0501', 'scene-0502', 'scene-0515', 'scene-0517']

        invalid_scenes = []
        for scene in tqdm.tqdm(nusc.scene, leave=False):
            if scene['name'] in whitelist:
                continue

            log = nusc.get('log', scene['log_token'])
            map_name = log['location']
            nusc_map = self.nusc_maps[map_name]
            ratio_valid = get_egoposes_on_drivable_ratio(nusc, nusc_map, scene['token'])
            if ratio_valid != 1.0:
                print('Error: Scene %s has a ratio of %f ego poses on the driveable area!'
                      % (scene['name'], ratio_valid))
                invalid_scenes.append(scene['name'])

        self.assertEqual(len(invalid_scenes), 0)


if __name__ == '__main__':
    unittest.main()
