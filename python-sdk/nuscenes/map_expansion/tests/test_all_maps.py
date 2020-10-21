import unittest
from collections import defaultdict
from typing import List

from nuscenes.map_expansion.map_api import NuScenesMap, locations


class TestAllMaps(unittest.TestCase):
    # TODO: preload map objects, make dataroot flexible

    def setUp(self):
        """ Initialize the map for each location. """
        self.nusc_maps = dict()
        for map_name in locations:
            if False:  # TODO map_name == 'singapore-onenorth':
                nusc_map = NuScenesMap(map_name=map_name, dataroot=os.path.expanduser('~'))  # TODO
            else:
                nusc_map = NuScenesMap(map_name=map_name)

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
                    print('Error: Not a lane or lane_connector: %s' % inout_lane_token)
                    disconnected.append(inout_lane_token)

        return disconnected

    @classmethod
    def drop_disconnected_lanes(cls, nusc_map: NuScenesMap) -> NuScenesMap:
        """ Remove any disconnected lanes. """

        # Get disconnected lanes.
        disconnected = cls.get_disconnected_lanes(nusc_map)

        # Remove lane.
        print('Before: ', len(nusc_map.lane))
        nusc_map.lane = [lane for lane in nusc_map.lane if lane['token'] not in disconnected]
        print('After: ', len(nusc_map.lane))

        # Remove lane_connector.
        print('Before: ', len(nusc_map.lane))
        nusc_map.lane_connector = [lane for lane in nusc_map.lane_connector if lane['token'] not in disconnected]
        print('After: ', len(nusc_map.lane))

        # Remove connectivity.
        print('Before: ', len(nusc_map.lane))
        for lane_token in disconnected:
            if lane_token in nusc_map.connectivity:
                del nusc_map.connectivity[lane_token]
        print('After: ', len(nusc_map.lane))

        # To fix the map class, we need to update some indices.
        nusc_map._make_token2ind()

        return nusc_map


if __name__ == '__main__':
    unittest.main()
