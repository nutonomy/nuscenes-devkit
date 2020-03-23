import unittest

from nuscenes.map_expansion.map_api import NuScenesMap


class TestInitAllMaps(unittest.TestCase):

    def test_init_all_maps(self):
        for my_map in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
            nusc_map = NuScenesMap(map_name=my_map)
