# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest
import os

from nuscenes.nuscenes import NuScenes


class TestNuScenes(unittest.TestCase):

    def test_load(self):
        """
        Creates a dummy result file and runs NuScenesEval.
        This is intended to simply exersize a large part of the code to catch typos and syntax errors.
        """

        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        nusc = NuScenes(version='v0.2', dataroot=os.environ['NUSCENES'], verbose=False)

        self.assertEqual(nusc.table_root, os.path.join(os.environ['NUSCENES'], 'v0.2'))


if __name__ == '__main__':
    unittest.main()
