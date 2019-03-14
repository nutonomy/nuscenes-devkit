# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest
import random
import json
import os
import shutil

from nuscenes.eval.detection.main import NuScenesEval
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class TestEndToEnd(unittest.TestCase):

    def test_simple(self):
        """
        Creates a dummy result file and runs NuScenesEval.
        This is intended to simply exersize a large part of the code to catch typos and syntax errors.
        """

        random.seed(43)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        nusc = NuScenes(version='v0.2', dataroot=os.environ['NUSCENES'], verbose=False)

        splits = create_splits_scenes()
        one_scene_token = nusc.field2token('scene', 'name', splits['val'][0])
        one_scene = nusc.get('scene', one_scene_token[0])

        def make_mock_entry(sample_token):
            return {
                'sample_token': sample_token,
                'translation': [1.0, 2.0, 3.0],
                'size': [1.0, 2.0, 3.0],
                'rotation': [1.0, 2.0, 2.0, 3.0],
                'velocity': [1.0, 2.0, 3.0],
                'detection_name': 'vehicle.car',
                'detection_score': random.random(),
                'attribute_scores': [.1, .2, .3, .4, .5, .6, .7, .8]
            }

        pred = {
            one_scene['first_sample_token']: [
                make_mock_entry(one_scene['first_sample_token']),
                make_mock_entry(one_scene['first_sample_token'])],
            one_scene['last_sample_token']: [
                make_mock_entry(one_scene['last_sample_token']),
                make_mock_entry(one_scene['last_sample_token'])],
        }

        res_mockup = 'nsc_eval.json'
        res_eval_folder = 'tmp'

        with open(res_mockup, 'w') as f:
            json.dump(pred, f)

        nusc_eval = NuScenesEval(nusc, res_mockup, eval_set='val', output_dir=res_eval_folder, verbose=True)
        nusc_eval.run_eval()

        # Trivial assert statement
        self.assertEqual(nusc_eval.output_dir, res_eval_folder)

        os.remove(res_mockup)
        shutil.rmtree(res_eval_folder)


if __name__ == '__main__':
    unittest.main()
