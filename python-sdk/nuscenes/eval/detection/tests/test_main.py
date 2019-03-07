# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest
import random
import json
import os
import shutil
from typing import Dict

from tqdm import tqdm
import numpy as np

from nuscenes.eval.detection.main import NuScenesEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class TestEndToEnd(unittest.TestCase):
    res_mockup = 'nsc_eval.json'
    res_eval_folder = 'tmp'

    def tearDown(self):
        if os.path.exists(self.res_mockup):
            os.remove(self.res_mockup)
        shutil.rmtree(self.res_eval_folder)

    @staticmethod
    def _mock_results(nusc) -> Dict[str, list]:
        """
        Creates "reasonable" results by looping through the full val-set, and adding 1 prediction per GT.
        Predictions will be permuted randomly along all axis.
        """

        def random_class(category_name):
            class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                           'traffic_cone', 'trailer', 'truck']
            tmp = category_to_detection_name(category_name)
            if tmp is not None and np.random.rand() < .9:
                return tmp
            else:
                return class_names[np.random.randint(0, 9)]

        random.seed(43)
        mock_results = {}
        splits = create_splits_scenes(nusc)
        val_samples = []
        for sample in nusc.sample:
            if nusc.get('scene', sample['scene_token'])['name'] in splits['val']:
                val_samples.append(sample)

        for sample in tqdm(val_samples):
            sample_res = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)

                sample_res.append(
                    {
                        'sample_token': sample['token'],
                        'translation': list(np.array(ann['translation']) + 10 * (np.random.rand(3) - 0.5)),
                        'size': list(np.array(ann['size']) * 2 * (np.random.rand(3) + 0.5)),
                        'rotation': list(np.array(ann['rotation']) + ((np.random.rand(4) - 0.5) * .1)),
                        'velocity': list(nusc.box_velocity(ann_token) * (np.random.rand(3) + 0.5)),
                        'detection_name': random_class(ann['category_name']),
                        'detection_score': random.random(),
                        'attribute_scores': list(np.random.rand(8))
                    }
                )
            mock_results[sample['token']] = sample_res
        return mock_results

    @unittest.skip("TODO unskip once done with the others")
    def test_simple(self):
        """
        Creates a dummy result file and runs NuScenesEval.
        This is intended to simply exersize a large part of the code to catch typos and syntax errors.
        """

        random.seed(43)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        nusc = NuScenes(version='v0.2', dataroot=os.environ['NUSCENES'], verbose=False)

        splits = create_splits_scenes(nusc)
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

        with open(self.res_mockup, 'w') as f:
            json.dump(pred, f)

        nusc_eval = NuScenesEval(nusc, self.res_mockup, eval_set='val', output_dir=self.res_eval_folder, verbose=True)
        nusc_eval.run_eval()

        # Trivial assert statement
        self.assertEqual(nusc_eval.output_dir, self.res_eval_folder)

    def test_delta(self):
        """
        This tests evaluates the score of a plausible, arbitrary and random set of predictions.
        The goal is to get some reasonable score.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """

        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        nusc = NuScenes(version='v0.2', dataroot=os.environ['NUSCENES'], verbose=False)

        with open(self.res_mockup, 'w') as f:
            json.dump(self._mock_results(nusc), f, indent=2)

        nusc_eval = NuScenesEval(nusc, self.res_mockup, eval_set='val', output_dir=self.res_eval_folder, verbose=True)
        nusc_eval.run_eval()

        # Trivial assert statement
        self.assertEqual(nusc_eval.output_dir, self.res_eval_folder)


if __name__ == '__main__':
    unittest.main()
