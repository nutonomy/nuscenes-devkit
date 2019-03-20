# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json
import os
import random
import shutil
import unittest
from typing import Dict

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.detection import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name, detection_name_to_rel_attributes
from nuscenes.utils.splits import create_splits_scenes


class TestMain(unittest.TestCase):
    res_mockup = 'nsc_eval.json'
    res_eval_folder = 'tmp'

    def tearDown(self):
        if os.path.exists(self.res_mockup):
            os.remove(self.res_mockup)
        if os.path.exists(self.res_eval_folder):
            shutil.rmtree(self.res_eval_folder)

    @staticmethod
    def _mock_results(nusc) -> Dict[str, list]:
        """
        Creates "reasonable" results by looping through the full val-set, and adding 1 prediction per GT.
        Predictions will be permuted randomly along all axes.
        """

        def random_class(category_name):
            class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                           'traffic_cone', 'trailer', 'truck']
            tmp = category_to_detection_name(category_name)
            if tmp is not None and np.random.rand() < .9:
                return tmp
            else:
                return class_names[np.random.randint(0, 9)]

        def random_attr(name):
            """
            This is the most straight-forward way to generate a random attribute.
            Not currently used b/c we want the test fixture to be back-wards compatible.
            """
            # Get relevant attributes.
            rel_attributes = detection_name_to_rel_attributes(name)

            if len(rel_attributes) == 0:
                # Empty string for classes without attributes.
                return ''
            else:
                # Pick a random attribute otherwise.
                return rel_attributes[np.random.randint(0, len(rel_attributes))]

        mock_results = {}
        splits = create_splits_scenes()
        val_samples = []
        for sample in nusc.sample:
            if nusc.get('scene', sample['scene_token'])['name'] in splits['val']:
                val_samples.append(sample)

        for sample in tqdm(val_samples):
            sample_res = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                detection_name = random_class(ann['category_name'])
                sample_res.append(
                    {
                        'sample_token': sample['token'],
                        'translation': list(np.array(ann['translation']) + 5 * (np.random.rand(3) - 0.5)),
                        'size': list(np.array(ann['size']) * 2 * (np.random.rand(3) + 0.5)),
                        'rotation': list(np.array(ann['rotation']) + ((np.random.rand(4) - 0.5) * .1)),
                        'velocity': list(nusc.box_velocity(ann_token)[:2] * (np.random.rand(3)[:2] + 0.5)),
                        'detection_name': detection_name,
                        'detection_score': random.random(),
                        'attribute_name': random_attr(detection_name)
                    })
            mock_results[sample['token']] = sample_res
        return mock_results

    def test_delta(self):
        """
        This tests runs the evaluation for an arbitrary random set of predictions.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """
        random.seed(42)
        np.random.seed(42)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        with open(self.res_mockup, 'w') as f:
            json.dump(self._mock_results(nusc), f, indent=2)

        cfg = config_factory('cvpr_2019')

        nusc_eval = NuScenesEval(nusc, cfg, self.res_mockup, eval_set='val', output_dir=self.res_eval_folder,
                                 verbose=False)
        metrics = nusc_eval.run()

        # 1. Score = 0.22082865720221012. Measured on the branch "release_v0.2" on March 7 2019.
        # 2. Score = 0.2199307290627096. Changed to measure center distance from the ego-vehicle.
        # 3. Score = 0.24954451673961747. Changed to 1.0-mini and cleaned up build script.
        # 4. Score = 0.20478832626986893. Updated treatment of cones, barriers, and other algo tunings.
        # 5. Score = 0.2043569666105005. AP calculation area is changed from >=min_recall to >min_recall.
        # 6. Score = 0.20636954644294506. After bike-rack filtering.
        # 7. Score - 0.20237925145690996. After TP reversion bug.
        self.assertAlmostEqual(metrics.weighted_sum, 0.20237925145690996)


if __name__ == '__main__':
    unittest.main()
