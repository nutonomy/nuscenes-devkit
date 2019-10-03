# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

import json
import os
import sys
import random
import shutil
import unittest
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.eval.tracking.constants import TRACKING_NAMES
from nuscenes.utils.splits import create_splits_scenes


class TestMain(unittest.TestCase):
    res_mockup = 'nusc_eval.json'
    res_eval_folder = 'tmp'

    def tearDown(self):
        if os.path.exists(self.res_mockup):
            os.remove(self.res_mockup)
        if os.path.exists(self.res_eval_folder):
            shutil.rmtree(self.res_eval_folder)

    @staticmethod
    def _mock_submission(nusc: NuScenes, split: str, add_errors: bool = False) -> Dict[str, dict]:
        """
        Creates "reasonable" submission (results and metadata) by looping through the mini-val set, adding 1 GT
        prediction per sample. Predictions will be permuted randomly along all axes.
        """

        def random_class(category_name: str, add_errors: bool = False) -> Optional[str]:
            # Alter 10% of the valid labels.
            class_names = sorted(TRACKING_NAMES)
            tmp = category_to_tracking_name(category_name)

            if tmp is None:
                return None
            else:
                if not add_errors or np.random.rand() < .9:
                    return tmp
                else:
                    return class_names[np.random.randint(0, len(class_names) - 1)]

        def random_id(instance_token: str, add_errors: bool = False) -> str:
            # Alter 10% of the valid ids to be a random string, which hopefully corresponds to a new track.
            if not add_errors or np.random.rand() < .9:
                _tracking_id = instance_token + '_pred'
            else:
                _tracking_id = str(np.random.randint(0, sys.maxsize))

            return _tracking_id

        mock_meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }
        mock_results = {}
        splits = create_splits_scenes()
        val_samples = []
        for sample in nusc.sample:
            if nusc.get('scene', sample['scene_token'])['name'] in splits[split]:
                val_samples.append(sample)

        for sample in tqdm(val_samples):
            sample_res = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                translation = list(np.array(ann['translation']))
                size = list(np.array(ann['size']))
                rotation = list(np.array(ann['rotation']))
                velocity = nusc.box_velocity(ann_token)[:2]
                tracking_id = random_id(ann['instance_token'], add_errors=add_errors)
                tracking_name = random_class(ann['category_name'], add_errors=add_errors)
                if tracking_name is None:
                    continue
                tracking_score = 1.0

                if add_errors:
                    translation += 5 * (np.random.rand(3) - 0.5)
                    size *= 2 * (np.random.rand(3) + 0.5)
                    rotation += (np.random.rand(4) - 0.5) * .1
                    velocity *= np.random.rand(3)[:2] + 0.5
                    tracking_score = random.random()

                sample_res.append(
                    {
                        'sample_token': sample['token'],
                        'translation': translation,
                        'size': size,
                        'rotation': rotation,
                        'velocity': list(velocity),
                        'tracking_id': tracking_id,
                        'tracking_name': tracking_name,
                        'tracking_score': tracking_score
                    })
            mock_results[sample['token']] = sample_res
        mock_submission = {
            'meta': mock_meta,
            'results': mock_results
        }
        return mock_submission

    def test_delta_mock(self):
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
            json.dump(self._mock_submission(nusc, 'mini_val', add_errors=True), f, indent=2)

        cfg = config_factory('tracking_nips_2019')
        nusc_eval = TrackingEval(nusc, cfg, self.res_mockup, eval_set='mini_val', output_dir=self.res_eval_folder,
                                 verbose=False)
        metrics, md_list = nusc_eval.evaluate()

        # 1. Score = TODO.
        self.assertAlmostEqual(metrics.mota, -1)  # TODO: set score

    def test_delta_gt(self):
        """
        This tests runs the evaluation with the ground truth used as predictions.
        This should result in a perfect score for every metric.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """
        random.seed(42)
        np.random.seed(42)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        with open(self.res_mockup, 'w') as f:
            json.dump(self._mock_submission(nusc, 'mini_val', add_errors=False), f, indent=2)

        cfg = config_factory('tracking_nips_2019')
        nusc_eval = TrackingEval(nusc, cfg, self.res_mockup, eval_set='mini_val', output_dir=self.res_eval_folder,
                                 verbose=False)
        metrics, md_list = nusc_eval.evaluate()

        # TODO: check more metrics
        self.assertAlmostEqual(metrics.mota, 1)


if __name__ == '__main__':
    TestMain().test_delta_gt()
