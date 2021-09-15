# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import json
import os
import random
import shutil
import sys
import unittest
from typing import Dict, Optional, Any

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.utils import category_to_tracking_name
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
    def _mock_submission(nusc: NuScenes,
                         split: str,
                         add_errors: bool = False) -> Dict[str, dict]:
        """
        Creates "reasonable" submission (results and metadata) by looping through the mini-val set, adding 1 GT
        prediction per sample. Predictions will be permuted randomly along all axes.
        :param nusc: NuScenes instance.
        :param split: Dataset split to use.
        :param add_errors: Whether to use GT or add errors to it.
        """
        # Get config.
        cfg = config_factory('tracking_nips_2019')

        def random_class(category_name: str, _add_errors: bool = False) -> Optional[str]:
            # Alter 10% of the valid labels.
            class_names = sorted(cfg.tracking_names)
            tmp = category_to_tracking_name(category_name)

            if tmp is None:
                return None
            else:
                if not _add_errors or np.random.rand() < .9:
                    return tmp
                else:
                    return class_names[np.random.randint(0, len(class_names) - 1)]

        def random_id(instance_token: str, _add_errors: bool = False) -> str:
            # Alter 10% of the valid ids to be a random string, which hopefully corresponds to a new track.
            if not _add_errors or np.random.rand() < .9:
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

        # Get all samples in the current evaluation split.
        splits = create_splits_scenes()
        val_samples = []
        for sample in nusc.sample:
            if nusc.get('scene', sample['scene_token'])['name'] in splits[split]:
                val_samples.append(sample)

        # Prepare results.
        instance_to_score = dict()
        for sample in tqdm(val_samples, leave=False):
            sample_res = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                translation = np.array(ann['translation'])
                size = np.array(ann['size'])
                rotation = np.array(ann['rotation'])
                velocity = nusc.box_velocity(ann_token)[:2]
                tracking_id = random_id(ann['instance_token'], _add_errors=add_errors)
                tracking_name = random_class(ann['category_name'], _add_errors=add_errors)

                # Skip annotations for classes not part of the detection challenge.
                if tracking_name is None:
                    continue

                # Skip annotations with 0 lidar/radar points.
                num_pts = ann['num_lidar_pts'] + ann['num_radar_pts']
                if num_pts == 0:
                    continue

                # If we randomly assign a score in [0, 1] to each box and later average over the boxes in the track,
                # the average score will be around 0.5 and we will have 0 predictions above that.
                # Therefore we assign the same scores to each box in a track.
                if ann['instance_token'] not in instance_to_score:
                    instance_to_score[ann['instance_token']] = random.random()
                tracking_score = instance_to_score[ann['instance_token']]
                tracking_score = np.clip(tracking_score + random.random() * 0.3, 0, 1)

                if add_errors:
                    translation += 4 * (np.random.rand(3) - 0.5)
                    size *= (np.random.rand(3) + 0.5)
                    rotation += (np.random.rand(4) - 0.5) * .1
                    velocity *= np.random.rand(3)[:2] + 0.5

                sample_res.append({
                        'sample_token': sample['token'],
                        'translation': list(translation),
                        'size': list(size),
                        'rotation': list(rotation),
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

    @unittest.skip
    def basic_test(self,
                   eval_set: str = 'mini_val',
                   add_errors: bool = False,
                   render_curves: bool = False) -> Dict[str, Any]:
        """
        Run the evaluation with fixed randomness on the specified subset, with or without introducing errors in the
        submission.
        :param eval_set: Which split to evaluate on.
        :param add_errors: Whether to use GT as submission or introduce additional errors.
        :param render_curves: Whether to render stats curves to disk.
        :return: The metrics returned by the evaluation.
        """
        random.seed(42)
        np.random.seed(42)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        if eval_set.startswith('mini'):
            version = 'v1.0-mini'
        elif eval_set == 'test':
            version = 'v1.0-test'
        else:
            version = 'v1.0-trainval'
        nusc = NuScenes(version=version, dataroot=os.environ['NUSCENES'], verbose=False)

        with open(self.res_mockup, 'w') as f:
            mock = self._mock_submission(nusc, eval_set, add_errors=add_errors)
            json.dump(mock, f, indent=2)

        cfg = config_factory('tracking_nips_2019')
        nusc_eval = TrackingEval(cfg, self.res_mockup, eval_set=eval_set, output_dir=self.res_eval_folder,
                                 nusc_version=version, nusc_dataroot=os.environ['NUSCENES'], verbose=False)
        metrics = nusc_eval.main(render_curves=render_curves)

        return metrics

    @unittest.skip
    def test_delta_mock(self,
                        eval_set: str = 'mini_val',
                        render_curves: bool = False):
        """
        This tests runs the evaluation for an arbitrary random set of predictions.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        :param eval_set: Which set to evaluate on.
        :param render_curves: Whether to render stats curves to disk.
        """
        # Run the evaluation with errors.
        metrics = self.basic_test(eval_set, add_errors=True, render_curves=render_curves)

        # Compare metrics to known solution.
        if eval_set == 'mini_val':
            self.assertAlmostEqual(metrics['amota'], 0.23766771095785147)
            self.assertAlmostEqual(metrics['amotp'], 1.5275400961369252)
            self.assertAlmostEqual(metrics['motar'], 0.3726570200013319)
            self.assertAlmostEqual(metrics['mota'], 0.25003943918566174)
            self.assertAlmostEqual(metrics['motp'], 1.2976508610883917)
        else:
            print('Skipping checks due to choice of custom eval_set: %s' % eval_set)

    @unittest.skip
    def test_delta_gt(self,
                      eval_set: str = 'mini_val',
                      render_curves: bool = False):
        """
        This tests runs the evaluation with the ground truth used as predictions.
        This should result in a perfect score for every metric.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        :param eval_set: Which set to evaluate on.
        :param render_curves: Whether to render stats curves to disk.
        """
        # Run the evaluation without errors.
        metrics = self.basic_test(eval_set, add_errors=False, render_curves=render_curves)

        # Compare metrics to known solution. Do not check:
        # - MT/TP (hard to figure out here).
        # - AMOTA/AMOTP (unachieved recall values lead to hard unintuitive results).
        if eval_set == 'mini_val':
            self.assertAlmostEqual(metrics['amota'], 1.0)
            self.assertAlmostEqual(metrics['amotp'], 0.0, delta=1e-5)
            self.assertAlmostEqual(metrics['motar'], 1.0)
            self.assertAlmostEqual(metrics['recall'], 1.0)
            self.assertAlmostEqual(metrics['mota'], 1.0)
            self.assertAlmostEqual(metrics['motp'], 0.0, delta=1e-5)
            self.assertAlmostEqual(metrics['faf'], 0.0)
            self.assertAlmostEqual(metrics['ml'], 0.0)
            self.assertAlmostEqual(metrics['fp'], 0.0)
            self.assertAlmostEqual(metrics['fn'], 0.0)
            self.assertAlmostEqual(metrics['ids'], 0.0)
            self.assertAlmostEqual(metrics['frag'], 0.0)
            self.assertAlmostEqual(metrics['tid'], 0.0)
            self.assertAlmostEqual(metrics['lgd'], 0.0)
        else:
            print('Skipping checks due to choice of custom eval_set: %s' % eval_set)


if __name__ == '__main__':
    unittest.main()
