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
from nuscenes.eval.detection.utils import category_to_detection_name, detection_name_to_rel_attributes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class TestEndToEnd(unittest.TestCase):
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
                        'velocity': list(nusc.box_velocity(ann_token) * (np.random.rand(3) + 0.5)),
                        'detection_name': detection_name,
                        'detection_score': random.random(),
                        'attribute_name': random_attr(detection_name)
                    })
            mock_results[sample['token']] = sample_res
        return mock_results

    @unittest.skip
    def test_delta(self):
        """
        This tests runs the evaluation for an arbitrary random set of predictions.
        This score is then captured in this very test such that if we change the eval code,
        this test will trigger if the results changed.
        """
        random.seed(42)
        np.random.seed(42)
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'

        cfg = DetectionConfig({
            "range": 40,
            "dist_fcn": "center_distance",
            "dist_ths": [2.0, 4.0],
            "dist_th_tp": 2.0,
            "metric_bounds": {
                "trans_err": 0.5,
                "vel_err": 1.5,
                "scale_err": 0.5,
                "orient_err": 1.570796,
                "attr_err": 1
            },
            "attributes": ["cycle.with_rider", "cycle.without_rider", "pedestrian.moving",
                           "pedestrian.sitting_lying_down", "pedestrian.standing", "vehicle.moving",
                           "vehicle.parked", "vehicle.stopped"],
            "min_recall": 0.1,
            "min_precision": 0.0,
            "weighted_sum_tp_metrics": ["trans_err", "scale_err", "orient_err"],
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
            "class_names": ["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
                            "pedestrian", "traffic_cone", "trailer", "truck"]
        })

        nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        with open(self.res_mockup, 'w') as f:
            json.dump(self._mock_results(nusc), f, indent=2)

        nusc_eval = NuScenesEval(nusc, cfg, self.res_mockup, eval_set='val', output_dir=self.res_eval_folder,
                                 verbose=False)
        metrics = nusc_eval.run()

        # Score of 0.22082865720221012 was measured on the branch "release_v0.2" on March 7 2019.
        # After changing to measure center distance from the ego-vehicle this changed to 0.2199307290627096
        # Changed to 1.0-mini. Cleaned up build script. So new basline at 0.24954451673961747
        self.assertAlmostEqual(metrics['weighted_sum'], 0.24954451673961747)


if __name__ == '__main__':
    unittest.main()
