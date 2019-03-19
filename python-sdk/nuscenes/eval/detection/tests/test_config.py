# nuScenes dev-kit.
# Code written by Alex Lang, 2019.
# Licensed under the Creative Commons [see licence.txt]

import unittest

from nuscenes.eval.detection.config import eval_detection_configs, config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig


class TestConfigs(unittest.TestCase):

    def test_DetectionConfig_serialization(self):
        """ Test serialization round trip """
        cfg = {
            'class_range': {
                'class1': 0,
                'class2': 0,
              },
            'dist_fcn': 'distance',
            'dist_ths': [0.0, 1.0],
            'dist_th_tp': 1.0,
            'min_recall': 0.0,
            'min_precision': 0.0,
            'max_boxes_per_sample': 1,
            'mean_ap_weight': 1
        }

        detect_config = DetectionConfig.deserialize(cfg)
        cfg_output = detect_config.serialize()

        self.assertEqual(cfg, cfg_output)

    def test_all_configs(self):
        """ Confirm that all configurations in eval_detection_configs work in the factory """

        for name in eval_detection_configs.items():
            detect_config = config_factory(name)
            self.assertTrue(isinstance(detect_config, DetectionConfig))
