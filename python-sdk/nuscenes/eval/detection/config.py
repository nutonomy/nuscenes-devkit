# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import json
import os

from nuscenes.eval.detection.data_classes import DetectionConfig


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    # Check if config exists.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, 'configs', '%s.json' % configuration_name)
    assert os.path.exists(cfg_path), \
        'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)

    return cfg
