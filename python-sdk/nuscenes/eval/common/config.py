# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import json
import os
from typing import Union

from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.tracking.data_classes import TrackingConfig


def config_factory(configuration_name: str) -> Union[DetectionConfig, TrackingConfig]:
    """
    Creates a *Config instance that can be used to initialize a *Eval instance, where * stands for Detection/Tracking.
    Note that this only works if the config file is located in the nuscenes/eval/common/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: *Config instance.
    """
    # Check if config exists.
    tokens = configuration_name.split('_')
    assert len(tokens) > 1, 'Error: Configuration name must be have prefix "detection_" or "tracking_"!'
    task = tokens[0]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, '..', task, 'configs', '%s.json' % configuration_name)
    assert os.path.exists(cfg_path), 'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    if task == 'detection':
        cfg = DetectionConfig.deserialize(data)
    elif task == 'tracking':
        cfg = TrackingConfig.deserialize(data)
    else:
        raise Exception('Error: Invalid config file name: %s' % configuration_name)

    return cfg
