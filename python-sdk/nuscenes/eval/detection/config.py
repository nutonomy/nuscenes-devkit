# nuScenes dev-kit.
# Code written by Alex Lang, 2019.
# Licensed under the Creative Commons [see licence.txt]

from nuscenes.eval.detection.data_classes import DetectionConfig

eval_detection_configs = {
    'cvpr_2019': {
        'class_range': {
            'car': 50,
            'truck': 50,
            'bus': 50,
            'trailer': 50,
            'construction_vehicle': 50,
            'pedestrian': 40,
            'motorcycle': 40,
            'bicycle': 40,
            'traffic_cone': 30,
            'barrier': 30
          },
        'dist_fcn': 'center_distance',
        'dist_ths': [0.5, 1.0, 2.0, 4.0],
        'dist_th_tp': 2.0,
        'min_recall': 0.1,
        'min_precision': 0.1,
        'max_boxes_per_sample': 500,
        'mean_ap_weight': 5
    }
}


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    assert configuration_name in eval_detection_configs.keys(), \
        'Requested unknown configuration {}'.format(configuration_name)

    return DetectionConfig.deserialize(eval_detection_configs[configuration_name])
