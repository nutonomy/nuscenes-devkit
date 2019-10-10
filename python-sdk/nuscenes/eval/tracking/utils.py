# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

from typing import Optional, List, Dict

import numpy as np

from nuscenes.eval.tracking.data_classes import TrackingMetrics


def category_to_tracking_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None


def print_final_metrics(metrics: TrackingMetrics) -> None:
    """
    Print metrics to stdout.
    :param metrics: The output of evaluate().
    """
    print('\n### Final results ###')

    # Print per-class metrics.
    metric_names = metrics.raw_metrics.keys()
    print('\nPer-class results:')
    print('\t\t', end='')
    print('\t'.join([m.upper() for m in metric_names]))

    class_names = metrics.class_names
    max_name_length = np.max([len(c) for c in class_names])
    for class_name in class_names:
        print('%s' % class_name.ljust(max_name_length), end='')

        for metric_name in metric_names:
            val = metrics.raw_metrics[metric_name][class_name]
            print_type = '%.3f' if np.isnan(val) or val != int(val) else '%d'
            print_str = '\t%s' % print_type
            print(print_str % val, end='')

        print()

    # Print high-level metrics.
    print('\nPer class-average results:')
    for metric_name in metric_names:
        print('%s\t%.3f' % (metric_name.upper(), metrics.compute_metric(metric_name, 'avg')))

    print('Eval time: %.1fs' % metrics.eval_time)
    print()


def print_threshold_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print only a subset of the metrics for the current class and threshold.
    :param metrics: A dictionary representation of the metrics.
    """
    # Retrieve metrics.
    threshold_str = list(metrics['mota'].keys())[0]
    mota = metrics['mota'][threshold_str]
    motp = metrics['motp_custom'][threshold_str]
    num_objects = metrics['num_objects'][threshold_str]
    num_false_positives = metrics['num_false_positives'][threshold_str]
    num_misses = metrics['num_misses'][threshold_str]

    # Print.
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('\t\t', 'MOTA', 'MOTP', 'P', 'TP', 'FP', 'FN'))
    print('%s\t%.3f\t%.3f\t%d\t%d\t%d\t%d'
          % (threshold_str, mota, motp, num_objects, num_objects - num_false_positives,
             num_false_positives, num_misses))
    print()
