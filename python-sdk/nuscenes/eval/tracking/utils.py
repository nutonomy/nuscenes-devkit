# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

from typing import Optional, Dict

import numpy as np
import motmetrics
from motmetrics.metrics import MetricsHost

from nuscenes.eval.tracking.data_classes import TrackingMetrics
from nuscenes.eval.tracking.metrics import motap, motp_custom, faf_custom, track_initialization_duration, \
    longest_gap_duration


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
    metric_names = metrics.label_metrics.keys()
    print('\nPer-class results:')
    print('\t\t', end='')
    print('\t'.join([m.upper() for m in metric_names]))

    class_names = metrics.class_names
    max_name_length = np.max([len(c) for c in class_names])
    for class_name in class_names:
        print('%s' % class_name.ljust(max_name_length), end='')

        for metric_name in metric_names:
            val = metrics.label_metrics[metric_name][class_name]
            print_format = '%.3f' if np.isnan(val) or val != int(val) else '%d'
            print('\t%s' % (print_format % val), end='')

        print()

    # Print high-level metrics.
    print('\nAggregated results:')
    for metric_name in metric_names:
        val = metrics.compute_metric(metric_name, 'all')
        print_format = '%.3f' if np.isnan(val) or val != int(val) else '%d'
        print('%s\t%s' % (metric_name.upper(), print_format % val))

    print('Eval time: %.1fs' % metrics.eval_time)
    print()


def print_threshold_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print only a subset of the metrics for the current class and threshold.
    :param metrics: A dictionary representation of the metrics.
    """
    # Retrieve metrics.
    assert len(metrics['mota'].keys()) == 1
    threshold_str = list(metrics['mota'].keys())[0]
    mota = metrics['mota'][threshold_str]
    motp = metrics['motp_custom'][threshold_str]
    recall = metrics['recall'][threshold_str]
    num_frames = metrics['num_frames'][threshold_str]
    num_objects = metrics['num_objects'][threshold_str]
    num_predictions = metrics['num_predictions'][threshold_str]
    num_false_positives = metrics['num_false_positives'][threshold_str]
    num_misses = metrics['num_misses'][threshold_str]
    num_switches = metrics['num_switches'][threshold_str]
    num_matches = metrics['num_matches'][threshold_str]

    # Print.
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s'
          % ('\t\t', 'MOTA', 'MOTP', 'Recall', 'Frames',
             'Gt', 'Gt-Mtch', 'Gt-Miss', 'Gt-IDS',
             'Pred', 'Pred-TP', 'Pred-FP', 'Pred-IDS',))
    print('%s\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d'
          % (threshold_str, mota, motp, recall, num_frames,
             num_objects, num_matches, num_misses, num_switches,
             num_predictions, num_matches, num_false_positives, num_switches))
    print()

    # Check metrics for consistency.
    assert num_objects == num_matches + num_misses + num_switches
    assert num_predictions == num_matches + num_false_positives + num_switches


def create_motmetrics() -> MetricsHost:
    """
    Creates a MetricsHost and populates it with default and custom metrics.
    It does not populate the global metrics which are more time consuming.
    :return The initialized MetricsHost object with default MOT metrics.
    """
    mh = MetricsHost()
    fields = [
        'num_frames', 'obj_frequencies', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses',
        'num_detections', 'num_objects', 'num_predictions', 'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'motp', 'mota', 'precision', 'recall', 'track_ratios'
    ]

    for field in fields:
        mh.register(getattr(motmetrics.metrics, field), formatter='{:d}'.format)

    # Register custom metrics.
    mh.register(motap,
                ['num_matches', 'num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                formatter='{:.2%}'.format, name='motap')
    mh.register(motp_custom,
                formatter='{:.2%}'.format, name='motp_custom')
    mh.register(faf_custom,
                formatter='{:.2%}'.format, name='faf_custom')
    mh.register(track_initialization_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='tid')
    mh.register(longest_gap_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='lgd')

    return mh
