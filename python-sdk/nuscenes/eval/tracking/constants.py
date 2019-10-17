# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']

AMOT_METRICS = ['AMOTA', 'AMOTP']
INTERNAL_METRICS = ['MOTAP']
LEGACY_METRICS = ['MOTA', 'MOTP', 'FAF', 'MT', 'ML', 'FP', 'FN', 'IDS', 'FRAG', 'TID', 'LGD']
DETECTION_METRICS = ['mAP', 'trans_err', 'scale_err', 'orient_err', 'vel_err']  # Excludes attr_err.
TRACKING_METRICS = [*AMOT_METRICS, *INTERNAL_METRICS, *LEGACY_METRICS]  # TODO: add *DETECTION_METRICS

PRETTY_TRACKING_NAMES = {
    'bicycle': 'Bicycle',
    'bus': 'Bus',
    'car': 'Car',
    'motorcycle': 'Motorcycle',
    'pedestrian': 'Pedestrian',
    'trailer': 'Trailer',
    'truck': 'Truck'
}

TRACKING_COLORS = {
    'bicycle': 'C7',
    'bus': 'C2',
    'car': 'C0',
    'motorcycle': 'C6',
    'pedestrian': 'C5',
    'trailer': 'C3',
    'truck': 'C1'
}

# Define mapping for metrics averaged over classes.
AVG_METRIC_MAP = {  # Mapping from average metric name to individual per-threshold metric name.
    'amota': 'motap',
    'amotp': 'motp'
}

# Define mapping for metrics that use motmetrics library.
MOT_METRIC_MAP = {  # Mapping from motmetrics names to metric names used here.
    'num_frames': '',  # Used in FAF.
    'num_objects': '',  # Used in MOTAP computation.
    'num_predictions': '',  # Only printed out.
    'num_matches': '',  # Used in MOTAP computation and printed out.
    'mota': 'mota',  # Traditional MOTA.
    'motap': 'motap',  # Only used in AMOTA.
    'motp_custom': 'motp',  # Traditional MOTP.
    'faf_custom': 'faf',
    'mostly_tracked': 'mt',
    'mostly_lost': 'ml',
    'num_false_positives': 'fp',
    'num_misses': 'fn',
    'num_switches': 'ids',
    'num_fragmentations': 'frag',
    'tid': 'tid',
    'lgd': 'lgd'
}
