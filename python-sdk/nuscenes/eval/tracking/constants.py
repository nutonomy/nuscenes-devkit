# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']

AMOT_METRICS = ['AMOTA', 'AMOTP']
LEGACY_METRICS = ['MOTA', 'MOTP', 'IDF1', 'FAF', 'MT', 'ML', 'FP', 'FN', 'IDS', 'FRAG']
DETECTION_METRICS = ['mAP', 'trans_err', 'scale_err', 'orient_err', 'vel_err']  # Excludes attr_err.
TRACKING_METRICS = [*AMOT_METRICS, *LEGACY_METRICS, *DETECTION_METRICS]

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
