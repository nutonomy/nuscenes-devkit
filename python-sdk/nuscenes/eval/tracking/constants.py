# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

TRACKING_NAMES = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']

AMOT_METRICS = ['AMOTA', 'AMOTP']
LEGACY_METRICS = ['MOTA', 'MOTP', 'IDF1', 'FAF', 'MT', 'ML', 'FP', 'FN', 'IDS', 'FRAG']
DETECTION_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err']  # Excludes attr_err.
TRACKING_METRICS = [*AMOT_METRICS, *LEGACY_METRICS, *DETECTION_METRICS]

PRETTY_TRACKING_NAMES = {'car': 'Car',
                         'truck': 'Truck',
                         'bus': 'Bus',
                         'trailer': 'Trailer',
                         'pedestrian': 'Pedestrian',
                         'motorcycle': 'Motorcycle',
                         'bicycle': 'Bicycle'}

TRACKING_COLORS = {'car': 'C0',
                   'truck': 'C1',
                   'bus': 'C2',
                   'trailer': 'C3',
                   'pedestrian': 'C5',
                   'motorcycle': 'C6',
                   'bicycle': 'C7'}
