# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.


AMOT_METRICS = ['amota', 'amotp']
INTERNAL_METRICS = ['recall', 'motar', 'gt']
LEGACY_METRICS = ['mota', 'motp', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
TRACKING_METRICS = [*AMOT_METRICS, *INTERNAL_METRICS, *LEGACY_METRICS]

# Define mapping for metrics averaged over classes.
AVG_METRIC_MAP = {  # Mapping from average metric name to individual per-threshold metric name.
    'amota': 'motar',
    'amotp': 'motp'
}

# Define mapping for metrics that use motmetrics library.
MOT_METRIC_MAP = {  # Mapping from motmetrics names to metric names used here.
    'num_frames': '',  # Used in FAF.
    'num_objects': 'gt',  # Used in MOTAR computation.
    'pred_frequencies': '', # Only needed in background.
    'num_predictions': '',  # Only printed out.
    'num_matches': 'tp',  # Used in MOTAR computation and printed out.
    'motar': 'motar',  # Only used in AMOTA.
    'mota_custom': 'mota',  # Traditional MOTA, but clipped below 0.
    'motp_custom': 'motp',  # Traditional MOTP.
    'faf': 'faf',
    'mostly_tracked': 'mt',
    'mostly_lost': 'ml',
    'num_false_positives': 'fp',
    'num_misses': 'fn',
    'num_switches': 'ids',
    'num_fragmentations_custom': 'frag',
    'recall': 'recall',
    'tid': 'tid',
    'lgd': 'lgd'
}
