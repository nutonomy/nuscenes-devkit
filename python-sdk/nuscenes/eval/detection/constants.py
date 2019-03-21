# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Licensed under the Creative Commons [see licence.txt]

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   'traffic_cone', 'barrier']
PRETTY_DETECTION_NAMES = {'car': 'Car', 'truck': 'Truck', 'bus': 'Bus', 'trailer': 'Trailer',
                          'construction_vehicle': 'Construction Vehicle', 'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle', 'bicycle': 'Bicycle', 'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

TP_METRICS_UNITS = {'trans_err': 'm', 'scale_err': '1-IOU', 'orient_err': 'rad', 'vel_err': 'm/s', 'attr_err': '1-acc'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9'}
