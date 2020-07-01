from typing import Dict, Iterable


def get_colormap() -> Dict[str, Iterable[int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": [0, 0, 0],  # Black.
        "human.pedestrian.adult": [0, 0, 230],  # Blue
        "human.pedestrian.child": [135, 206, 235],  # Skyblue,
        "human.pedestrian.wheelchair": [138, 43, 226],  # Blueviolet
        "human.pedestrian.stroller": [240, 128, 128],  # Lightcoral
        "human.pedestrian.personal_mobility": [219, 112, 147],  # Palevioletred
        "human.pedestrian.police_officer": [0, 0, 128],  # Navy,
        "human.pedestrian.construction_worker": [100, 149, 237],  # Cornflowerblue
        "animal": [70, 130, 180],  # Steelblue
        "vehicle.car": [255, 158, 0],  # Orange
        "vehicle.motorcycle": [255, 61, 99],  # Red
        "vehicle.bicycle": [220, 20, 60],  # Crimson
        "vehicle.bus.bendy": [255, 127, 80],  # Coral
        "vehicle.bus.rigid": [255, 69, 0],  # Orangered
        "vehicle.truck": [255, 99, 71],  # Tomato
        "vehicle.construction": [233, 150, 70],  # Darksalmon
        "vehicle.emergency.ambulance": [255, 83, 0],
        "vehicle.emergency.police": [255, 215, 0],  # Gold
        "vehicle.trailer": [255, 140, 0],  # Darkorange
        "movable_object.barrier": [112, 128, 144],  # Slategrey
        "movable_object.trafficcone": [47, 79, 79],  # Darkslategrey
        "movable_object.pushable_pullable": [105, 105, 105],  # Dimgrey
        "movable_object.debris": [210, 105, 30],  # Chocolate
        "static_object.bicycle_rack": [188, 143, 143],  # Rosybrown
        "flat.driveable_surface": [0, 207, 191],  # nuTonomy green
        "flat.sidewalk": [75, 0, 75],
        "flat.terrain": [112, 180, 60],
        "flat.other": [175, 0, 75],
        "static.manmade": [222, 184, 135],  # Burlywood
        "static.vegetation": [0, 175, 0],  # Green
        "static.other": [255, 228, 196]  # Bisque
    }

    return classname_to_color
