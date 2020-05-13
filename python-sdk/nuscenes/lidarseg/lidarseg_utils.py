# nuScenes dev-kit.
# Code written by Fong Whye Kit, 2020.

import colorsys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_stats(points_label: np.array, num_classes: int) -> List[int]:
    """
    Get frequency of each label in a point cloud.
    :param num_classes: The number of classes.
    :param points_label: A numPy array which contains the labels of the point cloud; e.g. np.array([2, 1, 34, ..., 38])
    :returns: An array which contains the counts of each label in the point cloud. The index of the point cloud
              corresponds to the index of the class label. E.g. [0, 2345, 12, 451] means that there are no points in
              class 0, there are 2345 points in class 1, there are 12 points in class 2 etc.
    """

    lidarseg_counts = [0] * num_classes  # Create as many bins as there are classes, and initialize all bins as 0.

    indices: np.ndarray = np.bincount(points_label)
    ii = np.nonzero(indices)[0]

    for class_idx, class_count in zip(ii, indices[ii]):
        lidarseg_counts[class_idx] += class_count  # Increment the count for the particular class name.

    return lidarseg_counts


def get_key_from_value(dictionary: dict, target_value: str):
    """
    Get the key belonging to a desired value in a dictionary. If there are multiple values which match the target_value,
    only the first key will be returned.
    :param dictionary: The dictionary to act on.
    :param target_value: The desired value.
    """
    return list(dictionary.keys())[list(dictionary.values()).index(target_value)]


def plt_to_cv2(points: np.array, coloring: np.array, im, imsize: Tuple[int, int] = (640, 360), dpi: int = 100):
    """
    Converts a scatter plot in matplotlib to an image in cv2. This is useful as cv2 is unable to do
    scatter plats.
    :param points: A numPy array (of size [2 x num_points] and type float) representing the pointcloud.
    :param coloring: A numPy array (of size [num_points] containing the color (in RGB, normalized
                     between 0 and 1) for each point.
    :param im: An image (e.g. a camera view) to put the scatter plot on.
    :param imsize: Size of image to render. The larger the slower this will run.
    :param dpi: Resolution of the output figure.
    :return: cv2 image with the scatter plot.
    """
    # Render lidarseg labels in image.
    fig = plt.figure(figsize=(imsize[0] / dpi, imsize[1] / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)

    # Convert from pyplot to cv2.
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)  # Put pixel buffer in numpy array.
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = cv2.resize(mat, imsize)

    return mat


def get_colormap() -> np.ndarray:
    default = [255, 0, 0]

    classname_to_color = {  # RGB.
        "noise": [0, 0, 0],  # Black.
        "human.pedestrian.adult": [255, 30, 30],
        "human.pedestrian.child": default,
        "human.pedestrian.wheelchair": default,
        "human.pedestrian.stroller": default,
        "human.pedestrian.personal_mobility": default,
        "human.pedestrian.police_officer": default,
        "human.pedestrian.construction_worker": default,
        "animal": default,
        "vehicle.car": [100, 150, 245],
        "vehicle.motorcycle": [30, 60, 150],
        "vehicle.bicycle": [100, 230, 245],
        "vehicle.bus.bendy": default,
        "vehicle.bus.rigid": default,
        "vehicle.truck": [80, 30, 180],
        "vehicle.construction": default,
        "vehicle.emergency.ambulance": default,
        "vehicle.emergency.police": default,
        "vehicle.trailer": default,
        "movable_object.barrier": default,
        "movable_object.trafficcone": default,
        "movable_object.pushable_pullable": default,
        "movable_object.debris": default,
        "static_object.bicycle_rack": default,
        "vehicle.on_rail": [0, 0, 255],  # Blue
        "vehicle.emergency.firetruck": default,
        "flat.driveable_surface": [255, 0, 255],
        "flat.sidewalk": [75, 0, 75],
        "flat.terrain": [150, 240, 80],
        "flat.other": [175, 0, 75],
        "static.manmade": default,
        "static.vegetation": [0, 175, 0],
        "static.other": default
    }

    coloring = dict(classname_to_color.copy())

    colormap = []
    for k, v in coloring.items():
        colormap.append(v)

    colormap = np.array(colormap) / 255  # Normalize RGB values to be between 0 and 1 for each channel.

    return colormap


def get_arbitrary_colormap(num_classes: int, random_seed: int = 2020) -> np.ndarray:
    """
    Create an arbitrary RGB colormap. Note that the RGB values are normalized between 0 and 1, not 0 and 255.
    :param num_classes: Number of colors to create.
    :param random_seed: The random seed to use.
    """
    num_classes = num_classes - 1  # No need colormap for class 0 as it will be fixed as black further down.

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colormap = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    np.random.seed(random_seed)  # Fix seed for consistent colors across runs.
    np.random.shuffle(colormap)  # Shuffle colors to de-correlate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    colormap = [(0, 0, 0)] + colormap  # Ensures that (class 0, which is noise) is black.
    colormap = np.array(colormap)  # Colormap is RGB with values for each channel normalized between 0 and 1.

    return colormap


def filter_colormap(colormap: np.array, classes_to_display: np.array) -> np.ndarray:
    """
    Given a colormap (in RGB) and a list of classes to display, return a colormap (in RGBA) with the opacity
    of the labels to be display set to 1.0 and those to be hidden set to 0.0
    :param colormap: [n x 3] array where each row consist of the RGB values for the corresponding class index
    :param classes_to_display: An array of classes to display (e.g. [1, 8, 32]). The array need not be ordered.
    :return (colormap <np.float: n, 4)>).

    colormap = np.array([[R1, G1, B1],             colormap = np.array([[1.0, 1.0, 1.0, 0.0],
                         [R2, G2, B2],   ------>                        [R2,  G2,  B2,  1.0],
                         ...,                                           ...,
                         Rn, Gn, Bn]])                                  [1.0, 1.0, 1.0, 0.0]])
    """
    for i in range(len(colormap)):
        if i not in classes_to_display:
            colormap[i] = [1.0, 1.0, 1.0]  # Mask labels to be hidden with 1.0 in all channels.

    # Convert the RGB colormap to an RGBA array, with the alpha channel set to zero whenever the R, G and B channels
    # are all equal to 1.0.
    alpha = np.array([~np.all(colormap == 1.0, axis=1) * 1.0])
    colormap = np.concatenate((colormap, alpha.T), axis=1)

    return colormap
