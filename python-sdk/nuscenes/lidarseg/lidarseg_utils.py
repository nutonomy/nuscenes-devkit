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
    :return: An array which contains the counts of each label in the point cloud. The index of the point cloud
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
    scatter plots.
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

    ax.axis('off')
    ax.margins(0, 0)

    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)

    # Convert from pyplot to cv2.
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer.buffer_rgba()).astype('uint8')  # Put pixel buffer in numpy array.
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = cv2.resize(mat, imsize)

    return mat


def get_colormap() -> np.ndarray:
    default = [255, 0, 0]

    classname_to_color = {  # RGB.
        "noise": [0, 0, 0],  # Black.
        "human.pedestrian.adult": [255, 30, 30],
        "human.pedestrian.child": [220, 20, 60],  # Crimson
        "human.pedestrian.wheelchair": [233, 150, 122],  # Darksalmon
        "human.pedestrian.stroller": [240, 128, 128],  # Lightcoral
        "human.pedestrian.personal_mobility": [219, 112, 147],  # Palevioletred
        "human.pedestrian.police_officer": [255, 204, 0],
        "human.pedestrian.construction_worker": [255, 165, 0],
        "animal": [255, 99, 71],  # Tomato
        "vehicle.car": [100, 150, 245],
        "vehicle.motorcycle": [30, 60, 150],
        "vehicle.bicycle": [100, 230, 245],
        "vehicle.bus.bendy": [70, 130, 180],  # Steelblue
        "vehicle.bus.rigid": [100, 149, 237],  # Cornflowerblue
        "vehicle.truck": [80, 30, 180],
        "vehicle.construction": [138, 43, 226],  # Blueviolet
        "vehicle.emergency.ambulance": [0, 0, 128],  # Navy
        "vehicle.emergency.police": [0, 0, 255],  # Blue
        "vehicle.trailer": [135, 206, 235],  # Skyblue
        "movable_object.barrier": [165, 42, 42],  # Brown
        "movable_object.trafficcone": [160, 82, 45],  # Sienna
        "movable_object.pushable_pullable": [139, 69, 19],  # Saddlebrown
        "movable_object.debris": [210, 105, 30],  # Chocolate
        "static_object.bicycle_rack": [188,143,143],  # Rosybrown
        "vehicle.on_rail": [0, 0, 255],  # Blue  TODO: remove
        "vehicle.emergency.firetruck": default,  # TODO: remove
        "flat.driveable_surface": [128, 64, 128],  # [255, 0, 255],
        "flat.sidewalk": [75, 0, 75],
        "flat.terrain": [112, 180, 60],  # [150, 240, 80],
        "flat.other": [175, 0, 75],
        "static.manmade": [222, 184, 135],  # Burlywood
        "static.vegetation": [0, 175, 0],
        "static.other": [255, 228, 196]  # Bisque
    }

    coloring = dict(classname_to_color.copy())

    colormap = []
    for k, v in coloring.items():
        colormap.append(v)

    colormap = np.array(colormap) / 255  # Normalize RGB values to be between 0 and 1 for each channel.

    return colormap


def get_arbitrary_colormap(num_classes: int, random_seed: int = 93) -> np.ndarray:
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
    :return: (colormap <np.float: n, 4)>).

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


def get_labels_in_coloring(color_legend: np.ndarray, coloring: np.ndarray) -> List[int]:
    """
    Find the class labels which are present in a pointcloud which has been projected onto an image.
    :param color_legend: A list of arrays in which each array corresponds to the RGB values of a class.
    :param coloring: A list of arrays in which each array corresponds to the RGB values of a point in the portion of
                     the pointcloud projected onto the image.
    :return: List of class indices which are present in the image.
    """

    def _array_in_list(arr: List, list_arrays: List) -> bool:
        """
        Check if an array is in a list of arrays.
        :param: arr: An array.
        :param: list_arrays: A list of arrays.
        :return: Whether the given array is in the list of arrays.
        """
        # Credits: https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        return next((True for elem in list_arrays if np.array_equal(elem, arr)), False)

    filter_lidarseg_labels = []

    # Get only the distinct colors present in the pointcloud so that we will not need to compare each color in
    # the color legend with every single point in the pointcloud later.
    distinct_colors = list(set(tuple(c) for c in coloring))

    for i, color in enumerate(color_legend):
        if _array_in_list(color, distinct_colors):
            filter_lidarseg_labels.append(i)

    return filter_lidarseg_labels
