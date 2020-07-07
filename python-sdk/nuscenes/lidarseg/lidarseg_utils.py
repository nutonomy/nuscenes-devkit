# nuScenes dev-kit.
# Code written by Fong Whye Kit, 2020.

import colorsys
from typing import Dict, Iterable, List, Tuple

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

    # Clear off the current figure to prevent an accumulation of figures in memory.
    plt.close('all')

    return mat


def colormap_to_colors(colormap: Dict[str, Iterable[int]], name2idx: Dict[str, int]) -> np.ndarray:
    """
    Create an array of RGB values from a colormap. Note that the RGB values are normalized
    between 0 and 1, not 0 and 255.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :param name2idx: A dictionary containing the mapping form class names to class index.
    :return: An array of colors.
    """
    colors = []
    for i, (k, v) in enumerate(colormap.items()):
        # Ensure that the indices from the colormap is same as the class indices.
        assert i == name2idx[k], 'Error: {} is of index {}, ' \
                                 'but it is of index {} in the colormap.'.format(k, name2idx[k], i)
        colors.append(v)

    colors = np.array(colors) / 255  # Normalize RGB values to be between 0 and 1 for each channel.

    return colors


def filter_colors(colors: np.array, classes_to_display: np.array) -> np.ndarray:
    """
    Given an array of RGB colors and a list of classes to display, return a colormap (in RGBA) with the opacity
    of the labels to be display set to 1.0 and those to be hidden set to 0.0
    :param colors: [n x 3] array where each row consist of the RGB values for the corresponding class index
    :param classes_to_display: An array of classes to display (e.g. [1, 8, 32]). The array need not be ordered.
    :return: (colormap <np.float: n, 4)>).

    colormap = np.array([[R1, G1, B1],             colormap = np.array([[1.0, 1.0, 1.0, 0.0],
                         [R2, G2, B2],   ------>                        [R2,  G2,  B2,  1.0],
                         ...,                                           ...,
                         Rn, Gn, Bn]])                                  [1.0, 1.0, 1.0, 0.0]])
    """
    for i in range(len(colors)):
        if i not in classes_to_display:
            colors[i] = [1.0, 1.0, 1.0]  # Mask labels to be hidden with 1.0 in all channels.

    # Convert the RGB colormap to an RGBA array, with the alpha channel set to zero whenever the R, G and B channels
    # are all equal to 1.0.
    alpha = np.array([~np.all(colors == 1.0, axis=1) * 1.0])
    colors = np.concatenate((colors, alpha.T), axis=1)

    return colors


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
