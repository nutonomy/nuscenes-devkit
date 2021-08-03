# nuScenes dev-kit.
# Code written by Fong Whye Kit, 2020.

from typing import Dict, Iterable, List, Tuple

import cv2
from matplotlib import axes
import matplotlib.patches as mpatches
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


def create_lidarseg_legend(labels_to_include_in_legend: List[int],
                           idx2name: Dict[int, str],
                           name2color: Dict[str, Tuple[int, int, int]],
                           ax: axes.Axes = None,
                           loc: str = 'upper center',
                           ncol: int = 3,
                           bbox_to_anchor: Tuple = None) -> None:
    """
    Given a list of class indices, the mapping from class index to class name, and the mapping from class name
    to class color, produce a legend which shows the color and the corresponding class name.
    :param labels_to_include_in_legend: Labels to show in the legend.
    :param idx2name: The mapping from class index to class name.
    :param name2color: The mapping from class name to class color.
    :param ax: Axes onto which to render.
    :param loc: The location of the legend.
    :param ncol: The number of columns that the legend has.
    :param bbox_to_anchor: A 2-tuple (x, y) which places the top-left corner of the legend specified by loc
                           at x, y. The origin is at the bottom-left corner and x and y are normalized between
                           0 and 1 (i.e. x > 1 and / or y > 1 will place the legend outside the plot.
    """

    recs = []
    classes_final = []
    classes = [name for idx, name in sorted(idx2name.items())]

    for i in range(len(classes)):
        if labels_to_include_in_legend is None or i in labels_to_include_in_legend:
            name = classes[i]
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=np.array(name2color[name]) / 255))

            # Truncate class names to only first 25 chars so that legend is not excessively long.
            classes_final.append(classes[i][:25])
    if ax is not None:
        ax.legend(recs, classes_final, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)
    else:
        plt.legend(recs, classes_final, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)


def paint_points_label(lidarseg_labels_filename: str, filter_lidarseg_labels: List[int],
                       name2idx: Dict[str, int], colormap: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Paint each label in a pointcloud with the corresponding RGB value, and set the opacity of the labels to
    be shown to 1 (the opacity of the rest will be set to 0); e.g.:
        [30, 5, 12, 34, ...] ------> [[R30, G30, B30, 0], [R5, G5, B5, 1], [R34, G34, B34, 1], ...]
    :param lidarseg_labels_filename: Path to the .bin file containing the labels.
    :param filter_lidarseg_labels: The labels for which to set opacity to zero; this is to hide those points
                                   thereby preventing them from being displayed.
    :param name2idx: A dictionary containing the mapping from class names to class indices.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
    """

    # Load labels from .bin file.
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]

    # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
    # get an array of RGB values where each color sits at the index in the array corresponding
    # to the class index.
    colors = colormap_to_colors(colormap, name2idx)  # Shape: [num_class, 3]

    if filter_lidarseg_labels is not None:
        # Ensure that filter_lidarseg_labels is an iterable.
        assert isinstance(filter_lidarseg_labels, (list, np.ndarray)), \
            'Error: filter_lidarseg_labels should be a list of class indices, eg. [9], [10, 21].'

        # Check that class indices in filter_lidarseg_labels are valid.
        assert all([0 <= x < len(name2idx) for x in filter_lidarseg_labels]), \
            'All class indices in filter_lidarseg_labels should ' \
            'be between 0 and {}'.format(len(name2idx) - 1)

        # Filter to get only the colors of the desired classes; this is done by setting the
        # alpha channel of the classes to be viewed to 1, and the rest to 0.
        colors = filter_colors(colors, filter_lidarseg_labels)  # Shape: [num_class, 4]

    # Paint each label with its respective RGBA value.
    coloring = colors[points_label]  # Shape: [num_points, 4]

    return coloring
