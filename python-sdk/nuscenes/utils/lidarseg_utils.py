from typing import Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import colorsys


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

    # ---------- render lidarseg labels in image ----------
    fig = plt.figure(figsize=(imsize[0] / dpi, imsize[1] / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    # ax.axis('off')
    # ---------- /render lidarseg labels in image ----------

    # ---------- convert from pyplot to cv2 ----------
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)  # put pixel buffer in numpy array
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = cv2.resize(mat, imsize)
    # ---------- /convert from pyplot to cv2 ----------

    return mat


def get_colormap() -> np.array:
    default = [255, 0, 0]

    classname_to_color = { # RGB
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
    }

    classname_scale_to_color = { # RGB
        "train": [0, 0, 255],
        "firetruck": default,
        "other_police": default,
        "driveable_surface": [255, 0, 255],
        "sidewalk": [75, 0, 75],
        "terrain_natural_surface": [150, 240, 80],
        "other_flat": [175, 0, 75],
        "man_made": default,
        "foliage_including_tree_and_bushes": [0, 175, 0],
        "other_static_object": default,
        "noise": [0, 0, 0]  # black
    }

    coloring = dict(classname_to_color.copy())
    coloring.update(classname_scale_to_color)
    # Note that if classname_scale_to_color and classname_to_color have overlapping keys, the
    # final value will be taken from classname_scale_to_color.
    print (coloring)

    colormap = []
    for k, v in coloring.items():
        colormap.append(v)

    colormap = np.array(colormap) / 255  # normalize RGB values to be between 0 and 1 for each channel

    return colormap


def get_arbitrary_colormap(num_classes) -> np.array:
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colormap = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    np.random.seed(2020)  # Fix seed for consistent colors across runs
    np.random.shuffle(colormap)  # Shuffle colors to de-correlate adjacent classes
    np.random.seed(None)  # Reset seed to default

    colormap = [(0, 0, 0)] + colormap  # TO-DO no need to add a zero class once lidarseg labels start with 0
    colormap = np.array(colormap)  # colormap is RGB with values for each channel normalized between 0 and 1

    return colormap


def filter_colormap(colormap: np.array, classes_to_display: np.array) -> np.array:
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
        if i not in classes_to_display:  # 1, 8, 31, 32, 38, 37, 40, 41:
            colormap[i] = [1.0, 1.0, 1.0]  # mask labels to be hidden with 1.0 in all channels

    # convert RGB colormap to an RGBA array, with the alpha channel set to zero
    # wherever the R, G and B channels are all equal to 1.0
    # alpha = np.array([~np.all(colormap == 1.0, axis=1) * 1.0])
    alpha = np.array([~np.all(colormap == 1.0, axis=1) * 1.0])
    colormap = np.concatenate((colormap, alpha.T), axis=1)

    return colormap
