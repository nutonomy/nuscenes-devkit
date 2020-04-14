import numpy as np
import colorsys


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
