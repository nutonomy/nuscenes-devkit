from typing import Tuple
import base64

import numpy as np
from pycocotools import mask as cocomask


def default_color(category_name: str) -> Tuple[int, int, int]: # TODO: updated these
    """ Provides a default colors based on the category names. """
    if 'cycle' in category_name:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name:
        return 255, 158, 0  # Orange
    elif 'human.pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    # These rest are the same in Scale's annotation tool.
    elif category_name == 'flat.driveable_surface':
        return 255, 145, 0
    elif category_name == 'flat':
        return 0, 230, 118
    elif category_name == 'vehicle.ego':
        return 213, 0, 249
    else:
        return 255, 0, 255  # Magenta


def annotation_name(attributes: dict,
                    category_name: str,
                    with_attributes: bool = False) -> str:
    """
    Returns the "name" of an annotation, including or not attribute states.
    :param attributes: The attribute dictionary.
    :param with_attributes: Whether to include annotation's attribute states in output string.
    :return: A human readable string describing the annotation.
    """
    outstr = category_name

    if with_attributes:
        attstate = [attribute['name'] for attribute in attributes]
        if len(attstate) > 0:
            outstr = outstr + "--" + '.'.join(attstate)

    return outstr


def mask_decode(mask: dict) -> np.ndarray:
    mask['counts'] = base64.b64decode(mask['counts'])
    return cocomask.decode(mask)
