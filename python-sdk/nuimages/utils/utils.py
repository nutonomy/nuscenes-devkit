# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

from typing import List
import base64

import numpy as np
from pycocotools import mask as cocomask


def annotation_name(attributes: List[dict],
                    category_name: str,
                    with_attributes: bool = False) -> str:
    """
    Returns the "name" of an annotation, optionally including the attributes.
    :param attributes: The attribute dictionary.
    :param category_name: Name of the object category.
    :param with_attributes: Whether to print the attributes alongside the category name.
    :return: A human readable string describing the annotation.
    """
    outstr = category_name

    if with_attributes:
        atts = [attribute['name'] for attribute in attributes]
        if len(atts) > 0:
            outstr = outstr + "--" + '.'.join(atts)

    return outstr


def mask_decode(mask: dict) -> np.ndarray:
    """
    Decode the mask from base64 string to binary string, then feed it to the external pycocotools library to get a mask.
    :param mask: The mask dictionary with fields `size` and `counts`.
    :return: A numpy array representing the binary mask for this class.
    """

    mask['counts'] = base64.b64decode(mask['counts'])
    return cocomask.decode(mask)
