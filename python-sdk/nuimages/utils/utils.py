# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

import base64
import os
from typing import List
import warnings

import matplotlib.font_manager
from PIL import ImageFont
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
    # Note that it is essential to copy the mask here. If we use the same variable we will overwrite the NuImage class
    # and cause the Jupyter Notebook to crash on some systems.
    new_mask = mask.copy()
    new_mask['counts'] = base64.b64decode(mask['counts'])
    return cocomask.decode(new_mask)


def get_font(fonts_valid: List[str] = None, font_size: int = 15) -> ImageFont:
    """
    Check if there is a desired font present in the user's system. If there is, use that font; otherwise, use a default
    font.
    :param fonts_valid: A list of fonts which are desirable.
    :param font_size: The size of the font to set. Note that if the default font is used, then the font size
        cannot be set.
    :return: An ImageFont object to use as the font in a PIL image.
    """
    # If there are no desired fonts supplied, use a hardcoded list of fonts which are desirable.
    if fonts_valid is None:
        fonts_valid = ['FreeSerif.ttf', 'FreeSans.ttf', 'Century.ttf', 'Calibri.ttf', 'arial.ttf']

    # Find a list of fonts within the user's system.
    fonts_in_sys = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # Sort the list of fonts to ensure that the desired fonts are always found in the same order.
    fonts_in_sys = sorted(fonts_in_sys)
    # Of all the fonts found in the user's system, check if any of them are desired.
    for font_in_sys in fonts_in_sys:
        if any(os.path.basename(font_in_sys) in s for s in fonts_valid):
            return ImageFont.truetype(font_in_sys, font_size)

    # If none of the fonts in the user's system are desirable, then use the default font.
    warnings.warn('No suitable fonts were found in your system. '
                  'A default font will be used instead (the font size will not be adjustable).')
    return ImageFont.load_default()
