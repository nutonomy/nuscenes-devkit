import os
from typing import Tuple, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Axis = Any


class BitMap:

    def __init__(self, dataroot: str, map_name: str, layer_name: str):
        """
        This class is used to render bitmap map layers. Currently these are:
        - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
        - basemap: The HD lidar basemap used for localization and as general context.

        :param dataroot: Path of the nuScenes dataset.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
            'boston-seaport'.
        :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
        """
        self.dataroot = dataroot
        self.map_name = map_name
        self.layer_name = layer_name

        self.image = self.load_bitmap()

    def load_bitmap(self) -> np.ndarray:
        """
        Load the specified bitmap.
        """
        # Load bitmap.
        if self.layer_name == 'basemap':
            map_path = os.path.join(self.dataroot, 'maps', 'basemap', self.map_name + '.png')
        elif self.layer_name == 'semantic_prior':
            map_hashes = {
                'singapore-onenorth': '53992ee3023e5494b90c316c183be829',
                'singapore-hollandvillage': '37819e65e09e5547b8a3ceaefba56bb2',
                'singapore-queenstown': '93406b464a165eaba6d9de76ca09f5da',
                'boston-seaport': '36092f0b03a857c6a3403e25b4b7aab3'
            }
            map_hash = map_hashes[self.map_name]
            map_path = os.path.join(self.dataroot, 'maps', map_hash + '.png')
        else:
            raise Exception('Error: Invalid bitmap layer: %s' % self.layer_name)

        # Convert to numpy.
        if os.path.exists(map_path):
            image = np.array(Image.open(map_path))
        else:
            raise Exception('Error: Cannot find %s %s! Please make sure that the map is correctly installed.'
                            % (self.layer_name, map_path))

        # Invert semantic prior colors.
        if self.layer_name == 'semantic_prior':
            image = image.max() - image

        return image

    def render(self, canvas_edge: Tuple[float, float], ax: Axis = None):
        """
        Render the bitmap.
        Note: Regardless of the image dimensions, the image will be rendered to occupy the entire map.
        :param canvas_edge: The dimension of the current map in meters (width, height).
        :param ax: Optional axis to render to.
        """
        if ax is None:
            ax = plt.subplot()
        x, y = canvas_edge
        if len(self.image.shape) == 2:
            ax.imshow(self.image, extent=[0, x, 0, y], cmap='gray')
        else:
            ax.imshow(self.image, extent=[0, x, 0, y])
