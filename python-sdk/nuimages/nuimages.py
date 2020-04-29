# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.
import sys
import os.path as osp
import json
import time
from typing import Any, List, Dict

import PIL
import PIL.ImageDraw
import PIL.ImageFont
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from nuimages.utils import default_color, annotation_name, mask_decode

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")


class NuImages:
    """
    Database class for nuImages to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v0.1',  # TODO: Add split
                 dataroot: str = '/data/sets/nuimages',
                 lazy: bool = True,
                 verbose: bool = True):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v0.1", ...).
        :param dataroot: Path to the tables and data.
        :param lazy: Whether to use lazy loading for the database tables.
        :param verbose: Whether to print status messages during load.
        """
        self.version = version
        self.dataroot = dataroot
        self.lazy = lazy
        self.verbose = verbose

        self.table_names = ['attribute', 'camera', 'category', 'image', 'log', 'object_ann', 'surface_ann']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Init reverse indexing.
        self._token2ind: Dict[str, dict] = dict()
        for table in self.table_names:
            self._token2ind[table] = None

        # Load tables directly if requested.
        if not lazy:
            # Explicitly init tables to help the IDE determine valid class members.
            self.attribute = self.__load_table__('attribute')
            self.camera = self.__load_table__('camera')
            self.category = self.__load_table__('category')
            self.image = self.__load_table__('image')
            self.log = self.__load_table__('log')
            self.object_ann = self.__load_table__('object_ann')
            self.surface_ann = self.__load_table__('surface_ann')

        # Initialize NuImagesExplorer class.
        self.explorer = NuImagesExplorer(self)

    def __getattr__(self, attr_name: str) -> Any:
        """
        Implement lazy loading for the database tables. Otherwise throw the default error.
        :param attr_name: The name of the variable to look for.
        :return: The dictionary that represents that table.
        """
        if attr_name in self.table_names:
            value = self.__load_table__(attr_name)
            self.__setattr__(attr_name, value)
            return self.__getattribute__(attr_name)
        else:
            raise AttributeError("Error: %r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        # Lazy loading: Compute reverse indices.
        if self._token2ind[table_name] is None:
            self._token2ind[table_name] = dict()
            for ind, member in enumerate(getattr(self, table_name)):
                self._token2ind[table_name][member['token']] = ind

        return self._token2ind[table_name][token]

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        start_time = time.time()
        table_path = osp.join(self.table_root, '{}.json'.format(table_name))
        assert osp.exists(table_path), 'Error: Table %s does not exist!' % table_name
        with open(table_path) as f:
            table = json.load(f)
        end_time = time.time()

        # Print a message to stdout.
        if self.verbose:
            print("Loaded {} {}(s) in {:.3f}s,".format(len(table), table_name, end_time - start_time))

        return table


class NuImagesExplorer:
    """
    Helper class to list and visualize NuImages data.
    These are meant to serve as tutorials and templates for working with the data.
    """

    def __init__(self, nuim: NuImages):
        self.nuim = nuim

    def list_attributes(self) -> None:
        pass # TODO

    def list_cameras(self) -> None:
        pass # TODO

    def list_categories(self) -> None:
        pass # TODO

    def list_images(self) -> None:
        pass # TODO

    def list_image(self, image_token: str) -> None:
        pass # TODO

    def list_logs(self) -> None:
        pass # TODO

    def render_image(self,
               image_token: str,
               with_annotations: bool = True,
               with_attributes: bool = False,
               box_tokens: List[str] = None,
               surface_tokens: List[str] = None,
               ax: Axes = None) -> PIL.Image:
        """
        Draws an image with annotations overlaid.
        :param with_annotations: Whether to draw all annotations.
        :param with_attributes: Whether to include attributes in the label tags.
        :param box_tokens: List of bounding box annotation tokens. If given only these annotations are drawn.
        :param surface_tokens: List of surface annotation tokens. If given only these annotations are drawn.
        :param ax: The matplotlib axes where the layer will get rendered or None to create new axes.
        :return: Image object.
        """
        # Get image data.
        image = self.nuim.get('image', image_token)
        im_path = osp.join(self.nuim.dataroot, image['filename_jpg'])
        im = PIL.Image.open(im_path)
        if not with_annotations:
            return im

        # Initialize drawing.
        #try:
        #    font = PIL.ImageFont.truetype('Ubuntu-B.ttf', 15)
        #except OSError:
        font = PIL.ImageFont.load_default()
        draw = PIL.ImageDraw.Draw(im, 'RGBA')

        # Load object instances.
        object_anns = [o for o in self.nuim.object_ann if o['image_token'] == image_token]
        if box_tokens is not None:
            object_anns = [o for o in object_anns if o['token'] in box_tokens]

        # Draw object instances.
        for ann in object_anns:
            # Get color, box, mask and name.
            category_token = ann['category_token']
            category_name = self.nuim.get('category', category_token)
            color = default_color(category_name)
            bbox = ann['bbox']
            mask = mask_decode(ann['mask'])
            name = annotation_name(self.nuim.attribute, category_name, with_attributes=with_attributes)

            # Draw rectangle, text and mask.
            draw.rectangle(bbox, outline=color)
            draw.text((bbox[0], bbox[1]), name, font=font)
            if mask:
                draw.bitmap((0, 0), PIL.Image.fromarray(mask * 128), fill=tuple(color + (128,)))

        # Load stuff / background regions.
        surface_anns = [o for o in self.nuim.surface_ann if o['image_token'] == image_token]
        if surface_tokens is not None:
            surface_anns = [o for o in object_anns if o['token'] in surface_tokens]

        # Draw stuff / background regions.
        for ann in surface_anns:
            # Get color and mask
            category_token = ann['category_token']
            category_name = self.nuim.get('category', category_token)
            color = default_color(category_name)
            mask = mask_decode(ann['mask'])

            if mask:
                draw.bitmap((0, 0), PIL.Image.fromarray(mask * 128), fill=tuple(color + (128,)))

        # Plot the image
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))
        ax.imshow(im)
        (width, height) = im.size
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(image_token)

        return im
