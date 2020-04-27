# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.
import sys
import os.path as osp
import json
import time
from typing import Any

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
        :param version: Version to load (e.g. "v1.0", ...).
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

        # Explicitly assign tables to help the IDE determine valid class members.
        if not lazy:
            self.attribute = self.__load_table__('attribute')
            self.camera = self.__load_table__('camera')
            self.category = self.__load_table__('category')
            self.image = self.__load_table__('image')
            self.log = self.__load_table__('log')
            self.object_ann = self.__load_table__('object_ann')
            self.surface_ann = self.__load_table__('surface_ann')

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

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        start_time = time.time()
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        end_time = time.time()

        # Print a message to stdout
        if self.verbose:
            print("Loaded {} {}(s) in {:.3f}s,".format(len(table), table_name, end_time - start_time))

        return table
