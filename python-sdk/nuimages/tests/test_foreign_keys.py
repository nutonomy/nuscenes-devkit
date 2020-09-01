# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import itertools
import os
import unittest
from collections import defaultdict
from typing import List, Dict, Any

from nuimages.nuimages import NuImages


class TestForeignKeys(unittest.TestCase):
    def __init__(self, _: Any = None, version: str = 'v1.0-mini', dataroot: str = None):
        """
        Initialize TestForeignKeys.
        Note: The second parameter is a dummy parameter required by the TestCase class.
        :param version: The NuImages version.
        :param dataroot: The root folder where the dataset is installed.
        """
        super().__init__()

        self.version = version
        if dataroot is None:
            self.dataroot = os.environ['NUIMAGES']
        else:
            self.dataroot = dataroot
        self.nuim = NuImages(version=self.version, dataroot=self.dataroot, verbose=False)

    def runTest(self) -> None:
        """
        Dummy function required by the TestCase class.
        """
        pass

    def test_foreign_keys(self) -> None:
        """
        Test that every foreign key points to a valid token.
        """
        # Index the tokens of all tables.
        index = dict()
        for table_name in self.nuim.table_names:
            print('Indexing table %s...' % table_name)
            table: list = self.nuim.__getattr__(table_name)
            tokens = [row['token'] for row in table]
            index[table_name] = set(tokens)

        # Go through each table and check the foreign_keys.
        for table_name in self.nuim.table_names:
            table: List[Dict[str, Any]] = self.nuim.__getattr__(table_name)
            if self.version.endswith('-test') and len(table) == 0:  # Skip test annotations.
                continue
            keys = table[0].keys()

            # Check 1-to-1 link.
            one_to_one_names = [k for k in keys if k.endswith('_token') and not k.startswith('key_')]
            for foreign_key_name in one_to_one_names:
                print('Checking one-to-one key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = foreign_key_name.replace('_token', '')
                foreign_tokens = set([row[foreign_key_name] for row in table])

                # Check all tokens are valid.
                if self.version.endswith('-mini') and foreign_table_name == 'category':
                    continue  # Mini does not cover all categories.
                foreign_index = index[foreign_table_name]
                self.assertTrue(foreign_tokens.issubset(foreign_index))

                # Check all tokens are covered.
                # By default we check that all tokens are covered. Exceptions are listed below.
                if table_name == 'object_ann':
                    if foreign_table_name == 'category':
                        remove = set([cat['token'] for cat in self.nuim.category if cat['name']
                                      in ['vehicle.ego', 'flat.driveable_surface']])
                        foreign_index = foreign_index.difference(remove)
                    elif foreign_table_name == 'sample_data':
                        foreign_index = None  # Skip as sample_datas may have no object_ann.
                elif table_name == 'surface_ann':
                    if foreign_table_name == 'category':
                        remove = set([cat['token'] for cat in self.nuim.category if cat['name']
                                      not in ['vehicle.ego', 'flat.driveable_surface']])
                        foreign_index = foreign_index.difference(remove)
                    elif foreign_table_name == 'sample_data':
                        foreign_index = None  # Skip as sample_datas may have no surface_ann.
                if foreign_index is not None:
                    self.assertEqual(foreign_tokens, foreign_index)

            # Check 1-to-many link.
            one_to_many_names = [k for k in keys if k.endswith('_tokens')]
            for foreign_key_name in one_to_many_names:
                print('Checking one-to-many key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = foreign_key_name.replace('_tokens', '')
                foreign_tokens_nested = [row[foreign_key_name] for row in table]
                foreign_tokens = set(itertools.chain(*foreign_tokens_nested))

                # Check that all tokens are valid.
                foreign_index = index[foreign_table_name]
                self.assertTrue(foreign_tokens.issubset(foreign_index))

                # Check all tokens are covered.
                if self.version.endswith('-mini') and foreign_table_name == 'attribute':
                    continue  # Mini does not cover all categories.
                if foreign_index is not None:
                    self.assertEqual(foreign_tokens, foreign_index)

            # Check prev and next.
            prev_next_names = [k for k in keys if k in ['previous', 'next']]
            for foreign_key_name in prev_next_names:
                print('Checking prev-next key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = table_name
                foreign_tokens = set([row[foreign_key_name] for row in table if len(row[foreign_key_name]) > 0])

                # Check that all tokens are valid.
                foreign_index = index[foreign_table_name]
                self.assertTrue(foreign_tokens.issubset(foreign_index))

    def test_prev_next(self) -> None:
        """
        Test that the prev and next points in sample_data cover all entries and have the correct ordering.
        """
        # Register all sample_datas.
        sample_to_sample_datas = defaultdict(lambda: [])
        for sample_data in self.nuim.sample_data:
            sample_to_sample_datas[sample_data['sample_token']].append(sample_data['token'])

        print('Checking prev-next pointers for completeness and correct ordering...')
        for sample in self.nuim.sample:
            # Compare the above sample_datas against those retrieved by using prev and next pointers.
            sd_tokens_pointers = self.nuim.get_sample_content(sample['token'])
            sd_tokens_all = sample_to_sample_datas[sample['token']]
            self.assertTrue(set(sd_tokens_pointers) == set(sd_tokens_all),
                            'Error: Inconsistency in prev/next pointers!')

            timestamps = []
            for sd_token in sd_tokens_pointers:
                sample_data = self.nuim.get('sample_data', sd_token)
                timestamps.append(sample_data['timestamp'])
            self.assertTrue(sorted(timestamps) == timestamps, 'Error: Timestamps not properly sorted!')


if __name__ == '__main__':
    # Runs the tests without aborting on error.
    for nuim_version in ['v1.0-train', 'v1.0-val', 'v1.0-test', 'v1.0-mini']:
        print('Running TestForeignKeys for version %s...' % nuim_version)
        test = TestForeignKeys(version=nuim_version)
        test.test_foreign_keys()
        test.test_prev_next()
        print()
