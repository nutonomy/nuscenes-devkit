import unittest
import os
from typing import List, Dict, Any
import itertools

from nuimages.nuimages import NuImages


class TestForeignKeys(unittest.TestCase):

    def setUp(self):
        self.nuim = NuImages(version='v1.0-val', dataroot=os.environ['NUIMAGES'], verbose=False)

    @unittest.skip
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
            keys = table[0].keys()

            # Check 1-to-1 link.
            one_to_one_names = [k for k in keys if k.endswith('_token')]
            for foreign_key_name in one_to_one_names:
                print('Checking one-to-one key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = foreign_key_name.replace('_token', '')
                foreign_tokens = set([row[foreign_key_name] for row in table])
                self.assertTrue(foreign_tokens.issubset(index[foreign_table_name]))

            # Check 1-to-many link.
            one_to_many_names = [k for k in keys if k.endswith('_tokens')]
            for foreign_key_name in one_to_many_names:
                print('Checking one-to-many key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = foreign_key_name.replace('_tokens', '')
                foreign_tokens_nested = [row[foreign_key_name] for row in table]
                foreign_tokens = set(itertools.chain(*foreign_tokens_nested))
                self.assertTrue(foreign_tokens.issubset(index[foreign_table_name]))

            # Check prev and next.
            prev_next_names = [k for k in keys if k in ['previous', 'next']]
            for foreign_key_name in prev_next_names:
                print('Checking prev-next key %s in table %s...' % (foreign_key_name, table_name))
                foreign_table_name = table_name
                foreign_tokens = set([row[foreign_key_name] for row in table if len(row[foreign_key_name]) > 0])
                self.assertTrue(foreign_tokens.issubset(index[foreign_table_name]))


if __name__ == '__main__':
    unittest.main()
