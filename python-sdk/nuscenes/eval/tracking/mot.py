"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on:

py-motmetrics at:
https://github.com/cheind/py-motmetrics

Notes by Michael Hoss:
For Python 3.10, we need to update the version of py-motmetrics to 1.4.0.
Then, to keep this code working, we need to change back the types of OId HId to object because they are
strings in nuscenes-devkit, whereas motmetrics changed these types to float from 1.1.3 to 1.4.0.
"""
from collections import OrderedDict
from itertools import count

import numpy as np
import pandas as pd
from motmetrics import MOTAccumulator

_INDEX_FIELDS = ['FrameId', 'Event']

class MOTAccumulatorCustom(MOTAccumulator):
    """This custom class was created by nuscenes-devkit to use a faster implementation of
    `new_event_dataframe_with_data` under compatibility with motmetrics<=1.1.3.
    Now that we use motmetrics==1.4.0, we need to use this custom implementation to use
    objects instead of strings for OId and HId.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """Create a new DataFrame filled with data.

        Params
        ------
        indices: dict
            dict of lists with fields 'FrameId' and 'Event'
        events: dict
            dict of lists with fields 'Type', 'OId', 'HId', 'D'
        """

        if len(events) == 0:
            return MOTAccumulatorCustom.new_event_dataframe()

        raw_type = pd.Categorical(
            events['Type'],
            categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'],
            ordered=False)
        series = [
            pd.Series(raw_type, name='Type'),
            pd.Series(events['OId'], dtype=object, name='OId'),  # OId is string in nuscenes-devkit
            pd.Series(events['HId'], dtype=object, name='HId'),  # HId is string in nuscenes-devkit
            pd.Series(events['D'], dtype=float, name='D')
        ]

        idx = pd.MultiIndex.from_arrays(
            # TODO What types are the indices FrameId and Event? string or int?
            [indices[field] for field in _INDEX_FIELDS],
            names=_INDEX_FIELDS)
        df = pd.concat(series, axis=1)
        df.index = idx
        return df

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),          # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=object)),      # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=object)),      # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),      # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True, return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = count()
        new_hid = count()

        r = MOTAccumulatorCustom.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulatorCustom):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                # pylint: disable=cell-var-from-loop
                # TODO TypeError: can only concatenate tuple (not "int") to tuple
                # This is likely because we have a multi-index dataframe r here (see new_event_dataframe())
                # See also https://stackoverflow.com/questions/39080555/pandas-get-level-values-for-multiple-columns
                # Playground code: https://onecompiler.com/python/422kn8tev
                """

                import pandas as pd

                a={"gk":[15,12,13,22,32,12],"mk":[12,21,23,22,56,12], "sf": [1,2,3,4,5,5]}
                df=pd.DataFrame(a)

                # B=df[df["mk"]>=21]

                # print(df)
                # print(B)

                df = df.set_index(["gk", "sf"])

                print(df)

                print("Experiment")
                print(df.index.get_level_values(1))
                print("First argument of max")
                print(df.index.get_level_values(0).max())
                print(df.index.get_level_values(0).max() +1) # the maximum value of the 0th index column incremented by 1
                print(df.index.get_level_values(1).max())
                print(df.index.get_level_values(1).max() +1)
                print("Second argument of max")
                print(df.index.get_level_values(0))
                print(df.index.get_level_values(0).unique())
                print(df.index.get_level_values(0).unique().shape)
                print(df.index.get_level_values(0).unique().shape[0])  # number of unique values in the 0th index column
                print("Final max evaluation")
                print(max(df.index.get_level_values(0).max() +1,df.index.get_level_values(0).unique().shape[0]))
                """

                next_frame_id = max(r.index.get_level_values(0).max() + 1, r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                # pylint: disable=cell-var-from-loop
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                # pylint: disable=cell-var-from-loop
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            # Avoid pandas warning. But is this legit/do we need such a column later on again?
            # copy = copy.dropna(axis=1, how='all')
            r = pd.concat((r, copy))
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r
