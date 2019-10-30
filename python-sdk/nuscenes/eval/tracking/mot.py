"""
This code is based on:

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import motmetrics
import pandas as pd


class MOTAccumulatorCustom(motmetrics.mot.MOTAccumulator):
    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """
        Create a new DataFrame filled with data.
        This version overwrites the original in MOTAccumulator achieves about 2x speedups.

        Params
        ------
        indices: list
            list of tuples (frameid, eventid)
        events: list
            list of events where each event is a list containing
            'Type', 'OId', HId', 'D'
        """

        idx = pd.MultiIndex.from_tuples(indices, names=['FrameId', 'Event'])
        df = pd.DataFrame(events, index=idx, columns=['Type', 'OId', 'HId', 'D'])
        return df

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df
