"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from collections import OrderedDict
from itertools import count
from nuscenes.eval.tracking.motmetrics.lap import linear_sum_assignment

class MOTAccumulator(object):
    """Manage tracking events.
    
    This class computes per-frame tracking events from a given set of object / hypothesis 
    ids and pairwise distances. Indended usage

        import nuscenes.eval.tracking.motmetrics as mm
        acc = mm.MOTAccumulator()
        acc.update(['a', 'b'], [0, 1, 2], dists, frameid=0)
        ...
        acc.update(['d'], [6,10], other_dists, frameid=76)        
        summary = mm.metrics.summarize(acc)
        print(mm.io.render_summary(summary))

    Update is called once per frame and takes objects / hypothesis ids and a pairwise distance
    matrix between those (see distances module for support). Per frame max(len(objects), len(hypothesis)) 
    events are generated. Each event type is one of the following
        - `'MATCH'` a match between a object and hypothesis was found
        - `'SWITCH'` a match between a object and hypothesis was found but differs from previous assignment
        - `'MISS'` no match for an object was found
        - `'FP'` no match for an hypothesis was found (spurious detections)
        - `'RAW'` events corresponding to raw input
    
    Events are tracked in a pandas Dataframe. The dataframe is hierarchically indexed by (`FrameId`, `EventId`),
    where `FrameId` is either provided during the call to `update` or auto-incremented when `auto_id` is set
    true during construction of MOTAccumulator. `EventId` is auto-incremented. The dataframe has the following
    columns 
        - `Type` one of `('MATCH', 'SWITCH', 'MISS', 'FP', 'RAW')`
        - `OId` object id or np.nan when `'FP'` or `'RAW'` and object is not present
        - `HId` hypothesis id or np.nan when `'MISS'` or `'RAW'` and hypothesis is not present
        - `D` distance or np.nan when `'FP'` or `'MISS'` or `'RAW'` and either object/hypothesis is absent
    
    From the events and associated fields the entire tracking history can be recovered. Once the accumulator 
    has been populated with per-frame data use `metrics.summarize` to compute statistics. See `metrics.compute_metrics`
    for a list of metrics computed.

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene." 
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    def __init__(self, auto_id=False, max_switch_time=float('inf')):
        """Create a MOTAccumulator.

        Params
        ------
        auto_id : bool, optional
            Whether or not frame indices are auto-incremented or provided upon
            updating. Defaults to false. Not specifying a frame-id when this value
            is true results in an error. Specifying a frame-id when this value is
            false also results in an error.

        max_switch_time : scalar, optional
            Allows specifying an upper bound on the timespan an unobserved but 
            tracked object is allowed to generate track switch events. Useful if groundtruth 
            objects leaving the field of view keep their ID when they reappear, 
            but your tracker is not capable of recognizing this (resulting in 
            track switch events). The default is that there is no upper bound
            on the timespan. In units of frame timestamps. When using auto_id
            in units of count.
        """

        self.auto_id = auto_id
        self.max_switch_time = max_switch_time
        self.reset()       

    def reset(self):
        """Reset the accumulator to empty state."""

        self._events = []
        self._indices = []
        #self.events = MOTAccumulator.new_event_dataframe()
        self.m = {} # Pairings up to current timestamp  
        self.last_occurrence = {} # Tracks most recent occurance of object
        self.dirty_events = True
        self.cached_events_df = None

    def update(self, oids, hids, dists, frameid=None):
        """Updates the accumulator with frame specific objects/detections.

        This method generates events based on the following algorithm [1]:
        1. Try to carry forward already established tracks. If any paired object / hypothesis
        from previous timestamps are still visible in the current frame, create a 'MATCH' 
        event between them.
        2. For the remaining constellations minimize the total object / hypothesis distance
        error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
        match create a 'SWITCH' else a 'MATCH' event.
        3. Create 'MISS' events for all remaining unassigned objects.
        4. Create 'FP' events for all remaining unassigned hypotheses.
        
        Params
        ------
        oids : N array 
            Array of object ids.
        hids : M array 
            Array of hypothesis ids.
        dists: NxM array
            Distance matrix. np.nan values to signal do-not-pair constellations.
            See `distances` module for support methods.  

        Kwargs
        ------
        frameId : id
            Unique frame id. Optional when MOTAccumulator.auto_id is specified during
            construction.

        Returns
        -------
        frame_events : pd.DataFrame
            Dataframe containing generated events

        References
        ----------
        1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
        EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
        """
        
        self.dirty_events = True
        oids = ma.array(oids, mask=np.zeros(len(oids)))
        hids = ma.array(hids, mask=np.zeros(len(hids)))  
        dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

        if frameid is None:            
            assert self.auto_id, 'auto-id is not enabled'
            if len(self._indices) > 0:
                frameid = self._indices[-1][0] + 1
            else:
                frameid = 0
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'
        
        eid = count()

        # 0. Record raw events

        no = len(oids)
        nh = len(hids)
        
        if no * nh > 0:
            for i in range(no):
                for j in range(nh):
                    self._indices.append((frameid, next(eid)))
                    self._events.append(['RAW', oids[i], hids[j], dists[i,j]])
        elif no == 0:
            for i in range(nh):
                self._indices.append((frameid, next(eid)))
                self._events.append(['RAW', np.nan, hids[i], np.nan])       
        elif nh == 0:
            for i in range(no):
                self._indices.append((frameid, next(eid)))
                self._events.append(['RAW', oids[i], np.nan, np.nan])

        if oids.size * hids.size > 0:    
            # 1. Try to re-establish tracks from previous correspondences
            for i in range(oids.shape[0]):
                if not oids[i] in self.m:
                    continue

                hprev = self.m[oids[i]]                    
                j, = np.where(hids==hprev)  
                if j.shape[0] == 0:
                    continue
                j = j[0]

                if np.isfinite(dists[i,j]):
                    oids[i] = ma.masked
                    hids[j] = ma.masked
                    self.m[oids.data[i]] = hids.data[j]
                    
                    self._indices.append((frameid, next(eid)))
                    self._events.append(['MATCH', oids.data[i], hids.data[j], dists[i, j]])

            # 2. Try to remaining objects/hypotheses
            dists[oids.mask, :] = np.nan
            dists[:, hids.mask] = np.nan
        
            rids, cids = linear_sum_assignment(dists)

            for i, j in zip(rids, cids):                
                if not np.isfinite(dists[i,j]):
                    continue
                
                o = oids[i]
                h = hids.data[j]
                is_switch = o in self.m and \
                            self.m[o] != h and \
                            abs(frameid - self.last_occurrence[o]) <= self.max_switch_time
                cat = 'SWITCH' if is_switch else 'MATCH'
                self._indices.append((frameid, next(eid)))
                self._events.append([cat, oids.data[i], hids.data[j], dists[i, j]])
                oids[i] = ma.masked
                hids[j] = ma.masked
                self.m[o] = h

        # 3. All remaining objects are missed
        for o in oids[~oids.mask]:
            self._indices.append((frameid, next(eid)))
            self._events.append(['MISS', o, np.nan, np.nan])
        
        # 4. All remaining hypotheses are false alarms
        for h in hids[~hids.mask]:
            self._indices.append((frameid, next(eid)))
            self._events.append(['FP', np.nan, h, np.nan])

        # 5. Update occurance state
        for o in oids.data:            
            self.last_occurrence[o] = frameid

        return frameid

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulator.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df
    
    @property
    def mot_events(self):
        df = self.events
        return df[df.Type != 'RAW']

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
        idx = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['FrameId','Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH'])
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

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """Create a new DataFrame filled with data.
        
        Params
        ------
        indices: list
            list of tuples (frameid, eventid)
        events: list
            list of events where each event is a list containing
            'Type', 'OId', HId', 'D'                    
        """

        idx = pd.MultiIndex.from_tuples(indices, names=['FrameId','Event'])
        df = pd.DataFrame(events, index=idx, columns=['Type', 'OId', 'HId', 'D'])
        return df
    


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

        r = MOTAccumulator.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulator):
                df = df.events

            copy = df.copy()
            infos = {}
            
            # Update index
            if update_frame_indices:
                next_frame_id = max(r.index.get_level_values(0).max()+1, r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0]+next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:                
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map
            
            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map
            
            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:            
            return r