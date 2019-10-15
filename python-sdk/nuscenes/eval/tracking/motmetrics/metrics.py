"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import division
from collections import OrderedDict, Iterable
import pandas as pd
import numpy as np
import inspect
import itertools
from nuscenes.eval.tracking.motmetrics.mot import MOTAccumulator
from nuscenes.eval.tracking.motmetrics.lap import linear_sum_assignment

class MetricsHost:
    """Keeps track of metrics and intra metric dependencies."""

    def __init__(self):
        self.metrics = OrderedDict()

    def register(self, fnc, deps='auto', name=None, helpstr=None, formatter=None):
        """Register a new metric.

        Params
        ------
        fnc : Function
            Function that computes the metric to be registered. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order of the argument passed is `df, result_dep1, result_dep2, ...`.

        Kwargs
        ------
        deps : string, list of strings or None, optional
            The dependencies of this metric. Each dependency is evaluated and the result
            is passed as argument to `fnc` as described above. If None is specified, the
            function does not have any dependencies. If a list of strings is given, dependencies
            for these metric strings are registered. If 'auto' is passed, the dependencies
            are deduced from argument inspection of the method. For this to work the argument 
            names have to be equal to the intended dependencies.
        name : string or None, optional
            Name identifier of this metric. If None is passed the name is deduced from
            function inspection.
        helpstr : string or None, optional 
            A description of what the metric computes. If no help message is given it
            is deduced from the docstring of the function.
        formatter: Format object, optional
            An optional default formatter when rendering metric results as string. I.e to
            render the result `0.35` as `35%` one would pass `{:.2%}.format`
        """        

        assert not fnc is None, 'No function given for metric {}'.format(name)

        if deps is None:
            deps = []
        elif deps is 'auto':            
            deps = inspect.getargspec(fnc).args[1:] # assumes dataframe as first argument

        if name is None:
            name = fnc.__name__ # Relies on meaningful function names, i.e don't use for lambdas

        if helpstr is None:
            helpstr = inspect.getdoc(fnc) if inspect.getdoc(fnc) else 'No description.'
            helpstr = ' '.join(helpstr.split())
            
        self.metrics[name] = {
            'name' : name,
            'fnc' : fnc,
            'deps' : deps,
            'help' : helpstr,
            'formatter' : formatter
        }

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v['name'] for v in self.metrics.values()]
    
    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return dict([(k, v['formatter']) for k, v in self.metrics.items() if not v['formatter'] is None])

    def list_metrics(self, include_deps=False):
        """Returns a dataframe containing names, descriptions and optionally dependencies for each metric."""
        cols = ['Name', 'Description', 'Dependencies']
        if include_deps:
            data = [(m['name'], m['help'], m['deps']) for m in self.metrics.values()]
        else:
            data = [(m['name'], m['help']) for m in self.metrics.values()]
            cols = cols[:-1]

        return pd.DataFrame(data, columns=cols)

    def list_metrics_markdown(self, include_deps=False):
        """Returns a markdown ready version of `list_metrics`."""
        df = self.list_metrics(include_deps=include_deps)
        fmt = [':---' for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        return df_formatted.to_csv(sep="|", index=False)

    def compute(self, df, metrics=None, return_dataframe=True, return_cached=False, name=None):
        """Compute metrics on the dataframe / accumulator.
        
        Params
        ------
        df : MOTAccumulator or pandas.DataFrame
            The dataframe to compute the metrics on
        
        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        return_dataframe : bool, optional
            Return the result as pandas.DataFrame (default) or dict.
        return_cached : bool, optional
           If true all intermediate metrics required to compute the desired metrics are returned as well.
        name : string, optional
            When returning a pandas.DataFrame this is the index of the row containing
            the computed metric values.
        """ 
        
        if isinstance(df, MOTAccumulator):
            df = df.events

        if metrics is None:
            metrics = self.names
        elif isinstance(metrics, str):
            metrics = [metrics]

        class DfMap : pass
        df_map = DfMap()
        df_map.full = df     
        df_map.raw = df[df.Type == 'RAW']
        df_map.noraw = df[df.Type != 'RAW']

        cache = {}
        for mname in metrics:
            cache[mname] = self._compute(df_map, mname, cache, parent='summarize')            

        if name is None:
            name = 0 

        if return_cached:
            data = cache
        else:
            data = OrderedDict([(k, cache[k]) for k in metrics])
            
        return pd.DataFrame(data, index=[name]) if return_dataframe else data     

    def compute_many(self, dfs, metrics=None, names=None, generate_overall=False):
        """Compute metrics on multiple dataframe / accumulators.
        
        Params
        ------
        dfs : list of MOTAccumulator or list of pandas.DataFrame
            The data to compute metrics on.
        
        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        names : list of string, optional
            The names of individual rows in the resulting dataframe.
        generate_overall : boolean, optional
            If true resulting dataframe will contain a summary row that is computed
            using the same metrics over an accumulator that is the concatentation of
            all input containers. In creating this temporary accumulator, care is taken
            to offset frame indices avoid object id collisions.

        Returns
        -------
        df : pandas.DataFrame
            A datafrom containing the metrics in columns and names in rows.
        """

        assert names is None or len(names) == len(dfs)

        if names is None:
            names = range(len(dfs))

        if generate_overall:
            dfs += [MOTAccumulator.merge_event_dataframes(dfs)]
            names += ['OVERALL']

        partials = [self.compute(acc, metrics=metrics, name=name) for acc, name in zip(dfs, names)]
        return pd.concat(partials)

    def _compute(self, df_map, name, cache, parent=None):
        """Compute metric and resolve dependencies."""
        assert name in self.metrics, 'Cannot find metric {} required by {}.'.format(name, parent)        

        minfo = self.metrics[name]
        vals = []
        for depname in minfo['deps']:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute(df_map, depname, cache, parent=name)
            vals.append(v)
        return minfo['fnc'](df_map, *vals)

def num_frames(df):
    """Total number of frames."""
    return df.full.index.get_level_values(0).unique().shape[0]

def obj_frequencies(df):
    """Total number of occurrences of individual objects over all frames."""
    return df.noraw.OId.value_counts()

def pred_frequencies(df):
    """Total number of occurrences of individual predictions over all frames."""
    return df.noraw.HId.value_counts()

def num_unique_objects(df, obj_frequencies):
    """Total number of unique object ids encountered."""
    return len(obj_frequencies)

def num_matches(df):
    """Total number matches."""
    return df.noraw.Type.isin(['MATCH']).sum()

def num_switches(df):
    """Total number of track switches."""
    return df.noraw.Type.isin(['SWITCH']).sum()

def num_false_positives(df):
    """Total number of false positives (false-alarms)."""
    return df.noraw.Type.isin(['FP']).sum()

def num_misses(df):
    """Total number of misses."""
    return df.noraw.Type.isin(['MISS']).sum()

def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    return num_matches + num_switches

def num_objects(df, obj_frequencies):
    """Total number of unique object appearances over all frames."""
    return obj_frequencies.sum()

def num_predictions(df, pred_frequencies):
    """Total number of unique prediction appearances over all frames."""
    return pred_frequencies.sum()

def num_predictions(df):
    """Total number of unique prediction appearances over all frames."""
    return df.noraw.HId.count()

def track_ratios(df, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""   
    tracked = df.noraw[df.noraw.Type != 'MISS']['OId'].value_counts()
    return tracked.div(obj_frequencies).fillna(0.)

def mostly_tracked(df, track_ratios):
    """Number of objects tracked for at least 80 percent of lifespan."""
    return track_ratios[track_ratios >= 0.8].count()

def partially_tracked(df, track_ratios):
    """Number of objects tracked between 20 and 80 percent of lifespan."""
    return track_ratios[(track_ratios >= 0.2) & (track_ratios < 0.8)].count()

def mostly_lost(df, track_ratios):
    """Number of objects tracked less than 20 percent of lifespan."""
    return track_ratios[track_ratios < 0.2].count()

def num_fragmentations(df, obj_frequencies):
    """Total number of switches from tracked to not tracked."""
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df.noraw[df.noraw.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()
    return fra

def motp(df, num_detections):
    """Multiple object tracker precision."""
    return df.noraw['D'].sum() / num_detections

def mota(df, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    return 1. - (num_misses + num_switches + num_false_positives) / num_objects

def precision(df, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    return num_detections / (num_false_positives + num_detections)

def recall(df, num_detections, num_objects):
    """Number of detections over number of objects."""
    return num_detections / num_objects

def id_global_assignment(df):
    """ID measures: Global min-cost assignment for ID measures."""

    oids = df.full['OId'].dropna().unique()
    hids = df.full['HId'].dropna().unique()
    hids_idx = dict((h,i) for i,h in enumerate(hids))

    hcs = [len(df.raw[(df.raw.HId==h)].groupby(level=0)) for h in hids]
    ocs = [len(df.raw[(df.raw.OId==o)].groupby(level=0)) for o in oids]

    no = oids.shape[0]
    nh = hids.shape[0]   

    df = df.raw.reset_index()    
    df = df.set_index(['OId','HId']) 
    df = df.sort_index(level=[0,1])

    fpmatrix = np.full((no+nh, no+nh), 0.)
    fnmatrix = np.full((no+nh, no+nh), 0.)
    fpmatrix[no:, :nh] = np.nan
    fnmatrix[:no, nh:] = np.nan 

    for r, oc in enumerate(ocs):
        fnmatrix[r, :nh] = oc
        fnmatrix[r,nh+r] = oc

    for c, hc in enumerate(hcs):
        fpmatrix[:no, c] = hc
        fpmatrix[c+no,c] = hc

    for r, o in enumerate(oids):
        df_o = df.loc[o, 'D'].dropna()
        for h, ex in df_o.groupby(level=0).count().iteritems():            
            c = hids_idx[h]

            fpmatrix[r,c] -= ex
            fnmatrix[r,c] -= ex

    costs = fpmatrix + fnmatrix    
    rids, cids = linear_sum_assignment(costs)

    return {
        'fpmatrix' : fpmatrix,
        'fnmatrix' : fnmatrix,
        'rids' : rids,
        'cids' : cids,
        'costs' : costs,
        'min_cost' : costs[rids, cids].sum()
    }

def idfp(df, id_global_assignment):
    """ID measures: Number of false positive matches after global min-cost matching."""
    rids, cids = id_global_assignment['rids'], id_global_assignment['cids']
    return id_global_assignment['fpmatrix'][rids, cids].sum()

def idfn(df, id_global_assignment):
    """ID measures: Number of false negatives matches after global min-cost matching."""
    rids, cids = id_global_assignment['rids'], id_global_assignment['cids']
    return id_global_assignment['fnmatrix'][rids, cids].sum()

def idtp(df, id_global_assignment, num_objects, idfn):
    """ID measures: Number of true positives matches after global min-cost matching."""
    return num_objects - idfn

def idp(df, idtp, idfp):
    """ID measures: global min-cost precision."""
    return idtp / (idtp + idfp)

def idr(df, idtp, idfn):
    """ID measures: global min-cost recall."""
    return idtp / (idtp + idfn)

def idf1(df, idtp, num_objects, num_predictions):
    """ID measures: global min-cost F1 score."""
    return 2 * idtp / (num_objects + num_predictions)

def create():
    """Creates a MetricsHost and populates it with default metrics."""
    m = MetricsHost()

    m.register(num_frames, formatter='{:d}'.format)
    m.register(obj_frequencies, formatter='{:d}'.format)    
    m.register(pred_frequencies, formatter='{:d}'.format)
    m.register(num_matches, formatter='{:d}'.format)
    m.register(num_switches, formatter='{:d}'.format)
    m.register(num_false_positives, formatter='{:d}'.format)
    m.register(num_misses, formatter='{:d}'.format)
    m.register(num_detections, formatter='{:d}'.format)
    m.register(num_objects, formatter='{:d}'.format)
    m.register(num_predictions, formatter='{:d}'.format)
    m.register(num_unique_objects, formatter='{:d}'.format)
    m.register(track_ratios)
    m.register(mostly_tracked, formatter='{:d}'.format)
    m.register(partially_tracked, formatter='{:d}'.format)
    m.register(mostly_lost, formatter='{:d}'.format)
    m.register(num_fragmentations)
    m.register(motp, formatter='{:.3f}'.format)
    m.register(mota, formatter='{:.1%}'.format)
    m.register(precision, formatter='{:.1%}'.format)
    m.register(recall, formatter='{:.1%}'.format)
    
    m.register(id_global_assignment)
    m.register(idfp)
    m.register(idfn)
    m.register(idtp)
    m.register(idp, formatter='{:.1%}'.format)
    m.register(idr, formatter='{:.1%}'.format)
    m.register(idf1, formatter='{:.1%}'.format)


    return m

motchallenge_metrics = [
    'idf1',
    'idp',
    'idr',
    'recall', 
    'precision', 
    'num_unique_objects', 
    'mostly_tracked', 
    'partially_tracked', 
    'mostly_lost', 
    'num_false_positives', 
    'num_misses',
    'num_switches',
    'num_fragmentations',
    'mota',
    'motp'
]
"""A list of all metrics from MOTChallenge."""
