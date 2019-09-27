"""
This code is based on Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py
"""
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes


class tData:
    """
        Utility class to load data.
    """

    def __init__(self, frame=-1, obj_type="unset", truncation=-1, occlusion=-1,
                 w=-1, h=-1, l=-1,
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """

        # Redundant
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score

        # Delete
        self.ignored = False
        self.valid = False
        self.truncation = truncation
        self.occlusion = occlusion

    def __str__(self):
        """
            Print read data.
        """

        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


class TrackingEvaluation(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	        - Multi-object tracking accuracy in [0,100]
             MOTP	        - Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	        - Multi-object tracking accuracy in [0,100] with log10(id-switches)
             id-switches    - number of id switches
             fragments      - number of fragmentations
             MT, PT, ML	    - number of mostly tracked, partially tracked and mostly lost trajectories
             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """

    def __init__(self,
                 nusc: NuScenes,
                 gt_boxes: EvalBoxes,
                 pred_boxes: EvalBoxes,
                 class_name: str,
                 mail,
                 num_sample_pts: int = 11):

        self.nusc = nusc
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.cls = class_name
        self.mail = mail
        self.num_sample_pts = num_sample_pts

        self.n_sequences = len(self.gt_boxes)

        # statistics and numbers for evaluation
        self.n_gt = 0  # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # number of ignored ground truth detections
        self.n_gts = []  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = []  # number of ground ignored truth detections PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_seq = []
        self.n_tr = 0  # number of tracker detections minus ignored tracker detections
        self.n_trs = []  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itr = 0  # number of ignored tracker detections
        self.n_itrs = []  # number of ignored tracker detections PER SEQUENCE
        self.n_igttr = 0  # number of ignored ground truth detections where the corresponding associated tracker detection is also ignored
        self.n_tr_trajectories = 0
        self.n_tr_seq = []
        self.MOTA = 0
        self.MOTP = 0
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0
        self.total_cost = 0
        self.itp = 0  # number of ignored true positives
        self.itps = []  # number of ignored true positives PER SEQUENCE
        self.tp = 0  # number of true positives including ignored true positives!
        self.tps = []  # number of true positives including ignored true positives PER SEQUENCE
        self.fn = 0  # number of false negatives WITHOUT ignored false negatives
        self.fns = []  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifn = 0  # number of ignored false negatives
        self.ifns = []  # number of ignored false negatives PER SEQUENCE
        self.fp = 0  # number of false positives
        # a bit tricky, the number of ignored false negatives and ignored true positives
        # is subtracted, but if both tracker detection and ground truth detection
        # are ignored this number is added again to avoid double counting
        self.fps = []  # above PER SEQUENCE
        self.mme = 0
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.n_sample_points = 500

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

        self.tracks_gt = self.create_tracks(self.gt_boxes)
        self.tracks_pred = self.create_tracks(self.pred_boxes)

    def create_tracks(self, all_boxes) -> Dict[str, Dict[str, TrackingBox]]:
        """
        Returns all tracks for all scenes. This can be applied either to GT or predictions.
        :return: The tracks.
        """

        # Group annotations wrt scene and track_id.
        tracks = {}
        for sample_token in all_boxes.sample_tokens:

            # Init scene.
            sample_record = self.nusc.get('sample', sample_token)
            scene_token = sample_record['scene_token']
            tracks[scene_token] = {}

            boxes: List[TrackingBox] = all_boxes.boxes[sample_token]
            for box in boxes:
                # Augment the boxes with timestamp. We will use timestamps to sort boxes in time later.
                box.timestamp = sample_record['timestamp']

                # Add box to tracks.
                if box.tracking_id not in tracks[scene_token].keys():
                    tracks[scene_token][box.tracking_id] = []
                tracks[scene_token][box.tracking_id].append(box)

        # Make sure the tracks are sorted in time.
        for scene_token, scene in tracks.items():
            for tracking_id, track in scene.items():
                scene[tracking_id] = sorted(track, key=lambda _box: _box.timestamp)
            tracks[scene_token] = scene

        return tracks

    def get_thresholds(self, scores, num_gt):
        # based on score of true positive to discretize the recall
        # may not be 11 due to not fully recall the results, all the results point has zero precision
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / float(num_gt)
            if i < (len(scores) - 1):
                r_recall = (i + 2) / float(num_gt)
            else:
                r_recall = l_recall
            if (r_recall - current_recall) < (current_recall - l_recall) and i < (len(scores) - 1):
                continue

            thresholds.append(score)
            current_recall += 1 / (self.num_sample_pts - 1.0)

        return thresholds

    def reset(self):
        self.n_gt = 0  # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # number of ignored ground truth detections
        self.n_tr = 0  # number of tracker detections minus ignored tracker detections
        self.n_itr = 0  # number of ignored tracker detections
        self.n_igttr = 0  # number of ignored ground truth detections where the corresponding associated tracker detection is also ignored

        self.MOTA = 0
        self.MOTP = 0

        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0

        self.total_cost = 0
        self.itp = 0
        self.tp = 0
        self.fn = 0
        self.ifn = 0
        self.fp = 0

        self.n_gts = []  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = []  # number of ground ignored truth detections PER SEQUENCE
        self.n_trs = []  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itrs = []  # number of ignored tracker detections PER SEQUENCE

        self.itps = []  # number of ignored true positives PER SEQUENCE
        self.tps = []  # number of true positives including ignored true positives PER SEQUENCE
        self.fns = []  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifns = []  # number of ignored false negatives PER SEQUENCE
        self.fps = []  # above PER SEQUENCE

        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    def compute_third_party_metrics(self, threshold=-10000):
        """
            Computes the metrics defined in
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9
        self.scores = list()

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0, 0
        for seq_idx in range(len(self.gt_boxes)):
            seq_gt = self.gt_boxes[seq_idx]
            seq_tracker_before: Dict[str, TrackingBox] = self.tracks[seq_idx]

            # remove the tracks with low confidence for each frame
            tracker_id_score = dict()
            for frame in range(len(seq_tracker_before)):
                tracks_tmp = seq_tracker_before[frame]
                for index in range(len(tracks_tmp)):
                    trk_tmp = tracks_tmp[index]
                    id_tmp = trk_tmp.track_id
                    score_tmp = trk_tmp.score

                    if id_tmp not in tracker_id_score.keys():
                        tracker_id_score[id_tmp] = list()
                    tracker_id_score[id_tmp].append(score_tmp)

            to_delete_id = list()
            for track_id, score_list in tracker_id_score.items():
                average_score = sum(score_list) / float(len(score_list))
                if average_score < threshold:
                    to_delete_id.append(track_id)

            seq_tracker = list()
            for frame in range(len(seq_tracker_before)):
                seq_tracker_frame = list()
                tracks_tmp = seq_tracker_before[frame]
                for index in range(len(tracks_tmp)):
                    trk_tmp = tracks_tmp[index]
                    id_tmp = trk_tmp.track_id
                    if id_tmp not in to_delete_id:
                        seq_tracker_frame.append(trk_tmp)
                seq_tracker.append(seq_tracker_frame)

            seq_trajectories = defaultdict(list)
            seq_ignored = defaultdict(list)

            # statistics over the current sequence, check the corresponding
            # variable comments in __init__ to get their meaning
            seqtp = 0
            seqitp = 0
            seqfn = 0
            seqifn = 0
            seqfp = 0
            seqigt = 0
            seqitr = 0

            last_ids = [[], []]

            n_gts = 0
            n_trs = 0

            for f in range(len(seq_gt)):  # go through each frame
                g = seq_gt[f]

                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)

                n_gts += len(g)
                n_trs += len(t)

                # use hungarian method to associate, using center distance 0..1 as cost
                # build cost matrix
                # row is gt, column is det
                cost_matrix = []
                this_ids = [[], []]
                for gg in g:
                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker = -1
                    gg.id_switch = 0
                    gg.fragmentation = 0
                    cost_row = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1 - center_distance(gg, tt)

                        # gating for center_distance
                        if c <= 1 - self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)  # = 1e9
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                if len(g) == 0:
                    cost_matrix = [[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # tmp variables for sanity checks and MODP computation
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc = 0  # this will sum up the overlaps for all true positives
                tmpcs = [0] * len(g)  # this will save the overlaps for all true positives
                # the reason is that some true positives might be ignored
                # later such that the corrsponding overlaps can
                # be subtracted from tmpc for MODP computation

                # mapping for tracker ids and ground truth ids
                for row, col in association_matrix:
                    # apply gating on boxoverlap # TODO: remove?
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        g[row].tracker = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid = True
                        g[row].distance = c
                        self.total_cost += 1 - c
                        tmpc += 1 - c
                        tmpcs[row] = 1 - c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp += 1

                        self.scores.append(t[col].score)

                    else:
                        g[row].tracker = -1
                        self.fn += 1
                        tmpfn += 1

                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0  # number of ignored tracker detections
                ignoredtrackers = dict()  # will associate the track_id with -1
                # if it is not ignored and 1 if it is
                # ignored;
                # this is used to avoid double counting ignored
                # cases, see the next loop

                for tt in t:
                    ignoredtrackers[tt.track_id] = -1
                    # ignore detection if it belongs to a neighboring class or is
                    # smaller or equal to the minimum height

                    # if ((self.cls == "car" and tt.obj_type == "van") or (
                    #         self.cls == "pedestrian" and tt.obj_type == "person_sitting")) and not tt.valid:
                    #     nignoredtracker += 1
                    #     tt.ignored = True
                    #     ignoredtrackers[tt.track_id] = 1
                    #     continue
                    # for d in dc:
                    #     # overlap = boxoverlap(tt,d,"a")
                    #     overlap = center_distance(tt, d)  # TODO: special considerations for dont care
                    #     if overlap > 0.5 and not tt.valid:
                    #         tt.ignored = True
                    #         nignoredtracker += 1
                    #         ignoredtrackers[tt.track_id] = 1
                    #         break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn = 0  # the number of ignored false negatives
                nignoredtp = 0  # the number of ignored true positives
                nignoredpairs = 0  # the number of ignored pairs, i.e. a true positive
                # which is ignored but where the associated tracker
                # detection has already been ignored

                gi = 0
                for gg in g:
                    if gg.tracker < 0:
                        if (self.cls == "car" and gg.obj_type == "van") or \
                           (self.cls == "pedestrian" and gg.obj_type == "person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1

                    elif gg.tracker >= 0:
                        if (self.cls == "car" and gg.obj_type == "van") or \
                           (self.cls == "pedestrian" and gg.obj_type == "person_sitting"):

                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                            # if the associated tracker detection is already ignored,
                            # we want to avoid double counting ignored detections
                            if ignoredtrackers[gg.tracker] > 0:
                                nignoredpairs += 1

                            # for computing MODP, the overlaps from ignored detections
                            # are subtracted
                            tmpc -= tmpcs[gi]
                    gi += 1

                # the below might be confusion, check the comments in __init__
                # to see what the individual statistics represent

                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp

                # count the number of ignored true positives
                self.itp += nignoredtp

                # adjust the number of ground truth objects considered
                self.n_gt -= (ignoredfn + nignoredtp)

                # count the number of ignored ground truth objects
                self.n_igt += ignoredfn + nignoredtp

                # count the number of ignored tracker objects
                self.n_itr += nignoredtracker

                # count the number of ignored pairs, i.e. associated tracker and
                # ground truth objects that are both ignored
                self.n_igttr += nignoredpairs

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                #
                tmpfn += len(g) - len(association_matrix) - ignoredfn
                self.fn += len(g) - len(association_matrix) - ignoredfn
                self.ifn += ignoredfn

                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t)
                tmpfp += len(t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                self.fp += len(t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                # tmpfp   = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
                # self.fp += len(t) - tmptp - nignoredtp

                # update sequence data
                seqtp += tmptp
                seqitp += nignoredtp
                seqfp += tmpfp
                seqfn += tmpfn
                seqifn += ignoredfn
                seqigt += ignoredfn + nignoredtp
                seqitr += nignoredtracker

                # sanity checks
                # - the number of true positives minues ignored true positives
                #   should be greater or equal to 0
                # - the number of false negatives should be greater or equal to 0
                # - the number of false positives needs to be greater or equal to 0
                #   otherwise ignored detections might be counted double
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false negatives (plus ignored ones)
                #   should match the total number of ground truth objects
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false positives
                #   plus the number of ignored tracker detections should
                #   match the total number of tracker detections; note that
                #   nignoredpairs is subtracted here to avoid double counting
                #   of ignored detection sin nignoredtp and nignoredtracker
                if tmptp < 0:
                    print(tmptp, nignoredtp)
                    raise NameError("Something went wrong! TP is negative")
                if tmpfn < 0:
                    print(tmpfn, len(g), len(association_matrix), ignoredfn, nignoredpairs)
                    raise NameError("Something went wrong! FN is negative")
                if tmpfp < 0:
                    print(tmpfp, len(t), tmptp, nignoredtracker, nignoredtp, nignoredpairs)
                    raise NameError("Something went wrong! FP is negative")
                if tmptp + tmpfn != len(g) - ignoredfn - nignoredtp:
                    print("seqidx", seq_idx)
                    print("frame ", f)
                    print("TP    ", tmptp)
                    print("FN    ", tmpfn)
                    print("FP    ", tmpfp)
                    print("nGT   ", len(g))
                    print("nAss  ", len(association_matrix))
                    print("ign GT", ignoredfn)
                    print("ign TP", nignoredtp)
                    raise NameError("Something went wrong! nGroundtruth is not TP+FN")
                if tmptp + tmpfp + nignoredtp + nignoredtracker - nignoredpairs != len(t):
                    print(seq_idx, f, len(t), tmptp, tmpfp)
                    print(len(association_matrix), association_matrix)
                    raise NameError("Something went wrong! nTracker is not TP+FP")

                # check for id switches or Fragmentations
                # frag will be more than id switch, switch happens only when id is different but detection exists
                # frag happens when id switch or detection is missing
                for i, tt in enumerate(this_ids[0]):
                    # print(i)
                    # print(tt)
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]  # id in current tracker corresponding to the gt tt
                        lid = last_ids[1][idx]  # id in last frame tracker corresponding to the gt tt
                        if tid != lid and lid != -1 and tid != -1:
                            # TODO: check if this makes sense without truncation logic
                            g[i].id_switch = 1
                            ids += 1
                        if tid != lid and lid != -1:
                            # TODO: check if this makes sense without truncation logic
                            g[i].fragmentation = 1
                            fr += 1

                # save current index
                last_ids = this_ids

            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx] = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored

            # self.num_gt += n_gts
            # gather statistics for "per sequence" statistics.
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)

        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (seq_trajectories, seq_ignored) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories) == 0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0] * 5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(), seq_ignored.values()):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue
                # all frames of this gt trajectory are not assigned to any detections
                if all([this == -1 for this in g]):
                    tmpML += 1
                    self.ML += 1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0] >= 0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt += 1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f - 1] != -1:
                        tmpId_switches += 1
                        self.id_switches += 1
                    if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
                        tmpFragments += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handled in for loop (g[f]!=-1)
                if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and not ign_g[f]:
                    tmpFragments += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    tmpMT += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML += 1
                    self.ML += 1
                else:  # 0.2 <= tracking_ratio <= 0.8
                    tmpPT += 1
                    self.PT += 1

        if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        if (self.recall + self.precision) == 0:
            self.F1 = 0.
        else:
            self.F1 = 2. * (self.precision * self.recall) / (self.precision + self.recall)
        if sum(self.n_frames) == 0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp / float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt == 0:
            self.MOTA = -float("inf")
        else:
            self.MOTA = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
        if self.tp == 0:
            self.MOTP = 0
        else:
            self.MOTP = self.total_cost / float(self.tp)

        self.num_gt = self.tp + self.fn
        return True

    def create_summary_details(self):
        """
            Generate and mail a summary of the results.
            If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += "evaluation: best results with single threshold".center(80, "=") + "\n"
        summary += self.print_entry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA) + "\n"
        summary += self.print_entry("Multiple Object Tracking Precision (MOTP)", float(self.MOTP)) + "\n"
        summary += "\n"
        summary += self.print_entry("Recall", self.recall) + "\n"
        summary += self.print_entry("Precision", self.precision) + "\n"
        summary += self.print_entry("F1", self.F1) + "\n"
        summary += self.print_entry("False Alarm Rate", self.FAR) + "\n"
        summary += "\n"
        summary += self.print_entry("Mostly Tracked", self.MT) + "\n"
        summary += self.print_entry("Partly Tracked", self.PT) + "\n"
        summary += self.print_entry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.print_entry("True Positives", self.tp) + "\n"
        # summary += self.print_entry("True Positives per Sequence", self.tps) + "\n"
        summary += self.print_entry("Ignored True Positives", self.itp) + "\n"
        # summary += self.print_entry("Ignored True Positives per Sequence", self.itps) + "\n"
        summary += self.print_entry("False Positives", self.fp) + "\n"
        # summary += self.print_entry("False Positives per Sequence", self.fps) + "\n"
        summary += self.print_entry("False Negatives", self.fn) + "\n"
        # summary += self.print_entry("False Negatives per Sequence", self.fns) + "\n"
        summary += self.print_entry("Ignored False Negatives", self.ifn) + "\n"
        # summary += self.print_entry("Ignored False Negatives per Sequence", self.ifns) + "\n"
        # summary += self.print_entry("Missed Targets", self.fn) + "\n"
        summary += self.print_entry("ID-switches", self.id_switches) + "\n"
        summary += self.print_entry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += self.print_entry("Ground Truth Objects (Total)", self.n_gt + self.n_igt) + "\n"
        # summary += self.print_entry("Ground Truth Objects (Total) per Sequence", self.n_gts) + "\n"
        summary += self.print_entry("Ignored Ground Truth Objects", self.n_igt) + "\n"
        # summary += self.print_entry("Ignored Ground Truth Objects per Sequence", self.n_igts) + "\n"
        summary += self.print_entry("Ground Truth Trajectories", self.n_gt_trajectories) + "\n"
        summary += "\n"
        summary += self.print_entry("Tracker Objects (Total)", self.n_tr) + "\n"
        # summary += self.print_entry("Tracker Objects (Total) per Sequence", self.n_trs) + "\n"
        summary += self.print_entry("Ignored Tracker Objects", self.n_itr) + "\n"
        # summary += self.print_entry("Ignored Tracker Objects per Sequence", self.n_itrs) + "\n"
        summary += self.print_entry("Tracker Trajectories", self.n_tr_trajectories) + "\n"
        # summary += "\n"
        # summary += self.print_entry("Ignored Tracker Objects with Associated Ignored Ground Truth Objects", self.n_igttr) + "\n"
        summary += "=" * 80

        return summary

    def create_summary_simple(self, threshold):
        """
            Generate and mail a summary of the results.
            If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += ("evaluation with confidence threshold %f" % threshold).center(80, "=") + "\n"
        summary += ' MOTA   MOTP   MODA   MODP    MT     ML     IDS  FRAG    F1   Prec  Recall  FAR     TP    FP    FN\n'

        summary += '{:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:5d}\n'.format( \
            self.MOTA, self.MOTP, self.MT, self.ML, self.id_switches, self.fragments, \
            self.F1, self.precision, self.recall, self.FAR, self.tp, self.fp, self.fn)
        summary += "=" * 80

        return summary

    def print_entry(self, key, val, width=(70, 10)):
        """
            Pretty print an entry in a table fashion.
        """

        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%d.4f" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def save_to_stats(self, dump, threshold=None):
        """
            Save the statistics in a whitespace separate file.
        """

        # write summary to file summary_cls.txt
        if threshold is None:
            summary = self.create_summary_details()
        else:
            summary = self.create_summary_simple(threshold)
        self.mail.msg(summary)  # mail or print the summary.
        print(summary, file=dump)


class Stat:
    """
        Utility class to load data.
    """

    def __init__(self,
                 cls,
                 suffix,
                 dump,
                 num_sample_pts: int = 11):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.mota = 0
        self.motp = 0
        self.F1 = 0
        self.precision = 0
        self.fp = 0
        self.fn = 0

        self.mota_list = list()
        self.motp_list = list()
        self.f1_list = list()
        self.precision_list = list()
        self.fp_list = list()
        self.fn_list = list()
        self.recall_list = list()

        self.cls = cls
        self.suffix = suffix
        self.dump = dump
        self.num_sample_pts = num_sample_pts

    def update(self, data):
        self.mota += data['mota']
        self.motp += data['motp']
        self.F1 += data['F1']
        self.precision += data['precision']
        self.fp += data['fp']
        self.fn += data['fn']

        self.mota_list.append(data['mota'])
        self.motp_list.append(data['motp'])
        self.f1_list.append(data['F1'])
        self.precision_list.append(data['precision'])
        self.fp_list.append(data['fp'])
        self.fn_list.append(data['fn'])
        self.recall_list.append(data['recall'])

    def output(self):
        self.ave_mota = self.mota / self.num_sample_pts
        self.ave_motp = self.motp / self.num_sample_pts
        self.ave_f1 = self.F1 / self.num_sample_pts
        self.ave_precision = self.precision / self.num_sample_pts

    def print_summary(self):
        summary = ""

        summary += "evaluation: average at all thresholds".center(80, "=") + "\n"
        summary += ' AMOTA  AMOTP  APrec\n'

        summary += '{:.4f} {:.4f} {:.4f}\n'.format(
            self.ave_mota, self.ave_motp, self.ave_precision)
        summary += "=" * 80
        print(summary, file=self.dump)

        return summary

    def plot_over_recall(self, data_list, title, y_name, save_path):
        # add extra zero at the end
        largest_recall = self.recall_list[-1]
        extra_zero = np.arange(largest_recall, 1, 0.01).tolist()
        len_extra = len(extra_zero)
        y_zero = [0] * len_extra

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(self.recall_list + extra_zero), np.array(data_list + y_zero))
        # ax.set_title(title, fontsize=20)
        ax.set_ylabel(y_name, fontsize=20)
        ax.set_xlabel('Recall', fontsize=20)
        ax.set_xlim(0.0, 1.0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        if y_name in ['MOTA', 'MOTP', 'F1', 'Precision']:
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(0.0, max(data_list))

        if y_name in ['MOTA', 'F1']:
            max_ind = np.argmax(np.array(data_list))
            # print(max_ind)
            plt.axvline(self.recall_list[max_ind], ymax=data_list[max_ind], color='r')
            plt.plot(self.recall_list[max_ind], data_list[max_ind], 'or', markersize=12)
            plt.text(self.recall_list[max_ind] - 0.05, data_list[max_ind] + 0.03, '%.2f' % (data_list[max_ind]),
                     fontsize=20)
        fig.savefig(save_path)
        # zxc

    def plot(self):
        save_dir = os.path.join("./results", 'plot')

        self.plot_over_recall(self.mota_list, 'MOTA - Recall Curve', 'MOTA',
                              os.path.join(save_dir, 'MOTA_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.motp_list, 'MOTP - Recall Curve', 'MOTP',
                              os.path.join(save_dir, 'MOTP_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.f1_list, 'F1 - Recall Curve', 'F1',
                              os.path.join(save_dir, 'F1_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.fp_list, 'False Positive - Recall Curve', 'False Positive',
                              os.path.join(save_dir, 'FP_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.fn_list, 'False Negative - Recall Curve', 'False Negative',
                              os.path.join(save_dir, 'FN_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))
        self.plot_over_recall(self.precision_list, 'Precision - Recall Curve', 'Precision',
                              os.path.join(save_dir, 'precision_recall_curve_%s_%s.pdf' % (self.cls, self.suffix)))

class Mail:
    """ Dummy class to print messages without sending e-mails"""
    def __init__(self,mailaddress):
        pass
    def msg(self,msg):
        print(msg)
    def finalize(self,success,benchmark,sha_key,mailaddress=None):
        if success:
            print("Results for %s (benchmark: %s) sucessfully created" % (benchmark,sha_key))
        else:
            print("Creating results for %s (benchmark: %s) failed" % (benchmark,sha_key))

