"""
Code written by Motional and the Robot Learning Lab, University of Freiburg.

Script to evaluate Panoptic nuScenes panoptic segmentation (PS) or panoptic tracking (PT) metrics.
Argument "task" could be one of ["segmentation", "tracking"], check eval/panoptic/README.md for more details of the
tasks. Note tracking results will be evaluated with both the PT and PS metrics.

Example usage to evaluate tracking metrics.
---------------------------------------------------------
python python-sdk/nuscenes/eval/panoptic/evaluate.py --result_path /data/panoptic_pred_results --eval_set mini_val
--task tracking --dataroot /data/sets/nuscenes --version v1.0-mini --out_dir /tmp/panoptic_eval_output
---------------------------------------------------------

Note, the panoptic prediction result directory should be as follows:
└── panoptic_results_dir
    └── panoptic
        └── {test, train, val, mini_train, mini_val} <- Contains the .npz files; a .npz file contains panoptic labels
        │                                                of the points in a point cloud.
        └── {test, train, val, mini_train, mini_val}
            └── submission.json                      <- contains certain information about the submission.
"""
import argparse
import json
import os
from typing import Any, Dict

import numpy as np
from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval
from nuscenes.eval.panoptic.panoptic_track_evaluator import PanopticTrackingEval
from nuscenes.eval.panoptic.utils import PanopticClassMapper, get_samples_in_panoptic_eval_set
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm


class NuScenesPanopticEval:
    """
    This is the official Panoptic nuScenes evaluation code. Results are written to the provided output_dir.
    Panoptic nuScenes uses the following metrics:
    - Panoptic Segmentation: we use the PQ (Panoptic Quality) metric: which is defined as:
      PQ = IOU/(TP + 0.5*FP + 0.5*FN).
    - Multi-object Panoptic Tracking: we use the PAT (Panoptic Tracking) metric, which is defined as:
      PAT = 2*PQ*TQ / (PQ + TQ) where TQ is as defined in the paper: 
      Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking 
      (https://arxiv.org/pdf/2109.03805.pdf)
    """

    def __init__(self,
                 nusc: NuScenes,
                 results_folder: str,
                 eval_set: str,
                 task: str,
                 min_inst_points: int,
                 out_dir: str = None,
                 verbose: bool = False):
        """
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param task: What panoptic task to evaluate on, 'segmentation' for panoptic segmentation evaluation only;
            and 'tracking' for both panoptic segmentation and multi-object panoptic tracking evaluation.
        :param min_inst_points: minimal number of instance points.
        :param out_dir: Output directory. The results is saved as 'out_dir/{task}-result.json' file;
        :param verbose: Whether to print messages during the evaluation.
        """
        assert hasattr(nusc, 'panoptic') and len(getattr(nusc, 'panoptic')) > 0,\
            f'Error: no panoptic ground truths found in {nusc.version}'

        supported_tasks = ['segmentation', 'tracking']
        if task not in supported_tasks:
            raise ValueError(f'Supported task must be one of: {supported_tasks}, got: {task} !')

        results_npz_folder = os.path.join(results_folder, 'panoptic', eval_set)
        assert os.path.exists(results_npz_folder), \
            f'Error: The folder containing the .npz files ({results_npz_folder}) does not exist.'

        self.nusc = nusc
        self.results_folder = results_folder
        self.eval_set = eval_set
        self.task = task
        self.verbose = verbose
        self.min_inst_points = min_inst_points
        self.out_dir = out_dir

        self.mapper = PanopticClassMapper(self.nusc)
        self.ignore_idx = self.mapper.ignore_class['index']
        self.id2name = {idx: name for name, idx in self.mapper.coarse_name_2_coarse_idx_mapping.items()}
        self.num_classes = len(self.mapper.coarse_name_2_coarse_idx_mapping)

        self.things = self.mapper.things.keys()
        self.stuff = self.mapper.stuff.keys()
        self.sample_tokens = get_samples_in_panoptic_eval_set(self.nusc, self.eval_set)

        if self.verbose:
            print(f'There are {self.num_classes} classes, {len(self.sample_tokens)} samples.')

        self.evaluator = dict(segmentation=PanopticEval(n_classes=self.num_classes,
                                                        ignore=[self.ignore_idx],
                                                        min_points=self.min_inst_points))
        if self.task == 'tracking':
            self.scene_name2tok = {rec['name']: rec['token'] for rec in nusc.scene}
            self.evaluator['tracking'] = PanopticTrackingEval(n_classes=self.num_classes,
                                                              min_stuff_cls_id=len(self.things) + 1,
                                                              ignore=[self.ignore_idx],
                                                              min_points=self.min_inst_points)

        self.eval_result_file = os.path.join(self.out_dir, self.task + '-result.json')
        if os.path.isfile(self.eval_result_file):
            os.remove(self.eval_result_file)

    def evaluate(self) -> None:
        """
        Evaluate metrics for task. For segmentation task, only panoptic segmentation metrics will be evaluated. For
        tracking task, besides the multi-object panoptic tracking metrics, single frame based panoptic segmentation
        metrics will be evaluated as well.
        """
        eval_results = {'segmentation': self.evaluate_segmentation()}
        if self.task == 'tracking':
            eval_results['tracking'] = self.evaluate_tracking()
        self.save_result(eval_results)

    def evaluate_segmentation(self) -> Dict[str, Any]:
        """
        Calculate panoptic segmentation metrics.
        :return: A dict of panoptic metrics for mean of all classes and each class.
            {
                "all": { "PQ": float, "SQ": float, "RQ": float, "mIoU": float, "PQ_dagger": float},
                "ignore": { "PQ": float, "SQ": float, "RQ": float, "IoU": float},
                "car": { "PQ": float, "SQ": float, "RQ": float, "IoU": float},
                ...
            }
        """
        for sample_token in tqdm(self.sample_tokens, disable=not self.verbose):
            sample = self.nusc.get('sample', sample_token)
            # Get the sample data token of the point cloud.
            sd_token = sample['data']['LIDAR_TOP']

            # Load the ground truth labels for the point cloud.
            panoptic_label_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', sd_token)['filename'])
            panoptic_label = load_bin_file(panoptic_label_filename, type='panoptic')

            # Filter eval classes.
            label_sem = self.mapper.convert_label(panoptic_label // 1000)
            label_inst = panoptic_label
            panoptic_pred_filename = os.path.join(self.results_folder, 'panoptic', self.eval_set,
                                                  sd_token + '_panoptic.npz')
            panoptic_pred = load_bin_file(panoptic_pred_filename, type='panoptic')
            pred_sem = panoptic_pred // 1000
            pred_inst = panoptic_pred

            # Get the confusion matrix between the ground truth and predictions. Update the confusion matrix for the
            # sample data into the confusion matrix for the eval set.
            self.evaluator['segmentation'].addBatch(pred_sem, pred_inst, label_sem, label_inst)

        mean_pq, mean_sq, mean_rq, class_all_pq, class_all_sq, class_all_rq = self.evaluator['segmentation'].getPQ()
        mean_iou, class_all_iou = self.evaluator['segmentation'].getSemIoU()

        results = self.wrap_result_segmentation(mean_pq, mean_sq, mean_rq, mean_iou, class_all_pq, class_all_sq,
                                                class_all_rq, class_all_iou)
        return results

    def wrap_result_segmentation(self,
                                 mean_pq: np.ndarray,
                                 mean_sq: np.ndarray,
                                 mean_rq: np.ndarray,
                                 mean_iou: np.ndarray,
                                 class_all_pq: np.ndarray,
                                 class_all_sq: np.ndarray,
                                 class_all_rq: np.ndarray,
                                 class_all_iou: np.ndarray) -> Dict[str, Any]:
        """
        Wrap panoptic segmentation results to dict format.
        :param mean_pq: <float64: 1>, Mean Panoptic Quality over all classes.
        :param mean_sq: <float64: 1>, Mean Segmentation Quality over all classes.
        :param mean_rq: <float64: 1>, Mean Recognition Quality over all classes.
        :param mean_iou: <float64: 1>, Mean IoU score over all classes.
        :param class_all_pq: <float64: num_classes,>, Panoptic Quality for each class.
        :param class_all_sq: <float64: num_classes,> Segmentation Quality for each class.
        :param class_all_rq: <float64: num_classes,>,  Recognition Quality for each class.
        :param class_all_iou: <float64: num_classes,>, IoU scores for each class.
        :return: A dict of panoptic segmentation metrics.
        """
        mean_pq, mean_sq, mean_rq, mean_iou = mean_pq.item(), mean_sq.item(), mean_rq.item(), mean_iou.item()
        class_all_pq = class_all_pq.flatten().tolist()
        class_all_sq = class_all_sq.flatten().tolist()
        class_all_rq = class_all_rq.flatten().tolist()
        class_all_iou = class_all_iou.flatten().tolist()

        results = dict()
        results["all"] = dict(PQ=mean_pq, SQ=mean_sq, RQ=mean_rq, mIoU=mean_iou)
        for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_pq, class_all_rq, class_all_sq, class_all_iou)):
            results[self.id2name[idx]] = dict(PQ=pq, SQ=sq, RQ=rq, IoU=iou)
        thing_pq_list = [float(results[c]["PQ"]) for c in self.things]
        stuff_iou_list = [float(results[c]["IoU"]) for c in self.stuff]
        results["all"]["PQ_dagger"] = np.mean(thing_pq_list + stuff_iou_list)

        return results

    def evaluate_tracking(self) -> Dict[str, Any]:
        """
        Calculate multi-object panoptic tracking metrics.
        :return: A dict of panoptic metrics for mean of all classes and each class.
            {
                "all": { "PAT": float, "PQ": float, "TQ": float, PTQ": float, "sPTQ": float, "LSTQ": float,
                         "mIoU": float, "S_assoc": float, "PTQ_dagger": float, "MOTSA": float, "sMOTSA": float,
                         "MOTSP": float},
                "ignore": { "PTQ": float, "sPTQ": float, "IoU": float},
                "car": { "PTQ": float, "sPTQ": float, "IoU": float},
                ...
            }
        """
        eval_scenes = create_splits_scenes(verbose=False)[self.eval_set]
        for scene in tqdm(eval_scenes, disable=not self.verbose):
            scene = self.nusc.get('scene', self.scene_name2tok[scene])
            cur_token, last_token = scene['first_sample_token'], scene['last_sample_token']
            pred_sem, pred_inst, label_sem, label_inst = [None], [None], [None], [None]

            while True:
                cur_sample = self.nusc.get('sample', cur_token)
                sd_token = cur_sample['data']['LIDAR_TOP']

                # Load the ground truth labels for the point cloud, filter evaluation classes.
                gt_label_file = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', sd_token)['filename'])
                panoptic_label = load_bin_file(gt_label_file, type='panoptic')
                label_sem.append(self.mapper.convert_label(panoptic_label // 1000))
                label_sem = label_sem[-2:]
                label_inst.append(panoptic_label)
                label_inst = label_inst[-2:]

                # Load predictions for the point cloud, filter evaluation classes.
                pred_file = os.path.join(self.results_folder, 'panoptic', self.eval_set, sd_token + '_panoptic.npz')
                panoptic_pred = load_bin_file(pred_file, type='panoptic')
                pred_sem.append(panoptic_pred // 1000)
                pred_sem = pred_sem[-2:]
                pred_inst.append(panoptic_pred)
                pred_inst = pred_inst[-2:]

                # Get the confusion matrix between the ground truth and predictions. Update the confusion matrix for
                # the sample data into the confusion matrix for the eval set.
                self.evaluator['tracking'].add_batch(scene['name'], pred_sem, pred_inst, label_sem, label_inst)
                if cur_token == last_token:
                    break
                cur_token = cur_sample['next']

        pat, mean_pq, mean_tq = self.evaluator['tracking'].get_pat()
        mean_ptq, class_all_ptq, mean_sptq, class_all_sptq = self.evaluator['tracking'].get_ptq()
        mean_iou, class_all_iou = self.evaluator['tracking'].getSemIoU()
        lstq, s_assoc = self.evaluator['tracking'].get_lstq()
        mean_motsa, mean_s_motsa, mean_motsp = self.evaluator['tracking'].get_motsa()

        results = self.wrap_result_mopt(pat=pat,
                                        mean_pq=mean_pq,
                                        mean_tq=mean_tq,
                                        mean_ptq=mean_ptq,
                                        class_all_ptq=class_all_ptq,
                                        mean_sptq=mean_sptq,
                                        class_all_sptq=class_all_sptq,
                                        mean_iou=mean_iou,
                                        class_all_iou=class_all_iou,
                                        lstq=lstq,
                                        s_assoc=s_assoc,
                                        mean_motsa=mean_motsa,
                                        mean_s_motsa=mean_s_motsa,
                                        mean_motsp=mean_motsp)

        return results

    def wrap_result_mopt(self,
                         pat: np.ndarray,
                         mean_pq: np.ndarray,
                         mean_tq: np.ndarray,
                         mean_ptq: np.ndarray,
                         class_all_ptq: np.ndarray,
                         mean_sptq: np.ndarray,
                         class_all_sptq: np.ndarray,
                         mean_iou: np.ndarray,
                         class_all_iou: np.ndarray,
                         lstq: np.ndarray,
                         s_assoc: np.ndarray,
                         mean_motsa: np.ndarray,
                         mean_s_motsa: np.ndarray,
                         mean_motsp: np.ndarray) -> Dict[str, Any]:
        """
        Wrap up MOPT results to dictionary.
        :param pat: <float64: 1>, Panoptic Tracking (PAT) score over all classes.
        :param mean_pq: <float64: 1>, Mean Panoptic Quality over all classes.
        :param mean_tq: <float64: 1>, Mean Tracking Quality over all temporally unique instances.
        :param mean_ptq: <float64: 1>, Mean PTQ score over all classes.
        :param mean_sptq: <float64: 1>, Mean soft-PTQ score over all classes.
        :param mean_iou: <float64: 1>, Mean IoU score over all classes.
        :param class_all_ptq: <float64: num_classes,>, PTQ scores for each class.
        :param class_all_sptq: <float64: num_classes,>, Soft-PTQ scores for each class.
        :param class_all_iou: <float64: num_classes,>, IoU scores for each class.
        :param lstq: <float64: 1>, LiDAR Segmentation and Tracking Quality (LSTQ) score over all classes.
        :param s_assoc: <float64: 1>, Association Score over all classes.
        :param mean_motsa: <float64: 1>, Mean MOTSA score over all thing classes.
        :param mean_s_motsa: <float64: 1>, Mean sMOTSA score over all thing classes.
        :param mean_motsp: <float64: 1>, Mean MOTSP score over all thing classes.
        :return: A dict of multi-object panoptic tracking metrics.
        """
        pat, mean_pq, mean_tq = pat.item(), mean_pq.item(), mean_tq.item()
        mean_ptq, mean_sptq, mean_iou = mean_ptq.item(), mean_sptq.item(), mean_iou.item()
        class_all_ptq = class_all_ptq.flatten().tolist()
        class_all_sptq = class_all_sptq.flatten().tolist()
        class_all_iou = class_all_iou.flatten().tolist()

        results = dict()
        results["all"] = dict(PAT=pat, PQ=mean_pq, TQ=mean_tq, PTQ=mean_ptq, sPTQ=mean_sptq,
                              LSTQ=lstq, mIoU=mean_iou, S_assoc=s_assoc, MOTSA=mean_motsa,
                              sMOTSA=mean_s_motsa, MOTSP=mean_motsp)
        for idx, (ptq, sptq, iou) in enumerate(zip(class_all_ptq, class_all_sptq, class_all_iou)):
            results[self.id2name[idx]] = dict(PTQ=ptq, sPTQ=sptq, IoU=iou)
        thing_ptq_list = [float(results[c]["PTQ"]) for c in self.things]
        stuff_iou_list = [float(results[c]["IoU"]) for c in self.stuff]
        results["all"]["PTQ_dagger"] = np.mean(thing_ptq_list + stuff_iou_list)

        return results

    def save_result(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Dump evaluation results to result.json
        :param results: {task_name: task_results}, evaluation results in a dictionary.
        """
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self.eval_result_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            raise ValueError(f'Invalid output dir: {self.out_dir}')

        if self.verbose:
            print(f"======\nPanoptic nuScenes {self.task} evaluation for {self.eval_set}")
            print(json.dumps(results, indent=4, sort_keys=False))
            print("======")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Panoptic nuScenes results.')
    parser.add_argument('--result_path', type=str, help='The path to the results folder.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--task', type=str, default='segmentation',
                        help='What task to evaluate, segmentation or tracking.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--min_inst_points', type=int, default=15,
                        help='Lower bound for the number of points to be considered instance.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print to stdout.')
    parser.add_argument('--out_dir', type=str, default=None, help='Folder to write the panoptic labels to.')
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else f'Panoptic-nuScenes-{args.version}'
    task = args.task
    # Overwrite with task from submission.json if the file exists.
    submission_file = os.path.join(args.result_path, args.eval_set, 'submission.json')
    if os.path.exists(submission_file):
        print(submission_file)
        with open(submission_file, 'r') as f:
            data = json.load(f)
            if 'meta' in data and 'task' in data['meta']:
                task = data['meta']['task']

    supported_tasks = ['segmentation', 'tracking']
    if task not in supported_tasks:
        raise ValueError(f'Supported task must be one of: {supported_tasks}, got: {task} !')

    print(f'Start {task} evaluation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)

    evaluator = NuScenesPanopticEval(nusc=nusc,
                                     results_folder=args.result_path,
                                     eval_set=args.eval_set,
                                     task=task,
                                     min_inst_points=args.min_inst_points,
                                     out_dir=out_dir,
                                     verbose=args.verbose)
    evaluator.evaluate()
    print(f'Evaluation results saved at {args.out_dir}/{task}-result.json. \nFinished {task} evaluation.')


if __name__ == '__main__':
    main()
