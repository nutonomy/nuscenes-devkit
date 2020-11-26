import os
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.evaluate import LidarSegEval
from nuscenes.eval.lidarseg.utils import ConfusionMatrix, load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud


class LidarSegEvalStratified(LidarSegEval):
    """
    This class extends the LidarSegEval class to provide... TODO
    Stratify by radial distance
    """
    def __init__(self, nusc: NuScenes,
                 results_folder: str,
                 eval_set: str,
                 # sample_size: int,
                 strata_list: Tuple[Tuple[float]] = ((0, 20), (20, 40), (40, 60)),
                 # is_render_bad_samples: bool = True,
                 verbose: bool = True):

        super().__init__(nusc, results_folder, eval_set, verbose)

        self.strata_list = strata_list

        # Create a list of confusion matrices, one for each strata.
        self.global_cm = [ConfusionMatrix(self.num_classes, self.ignore_idx) for i in range(len(strata_list))]

    def evaluate(self):
        """

        Overwrites the `evaluate` method in the LidarSegEval class.
        """
        for i, strata in enumerate(self.strata_list):
            print('Evaluating for strata {}m to {}m...'.format(strata[0], strata[1]))

            for sample_token in tqdm(self.sample_tokens, disable=not self.verbose):
                sample = self.nusc.get('sample', sample_token)

                # 1. Get the sample data token of the point cloud.
                sd_token = sample['data']['LIDAR_TOP']

                # 2. Get the indices of the points belonging to the strata.
                pointsensor = self.nusc.get('sample_data', sd_token)
                pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
                points = pc.points  # [4, N]
                _, filtered_idxs = self.filter_pointcloud_by_depth(points, min_depth=strata[0], max_depth=strata[1])

                # 3. Load the ground truth labels for the point cloud.
                lidarseg_label_filename = os.path.join(self.nusc.dataroot,
                                                       self.nusc.get('lidarseg', sd_token)['filename'])
                lidarseg_label = load_bin_file(lidarseg_label_filename)
                lidarseg_label = self.mapper.convert_label(lidarseg_label)  # Map the labels as necessary.
                lidarseg_label = lidarseg_label[filtered_idxs]  # Filter to get only labels belonging to the strata.

                # 4. Load the predictions for the point cloud.
                lidarseg_pred_filename = os.path.join(self.results_folder, 'lidarseg',
                                                      self.eval_set, sd_token + '_lidarseg.bin')
                lidarseg_pred = load_bin_file(lidarseg_pred_filename)
                lidarseg_pred = lidarseg_pred[filtered_idxs]  # Filter to get only labels belonging to the strata.

                # 5. Update the confusion matrix for the sample data into the confusion matrix for the eval set.
                self.global_cm[i].update(lidarseg_label, lidarseg_pred)

        stratified_per_class_metrics = self.get_stratified_per_class_metrics()
        self.render_stratified_per_class_metrics(stratified_per_class_metrics, '/home/whye/Desktop/logs/hi.png')

    def render_stratified_per_class_metrics(self, stratified_per_class_metrics, filename: str) -> None:
        """

        """
        stratified_classes = {cls_name: [] for cls_name in self.id2name.values()}

        for strata_metrics in stratified_per_class_metrics:
            for cls, cls_iou in strata_metrics['iou_per_class'].items():
                stratified_classes[cls].append(cls_iou)

        print(stratified_classes)
        _, axes = plt.subplots(self.num_classes, 1, figsize=(5 * len(self.strata_list), 100))
        for i, (cls, cls_strata) in enumerate(stratified_classes.items()):
            if cls == 'ignore':
                continue   # TODO find better way to ignore class.
            axes[i].bar([' to '.join(map(str, rng)) for rng in self.strata_list], cls_strata, color='green')
            axes[i].set_xlabel('Interval', fontsize=15)
            axes[i].set_ylabel('IOU', fontsize=15)
            axes[i].set_title('Stratified results for {}'.format(cls), fontsize=20)
            axes[i].tick_params(axis='both', which='major', labelsize=15)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def get_stratified_per_class_metrics(self) -> List[Dict]:
        """

        """
        stratified_per_class_iou = []
        for strata_cm in self.global_cm:
            iou_per_class = strata_cm.get_per_class_iou()
            miou = strata_cm.get_mean_iou()
            freqweighted_iou = strata_cm.get_freqweighted_iou()

            # Put everything nicely into a dict.
            results = {'iou_per_class': {self.id2name[i]: class_iou for i, class_iou in enumerate(iou_per_class)},
                       'miou': miou,
                       'freq_weighted_iou': freqweighted_iou}

            stratified_per_class_iou.append(results)

        return stratified_per_class_iou

    @staticmethod
    def filter_pointcloud_by_depth(points: LidarPointCloud,
                                   min_depth: float = 0,
                                   max_depth: float = None) -> Union[LidarPointCloud, List[int]]:
        """
        Radial distance  # TODO
        """
        points = points.T  # Transpose the matrix to make manipulation for convenient ([4, N] to [N, 4]).

        depth = np.linalg.norm(points[:, :2], axis=1)

        assert min_depth >= 0, 'Error: min_depth cannot be negative.'
        min_depth_idxs = np.where(depth > min_depth)[0]
        # print('min: ', len(min_depth_idxs))  # TODO
        # print(min_depth_idxs)  # TODO

        if max_depth is not None:
            assert max_depth >= 0, 'Error: max_depth cannot be negative.'
            max_depth_idxs = np.where(depth <= max_depth)[0]
            # print('max: ', len(max_depth_idxs))  # TODO
            # print(max_depth_idxs)  # TODO

            filtered_idxs = np.intersect1d(min_depth_idxs, max_depth_idxs)
        else:
            filtered_idxs = min_depth_idxs

        # print(len(filtered_idxs))  # TODO
        points = points[filtered_idxs]
        points = points.T  # Transpose the matrix back ([N, 4] to [4, N]).

        return points, filtered_idxs

    def viz_need_or_not(self):
        """
               import matplotlib.pyplot as plt
               fig, ax = plt.subplots(1, 1, figsize=(9, 16))
               ax.scatter(points[0, :], points[1, :], s=0.2)
               plt.gca().set_aspect('equal', adjustable='box')
               plt.show()
               """
        """
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        """


def visualize_semantic_differences_bev(sample_token: str, lidarseg_preds_bin_path: str): #  -> axes.Axes:
    """
    Visualize semantic difference of lidar segmentation results in bird's eye view.
    :param sample_token: Unique identifier.
    :param lidarseg_preds_bin_path: LidarSegmentationResults class.
    :param
    """
    pc = seg_res.data[token].point_cloud
    gt = seg_res.data[token].gt
    est = seg_res.data[token].est

    # red: wrong label, green: correct label
    id2color_for_diff_bev = {0: (191, 41, 0, 255), 1: (50, 168, 82, 255)}
    id2color = {_id: label.color for _id, label in seg_res.labelmap.items()}

    # Plot birdsview gt, est and difference.
    _, axes = plt.subplots(1, 3, figsize=(10 * 3, 10))

    xrange = seg_res.meta.xrange
    yrange = seg_res.meta.yrange

    if class_list is not None:
        mask = np.logical_or(np.isin(gt, class_list), np.isin(est, class_list))
        gt = gt[mask]
        pc = pc[mask]
        est = est[mask]
    pointcloud_gt = LidarPointCloud(np.concatenate((pc.T, np.atleast_2d(gt)), axis=0))
    pointcloud_gt.render_label(axes[0], id2color=id2color, x_lim=xrange, y_lim=yrange)
    pointcloud_est = LidarPointCloud(np.concatenate((pc.T, np.atleast_2d(est)), axis=0))
    pointcloud_est.render_label(axes[1], id2color=id2color, x_lim=xrange, y_lim=yrange)
    pointcloud_diff = LidarPointCloud(np.concatenate((pc.T, np.atleast_2d((gt == est).astype(np.int))), axis=0))
    pointcloud_diff.render_label(axes[2], id2color=id2color_for_diff_bev, x_lim=xrange, y_lim=yrange)

    axes[0].set_title('Raw Point Cloud with GT Labels')
    axes[1].set_title('Raw Point Cloud with EST Labels')
    axes[2].set_title('Errors (Correct: Green, Mislabeled: Red)')

    return axes


if __name__ == '__main__':
    result_path_ = '/home/whye/Desktop/logs/lidarseg_nips20/venice/maplsn_rangeview'
    eval_set_ = 'test'
    dataroot_ = '/data/sets/nuscenes'
    version_ = 'v1.0-test'
    verbose_ = True

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEvalStratified(nusc_, result_path_, eval_set=eval_set_, verbose=verbose_)
    evaluator.evaluate()
