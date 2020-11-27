import os
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.evaluate import LidarSegEval
from nuscenes.eval.lidarseg.utils import ConfusionMatrix, load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud


class LidarSegEvalStratified(LidarSegEval):
    """
    Extends the LidarSegEval class to provide an evaluation which is stratified by radial distance from the ego lidar.
    """
    def __init__(self, nusc: NuScenes,
                 results_folder: str,
                 output_dir: str,
                 eval_set: str,
                 # sample_size: int,
                 strata_list: Tuple[Tuple[float]] = ((0, 20), (20, 40), (40, 60), (60, 80)),
                 # is_render_bad_samples: bool = True,
                 verbose: bool = True):
        """
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder where the results are stored.
        :param output_dir: Folder to save plots and results to.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param strata_list: The strata to evaluate by, in meters.
        :param verbose: Whether to print messages during the evaluation.
        """
        super().__init__(nusc, results_folder, eval_set, verbose)

        self.output_dir = output_dir
        self.strata_list = strata_list

        self.ignore_name = self.mapper.ignore_class['name']

        # Create a list of confusion matrices, one for each strata.
        self.global_cm = [ConfusionMatrix(self.num_classes, self.ignore_idx) for i in range(len(strata_list))]

    def evaluate(self) -> None:
        """
        Performs the actual evaluation. Overwrites the `evaluate` method in the LidarSegEval class.
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
        self.render_stratified_per_class_metrics(stratified_per_class_metrics,
                                                 os.path.join(self.output_dir, 'stratified_iou_per_class.png'))

    def render_stratified_per_class_metrics(self, stratified_per_class_metrics: List[Dict], filename: str) -> None:
        """
        Renders the stratified per class metrics.
        :param stratified_per_class_metrics: A list of dictionaries where each entry corresponds to each strata; each
                                             dictionary contains the iou_per_class, miou, and freq_weighted_iou.
        :param filename: Filename to save the render as.
        """
        stratified_classes = {cls_name: [] for cls_name in self.id2name.values()}
        for strata_metrics in stratified_per_class_metrics:
            for cls, cls_iou in strata_metrics['iou_per_class'].items():
                stratified_classes[cls].append(cls_iou)

        # Delete the ignored class from the dictionary.
        stratified_classes.pop(self.ignore_name, None)

        plot_num_cols = 4
        plot_num_rows = int(np.ceil(len(stratified_classes) / plot_num_cols))
        _, axes = plt.subplots(plot_num_rows, plot_num_cols, figsize=(15 * len(self.strata_list), 300))

        gs = gridspec.GridSpec(plot_num_rows, plot_num_cols)
        fig = plt.figure()
        for n, (cls, cls_strata) in enumerate(stratified_classes.items()):
            ax = fig.add_subplot(gs[n])
            ax.bar([' to '.join(map(str, rng)) for rng in self.strata_list], cls_strata, color='green')
            ax.set_xlabel('Interval', fontsize=3)
            ax.set_ylabel('IOU', fontsize=3)
            ax.set_ylim(top=1.1)  # Make y-axis slightly higher to accommodate tag.
            ax.set_title('Stratified results for {}'.format(cls), fontsize=4)
            ax.tick_params(axis='both', which='major', labelsize=3)

            # Loop to add a tag to each bar.
            for j, rect in enumerate(ax.patches):
                ax.text(rect.get_x() + rect.get_width() / 2., rect.get_y() + rect.get_height() + 0.01,
                        '{:.4f}'.format(cls_strata[j]),
                        ha='center', va='bottom', fontsize=3)

        plt.tight_layout()
        plt.savefig(filename, dpi=250)
        plt.close()

    def get_stratified_per_class_metrics(self) -> List[Dict]:
        """
        Gets the metrics per class for each strata:
        :return: A list of dictionaries where each entry corresponds to each strata; each dictionary contains the
                 iou_per_class, miou, and freq_weighted_iou.
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
        Filters the point cloud such that only points which are within a certain radial range from the ego lidar are
        selected.
        :param points: The point cloud to be filtered.
        :param min_depth: Points to be further than this distance from the ego lidar, in meters.
        :param max_depth: Points to be at most this far from the ego lidar, in meters. If None, then max_depth is
                          effectively the distance of the furthest point from the ego lidar.
        """
        points = points.T  # Transpose the matrix to make manipulation more convenient ([4, N] to [N, 4]).

        depth = np.linalg.norm(points[:, :2], axis=1)

        assert min_depth >= 0, 'Error: min_depth cannot be negative.'
        min_depth_idxs = np.where(depth > min_depth)[0]

        if max_depth is not None:
            assert max_depth >= 0, 'Error: max_depth cannot be negative.'
            max_depth_idxs = np.where(depth <= max_depth)[0]

            filtered_idxs = np.intersect1d(min_depth_idxs, max_depth_idxs)
        else:
            filtered_idxs = min_depth_idxs

        points = points[filtered_idxs]
        points = points.T  # Transpose the matrix back ([N, 4] to [4, N]).

        return points, filtered_idxs


def viz_need_or_not():
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


def visualize_semantic_differences_bev(nusc, sample_token: str, lidarseg_preds_bin_path: str = None):  # -> axes.Axes:
    """
    Visualize semantic difference of lidar segmentation results in bird's eye view.
    :param sample_token: Unique identifier.
    :param lidarseg_preds_bin_path: LidarSegmentationResults class.
    """
    """
    # pc = seg_res.data[token].point_cloud
    pointsensor = self.nusc.get('sample_data', sd_token)
    pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    points = pc.points.T  # [N, 4]
    """
    sample = nusc.get('sample', sample_token)

    # Get the sample data token of the point cloud.
    sd_token = sample['data']['LIDAR_TOP']

    # Init axes.
    fig, ax = plt.subplots(1, 3, figsize=(10 * 3, 10))
    nusc.render_sample_data(sd_token, ax=ax[0], with_anns=True, show_lidarseg=True, show_lidarseg_legend=False)
    nusc.render_sample_data(sd_token, ax=ax[1], with_anns=True, show_lidarseg=True, show_lidarseg_legend=False,)
                            # lidarseg_preds_bin_path=)

    filename = '/home/whye/Desktop/logs/hi2.png'  # TODO
    plt.savefig(filename)


def haha():
    # gt = seg_res.data[token].gt
    # est = seg_res.data[token].est

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
    out_path = '/home/whye/Desktop/logs'
    eval_set_ = 'test'
    dataroot_ = '/data/sets/nuscenes'
    version_ = 'v1.0-test'
    verbose_ = True

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEvalStratified(nusc_, result_path_, out_path, eval_set=eval_set_, verbose=verbose_)
    evaluator.evaluate()
    # visualize_semantic_differences_bev(nusc_, nusc_.sample[0]['token'])
