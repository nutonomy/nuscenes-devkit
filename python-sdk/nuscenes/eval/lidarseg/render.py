import argparse
import json
import os
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.evaluate import LidarSegEval
from nuscenes.eval.lidarseg.utils import ConfusionMatrix, LidarsegClassMapper
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors
from nuscenes.utils.data_classes import LidarSegPointCloud


class LidarSegEvalStratified(LidarSegEval):
    """
    Extends the LidarSegEval class to provide an evaluation which is stratified by radial distance from the ego lidar.
    """
    def __init__(self, nusc: NuScenes,
                 results_folder: str,
                 output_dir: str,
                 eval_set: str,
                 strata_list: Tuple[Tuple[float]] = ((0, 10), (10, 20), (20, 30), (30, 40), (40, None)),
                 verbose: bool = True):
        """
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder where the results are stored.
        :param output_dir: Folder to save plots and results to.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param strata_list: The strata to evaluate on, in meters.
        :param verbose: Whether to print messages during the evaluation.
        """
        super().__init__(nusc, results_folder, eval_set, verbose)

        self.output_dir = output_dir
        assert os.path.exists(self.output_dir), 'Error: {} does not exist.'.format(self.output_dir)

        self.strata_list = strata_list
        # Get the display names for each stratum (e.g. (40, 60) -> '40 to 60'; if the upper bound of a stratum is
        # None (e.g. (40, None)), then the display name will be '40+'.
        self.strata_names = ['{}m to {}m'.format(stratum[0], stratum[1])
                             if stratum[1] is not None else str(stratum[0]) + 'm+'
                             for i, stratum in enumerate(self.strata_list)]

        self.ignore_name = self.mapper.ignore_class['name']

        # Create a list of confusion matrices, one for each stratum.
        self.global_cm = [ConfusionMatrix(self.num_classes, self.ignore_idx) for i in range(len(strata_list))]

        # After running the evaluation, a list of dictionaries where each entry corresponds to each stratum and is of
        # the following format:
        # [
        #     {
        #         "iou_per_class": {
        #             "ignore": NaN,
        #             "class_a": 0.8042956640279008,
        #             ...
        #         },
        #         "miou": 0.7626268050947295,
        #         "freq_weighted_iou": 0.8906460292535451
        #     },
        #     ...  # More stratum.
        # ]
        self.stratified_per_class_metrics = None

    def evaluate(self) -> None:
        """ Performs the actual evaluation. Overwrites the `evaluate` method in the LidarSegEval class. """
        for i, stratum in enumerate(self.strata_list):
            if self.verbose:
                print('Evaluating for stratum {}...'.format(self.strata_names[i]))
            for sample_token in tqdm(self.sample_tokens, disable=not self.verbose):
                sample = self.nusc.get('sample', sample_token)

                # 1. Get the sample data token of the point cloud.
                sd_token = sample['data']['LIDAR_TOP']
                pointsensor = self.nusc.get('sample_data', sd_token)
                pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])

                # 2. Load the ground truth labels for the point cloud.
                gt_path = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', sd_token)['filename'])
                gt = LidarSegPointCloud(pcl_path, gt_path)
                gt.labels = self.mapper.convert_label(gt.labels)  # Map the labels as necessary.

                # 3. Load the predictions for the point cloud.
                pred_path = os.path.join(self.results_folder, 'lidarseg', self.eval_set, sd_token + '_lidarseg.bin')
                pred = LidarSegPointCloud(pcl_path, pred_path)

                # 4. Filter to get only labels belonging to the stratum.
                gt = self.filter_pointcloud_by_depth(gt, min_depth=stratum[0], max_depth=stratum[1])
                pred = self.filter_pointcloud_by_depth(pred, min_depth=stratum[0], max_depth=stratum[1])

                # 5. Update the confusion matrix for the sample data into the confusion matrix for the eval set.
                self.global_cm[i].update(gt.labels, pred.labels)

        self.stratified_per_class_metrics = self.get_stratified_per_class_metrics()
        if self.verbose:
            print(json.dumps(self.stratified_per_class_metrics, indent=4))

        if self.output_dir:
            self.render_stratified_per_class_metrics(os.path.join(self.output_dir, 'stratified_iou_per_class.png'))
            self.render_stratified_overall_metrics(os.path.join(self.output_dir, 'stratified_iou_overall.png'))

    def render_stratified_overall_metrics(self, filename: str, dpi: int = 100) -> None:
        """
        Renders the stratified overall metrics (i.e. the classes are aggregated for each stratum).
        :param filename: Filename to save the render as.
        :param dpi: Resolution of the output figure.
        """
        stratified_iou = {stratum_name: self.stratified_per_class_metrics[i]['miou']
                          for i, stratum_name in enumerate(self.strata_names)}

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(list(stratified_iou.keys()),
               list(stratified_iou.values()), color='grey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels(self.strata_names, rotation=45, horizontalalignment='right')
        ax.set_ylabel('mIOU', fontsize=15)
        ax.set_ylim(top=1.1)  # Make y-axis slightly higher to accommodate tag.
        ax.set_title('Distance vs. mIOU for all classes', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Loop to add a tag to each bar.
        for j, rect in enumerate(ax.patches):
            ax.text(rect.get_x() + rect.get_width() / 2., rect.get_y() + rect.get_height() + 0.01,
                    '{:.4f}'.format(list(stratified_iou.values())[j]),
                    ha='center', va='bottom', fontsize=15)

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)

        if self.verbose:
            plt.show()

    def render_stratified_per_class_metrics(self, filename: str, dpi: int = 250) -> None:
        """
        Renders the stratified per class metrics.
        :param filename: Filename to save the render as.
        :param dpi: Resolution of the output figure.
        """
        stratified_classes = {cls_name: [] for cls_name in self.id2name.values()}
        for stratum_metrics in self.stratified_per_class_metrics:
            for cls, cls_iou in stratum_metrics['iou_per_class'].items():
                stratified_classes[cls].append(cls_iou)

        # Delete the ignored class from the dictionary.
        stratified_classes.pop(self.ignore_name, None)

        plot_num_cols = 4
        plot_num_rows = int(np.ceil(len(stratified_classes) / plot_num_cols))

        gs = gridspec.GridSpec(plot_num_rows, plot_num_cols)
        fig = plt.figure()
        for n, (cls, cls_strata) in enumerate(stratified_classes.items()):
            ax = fig.add_subplot(gs[n])
            ax.bar(self.strata_names, cls_strata, color=np.array(self.mapper.coarse_colormap[cls]) / 255)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticklabels(self.strata_names, rotation=45, horizontalalignment='right')
            ax.set_ylabel('IOU', fontsize=3)
            ax.set_ylim(top=1.1)  # Make y-axis slightly higher to accommodate tag.
            ax.set_title('Distance vs. IOU for {}'.format(cls), fontsize=4)
            ax.tick_params(axis='both', which='major', labelsize=3)

            # Loop to add a tag to each bar.
            for j, rect in enumerate(ax.patches):
                ax.text(rect.get_x() + rect.get_width() / 2., rect.get_y() + rect.get_height() + 0.01,
                        '{:.4f}'.format(cls_strata[j]),
                        ha='center', va='bottom', fontsize=3)

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)

        if self.verbose:
            plt.show()

    def get_stratified_per_class_metrics(self) -> List[Dict]:
        """
        Gets the metrics per class for each stratum:
        :return: A list of dictionaries where each entry corresponds to each stratum; each dictionary contains the
                 iou_per_class, miou, and freq_weighted_iou.
        """
        stratified_per_class_iou = []
        for stratum_cm in self.global_cm:
            iou_per_class = stratum_cm.get_per_class_iou()
            miou = stratum_cm.get_mean_iou()
            freqweighted_iou = stratum_cm.get_freqweighted_iou()

            # Put everything nicely into a dict.
            results = {'iou_per_class': {self.id2name[i]: class_iou for i, class_iou in enumerate(iou_per_class)},
                       'miou': miou,
                       'freq_weighted_iou': freqweighted_iou}

            stratified_per_class_iou.append(results)

        return stratified_per_class_iou

    @staticmethod
    def filter_pointcloud_by_depth(pc: LidarSegPointCloud,
                                   min_depth: float = 0,
                                   max_depth: float = None) -> LidarSegPointCloud:
        """
        Filters the point cloud such that only points which are within a certain radial range from the ego lidar are
        selected.
        :param pc: The point cloud to be filtered.
        :param min_depth: Points to be further than this distance from the ego lidar, in meters.
        :param max_depth: Points to be at most this far from the ego lidar, in meters. If None, then max_depth is
                          effectively the distance of the furthest point from the ego lidar.
        :return: The filtered point cloud.
        """
        depth = np.linalg.norm(pc.points[:, :2], axis=1)

        assert min_depth >= 0, 'Error: min_depth cannot be negative.'
        min_depth_idxs = np.where(depth > min_depth)[0]

        # Get the indices of the points belonging to the radial range.
        if max_depth is not None:
            assert max_depth >= 0, 'Error: max_depth cannot be negative.'
            max_depth_idxs = np.where(depth <= max_depth)[0]

            filtered_idxs = np.intersect1d(min_depth_idxs, max_depth_idxs)
        else:
            filtered_idxs = min_depth_idxs

        pc.points, pc.labels = pc.points[filtered_idxs], pc.labels[filtered_idxs]

        return pc


def visualize_semantic_differences_bev(nusc: NuScenes,
                                       sample_token: str,
                                       lidarseg_preds_folder: str = None,
                                       axes_limit: float = 40,
                                       dot_size: int = 5,
                                       out_path: str = None) -> None:
    """
    Visualize semantic difference of lidar segmentation results in bird's eye view.
    :param nusc: A NuScenes object.
    :param sample_token: Unique identifier.
    :param lidarseg_preds_folder: A path to the folder which contains the user's lidar segmentation predictions for
                                  the scene. The naming convention of each .bin file in the folder should be
                                  named in this format: <lidar_sample_data_token>_lidarseg.bin.
    :param axes_limit: Axes limit for plot (measured in meters).
    :param dot_size: Scatter plot dot size.
    :param out_path: Path to save visualization to (e.g. /save/to/here/bev_diff.png).
    """
    mapper = LidarsegClassMapper(nusc)

    sample = nusc.get('sample', sample_token)

    # Get the sample data token of the point cloud.
    sd_token = sample['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', sd_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])

    # Load the ground truth labels for the point cloud.
    gt_path = os.path.join(nusc.dataroot, nusc.get('lidarseg', sd_token)['filename'])
    gt = LidarSegPointCloud(pcl_path, gt_path)
    gt.labels = mapper.convert_label(gt.labels)  # Map the labels as necessary.

    # Load the predictions for the point cloud.
    preds_path = os.path.join(lidarseg_preds_folder, sd_token + '_lidarseg.bin')
    preds = LidarSegPointCloud(pcl_path, preds_path)

    # Do not compare points which are ignored.
    ignored_points_idxs = np.where(gt.labels != mapper.ignore_class['index'])[0]
    gt.labels = gt.labels[ignored_points_idxs]
    gt.points = gt.points[ignored_points_idxs]
    preds.labels = preds.labels[ignored_points_idxs]
    preds.points = preds.points[ignored_points_idxs]

    # Init axes.
    fig, axes = plt.subplots(1, 3, figsize=(10 * 3, 10), sharex='all', sharey='all')

    # Render ground truth and predictions.
    gt.render(mapper.coarse_colormap, mapper.coarse_name_2_coarse_idx_mapping, ax=axes[0])
    preds.render(mapper.coarse_colormap, mapper.coarse_name_2_coarse_idx_mapping, ax=axes[1])

    # Render errors.
    id2color_for_diff_bev = {0: (191, 41, 0, 255),  # red: wrong label
                             1: (50, 168, 82, 255)}  # green: correct label
    colors_for_diff_bev = colormap_to_colors(id2color_for_diff_bev, {0: 0, 1: 1})
    mask = np.array(gt.labels == preds.labels).astype(int)  # Convert array from bool to int.
    axes[2].scatter(gt.points[:, 0], gt.points[:, 1], c=colors_for_diff_bev[mask], s=dot_size)
    axes[2].set_title('Errors (Correct: Green, Mislabeled: Red)')

    # Limit visible range for all subplots.
    plt.xlim(-axes_limit, axes_limit)
    plt.ylim(-axes_limit, axes_limit)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes-lidarseg results.')
    parser.add_argument('--result_path', type=str,
                        help='The path to the results folder.')
    parser.add_argument('--out_path', type=str,
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = args.result_path
    out_path_ = args.out_path
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = args.verbose

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEvalStratified(nusc_,
                                       result_path_,
                                       out_path_,
                                       eval_set=eval_set_,
                                       verbose=verbose_)
    evaluator.evaluate()
