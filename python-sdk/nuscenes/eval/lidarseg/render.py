import argparse
import json
import os
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.evaluate import LidarSegEval
from nuscenes.eval.lidarseg.utils import ConfusionMatrix, LidarsegClassMapper, load_bin_file, LidarSegPointCloud
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend, get_labels_in_coloring
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
                 strata_list: Tuple[Tuple[float]] = ((0, 20), (20, 40), (40, None)),
                 # is_render_bad_samples: bool = True,
                 render_viz: bool = True,
                 verbose: bool = True):
        """
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder where the results are stored.
        :param output_dir: Folder to save plots and results to.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param strata_list: The strata to evaluate by, in meters.
        :param render_viz: Whether to render and save the visualizations to output_dir.
        :param verbose: Whether to print messages during the evaluation.
        """
        super().__init__(nusc, results_folder, eval_set, verbose)

        self.output_dir = output_dir

        self.strata_list = strata_list
        # Get the display names for each strata (e.g. (40, 60) -> '40 to 60'; if the upper bound of a strata is
        # None (e.g. (40, None)), then the display name will be '40+'.
        self.strata_names = ['{}m to {}m'.format(rng[0], rng[1]) if rng[1] is not None else str(rng[0]) + 'm+'
                             for i, rng in enumerate(self.strata_list)]

        self.render_viz = render_viz

        self.ignore_name = self.mapper.ignore_class['name']

        # Create a list of confusion matrices, one for each strata.
        self.global_cm = [ConfusionMatrix(self.num_classes, self.ignore_idx) for i in range(len(strata_list))]

        # After running the evaluation, a list of dictionaries where each entry corresponds to each strata and is of
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
        #     ...  # More strata.
        # ]
        self.stratified_per_class_metrics = None

    def evaluate(self) -> None:
        """
        Performs the actual evaluation. Overwrites the `evaluate` method in the LidarSegEval class.
        """
        for i, strata in enumerate(self.strata_list):
            print('Evaluating for strata {}...'.format(self.strata_names[i]))
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

        self.stratified_per_class_metrics = self.get_stratified_per_class_metrics()
        if self.verbose:
            print(json.dumps(self.stratified_per_class_metrics, indent=4))

        if self.render_viz:
            self.render_stratified_per_class_metrics(os.path.join(self.output_dir, 'stratified_iou_per_class.png'))
            self.render_stratified_overall_metrics(os.path.join(self.output_dir, 'stratified_iou_overall.png'))

    def render_stratified_overall_metrics(self, filename: str, dpi: int = 100) -> None:
        """
        Renders the stratified overall metrics (i.e. the classes are aggregated for each strata).
        :param filename: Filename to save the render as.
        :param dpi: Resolution of the output figure.
        """
        stratified_iou = {strata_name: self.stratified_per_class_metrics[i]['miou'] for i, strata_name in
                          enumerate(self.strata_names)}

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(list(stratified_iou.keys()),
               list(stratified_iou.values()), color='grey')
        ax.set_xlabel('Interval / m', fontsize=15)
        ax.set_ylabel('mIOU', fontsize=15)
        ax.set_ylim(top=1.1)  # Make y-axis slightly higher to accommodate tag.
        ax.set_title('Stratified results for mIOU', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Loop to add a tag to each bar.
        for j, rect in enumerate(ax.patches):
            ax.text(rect.get_x() + rect.get_width() / 2., rect.get_y() + rect.get_height() + 0.01,
                    '{:.4f}'.format(list(stratified_iou.values())[j]),
                    ha='center', va='bottom', fontsize=15)

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def render_stratified_per_class_metrics(self, filename: str, dpi: int = 150) -> None:
        """
        Renders the stratified per class metrics.
        :param filename: Filename to save the render as.
        :param dpi: Resolution of the output figure.
        """
        stratified_classes = {cls_name: [] for cls_name in self.id2name.values()}
        for strata_metrics in self.stratified_per_class_metrics:
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
            ax.bar(self.strata_names, cls_strata, color=np.array(self.mapper.coarse_colormap[cls]) / 255)
            ax.set_xlabel('Interval / m', fontsize=3)
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
        plt.savefig(filename, dpi=dpi)
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


def visualize_semantic_differences_bev(nusc,
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
    """
        if class_list is not None:
        mask = np.logical_or(np.isin(gt, class_list), np.isin(est, class_list))
        gt = gt[mask]
        pc = pc[mask]
        est = est[mask]
    """
    ignored_points_idxs = np.where(gt.labels != mapper.ignore_class['index'])[0]
    gt.labels = gt.labels[ignored_points_idxs]
    gt.points = gt.points[ignored_points_idxs]
    preds.labels = preds.labels[ignored_points_idxs]
    preds.points = preds.points[ignored_points_idxs]

    # Init axes.
    fig, axes = plt.subplots(1, 3, figsize=(10 * 3, 10), sharex='all', sharey='all')

    gt.render(mapper.coarse_colormap, mapper.coarse_name_2_coarse_idx_mapping, ax=axes[0])
    preds.render(mapper.coarse_colormap, mapper.coarse_name_2_coarse_idx_mapping, ax=axes[1])

    # Render errors.
    id2color_for_diff_bev = {0: (191, 41, 0, 255),  # red: wrong label
                             1: (50, 168, 82, 255)}  # green: correct label
    colors_for_diff_bev = colormap_to_colors(id2color_for_diff_bev, {0: 0, 1: 1})
    mask = np.array(gt.labels == preds.labels).astype(int)  # need to convert array from bool to int
    axes[2].scatter(gt.points[:, 0], gt.points[:, 1], c=colors_for_diff_bev[mask], s=dot_size)
    axes[2].set_title('Errors (Correct: Green, Mislabeled: Red)')

    # Limit visible range for all subplots.
    plt.xlim(-axes_limit, axes_limit)
    plt.ylim(-axes_limit, axes_limit)

    if out_path:
        plt.savefig(out_path)

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

    """
    result_path_ = '/home/whye/Desktop/logs/lidarseg_nips20/venice/maplsn_rangeview'
    out_path_ = '/home/whye/Desktop/logs'
    eval_set_ = 'test'
    dataroot_ = '/data/sets/nuscenes'
    version_ = 'v1.0-test'
    verbose_ = True
    """

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEvalStratified(nusc_,
                                       result_path_,
                                       out_path_,
                                       eval_set=eval_set_,
                                       # strata_list=???,
                                       # render_viz=render_viz,
                                       verbose=verbose_)
    evaluator.evaluate()
    # visualize_semantic_differences_bev(nusc_, nusc_.sample[0]['token'],
    #                                    # '/home/whye/Desktop/logs/lidarseg_nips20/venice/fusion_train/lidarseg/test/')
    #                                    '/home/whye/Desktop/logs/lidarseg_nips20/others/d625f4a4-bb7f-4ccd-9189-306b1ff5c6eb/lidarseg/test',
    #                                    out_path='/home/whye/Desktop/logs/hi2.png')
