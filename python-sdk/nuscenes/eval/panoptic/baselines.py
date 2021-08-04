from nuscenes.nuscenes import NuScenes
from nuscenes.eval.panoptic.evaluate import NuScenesPanopticEval
from nuscenes.eval.panoptic.merge_lidarseg_and_tracking import generate_panoptic_labels


def get_baseline(nusc: NuScenes,
                 path_to_lidarseg_preds: str,
                 path_to_tracking_preds: str,
                 out_dir: str,
                 verbose: bool = False):
    """
    # TODO
    """
    generate_panoptic_labels(nusc=nusc,
                             seg_folder=path_to_lidarseg_preds,
                             track_json=path_to_tracking_preds,
                             eval_set=nusc.version,
                             out_dir=out_dir,
                             verbose=verbose)


def main(out_dir: str, version: str, dataroot: str = '/data/sets/nuscenes', verbose: bool = False,
         evaluate: bool = True):
    """
    # TODO
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

    print('Fuse predictions from lidarseg and tracking to get predictions for panoptic tracking')
    # get_baselines(nusc, )
    print(f'Panoptic tracking predictions saved at {out_dir}. \nFinished.')

    if evaluate:
        NuScenesPanopticEval(nusc=nusc,
                             results_folder='/home/whye/Desktop/logs/panoptic/',
                             eval_set='test',
                             task='tracking',  # OR 'segmentation',
                             min_inst_points=15,  # default
                             out_dir='/home/whye/Desktop/logs/panoptic/panoptic_eval',
                             verbose=verbose)


# if __name__ == '__main__':
#     main()
