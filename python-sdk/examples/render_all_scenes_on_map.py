import argparse

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sklearn.metrics  # Requires sklearn. Please run `pip install sklearn` inside your virtualenv.

from nuscenes_utils.nuscenes import NuScenes


def render_all_scenes_on_map(nusc: NuScenes, log_location: str) -> None:
    """
    Renders the ego poses for all scenes on a particular map. Also counts the number of ego poses that were on the
    semantic prior area (drivable surface + sidewalks).
    More general version of NuScenes.render_scene_on_map().
    :param nusc: NuScenes object.
    :param log_location: Name of the location, e.g. "singapore-onenorth", "boston-seaport".
    """

    # For the purpose of this demo, subsample the mask by a factor of 25.
    demo_ss_factor = 25.0

    on_drivable_cnt = 0

    log_tokens = [l['token'] for l in nusc.log if l['location'] == log_location]
    scene_tokens = [s['token'] for s in nusc.scene if s['log_token'] in log_tokens]

    map_poses = []

    for i, scene_token in enumerate(scene_tokens):
        print('Processing scene %d of %d...' % (i + 1, len(scene_tokens)))

        # Get records from NuScenes database.
        scene_record = nusc.get('scene', scene_token)
        log_record = nusc.get('log', scene_record['log_token'])
        map_record = nusc.get('map', log_record['map_token'])

        # map_record['mask'].mask holds a MapMask instance that we need below.
        map_mask = map_record['mask']

        # Now draw the map mask


        # For each sample in the scene, plot the ego pose.
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
        for sample_token in sample_tokens:
            sample_record = nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the LIDAR_TOP sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])

            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            # Recover the ego pose. A 1 is added at the end to make it homogenous coordinates.
            pose = np.array(pose_record['translation'] + [1])

            # Calculate the pose on the map.
            map_pose = np.dot(map_mask.transform_matrix, pose) / demo_ss_factor
            map_pose = map_pose[:2]
            map_poses.append(map_pose)

            # Check if outside semantic prior area.
            on_drivable_cnt += map_mask.is_on_mask(pose[0], pose[1])

    # Compute number of close ego poses
    map_poses = np.vstack(map_poses)
    dists = sklearn.metrics.pairwise.euclidean_distances(map_poses, map_poses)
    close_dist = 50
    close_poses = np.sum(dists < close_dist, axis=0)

    # Plot
    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    mask = Image.fromarray(map_mask.mask)
    axes.imshow(mask.resize((int(mask.size[0] / demo_ss_factor), int(mask.size[1] / demo_ss_factor)),
                            resample=Image.NEAREST))
    title = 'Number of ego poses within {}m in {}'.format(close_dist, log_location)
    axes.set_title(title)
    sc = axes.scatter(map_poses[:, 0], map_poses[:, 1], c=close_poses)
    plt.colorbar(sc)
    plt.show()
    import pdb; pdb.set_trace()

    print('For location {}, {} ego poses ({:.1f}%) were on the semantic prior area'.format(
        log_location, on_drivable_cnt, 100 * on_drivable_cnt / len(sample_tokens)))


if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description='Display all ego pose locations on the specified maps.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_location', default='singapore-onenorth', type=str,
                        help='Log location, e.g. singapore-onenorth')
    args = parser.parse_args()

    nusc = NuScenes()
    render_all_scenes_on_map(nusc, log_location=args.log_location)
