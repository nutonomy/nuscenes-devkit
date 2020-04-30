print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import argparse
import json
import os

from nuscenes import NuScenes


def main(nusc):
    in_mini, in_lidarseg = make_mini_from_lidarseg(nusc)

    to_check = 8
    sample_token = get_single_sample_token(nusc, in_mini, to_check)
    # sample_token = '9c7c7d5d109c40fcaecd3c422d37b4f6'

    # nusc.render_scene_channel(nusc.scene[-1]['token'], 'CAM_FRONT', (1280, 720))
    # nusc.render_scene(nusc.scene[0]['token'])

    mapping_temp = dict()  # Psuedo mapping
    for i in range(41):
        mapping_temp[i] = str(i)
    nusc.get_sample_lidarseg_stats(sample_token, mapping_temp, sort_counts=True)
    quit()
    # ---------- render lidarseg labels in BEV of pc ----------
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['LIDAR_TOP']

    # sample_data_token = "b367d4bddc8641b7bc69d7566d126f28"  # CAM_FRONT_LEFT
    # sample_data_token = "03be4e37936943d2bd991b5351baf82c"  # CAM_BACK
    # sample_data_token = "2abaed501018421fb4e6adc52b99db12"  # LIDAR but sample_data_token is not from a key_frame

    nusc.render_sample_data(sample_data_token,
                            show_lidarseg_labels=True,
                            filter_lidarseg_labels=[32, 1],
                            out_path=os.path.expanduser('~/Desktop/test1.png'))
    # ---------- /render lidarseg labels in BEV of pc ----------

    # ---------- render lidarseg labels in image ----------
    nusc.render_pointcloud_in_image(sample_token,
                                    pointsensor_channel='LIDAR_TOP',
                                    camera_channel='CAM_FRONT',
                                    render_intensity=True,
                                    show_lidarseg_labels=True,
                                    filter_lidarseg_labels=[32, 1],
                                    out_path=os.path.expanduser('~/Desktop/test2.png'),
                                    render_if_no_points=False,
                                    verbose=True)
    # ---------- /render lidarseg labels in image ----------

    # ---------- render sample (i.e. lidar, radar and all cameras) ----------
    nusc.render_sample(sample_token, out_path=os.path.expanduser('~/Desktop/test3.png'),
                       show_lidarseg_labels=True, verbose=False)
    # ---------- /render sample (i.e. lidar, radar and all cameras) ----------

    # ---------- render scene for a given cam sensor with lidarseg labels ----------
    nusc.render_camera_channel_with_pointclouds(nusc.scene[0]['token'], 'CAM_BACK',
                                                out_folder=os.path.expanduser('~/Desktop/my_rendered_scene.avi'),
                                                filter_lidarseg_labels=[6],  # [32, 1],
                                                render_if_no_points=True,
                                                verbose=True,
                                                imsize=(1280, 720))
    # ---------- /render scene for a given cam sensor with lidarseg labels ----------

    # ---------- render scene for all cameras with lidarseg labels ----------
    nusc.render_scene_with_pointclouds_for_all_cameras(nusc.scene[0]['token'],
                                                       out_path=os.path.expanduser('~/Desktop/all_cams_lidarseg.avi'),
                                                       filter_lidarseg_labels=[32, 1],
                                                       imsize=(640, 360))
    # ---------- /render scene for all cameras with lidarseg labels ----------


def load_table(path_to_json) -> dict:
    """ Loads a table. """
    with open(path_to_json) as f:
        table = json.load(f)
    return table


def make_mini_from_lidarseg(nusc):
    lidar_seg_annots = nusc.lidarseg

    in_mini = []
    in_lidarseg = []

    count = 0
    for i in range(len(lidar_seg_annots)):
        try_lidar_tok = lidar_seg_annots[i]['sample_data_token']

        try:
            entry = nusc.get('sample_data', try_lidar_tok)
            in_mini.append(entry)
            in_lidarseg.append((lidar_seg_annots[i]))
            count += 1
        except:
            continue

    assert len(in_mini) == count
    print('%d of lidarseg annotations exist in v1.0-mini' % count)

    return in_mini, in_lidarseg


def get_single_sample_token(nusc, in_mini, to_check=257):
    # print(in_lidarseg[to_check])
    print(in_mini[to_check])

    sample = nusc.get('sample', in_mini[to_check]['sample_token'])
    # print(sample)
    scene = nusc.get('scene', sample['scene_token'])
    # print(scene)
    print(scene['name'])

    sample_token = in_mini[to_check]['sample_token']

    return sample_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test drive the visualizations for the nuScenes-lidarseg devkit')
    parser.add_argument('-d', '--data_dir', type=str, default='/data/sets/nuscenes',
                        help='Path to where the lidarseg dataset is.')
    parser.add_argument('-v', '--version', type=str, default='v1.0-mini',
                        help='Version of nuScenes; only v1.0-trainval, v1.0-test or v1.0-mini.')
    args = parser.parse_args()

    nusc_obj = NuScenes(version=args.version, dataroot=args.data_dir, verbose=True)

    main(nusc_obj)
