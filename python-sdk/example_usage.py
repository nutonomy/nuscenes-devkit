print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import os

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm

def main():
    root = '/home/whye/Desktop/nuscenes_o'
    nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
    # nusc.list_categories()
    # print (nusc.lidarseg[0])
    # print (nusc.sample_annotation[0])

    # for classname, freq in sorted(lidarseg_counts.items()):
    #     print('{:27} nbr_points={:9}'.format(classname[:27], freq))

    '''
    {
        "token": "36b7b02f0f034f0595e3437a85554151",
        "log_token": "08ba46dd716d42a69d108638fef5bbb9",
        "nbr_samples": 40,
        "first_sample_token": "305227a63e184c378cc9e36fd60382ca",
        "last_sample_token": "b8fc5f72c1d84318a66ff9dac7308fef",
        "name": "scene-0248",
        "description": "Wait at intersection, many peds"
    },
    '''

    # sample_data_token = 'd9ee706fc0e1481a82e1d1d2788b38f1'

    my_scene = nusc.get('scene', "36b7b02f0f034f0595e3437a85554151")
    my_sample = nusc.get('sample', my_scene['first_sample_token'])
    sample_data_token = my_sample['data']['LIDAR_TOP']

    def get_stats(points_label):
        lidarseg_counts = [0] * (max(points_label) + 1)

        indices = np.bincount(points_label)
        ii = np.nonzero(indices)[0]

        for class_idx, class_count in zip(ii, indices[ii]):
            # print(class_idx, class_count)
            lidarseg_counts[class_idx] += class_count  # increment the count for the particular class name
        # print(lidarseg_counts)

        return lidarseg_counts

    lidarseg_labels_filename = os.path.join('/home/whye/Desktop/nuscenes_o', 'lidarseg', sample_data_token + '_lidarseg.bin')
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
    out = get_stats(points_label)
    print(out)
    print(len(out))

    nusc.render_sample_data(sample_data_token, show_lidarseg_labels=True, underlay_map=True, with_anns=True)


def render_scene_with_pointclouds(nusc, scene, camera_channel, filter_lidarseg_labels, out_folder) -> None:
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    total_num_samples = scene['nbr_samples']
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']

    current_token = first_sample_token
    keep_looping = True
    i = 0
    while keep_looping:
        if current_token == last_sample_token:
            keep_looping = False

        sample_record = nusc.get('sample', current_token)

        # ---------- get filename of image ----------
        camera_token = sample_record['data'][camera_channel]
        cam = nusc.get('sample_data', camera_token)
        filename = os.path.basename(cam['filename'])
        # ---------- /get filename of image ----------

        # ---------- render lidarseg labels in image ----------
        nusc.render_pointcloud_in_image(sample_record['token'],
                                        pointsensor_channel='LIDAR_TOP',
                                        camera_channel=camera_channel,
                                        render_intensity=False,
                                        show_lidarseg_labels=True,
                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                        out_path=os.path.join(out_folder, filename))
        plt.close('all')  # To prevent figures from accumulating in memory
        # ---------- /render lidarseg labels in image ----------

        next_token = sample_record['next']
        current_token = next_token

        i += 1

    print(total_num_samples, i)
    assert total_num_samples == i


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


def test_viz(nusc):
    in_mini, in_lidarseg = make_mini_from_lidarseg(nusc)

    to_check = 63
    sample_token = get_single_sample_token(nusc, in_mini, to_check)
    # sample_token = '9c7c7d5d109c40fcaecd3c422d37b4f6'

    # ---------- render lidarseg labels in BEV of pc ----------
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['LIDAR_TOP']
    # sample_data_token = "b367d4bddc8641b7bc69d7566d126f28"  # CAM_FRONT_LEFT
    # sample_data_token = "03be4e37936943d2bd991b5351baf82c"  # CAM_BACK
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
                                    out_path=os.path.expanduser('~/Desktop/test2.png'))
    # ---------- /render lidarseg labels in image ----------

    # ---------- render sample (i.e. lidar, radar and all cameras) ----------
    # note: lidarseg labels will not be rendered
    nusc.render_sample(sample_token, out_path=os.path.expanduser('~/Desktop/test3.png'))
    # ---------- /render sample (i.e. lidar, radar and all cameras) ----------


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/whye/Desktop/nuscenes_o', verbose=True)

    # render_scene_with_pointclouds(nusc, nusc.scene[0], 'CAM_BACK', [32, 1, 36],
    #                               os.path.expanduser('~/Desktop/CAM_BACK'))

    # main()
    test_viz(nusc)