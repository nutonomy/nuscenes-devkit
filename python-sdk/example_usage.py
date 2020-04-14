print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import os

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

    sample_token = '9c7c7d5d109c40fcaecd3c422d37b4f6'
    nusc.render_pointcloud_in_image(sample_token,
                                    pointsensor_channel='LIDAR_TOP',
                                    camera_channel='CAM_FRONT',
                                    render_intensity=True,
                                    show_lidarseg_labels=True,
                                    filter_lidarseg_labels=[32, 1]
                                    )

    import sys
    sys.exit()

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

# def splitter(lidarseg_full):


def qwert():
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/whye/Desktop/nuscenes_o', verbose=True)
    try_single = False

    if not try_single:
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

        # to_check = 257
        to_check = 63
        print(in_lidarseg[to_check])
        print(in_mini[to_check])
        sample = nusc.get('sample', in_mini[to_check]['sample_token'])
        print (sample)
        scene = nusc.get('scene', sample['scene_token'])
        print(scene)
        print(scene['name'])

        assert len(in_mini) == count
        print('%d of lidarseg annotations exist in v1.0-mini' % count)

    if not try_single:
        sample_data_token = in_mini[to_check]['token']
    else:
        sample_data_token = 'd9ee706fc0e1481a82e1d1d2788b38f1'
    # nusc.render_sample_data(sample_data_token,
    #                         show_lidarseg_labels=True)

    if not try_single:
        sample_token = in_mini[to_check]['sample_token']
    else:
        sample_token = '9c7c7d5d109c40fcaecd3c422d37b4f6'
    nusc.render_pointcloud_in_image(sample_token,
                                    pointsensor_channel='LIDAR_TOP',
                                    # pointsensor_channel='RADAR_FRONT',
                                    camera_channel='CAM_FRONT',
                                    render_intensity=True,
                                    show_lidarseg_labels=True)


def eg_seperated():
    from nuscenes_lidarseg import NuScenesLidarseg

    nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)

    nusc_ls = NuScenesLidarseg(dataroot='/data/sets/nuscenes-lidarseg',
                               dataroot_nuscenes='/data/sets/Desktop/nuscenes')

    nusc_ls.list_lidarseg_categories()

    lidar_seg_annots = nusc_ls.lidarseg

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

    to_check = 310
    print(in_lidarseg[to_check])
    print(in_mini[to_check])

    assert len(in_mini) == count
    print('%d of lidarseg annotations exist in v1.0-mini' % count)

    sample_data_token = in_mini[to_check]['token']

    nusc_ls.render_sample_lidarseg_data(sample_data_token)


if __name__ == '__main__':
    main()
    # qwert()