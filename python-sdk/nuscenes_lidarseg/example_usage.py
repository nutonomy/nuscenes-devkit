# import sys
# sys.path.insert(0, '/home/whyekit/Desktop/nuscenes-devkit/nuscenes/')

from nuscenes import NuScenes


def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/whyekit/Desktop/nuscenes', verbose=True)

    '''
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

    to_check = 310
    print(in_lidarseg[to_check])
    print(in_mini[to_check])

    assert len(in_mini) == count
    print('%d of lidarseg annotations exist in v1.0-mini' % count)

    sample_data_token = in_mini[to_check]['token']
    '''
    sample_data_token = 'd9ee706fc0e1481a82e1d1d2788b38f1'
    nusc.render_sample_data(sample_data_token) #, show_lidarseg_labels=False)


def eg_seperated():
    from nuscenes_lidarseg import NuScenesLidarseg

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/whyekit/Desktop/nuscenes', verbose=True)

    nusc_ls = NuScenesLidarseg(dataroot='/home/whyekit/Desktop/nuscenes-lidarseg',
                               dataroot_nuscenes='/home/whyekit/Desktop/nuscenes')

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