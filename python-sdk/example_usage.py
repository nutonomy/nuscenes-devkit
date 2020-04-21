print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from functools import partial
import json
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm


def main(nusc_class):
    # nusc.list_categories()
    # print (nusc.lidarseg[0])
    # print (nusc.sample_annotation[0])

    # for classname, freq in sorted(lidarseg_counts.items()):
    #     print('{:27} nbr_points={:9}'.format(classname[:27], freq))

    classes = [40, 41]  # [40, 41]
    out_folder = os.path.expanduser('~/Desktop/for_VOs')

    for class_to_render in classes:
        print('Checking {}...'.format(class_to_render))
        out_folder_class = os.path.join(out_folder, str(class_to_render))
        if not os.path.exists(out_folder_class):
            os.makedirs(out_folder_class)

        render_cams_with_lidarseg_for_all_scenes(nusc_class, out_folder_class, [class_to_render])


def load_table(path_to_json) -> dict:
    """ Loads a table. """
    with open(path_to_json) as f:
        table = json.load(f)
    return table


def render_cams_with_lidarseg_for_all_scenes(nusc, out_root, filter_classes=None, do_multiprocessing=True) -> None:
    assert os.path.isdir(out_root), 'ERROR: {} does not exist.'.format(out_root)

    cam_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    for cam_channel in cam_channels:
        print('Working on {}...'.format(cam_channel))
        out_subfolder = os.path.join(out_root, cam_channel)

        start_time = time.time()
        procs = []
        func = partial(render_scene_channel_with_pointclouds,
                       nusc=nusc, camera_channel=cam_channel,
                       filter_lidarseg_labels=filter_classes,
                       out_folder=out_subfolder)

        for scene_entry in nusc.scene:
            if do_multiprocessing:
                proc = mp.Process(target=func, args=(scene_entry['token'],))
                procs.append(proc)
                proc.start()
            else:
                render_scene_channel_with_pointclouds(scene_entry['token'], nusc, cam_channel, filter_classes,
                                                      out_subfolder)

        if do_multiprocessing:
            for proc in procs:
                proc.join()
                proc.close()

        num_converted = len([name for name in os.listdir(out_subfolder)])

        print('Rendered {} images for {} in {:.3f} minutes'.format(
            num_converted, cam_channel, (time.time() - start_time) / 60))

        samples_folder = os.path.join(nusc.dataroot, 'samples', cam_channel)
        num_originals = len([name for name in os.listdir(samples_folder)
                             if os.path.splitext(name)[1] in ['.jpg', '.png', '.JPG', '.PNG']])

        # TODO uncomment after done with VOs
        # assert num_converted == num_originals, 'ERROR: There were {} originals in {} but {} were converted and ' \
        #                                        'stored at {}. Pls check.'.format(num_originals, samples_folder,
        #                                                                          num_converted, out_subfolder)


def render_scene_channel_with_pointclouds(scene_token, nusc, camera_channel, filter_lidarseg_labels,
                                          out_folder) -> None:
    valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    assert camera_channel in valid_channels, 'Input camera channel {} not valid.'.format(camera_channel)

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    scene_record = nusc.get('scene', scene_token)

    total_num_samples = scene_record['nbr_samples']
    first_sample_token = scene_record['first_sample_token']
    last_sample_token = scene_record['last_sample_token']

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

    assert total_num_samples == i, 'ERROR: There were supposed to be {} frames, ' \
                                   'but only {} frames were rendered'.format(total_num_samples, i)


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
                                    out_path=os.path.expanduser('~/Desktop/test2.png'))
    # ---------- /render lidarseg labels in image ----------

    # ---------- render sample (i.e. lidar, radar and all cameras) ----------
    nusc.render_sample(sample_token, out_path=os.path.expanduser('~/Desktop/test3.png'),
                       show_lidarseg_labels=True)
    # ---------- /render sample (i.e. lidar, radar and all cameras) ----------

    # ---------- render scene for a given sensor ----------
    nusc.render_scene_channel(nusc.scene[0]['token'],
                              channel='CAM_FRONT',
                              out_path=os.path.expanduser('~/Desktop/test4.avi'))
    # ---------- /render scene for a given sensor ----------

    # ---------- render scene for a given cam sensor with lidarseg labels ----------
    render_scene_channel_with_pointclouds(nusc.scene[0]['token'], nusc, 'CAM_FRONT_LEFT', [32, 1, 36],
                                          os.path.expanduser('~/Desktop/CAM_FRONT_LEFT'))
    # ---------- /render scene for a given cam sensor with lidarseg labels ----------

if __name__ == '__main__':
    nusc_class = NuScenes(version='v1.0-mini', dataroot='/home/whye/Desktop/nuscenes_o', verbose=True)

    main(nusc_class)
    # test_viz(nusc_class)