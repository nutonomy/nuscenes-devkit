"""
Exports a video of each scene (with annotations) to disk.
"""
import os

from nuscenes_utils.nuscenes import NuScenes

# Load NuScenes class
nusc = NuScenes()
scene_tokens = [s['token'] for s in nusc.scene]

# Create output directory
out_dir = os.path.expanduser('~/nuscenes-visualization/scene-videos')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Write videos to disk
for scene_token in scene_tokens:
    scene = nusc.get('scene', scene_token)
    print('Writing scene %s' % scene['name'])
    out_path = os.path.join(out_dir, scene['name']) + '.avi'
    if not os.path.exists(out_path):
        nusc.render_scene(scene['token'], out_path=out_path)