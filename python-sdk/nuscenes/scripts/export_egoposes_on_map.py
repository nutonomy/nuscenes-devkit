# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

"""
Exports an image for each map location with all the ego poses drawn on the map.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from nuscenes import NuScenes

# Load NuScenes class
nusc = NuScenes(dataroot='/data/sets/nuscenes', version='v1.0-trainval')
locations = np.unique([l['location'] for l in nusc.log])

# Create output directory
out_dir = os.path.expanduser('~/nuscenes-visualization/map-poses')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

for location in locations:
    print('Rendering map %s...' % location)
    nusc.render_egoposes_on_map(location)
    out_path = os.path.join(out_dir, 'egoposes-%s.png' % location)
    plt.tight_layout()
    plt.savefig(out_path)
