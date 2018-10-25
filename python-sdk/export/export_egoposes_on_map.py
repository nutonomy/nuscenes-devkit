"""
Exports an image for each map location with all the ego poses drawn on the map.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from nuscenes_utils.nuscenes import NuScenes

# Load NuScenes class
nusc = NuScenes()
locations = np.unique([l['location'] for l in nusc.log])

# Create output directory
out_dir = os.path.expanduser('~/nuscenes-visualization/map-poses')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

for location in locations:
    nusc.render_egoposes_on_map(log_location=location)
    out_path = os.path.join(out_dir, 'egoposes-%s.png' % location)
    plt.tight_layout()
    plt.savefig(out_path)