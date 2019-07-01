from nuscenes.map_expansion.map_api import NuscenesMap
import matplotlib.pyplot as plt

nusc_map = NuscenesMap(dataroot='/data/sets/nuscenes', map_name='singapore-onenorth')

patch_box = None
patch_angle = 0  # Default orientation where North is up
layer_names = ['drivable_area', 'ped_crossing']
canvas_size = None
map_mask = nusc_map.explorer.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=(15, 15))
