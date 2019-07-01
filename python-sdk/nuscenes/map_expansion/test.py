from nuscenes.map_expansion.map_api import NuscenesMap
import matplotlib.pyplot as plt 

map_graph = NuscenesMap(dataroot='/data/sets/nuscenes', map_name='singapore-onenorth')

patch_box = (400, 1500, 500, 500)
patch_angle = 0
layer_names = None
figsize = (40, 10)
canvas_size = (1000, 1000)
fig, ax = map_graph.render_map_mask(patch_box, patch_angle, layer_names, figsize, canvas_size)
