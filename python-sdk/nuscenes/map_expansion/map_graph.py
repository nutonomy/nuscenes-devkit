# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# Licensed under the Creative Commons [see license.txt]

import os
import json
import random
import descartes
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle, Arrow
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

# Recommended style to use as the plots will show grids.
plt.style.use('seaborn-whitegrid')


class MapGraph:
    """ MapGraph database class for querying and retrieving information from the database. """

    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        """
        Loads the layers, create reverse indices and shortcuts, initializes the explorer class.
        :param dataroot: Path to the layers in the form of a .json file.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`,
        `boston-seaport` that we want to load.
        """
        assert map_name in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']

        self.dataroot = dataroot
        self.map_name = map_name

        self.json_fname = os.path.join(self.dataroot, "maps", "{}.json".format(self.map_name))

        self.geometric_layers = ['polygon', 'line', 'node']

        # These are the non-geometric layers which have polygons as the geometric descriptors.
        self.non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']

        # These are the non-geometric layers which have line strings as the geometric descriptors.
        self.non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']
        self.non_geometric_layers = self.non_geometric_polygon_layers + self.non_geometric_line_layers
        self.layer_names = self.geometric_layers + self.non_geometric_polygon_layers + self.non_geometric_line_layers

        with open(self.json_fname, 'r') as fh:
            self.json_obj = json.load(fh)

        self.canvas_edge = self.json_obj['canvas_edge']
        self._load_layers()
        self._make_token2ind()
        self._make_shortcuts()

        self.explorer = MapGraphExplorer(self)

    def _load_layer(self, layer_name: str) -> List[dict]:
        """
        Returns a list of records corresponding to the layer name.
        :param layer_name: Name of the layer that will be loaded.
        :return: A list of records corresponding to a layer.
        """
        return self.json_obj[layer_name]

    def _load_layers(self) -> None:
        """ Loads each available layer. """

        # Explicit assignment of layers are necessary to help the IDE determine valid class members.
        self.polygon = self._load_layer('polygon')
        self.line = self._load_layer('line')
        self.node = self._load_layer('node')
        self.drivable_area = self._load_layer('drivable_area')
        self.road_segment = self._load_layer('road_segment')
        self.road_block = self._load_layer('road_block')
        self.lane = self._load_layer('lane')
        self.ped_crossing = self._load_layer('ped_crossing')
        self.walkway = self._load_layer('walkway')
        self.stop_line = self._load_layer('stop_line')
        self.carpark_area = self._load_layer('carpark_area')
        self.road_divider = self._load_layer('road_divider')
        self.lane_divider = self._load_layer('lane_divider')
        self.traffic_light = self._load_layer('traffic_light')

    def _make_token2ind(self) -> None:
        """ Store the mapping from token to layer index for each layer. """
        self._token2ind = dict()
        for layer_name in self.layer_names:
            self._token2ind[layer_name] = dict()

            for ind, member in enumerate(getattr(self, layer_name)):
                self._token2ind[layer_name][member['token']] = ind

    def _make_shortcuts(self) -> None:
        """ Makes the record shortcuts. """

        # Makes a shortcut between non geometric records to their nodes.
        for layer_name in self.non_geometric_polygon_layers:
            if layer_name == 'drivable_area':  # Drivable area has more than one geometric representation.
                pass
            else:
                for record in self.__dict__[layer_name]:
                    polygon_obj = self.get('polygon', record['polygon_token'])
                    record['exterior_node_tokens'] = polygon_obj['exterior_node_tokens']
                    record['holes'] = polygon_obj['holes']

        for layer_name in self.non_geometric_line_layers:
            for record in self.__dict__[layer_name]:
                record['node_tokens'] = self.get('line', record['line_token'])['node_tokens']

        # Makes a shortcut between stop lines to their cues, there's different cues for different types of stop line.
        # Refer to `_get_stop_line_cue()` for details.
        for record in self.stop_line:
            cue = self._get_stop_line_cue(record)
            record['cue'] = cue

        # Makes a shortcut between lanes to their lane divider segment nodes.
        for record in self.lane:
            record['left_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
                                                         record['left_lane_divider_segments']]
            record['right_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
                                                          record['right_lane_divider_segments']]

    def _get_stop_line_cue(self, stop_line_record: dict) -> List[dict]:
        """
        Get the different cues for different types of stop lines.
        :param stop_line_record: A single stop line record.
        :return: The cue for that stop line.
        """
        if stop_line_record['stop_line_type'] in ['PED_CROSSING', 'TURN_STOP']:
            return [self.get('ped_crossing', token) for token in stop_line_record['ped_crossing_tokens']]
        elif stop_line_record['stop_line_type'] in ['STOP_SIGN', 'YIELD']:
            return []
        elif stop_line_record['stop_line_type'] == 'TRAFFIC_LIGHT':
            return [self.get('traffic_light', token) for token in stop_line_record['traffic_light_tokens']]

    def get(self, layer_name: str, token: str) -> dict:
        """
        Returns a record from the layer in constant runtime.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: A single layer record.
        """
        assert layer_name in self.layer_names, "Layer {} not found".format(layer_name)

        return getattr(self, layer_name)[self.getind(layer_name, token)]

    def getind(self, layer_name: str, token: str) -> int:
        """
        This returns the index of the record in a layer in constant runtime.
        :param layer_name: Name of the layer we are interested in.
        :param token: Token of the record.
        :return: The index of the record in the layer, layer is an array.
        """
        return self._token2ind[layer_name][token]

    def render_record(self,
                      layer_name: str,
                      token: str,
                      alpha: float = 0.5,
                      figsize: Tuple[int, int] = (15, 15),
                      other_layers: List[str] = None) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
         Render a single map graph record. By default will also render 3 layers which are `drivable_area`, `lane`,
         and `walkway` unless specified by `other_layers`.
         :param layer_name: Name of the layer that we are interested in.
         :param token: Token of the record that you want to render.
         :param alpha: The opacity of each layer that gets rendered.
         :param figsize: Size of the whole figure.
         :param other_layers: What other layers to render aside from the one specified in `layer_name`.
         :return: The matplotlib figure and axes of the rendered layers.
         """
        return self.explorer.render_record(layer_name, token, alpha, figsize, other_layers)

    def render_layers(self,
                      layer_names: List[str],
                      alpha: float = 0.5,
                      figsize: Tuple[int, int] = (15, 15)) -> Tuple[Figure, Axes]:
        """
        Render a list of layer names.
        :param layer_names: A list of layer names.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_layers(layer_names, alpha, figsize)

    def render_map_patch(self,
                         box_coords: Tuple[float, float, float, float],
                         layer_names: List[str] = None,
                         alpha: float = 0.5,
                         figsize: Tuple[int, int] = (15, 15)) -> Tuple[Figure, Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_map_patch(box_coords, layer_names, alpha, figsize)

    def get_records_in_patch(self,
                             box_coords: Tuple[float, float, float, float],
                             layer_names: List[str] = None,
                             mode: str = 'intersect') -> Dict[str, List[str]]:
        """
        Get all the record token that intersects or is within a particular rectangular patch.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: Names of the layers that we want to retrieve in a particular patch. By  default will always
        look at the all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch, "within" will return
        all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """
        return self.explorer.get_records_in_patch(box_coords, layer_names, mode)

    def is_record_in_patch(self,
                           layer_name: str,
                           token: str,
                           box_coords: Tuple[float, float, float, float],
                           mode: str = 'intersect') -> bool:
        """
        Query whether a particular record is in a rectangular patch
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch, "within" will
        return True if the geometric object is within the patch.
        :return: Boolean value on whether a particular record intersects or within a particular patch.
        """
        return self.explorer.is_record_in_patch(layer_name, token, box_coords, mode)

    def layers_on_point(self, x: float, y: float) -> Dict[str, str]:
        """
        Returns all the polygonal layers that a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :return: All the polygonal layers that a particular point is on. {<layer name>: <list of tokens>}
        """
        return self.explorer.layers_on_point(x, y)

    def record_on_point(self, x: float, y: float, layer_name: str) -> str:
        """
        Query what record of a layer a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_name: The non geometric polygonal layer name that we are interested in.
        :return: The record token of a layer a particular point is on.
        """
        return self.explorer.record_on_point(x, y, layer_name)

    def extract_polygon(self, polygon_token: str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token.
        :param polygon_token: The token of the polygon record.
        :return: The polygon wrapped in a shapely Polygon object.
        """
        return self.explorer.extract_polygon(polygon_token)

    def extract_line(self, line_token: str) -> LineString:
        """
        Construct a shapely LineString object out of a line token.
        :param line_token: The token of the line record.
        :return: The line wrapped in a LineString object.
        """
        return self.explorer.extract_line(line_token)

    def get_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the geometric object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the line representation.
        """
        return self.explorer.get_bounds(layer_name, token)


class MapGraphExplorer:
    """ Helper class to explore the nuScenes map data. """
    def __init__(self,
                 map_graph: MapGraph,
                 representative_layers: Tuple[str] = ('drivable_area', 'lane', 'walkway'),
                 color_map: dict = None):
        """
        :param map_graph: MapGraph database class.
        :param representative_layers: These are the layers that we feel are representative of the whole mapping data.
        :param color_map: Color map.
        """
        # Mutable default argument.
        if color_map is None:
            color_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

        self.map_graph = map_graph
        self.representative_layers = representative_layers
        self.color_map = color_map

        self.canvas_max_x = self.map_graph.canvas_edge[0]
        self.canvas_min_x = 0
        self.canvas_max_y = self.map_graph.canvas_edge[1]
        self.canvas_min_y = 0
        self.canvas_aspect_ratio = (self.canvas_max_x - self.canvas_min_x) / (self.canvas_max_y - self.canvas_min_y)

    def render_record(self,
                      layer_name: str,
                      token: str,
                      alpha: float,
                      figsize: Tuple[int, int],
                      other_layers: List[str] = None) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Render a single map graph record . By default will also render 3 layers which are `drivable_area`, `lane`,
        and `walkway` unless specified by `other_layers`.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record that you want to render.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :param other_layers: What other layers to render aside from the one specified in `layer_name`.
        :return: The matplotlib figure and axes of the rendered layers.
        """

        if other_layers is None:
            other_layers = list(self.representative_layers)

        for other_layer in other_layers:
            if other_layer not in self.map_graph.non_geometric_layers:
                raise ValueError("{} is not a non geometric layer".format(layer_name))

        x1, y1, x2, y2 = self.map_graph.get_bounds(layer_name, token)

        local_width = x2 - x1
        local_height = y2 - y1
        local_aspect_ratio = local_width / local_height

        # We obtained the values 0.65 and 0.66 by trials.
        fig = plt.figure(figsize=figsize)
        global_ax = fig.add_axes([0, 0, 0.65, 0.65 / self.canvas_aspect_ratio])
        local_ax = fig.add_axes([0.66, 0.66 / self.canvas_aspect_ratio, 0.34, 0.34 / local_aspect_ratio])

        # To make sure the sequence of the layer overlays is always consistent after typesetting set().
        random.seed('nutonomy')

        layer_names = other_layers + [layer_name]
        layer_names = list(set(layer_names))

        for layer in layer_names:
            self._render_layer(global_ax, layer, alpha)

        for layer in layer_names:
            self._render_layer(local_ax, layer, alpha)

        if layer_name == 'drivable_area':
            # Bad output aesthetically if we add spacing between the objects and the axes for drivable area.
            local_ax_xlim = (x1, x2)
            local_ax_ylim = (y1, y2)
        else:
            # Add some spacing between the object and the axes.
            local_ax_xlim = (x1 - local_width / 3, x2 + local_width / 3)
            local_ax_ylim = (y1 - local_height / 3, y2 + local_height / 3)

            # Draws the rectangular patch on the local_ax.
            local_ax.add_patch(Rectangle((x1, y1), local_width, local_height, linestyle='-.', color='red', fill=False,
                                         lw=2))

        local_ax.set_xlim(*local_ax_xlim)
        local_ax.set_ylim(*local_ax_ylim)
        local_ax.set_title('Local View')

        global_ax.set_xlim(self.canvas_min_x, self.canvas_max_x)
        global_ax.set_ylim(self.canvas_min_y, self.canvas_max_y)
        global_ax.set_title('Global View')
        global_ax.legend()

        # Adds the zoomed in effect to the plot.
        mark_inset(global_ax, local_ax, loc1=2, loc2=4, color='black')

        return fig, (global_ax, local_ax)

    def render_layers(self, layer_names: List[str], alpha: float, figsize: Tuple[int, int]) -> Tuple[Figure, Axes]:
        """
        Render a list of layers.
        :param layer_names: A list of layer names.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1 / self.canvas_aspect_ratio])

        ax.set_xlim(self.canvas_min_x, self.canvas_max_x)
        ax.set_ylim(self.canvas_min_y, self.canvas_max_y)

        layer_names = list(set(layer_names))

        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha)

        ax.legend()

        return fig, ax

    def render_map_patch(self,
                         box_coords: Tuple[float, float, float, float],
                         layer_names: List[str] = None,
                         alpha: float = 0.5,
                         figsize: Tuple[int, int] = (15, 15)) -> Tuple[Figure, Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :return: The matplotlib figure and axes of the rendered layers.
        """

        x_min, y_min, x_max, y_max = box_coords

        if layer_names is None:
            layer_names = self.map_graph.non_geometric_layers

        fig = plt.figure(figsize=figsize)

        local_width = x_max - x_min
        local_height = y_max - y_min
        local_aspect_ratio = local_width / local_height

        ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha)

        ax.set_xlim(x_min - local_width / 3, x_max + local_width / 3)
        ax.set_ylim(y_min - local_height / 3, y_max + local_height / 3)
        ax.add_patch(Rectangle((x_min, y_min), local_width, local_height, fill=False, linestyle='-.', color='red',
                               lw=2))
        ax.text(x_min, y_min + local_height / 2, "{} m".format(local_height), dict(fontsize=12))
        ax.text(x_min + local_width / 2, y_min, "{} m".format(local_width), dict(fontsize=12))

        ax.legend()

        return fig, ax

    def get_records_in_patch(self,
                             box_coords: Tuple[float, float, float, float],
                             layer_names: List[str] = None,
                             mode: str = 'intersect') -> Dict[str, List[str]]:
        """
        Get all the record token that intersects or within a particular rectangular patch.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: Names of the layers that we want to retrieve in a particular patch.
            By default will always look for all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch,
            "within" will return all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """
        if mode not in ['intersect', 'within']:
            raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

        if layer_names is None:
            layer_names = self.map_graph.non_geometric_layers

        records_in_patch = dict()
        for layer_name in layer_names:
            layer_records = []
            for record in getattr(self.map_graph, layer_name):
                token = record['token']
                if self.is_record_in_patch(layer_name, token, box_coords, mode):
                    layer_records.append(token)

            records_in_patch.update({layer_name: layer_records})

        return records_in_patch

    def is_record_in_patch(self,
                           layer_name: str,
                           token: str,
                           box_coords: Tuple[float, float, float, float],
                           mode: str = 'intersect') -> bool:
        """
        Query whether a particular record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular record intersects or is within a particular patch.
        """
        if mode not in ['intersect', 'within']:
            raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

        if layer_name in self.map_graph.non_geometric_polygon_layers:
            return self._is_polygon_record_in_patch(token, layer_name, box_coords, mode)
        elif layer_name in self.map_graph.non_geometric_line_layers:
            return self._is_line_record_in_patch(token, layer_name, box_coords,  mode)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def layers_on_point(self, x: float, y: float) -> Dict[str, str]:
        """
        Returns all the polygonal layers that a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :return: All the polygonal layers that a particular point is on.
        """
        layers_on_point = dict()
        for layer_name in self.map_graph.non_geometric_polygon_layers:
            layers_on_point.update({layer_name: self.record_on_point(x, y, layer_name)})

        return layers_on_point

    def record_on_point(self, x, y, layer_name) -> str:
        """
        Query what record of a layer a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_name: The non geometric polygonal layer name that we are interested in.
        :return: The first token of a layer a particular point is on or '' if no layer is found.
        """
        if layer_name not in self.map_graph.non_geometric_polygon_layers:
            raise ValueError("{} is not a polygon layer".format(layer_name))

        point = Point(x, y)
        records = getattr(self.map_graph, layer_name)

        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_graph.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                for polygon in polygons:
                    if point.within(polygon):
                        return record['token']
                    else:
                        pass
        else:
            for record in records:
                polygon = self.map_graph.extract_polygon(record['polygon_token'])
                if point.within(polygon):
                    return record['token']
                else:
                    pass

        # If nothing is found, return an empty string.
        return ''

    def extract_polygon(self, polygon_token: str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token.
        :param polygon_token: The token of the polygon record.
        :return: The polygon wrapped in a shapely Polygon object.
        """
        polygon_record = self.map_graph.get('polygon', polygon_token)
        exterior_coords = [(self.map_graph.get('node', token)['x'], self.map_graph.get('node', token)['y'])
                           for token in polygon_record['exterior_node_tokens']]

        interiors = []
        for hole in polygon_record['holes']:
            interior_coords = [(self.map_graph.get('node', token)['x'], self.map_graph.get('node', token)['y'])
                               for token in hole['node_tokens']]
            if len(interior_coords) > 0:  # Add only non-empty holes.
                interiors.append(interior_coords)

        return Polygon(exterior_coords, interiors)

    def extract_line(self, line_token: str) -> LineString:
        """
        Construct a shapely LineString object out of a line token.
        :param line_token: The token of the line record.
        :return: The line wrapped in a LineString object.
        """
        line_record = self.map_graph.get('line', line_token)
        line_nodes = [(self.map_graph.get('node', token)['x'], self.map_graph.get('node', token)['y'])
                      for token in line_record['node_tokens']]

        return LineString(line_nodes)

    def get_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the geometric object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of the line representation.
        """
        if layer_name in self.map_graph.non_geometric_polygon_layers:
            return self._get_polygon_bounds(layer_name, token)
        elif layer_name in self.map_graph.non_geometric_line_layers:
            return self._get_line_bounds(layer_name, token)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _get_polygon_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the extremities of the polygon object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the polygon or polygons (for drivable_area) representation.
        """
        if layer_name not in self.map_graph.non_geometric_polygon_layers:
            raise ValueError("{} is not a record with polygon representation".format(token))

        record = self.map_graph.get(layer_name, token)

        if layer_name == 'drivable_area':
            polygons = [self.map_graph.get('polygon', polygon_token) for polygon_token in record['polygon_tokens']]
            exterior_node_coords = []

            for polygon in polygons:
                nodes = [self.map_graph.get('node', node_token) for node_token in polygon['exterior_node_tokens']]
                node_coords = [(node['x'], node['y']) for node in nodes]
                exterior_node_coords.extend(node_coords)

            exterior_node_coords = np.array(exterior_node_coords)
        else:
            exterior_nodes = [self.map_graph.get('node', token) for token in record['exterior_node_tokens']]
            exterior_node_coords = np.array([(node['x'], node['y']) for node in exterior_nodes])

        xs = exterior_node_coords[:, 0]
        ys = exterior_node_coords[:, 1]

        x2 = xs.max()
        x1 = xs.min()
        y2 = ys.max()
        y1 = ys.min()

        return x1, y1, x2, y2

    def _get_line_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the line object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the line representation.
        """
        if layer_name not in self.map_graph.non_geometric_line_layers:
            raise ValueError("{} is not a record with line representation".format(token))

        record = self.map_graph.get(layer_name, token)
        nodes = [self.map_graph.get('node', node_token) for node_token in record['node_tokens']]
        node_coords = [(node['x'], node['y']) for node in nodes]
        node_coords = np.array(node_coords)

        xs = node_coords[:, 0]
        ys = node_coords[:, 1]

        x2 = xs.max()
        x1 = xs.min()
        y2 = ys.max()
        y1 = ys.min()

        return x1, y1, x2, y2

    def _is_polygon_record_in_patch(self,
                                    token: str,
                                    layer_name: str,
                                    box_coords: Tuple[float, float, float, float],
                                    mode: str = 'intersect') -> bool:
        """
        Query whether a particular polygon record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular polygon record intersects or is within a particular patch.
        """
        if layer_name not in self.map_graph.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        x_min, y_min, x_max, y_max = box_coords
        record = self.map_graph.get(layer_name, token)
        rectangular_patch = box(x_min, y_min, x_max, y_max)

        if layer_name == 'drivable_area':
            polygons = [self.map_graph.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
            geom = MultiPolygon(polygons)
        else:
            geom = self.map_graph.extract_polygon(record['polygon_token'])

        if mode == 'intersect':
            return geom.intersects(rectangular_patch)
        elif mode == 'within':
            return geom.within(rectangular_patch)

    def _is_line_record_in_patch(self,
                                 token: str,
                                 layer_name: str,
                                 box_coords: Tuple[float, float, float, float],
                                 mode: str = 'intersect') -> bool:
        """
        Query whether a particular line record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular line  record intersects or is within a particular patch.
        """
        if layer_name not in self.map_graph.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        x_min, y_min, x_max, y_max = box_coords

        record = self.map_graph.get(layer_name, token)
        node_recs = [self.map_graph.get('node', node_token) for node_token in record['node_tokens']]
        node_coords = [[node['x'], node['y']] for node in node_recs]
        node_coords = np.array(node_coords)

        cond_x = np.logical_and(node_coords[:, 0] < x_max, node_coords[:, 0] > x_min)
        cond_y = np.logical_and(node_coords[:, 1] < y_max, node_coords[:, 0] > y_min)
        cond = np.logical_and(cond_x, cond_y)

        if mode == 'intersect':
            return np.any(cond)
        elif mode == 'within':
            return np.all(cond)

    def _render_layer(self, ax: Axes, layer_name: str, alpha: float) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        """
        if layer_name in self.map_graph.non_geometric_polygon_layers:
            self._render_polygon_layer(ax, layer_name, alpha)
        elif layer_name in self.map_graph.non_geometric_line_layers:
            self._render_line_layer(ax, layer_name, alpha)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _render_polygon_layer(self, ax: Axes, layer_name: str, alpha: float) -> None:
        """
        Renders an individual non-geometric polygon layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        """
        if layer_name not in self.map_graph.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        first_time = True
        records = getattr(self.map_graph, layer_name)
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_graph.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    if first_time:
                        label = layer_name
                        first_time = False
                    else:
                        label = None
                    ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label))
        else:
            for record in records:
                polygon = self.map_graph.extract_polygon(record['polygon_token'])

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None

                ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                    label=label))

    def _render_line_layer(self, ax: Axes, layer_name: str, alpha: float) -> None:
        """
        Renders an individual non-geometric line layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        """
        if layer_name not in self.map_graph.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        first_time = True
        records = getattr(self.map_graph, layer_name)
        for record in records:
            if first_time:
                label = layer_name
                first_time = False
            else:
                label = None
            line = self.map_graph.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            if layer_name == 'traffic_light':
                # Draws an arrow with the physical traffic light as the starting point, pointing to the direction on
                # where the traffic light points.
                ax.add_patch(Arrow(xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], color=self.color_map[layer_name],
                                   label=label))
            else:
                ax.plot(xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label)
