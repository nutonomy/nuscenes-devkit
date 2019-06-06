{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# nuScenes Map Extension Tutorial"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is the tutorial for the nuScenes map extension. In particular, the MapGraph data class. \n",
    "\n",
    "This tutorial will go through the description of each layers, how we retrieve and query a certain record within the map graph layers, render methods, and advanced data exploration\n",
    "\n",
    "In database terms, layers are basically tables of the map database in which we assign arbitrary parts of the maps with informative labels such as `traffic_lights`, `stop_lines`, `road_segments`, etc. Refer to the discussion on layers for more details."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup\n",
    "To install the map extension, please download the files from https://www.nuscenes.org/download and copy the files into your nuScenes map folder, e.g. `/data/sets/nuscenes/maps`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialization\n",
    "\n",
    "We will be working with the `singapore-onenorth` map. The `MapGraph` can be initialized as follows:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'nuscenes.map_expansion'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-85ccc869ecdf>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mnuscenes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmap_expansion\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnuscenes_map\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mMapGraph\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mmap_graph\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mMapGraph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataroot\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'/data/sets/nuscenes'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmap_name\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'singapore-onenorth'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'nuscenes.map_expansion'"
     ]
    }
   ],
   "source": [
    "from nuscenes.map_expansion.nuscenes_map import MapGraph\n",
    "import matplotlib.pyplot as plt \n",
    "\n",
    "map_graph = MapGraph(dataroot='/data/sets/nuscenes', map_name='singapore-onenorth')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Layers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us look at the map layers that this data class holds:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.layer_names"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our map database consists of multiple **layers**. Where each layer is made up of **records**. Each record will have a token identifier.\n",
    "\n",
    "We see how our map layers are divided into two types of layers. One set of layer belong to the `geometric_layers` group, another set of layers belongs to the `non_geometric_layers` group.  \n",
    "1. `geometric_layers` serve as physical \"descriptors\" of the maps:\n",
    "    - Nodes are the most primitive geometric records.\n",
    "    - Lines consist of two or more nodes. Formally, one `Line` record can consist of more than one line segment.\n",
    "    - Polygons consist of three or more nodes. A polygon can have holes, thus distorting its formal definition. Holes are defined as a sequence of nodes that forms the perimeter of the polygonal hole.\n",
    "    \n",
    "    \n",
    "2. `non_geometric_layers` are overlapping layers that serve as labels of the physical entities. They can have more than one geometric representation (such as `drivable_areas`) but must be strictly of one type. (e.g. `road_segment` is represented by one polygon object, `lane_divider` is represented by one line object)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Geometric layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.geometric_layers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### a. Node\n",
    "The most primitive geometric record in our map database. This is the only layer that has an explicit coordinate field associated with it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_node = map_graph.node[0]\n",
    "sample_node"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### b. Line\n",
    "\n",
    "Similar to `Polygon` discussed above. The definition of `Line` here does not follow the formal definition as it can consist of *more than two* nodes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_line = map_graph.line[2]\n",
    "sample_line"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### c. Polygon \n",
    "The definition of \"Polygon\" here does not conform to the formal definition of polygons which is a 'plane figure bounded by a closed curve' as it may contain holes.\n",
    "\n",
    "Every polygon record comprises of a list of exterior nodes, and zero or more list(s) of nodes that constitutes (zero or more) holes.\n",
    "\n",
    "Let's look at one polygon record"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_polygon = map_graph.polygon[3]\n",
    "sample_polygon.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_polygon['exterior_node_tokens']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_holes = sample_polygon['holes'][0]\n",
    "sample_holes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Non geometric layers\n",
    "\n",
    "Every non geometric layers are associated with an geometric object. To reiterate, the concept of \"non-geometric\" here does not mean that a layer does not possess any geometric entity. In fact, every non-geometric layer is associated with at least one geometric entity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.non_geometric_layers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### a. Drivable Area\n",
    "Drivable area is defined as the area where the car can drive, without consideration for driving direction or legal restrictions. This is the only layer in which the record can be represented by more than one geometric entity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_drivable_area = map_graph.drivable_area[0]\n",
    "sample_drivable_area"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('drivable_area', sample_drivable_area['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### b. Road Segment\n",
    "\n",
    "A segment of road on a drivable area. It has an `is_intersection` flag which denotes whether a particular road segment is an intersection.\n",
    "\n",
    "It may or may not have an association with a `drivable area` record from its `drivable_area_token` field."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "sample_road_segment = map_graph.road_segment[600]\n",
    "sample_road_segment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As observed, for all non geometric objects except `drivable_area`, we provide a shortcut to its `nodes`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('road_segment', sample_road_segment['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's take a look at a `road_segment` record with `is_intersection == True`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_intersection_road_segment = map_graph.road_segment[3]\n",
    "sample_intersection_road_segment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('road_segment', sample_intersection_road_segment['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### c. Road Block\n",
    "Road blocks are always within a road segment. It will always have the same traffic direction within its area.\n",
    "\n",
    "Within a road block, the number of lanes is consistent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_road_block = map_graph.road_block[0]\n",
    "sample_road_block"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Every road block has a `from_edge_line_token` and `to_edge_line_token` that denotes its traffic direction."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### d. Lanes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lanes are parts of the road that are designed for a single line of vehicle path."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "sample_lane_record = map_graph.lane[600]\n",
    "sample_lane_record"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('lane', sample_lane_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Aside from the token and the geometric representation, it has several fields:\n",
    "- `lane_type` denotes whether cars or bikes are allowed to navigate through that lane.\n",
    "- `from_edge_line_token` and `to_edge_line_token` denotes their traffic direction\n",
    "- `left_lane_divider_segments` and `right_lane_divider_segment` denotes their lane dividers.\n",
    "- `left_lane_divider_segment_nodes` and `right_lane_divider_segment_nodes` denotes the nodes that makes up the lane dividers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### e. Pedestrian Crossing\n",
    "It is the physical world's pedestrian crossing. Each pedestrian crossing record has to be on a road segment. It has the `road_segment_token` field which denotes the `road_segment` record it is associated with."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_ped_crossing_record = map_graph.ped_crossing[0]\n",
    "sample_ped_crossing_record"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('ped_crossing', sample_ped_crossing_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### f. Walkway\n",
    "It is the physical world's walk way or side walk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_walkway_record = map_graph.walkway[0]\n",
    "sample_walkway_record"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('walkway', sample_walkway_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### g. Stop Line\n",
    "The physical world's stop line, even though the name implies that it should possess a `line` geometric representation, in reality its physical representation is an **area where the ego vehicle must stop.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_stop_line_record = map_graph.stop_line[1]\n",
    "sample_stop_line_record"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It has several trivial attributes\n",
    "- `stop_line_type`, the type of the stop line, this represents the reasons why the ego vehicle would stop         \n",
    "- `ped_crossing_tokens` denotes the association information if the `stop_line_type` is `PED_CROSSING`.\n",
    "- `traffic_light_tokens` denotes the association information if the `stop_line_type` is `TRAFFIC_LIGHT`.\n",
    "- `road_block_token` denotes the association information to a `road_block`, can be empty by default. \n",
    "- `cues` field contains the reason on why this this record is a `stop_line`. An area can be a stop line due to multiple reasons:\n",
    "    - Cues for `stop_line_type` of \"PED_CROSSING\" or \"TURN_STOP\" are `ped_crossing` records.\n",
    "    - Cues for `stop_line_type` of TRAFFIC_LIGHT\" are `traffic_light` records.\n",
    "    - No cues for `stop_line_type` of \"STOP_SIGN\" or \"YIELD\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('stop_line', sample_stop_line_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### h. Carpark Area\n",
    "A car park or a parking lot area."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_carpark_area_record = map_graph.carpark_area[1]\n",
    "sample_carpark_area_record"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It has several trivial attributes:\n",
    "- `orientation` denotes the direction of cars parked in radians.\n",
    "- `road_block_token` denotes the association information to a `road_block`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('carpark_area', sample_carpark_area_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### i. Road Divider\n",
    "A divider that separates one road block from another."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_road_divider_record = map_graph.road_divider[0]\n",
    "sample_road_divider_record"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`road_segment_token` saves the association information to a `road_segment`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('road_divider', sample_road_divider_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### j. Lane Divider\n",
    "Lane divider comes in between lanes that point in the same traffic direction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_lane_divider_record = map_graph.lane_divider[0]\n",
    "sample_lane_divider_record"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `lane_divider_segments` field consist of different `node`s and their respective `segment_type`s which denotes their physical appearance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('lane_divider', sample_lane_divider_record['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### l. Traffic Light\n",
    "A physical world's traffic light."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_traffic_light_record = map_graph.traffic_light[0]\n",
    "sample_traffic_light_record"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It has several trivial attributes:\n",
    "1. `traffic_light_type` denotes whether the traffic light is oriented horizontally or vertically.\n",
    "2. `from_road_block_tokens` denotes from which road block the traffic light guides.\n",
    "3. `items` are the bulbs for that traffic light.\n",
    "4. `pose` denotes the pose of the traffic light."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's examine the `items` field"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_traffic_light_record['items']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As mentioned, every entry in the `items` field is a traffic light bulb. It has the `color` information, the `shape` information, `rel_pos` which is the relative position, and the `to_road_block_tokens` that denotes to which road blocks the traffic light bulb is guiding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.json_obj['lane'][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Render Methods"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Rendering multiple layers\n",
    "\n",
    "The `MapGraph` class makes it possible to render multiple map layers on a matplotlib figure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_layers(map_graph.non_geometric_layers, figsize=(15,15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Rendering a particular record of the map layer\n",
    "\n",
    "We can render a record, which will show its global and local view"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('road_segment', map_graph.road_segment[600]['token'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "fig, ax = map_graph.render_record('stop_line', map_graph.stop_line[14]['token'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Advanced Data Exploration"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's render a particular patch on the map:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "patch = (300, 1000, 500, 1200)\n",
    "fig, ax = map_graph.render_map_patch(patch, map_graph.non_geometric_layers, figsize=(10, 10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A lot of layers can be seen in this patch. Lets retrieve all map records that are in this patch.\n",
    "\n",
    "- The option \"`within`\" will return all non geometric records that ***are within*** the map patch\n",
    "- The option \"`intersect`\" will return all non geometric records that ***intersect*** the map patch\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "records_within_patch = map_graph.get_records_in_patch(patch, map_graph.non_geometric_layers, mode='within')\n",
    "records_intersect_patch = map_graph.get_records_in_patch(patch, map_graph.non_geometric_layers, mode='intersect')\n",
    "print('Found %d records within the patch and %d records that intersect it.' % (len(records_within_patch), len(records_intersect_patch)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see that using the option `intersect` yields at least as many records as `within`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "records_within_patch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "records_intersect_patch"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's check what are the layers that are on point `(740, 960)`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.layers_on_point(740, 960)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Looking at the above plot. Point `(760, 925)` seems to be on a stop line. Lets verify that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.record_on_point(760, 925, 'stop_line')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's look at the bounds/extremities of that record"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_graph.get_bounds('stop_line', 'ac0a935f-99af-4dd4-95e3-71c92a5e58b1')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
