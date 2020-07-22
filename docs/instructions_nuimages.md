# nuImages Annotator Instructions

# Overview
- [Introduction](#introduction)
- [Objects](#objects)
  - [Bounding Boxes](#bounding-boxes)
  - [Instance Segmentation](#instance-segmentation)
  - [Attributes](#attributes)
- [Surfaces](#surfaces)
  - [Semantic Segmentation](#semantic-segmentation)



# Introduction
In nuImages, we annotate objects with 2d boxes, instance masks and 2d segmentation masks. All the labels and attributes from nuScenes are carried over into nuImages.
We have also [added more attributes](#attributes) in nuImages. For segmentation, we have included ["stuff" (background) classes](#surfaces).



# Objects
nuImages contains the same [object classes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#labels),
while the [attributes](#attributes) are a superset of the [attributes in nuScenes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#attributes).

## Bounding Boxes
### General Instructions
 - Draw bounding boxes around all objects that are in the list of [object classes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#labels).
 - Do not apply more than one box to a single object.
 - If an object is occluded, then draw the bounding box to include the occluded part of the object according to your best guess.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_200/occluded_vehicle_8.png)]() [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_200/occluded_vehicle_9.png)]() 
 - If an object is cut off at the edge of the image, then the bounding box should stop at the image boundary.
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_200/mirror_image_1.png)]() 
 - If an object has extremities, the bounding box should include **all** the extremities (exceptions are the side view mirrors and antennas of vehicles).
 Note that this differs [from how the instance masks are annotated](#instance-segmentation), in which the extremities are included in the masks.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_200/extremity_5.png)]() [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_200/extremity_4.png)]()
 - Only label objects if the object is clear enough to be certain of what it is. If an object is so blurry it cannot be known, do not label the object.
 - Do not label an object if its height is less than 10 pixels.
 - Do not label an object if its less than 20% visible, unless you can confidently tell what the object is.
 An object can have low visibility when it is occluded or cut off by the image.
 The clarity and orientation of the object does not influence its visibility. 
 
### Detailed Instructions 
 - `human.pedestrian.*`
   - In nighttime images, annotate the pedestrian only when either the body part(s) of a person is clearly visible (leg, arm, head etc.), or the person is clearly in motion.
   
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_400/nighttime_pedestrian_fp_1.png)]() [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_400/nighttime_pedestrian_fp_2.png)]() 
 - `vehicle.*`
   - In nighttime images, annotate a vehicle only when a pair of lights is clearly visible (break or head or hazard lights), and it is clearly on the road surface.
   
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_400/nighttime_vehicle_fp_1.png)]() 
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_400/nighttime_vehicle_fp_2.png)]() 
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/example_imgs/resized_400/nighttime_vehicle_fn_1.png)]()
   
[Top](#overview)
   
## Instance Segmentation
### General Instructions
 - Given a bounding box, outline the **visible** parts of the object enclosed within the bounding box using a polygon.
 - Each pixel on the image should be assigned to at most one object instance (i.e. the polygons should not overlap).
 - There should not be a discrepancy of more than 2 pixels between the edge of the object instance and the polygon.
 - If an object is occluded by another object whose width is less than 5 pixels (e.g. a thin fence), then the external object can be included in the polygon.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst11.png)]()
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst12.png)]()
 - If an object is loosely covered by another object (e.g. branches, bushes), do not create several polygons for visible areas that are less than 15 pixels in diameter.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst_seg_v2_19.png)]()
 - If an object enclosed by the bounding box is occluded by another foreground object but has a visible area through a glass window (like for cars / vans / trucks), 
 do not create a polygon on that visible area.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst16.png)]()
 - If an object has a visible area through a hole of another foreground object, create a polygon on the visible area.
 Exemptions would be holes from bicycle / motorcycles / bike racks and holes that are less than 15 pixels diameter.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst17.png)]()
 - If a static / moveable object has another object attached to it (signboard, rope), include it in the annotation.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst_seg_v2_21.png)]()
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst_seg_v2_22.png)]()
 - If parts of an object are not visible due to lighting and / or shadow, it is best to have an educated guess on the non-visible areas of the object.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst24.png)]()
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst25.png)]()
 
### Detailed Instructions 
 - `vehicle.*`
   - Include extremities (e.g. side view mirrors, taxi heads, police sirens, etc.); exceptions are the crane arms on construction vehicles.
   
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst8.png)]()
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/instance_seg_examples/inst10.png)]()
 - `static_object.bicycle_rack`
   - All bicycles in a bicycle rack should not be annotated.

[Top](#overview)

## Attributes
In nuImages, each object comes with a box, a mask and a set of attributes. 
The following attributes are in **addition** to the [existing ones in nuScenes]((https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#attributes)):

|  Attribute | Short Description |
| --- | --- |
| vehicle_light.emergency.flashing | The emergency lights on the vehicle are flashing. |
| vehicle_light.emergency.not_flashing | The emergency lights on the vehicle are not flashing. |
| vertical_position.off_ground | The object is not in the ground (e.g. it is flying, falling, jumping or positioned in a tree or on a vehicle). |
| vertical_position.on_ground | The object is on the ground plane. |

[Top](#overview)


# Surfaces
nuImages includes surface classes as well:

|  Label | Short Description |
| --- | --- |
| [`flat.driveable_surface`](#1-flatdriveable_surface) | All paved or unpaved surfaces that a car can drive on with no concern of traffic rules. |
| [`vehicle.ego`](#2-vehicleego) | The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image. |

### 1. flat.driveable_surface
[### TODO add examples images of class]
### 2. vehicle.ego
[### TODO add examples images of class]

## Semantic Segmentation
### General Instructions
 - Only annotate a surface if its length and width are **both** greater than 20 pixels.
 - Annotations should tightly bound the edges of the area(s) of interest. 
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/semantic_seg_examples/new_endpoint/no_gaps.png)]()
 - If two areas/objects of interest are adjacent to each other, there should be no gap between the two annotations.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/semantic_seg_examples/new_endpoint/polygon_tight.png)]()
 - Annotate a surface only as far as it is clearly visible.
 
 [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/semantic_seg_examples/ground/rules1.png)]()
 - If a surface is occluded (e.g. by branches, trees, fence poles), only annotate the visible areas (which are more than 20 pixels in length and width).
 
 [![](https://camo.githubusercontent.com/dfbfa4c8a61590f81761ba1e7b47d68fe5d92773/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6e75746f6e6f6d792d696d64622d64756d6d792f73656d616e7469635f7365675f6578616d706c65732f6e65775f656e64706f696e742f6275736865732e706e67)]()
 - If a surface is covered by dirt or snow of less than 20 cm in height, include the dirt or snow in the annotation (since it can be safely driven over).
 
 [![](https://camo.githubusercontent.com/c48c2fcbf2b8d59cf123680cffda0f03223b38f8/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6e75746f6e6f6d792d696d64622d64756d6d792f73656d616e7469635f7365675f6578616d706c65732f67726f756e642f72756c657331322e706e67)]()
 - If a surface has puddles in it, always include them in the annotation.
 - Do not annotate reflections of surfaces.

### Detailed Instructions 
 - `flat.driveable_surface`
   - Include surfaces blocked by road blockers or pillars as long as they are the same surface as the driveable surface.
   [![](https://s3.amazonaws.com/nutonomy-imdb-dummy/semantic_seg_examples/driveable_surface14.png)]()

[Top](#overview)
