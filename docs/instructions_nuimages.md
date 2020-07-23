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
 
 [![](occluded_vehicle_8.png)]() [![](occluded_vehicle_9.png)]() 
 - If an object is cut off at the edge of the image, then the bounding box should stop at the image boundary.
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
 
 [![](mirror_image_1.png)]() 
 - If an object has extremities, the bounding box should include **all** the extremities (exceptions are the side view mirrors and antennas of vehicles).
 Note that this differs [from how the instance masks are annotated](#instance-segmentation), in which the extremities are included in the masks.
 
 [![](extremity_5.png)]() [![](extremity_4.png)]()
 - Only label objects if the object is clear enough to be certain of what it is. If an object is so blurry it cannot be known, do not label the object.
 - Do not label an object if its height is less than 10 pixels.
 - Do not label an object if its less than 20% visible, unless you can confidently tell what the object is.
 An object can have low visibility when it is occluded or cut off by the image.
 The clarity and orientation of the object does not influence its visibility. 
 
### Detailed Instructions 
 - `human.pedestrian.*`
   - In nighttime images, annotate the pedestrian only when either the body part(s) of a person is clearly visible (leg, arm, head etc.), or the person is clearly in motion.
   
   [![](nighttime_pedestrian_fp_1.png)]() [![](nighttime_pedestrian_fp_2.png)]() 
 - `vehicle.*`
   - In nighttime images, annotate a vehicle only when a pair of lights is clearly visible (break or head or hazard lights), and it is clearly on the road surface.
   
   [![](nighttime_vehicle_fp_1.png)]() 
   [![](nighttime_vehicle_fp_2.png)]() 
   [![](nighttime_vehicle_fn_1.png)]()
   
[Top](#overview)
   
## Instance Segmentation
### General Instructions
 - Given a bounding box, outline the **visible** parts of the object enclosed within the bounding box using a polygon.
 - Each pixel on the image should be assigned to at most one object instance (i.e. the polygons should not overlap).
 - There should not be a discrepancy of more than 2 pixels between the edge of the object instance and the polygon.
 - If an object is occluded by another object whose width is less than 5 pixels (e.g. a thin fence), then the external object can be included in the polygon.
 
 [![](inst11.png)]()
 [![](inst12.png)]()
 - If an object is loosely covered by another object (e.g. branches, bushes), do not create several polygons for visible areas that are less than 15 pixels in diameter.
 
 [![](inst_seg_v2_19.png)]()
 - If an object enclosed by the bounding box is occluded by another foreground object but has a visible area through a glass window (like for cars / vans / trucks), 
 do not create a polygon on that visible area.
 
 [![](inst16.png)]()
 - If an object has a visible area through a hole of another foreground object, create a polygon on the visible area.
 Exemptions would be holes from bicycle / motorcycles / bike racks and holes that are less than 15 pixels diameter.
 
 [![](inst17.png)]()
 - If a static / moveable object has another object attached to it (signboard, rope), include it in the annotation.
 
 [![](inst_seg_v2_21.png)]()
 [![](inst_seg_v2_22.png)]()
 - If parts of an object are not visible due to lighting and / or shadow, it is best to have an educated guess on the non-visible areas of the object.
 
 [![](inst24.png)]()
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
 
 [![](inst25.png)]()
 
### Detailed Instructions 
 - `vehicle.*`
   - Include extremities (e.g. side view mirrors, taxi heads, police sirens, etc.); exceptions are the crane arms on construction vehicles.
   
   [![](inst8.png)]()
   [![](inst10.png)]()
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
![driveable_1](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/driveable_1.png)
![driveable_2](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/driveable_2.png)
![driveable_3](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/driveable_3.png)
![driveable_4](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/driveable_4.png)

### 2. vehicle.ego
![ego_1](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/ego_1.png)
![ego_2](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/ego_2.png)
![ego_3](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/ego_3.png)
![ego_4](https://www.nuscenes.org/public/images/taxonomy_imgs/nuimages/ego_4.png)

## Semantic Segmentation
### General Instructions
 - Only annotate a surface if its length and width are **both** greater than 20 pixels.
 - Annotations should tightly bound the edges of the area(s) of interest. 
 
 [![](no_gaps.png)]()
 - If two areas/objects of interest are adjacent to each other, there should be no gap between the two annotations.
 
 [![](polygon_tight.png)]()
 - Annotate a surface only as far as it is clearly visible.
 
 [![](rules1.png)]()
 - If a surface is occluded (e.g. by branches, trees, fence poles), only annotate the visible areas (which are more than 20 pixels in length and width).
 
 [![]()]()
 - If a surface is covered by dirt or snow of less than 20 cm in height, include the dirt or snow in the annotation (since it can be safely driven over).
 
 [![]()]()
 - If a surface has puddles in it, always include them in the annotation.
 - Do not annotate reflections of surfaces.

### Detailed Instructions 
 - `flat.driveable_surface`
   - Include surfaces blocked by road blockers or pillars as long as they are the same surface as the driveable surface.
   [![](dirveable_surface14.png)]()

[Top](#overview)
