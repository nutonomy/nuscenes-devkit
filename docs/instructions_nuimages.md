# nuImages Annotator Instructions

# Overview
- [Introduction](#introduction)
- [Objects](#objects)
  - [Bounding Boxes](#bounding-boxes)
  - [Instance Segmentation](#instance-segmentation)
- [Surfaces](#surfaces)
  - [Semantic Segmentation](#semantic-segmentation)



# Introduction
In nuImages, we annotate objects with 2d boxes, instance masks and 2d segmentation masks. All the labels and attributes from nuScenes are carried over into nuImages.



# Objects
nuImages contains the same [object classes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#labels) 
and [attributes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#attributes) from nuScenes.


## Bounding Boxes
### General Instructions
 - Draw bounding boxes around all objects that are in the list of [object classes](https://github.com/nutonomy/nuscenes-devkit/tree/master/docs/instructions_nuscenes.md#labels)
 - Do not apply more than one box to a single object.
 - If an object is occluded or cut off by the edge of the image, then draw the bounding box to include the occluded part of the object according to your best guess.
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
 - If an object has extremities, the bounding box should include **all** the extremities. 
 (Exceptions are the side view mirrors and antennas of vehicles).
 - Only label objects if the object is clear enough to be certain of what it is. 
 If an object is so blurry it cannot be known, do not label the object.
 - Do not label an object if its height is less than 10 pixels.
 - Do not label an object if its less than 20% visible. 
 The clarity and orientation of the object does not influence its visibility. 
 An object can have low visibility when it is occluded or cut off by the image.
 However, even when the object is less than 20% visible, it should be labeled if you can confidently tell what the object is.
 
### Detailed Instructions 
 - `human.pedestrian.*`
   - People inside / on vehicles should not be annotated as pedestrians. They are considered as appendages of vehicles.
   - In nighttime images, annotate the pedestrian only when either body part(s) of a person is clearly visible 
   (leg, arm, head etc.), or the person is clearly in motion.
   - If a pedestrian is carrying an object (e.g. children, bags, umbrellas tools), the object should be included 
   in the bounding box for the pedestrian.
   - If two or more pedestrians are carrying the same object, the bounding box of only one of them will include the object.
   - If a pedestrian is pulling or pushing an object, the pedestrian and the object should be annotated separately.
   - If a person is in a stroller / wheelchair, include the person in the annotation for `human.pedestrian.stroller` / `human.pedestrian.wheelchair`.
   - Pedestrians pushing strollers / wheelchairs should be labeled separately.
   - If a person in on a personal mobility vehicle (e.g. skateboard, Segway, scooter), include the person in the annotation for `human.pedestrian.personal_mobility`.
   - If there is uncertainty about the type of the pedestrian (`human.pedestrian.adult` vs `human.pedestrian.child` vs `human.pedestrian.construction_worker`, etc.),
    choose `human.pedestrian.adult`.
 - `vehicle.*`
   - In nighttime images annotate, the vehicles only when a pair of lights is clearly visible (break or head or hazard lights), 
   and it is clearly on the road surface.
   - Do not include side view mirrors and antennas of vehicles.
   - If a vehicle is primarily designed to haul cargo, label it as `vehicle.truck`.
   - If a vehicle is primarily designed to carry more than 10 people, label it as `vehicle.bus.*`.
   - If the pivoting joint of a bus cannot be seen, annotate it as `vehicle.bus.rigid`.
   - Trailers hauled after a semi-tractor should be labeled as `vehicle.trailer`.
   - Vehicles used for hauling rocks or building materials are considered `vehicle.truck` rather than `vehicle.construction`.
   - For `vehicle.motorcycle`:
     - If there is a rider, include the rider in the box.
     - If there is a passenger, include the passenger in the box.
     - If there is a pedestrian standing next to the motorcycle, do not include the pedestrian in the annotation.
     - If there is a sidecar attached to the motorcycle, include the sidecar in the box.
   - For `vehicle.bicycle`:
     - If there is a rider, include the rider in the box.
     - If there is a passenger, include the passenger in the box.
     - If there is a pedestrian standing next to the bicycle, do NOT include in the annotation.
   - For `vehicle.bicycle`:
     - Bicycles which are not part of a bicycle rack should be annotated as bicycles separately, 
     rather than collectively as `static_object.bicycle_rack`.
   - For `vehicle.trailer`:
     - A vehicle towed by another vehicle should be labeled with its corresponding vehicle type (not as `vehicle.trailer`).
 - `movable_object.*`
   - Movable objects do not include pedestrians, bicycles, wheel-chairs, and strollers.
   - Do not label movable signs or small barriers not on the road.
   - Do not label permanent pedestrian walk signs, even when they are on the road.
   - Do not label permanently mounted poles.
   - For `movable_object.trafficcone`:
     - Traffic cones do not necessarily have to be cone-shaped (e.g. there may be traffic cones which are orange-striped barrels).
     - Permanently mounted traffic delineator posts are NOT traffic cones.
     - Do not label traffic cones appended to vehicles.
   - For `movable_object.barrier`:
     - If there are multiple barriers either connected or just placed next to each other, they should be annotated separately.
     - If the barriers are installed permanently, then do not include them.

[Top](#overview)
   
## Instance Segmentation
### General Instructions
 - Given a bounding box, outline the **visible** parts of the object enclosed within the bounding box using a polygon.
 - Each pixel on the image should be assigned to at most one object instance (i.e. the polygons should not overlap).
 - There should be not be a discrepancy of more than 2 pixels discrepancy between the edge of the object instance and the polygon.
 - If an object is occluded by another object whose width is less than 5 pixels (e.g. a thin fence), 
 then the external object can be included in the polygon.
 - If an object is loosely covered by another object (e.g. branches bushes), do not create several polygons for visible areas that are less than 15 pixels in diameter.
 - If an object is enclosed by the bounding box is occluded by another foreground object but has a visible area through a glass window (like for cars / vans / trucks), 
 not create a polygon on that visible area.
 - If an object has a visible area through a hole of another foreground object, create a polygon on the visible area. 
 Exemptions would be holes from bicycle / motorcycles / bike racks and holes that are less than 15 pixels diameter.
 - If a static / moveable object has another object attached to it (signboard, rope), include it in the annotation.
 - If parts of an object is not visible due to lighting and / or shadow, it is best to have an educated guess on the non-visible area of the object.
 - If an object is reflected clearly in a glass window, then the reflection should be annotated.
### Detailed Instructions 
 - `human.pedestrian.*` 
  - If a pedestrian is carrying an object (i.e. bags, umbrellas, tools), include it in the polygon. 
  If the pedestrian is dragging an object, then do not include it.
 - `vehicle.*`
   - Include extremities (e.g. side view mirrors, taxi heads, police sirens, etc.)
   (Exceptions are the crane arms on construction vehicles).
   - If a person in on a personal mobility vehicle (e.g. skateboard, Segway, scooter), include the person in the annotation for `human.pedestrian.personal_mobility`.
   - If a vehicle is carrying people or bicycles, these are considered part of the object and should be included in the polygon.
 - `static_object.bicycle_rack`
   - All bicycles in a bicycle rack should be annotated.

[Top](#overview)


# Surfaces
nuImages includes surface classes as well:

|  Label | Short Description |
| --- | --- |
| [`flat.driveable_surface`](#1-flatdriveable_surface) | All paved or unpaved surfaces that a car can drive on with no concern of traffic rules. |
| [`vehicle.ego`](#2-vehicleego) | The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image. |

### 1. flat.driveable_surface
### 2. vehicle.ego


## Semantic Segmentation
### General Instructions
 - Only annotate a surface if its length and width are **both** greater than 20 pixels.
 - Annotations should tightly bound the edges of the area(s) of interest. 
 - If two areas/objects of interest are adjacent to each other, there should be no gap between the two annotations.
 - Annotate a surface only as far as it is clearly visible.
 - If a surface is occluded (e.g. by branches, trees, fence poles), only annotate the visible areas 
 (which are more than 20 pixels in length and width).
 - If a surface is covered by dirt or snow of less than 20 cm in height, include the dirt or snow in the annotation 
 (since it can be safely driven over).
 - If a surface has puddles in it, always include them in the annotation.
 - Do not annotate reflections of surfaces.
 
### Detailed Instructions 
 - `flat.driveable_surface`
   - All lanes and directions of traffic should be included. 
   Road, driveways (including the space where the driveway intersects the sidewalk), pedestrian crosswalks, 
   parking lots and road shoulders should also be included. Obstacles on the road (e.g vehicles, pedestrians) should be excluded from the outline.
   - Include walkable surface between traffic islands but excludes the elevated traffic islands.
   - Do not include sidewalks.
   - Include surface blocked by road blockers or pillars as long as they are the same surface from the driveable surface.
 - `vehicle.ego`
   - The entire ego vehicle should be annotated, including any sensors that can be seen in the frame.

[Top](#overview)
