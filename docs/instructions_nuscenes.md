# nuScenes Annotator Instructions 

# Overview
- [Instructions](#instructions)
- [Special Rules](#special-rules)
- [Labels](#labels)
- [Attributes](#attributes)
- [Detailed Instructions and Examples](#detailed-instructions-and-examples) 

# Instructions
+ Draw 3D bounding boxes around all objects from the [labels](#labels) list, and label them according to the instructions below. 
+ **Do not** apply more than one box to a single object.
+ Check every cuboid in every frame, to make sure all points are inside the cuboid and **look reasonable in the image view**.
+ For nighttime or rainy scenes, annotate objects as if these are daytime or normal weather scenes.

# Special Rules 
+ **Minimum number of points** : 
    + Label any target object containing **at least 1 LIDAR or RADAR point**, as long as you can be reasonably sure you know the location and shape of the object. Use your best judgment on correct cuboid position, sizing, and heading. 
+ **Cuboid Sizing** : 
    + **Cuboids must be very tight.** Draw the cuboid as close as possible to the edge of the object without excluding any LIDAR points. There should be almost no visible space between the cuboid border and the closest point on the object. 
+ **Extremities** : 
    + **If** an object has extremities (eg. arms and legs of pedestrians), **then** the bounding box should include the extremities. 
    + **Exception**: Do not include vehicle side view mirrors. Also, do not include other vehicle extremities (crane arms etc.) that are above 1.5 meters high. 
+ **Carried Object** : 
    + If a pedestrian is carrying an object (bags, umbrellas, tools etc.), such object will be included in the bounding box for the pedestrian. If two or more pedestrians are carrying the same object, the bounding box of only one of them will include the object.
+ **Stationary Objects** :
    + Sometimes stationary objects move over time due to errors in the localization. If a stationary object’s points shift over time, please create a separate cuboid for every frame.
+ **Use Pictures**:
    + For objects with few LIDAR or RADAR points, use the images to make sure boxes are correctly sized. If you see that a cuboid is too short in the image view, adjust it to cover the entire object based on the image view.
+ **Visibility Attribute** : 
    + The visibility attribute specifies the percentage of object pixels visible in the panoramic view of all cameras. 
    + ![](https://www.nuscenes.org/public/images/taxonomy_imgs/lidar_visibility_1.png)

# Labels 
**For every bounding box, include one of the following labels:**
1. **[Car or Van or SUV](#car-or-van-or-suv)**: Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, SUVs and jeeps.   

2. **[Truck](#truck)**: Vehicles primarily designed to haul cargo including pick-ups, lorrys, trucks and semi-tractors. Trailers hauled after a semi-tractor should be labeled as "Trailer".

    - **[Pickup Truck](#pickup-truck)**: A pickup truck is a light duty truck with an enclosed cab and an open or closed cargo area. A pickup truck can be intended primarily for hauling cargo or for personal use. 

    - **[Front Of Semi Truck](#front-of-semi-truck)**: Tractor part of a semi trailer truck. Trailers hauled after a semi-tractor should be labeled as a trailer. 

5. **[Bendy Bus](#bendy-bus)**: Buses and shuttles designed to carry more than 10 people and comprises two or more rigid sections linked by a pivoting joint. Annotate each section of the bendy bus individually. 

6. **[Rigid Bus](#rigid-bus)**: Rigid buses and shuttles designed to carry more than 10 people.

7. **[Construction Vehicle](#construction-vehicle)**: Vehicles primarily designed for construction. Typically very slow moving or stationary. Cranes and extremities of construction vehicles are only included in annotations if they interfere with traffic. Trucks used to hauling rocks or building materials are considered trucks rather than construction vehicles. 

8. **[Motorcycle](#motorcycle)**: Gasoline or electric powered 2-wheeled vehicle designed to move rapidly (at the speed of standard cars) on the road surface. This category includes all motorcycles, vespas and scooters. It also includes light 3-wheel vehicles, often with a light plastic roof and open on the sides, that tend to be common in Asia. If there is a rider and/or passenger, include them in the box.

9. **[Bicycle](#bicycle)**: Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on road surface, sidewalks or bicycle paths. If there is a rider and/or passenger, include them in the box.

10. **[Bicycle Rack](#bicycle-rack)**: Area or device intended to park or secure the bicycles in a row. It includes all the bicycles parked in it and any empty slots that are intended for parking bicycles. Bicycles that are not part of the rack should not be included. Instead they should be annotated as bicycles separately.

11. **[Trailer](#trailer)**: Any vehicle trailer, both for trucks, cars and motorcycles (regardless of whether currently being towed or not). For semi-trailers (containers) label the truck itself as "Truck".

12. **[Police Vehicle](#police-vehicle)**: All types of police vehicles including police bicycles and motorcycles. 

13. **[Ambulance](#ambulance)**: All types of ambulances. 

14. **[Adult Pedestrian](#adult-pedestrian)**: An adult pedestrian moving around the cityscape. Mannequins should also be annotated as Adult Pedestrian. 

15. **[Child Pedestrian](#child-pedestrian)**: A child pedestrian moving around the cityscape. 

16. **[Construction Worker](#construction-worker)**: A human in the scene whose main purpose is construction work.

17. **[Stroller](#stroller)**: Any stroller. If a person is in the stroller, include in the annotation. If a pedestrian pushing the stroller, then they should be labeled separately. 

18. **[Wheelchair](#wheelchair)**: Any type of wheelchair. If a pedestrian is pushing the wheelchair then they should be labeled separately.

19. **[Portable Personal Mobility Vehicle](#portable-personal-mobility-vehicle)**: A small electric or self-propelled vehicle, e.g. skateboard, segway, or scooters, on which the person typically travels in a upright position. Driver and (if applicable) rider should be included in the bounding box along with the vehicle. 

20. **[Police Officer](#police-officer)**: Any type of police officer, regardless whether directing the traffic or not.

21. **[Animal](#animal)**: All animals, e.g. cats, rats, dogs, deer, birds. 

22. **[Traffic Cone](#traffic-cone)**: All types of traffic cones.

23. **[Temporary Traffic Barrier](#temporary-traffic-barrier)**: Any metal, concrete or water barrier temporarily placed in the scene in order to re-direct vehicle or pedestrian traffic. In particular, includes barriers used at construction zones. If there are multiple barriers either connected or just placed next to each other, they should be annotated separately.

24. **[Pushable Pullable Object](#pushable-pullable-object)**: Objects that a pedestrian may push or pull. For example dolleys, wheel barrows, garbage-bins with wheels, or shopping carts. Typically not designed to carry humans.

25. **[Debris](#debris)**: Debris or movable object that is too large to be driven over safely. Includes misc. things like trash bags, temporary road-signs, objects around construction zones, and trash cans. 

# Attributes 
1. **For every object, include the attribute:** 
    + **Visibility**: 
        + **0%-40%**: The object is 0% to 40% visible in panoramic view of all cameras.
        + **41%-60%**: The object is 41% to 60% visible in panoramic view of all cameras.
        + **61%-80%**: The object is 61% to 80% visible in panoramic view of all cameras.
        + **81%-100%**: The object is 81% to 100% visible in panoramic view of all cameras.
    + This attribute specifies the percentage of an object visible through the cameras. For this estimation to be carried out, all the different camera views should be considered as one and the visible portion would be gauged in the resulting **panoramic view**
    + ![](https://www.nuscenes.org/public/images/taxonomy_imgs/lidar_visibility_1.png)
2. **For each vehicle with four or more wheels, select the status:** 
    + **Vehicle Activity**: 
        + **Parked**: Vehicle is stationary (usually for longer duration) with no immediate intent to move.
        + **Stopped**: Vehicle, with a driver/rider in/on it, is currently stationary but has an intent to move.
        + **Moving**: Vehicle is moving.
3. **For each bicycle, motorcycle and portable personal mobility vehicle, select the rider status.** 
    + **Has Rider**: 
        + **Yes**: There is a rider on the bicycle or motorcycle.
        + **No**: There is NO rider on the bicycle or motorcycle.
4. **For each human in the scene, select the status** 
    + **Human Activity**: 
        + **Sitting or Lying Down**: The human is sitting or lying down.
        + **Standing**: The human is standing.
        + **Moving**: The human is moving. 

<br><br><br>
 # Detailed Instructions and Examples 
 
Bounding Box color convention in example images: 
 + **Green**: Objects like this should be annotated 
 + **Red**: Objects like this should not be annotated 

 
## Car or Van or SUV
+ Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, SUVs and jeeps.  
    + If the vehicle is designed to carry more than 10 people label it is a bus. 
    + If it is primarily designed to haul cargo it is a truck. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/personal_vehicle_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/personal_vehicle_4.jpg)

 [Top](#overview)
## Truck 
+ Vehicles primarily designed to haul cargo including pick-ups, lorrys, trucks and semi-tractors. Trailers hauled after a semi-tractor should be labeled as vehicle.trailer.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/truck_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/truck_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/truck_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/truck_5.jpg)

**Pickup Truck** 
+ A pickup truck is a light duty truck with an enclosed cab and an open or closed cargo area. A pickup truck can be intended primarily for hauling cargo or for personal use.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pickup_truck_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pickup_truck_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pickup_truck_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pickup_truck_5.jpg)

**Front Of Semi Truck**
+ Tractor part of a semi trailer truck. Trailers hauled after a semi-tractor should be labeled as a trailer. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_2.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_3.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_5.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_6.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_7.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/front_of_semi_truck_8.png)

 [Top](#overview)
## Bendy Bus 
+ Buses and shuttles designed to carry more than 10 people and comprises two or more rigid sections linked by a pivoting joint. 
    + Annotate each section of the bendy bus individually. 
    + If you cannot see the pivoting joint of the bus, annotate it as **rigid bus**. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bendy_bus_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bendy_bus_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bendy_bus_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bendy_bus_4.jpg)

 [Top](#overview)
## Rigid Bus 
+ Rigid buses and shuttles designed to carry more than 10 people. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/rigid_bus_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/rigid_bus_3.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/rigid_bus_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/rigid_bus_5.jpg)

 [Top](#overview)
## Construction Vehicle 
+ Vehicles primarily designed for construction. Typically very slow moving or stationary. 
    + Trucks used to hauling rocks or building materials are considered as truck rather than construction vehicles. 
    + Cranes and extremities of construction vehicles are only included in annotations if they interferes with traffic. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_vehicle_7.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_vehicle_8.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_vehicle_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_vehicle_9.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_vehicle_6.jpg) 

 [Top](#overview)
## Motorcycle 
+ Gasoline or electric powered 2-wheeled vehicle designed to move rapidly (at the speed of standard cars) on the road surface. This category includes all motorcycles, vespas and scooters. It also includes light 3-wheel vehicles, often with a light plastic roof and open on the sides, that tend to be common in Asia. 
    + If there is a rider, include the rider in the box.
    + If there is a passenger, include the passenger in the box. 
    + If there is a pedestrian standing next to the motorcycle, do NOT include in the annotation. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_5.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_6.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_7.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/motorcycle_8.jpg)

 [Top](#overview)
## Bicycle 
+ Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on road surface, sidewalks or bicycle paths. 
    + If there is a rider, include the rider in the box 
    + If there is a passenger, include the passenger in the box 
    + If there is a pedestrian standing next to the bicycle, do NOT include in the annotation 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bicycle_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bicycle_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bicycle_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bicycle_4.jpg) 

 [Top](#overview)
## Bicycle Rack
+ Area or device intended to park or secure the bicycles in a row. It includes all the bicycles parked in it and any empty slots that are intended for parking bicycles.
    + Bicycles that are not part of the rack should not be included. Instead they should be annotated as bicycles separately.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_2.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_3.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_4.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_5.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/bike_rack_6.png)

 [Top](#overview)
## Trailer 
+ Any vehicle trailer, both for trucks, cars and motorcycles (regardless of whether currently being towed or not). For semi-trailers (containers) label the truck itself as "front of semi truck".
    + A vehicle towed by another vehicle should be labeled as vehicle (not as trailer). 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_4.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_5.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_6.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_7.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_8.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_9.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/trailer_10.jpg)

 [Top](#overview)
## Police Vehicle 
+ All types of police vehicles including police bicycles and motorcycles. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/police_vehicle_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/police_vehicle_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/police_vehicle_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/police_vehicle_3.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/police_vehicle_4.png)

 [Top](#overview)
## Ambulance 
+ All types of ambulances. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/ambulance_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/ambulance_3.jpg)

 [Top](#overview)
## Adult Pedestrian 
+ An adult pedestrian moving around the cityscape. 
    + Mannequins should also be treated as adult pedestrian.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/mannequin_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/adult_pedestrian_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/adult_pedestrian_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/adult_pedestrian_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/adult_pedestrian_5.jpg)

 [Top](#overview)
## Child Pedestrian 
+ A child pedestrian moving around the cityscape. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/child_pedestrian_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/child_pedestrian_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/child_pedestrian_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/child_pedestrian_5.jpg)

 [Top](#overview)
## Construction Worker 
+ A human in the scene whose main purpose is construction work. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_worker_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_worker_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_worker_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/construction_worker_4.jpg)

 [Top](#overview)
## Stroller 
+ Any stroller 
    + If a person is in the stroller, include in the annotation. 
    + Pedestrians pushing strollers should be labeled separately.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/stroller_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/stroller_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/stroller_5.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/stroller_6.jpg)

 [Top](#overview)
## Wheelchair 
+ Any type of wheelchair 
    + If a person is in the wheelchair, include in the annotation. 
    + Pedestrians pushing wheelchairs should be labeled separately.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/wheelchair_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/wheelchair_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/wheelchair_3.jpg)

 [Top](#overview)
## Portable Personal Mobility Vehicle
+ A small electric or self-propelled vehicle, e.g. skateboard, segway, or scooters, on which the person typically travels in a upright position. Driver and (if applicable) rider should be included in the bounding box along with the vehicle. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/personal_mobility_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/personal_mobility_3.png)

 [Top](#overview)
## Police Officer
+ Any type of police officer, regardless whether directing the traffic or not.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_police_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_police_2.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/other_police_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/other_police_2.png)

 [Top](#overview)
## Animal 
+ All animals, e.g. cats, rats, dogs, deer, birds. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/small_animal_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/small_animal_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/large_animal_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/large_animal_3.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/large_animal_4.png)

 [Top](#overview)
## Traffic Cone 
+ All types of traffic cones.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_cone_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_cone_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_cone_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_cone_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/traffic_cone_5.jpg)

 [Top](#overview)
## Temporary Traffic Barrier 
+ Any metal, concrete or water barrier temporarily placed in the scene in order to re-direct vehicle or pedestrian traffic. In particular, includes barriers used at construction zones. 
    + If there are multiple barriers either connected or just placed next to each other, they should be annotated separately. 
    + If barriers are installed permanently, then do NOT include them.

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_1.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_6.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_2.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_3.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_4.jpg)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/temporary_traffic_barrier_5.png)

 [Top](#overview)
## Pushable Pullable Object 
+ Objects that a pedestrian may push or pull. For example dolleys, wheel barrows, garbage-bins with wheels, or shopping carts. Typically not designed to carry humans. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pushable_pullable_2.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pushable_pullable_4.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pushable_pullable_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/pushable_pullable_3.png)

 [Top](#overview)
## Debris 
+ Debris or movable object that is left **on the driveable surface** that is too large to be driven over safely, e.g tree branch, full trash bag etc. 

    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/movable_obstacle_1.png)
    ![](https://www.nuscenes.org/public/images/taxonomy_imgs/movable_obstacle_2.png)

 [Top](#overview)
