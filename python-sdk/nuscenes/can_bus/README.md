# nuScenes CAN bus expansion
This page describes the Controller Area Network (CAN) bus expansion for the nuScenes dataset.
This is additional information that was published in January 2020 after the initial nuScenes release in March 2019.
The data can be used for tasks such as trajectory estimation, object detection and tracking.

# Overview
- [Introduction](#introduction)
  - [Notation](#notation)
- [Derived messages](#derived-messages)
  - [Meta](#meta)
  - [Route](#route)
- [CAN bus messages](#can-bus-messages)
  - [IMU](#imu)
  - [Pose](#pose)
  - [Steer Angle Feedback](#steer-angle-feedback)
  - [Vehicle Monitor](#vehicle-monitor)
  - [Zoe Sensors](#zoe-sensors)
  - [Zoe Vehicle Info](#zoe-vehicle-info)

## Introduction
The nuScenes dataset provides sensor data and annotations for 1000 scenes.
The CAN bus expansion includes additional information for these scenes.
The [CAN bus](https://copperhilltech.com/a-brief-introduction-to-controller-area-network/) is used for communication in automobiles and includes low-level messages regarding position, velocity, acceleration, steering, lights, battery and many more.
In addition to this raw data we also provide some meta data, such as statistics of the different message types.
Note that the CAN bus data is highly experimental.
Some data may be redundant across different messages.
Finally we extract a snippet of the route that the vehicle is currently travelling on.

### Notation
All messages of a particular type are captured in a file of the format `scene_0001_message.json`, where `0001` indicates the scene id and `message` the message name.
The messages (except *route*) contain different keys and values.
Below we notate the dimensionality as \[d\] to indicate that a value has d dimensions.
  
## Derived messages
Here we store additional information that is derived from various [CAN bus messages](#can-bus-messages) below.
These messages are timeless and therefore do not provide the `utime` timestamp common to the CAN bus messages.

### Meta
Format: `scene_0001_meta.json`

This meta file summarizes all CAN bus messages (except *route*) and provides some statistics that may be helpful to understand the data.
- message_count: \[1\] How many messages of this type were logged.
- message_freq: \[1\] The message frequency computed from timestamp and message_count.
- timespan: \[1\] How many seconds passed from first to last message in a scene. Usually around 20s.
- var_stats: (dict) Contains the maximum, mean, minimum and standard deviation for both the raw values and the differences of two consecutive values.

### Route
Format: `scene_0001_route.json`

Our vehicles follow predefined routes through the city.
The baseline route is the recommended navigation path for the vehicle to follow.
This is an ideal route that does not take into account any blocking objects or road closures.
The route contains the relevant section of the current scene and around 50m before and after it.
The route is stored as a list of 2-tuples (x, y) in meters on the current nuScenes map.
The data is recorded at approximately 50Hz.
For 3% of the scenes this data is not available as the drivers were not following any route.

## CAN bus messages
Here we list the raw CAN bus messages.
We store each type of message in a separate file for each scene (e.g. `scene-0001_ms_imu.json`).
Messages are stored in chronological order in the above file. 
Each message has the following field:
- utime: \[1\] The integer timestamp in microseconds that the actual measurement took place (e.g. 1531883549954657).
For the *Zoe Sensors* and *Zoe Vehicle Info* messages this info is not directly available and is therefore replaced by the timestamp when the CAN bus message was received.

### IMU 
Frequency: 100Hz

Format: `scene_0001_imu.json`

- linear_accel: \[3\] Acceleration vector (x, y, z) in the IMU frame in m/s/s.
- q: \[4\] Quaternion that transforms from IMU coordinates to a fixed reference frame. The yaw of this reference frame is arbitrary, determined by the IMU. However, the x-y plane of the reference frame is perpendicular to gravity, and z points up. 
- rotation_rate: \[3\] Angular velocity in rad/s around the x, y, and z axes, respectively, in the IMU coordinate frame.

### Pose
Frequency: 50Hz

Format: `scene_0001_pose.json`

The current pose of the ego vehicle, sampled at 50Hz.
- accel: \[3\] Acceleration vector in the ego vehicle frame in m/s/s.
- orientation: \[4\]  The rotation vector in the ego vehicle frame.
- pos: \[3\] The position (x, y, z) in meters in the global frame. This is identical to the [nuScenes ego pose](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose), but sampled at a higher frequency.
- rotation_rate: \[3\] The angular velocity vector of the vehicle in rad/s. This is expressed in the ego vehicle frame.
- vel: \[3\] The velocity in m/s, expressed in the ego vehicle frame.
 
### Steer Angle Feedback
Frequency: 100Hz

Format: `scene_0001_steeranglefeedback.json`

- value: \[1\] Steering angle feedback in radians in range \[-7.7, 6.3\]. 0 indicates no steering, positive values indicate left turns, negative values right turns.

### Vehicle Monitor
Frequency: 2Hz

Format: `scene_0001_vehicle_monitor.json`

- available_distance: \[1\] Available vehicle range given the current battery level in kilometers.
- battery_level: \[1\] Current battery level in range \[0, 100\].
- brake: \[1\] Braking pressure in bar. An integer in range \[0, 126\]. 
- brake_switch: \[1\] Brake switch as an integer, 1 (pedal not pressed), 2 (pedal pressed) or 3 (pedal confirmed pressed).
- gear_position: \[1\] The gear position as an integer, typically 0 (parked) or 7 (driving).
- left_signal: \[1\] Left turning signal as an integer, 0 (inactive) or 1 (active).
- rear_left_rpm: \[1\] Rear left brake speed in revolutions per minute.
- rear_right_rpm: \[1\] Rear right brake speed in revolutions per minute.
- right_signal: \[1\] Right turning signal as an integer, 0 (inactive) or 1 (active).
- steering: \[1\] Steering angle in degrees at a resolution of 0.1 in range \[-780, 779.9\].
- steering_speed: \[1\] Steering speed in degrees per second in range \[-465, 393\].
- throttle: \[1\] Throttle pedal position as an integer in range \[0, 1000\].
- vehicle_speed: \[1\] Vehicle speed in km/h at a resolution of 0.01. 
- yaw_rate: \[1\] Yaw turning rate in degrees per second at a resolution of 0.1.

### Zoe Sensors
Frequency: 794-973Hz

Format: `scene_0001_zoesensors.json`

- brake_sensor: \[1\] Vehicle brake sensor in range \[0.166, 0.631\]. High values indicate braking.
- steering_sensor: \[1\] Vehicle steering sensor in range \[0.176, 0.252\].
- throttle_sensor: \[1\] Vehicle throttle sensor in range \[0.105, 0.411]\.

### Zoe Vehicle Info
Frequency: 100Hz

Format: `scene_0001_zoe_veh_info.json`

- FL_wheel_speed: \[1\] Front left wheel speed. The unit is rounds per minute with a resolution of 0.0417rpm.
- FR_wheel_speed: \[1\] Front right wheel speed. The unit is rounds per minute with a resolution of 0.0417rpm.
- RL_wheel_speed: \[1\] Rear left wheel speed. The unit is rounds per minute with a resolution of 0.0417rpm.
- RR_wheel_speed: \[1\] Rear right wheel speed. The unit is rounds per minute with a resolution of 0.0417rpm.
- left_solar: \[1\] Zoe vehicle left solar sensor value as an integer.
- longitudinal_accel: \[1\] Longitudinal acceleration in meters per second squared at a resolution of 0.05.
- meanEffTorque: \[1\] Actual torque delivered by the engine in Newton meters at a resolution of 0.5. Values in range \[-400, 1647\], offset by -400.
- odom: \[1\] Odometry distance travelled modulo vehicle circumference. Values are in centimeters in range \[0, 124\]. Note that due to the low sampling frequency these values are only useful at low speeds.
- odom_speed: \[1\] Vehicle speed in km/h. Values in range \[0, 60\]. For a higher sampling rate refer to the pose.vel message.
- pedal_cc: \[1\] Throttle value. Values in range \[0, 1000\].
- regen: \[1\] Coasting throttle. Values in range \[0, 100\].
- requestedTorqueAfterProc: \[1\] Input torque requested in Newton meters at a resolution of 0.5. Values in range \[-400, 1647\], offset by -400.
- right_solar: \[1\] Zoe vehicle right solar sensor value as an integer.
- steer_corrected: \[1\] Steering angle (steer_raw) corrected by an offset (steer_offset_can).
- steer_offset_can: \[1\] Steering angle offset in degrees, typically -12.6.
- steer_raw: \[1\] Raw steering angle in degrees.
- transversal_accel: \[1\] Transversal acceleration in g at a resolution of 0.004.