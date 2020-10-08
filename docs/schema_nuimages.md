nuImages schema
==========
This document describes the database schema used in nuImages.
All annotations and meta data (including calibration, maps, vehicle coordinates etc.) are covered in a relational database.
The database tables are listed below.
Every row can be identified by its unique primary key `token`.
Foreign keys such as `sample_token` may be used to link to the `token` of the table `sample`.
Please refer to the [tutorial](https://www.nuscenes.org/nuimages#tutorials) for an introduction to the most important database tables.

![](https://www.nuscenes.org/public/images/nuimages-schema.svg)

attribute
---------
An attribute is a property of an instance that can change while the category remains the same.
Example: a vehicle being parked/stopped/moving, and whether or not a bicycle has a rider.
The attributes in nuImages are a superset of those in nuScenes.
```
attribute {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Attribute name.
   "description":             <str> -- Attribute description.
}
```

calibrated_sensor
---------
Definition of a particular camera as calibrated on a particular vehicle.
All extrinsic parameters are given with respect to the ego vehicle body frame.
Contrary to nuScenes, all camera images come distorted and unrectified.
```
calibrated_sensor {
   "token":                   <str> -- Unique record identifier.
   "sensor_token":            <str> -- Foreign key pointing to the sensor type.
   "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
   "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
   "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
   "camera_distortion":       <float> [5 or 6] -- Camera calibration parameters [k1, k2, p1, p2, k3, k4]. We use the 5 parameter camera convention of the CalTech camera calibration toolbox, that is also used in OpenCV. Only for fish-eye lenses in CAM_BACK do we use the 6th parameter (k4).
}
```

category
---------
Taxonomy of object categories (e.g. vehicle, human). 
Subcategories are delineated by a period (e.g. `human.pedestrian.adult`).
The categories in nuImages are the same as in nuScenes (w/o lidarseg), plus `flat.driveable_surface`.
```
category {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Category name. Subcategories indicated by period.
   "description":             <str> -- Category description.
}
```

ego_pose
---------
Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system of the log's map.
The ego_pose is the output of a lidar map-based localization algorithm described in our paper.
The localization is 2-dimensional in the x-y plane.
Warning: nuImages is collected from almost 500 logs with different maps versions.
Therefore the coordinates **should not be compared across logs** or rendered on the semantic maps of nuScenes.
```
ego_pose {
   "token":                   <str> -- Unique record identifier.
   "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
   "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
   "timestamp":               <int> -- Unix time stamp.
   "rotation_rate":           <float> [3] -- The angular velocity vector (x, y, z) of the vehicle in rad/s. This is expressed in the ego vehicle frame.
   "acceleration":            <float> [3] -- Acceleration vector (x, y, z) in the ego vehicle frame in m/s/s. The z value is close to the gravitational acceleration `g = 9.81 m/s/s`.
   "speed":                   <float> -- The speed of the ego vehicle in the driving direction in m/s.
}
```

log
---------
Information about the log from which the data was extracted.
```
log {
   "token":                   <str> -- Unique record identifier.
   "logfile":                 <str> -- Log file name.
   "vehicle":                 <str> -- Vehicle name.
   "date_captured":           <str> -- Date (YYYY-MM-DD).
   "location":                <str> -- Area where log was captured, e.g. singapore-onenorth.
}
```

object_ann
---------
The annotation of a foreground object (car, bike, pedestrian) in an image.
Each foreground object is annotated with a 2d box, a 2d instance mask and category-specific attributes.
```
object_ann {
    "token":                  <str> -- Unique record identifier.
    "sample_data_token":      <str> -- Foreign key pointing to the sample data, which must be a keyframe image.
    "category_token":         <str> -- Foreign key pointing to the object category.
    "attribute_tokens":       <str> [n] -- Foreign keys. List of attributes for this annotation.
    "bbox":                   <int> [4] -- Annotated amodal bounding box. Given as [xmin, ymin, xmax, ymax].
    "mask":                   <RLE> -- Run length encoding of instance mask using the pycocotools package.
}
```

sample_data
---------
Sample_data contains the images and information about when they were captured.
Sample_data covers all images, regardless of whether they are a keyframe or not.
Only keyframes are annotated.
For every keyframe, we also include up to 6 past and 6 future sweeps at 2 Hz.
We can navigate between consecutive images using the `prev` and `next` pointers.
The sample timestamp is inherited from the keyframe camera sample_data timestamp.
```
sample_data {
   "token":                   <str> -- Unique record identifier.
   "sample_token":            <str> -- Foreign key. Sample to which this sample_data is associated.
   "ego_pose_token":          <str> -- Foreign key.
   "calibrated_sensor_token": <str> -- Foreign key.
   "filename":                <str> -- Relative path to data-blob on disk.
   "fileformat":              <str> -- Data file format.
   "width":                   <int> -- If the sample data is an image, this is the image width in pixels.
   "height":                  <int> -- If the sample data is an image, this is the image height in pixels.
   "timestamp":               <int> -- Unix time stamp.
   "is_key_frame":            <bool> -- True if sample_data is part of key_frame, else False.
   "next":                    <str> -- Foreign key. Sample data from the same sensor that follows this in time. Empty if end of scene.
   "prev":                    <str> -- Foreign key. Sample data from the same sensor that precedes this in time. Empty if start of scene.
}
```

sample
---------
A sample is an annotated keyframe selected from a large pool of images in a log.
Every sample has up to 13 camera sample_datas corresponding to it.
These include the keyframe, which can be accessed via `key_camera_token`.
```
sample {
   "token":                   <str> -- Unique record identifier.
   "timestamp":               <int> -- Unix time stamp.
   "log_token":               <str> -- Foreign key pointing to the log.
   "key_camera_token":        <str> -- Foreign key of the sample_data corresponding to the camera keyframe.
}
```

sensor
---------
A specific sensor type.
```
sensor {
   "token":                   <str> -- Unique record identifier.
   "channel":                 <str> -- Sensor channel name.
   "modality":                <str> -- Sensor modality. Always "camera" in nuImages.
}
```

surface_ann
---------
The annotation of a background object (driveable surface) in an image.
Each background object is annotated with a 2d semantic segmentation mask.
```
surface_ann {
   "token":                   <str> -- Unique record identifier.
    "sample_data_token":      <str> -- Foreign key pointing to the sample data, which must be a keyframe image.
    "category_token":         <str> -- Foreign key pointing to the surface category.
    "mask":                   <RLE> -- Run length encoding of segmentation mask using the pycocotools package.
}
```
