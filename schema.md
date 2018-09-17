Database schema
==========
*(this schema is copied from the [devkit](https://github.com/nutonomy/nuscenes-devkit/blob/master/schema.md) and may be outdated)*

category
---------

Taxonomy of object categories (e.g. vehicle, human). 
Subcategories are delineated by a period.
```
category {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Category name. Subcategories indicated by period.
   "description":             <str> -- Category description.
}
```
attribute
---------

An attribute is a property of an instance that can change while the category remains the same. 
 Example: vehicle pose, and whether or not a bicycle has a rider.
```
attribute {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Attribute name.
   "description":             <str> -- Attribute description.
}
```
visibility
---------

The visibility of an instance is the fraction of pixels visible in all 6 images. Binned into 5 bins of 20%.
```
visibility {
   "token":                   <str> -- Unique record identifier.
   "level":                   <str> -- Visibility level.
   "description":             <str> -- Description of visibility level.
}
```
instance
---------

An object instance, e.g. particular vehicle. This table is an enumeration of all object 
instances we observed. Note that instances are not tracked across scenes.
```
instance {
   "token":                   <str> -- Unique record identifier.
   "category_token":          <str> -- Foreign key. Object instance category.
   "nbr_annotations":         <int> -- Number of annotations of this instance.
   "first_annotation_token":  <str> -- Foreign key. Points to the first annotation of this instance.
   "last_annotation_token":   <str> -- Foreign key. Points to the last annotation of this instance.
}
```
sensor
---------

A specific sensor type.
```
sensor {
   "token":                   <str> -- Unique record identifier.
   "channel":                 <str> -- Sensor channel name.
   "modality":                <str> {camera, lidar, radar} -- Sensor modality. Supports category(ies) in brackets.
}
```
calibrated_sensor
---------

Definition of a particular sensor as calibrated on a particular vehicle. All extrinsic parameters are 
given with respect to the ego vehicle body frame.
```
calibrated_sensor {
   "token":                   <str> -- Unique record identifier.
   "sensor_token":            <str> -- Foreign key pointing to the sensor type.
   "translation":             <float> [3] -- Coordinate system origin: x, y, z.
   "rotation":                <float> [4] -- Coordinate system orientation in quaternions.
   "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration matrix.
   "camera_distortion":       <float> [*] -- Distortion per convention of the CalTech camera calibration toolbox. Can be 5-10 coefficients.
}
```
ego_pose
---------

Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system.
```
ego_pose {
   "token":                   <str> -- Unique record identifier.
   "translation":             <float> [3] -- Coordinate system origin: x, y, z.
   "rotation":                <float> [4] -- Coordinate system orientation in quaternions.
   "timestamp":               <int> -- Unix time stamp.
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
scene
---------

A scene is a 20s long sequence of consecutive frames extracted from a log. 
Multiple scenes can come from the same log. 
Note that object identities (instance tokens) are not preserved across scenes.
```
scene {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Short string identifier.
   "description":             <str> -- Longer description of the scene.
   "log_token":               <str> -- Foreign key. Points to log from where the data was extracted.
   "nbr_samples":             <int> -- Number of samples in this scene.
   "first_sample_token":      <str> -- Foreign key. Points to the first sample in scene.
   "last_sample_token":       <str> -- Foreign key. Points to the last sample in scene.
}
```
sample
---------

A sample is data collected at (approximately) the same timestamp as part of a single LIDAR sweep.
```
sample {
   "token":                   <str> -- Unique record identifier.
   "timestamp":               <int> -- Unix time stamp.
   "scene_token":             <str> -- Foreign key pointing to the scene.
   "next":                    <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
   "prev":                    <str> -- Foreign key. Sample that precedes this in time. Empty if start of scene.
}
```
sample_data
---------

A sensor data e.g. image, point cloud, radar return. For sample_data with is_key_frame=True, the time-stamps 
should be very close to the sample it points to. For non key-frames the sample_data points to the 
sample that follows closest in time.
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
sample_annotation
---------

A geometry defining the position of an object seen in a sample. All location data is given with respect 
to the world coordinate system.
```
sample_annotation {
   "token":                   <str> -- Unique record identifier.
   "sample_token":            <str> -- Foreign key. NOTE: this points to a sample NOT a sample_data since annotations are done on the sample level taking all relevant sample_data into account.
   "instance_token":          <str> -- Foreign key. Which object instance is this annotating. An instance can have multiple annotations over time.
   "attribute_tokens":        <str> [n] -- Foreign keys. List of attributes for this annotation. Attributes can change over time, so they belong here, not in the object table.
   "visibility_token":        <str> -- Foreign key. Visibility may also change over time.
   "translation":             <float> [3] -- Bounding box location as center_x, center_y, center_z.
   "size":                    <float> [3] -- Bounding box size as width, length, height.
   "rotation":                <float> [4] -- Bounding box orientation in quaternions.
   "next":                    <str> -- Foreign key. Sample annotation from the same object instance that follows this in time. Empty if this is the last annotation for this object.
   "prev":                    <str> -- Foreign key. Sample annotation from the same object instance that precedes this in time. Empty if this is the first annotation for this object.
}
```
map
---------

Map data that is stored as binary semantic masks from a top-down view.
```
map {
   "token":                   <str> -- Unique record identifier.
   "log_token":               <str> -- Foreign key.
   "category":                <str> -- Map category, e.g. semantic_prior for drivable surface and sidewalk
   "filename":                <str> -- Relative path to the file with the map mask.
}
```
