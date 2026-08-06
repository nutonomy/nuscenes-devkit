use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use ahash::{HashMap, HashMapExt};
use enum_map::EnumMap;
use log::debug;
use num_format::ToFormattedString;
use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyValueError};
use pyo3::types::PyDict;
use pyo3::{IntoPyObjectExt, prelude::*};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;

use crate::common::SensorChannel;
use crate::domain::*;
use crate::model::*;
use crate::proxy::*;
use crate::table::{AsRefToken, Table};

macro_rules! load_table {
    ($scope:ident, $root:expr, $filename:expr, $target:ident) => {
        $scope.spawn(|_| {
            let _ = $target.set(load_json(&$root, $filename));
        });
    };
}

fn load_json<T>(path: &Path, name: &str) -> PyResult<T>
where
    T: Deserialize<'static>,
{
    let path = path.join(name);
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!("{name} not found at {}", path.display())));
    }
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("Failed to open {name}: {}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| PyValueError::new_err(format!("Failed to map {name}: {}", e)))?;
    let mmap = Box::leak(Box::new(mmap));
    let json =
        std::str::from_utf8(mmap).map_err(|e| PyValueError::new_err(format!("Invalid UTF-8 token found: {e}")))?;
    serde_json::from_str(&json).map_err(|e| PyValueError::new_err(format!("Failed to parse {name}: {}", e)))
}

#[pyclass(module = "nuscenes._lib")]
/// Class for loading tables and querying data from the nuScenes dataset.
pub struct Tables {
    version: String,
    dataroot: String,
    // Vehicle
    log: Table<Log<'static>>,
    map: Table<Map<'static>>,
    sensor: Table<Sensor>,
    calib: Table<CalibratedSensor>,
    // Extraction
    scene: Table<Scene<'static>>,
    sample: Table<Sample>,
    sample_data: Table<SampleData<'static>>,
    ego_pose: Table<EgoPose>,
    // Annotation
    instance: Table<Instance>,
    sample_ann: Table<SampleAnnotation<'static>>,
    // Taxonomy
    category: Table<Category<'static>>,
    attribute: Table<Attribute<'static>>,
    // Extensions
    lidarseg: Option<Table<LidarSeg<'static>>>,
    panoptic: Option<Table<Panoptic<'static>>>,
}

#[pymethods]
impl Tables {
    #[new]
    fn new(version: &str, dataroot: &str) -> PyResult<Self> {
        let table_root = Path::new(dataroot).join(version);
        if !table_root.exists() {
            return Err(PyFileNotFoundError::new_err(format!("Dataset not found at {}", table_root.display())));
        }

        let start_time = Instant::now();
        debug!(target: "nuscenes", "======\nLoading NuScenes tables for version {}...", version);

        let log: OnceLock<PyResult<Vec<LogModel>>> = OnceLock::new();
        let map: OnceLock<PyResult<Vec<MapModel>>> = OnceLock::new();
        let sensor: OnceLock<PyResult<Vec<SensorModel>>> = OnceLock::new();
        let calib: OnceLock<PyResult<Vec<CalibratedSensorModel>>> = OnceLock::new();
        let scene: OnceLock<PyResult<Vec<SceneModel>>> = OnceLock::new();
        let sample: OnceLock<PyResult<Vec<SampleModel>>> = OnceLock::new();
        let sample_data: OnceLock<PyResult<Vec<SampleDataModel>>> = OnceLock::new();
        let ego_pose: OnceLock<PyResult<Vec<EgoPoseModel>>> = OnceLock::new();
        let instance: OnceLock<PyResult<Vec<InstanceModel>>> = OnceLock::new();
        let sample_ann: OnceLock<PyResult<Vec<SampleAnnotationModel>>> = OnceLock::new();
        let category: OnceLock<PyResult<Vec<CategoryModel>>> = OnceLock::new();
        let attribute: OnceLock<PyResult<Vec<AttributeModel>>> = OnceLock::new();
        let lidarseg: OnceLock<PyResult<Vec<LidarSegModel>>> = OnceLock::new();
        let panoptic: OnceLock<PyResult<Vec<PanopticModel>>> = OnceLock::new();

        rayon::scope(|s| {
            load_table!(s, table_root, "log.json", log);
            load_table!(s, table_root, "map.json", map);
            load_table!(s, table_root, "sensor.json", sensor);
            load_table!(s, table_root, "calibrated_sensor.json", calib);
            load_table!(s, table_root, "scene.json", scene);
            load_table!(s, table_root, "sample.json", sample);
            load_table!(s, table_root, "sample_data.json", sample_data);
            load_table!(s, table_root, "ego_pose.json", ego_pose);
            load_table!(s, table_root, "instance.json", instance);
            load_table!(s, table_root, "sample_annotation.json", sample_ann);
            load_table!(s, table_root, "category.json", category);
            load_table!(s, table_root, "attribute.json", attribute);
            if table_root.join("lidarseg.json").exists() {
                load_table!(s, table_root, "lidarseg.json", lidarseg);
            }
            if table_root.join("panoptic.json").exists() {
                load_table!(s, table_root, "panoptic.json", panoptic);
            }
        });

        // into_iter() is used instead of into_par_iter() for tables that are relatively small.
        let log = log.into_inner().unwrap()?; // 68
        let map = Table::new(map.into_inner().unwrap()?.into_iter().map(Map::from).collect()); // 4
        let sensor = Table::new(sensor.into_inner().unwrap()?.into_iter().map(Sensor::from).collect()); // 12
        let calib = Table::new(calib.into_inner().unwrap()?.into_par_iter().map(CalibratedSensor::from).collect()); // 10200

        let scene = Table::new(scene.into_inner().unwrap()?.into_par_iter().map(Scene::from).collect()); // 850
        let sample = sample.into_inner().unwrap()?; // 34,149
        let sample_data = sample_data.into_inner().unwrap()?; // 2,631,083
        let ego_pose = Table::new(ego_pose.into_inner().unwrap()?.into_par_iter().map(EgoPose::from).collect()); // 2,631,083

        let instance = Table::new(instance.into_inner().unwrap()?.into_par_iter().map(Instance::from).collect()); // 64,386
        let sample_ann = sample_ann.into_inner().unwrap()?; // 1,166,187

        let category = Table::new(category.into_inner().unwrap()?.into_iter().map(Category::from).collect()); // 23
        let attribute = Table::new(attribute.into_inner().unwrap()?.into_iter().map(Attribute::from).collect()); // 8

        let lidarseg = lidarseg
            .into_inner()
            .transpose()?
            .map(|lidarseg| Table::new(lidarseg.into_par_iter().map(LidarSeg::from).collect()));
        let panoptic = panoptic
            .into_inner()
            .transpose()?
            .map(|panoptic| Table::new(panoptic.into_par_iter().map(Panoptic::from).collect()));

        // TODO: Load image annotations table created by `export_2d_annotations_as_json()`

        let elapsed = start_time.elapsed();
        let locale = num_format::Locale::en;
        debug!(target: "nuscenes", "{:>9} log", log.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} map", map.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} sensor", sensor.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} calibrated sensor", calib.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} scene", scene.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} sample", sample.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} sample data", sample_data.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} ego pose", ego_pose.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} instance", instance.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} sample annotation", sample_ann.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} category", category.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "{:>9} attribute", attribute.len().to_formatted_string(&locale));
        debug!(target: "nuscenes", "Done loading in {:.3} seconds", elapsed.as_secs_f32());

        let start_time = Instant::now();
        debug!(target: "nuscenes", "Reverse indexing...");

        let log = {
            let log_to_map: HashMap<[u8; 16], [u8; 16]> =
                map.iter().flat_map(|map| map.log_tokens.iter().map(move |tok| (*tok, map.token))).collect();
            let log =
                log.into_iter().map(|log| Log::from_model(log_to_map.get(&log.token).copied().unwrap(), log)).collect();
            Table::new(log)
        };

        let sample_data = {
            let sample_data = sample_data
                .into_par_iter()
                .map(|sd| {
                    let cal = calib.get(&sd.calibrated_sensor_token).unwrap();
                    let sen = sensor.get(&cal.sensor_token).unwrap();
                    SampleData::from_model(sen.modality, sen.channel, sd)
                })
                .collect();
            Table::new(sample_data)
        };

        let sample_ann = {
            let sample_ann = sample_ann
                .into_par_iter()
                .map(|ann| {
                    let ins = instance.get(&ann.instance_token).unwrap();
                    let cat = category.get(&ins.category_token).unwrap();
                    SampleAnnotation::from_model(cat.name.clone(), ann)
                })
                .collect();
            Table::new(sample_ann)
        };

        let sample = {
            let sample_to_sd: HashMap<[u8; 16], EnumMap<SensorChannel, [u8; 16]>> =
                sample_data.iter().fold(HashMap::with_capacity(sample.len()), |mut acc, sd| {
                    let map = acc.entry(sd.sample_token).or_default();
                    map[sd.channel] = sd.token;
                    acc
                });
            let sample_to_ann: HashMap<[u8; 16], Vec<[u8; 16]>> =
                sample_ann.iter().fold(HashMap::with_capacity(sample.len()), |mut acc, ann| {
                    acc.entry(ann.sample_token).or_default().push(ann.token);
                    acc
                });
            let sample = sample
                .into_par_iter()
                .map(|sample| {
                    let data = sample_to_sd.get(&sample.token).cloned().unwrap_or_default();
                    let anns = sample_to_ann.get(&sample.token).cloned().unwrap_or_default().into_boxed_slice();
                    Sample::from_model(data, anns, sample)
                })
                .collect();
            Table::new(sample)
        };

        let elapsed = start_time.elapsed();
        debug!(target: "nuscenes", "Done reverse indexing in {:.3} seconds\n======", elapsed.as_secs_f32());

        Ok(Self {
            version: version.to_string(),
            dataroot: dataroot.to_string(),
            log,
            map,
            sensor,
            calib,
            scene,
            sample,
            sample_data,
            ego_pose,
            instance,
            sample_ann,
            category,
            attribute,
            lidarseg,
            panoptic,
        })
    }

    #[getter]
    fn log(&self) -> LogView {
        LogView { data: self.log.data.clone() }
    }

    #[getter]
    fn map(&self) -> MapView {
        MapView { data: self.map.data.clone() }
    }

    #[getter]
    fn sensor(&self) -> SensorView {
        SensorView { data: self.sensor.data.clone() }
    }

    #[getter]
    fn calibrated_sensor(&self) -> CalibratedSensorView {
        CalibratedSensorView { data: self.calib.data.clone() }
    }

    #[getter]
    fn scene(&self) -> SceneView {
        SceneView { data: self.scene.data.clone() }
    }

    #[getter]
    fn sample(&self) -> SampleView {
        SampleView { data: self.sample.data.clone() }
    }

    #[getter]
    fn sample_data(&self) -> SampleDataView {
        SampleDataView { data: self.sample_data.data.clone() }
    }

    #[getter]
    fn ego_pose(&self) -> EgoPoseView {
        EgoPoseView { data: self.ego_pose.data.clone() }
    }

    #[getter]
    fn instance(&self) -> InstanceView {
        InstanceView { data: self.instance.data.clone() }
    }

    #[getter]
    fn sample_annotation(&self) -> SampleAnnotationView {
        SampleAnnotationView { data: self.sample_ann.data.clone() }
    }

    #[getter]
    fn category(&self) -> CategoryView {
        CategoryView { data: self.category.data.clone() }
    }

    #[getter]
    fn attribute(&self) -> AttributeView {
        AttributeView { data: self.attribute.data.clone() }
    }

    #[getter]
    fn lidarseg(&self) -> Option<LidarSegView> {
        self.lidarseg.as_ref().map(|t| LidarSegView { data: t.data.clone() })
    }

    #[getter]
    fn panoptic(&self) -> Option<PanopticView> {
        self.panoptic.as_ref().map(|t| PanopticView { data: t.data.clone() })
    }

    fn get<'py>(slf: PyRef<'py, Self>, table: &str, token: &str) -> PyResult<Bound<'py, PyDict>> {
        let mut bytes = [0u8; 16];
        hex::decode_to_slice(token, &mut bytes)
            .map_err(|_| PyValueError::new_err(format!("Invalid token format: {token}")))?;

        match table {
            "log" => slf.lookup_in_table(slf.py(), &slf.log, &bytes),
            "map" => slf.lookup_in_table(slf.py(), &slf.map, &bytes),
            "sensor" => slf.lookup_in_table(slf.py(), &slf.sensor, &bytes),
            "calibrated_sensor" => slf.lookup_in_table(slf.py(), &slf.calib, &bytes),
            "scene" => slf.lookup_in_table(slf.py(), &slf.scene, &bytes),
            "sample" => slf.lookup_in_table(slf.py(), &slf.sample, &bytes),
            "sample_data" => slf.lookup_in_table(slf.py(), &slf.sample_data, &bytes),
            "ego_pose" => slf.lookup_in_table(slf.py(), &slf.ego_pose, &bytes),
            "instance" => slf.lookup_in_table(slf.py(), &slf.instance, &bytes),
            "sample_annotation" => slf.lookup_in_table(slf.py(), &slf.sample_ann, &bytes),
            "category" => slf.lookup_in_table(slf.py(), &slf.category, &bytes),
            "attribute" => slf.lookup_in_table(slf.py(), &slf.attribute, &bytes),
            "lidarseg" => slf
                .lidarseg
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("lidarseg not loaded due to missing 'lidarseg.json'"))
                .and_then(|t| slf.lookup_in_table(slf.py(), t, &bytes)),
            "panoptic" => slf
                .panoptic
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("panoptic is not loaded due to missing 'panoptic.json'"))
                .and_then(|t| slf.lookup_in_table(slf.py(), t, &bytes)),
            // Usually unreachable, checks should be done in Python.
            _ => Err(PyKeyError::new_err(format!("Table '{table}' not found"))),
        }
    }

    fn __getstate__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyAny>> {
        let state = PyDict::new(slf.py());
        state.set_item("version", slf.version.clone())?;
        state.set_item("dataroot", slf.dataroot.clone())?;
        state.into_bound_py_any(slf.py())
    }

    fn __setstate__(mut slf: PyRefMut<'_, Self>, state: Bound<'_, PyDict>) -> PyResult<()> {
        let version: String = state.get_item("version")?.unwrap().extract()?;
        let dataroot: String = state.get_item("dataroot")?.unwrap().extract()?;
        let new_instance = Self::new(&version, &dataroot)?;
        *slf = new_instance;
        Ok(())
    }

    fn __reduce__(slf: PyRef<'_, Self>) -> PyResult<(Bound<'_, PyAny>, (String, String))> {
        let cls = slf.py().get_type::<Tables>();
        let cls = cls.into_bound_py_any(slf.py())?;
        Ok((cls, (slf.version.clone(), slf.dataroot.clone())))
    }
}

impl Tables {
    fn lookup_in_table<'py, T: ToPyDict + AsRefToken>(
        &self, py: Python<'py>, table: &Table<T>, token: &[u8; 16],
    ) -> PyResult<Bound<'py, PyDict>> {
        table.get(token).map(|d| d.to_py_dict(py)).ok_or_else(|| PyKeyError::new_err(hex::encode(token)))?
    }
}
