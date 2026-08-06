use std::borrow::Cow;
use std::collections::HashMap;

use enum_map::EnumMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::ToPyDict;
use crate::common::{SensorChannel, SensorModality};
use crate::model::{EgoPoseModel, SampleDataModel, SampleModel, SceneModel};
use crate::table::AsRefToken;

#[derive(Clone, Debug)]
pub struct EgoPose {
    pub token: [u8; 16],

    pub timestamp: u64,

    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub token: [u8; 16],
    pub scene_token: [u8; 16],

    pub prev: Option<[u8; 16]>,
    pub next: Option<[u8; 16]>,

    pub timestamp: u64,

    pub data: EnumMap<SensorChannel, [u8; 16]>, // From sample data
    pub anns: Box<[[u8; 16]]>,                  // From sample annotations
}

#[derive(Clone, Debug)]
pub struct SampleData<'a> {
    pub token: [u8; 16],
    pub sample_token: [u8; 16],
    pub ego_pose_token: [u8; 16],
    pub calibrated_sensor_token: [u8; 16],

    pub prev: Option<[u8; 16]>,
    pub next: Option<[u8; 16]>,

    pub fileformat: Cow<'a, str>,
    pub filename: Cow<'a, str>,

    pub timestamp: u64,
    pub is_key_frame: bool,
    pub height: u16,
    pub width: u16,

    pub modality: SensorModality, // From calibrated sensor -> sensor
    pub channel: SensorChannel,   // From calibrated sensor -> sensor
}

#[derive(Clone, Debug)]
pub struct Scene<'a> {
    pub token: [u8; 16],
    pub log_token: [u8; 16],

    pub name: Cow<'a, str>,
    pub desc: Cow<'a, str>,

    pub nbr_samples: u16,
    pub first_sample_token: [u8; 16],
    pub last_sample_token: [u8; 16],
}

impl<'a> SampleData<'a> {
    pub fn from_model(modality: SensorModality, channel: SensorChannel, model: SampleDataModel<'a>) -> Self {
        Self {
            token: model.token,
            sample_token: model.sample_token,
            ego_pose_token: model.ego_pose_token,
            calibrated_sensor_token: model.calibrated_sensor_token,

            prev: model.prev,
            next: model.next,

            fileformat: model.fileformat,
            filename: model.filename,

            timestamp: model.timestamp,
            is_key_frame: model.is_key_frame,
            height: model.height,
            width: model.width,

            modality,
            channel,
        }
    }
}

impl Sample {
    pub fn from_model(data: EnumMap<SensorChannel, [u8; 16]>, anns: Box<[[u8; 16]]>, model: SampleModel) -> Self {
        Self {
            token: model.token,
            scene_token: model.scene_token,
            prev: model.prev,
            next: model.next,
            timestamp: model.timestamp,
            data,
            anns,
        }
    }
}

impl From<EgoPoseModel> for EgoPose {
    fn from(model: EgoPoseModel) -> Self {
        Self {
            token: model.token,
            timestamp: model.timestamp,
            translation: model.translation,
            rotation: model.rotation,
        }
    }
}

impl AsRefToken for EgoPose {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl ToPyDict for EgoPose {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("translation", self.translation)?;
        dict.set_item("rotation", self.rotation)?;

        Ok(dict)
    }
}

impl AsRefToken for Sample {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl ToPyDict for Sample {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let data: HashMap<_, _> = self.data.iter().map(|(key, val)| (key.to_string(), hex::encode(val))).collect();
        let anns: Vec<_> = self.anns.iter().map(hex::encode).collect();

        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("scene_token", hex::encode(self.scene_token))?;
        dict.set_item("prev", self.prev.map(hex::encode).unwrap_or_default())?;
        dict.set_item("next", self.next.map(hex::encode).unwrap_or_default())?;
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("data", data)?;
        dict.set_item("anns", anns)?;

        Ok(dict)
    }
}

impl<'a> AsRefToken for SampleData<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for SampleData<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("sample_token", hex::encode(self.sample_token))?;
        dict.set_item("ego_pose_token", hex::encode(self.ego_pose_token))?;
        dict.set_item("calibrated_sensor_token", hex::encode(self.calibrated_sensor_token))?;
        dict.set_item("prev", self.prev.map(hex::encode).unwrap_or_default())?;
        dict.set_item("next", self.next.map(hex::encode).unwrap_or_default())?;
        dict.set_item("fileformat", self.fileformat.as_ref())?;
        dict.set_item("filename", self.filename.as_ref())?;
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("is_key_frame", self.is_key_frame)?;
        dict.set_item("height", self.height)?;
        dict.set_item("width", self.width)?;
        dict.set_item("modality", self.modality.to_string())?;
        dict.set_item("channel", self.channel.to_string())?;

        Ok(dict)
    }
}

impl<'a> From<SceneModel<'a>> for Scene<'a> {
    fn from(model: SceneModel<'a>) -> Self {
        Self {
            token: model.token,
            log_token: model.log_token,
            name: model.name,
            desc: model.description,
            nbr_samples: model.nbr_samples,
            first_sample_token: model.first_sample_token,
            last_sample_token: model.last_sample_token,
        }
    }
}

impl<'a> AsRefToken for Scene<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Scene<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("log_token", hex::encode(self.log_token))?;
        dict.set_item("nbr_samples", self.nbr_samples)?;
        dict.set_item("first_sample_token", hex::encode(self.first_sample_token))?;
        dict.set_item("last_sample_token", hex::encode(self.last_sample_token))?;
        dict.set_item("name", self.name.as_ref())?;
        dict.set_item("description", self.desc.as_ref())?;

        Ok(dict)
    }
}
