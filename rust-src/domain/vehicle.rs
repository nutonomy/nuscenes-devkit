use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::ToPyDict;
use crate::common::{SensorChannel, SensorModality};
use crate::model::{CalibratedSensorModel, LogModel, MapModel, SensorModel};
use crate::table::AsRefToken;

#[derive(Clone, Debug)]
pub struct CalibratedSensor {
    pub token: [u8; 16],
    pub sensor_token: [u8; 16],

    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub camera_intrinsic: Option<[[f32; 3]; 3]>,
}

#[derive(Clone, Debug)]
pub struct Log<'a> {
    pub token: [u8; 16],

    pub logfile: Cow<'a, str>,
    pub vehicle: Cow<'a, str>,
    pub location: Cow<'a, str>,
    pub date_captured: Cow<'a, str>,

    pub map_token: [u8; 16], // From Map
}

#[derive(Clone, Debug)]
pub struct Map<'a> {
    pub token: [u8; 16],
    pub log_tokens: Box<[[u8; 16]]>,

    // [TODO]: Change to enums
    pub category: Cow<'a, str>,
    pub filename: Cow<'a, str>,
    // pub mask: MapMask,  // From where?
}

#[derive(Clone, Debug)]
pub struct Sensor {
    pub token: [u8; 16],

    pub channel: SensorChannel,
    pub modality: SensorModality,
}

impl<'a> Log<'a> {
    pub fn from_model(map_token: [u8; 16], model: LogModel<'a>) -> Self {
        Self {
            token: model.token,
            logfile: model.logfile,
            vehicle: model.vehicle,
            location: model.location,
            date_captured: model.date_captured,
            map_token,
        }
    }
}

impl From<CalibratedSensorModel> for CalibratedSensor {
    fn from(model: CalibratedSensorModel) -> Self {
        Self {
            token: model.token,
            sensor_token: model.sensor_token,
            translation: model.translation,
            rotation: model.rotation,
            camera_intrinsic: model.camera_intrinsic,
        }
    }
}

impl AsRefToken for CalibratedSensor {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl ToPyDict for CalibratedSensor {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("sensor_token", hex::encode(self.sensor_token))?;
        dict.set_item("translation", self.translation)?;
        dict.set_item("rotation", self.rotation)?;
        dict.set_item("camera_intrinsic", self.camera_intrinsic.map(|i| i.to_vec()).unwrap_or_default())?;

        Ok(dict)
    }
}

impl<'a> AsRefToken for Log<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Log<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("vehicle", self.vehicle.as_ref())?;
        dict.set_item("date_captured", self.date_captured.as_ref())?;
        dict.set_item("location", self.location.as_ref())?;
        dict.set_item("map_token", hex::encode(self.map_token))?;

        Ok(dict)
    }
}

impl<'a> From<MapModel<'a>> for Map<'a> {
    fn from(model: MapModel<'a>) -> Self {
        Self {
            token: model.token,
            log_tokens: model.log_tokens,
            category: model.category,
            filename: model.filename,
        }
    }
}

impl<'a> AsRefToken for Map<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Map<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("log_tokens", self.log_tokens.iter().map(hex::encode).collect::<Vec<_>>())?;
        dict.set_item("category", self.category.as_ref())?;
        dict.set_item("filename", self.filename.as_ref())?;

        Ok(dict)
    }
}

impl From<SensorModel> for Sensor {
    fn from(model: SensorModel) -> Self {
        Self { token: model.token, channel: model.channel, modality: model.modality }
    }
}

impl AsRefToken for Sensor {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl ToPyDict for Sensor {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("channel", self.channel.to_string())?;
        dict.set_item("modality", self.modality.to_string())?;

        Ok(dict)
    }
}
