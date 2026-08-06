use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::domain::ToPyDict;
use crate::model::{LidarSegModel, PanopticModel};
use crate::table::AsRefToken;

#[derive(Clone, Debug)]
pub struct LidarSeg<'a> {
    pub token: [u8; 16],
    pub sample_data_token: [u8; 16],

    pub filename: Cow<'a, str>,
}

#[derive(Clone, Debug)]
pub struct Panoptic<'a> {
    pub token: [u8; 16],
    pub sample_data_token: [u8; 16],

    pub filename: Cow<'a, str>,
}

impl<'a> From<LidarSegModel<'a>> for LidarSeg<'a> {
    fn from(model: LidarSegModel<'a>) -> Self {
        Self { token: model.token, sample_data_token: model.sample_data_token, filename: model.filename }
    }
}

impl<'a> AsRefToken for LidarSeg<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for LidarSeg<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("sample_data_token", hex::encode(self.sample_data_token))?;
        dict.set_item("filename", self.filename.as_ref())?;

        Ok(dict)
    }
}

impl<'a> From<PanopticModel<'a>> for Panoptic<'a> {
    fn from(model: PanopticModel<'a>) -> Self {
        Self { token: model.token, sample_data_token: model.sample_data_token, filename: model.filename }
    }
}

impl<'a> AsRefToken for Panoptic<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Panoptic<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("sample_data_token", hex::encode(self.sample_data_token))?;
        dict.set_item("filename", self.filename.as_ref())?;

        Ok(dict)
    }
}
