use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::ToPyDict;
use crate::common::Visibility;
use crate::model::{InstanceModel, SampleAnnotationModel};
use crate::table::AsRefToken;

#[derive(Clone, Debug)]
pub struct Instance {
    pub token: [u8; 16],
    pub category_token: [u8; 16],

    pub nbr_annotations: u32,
    pub first_annotation_token: [u8; 16],
    pub last_annotation_token: [u8; 16],
}

#[derive(Clone, Debug)]
pub struct SampleAnnotation<'a> {
    pub token: [u8; 16],
    pub sample_token: [u8; 16],
    pub instance_token: [u8; 16],
    pub attribute_tokens: Box<[[u8; 16]]>,

    pub prev: Option<[u8; 16]>,
    pub next: Option<[u8; 16]>,

    pub visibility: Visibility,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub size: [f32; 3],

    pub num_lidar_pts: u32,
    pub num_radar_pts: u32,

    pub category_name: Cow<'a, str>, // From instance -> category
}

impl<'a> SampleAnnotation<'a> {
    pub fn from_model(category_name: Cow<'a, str>, model: SampleAnnotationModel) -> Self {
        Self {
            token: model.token,
            sample_token: model.sample_token,
            instance_token: model.instance_token,
            attribute_tokens: model.attribute_tokens,
            prev: model.prev,
            next: model.next,
            visibility: model.visibility,
            translation: model.translation,
            rotation: model.rotation,
            size: model.size,
            num_lidar_pts: model.num_lidar_pts,
            num_radar_pts: model.num_radar_pts,
            category_name,
        }
    }
}

impl From<InstanceModel> for Instance {
    fn from(model: InstanceModel) -> Self {
        Self {
            token: model.token,
            category_token: model.category_token,
            nbr_annotations: model.nbr_annotations,
            first_annotation_token: model.first_annotation_token,
            last_annotation_token: model.last_annotation_token,
        }
    }
}

impl AsRefToken for Instance {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl ToPyDict for Instance {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("category_token", hex::encode(self.category_token))?;
        dict.set_item("nbr_annotations", self.nbr_annotations)?;
        dict.set_item("first_annotation_token", hex::encode(self.first_annotation_token))?;
        dict.set_item("last_annotation_token", hex::encode(self.last_annotation_token))?;

        Ok(dict)
    }
}

impl<'a> AsRefToken for SampleAnnotation<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for SampleAnnotation<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("sample_token", hex::encode(self.sample_token))?;
        dict.set_item("instance_token", hex::encode(self.instance_token))?;
        dict.set_item("attribute_tokens", self.attribute_tokens.iter().map(hex::encode).collect::<Vec<_>>())?;
        dict.set_item("prev", self.prev.map(hex::encode).unwrap_or_default())?;
        dict.set_item("next", self.next.map(hex::encode).unwrap_or_default())?;
        dict.set_item("visibility", self.visibility.to_string())?;
        dict.set_item("translation", self.translation)?;
        dict.set_item("rotation", self.rotation)?;
        dict.set_item("size", self.size)?;
        dict.set_item("num_lidar_pts", self.num_lidar_pts)?;
        dict.set_item("num_radar_pts", self.num_radar_pts)?;
        dict.set_item("category_name", self.category_name.as_ref())?;

        Ok(dict)
    }
}
