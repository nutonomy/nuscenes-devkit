use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::ToPyDict;
use crate::model::{AttributeModel, CategoryModel};
use crate::table::AsRefToken;

#[derive(Clone, Debug)]
pub struct Attribute<'a> {
    pub token: [u8; 16],

    pub name: Cow<'a, str>,
    pub desc: Cow<'a, str>,
}

#[derive(Clone, Debug)]
pub struct Category<'a> {
    pub token: [u8; 16],

    pub name: Cow<'a, str>,
    pub desc: Cow<'a, str>,

    pub index: Option<u32>, // Only for lidarseg
}

impl<'a> From<AttributeModel<'a>> for Attribute<'a> {
    fn from(model: AttributeModel<'a>) -> Self {
        Self { token: model.token, name: model.name, desc: model.description }
    }
}

impl<'a> AsRefToken for Attribute<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Attribute<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("name", self.name.as_ref())?;
        dict.set_item("description", self.desc.as_ref())?;

        Ok(dict)
    }
}

impl<'a> From<CategoryModel<'a>> for Category<'a> {
    fn from(model: CategoryModel<'a>) -> Self {
        Self { token: model.token, name: model.name, desc: model.description, index: model.index }
    }
}

impl<'a> AsRefToken for Category<'a> {
    fn as_ref_token(&self) -> [u8; 16] {
        self.token
    }
}

impl<'a> ToPyDict for Category<'a> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("token", hex::encode(self.token))?;
        dict.set_item("name", self.name.as_ref())?;
        dict.set_item("description", self.desc.as_ref())?;
        if let Some(index) = self.index {
            dict.set_item("index", index)?;
        }

        Ok(dict)
    }
}
