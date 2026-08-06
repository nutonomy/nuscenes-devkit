use pyo3::prelude::*;
use pyo3::types::PyDict;

mod annotation;
mod extensions;
mod extraction;
mod taxonomy;
mod vehicle;

pub use annotation::*;
pub use extensions::*;
pub use extraction::*;
pub use taxonomy::*;
pub use vehicle::*;

pub trait ToPyDict {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;
}
