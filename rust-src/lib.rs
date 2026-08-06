mod common;
mod domain;
mod model;
mod nusc;
mod proxy;
mod table;

/// A module for loading and querying nuScenes tables implemented in Rust
#[pyo3::pymodule]
mod _lib {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::nusc::Tables;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }
}
