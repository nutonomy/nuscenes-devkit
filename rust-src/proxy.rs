use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PySlice};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::domain::*;

macro_rules! define_view {
    ($view_name:ident, $iter_name:ident, $model_type:ty) => {
        #[pyclass(sequence)]
        pub struct $view_name {
            pub data: Arc<Box<[$model_type]>>,
        }

        #[pymethods]
        impl $view_name {
            fn __len__(&self) -> usize {
                self.data.len()
            }

            fn __getitem__<'py>(slf: PyRef<'py, Self>, index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                if let Ok(slice) = index.cast::<PySlice>() {
                    let indices = slice.indices(slf.data.len() as isize)?;
                    (indices.start..indices.stop)
                        .step_by(indices.step as usize)
                        .map(|i| slf.data[i as usize].to_py_dict(slf.py()))
                        .collect::<PyResult<Vec<_>>>()?
                        .into_bound_py_any(slf.py())
                } else if let Ok(index) = index.extract::<usize>() {
                    slf.data
                        .get(index)
                        .ok_or_else(|| PyIndexError::new_err("Index out of range"))
                        .and_then(|d| d.to_py_dict(slf.py()))
                        .and_then(|d| d.into_bound_py_any(slf.py()))
                } else {
                    Err(PyTypeError::new_err("Table indices must be slice or integers"))
                }
            }

            fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<$iter_name>> {
                let iter = $iter_name { data: slf.data.clone(), index: 0 };
                Py::new(slf.py(), iter)
            }
        }

        #[pyclass]
        pub struct $iter_name {
            pub data: Arc<Box<[$model_type]>>,
            pub index: usize,
        }

        #[pymethods]
        impl $iter_name {
            pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            pub fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Bound<'_, PyDict>> {
                if slf.index >= slf.data.len() {
                    return None;
                }

                let item = slf.data[slf.index].to_py_dict(slf.py()).ok();
                slf.index += 1;
                item
            }
        }
    };
}

// Annotation
define_view!(InstanceView, InstanceIter, Instance);
define_view!(SampleAnnotationView, SampleAnnotationIter, SampleAnnotation<'static>);

// Extraction
define_view!(EgoPoseView, EgoPoseIter, EgoPose);
define_view!(SampleView, SampleIter, Sample);
define_view!(SampleDataView, SampleDataIter, SampleData<'static>);
define_view!(SceneView, SceneIter, Scene<'static>);

// Taxonomy
define_view!(AttributeView, AttributeIter, Attribute<'static>);
define_view!(CategoryView, CategoryIter, Category<'static>);

// Vehicle
define_view!(CalibratedSensorView, CalibratedSensorIter, CalibratedSensor);
define_view!(LogView, LogIter, Log<'static>);
define_view!(MapView, MapIter, Map<'static>);
define_view!(SensorView, SensorIter, Sensor);

// Extension
define_view!(LidarSegView, LidarSegIter, LidarSeg<'static>);
define_view!(PanopticView, PanopticIter, Panoptic<'static>);

// #[pyclass(sequence)]
// pub struct EgoPoseView {
//     pub data: Arc<Box<[EgoPose]>>,
// }

// #[pymethods]
// impl EgoPoseView {
//     fn __getitem__<'py>(slf: PyRef<'py, Self>, index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
//         if let Ok(slice) = index.cast::<PySlice>() {
//             let indices = slice.indices(slf.data.len() as isize)?;
//             (indices.start..indices.stop)
//                 .step_by(indices.step as usize)
//                 .map(|i| slf.data[i as usize].to_py_dict(slf.py()))
//                 .collect::<PyResult<Vec<_>>>()?
//                 .into_bound_py_any(slf.py())
//         } else if let Ok(index) = index.extract::<usize>() {
//             slf.data
//                 .get(index)
//                 .ok_or_else(|| PyIndexError::new_err("Index out of range"))
//                 .and_then(|d| d.to_py_dict(slf.py()))
//                 .and_then(|d| d.into_bound_py_any(slf.py()))
//         } else {
//             Err(PyTypeError::new_err("Table indices must be slice or integers"))
//         }
//     }
// }
