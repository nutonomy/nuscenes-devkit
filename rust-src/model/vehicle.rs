use std::borrow::Cow;

use serde::Deserialize;
use serde_with::hex::Hex;
use serde_with::serde_as;

use super::EmptyMatrix3AsNone;
use crate::common::{SensorChannel, SensorModality};

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct CalibratedSensorModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub sensor_token: [u8; 16],

    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    #[serde_as(as = "EmptyMatrix3AsNone")]
    pub camera_intrinsic: Option<[[f32; 3]; 3]>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LogModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],

    #[serde(borrow)]
    pub logfile: Cow<'a, str>,
    #[serde(borrow)]
    pub vehicle: Cow<'a, str>,
    #[serde(borrow)]
    pub date_captured: Cow<'a, str>,
    #[serde(borrow)]
    pub location: Cow<'a, str>,

    #[serde(skip)]
    pub map_token: [u8; 16], // Reverse index from Map
}

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct MapModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde_as(as = "Box<[Hex]>")]
    pub log_tokens: Box<[[u8; 16]]>,

    #[serde(borrow)]
    pub category: Cow<'a, str>,
    #[serde(borrow)]
    pub filename: Cow<'a, str>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SensorModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],

    pub channel: SensorChannel,
    pub modality: SensorModality,
}
