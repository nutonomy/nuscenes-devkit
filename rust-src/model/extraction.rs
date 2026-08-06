use std::borrow::Cow;

use serde::Deserialize;

use super::deserialize_token;

#[derive(Clone, Debug, Deserialize)]
pub struct EgoPoseModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],

    pub timestamp: u64,

    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Clone, Debug, Deserialize)]
pub struct SampleModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub scene_token: [u8; 16],

    #[serde(deserialize_with = "deserialize_token")]
    pub prev: Option<[u8; 16]>,
    #[serde(deserialize_with = "deserialize_token")]
    pub next: Option<[u8; 16]>,

    pub timestamp: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SampleDataModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub sample_token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub ego_pose_token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub calibrated_sensor_token: [u8; 16],

    #[serde(deserialize_with = "deserialize_token")]
    pub prev: Option<[u8; 16]>,
    #[serde(deserialize_with = "deserialize_token")]
    pub next: Option<[u8; 16]>,

    #[serde(borrow)]
    pub fileformat: Cow<'a, str>,
    #[serde(borrow)]
    pub filename: Cow<'a, str>,

    pub timestamp: u64,
    pub is_key_frame: bool,
    pub height: u16,
    pub width: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SceneModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub log_token: [u8; 16],

    #[serde(borrow)]
    pub name: Cow<'a, str>,
    #[serde(borrow)]
    pub description: Cow<'a, str>,

    pub nbr_samples: u16,
    #[serde(with = "hex::serde")]
    pub first_sample_token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub last_sample_token: [u8; 16],
}
