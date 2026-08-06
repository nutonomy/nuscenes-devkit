use serde::Deserialize;
use serde_with::hex::Hex;
use serde_with::serde_as;

use super::deserialize_token;
use crate::common::Visibility;

#[derive(Clone, Debug, Deserialize)]
pub struct InstanceModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub category_token: [u8; 16],

    pub nbr_annotations: u32,
    #[serde(with = "hex::serde")]
    pub first_annotation_token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub last_annotation_token: [u8; 16],
}

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct SampleAnnotationModel {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub sample_token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub instance_token: [u8; 16],

    #[serde_as(as = "Box<[Hex]>")]
    pub attribute_tokens: Box<[[u8; 16]]>,

    #[serde(deserialize_with = "deserialize_token")]
    pub prev: Option<[u8; 16]>,
    #[serde(deserialize_with = "deserialize_token")]
    pub next: Option<[u8; 16]>,

    #[serde(rename = "visibility_token")]
    pub visibility: Visibility,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub size: [f32; 3],

    pub num_lidar_pts: u32,
    pub num_radar_pts: u32,
}
