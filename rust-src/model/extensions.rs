use std::borrow::Cow;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct LidarSegModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub sample_data_token: [u8; 16],

    #[serde(borrow)]
    pub filename: Cow<'a, str>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PanopticModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],
    #[serde(with = "hex::serde")]
    pub sample_data_token: [u8; 16],

    #[serde(borrow)]
    pub filename: Cow<'a, str>,
}
