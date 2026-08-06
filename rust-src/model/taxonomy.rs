use std::borrow::Cow;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct AttributeModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],

    #[serde(borrow)]
    pub name: Cow<'a, str>,
    #[serde(borrow)]
    pub description: Cow<'a, str>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CategoryModel<'a> {
    #[serde(with = "hex::serde")]
    pub token: [u8; 16],

    #[serde(borrow)]
    pub name: Cow<'a, str>,
    #[serde(borrow)]
    pub description: Cow<'a, str>,

    pub index: Option<u32>, // Only in lidarseg
}
