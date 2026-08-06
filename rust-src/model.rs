use serde::{Deserialize, Deserializer};
use serde_with::DeserializeAs;

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

#[inline]
fn decode_hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[inline]
fn parse_hex(s: &str) -> Option<[u8; 16]> {
    let bytes = s.as_bytes();
    let mut out = [0u8; 16];
    for i in 0..16 {
        let hi = decode_hex_nibble(bytes[2 * i])?;
        let lo = decode_hex_nibble(bytes[2 * i + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn deserialize_token<'de, D>(deserializer: D) -> Result<Option<[u8; 16]>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<&str>::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some("") => Ok(None),
        Some(s) if s.len() != 32 => Err(serde::de::Error::custom("Invalid token length")),
        Some(s) => Ok(parse_hex(s)),
    }
}

pub struct EmptyMatrix3AsNone;

impl<'de> DeserializeAs<'de, Option<[[f32; 3]; 3]>> for EmptyMatrix3AsNone {
    fn deserialize_as<D>(deserializer: D) -> Result<Option<[[f32; 3]; 3]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mat = Vec::<[f32; 3]>::deserialize(deserializer)?;
        if mat.is_empty() {
            return Ok(None);
        }

        let arr: [[f32; 3]; 3] = mat.try_into().map_err(|_| serde::de::Error::custom("Expected a 3x3 matrix"))?;
        Ok(Some(arr))
    }
}
