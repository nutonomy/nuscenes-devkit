use std::fmt::Display;

use enum_map::Enum;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, Enum, Hash, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SensorChannel {
    CamBack,
    CamBackLeft,
    CamBackRight,
    CamFront,
    CamFrontLeft,
    CamFrontRight,
    LidarTop,
    RadarBackLeft,
    RadarBackRight,
    RadarFront,
    RadarFrontLeft,
    RadarFrontRight,
}

#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SensorModality {
    Camera,
    Lidar,
    Radar,
}

impl Display for SensorModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Camera => "camera",
            Self::Lidar => "lidar",
            Self::Radar => "radar",
        };
        write!(f, "{name}")
    }
}

impl Display for SensorChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::CamBack => "CAM_BACK",
            Self::CamBackLeft => "CAM_BACK_LEFT",
            Self::CamBackRight => "CAM_BACK_RIGHT",
            Self::CamFront => "CAM_FRONT",
            Self::CamFrontLeft => "CAM_FRONT_LEFT",
            Self::CamFrontRight => "CAM_FRONT_RIGHT",
            Self::LidarTop => "LIDAR_TOP",
            Self::RadarBackLeft => "RADAR_BACK_LEFT",
            Self::RadarBackRight => "RADAR_BACK_RIGHT",
            Self::RadarFront => "RADAR_FRONT",
            Self::RadarFrontLeft => "RADAR_FRONT_LEFT",
            Self::RadarFrontRight => "RADAR_FRONT_RIGHT",
        };
        write!(f, "{name}")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq)]
pub enum Visibility {
    #[serde(rename = "1")]
    V0_40,
    #[serde(rename = "2")]
    V40_60,
    #[serde(rename = "3")]
    V60_80,
    #[serde(rename = "4")]
    V80_100,
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::V0_40 => "v0-40",
            Self::V40_60 => "v40-60",
            Self::V60_80 => "v60-80",
            Self::V80_100 => "v80-100",
        };
        write!(f, "{name}")
    }
}
