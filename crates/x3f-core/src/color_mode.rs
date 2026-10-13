//! Select a camera color mode when building a DNG profile from X3F metadata.

use crate::Error;
use std::{fmt, str::FromStr};

/// Choose the camera's color and tone settings without changing the raw samples.
/// Use with `ProcessOptions::dng_color_mode`. The X3F must contain calibration for
/// the selected mode, so not every file supports every variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Standard color and tone.
    Standard,
    /// Neutral color and reduced contrast.
    Neutral,
    /// Vivid color and increased contrast.
    Vivid,
    /// Portrait color and tone.
    Portrait,
    /// Landscape color and tone.
    Landscape,
    /// Foveon Classic Blue color and tone.
    FcBlue,
}

impl ColorMode {
    /// Return the exact preset name used by the X3F metadata and CLI selector.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Neutral => "Neutral",
            Self::Vivid => "Vivid",
            Self::Portrait => "Portrait",
            Self::Landscape => "Landscape",
            Self::FcBlue => "FCBlue",
        }
    }
}

impl fmt::Display for ColorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ColorMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "Standard" => Ok(Self::Standard),
            "Neutral" => Ok(Self::Neutral),
            "Vivid" => Ok(Self::Vivid),
            "Portrait" => Ok(Self::Portrait),
            "Landscape" => Ok(Self::Landscape),
            "FCBlue" => Ok(Self::FcBlue),
            _ => Err(Error::InvalidData(format!(
                "unsupported color mode: {value}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_names_round_trip() {
        for mode in [
            ColorMode::Standard,
            ColorMode::Neutral,
            ColorMode::Vivid,
            ColorMode::Portrait,
            ColorMode::Landscape,
            ColorMode::FcBlue,
        ] {
            assert_eq!(mode.as_str().parse::<ColorMode>().unwrap(), mode);
            assert_eq!(mode.to_string(), mode.as_str());
        }
        for name in [
            "",
            "standard",
            "Auto",
            "Monochrome",
            "Fov Classic Blue",
            "unknown",
        ] {
            assert!(name.parse::<ColorMode>().is_err());
        }
        assert!(crate::ProcessOptions::default().dng_color_mode.is_none());
    }
}
