//! Per-conversion cancellation and recoverable errors for Rust callers.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum Error {
    Cancelled,
    InvalidData(&'static str),
    Io(std::io::Error),
    Allocation,
    Gpu(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => f.write_str("conversion cancelled"),
            Self::InvalidData(message) => write!(f, "invalid X3F data: {message}"),
            Self::Io(error) => error.fmt(f),
            Self::Allocation => {
                f.write_str("X3F allocation exceeds available memory or safety limit")
            }
            Self::Gpu(message) => write!(f, "GPU denoising unavailable: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Default)]
pub struct Control<'a> {
    cancel: Option<&'a AtomicBool>,
    window: Option<(u64, u64)>,
    pub(crate) denoise_backend: crate::DenoiseBackend,
    pub(crate) denoise_options: crate::DenoiseOptions,
}

impl<'a> Control<'a> {
    pub const fn new(cancel: &'a AtomicBool) -> Self {
        Self {
            cancel: Some(cancel),
            window: None,
            denoise_backend: crate::DenoiseBackend::Auto,
            denoise_options: crate::DenoiseOptions::DEFAULT,
        }
    }
    pub const fn none() -> Self {
        Self {
            cancel: None,
            window: None,
            denoise_backend: crate::DenoiseBackend::Auto,
            denoise_options: crate::DenoiseOptions::DEFAULT,
        }
    }
    /// Choose a denoising backend without changing another conversion's state.
    pub const fn with_denoise_backend(mut self, backend: crate::DenoiseBackend) -> Self {
        self.denoise_backend = backend;
        self
    }
    /// Restrict input reads to a frame at `base` with the given byte length.
    pub const fn with_window(mut self, base: u64, length: u64) -> Self {
        self.window = Some((base, length));
        self
    }

    pub(crate) const fn window(self) -> Option<(u64, u64)> {
        self.window
    }

    pub fn check(self) -> Result<()> {
        if self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn with_denoise_options(mut self, options: crate::DenoiseOptions) -> Result<Self> {
        options.validate().map_err(Error::InvalidData)?;
        self.denoise_options = options;
        Ok(self)
    }
}
