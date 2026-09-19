//! Per-conversion cancellation and recoverable errors for Rust callers.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum Error {
    Cancelled,
    InvalidData(&'static str),
    Io(std::io::Error),
    Allocation,
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
}

impl<'a> Control<'a> {
    pub const fn new(cancel: &'a AtomicBool) -> Self {
        Self {
            cancel: Some(cancel),
        }
    }
    pub const fn none() -> Self {
        Self { cancel: None }
    }
    pub fn check(self) -> Result<()> {
        if self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
