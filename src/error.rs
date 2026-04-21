use std::{
    error::Error as StdError,
    num::{ParseFloatError, ParseIntError, TryFromIntError},
    result::Result as StdResult,
    str::Utf8Error,
    time::SystemTimeError,
};

use thiserror::Error;

pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;
pub type Result<T, E = AppError> = StdResult<T, E>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Message(String),
    #[error("{context}")]
    Context {
        context: String,
        #[source]
        source: BoxError,
    },
    #[error("failed to resolve project root from manifest directory: {0}")]
    ProjectRootResolution(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    EnvVar(#[from] std::env::VarError),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    TomlDeserialize(#[from] toml::de::Error),
    #[error(transparent)]
    TomlSerialize(#[from] toml::ser::Error),
    #[error(transparent)]
    Regex(#[from] regex::Error),
    #[error(transparent)]
    Ureq(#[from] ureq::Error),
    #[error(transparent)]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error(transparent)]
    ParseInt(#[from] ParseIntError),
    #[error(transparent)]
    ParseFloat(#[from] ParseFloatError),
    #[error(transparent)]
    TryFromInt(#[from] TryFromIntError),
    #[error(transparent)]
    Utf8(#[from] Utf8Error),
    #[error(transparent)]
    SystemTime(#[from] SystemTimeError),
    #[error(transparent)]
    RawWindowHandle(#[from] raw_window_handle::HandleError),
}

impl AppError {
    #[must_use]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    #[must_use]
    pub fn context(
        context: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Context {
            context: context.into(),
            source: Box::new(source),
        }
    }
}

pub trait ContextExt<T> {
    fn context(self, context: impl Into<String>) -> Result<T>;
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: Into<String>,
        F: FnOnce() -> C;
}

impl<T, E> ContextExt<T> for StdResult<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|source| AppError::context(context, source))
    }

    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: Into<String>,
        F: FnOnce() -> C,
    {
        self.map_err(|source| AppError::context(f(), source))
    }
}

impl<T> ContextExt<T> for Option<T> {
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| AppError::message(context))
    }

    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: Into<String>,
        F: FnOnce() -> C,
    {
        self.ok_or_else(|| AppError::message(f()))
    }
}

impl From<String> for AppError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl From<&str> for AppError {
    fn from(message: &str) -> Self {
        Self::Message(message.to_owned())
    }
}

impl From<BoxError> for AppError {
    fn from(source: BoxError) -> Self {
        let message = source.to_string();
        Self::Context {
            context: message,
            source,
        }
    }
}

#[macro_export]
macro_rules! app_error {
    ($format:literal, $($arg:tt)+) => {
        $crate::error::AppError::message(format!($format, $($arg)+))
    };
    ($message:literal $(,)?) => {
        $crate::error::AppError::message(format!($message))
    };
    ($message:expr $(,)?) => {
        $crate::error::AppError::message($message)
    };
}

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {{
        return Err($crate::app_error!($($arg)*))
    }};
}

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $($arg:tt)*) => {{
        if !$condition {
            $crate::bail!($($arg)*);
        }
    }};
}
