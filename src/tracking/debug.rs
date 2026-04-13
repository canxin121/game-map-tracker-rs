use crate::domain::tracker::TrackerEngineKind;
use strum::Display;

#[derive(Debug, Clone)]
pub struct DebugField {
    pub label: String,
    pub value: String,
}

impl DebugField {
    #[must_use]
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugImage {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub format: DebugImageFormat,
    pub kind: DebugImageKind,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugImageFormat {
    Gray8,
    Rgba8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum DebugImageKind {
    #[strum(to_string = "Snapshot")]
    Snapshot,
    #[strum(to_string = "Heatmap")]
    Heatmap,
    #[strum(to_string = "Mask")]
    Mask,
}

impl DebugImage {
    #[must_use]
    pub fn new(label: impl Into<String>, width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            label: label.into(),
            width,
            height,
            format: DebugImageFormat::Gray8,
            kind: DebugImageKind::Snapshot,
            pixels,
        }
    }

    #[must_use]
    pub fn rgba(
        label: impl Into<String>,
        width: u32,
        height: u32,
        kind: DebugImageKind,
        pixels: Vec<u8>,
    ) -> Self {
        Self {
            label: label.into(),
            width,
            height,
            format: DebugImageFormat::Rgba8,
            kind,
            pixels,
        }
    }

    #[must_use]
    pub fn with_kind(mut self, kind: DebugImageKind) -> Self {
        self.kind = kind;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TrackingDebugSnapshot {
    pub engine: TrackerEngineKind,
    pub frame_index: u64,
    pub stage_label: String,
    pub images: Vec<DebugImage>,
    pub fields: Vec<DebugField>,
}
