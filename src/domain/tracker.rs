use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString};

use crate::domain::geometry::WorldPoint;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Display,
    EnumIter,
    EnumString,
    Serialize,
    Deserialize,
    JsonSchema,
)]
pub enum TrackerEngineKind {
    #[strum(to_string = "传统图像匹配")]
    RustTemplate,
    #[strum(to_string = "AI 图像识别")]
    CandleAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Serialize, Deserialize, JsonSchema)]
pub enum TrackerLifecycle {
    Idle,
    Running,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Serialize, Deserialize, JsonSchema)]
pub enum TrackingSource {
    ManualPreview,
    TemplateSearch,
    LocalTrack,
    GlobalRelocate,
    InertialHold,
    CandleEmbedding,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PositionEstimate {
    pub world: WorldPoint,
    pub found: bool,
    pub inertial: bool,
    pub heading_degrees: Option<f32>,
    pub source: TrackingSource,
    pub match_score: Option<f32>,
}

impl PositionEstimate {
    #[must_use]
    pub fn manual(world: WorldPoint) -> Self {
        Self {
            world,
            found: true,
            inertial: false,
            heading_degrees: None,
            source: TrackingSource::ManualPreview,
            match_score: None,
        }
    }

    #[must_use]
    pub fn tracked(
        world: WorldPoint,
        source: TrackingSource,
        match_score: Option<f32>,
        inertial: bool,
    ) -> Self {
        Self {
            world,
            found: true,
            inertial,
            heading_degrees: None,
            source,
            match_score,
        }
    }
}
