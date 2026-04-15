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
    #[strum(
        to_string = "多尺度模板匹配",
        serialize = "MultiScaleTemplateMatch",
        serialize = "RustTemplate"
    )]
    #[serde(rename = "multi_scale_template_match", alias = "RustTemplate")]
    MultiScaleTemplateMatch,
    #[strum(
        to_string = "卷积特征匹配",
        serialize = "ConvolutionFeatureMatch",
        serialize = "CandleAi"
    )]
    #[serde(rename = "convolution_feature_match", alias = "CandleAi")]
    ConvolutionFeatureMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Serialize, Deserialize, JsonSchema)]
pub enum TrackerLifecycle {
    Idle,
    Running,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Serialize, Deserialize, JsonSchema)]
pub enum TrackingSource {
    #[strum(to_string = "手动预览")]
    ManualPreview,
    #[strum(to_string = "模板搜索")]
    TemplateSearch,
    #[strum(to_string = "局部锁定")]
    LocalTrack,
    #[strum(to_string = "全局重定位")]
    GlobalRelocate,
    #[strum(to_string = "惯性保位")]
    InertialHold,
    #[strum(to_string = "卷积特征匹配")]
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
