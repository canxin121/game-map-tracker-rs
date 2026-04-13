use derive_more::{Display, From};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::{
    geometry::{WorldPoint, WorldRect},
    marker::MarkerStyle,
};

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Display, From, Default, Serialize, Deserialize, JsonSchema,
)]
pub struct RouteId(pub String);

impl RouteId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Display, From, Default, Serialize, Deserialize, JsonSchema,
)]
pub struct RoutePointId(pub String);

impl RoutePointId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RoutePoint {
    #[serde(default = "RoutePointId::new")]
    pub id: RoutePointId,
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub style: MarkerStyle,
}

impl RoutePoint {
    #[must_use]
    pub fn new(label: impl Into<String>, world: WorldPoint) -> Self {
        Self {
            id: RoutePointId::new(),
            x: world.x,
            y: world.y,
            label: Some(label.into()),
            note: String::new(),
            radius: None,
            style: MarkerStyle::default(),
        }
    }

    #[must_use]
    pub fn world(&self) -> WorldPoint {
        WorldPoint::new(self.x, self.y)
    }

    #[must_use]
    pub fn display_label(&self) -> &str {
        self.label
            .as_deref()
            .filter(|label| !label.trim().is_empty())
            .unwrap_or("未命名节点")
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.style = self.style.normalized();
        self.radius = self.radius.map(|radius| radius.max(0.0));
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RouteMetadata {
    pub id: RouteId,
    pub file_name: String,
    pub display_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RouteDocument {
    #[serde(default = "RouteId::new")]
    pub id: RouteId,
    #[serde(default)]
    pub name: String,
    #[serde(rename = "loop", default)]
    pub looped: bool,
    #[serde(default, alias = "description")]
    pub notes: String,
    #[serde(default = "default_group_visible")]
    pub visible: bool,
    #[serde(default)]
    pub default_style: MarkerStyle,
    #[serde(default)]
    pub points: Vec<RoutePoint>,
    #[serde(skip)]
    pub metadata: RouteMetadata,
}

impl RouteDocument {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: RouteId::new(),
            name: name.into(),
            looped: false,
            notes: String::new(),
            visible: true,
            default_style: MarkerStyle::default(),
            points: Vec::new(),
            metadata: RouteMetadata::default(),
        }
    }

    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    #[must_use]
    pub fn display_name(&self) -> &str {
        if !self.name.trim().is_empty() {
            &self.name
        } else if !self.metadata.display_name.trim().is_empty() {
            &self.metadata.display_name
        } else {
            "未命名标记组"
        }
    }

    #[must_use]
    pub fn display_notes(&self) -> &str {
        self.notes.trim()
    }

    #[must_use]
    pub fn bounds(&self) -> Option<WorldRect> {
        let mut points = self.points.iter();
        let first = points.next()?;
        let mut rect = WorldRect::from_point(first.world());
        for point in points {
            rect.include(point.world());
        }
        Some(rect)
    }

    #[must_use]
    pub fn first_point(&self) -> Option<WorldPoint> {
        self.points.first().map(RoutePoint::world)
    }

    #[must_use]
    pub fn effective_style(&self, point: &RoutePoint) -> MarkerStyle {
        let mut style = self.default_style.clone().normalized();
        let point_style = point.style.clone().normalized();
        style.icon = point_style.icon;
        if !point_style.color_hex.trim().is_empty() {
            style.color_hex = point_style.color_hex;
        }
        style.size_px = point_style.size_px;
        style
    }

    #[must_use]
    pub fn find_point(&self, point_id: &RoutePointId) -> Option<&RoutePoint> {
        self.points.iter().find(|point| &point.id == point_id)
    }

    pub fn find_point_mut(&mut self, point_id: &RoutePointId) -> Option<&mut RoutePoint> {
        self.points.iter_mut().find(|point| &point.id == point_id)
    }

    pub fn remove_point(&mut self, point_id: &RoutePointId) -> Option<RoutePoint> {
        let index = self.points.iter().position(|point| &point.id == point_id)?;
        Some(self.points.remove(index))
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.visible = self.visible || self.points.is_empty();
        self.default_style = self.default_style.normalized();
        self.points = self
            .points
            .into_iter()
            .map(RoutePoint::normalized)
            .collect();
        self
    }
}

const fn default_group_visible() -> bool {
    true
}
