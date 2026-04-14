use derive_more::{Display, From};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use crate::domain::geometry::WorldPoint;

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Display, From, Default, Serialize, Deserialize, JsonSchema,
)]
pub struct MarkerId(pub String);

impl MarkerId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Display, From, Default, Serialize, Deserialize, JsonSchema,
)]
pub struct MarkerGroupId(pub String);

impl MarkerGroupId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, From, Serialize, JsonSchema)]
#[display("{_0}")]
#[serde(transparent)]
pub struct MarkerIconStyle(pub String);

impl MarkerIconStyle {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(normalize_marker_icon_name(&name.into()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn normalized(&self) -> Self {
        Self::new(self.0.clone())
    }
}

impl Default for MarkerIconStyle {
    fn default() -> Self {
        Self::new(default_marker_icon_name())
    }
}

impl From<&str> for MarkerIconStyle {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for MarkerIconStyle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::new(raw))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MarkerStyle {
    #[serde(default)]
    pub icon: MarkerIconStyle,
    #[serde(default = "default_marker_color")]
    pub color_hex: String,
    #[serde(default = "default_marker_size")]
    pub size_px: f32,
}

impl Default for MarkerStyle {
    fn default() -> Self {
        Self {
            icon: MarkerIconStyle::default(),
            color_hex: default_marker_color(),
            size_px: default_marker_size(),
        }
    }
}

impl MarkerStyle {
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.icon = self.icon.normalized();
        self.color_hex = normalize_hex_color(&self.color_hex);
        self.size_px = self.size_px.clamp(14.0, 64.0);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MapMarker {
    #[serde(default = "MarkerId::new")]
    pub id: MarkerId,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub note: String,
    pub world: WorldPoint,
    #[serde(default)]
    pub style: MarkerStyle,
}

impl MapMarker {
    #[must_use]
    pub fn new(label: impl Into<String>, world: WorldPoint) -> Self {
        Self {
            id: MarkerId::new(),
            label: label.into(),
            note: String::new(),
            world,
            style: MarkerStyle::default(),
        }
    }

    #[must_use]
    pub fn display_label(&self) -> &str {
        if self.label.trim().is_empty() {
            "未命名标记"
        } else {
            &self.label
        }
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.style = self.style.normalized();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MarkerGroup {
    #[serde(default = "MarkerGroupId::new")]
    pub id: MarkerGroupId,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_group_visible")]
    pub visible: bool,
    #[serde(default)]
    pub default_style: MarkerStyle,
    #[serde(default)]
    pub markers: Vec<MapMarker>,
}

impl MarkerGroup {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: MarkerGroupId::new(),
            name: name.into(),
            description: String::new(),
            visible: true,
            default_style: MarkerStyle::default(),
            markers: Vec::new(),
        }
    }

    #[must_use]
    pub fn display_name(&self) -> &str {
        if self.name.trim().is_empty() {
            "未命名分组"
        } else {
            &self.name
        }
    }

    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    #[must_use]
    pub fn effective_style(&self, marker: &MapMarker) -> MarkerStyle {
        let mut style = self.default_style.clone().normalized();
        let marker_style = marker.style.clone().normalized();
        style.icon = marker_style.icon;
        if !marker_style.color_hex.trim().is_empty() {
            style.color_hex = marker_style.color_hex;
        }
        style.size_px = marker_style.size_px;
        style
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.default_style = self.default_style.normalized();
        self.markers = self
            .markers
            .into_iter()
            .map(MapMarker::normalized)
            .collect();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MarkerCatalog {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub groups: Vec<MarkerGroup>,
}

impl Default for MarkerCatalog {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            groups: Vec::new(),
        }
    }
}

impl MarkerCatalog {
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.groups.iter().map(MarkerGroup::marker_count).sum()
    }

    #[must_use]
    pub fn visible_group_count(&self) -> usize {
        self.groups.iter().filter(|group| group.visible).count()
    }

    #[must_use]
    pub fn find_group_index(&self, group_id: &MarkerGroupId) -> Option<usize> {
        self.groups.iter().position(|group| &group.id == group_id)
    }

    #[must_use]
    pub fn find_group(&self, group_id: &MarkerGroupId) -> Option<&MarkerGroup> {
        self.groups.iter().find(|group| &group.id == group_id)
    }

    pub fn find_group_mut(&mut self, group_id: &MarkerGroupId) -> Option<&mut MarkerGroup> {
        self.groups.iter_mut().find(|group| &group.id == group_id)
    }

    #[must_use]
    pub fn find_marker(
        &self,
        group_id: &MarkerGroupId,
        marker_id: &MarkerId,
    ) -> Option<&MapMarker> {
        self.find_group(group_id)?
            .markers
            .iter()
            .find(|marker| &marker.id == marker_id)
    }

    pub fn find_marker_mut(
        &mut self,
        group_id: &MarkerGroupId,
        marker_id: &MarkerId,
    ) -> Option<&mut MapMarker> {
        self.find_group_mut(group_id)?
            .markers
            .iter_mut()
            .find(|marker| &marker.id == marker_id)
    }

    pub fn remove_group(&mut self, group_id: &MarkerGroupId) -> Option<MarkerGroup> {
        let index = self.find_group_index(group_id)?;
        Some(self.groups.remove(index))
    }

    pub fn remove_marker(
        &mut self,
        group_id: &MarkerGroupId,
        marker_id: &MarkerId,
    ) -> Option<MapMarker> {
        let group = self.find_group_mut(group_id)?;
        let index = group
            .markers
            .iter()
            .position(|marker| &marker.id == marker_id)?;
        Some(group.markers.remove(index))
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.schema_version = self.schema_version.max(default_schema_version());
        self.groups = self
            .groups
            .into_iter()
            .map(MarkerGroup::normalized)
            .collect();
        self
    }
}

#[must_use]
pub fn normalize_hex_color(value: &str) -> String {
    let hex = value.trim().trim_start_matches('#');
    if hex.len() == 6 && hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        format!("#{}", hex.to_uppercase())
    } else {
        default_marker_color()
    }
}

fn default_schema_version() -> u32 {
    1
}

fn default_group_visible() -> bool {
    true
}

fn default_marker_color() -> String {
    "#FF6B6B".to_owned()
}

fn default_marker_size() -> f32 {
    24.0
}

fn default_marker_icon_name() -> &'static str {
    "黑晶琉璃"
}

fn normalize_marker_icon_name(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default_marker_icon_name().to_owned();
    }

    legacy_marker_icon_name(trimmed)
        .unwrap_or(trimmed)
        .to_owned()
}

fn legacy_marker_icon_name(value: &str) -> Option<&'static str> {
    match value {
        "Pin" | "Circle" | "Square" | "Diamond" | "Triangle" | "Cross" | "Star" => Some("黑晶琉璃"),
        "701" => Some("黑晶琉璃"),
        "702" => Some("黄石榴石"),
        "703" => Some("蓝晶碧玺"),
        "704" => Some("紫莲刚玉"),
        "705" => Some("向阳花"),
        "706" => Some("喵喵草"),
        "707" => Some("蓝掌"),
        "708" => Some("睡铃"),
        "709" => Some("天使草"),
        "710" => Some("石耳"),
        "711" => Some("伞伞菌"),
        "712" => Some("蜜黄菌"),
        "713" => Some("喷气菇"),
        "714" => Some("凤眼莲"),
        "715" => Some("蜂窝"),
        "716" => Some("星霜花"),
        "717" => Some("荧光兰"),
        "718" => Some("大嘴花"),
        "719" => Some("流星兰"),
        "720" => Some("紫晶菇"),
        "721" => Some("海桑花"),
        "722" => Some("海星石"),
        "723" => Some("彩玉花"),
        "724" => Some("象牙花"),
        "725" => Some("风卷草"),
        "726" => Some("海珊瑚"),
        "727" => Some("海神花"),
        "728" => Some("紫雀花"),
        "729" => Some("恶魔雪茄"),
        "730" => Some("骨片"),
        "731" => Some("花星角"),
        "732" => Some("火焰花"),
        "733" => Some("雪菇"),
        "734" => Some("幽幽草"),
        "735" => Some("幽幽鬼火"),
        "736" => Some("藻羽花"),
        "737" => Some("洋红珊瑚"),
        _ => None,
    }
}
