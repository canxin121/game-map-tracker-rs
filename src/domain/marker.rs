use derive_more::{Display, From};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum::{Display as StrumDisplay, EnumIter};
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

macro_rules! marker_icon_presets {
    ($($(#[$meta:meta])* $variant:ident => $code:literal),+ $(,)?) => {
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            Serialize,
            Deserialize,
            JsonSchema,
            StrumDisplay,
            EnumIter,
        )]
        pub enum MarkerIconStyle {
            $(
                $(#[$meta])*
                #[serde(rename = $code)]
                #[strum(to_string = $code)]
                $variant,
            )+
        }

        impl MarkerIconStyle {
            #[must_use]
            pub const fn asset_code(self) -> &'static str {
                match self {
                    $(Self::$variant => $code,)+
                }
            }

            #[must_use]
            pub const fn asset_path(self) -> &'static str {
                match self {
                    $(Self::$variant => concat!("assets/points/", $code, ".png"),)+
                }
            }
        }
    };
}

marker_icon_presets! {
    #[serde(
        alias = "Pin",
        alias = "Circle",
        alias = "Square",
        alias = "Diamond",
        alias = "Triangle",
        alias = "Cross",
        alias = "Star"
    )]
    Icon701 => "701",
    Icon702 => "702",
    Icon703 => "703",
    Icon704 => "704",
    Icon705 => "705",
    Icon706 => "706",
    Icon707 => "707",
    Icon708 => "708",
    Icon709 => "709",
    Icon710 => "710",
    Icon711 => "711",
    Icon712 => "712",
    Icon713 => "713",
    Icon714 => "714",
    Icon715 => "715",
    Icon716 => "716",
    Icon717 => "717",
    Icon718 => "718",
    Icon719 => "719",
    Icon720 => "720",
    Icon721 => "721",
    Icon722 => "722",
    Icon723 => "723",
    Icon724 => "724",
    Icon725 => "725",
    Icon726 => "726",
    Icon727 => "727",
    Icon728 => "728",
    Icon729 => "729",
    Icon730 => "730",
    Icon731 => "731",
    Icon732 => "732",
    Icon733 => "733",
    Icon734 => "734",
    Icon735 => "735",
    Icon736 => "736",
    Icon737 => "737",
}

impl Default for MarkerIconStyle {
    fn default() -> Self {
        Self::Icon701
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
