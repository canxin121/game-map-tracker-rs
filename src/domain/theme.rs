use derive_more::Display;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum::EnumIter;

#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    Display,
    EnumIter,
    Serialize,
    Deserialize,
    JsonSchema,
    PartialEq,
    Eq,
    Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    #[default]
    #[display("跟随系统")]
    FollowSystem,
    #[display("浅色")]
    Light,
    #[display("深色")]
    Dark,
}

impl ThemePreference {
    pub const fn description(self) -> &'static str {
        match self {
            Self::FollowSystem => "窗口跟随系统外观变化自动切换。",
            Self::Light => "始终使用浅色主题。",
            Self::Dark => "始终使用深色主题。",
        }
    }
}
