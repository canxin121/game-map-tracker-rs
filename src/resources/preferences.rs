use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::domain::theme::ThemePreference;

const UI_PREFERENCES_FILE_NAME: &str = ".game-map-tracker-rs.toml";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(default)]
pub struct UiPreferences {
    pub theme_mode: ThemePreference,
    #[serde(default = "default_auto_focus_enabled")]
    pub auto_focus_enabled: bool,
    #[serde(default = "default_tracker_point_popup_enabled")]
    pub tracker_point_popup_enabled: bool,
}

impl Default for UiPreferences {
    fn default() -> Self {
        Self {
            theme_mode: ThemePreference::default(),
            auto_focus_enabled: default_auto_focus_enabled(),
            tracker_point_popup_enabled: default_tracker_point_popup_enabled(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UiPreferencesRepository;

impl UiPreferencesRepository {
    pub fn path_for(project_root: &Path) -> PathBuf {
        project_root.join(UI_PREFERENCES_FILE_NAME)
    }

    pub fn load(project_root: &Path) -> Result<UiPreferences> {
        let path = Self::path_for(project_root);
        if !path.exists() {
            return Ok(UiPreferences::default());
        }

        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read ui preferences at {}", path.display()))?;
        toml::from_str::<UiPreferences>(&raw)
            .with_context(|| format!("failed to parse ui preferences at {}", path.display()))
    }

    pub fn save(project_root: &Path, preferences: &UiPreferences) -> Result<PathBuf> {
        let path = Self::path_for(project_root);
        Self::save_to_path(&path, preferences)?;
        Ok(path)
    }

    fn save_to_path(path: &Path, preferences: &UiPreferences) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create ui preferences parent {}",
                    parent.display()
                )
            })?;
        }
        let raw = toml::to_string_pretty(preferences)
            .context("failed to serialize ui preferences as TOML")?;
        fs::write(path, raw)
            .with_context(|| format!("failed to write ui preferences at {}", path.display()))
    }
}

const fn default_auto_focus_enabled() -> bool {
    true
}

const fn default_tracker_point_popup_enabled() -> bool {
    true
}
