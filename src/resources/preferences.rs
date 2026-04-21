use std::{
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    domain::theme::ThemePreference,
    error::{ContextExt as _, Result},
};

const UI_PREFERENCES_FILE_NAME: &str = ".game-map-tracker-rs.toml";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(default)]
pub struct UiPreferences {
    pub theme_mode: ThemePreference,
    #[serde(default = "default_auto_focus_enabled")]
    pub auto_focus_enabled: bool,
    #[serde(default = "default_tracker_point_popup_enabled")]
    pub tracker_point_popup_enabled: bool,
    #[serde(default = "default_debug_mode_enabled")]
    pub debug_mode_enabled: bool,
    #[serde(default = "default_test_case_capture_enabled")]
    pub test_case_capture_enabled: bool,
}

impl Default for UiPreferences {
    fn default() -> Self {
        Self {
            theme_mode: ThemePreference::default(),
            auto_focus_enabled: default_auto_focus_enabled(),
            tracker_point_popup_enabled: default_tracker_point_popup_enabled(),
            debug_mode_enabled: default_debug_mode_enabled(),
            test_case_capture_enabled: default_test_case_capture_enabled(),
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
            info!(path = %path.display(), "ui preferences file not found, using defaults");
            return Ok(UiPreferences::default());
        }

        debug!(path = %path.display(), "loading ui preferences");
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read ui preferences at {}", path.display()))?;
        let preferences = toml::from_str::<UiPreferences>(&raw)
            .with_context(|| format!("failed to parse ui preferences at {}", path.display()))?;
        info!(
            path = %path.display(),
            debug_mode_enabled = preferences.debug_mode_enabled,
            auto_focus_enabled = preferences.auto_focus_enabled,
            tracker_point_popup_enabled = preferences.tracker_point_popup_enabled,
            test_case_capture_enabled = preferences.test_case_capture_enabled,
            "loaded ui preferences"
        );
        Ok(preferences)
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
            .with_context(|| format!("failed to write ui preferences at {}", path.display()))?;
        info!(
            path = %path.display(),
            debug_mode_enabled = preferences.debug_mode_enabled,
            auto_focus_enabled = preferences.auto_focus_enabled,
            tracker_point_popup_enabled = preferences.tracker_point_popup_enabled,
            test_case_capture_enabled = preferences.test_case_capture_enabled,
            "saved ui preferences"
        );
        Ok(())
    }
}

const fn default_auto_focus_enabled() -> bool {
    true
}

const fn default_tracker_point_popup_enabled() -> bool {
    true
}

const fn default_debug_mode_enabled() -> bool {
    false
}

const fn default_test_case_capture_enabled() -> bool {
    false
}
