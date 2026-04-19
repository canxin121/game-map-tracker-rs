use std::{fmt, fs, path::Path, str::FromStr};

use anyhow::{Context as _, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const CONFIG_FILE_NAME: &str = "config.toml";
const DEFAULT_TELEPORT_LINK_DISTANCE: f32 = 450.0;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct CaptureRegion {
    pub top: i32,
    pub left: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct LocalSearchConfig {
    pub enabled: bool,
    pub radius_px: u32,
    pub lock_fail_threshold: u32,
    pub max_accepted_jump_px: u32,
    pub reacquire_jump_threshold_px: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct AiTrackingConfig {
    pub refresh_rate_ms: u64,
    pub confidence_threshold: f32,
    pub min_match_count: usize,
    pub ransac_threshold: f32,
    pub scan_size: u32,
    pub scan_step: u32,
    pub track_radius: u32,
    pub device: AiDevicePreference,
    pub device_index: usize,
    pub weights_path: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiDevicePreference {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct TemplateTrackingConfig {
    pub refresh_rate_ms: u64,
    pub local_downscale: u32,
    pub global_downscale: u32,
    pub global_refine_radius_px: u32,
    pub local_match_threshold: f32,
    pub global_match_threshold: f32,
    pub mask_outer_radius: f32,
    pub mask_inner_radius: f32,
    pub device: AiDevicePreference,
    pub device_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct MinimapPresenceProbeConfig {
    pub enabled: bool,
    pub top: i32,
    pub left: i32,
    pub width: u32,
    pub height: u32,
    pub match_threshold: f32,
    pub device: AiDevicePreference,
    pub device_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct NetworkConfig {
    pub http_port: u16,
    pub websocket_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct AppConfig {
    pub minimap: CaptureRegion,
    pub minimap_presence_probe: MinimapPresenceProbeConfig,
    pub window_geometry: String,
    pub view_size: u32,
    pub max_lost_frames: u32,
    pub teleport_link_distance: f32,
    pub local_search: LocalSearchConfig,
    pub ai: AiTrackingConfig,
    pub template: TemplateTrackingConfig,
    pub network: NetworkConfig,
}

impl Default for CaptureRegion {
    fn default() -> Self {
        Self {
            top: 54,
            left: 2278,
            width: 255,
            height: 235,
        }
    }
}

impl Default for LocalSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            radius_px: 280,
            lock_fail_threshold: 5,
            max_accepted_jump_px: 500,
            reacquire_jump_threshold_px: 500,
        }
    }
}

impl Default for AiTrackingConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 200,
            confidence_threshold: 0.25,
            min_match_count: 6,
            ransac_threshold: 8.0,
            scan_size: 1600,
            scan_step: 1400,
            track_radius: 500,
            device: AiDevicePreference::default(),
            device_index: 0,
            weights_path: None,
        }
    }
}

impl Default for AiDevicePreference {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for AiDevicePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
        };
        f.write_str(value)
    }
}

impl FromStr for AiDevicePreference {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            _ => Err("ai.device 必须是 cpu、cuda、vulkan 或 metal。".to_owned()),
        }
    }
}

impl Default for TemplateTrackingConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 120,
            local_downscale: 4,
            global_downscale: 8,
            global_refine_radius_px: 480,
            local_match_threshold: 0.45,
            global_match_threshold: 0.40,
            mask_outer_radius: 0.96,
            mask_inner_radius: 0.16,
            device: AiDevicePreference::default(),
            device_index: 0,
        }
    }
}

impl Default for MinimapPresenceProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            top: 0,
            left: 0,
            width: 0,
            height: 0,
            match_threshold: 0.62,
            device: AiDevicePreference::Cpu,
            device_index: 0,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            http_port: 3000,
            websocket_port: 8765,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            minimap: CaptureRegion::default(),
            minimap_presence_probe: MinimapPresenceProbeConfig::default(),
            window_geometry: "400x400+1500+100".to_owned(),
            view_size: 400,
            max_lost_frames: 50,
            teleport_link_distance: DEFAULT_TELEPORT_LINK_DISTANCE,
            local_search: LocalSearchConfig::default(),
            ai: AiTrackingConfig::default(),
            template: TemplateTrackingConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn normalize_in_place(&mut self) {
        if self
            .ai
            .weights_path
            .as_deref()
            .is_some_and(is_legacy_default_ai_weights_path)
        {
            self.ai.weights_path = None;
        }
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.normalize_in_place();
        self
    }
}

fn is_legacy_default_ai_weights_path(path: &str) -> bool {
    let normalized = path.trim().replace('\\', "/");
    matches!(
        normalized.as_str(),
        "models/tracker_encoder.safetensors" | "models/candle_edge_bank.safetensors"
    )
}

impl MinimapPresenceProbeConfig {
    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    #[must_use]
    pub fn capture_region(&self) -> Option<CaptureRegion> {
        self.is_configured().then_some(CaptureRegion {
            top: self.top,
            left: self.left,
            width: self.width,
            height: self.height,
        })
    }
}

pub fn load_existing_config(project_root: &Path) -> Result<AppConfig> {
    let path = project_root.join(CONFIG_FILE_NAME);
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let mut config = toml::from_str::<AppConfig>(&raw)
        .with_context(|| format!("failed to parse config file at {}", path.display()))?;
    config.normalize_in_place();
    Ok(config)
}

pub fn save_config(project_root: &Path, config: &AppConfig) -> Result<std::path::PathBuf> {
    let path = project_root.join(CONFIG_FILE_NAME);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config parent {}", parent.display()))?;
    }

    let mut normalized = config.clone();
    normalized.normalize_in_place();
    let raw =
        toml::to_string_pretty(&normalized).context("failed to serialize config file as TOML")?;
    fs::write(&path, raw)
        .with_context(|| format!("failed to write config file at {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn temp_config_root(test_name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("game-map-tracker-rs-{test_name}-{unique}"));
        fs::create_dir_all(&path).expect("failed to create temp config root");
        path
    }

    #[test]
    fn normalized_preserves_minimap_presence_probe_enabled_flag() {
        let mut config = AppConfig::default();
        config.minimap_presence_probe.enabled = false;

        let normalized = config.normalized();

        assert!(!normalized.minimap_presence_probe.enabled);
    }

    #[test]
    fn load_existing_config_preserves_minimap_presence_probe_enabled_flag() {
        let root = temp_config_root("load-config");
        let mut config = AppConfig::default();
        config.minimap_presence_probe.enabled = false;
        let raw = toml::to_string_pretty(&config).expect("failed to serialize test config");
        fs::write(root.join(CONFIG_FILE_NAME), raw).expect("failed to write test config");

        let loaded = load_existing_config(&root).expect("failed to load config");

        assert!(!loaded.minimap_presence_probe.enabled);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn save_config_persists_minimap_presence_probe_enabled_flag() {
        let root = temp_config_root("save-config");
        let mut config = AppConfig::default();
        config.minimap_presence_probe.enabled = false;

        let path = save_config(&root, &config).expect("failed to save config");
        let loaded = load_existing_config(path.parent().expect("config path should have parent"))
            .expect("failed to reload saved config");

        assert!(!loaded.minimap_presence_probe.enabled);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn normalized_clears_legacy_default_ai_weights_path() {
        let mut config = AppConfig::default();
        config.ai.weights_path = Some("models\\tracker_encoder.safetensors".to_owned());

        let normalized = config.normalized();

        assert_eq!(normalized.ai.weights_path, None);
    }

    #[test]
    fn normalized_preserves_custom_ai_weights_path() {
        let mut config = AppConfig::default();
        config.ai.weights_path = Some("custom/tracker_encoder.safetensors".to_owned());

        let normalized = config.normalized();

        assert_eq!(
            normalized.ai.weights_path,
            Some("custom/tracker_encoder.safetensors".to_owned())
        );
    }
}
