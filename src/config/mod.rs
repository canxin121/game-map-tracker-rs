use std::{fs, path::Path};

use anyhow::{Context as _, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const CONFIG_FILE_NAME: &str = "config.toml";
const DEFAULT_TELEPORT_LINK_DISTANCE: f32 = 320.0;

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
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default)]
pub struct SiftTrackingConfig {
    pub refresh_rate_ms: u64,
    pub clahe_limit: f32,
    pub match_ratio: f32,
    pub min_match_count: usize,
    pub ransac_threshold: f32,
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
    pub weights_path: Option<String>,
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
    pub window_geometry: String,
    pub view_size: u32,
    pub max_lost_frames: u32,
    pub teleport_link_distance: f32,
    pub local_search: LocalSearchConfig,
    pub sift: SiftTrackingConfig,
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
        }
    }
}

impl Default for SiftTrackingConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 50,
            clahe_limit: 3.0,
            match_ratio: 0.9,
            min_match_count: 5,
            ransac_threshold: 8.0,
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
            weights_path: None,
        }
    }
}

impl Default for TemplateTrackingConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 120,
            local_downscale: 4,
            global_downscale: 16,
            global_refine_radius_px: 480,
            local_match_threshold: 0.53,
            global_match_threshold: 0.45,
            mask_outer_radius: 0.96,
            mask_inner_radius: 0.16,
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
            window_geometry: "400x400+1500+100".to_owned(),
            view_size: 400,
            max_lost_frames: 50,
            teleport_link_distance: DEFAULT_TELEPORT_LINK_DISTANCE,
            local_search: LocalSearchConfig::default(),
            sift: SiftTrackingConfig::default(),
            ai: AiTrackingConfig::default(),
            template: TemplateTrackingConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

pub fn load_existing_config(project_root: &Path) -> Result<AppConfig> {
    let path = project_root.join(CONFIG_FILE_NAME);
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    toml::from_str::<AppConfig>(&raw)
        .with_context(|| format!("failed to parse config file at {}", path.display()))
}

pub fn save_config(project_root: &Path, config: &AppConfig) -> Result<std::path::PathBuf> {
    let path = project_root.join(CONFIG_FILE_NAME);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config parent {}", parent.display()))?;
    }

    let raw =
        toml::to_string_pretty(config).context("failed to serialize config file as TOML")?;
    fs::write(&path, raw)
        .with_context(|| format!("failed to write config file at {}", path.display()))?;
    Ok(path)
}
