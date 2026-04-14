use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, anyhow};
use directories::ProjectDirs;

use crate::{
    config::{AppConfig, CONFIG_FILE_NAME},
    resources::{BwikiCachePaths, RouteRepository},
};

const DATA_DIR_ENV: &str = "GAME_MAP_TRACKER_RS_DATA_DIR";
const APP_QUALIFIER: &str = "io";
const APP_ORGANIZATION: &str = "rocom";
const APP_NAME: &str = "game-map-tracker-rs";

#[derive(Debug, Clone)]
pub struct WorkspaceBootstrap {
    pub workspace_root: PathBuf,
}

impl WorkspaceBootstrap {
    pub fn prepare() -> Result<Self> {
        let workspace_root = data_dir_override_path()?.unwrap_or(default_workspace_root()?);
        ensure_default_config(&workspace_root)?;
        ensure_workspace_layout(&workspace_root)?;
        remove_obsolete_workspace_paths(&workspace_root)?;
        let _ = RouteRepository::normalize_directory(&workspace_root.join("routes"))?;
        Ok(Self { workspace_root })
    }
}

fn data_dir_override_path() -> Result<Option<PathBuf>> {
    match env::var(DATA_DIR_ENV) {
        Ok(path) => {
            let path = PathBuf::from(path);
            if path.as_os_str().is_empty() {
                return Ok(None);
            }
            Ok(Some(path))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow!("failed to read {DATA_DIR_ENV}: {error}")),
    }
}

fn default_workspace_root() -> Result<PathBuf> {
    let dirs = ProjectDirs::from(APP_QUALIFIER, APP_ORGANIZATION, APP_NAME)
        .ok_or_else(|| anyhow!("failed to resolve project directories"))?;
    Ok(dirs.data_local_dir().to_path_buf())
}

fn ensure_default_config(root: &Path) -> Result<()> {
    fs::create_dir_all(root)
        .with_context(|| format!("failed to create workspace root {}", root.display()))?;

    let config_path = root.join(CONFIG_FILE_NAME);
    if !config_path.exists() {
        let raw = toml::to_string_pretty(&AppConfig::default())
            .context("failed to serialize default config as TOML")?;
        fs::write(&config_path, raw)
            .with_context(|| format!("failed to write default config {}", config_path.display()))?;
    }

    Ok(())
}

fn ensure_workspace_layout(root: &Path) -> Result<()> {
    for path in [root.join("routes"), root.join("cache").join("bwiki")] {
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create workspace directory {}", path.display()))?;
    }
    BwikiCachePaths::new(root.join("cache").join("bwiki")).ensure_directories()?;
    Ok(())
}

fn remove_obsolete_workspace_paths(root: &Path) -> Result<()> {
    for file in ["config.json", ".game-map-tracker-rs.json"] {
        let path = root.join(file);
        if path.is_file() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove obsolete file {}", path.display()))?;
        }
    }

    let legacy_dir = root.join("legacy");
    if legacy_dir.is_dir() {
        fs::remove_dir_all(&legacy_dir).with_context(|| {
            format!(
                "failed to remove obsolete legacy workspace {}",
                legacy_dir.display()
            )
        })?;
    }

    for obsolete_dir in [
        root.join("assets").join("map"),
        root.join("assets").join("points"),
    ] {
        if obsolete_dir.is_dir() {
            fs::remove_dir_all(&obsolete_dir).with_context(|| {
                format!(
                    "failed to remove obsolete runtime asset directory {}",
                    obsolete_dir.display()
                )
            })?;
        }
    }

    let assets_root = root.join("assets");
    if assets_root.is_dir() && fs::read_dir(&assets_root)?.next().is_none() {
        fs::remove_dir(&assets_root).with_context(|| {
            format!(
                "failed to remove empty obsolete asset root {}",
                assets_root.display()
            )
        })?;
    }

    Ok(())
}
