use std::path::{Path, PathBuf};

use tracing::info;

use crate::{
    config::{AppConfig, CONFIG_FILE_NAME, load_existing_config},
    domain::{geometry::MapDimensions, route::RouteDocument},
    error::Result,
    resources::{BwikiCachePaths, RouteRepository, default_map_dimensions},
};

#[derive(Debug, Clone)]
pub struct AssetManifest {
    pub config_path: PathBuf,
    pub routes_dir: PathBuf,
    pub bwiki_cache_dir: PathBuf,
    pub map_dimensions: MapDimensions,
}

#[derive(Debug, Clone)]
pub struct WorkspaceLoadReport {
    pub group_count: usize,
    pub point_count: usize,
    pub map_dimensions: MapDimensions,
}

#[derive(Debug, Clone)]
pub struct WorkspaceSnapshot {
    pub project_root: PathBuf,
    pub config: AppConfig,
    pub assets: AssetManifest,
    pub groups: Vec<RouteDocument>,
    pub report: WorkspaceLoadReport,
}

impl WorkspaceSnapshot {
    pub fn load(project_root: impl Into<PathBuf>) -> Result<Self> {
        let project_root = project_root.into();
        info!(project_root = %project_root.display(), "loading workspace snapshot");
        let assets = discover_assets(&project_root)?;
        let config = load_existing_config(&project_root)?;
        let groups = RouteRepository::load_all(&assets.routes_dir)?;
        let point_count = groups.iter().map(RouteDocument::point_count).sum();
        let report = WorkspaceLoadReport {
            group_count: groups.len(),
            point_count,
            map_dimensions: assets.map_dimensions,
        };
        info!(
            project_root = %project_root.display(),
            group_count = report.group_count,
            point_count = report.point_count,
            "workspace snapshot loaded"
        );

        Ok(Self {
            project_root,
            config,
            assets,
            groups,
            report,
        })
    }
}

fn discover_assets(project_root: &Path) -> Result<AssetManifest> {
    let config_path = project_root.join(CONFIG_FILE_NAME);
    let routes_dir = project_root.join("routes");
    let bwiki_cache_dir = project_root.join("cache").join("bwiki");
    BwikiCachePaths::new(&bwiki_cache_dir).ensure_directories()?;
    let map_dimensions = default_map_dimensions();

    Ok(AssetManifest {
        config_path,
        routes_dir,
        bwiki_cache_dir,
        map_dimensions,
    })
}
