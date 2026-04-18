use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::domain::route::{RouteDocument, RouteId, RouteMetadata, RoutePointId};

#[derive(Debug, Default, Clone)]
pub struct RouteImportReport {
    pub imported_count: usize,
    pub imported_point_count: usize,
    pub first_imported_group_id: Option<RouteId>,
    pub failed_sources: Vec<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RouteRepository;

impl RouteRepository {
    pub fn load_all(routes_dir: &Path) -> Result<Vec<RouteDocument>> {
        if !routes_dir.exists() {
            fs::create_dir_all(routes_dir).with_context(|| {
                format!("failed to create routes directory {}", routes_dir.display())
            })?;
            info!(routes_dir = %routes_dir.display(), "created missing routes directory");
            return Ok(Vec::new());
        }

        let mut files = fs::read_dir(routes_dir)
            .with_context(|| format!("failed to scan routes directory {}", routes_dir.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| is_json_file(&entry.path()))
            .collect::<Vec<_>>();

        files.sort_by_key(|entry| entry.file_name());

        let routes = files
            .into_iter()
            .map(|entry| Self::load_path(&entry.path()))
            .collect::<Result<Vec<_>>>()?;
        info!(
            routes_dir = %routes_dir.display(),
            route_count = routes.len(),
            "loaded route documents"
        );
        Ok(routes)
    }

    pub fn load_path(path: &Path) -> Result<RouteDocument> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read route file {}", path.display()))?;
        let mut route = serde_json::from_str::<RouteDocument>(&raw)
            .with_context(|| format!("failed to parse route file {}", path.display()))?
            .normalized();

        let file_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| Self::suggested_file_name(route.display_name()));
        let display_name = path
            .file_stem()
            .map(|stem| {
                stem.to_string_lossy()
                    .split("__")
                    .next()
                    .unwrap_or_default()
                    .to_owned()
            })
            .unwrap_or_else(|| route.name.clone());

        route.id = RouteId(file_name.clone());
        route.metadata = RouteMetadata {
            id: route.id.clone(),
            file_name,
            display_name,
        };

        Ok(route)
    }

    pub fn save(path: &Path, route: &RouteDocument) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create route parent directory {}",
                    parent.display()
                )
            })?;
        }

        let body = serde_json::to_string_pretty(&route.clone().normalized())
            .context("failed to serialize route group")?;
        fs::write(path, body)
            .with_context(|| format!("failed to write route file {}", path.display()))?;
        debug!(
            path = %path.display(),
            route_id = %route.id,
            point_count = route.point_count(),
            "saved route document"
        );
        Ok(())
    }

    pub fn normalize_directory(routes_dir: &Path) -> Result<usize> {
        let routes = Self::load_all(routes_dir)?;
        for route in &routes {
            let target = routes_dir.join(&route.metadata.file_name);
            Self::save(&target, route)?;
        }
        Ok(routes.len())
    }

    pub fn collect_import_files(root: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        collect_json_files(root, &mut files)?;
        files.sort();
        Ok(files)
    }

    pub fn import_paths<I>(paths: I, routes_dir: &Path) -> Result<RouteImportReport>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        fs::create_dir_all(routes_dir).with_context(|| {
            format!("failed to create routes directory {}", routes_dir.display())
        })?;

        let existing_routes = Self::load_all(routes_dir)?;
        let mut used_file_names = existing_routes
            .iter()
            .map(|route| route.metadata.file_name.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let mut report = RouteImportReport::default();

        for source in paths {
            if !source.is_file() || !is_json_file(&source) {
                continue;
            }

            match Self::load_path(&source) {
                Ok(mut route) => {
                    ensure_unique_point_ids(&mut route);
                    let display_name = route.display_name().to_owned();
                    let file_name =
                        allocate_import_file_name(&source, &display_name, &used_file_names);
                    let target = routes_dir.join(&file_name);
                    route.id = RouteId(file_name.clone());
                    route.metadata = RouteMetadata {
                        id: route.id.clone(),
                        file_name: file_name.clone(),
                        display_name,
                    };

                    match Self::save(&target, &route) {
                        Ok(()) => {
                            used_file_names.insert(file_name.to_ascii_lowercase());
                            report.imported_count += 1;
                            report.imported_point_count += route.point_count();
                            report
                                .first_imported_group_id
                                .get_or_insert_with(|| route.id.clone());
                            info!(
                                source = %source.display(),
                                target = %target.display(),
                                route_id = %route.id,
                                point_count = route.point_count(),
                                "imported route document"
                            );
                        }
                        Err(error) => {
                            warn!(
                                source = %source.display(),
                                error = %error,
                                "failed to save imported route document"
                            );
                            report
                                .failed_sources
                                .push(format!("{}: {error:#}", source.display()));
                        }
                    }
                }
                Err(error) => {
                    warn!(
                        source = %source.display(),
                        error = %error,
                        "failed to load imported route document"
                    );
                    report
                        .failed_sources
                        .push(format!("{}: {error:#}", source.display()));
                }
            }
        }

        info!(
            routes_dir = %routes_dir.display(),
            imported_count = report.imported_count,
            imported_point_count = report.imported_point_count,
            failed_count = report.failed_sources.len(),
            "finished importing route documents"
        );
        Ok(report)
    }

    pub fn delete(path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }

        fs::remove_file(path)
            .with_context(|| format!("failed to delete route file {}", path.display()))?;
        info!(path = %path.display(), "deleted route document");
        Ok(())
    }

    #[must_use]
    pub fn suggested_file_name(name: &str) -> String {
        let stem = sanitize_file_stem(name);
        format!("{stem}.json")
    }

    #[must_use]
    pub fn random_file_name() -> String {
        format!("route_{}.json", Uuid::new_v4().simple())
    }
}

fn collect_json_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to scan import directory {}", root.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read directory {}", root.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, files)?;
        } else if is_json_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

fn allocate_import_file_name(
    source: &Path,
    route_name: &str,
    used_file_names: &HashSet<String>,
) -> String {
    let preferred_stem = source
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or_else(|| route_name.to_owned());
    let stem = sanitize_file_stem(&preferred_stem);

    let mut candidate = format!("{stem}.json");
    let mut counter = 2usize;
    while used_file_names.contains(&candidate.to_ascii_lowercase()) {
        candidate = format!("{stem}_{counter}.json");
        counter += 1;
    }
    candidate
}

fn ensure_unique_point_ids(route: &mut RouteDocument) {
    let mut used_point_ids = HashSet::new();
    for point in &mut route.points {
        if point.id.0.trim().is_empty() || used_point_ids.contains(&point.id.0) {
            point.id = RoutePointId::new();
            while used_point_ids.contains(&point.id.0) {
                point.id = RoutePointId::new();
            }
        }
        used_point_ids.insert(point.id.0.clone());
    }
}

fn is_json_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

fn sanitize_file_stem(name: &str) -> String {
    let mut stem = name
        .trim()
        .chars()
        .map(|ch| match ch {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            control if control.is_control() => '_',
            other => other,
        })
        .collect::<String>()
        .trim_matches('.')
        .trim()
        .to_owned();

    if stem.is_empty() {
        stem = "route".to_owned();
    }

    stem
}
