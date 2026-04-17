use std::{
    env, fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use directories::ProjectDirs;
use image::GrayImage;

use crate::{
    config::{AppConfig, CONFIG_FILE_NAME},
    resources::{
        AssetManifest, BwikiCachePaths, WorkspaceLoadReport, WorkspaceSnapshot,
        default_map_dimensions,
    },
};

const APP_QUALIFIER: &str = "io";
const APP_ORGANIZATION: &str = "rocom";
const APP_NAME: &str = "game-map-tracker-rs";
const DATA_DIR_ENV: &str = "GAME_MAP_TRACKER_RS_DATA_DIR";

pub(crate) fn build_test_workspace(config: AppConfig, namespace: &str) -> WorkspaceSnapshot {
    let project_root = env::temp_dir()
        .join("game-map-tracker-rs-test-workspace")
        .join(namespace);
    let bwiki_cache_dir = runtime_workspace_root().join("cache").join("bwiki");
    let tracking_cache_dir = project_root.join("cache").join("tracking");
    let routes_dir = project_root.join("routes");

    fs::create_dir_all(&tracking_cache_dir)
        .unwrap_or_else(|error| panic!("failed to create test tracking cache: {error:#}"));
    fs::create_dir_all(&routes_dir)
        .unwrap_or_else(|error| panic!("failed to create test routes directory: {error:#}"));
    BwikiCachePaths::new(&bwiki_cache_dir)
        .ensure_directories()
        .unwrap_or_else(|error| panic!("failed to prepare shared BWiki cache: {error:#}"));

    let map_dimensions = default_map_dimensions();
    WorkspaceSnapshot {
        project_root: project_root.clone(),
        config,
        assets: AssetManifest {
            config_path: project_root.join(CONFIG_FILE_NAME),
            routes_dir,
            bwiki_cache_dir,
            map_dimensions,
        },
        groups: Vec::new(),
        report: WorkspaceLoadReport {
            group_count: 0,
            point_count: 0,
            map_dimensions,
        },
    }
}

pub(crate) fn sample_world_positions(
    image: &GrayImage,
    view_size: u32,
    desired: usize,
) -> Vec<(u32, u32)> {
    let min_center = align_to(view_size / 2 + 32, 4);
    let max_x = align_to(image.width().saturating_sub(view_size / 2 + 32), 4);
    let max_y = align_to(image.height().saturating_sub(view_size / 2 + 32), 4);
    if max_x <= min_center || max_y <= min_center {
        return vec![(min_center, min_center)];
    }

    let stride = align_to((view_size / 2).max(160), 4).max(4);
    let radius = (view_size / 2).clamp(96, 180);
    let mut candidates = Vec::new();

    for y in stepped_positions(min_center, max_y, stride) {
        for x in stepped_positions(min_center, max_x, stride) {
            candidates.push((local_texture_score(image, x, y, radius), x, y));
        }
    }

    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    let min_separation = view_size.max(600);
    let mut points = Vec::new();
    for (_, x, y) in candidates {
        if points
            .iter()
            .all(|(px, py)| x.abs_diff(*px) + y.abs_diff(*py) >= min_separation)
        {
            points.push((x, y));
        }
        if points.len() >= desired {
            return points;
        }
    }

    for (x, y) in fallback_grid_positions(min_center, max_x, max_y) {
        if points
            .iter()
            .all(|(px, py)| x.abs_diff(*px) + y.abs_diff(*py) >= min_separation / 2)
        {
            points.push((x, y));
        }
        if points.len() >= desired {
            break;
        }
    }

    points
}

pub(crate) fn timed<T>(operation: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let value = operation();
    (value, started.elapsed())
}

pub(crate) fn benchmark_repeated<T>(
    warmups: usize,
    iterations: usize,
    mut operation: impl FnMut() -> T,
) -> (T, Duration) {
    assert!(iterations > 0, "benchmark iterations must be > 0");

    for _ in 0..warmups {
        black_box(operation());
    }

    let started = Instant::now();
    let mut last = None;
    for _ in 0..iterations {
        last = Some(operation());
    }
    let elapsed = started.elapsed();

    (
        last.expect("benchmark_repeated must execute at least once"),
        elapsed,
    )
}

pub(crate) fn print_perf_ms(scope: &str, metric: &str, duration: Duration) {
    println!(
        "[perf][{scope}] {metric}={:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
}

pub(crate) fn print_perf_per_op(scope: &str, metric: &str, iterations: usize, total: Duration) {
    let total_ms = total.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations.max(1) as f64;
    let throughput = if total.as_secs_f64() > 0.0 {
        iterations as f64 / total.as_secs_f64()
    } else {
        f64::INFINITY
    };
    println!(
        "[perf][{scope}] {metric} avg={avg_ms:.2}ms/op throughput={throughput:.2}ops/s iterations={iterations} total={total_ms:.2}ms",
    );
}

fn runtime_workspace_root() -> PathBuf {
    if let Ok(path) = env::var(DATA_DIR_ENV) {
        let path = PathBuf::from(path);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }

    ProjectDirs::from(APP_QUALIFIER, APP_ORGANIZATION, APP_NAME)
        .unwrap_or_else(|| panic!("failed to resolve runtime workspace root"))
        .data_local_dir()
        .to_path_buf()
}

fn align_to(value: u32, step: u32) -> u32 {
    (value / step.max(1)) * step.max(1)
}

fn stepped_positions(start: u32, end: u32, step: u32) -> Vec<u32> {
    let mut positions = Vec::new();
    let mut current = start;
    while current <= end {
        positions.push(current);
        current = current.saturating_add(step.max(1));
        if current == positions.last().copied().unwrap_or(current) {
            break;
        }
    }
    if positions.last().copied() != Some(end) {
        positions.push(end);
    }
    positions
}

fn fallback_grid_positions(min_center: u32, max_x: u32, max_y: u32) -> Vec<(u32, u32)> {
    let anchors_x = [min_center, (min_center + max_x) / 2, max_x];
    let anchors_y = [min_center, (min_center + max_y) / 2, max_y];
    let mut positions = Vec::new();
    for y in anchors_y {
        for x in anchors_x {
            positions.push((align_to(x, 4), align_to(y, 4)));
        }
    }
    positions
}

fn local_texture_score(image: &GrayImage, center_x: u32, center_y: u32, radius: u32) -> u64 {
    let left = center_x.saturating_sub(radius);
    let top = center_y.saturating_sub(radius);
    let right = (center_x + radius).min(image.width().saturating_sub(1));
    let bottom = (center_y + radius).min(image.height().saturating_sub(1));
    let sample_step = (radius / 10).max(8);
    let mut score = 0u64;

    let mut y = top + sample_step;
    while y <= bottom {
        let mut x = left + sample_step;
        while x <= right {
            let value = i32::from(image.get_pixel(x, y).0[0]);
            let left_value = i32::from(image.get_pixel(x - sample_step, y).0[0]);
            let top_value = i32::from(image.get_pixel(x, y - sample_step).0[0]);
            score += (value - left_value).unsigned_abs() as u64;
            score += (value - top_value).unsigned_abs() as u64;
            x = x.saturating_add(sample_step);
        }
        y = y.saturating_add(sample_step);
    }

    score
}
