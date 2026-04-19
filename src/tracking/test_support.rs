use std::{
    env, fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use directories::ProjectDirs;
use image::{
    GrayImage, Luma,
    imageops::{FilterType, crop_imm, replace, resize},
};

#[cfg(all(feature = "ai-burn", burn_vulkan_backend))]
use crate::tracking::burn_support::available_burn_device_descriptors;
use crate::{
    config::{AiDevicePreference, AppConfig, CONFIG_FILE_NAME, load_existing_config},
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

pub(crate) fn runtime_config_or_default() -> AppConfig {
    load_existing_config(&runtime_workspace_root()).unwrap_or_else(|error| {
        eprintln!("failed to load runtime config for stress tests: {error:#}");
        AppConfig::default()
    })
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

pub(crate) fn stress_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

pub(crate) fn stress_env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
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

#[derive(Debug, Clone)]
pub(crate) struct StressPathCase {
    pub start: (u32, u32),
    pub locals: Vec<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub(crate) struct StressFailure {
    pub case_index: usize,
    pub step_index: usize,
    pub stage: &'static str,
    pub expected: (u32, u32),
    pub actual: Option<(f32, f32)>,
    pub score: Option<f32>,
    pub source: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StressRoundStats {
    pub global_total: usize,
    pub global_success: usize,
    pub local_total: usize,
    pub local_success: usize,
    pub failures: Vec<StressFailure>,
}

impl StressRoundStats {
    pub(crate) fn global_accuracy(&self) -> f32 {
        ratio(self.global_success, self.global_total)
    }

    pub(crate) fn local_accuracy(&self) -> f32 {
        ratio(self.local_success, self.local_total)
    }

    pub(crate) fn overall_accuracy(&self) -> f32 {
        ratio(
            self.global_success + self.local_success,
            self.global_total + self.local_total,
        )
    }
}

pub(crate) fn synthetic_capture_from_map(
    map: &GrayImage,
    config: &AppConfig,
    center: (u32, u32),
) -> GrayImage {
    let half = config.view_size / 2;
    let left = center.0.saturating_sub(half);
    let top = center.1.saturating_sub(half);
    let crop = crop_imm(map, left, top, config.view_size, config.view_size).to_image();

    let diameter_px = config.minimap.width.min(config.minimap.height).max(1);
    let diameter_px = ((diameter_px as f32) * config.template.mask_outer_radius).round() as u32;
    let minimap = resize(&crop, diameter_px, diameter_px, FilterType::Triangle);
    let mut canvas = GrayImage::from_pixel(config.minimap.width, config.minimap.height, Luma([0]));
    let offset_x = i64::from((config.minimap.width - diameter_px) / 2);
    let offset_y = i64::from((config.minimap.height - diameter_px) / 2);
    replace(&mut canvas, &minimap, offset_x, offset_y);
    canvas
}

pub(crate) fn random_stress_paths(
    image: &GrayImage,
    view_size: u32,
    case_count: usize,
    local_steps: usize,
    min_step: u32,
    max_step: u32,
    seed: u64,
) -> Vec<StressPathCase> {
    let min_center = align_to(view_size / 2 + 32, 4);
    let max_x = align_to(image.width().saturating_sub(view_size / 2 + 32), 4);
    let max_y = align_to(image.height().saturating_sub(view_size / 2 + 32), 4);
    if max_x <= min_center || max_y <= min_center {
        return vec![StressPathCase {
            start: (min_center, min_center),
            locals: Vec::new(),
        }];
    }

    let mut rng = DeterministicRng::new(seed);
    let texture_radius = local_texture_radius(view_size);
    let texture_floor =
        estimated_texture_floor(image, texture_radius, min_center, max_x, max_y, seed);
    let mut cases = Vec::with_capacity(case_count);
    let mut attempts = 0usize;
    while cases.len() < case_count && attempts < case_count.saturating_mul(256).max(256) {
        attempts += 1;
        let start = random_world_point(&mut rng, min_center, max_x, max_y);
        if local_texture_score(image, start.0, start.1, texture_radius) < texture_floor {
            continue;
        }
        let locals = build_local_path(
            &mut rng,
            image,
            start,
            local_steps,
            min_step,
            max_step,
            min_center,
            max_x,
            max_y,
            texture_radius,
            texture_floor,
        );
        cases.push(StressPathCase { start, locals });
    }

    while cases.len() < case_count {
        let start = random_world_point(&mut rng, min_center, max_x, max_y);
        let locals = build_local_path(
            &mut rng,
            image,
            start,
            local_steps,
            min_step,
            max_step,
            min_center,
            max_x,
            max_y,
            texture_radius,
            0,
        );
        cases.push(StressPathCase { start, locals });
    }
    cases
}

pub(crate) fn write_stress_report(
    engine: &str,
    round: usize,
    stats: &StressRoundStats,
    note: &str,
) -> PathBuf {
    let root = env::temp_dir()
        .join("game-map-tracker-rs-stress")
        .join(engine);
    fs::create_dir_all(&root).expect("failed to create stress report directory");
    let path = root.join(format!("round-{round:02}.txt"));
    let mut report = String::new();
    report.push_str(&format!("engine={engine}\n"));
    report.push_str(&format!("round={round}\n"));
    report.push_str(&format!(
        "global={}/{} ({:.2}%)\n",
        stats.global_success,
        stats.global_total,
        stats.global_accuracy() * 100.0
    ));
    report.push_str(&format!(
        "local={}/{} ({:.2}%)\n",
        stats.local_success,
        stats.local_total,
        stats.local_accuracy() * 100.0
    ));
    report.push_str(&format!(
        "overall_accuracy={:.2}%\n",
        stats.overall_accuracy() * 100.0
    ));
    report.push_str(&format!("note={note}\n"));
    report.push_str("failures:\n");
    for failure in &stats.failures {
        let actual = failure
            .actual
            .map(|(x, y)| format!("{x:.1},{y:.1}"))
            .unwrap_or_else(|| "none".to_owned());
        let score = failure
            .score
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "--".to_owned());
        let source = failure.source.clone().unwrap_or_else(|| "--".to_owned());
        report.push_str(&format!(
            "case={} step={} stage={} expected={},{} actual={} score={} source={} note={}\n",
            failure.case_index,
            failure.step_index,
            failure.stage,
            failure.expected.0,
            failure.expected.1,
            actual,
            score,
            source,
            failure.note
        ));
    }
    fs::write(&path, report).expect("failed to write stress report");
    path
}

#[cfg(all(feature = "ai-burn", burn_vulkan_backend))]
pub(crate) fn require_vulkan_discrete_ordinal() -> (usize, String) {
    let descriptors = available_burn_device_descriptors(AiDevicePreference::Vulkan);
    descriptors
        .into_iter()
        .find(|descriptor| descriptor.name.starts_with("独显 GPU"))
        .map(|descriptor| (descriptor.ordinal, descriptor.name))
        .unwrap_or_else(|| panic!("no Vulkan discrete GPU is available for stress tests"))
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

fn local_texture_radius(view_size: u32) -> u32 {
    (view_size / 2).clamp(96, 180)
}

fn estimated_texture_floor(
    image: &GrayImage,
    radius: u32,
    min_center: u32,
    max_x: u32,
    max_y: u32,
    seed: u64,
) -> u64 {
    let mut rng = DeterministicRng::new(seed ^ 0xa5a5_5a5a_1f2e_3d4c);
    let mut scores = Vec::with_capacity(128);
    for _ in 0..128 {
        let point = random_world_point(&mut rng, min_center, max_x, max_y);
        scores.push(local_texture_score(image, point.0, point.1, radius));
    }
    if scores.is_empty() {
        return 0;
    }

    scores.sort_unstable();
    scores[scores.len() / 5]
}

fn ratio(successes: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        successes as f32 / total as f32
    }
}

fn random_world_point(
    rng: &mut DeterministicRng,
    min_center: u32,
    max_x: u32,
    max_y: u32,
) -> (u32, u32) {
    (
        align_to(rng.u32_inclusive(min_center, max_x), 4),
        align_to(rng.u32_inclusive(min_center, max_y), 4),
    )
}

fn build_local_path(
    rng: &mut DeterministicRng,
    image: &GrayImage,
    start: (u32, u32),
    local_steps: usize,
    min_step: u32,
    max_step: u32,
    min_center: u32,
    max_x: u32,
    max_y: u32,
    texture_radius: u32,
    texture_floor: u64,
) -> Vec<(u32, u32)> {
    let mut current = start;
    let mut path = Vec::with_capacity(local_steps);
    for _ in 0..local_steps {
        let mut next = current;
        for _ in 0..24 {
            let step_x = rng.signed_step(min_step, max_step);
            let step_y = rng.signed_step(min_step, max_step);
            if step_x == 0 && step_y == 0 {
                continue;
            }
            let candidate = (
                align_to(
                    (current.0 as i32 + step_x).clamp(min_center as i32, max_x as i32) as u32,
                    4,
                ),
                align_to(
                    (current.1 as i32 + step_y).clamp(min_center as i32, max_y as i32) as u32,
                    4,
                ),
            );
            if candidate != current
                && local_texture_score(image, candidate.0, candidate.1, texture_radius)
                    >= texture_floor
            {
                next = candidate;
                break;
            }
        }
        current = next;
        path.push(current);
    }
    path
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn u32_inclusive(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        let span = max - min + 1;
        min + (self.next_u32() % span)
    }

    fn signed_step(&mut self, min_step: u32, max_step: u32) -> i32 {
        let magnitude = self.u32_inclusive(min_step.min(max_step), max_step.max(min_step)) as i32;
        match self.next_u32() & 0b11 {
            0 => magnitude,
            1 => -magnitude,
            2 => magnitude / 2,
            _ => -(magnitude / 2),
        }
    }
}
