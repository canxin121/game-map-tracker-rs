use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, unbounded};
use image::{
    GrayImage, Luma, RgbaImage,
    imageops::{FilterType, replace, resize},
};
use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};
use ureq::Agent;

use crate::domain::geometry::{MapDimensions, WorldPoint};

const USER_AGENT: &str = "game-map-tracker-rs-bwiki-runtime/1.0";
const DATASET_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
const TILE_SIZE: u32 = 256;
pub const BWIKI_WORLD_ZOOM: u8 = 8;
const TILE_URL_TEMPLATE: &str =
    "https://wiki-dev-patch-oss.oss-cn-hangzhou.aliyuncs.com/res/lkwg/map-3.0/{z}/tile-{x}_{y}.png";
const TYPE_PARSE_URL: &str = "https://wiki.biligame.com/rocom/api.php?action=parse&page=Data:Mapnew/type/json&prop=text&format=json&formatversion=2";
const POINT_PARSE_URL: &str = "https://wiki.biligame.com/rocom/api.php?action=parse&page=Data:Mapnew/point.json&prop=text&format=json&formatversion=2";
const TRACKING_POI_MARK_TYPES: [u32; 7] = [201, 202, 203, 204, 205, 206, 210];
const TRACKING_POI_ICON_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
const TRACKING_POI_ICON_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BwikiTileZoom {
    pub zoom: u8,
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
}

impl BwikiTileZoom {
    #[must_use]
    pub const fn width_tiles(self) -> u32 {
        (self.max_x - self.min_x + 1) as u32
    }

    #[must_use]
    pub const fn height_tiles(self) -> u32 {
        (self.max_y - self.min_y + 1) as u32
    }

    #[must_use]
    pub const fn world_tile_size(self) -> u32 {
        TILE_SIZE << (BWIKI_WORLD_ZOOM - self.zoom)
    }

    #[must_use]
    pub const fn world_width(self) -> u32 {
        self.width_tiles() * self.world_tile_size()
    }

    #[must_use]
    pub const fn world_height(self) -> u32 {
        self.height_tiles() * self.world_tile_size()
    }

    #[must_use]
    pub const fn source_world_pixel_size(self) -> u32 {
        1u32 << (BWIKI_WORLD_ZOOM - self.zoom)
    }
}

pub const BWIKI_TILE_ZOOMS: [BwikiTileZoom; 5] = [
    BwikiTileZoom {
        zoom: 4,
        min_x: -2,
        max_x: 1,
        min_y: -2,
        max_y: 1,
    },
    BwikiTileZoom {
        zoom: 5,
        min_x: -3,
        max_x: 2,
        min_y: -3,
        max_y: 2,
    },
    BwikiTileZoom {
        zoom: 6,
        min_x: -6,
        max_x: 5,
        min_y: -5,
        max_y: 4,
    },
    BwikiTileZoom {
        zoom: 7,
        min_x: -12,
        max_x: 11,
        min_y: -9,
        max_y: 8,
    },
    BwikiTileZoom {
        zoom: 8,
        min_x: -24,
        max_x: 23,
        min_y: -18,
        max_y: 17,
    },
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BwikiTypeDefinition {
    pub category: String,
    pub mark_type: u32,
    pub name: String,
    pub icon_url: String,
    pub point_count: usize,
    pub type_known: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BwikiPointRecord {
    pub mark_type: u32,
    pub title: String,
    pub id: String,
    pub raw_lat: i32,
    pub raw_lng: i32,
    pub world: WorldPoint,
    pub uid: String,
    pub layer: String,
    pub time: Option<u64>,
    pub version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BwikiDataset {
    pub fetched_at_epoch_secs: u64,
    pub types: Vec<BwikiTypeDefinition>,
    pub points_by_type: BTreeMap<u32, Vec<BwikiPointRecord>>,
}

impl BwikiDataset {
    #[must_use]
    pub fn type_by_mark_type(&self, mark_type: u32) -> Option<&BwikiTypeDefinition> {
        self.types.iter().find(|item| item.mark_type == mark_type)
    }

    #[must_use]
    pub fn type_by_icon_name(&self, icon_name: &str) -> Option<&BwikiTypeDefinition> {
        let normalized = icon_name.trim();
        if normalized.is_empty() {
            return None;
        }

        self.types
            .iter()
            .find(|item| item.name == normalized)
            .or_else(|| {
                normalized
                    .parse::<u32>()
                    .ok()
                    .and_then(|mark_type| self.type_by_mark_type(mark_type))
            })
    }

    #[must_use]
    pub fn sorted_category_names(&self) -> Vec<String> {
        let mut names = self
            .types
            .iter()
            .map(|item| item.category.clone())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        names
    }

    #[must_use]
    pub fn total_point_count(&self) -> usize {
        self.points_by_type.values().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone)]
pub struct BwikiCachePaths {
    pub root: PathBuf,
    pub data_dir: PathBuf,
    pub icons_dir: PathBuf,
    pub tiles_dir: PathBuf,
}

impl BwikiCachePaths {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            data_dir: root.join("data"),
            icons_dir: root.join("icons"),
            tiles_dir: root.join("tiles"),
            root,
        }
    }

    pub fn ensure_directories(&self) -> Result<()> {
        let legacy_stitched_dir = self.root.join("stitched");
        if legacy_stitched_dir.is_dir() {
            fs::remove_dir_all(&legacy_stitched_dir).with_context(|| {
                format!(
                    "failed to remove legacy BWiki stitched cache directory {}",
                    legacy_stitched_dir.display()
                )
            })?;
        }

        for path in [&self.root, &self.data_dir, &self.icons_dir, &self.tiles_dir] {
            fs::create_dir_all(path).with_context(|| {
                format!("failed to create BWiki cache directory {}", path.display())
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BwikiResourceManager {
    inner: Arc<BwikiManagerInner>,
}

#[derive(Debug)]
struct BwikiManagerInner {
    cache: BwikiCachePaths,
    state: Mutex<BwikiManagerState>,
    version: AtomicU64,
    job_tx: Sender<BwikiJob>,
}

#[derive(Debug, Default)]
struct BwikiManagerState {
    dataset: Option<Arc<BwikiDataset>>,
    queued_jobs: HashSet<BwikiJobKey>,
    ready_icons: HashMap<u32, PathBuf>,
    last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BwikiJobKey {
    RefreshDataset,
    DownloadIcon { mark_type: u32 },
    DownloadTile { zoom: u8, x: i32, y: i32 },
}

#[derive(Debug, Clone)]
enum BwikiJob {
    RefreshDataset,
    DownloadIcon { mark_type: u32, icon_url: String },
    DownloadTile { zoom: u8, x: i32, y: i32 },
}

impl BwikiJob {
    fn key(&self) -> BwikiJobKey {
        match self {
            Self::RefreshDataset => BwikiJobKey::RefreshDataset,
            Self::DownloadIcon { mark_type, .. } => BwikiJobKey::DownloadIcon {
                mark_type: *mark_type,
            },
            Self::DownloadTile { zoom, x, y } => BwikiJobKey::DownloadTile {
                zoom: *zoom,
                x: *x,
                y: *y,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParseResponse {
    parse: ParsePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParsePayload {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawTypeCollection {
    data: Vec<RawTypeDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawTypeDefinition {
    #[serde(rename = "type")]
    category: String,
    #[serde(rename = "markType")]
    mark_type: u32,
    #[serde(rename = "markTypeName")]
    name: String,
    #[serde(default)]
    icon: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawPointDefinition {
    #[serde(rename = "markType")]
    mark_type: u32,
    #[serde(default)]
    title: String,
    id: String,
    point: RawPointCoordinate,
    #[serde(default)]
    uid: String,
    #[serde(default)]
    layer: String,
    #[serde(default)]
    time: Option<u64>,
    #[serde(default)]
    version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawPointCoordinate {
    lat: i32,
    lng: i32,
}

impl BwikiResourceManager {
    pub fn new(cache_root: impl Into<PathBuf>) -> Result<Self> {
        let cache = BwikiCachePaths::new(cache_root);
        cache.ensure_directories()?;
        info!(cache_root = %cache.root.display(), "initializing BWiki resource manager");

        let (job_tx, job_rx) = unbounded();
        let inner = Arc::new(BwikiManagerInner {
            cache,
            state: Mutex::new(BwikiManagerState::default()),
            version: AtomicU64::new(0),
            job_tx,
        });

        for index in 0..4 {
            let worker_inner = inner.clone();
            let worker_rx = job_rx.clone();
            thread::Builder::new()
                .name(format!("bwiki-cache-{index}"))
                .spawn(move || run_bwiki_worker(worker_inner, worker_rx))
                .map_err(|error| anyhow!("failed to spawn BWiki cache worker: {error}"))?;
        }

        info!(cache_root = %inner.cache.root.display(), worker_count = 4, "BWiki resource manager ready");
        Ok(Self { inner })
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner.version.load(Ordering::SeqCst)
    }

    #[must_use]
    pub fn cache_paths(&self) -> &BwikiCachePaths {
        &self.inner.cache
    }

    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.inner.state.lock().last_error.clone()
    }

    #[must_use]
    pub fn dataset_snapshot(&self) -> Option<Arc<BwikiDataset>> {
        if let Some(dataset) = self.inner.state.lock().dataset.clone() {
            return Some(dataset);
        }

        let cache_path = dataset_cache_path(&self.inner.cache);
        if let Ok(dataset) = load_dataset_from_cache(&cache_path) {
            let dataset = Arc::new(dataset);
            self.inner.state.lock().dataset = Some(dataset.clone());
            return Some(dataset);
        }

        None
    }

    pub fn ensure_dataset_loaded(&self) {
        let cache_path = dataset_cache_path(&self.inner.cache);
        let stale = cache_is_stale(&cache_path, DATASET_CACHE_TTL);
        if self.dataset_snapshot().is_none() || stale {
            debug!(
                cache_path = %cache_path.display(),
                stale,
                "queueing BWiki dataset refresh"
            );
            self.enqueue(BwikiJob::RefreshDataset);
        }
    }

    pub fn refresh_dataset(&self) {
        info!("explicitly refreshing BWiki dataset");
        self.enqueue(BwikiJob::RefreshDataset);
    }

    #[must_use]
    pub fn is_dataset_refresh_pending(&self) -> bool {
        self.inner
            .state
            .lock()
            .queued_jobs
            .contains(&BwikiJobKey::RefreshDataset)
    }

    #[must_use]
    pub fn resolve_icon_definition(&self, icon_name: &str) -> Option<BwikiTypeDefinition> {
        self.ensure_dataset_loaded();
        let dataset = self.dataset_snapshot()?;
        dataset.type_by_icon_name(icon_name).cloned()
    }

    #[must_use]
    pub fn ensure_icon_path(&self, mark_type: u32, icon_url: &str) -> Option<PathBuf> {
        if icon_url.trim().is_empty() {
            return None;
        }

        if let Some(path) = self.inner.state.lock().ready_icons.get(&mark_type).cloned() {
            return Some(path);
        }

        let path = icon_cache_path(&self.inner.cache, mark_type, icon_url);
        if path.is_file() {
            self.inner
                .state
                .lock()
                .ready_icons
                .insert(mark_type, path.clone());
            return Some(path);
        }
        self.enqueue(BwikiJob::DownloadIcon {
            mark_type,
            icon_url: icon_url.to_owned(),
        });
        None
    }

    #[must_use]
    pub fn ensure_icon_path_by_name(&self, icon_name: &str) -> Option<PathBuf> {
        let definition = self.resolve_icon_definition(icon_name)?;
        self.ensure_icon_path(definition.mark_type, &definition.icon_url)
    }

    #[must_use]
    pub fn ensure_tile_path(&self, zoom: u8, x: i32, y: i32) -> Option<PathBuf> {
        if !tile_is_inside_bounds(zoom, x, y) {
            return None;
        }

        let path = tile_cache_path(&self.inner.cache, zoom, x, y);
        if path.is_file() {
            return Some(path);
        }
        self.enqueue(BwikiJob::DownloadTile { zoom, x, y });
        None
    }

    pub fn ensure_tile_ready_blocking(&self, zoom: u8, x: i32, y: i32) -> Result<PathBuf> {
        let path = ensure_tile_blocking(&self.inner.cache.root, zoom, x, y)?;
        Ok(path)
    }

    fn enqueue(&self, job: BwikiJob) {
        let key = job.key();
        let mut state = self.inner.state.lock();
        if state.queued_jobs.contains(&key) {
            return;
        }
        state.queued_jobs.insert(key.clone());
        debug!(job = ?key, "queued BWiki background job");
        if let Err(error) = self.inner.job_tx.send(job) {
            state.last_error = Some(format!("failed to queue BWiki job: {error}"));
            error!(job = ?key, error = %error, "failed to queue BWiki background job");
        }
    }
}

pub fn default_map_dimensions() -> MapDimensions {
    let world = zoom_world_bounds(BWIKI_WORLD_ZOOM).expect("missing z8 range");
    MapDimensions {
        width: world.world_width(),
        height: world.world_height(),
    }
}

#[must_use]
pub fn zoom_world_bounds(zoom: u8) -> Option<BwikiTileZoom> {
    BWIKI_TILE_ZOOMS
        .iter()
        .copied()
        .find(|item| item.zoom == zoom)
}

#[must_use]
fn world_origin() -> WorldPoint {
    let world = zoom_world_bounds(BWIKI_WORLD_ZOOM).expect("missing z8 range");
    WorldPoint::new(
        (-world.min_x) as f32 * TILE_SIZE as f32,
        (-world.min_y) as f32 * TILE_SIZE as f32,
    )
}

#[must_use]
pub fn tile_coordinate_to_world_origin(zoom: u8, x: i32, y: i32) -> Option<WorldPoint> {
    let range = zoom_world_bounds(zoom)?;
    let origin = world_origin();
    let world_tile_size = range.world_tile_size() as f32;
    Some(WorldPoint::new(
        x as f32 * world_tile_size + origin.x,
        y as f32 * world_tile_size + origin.y,
    ))
}

#[must_use]
pub fn world_to_tile_coordinate(zoom: u8, world: WorldPoint) -> Option<(i32, i32)> {
    let range = zoom_world_bounds(zoom)?;
    let origin = world_origin();
    let world_tile_size = range.world_tile_size() as f32;
    Some((
        ((world.x - origin.x) / world_tile_size).floor() as i32,
        ((world.y - origin.y) / world_tile_size).floor() as i32,
    ))
}

#[must_use]
pub fn preferred_display_tile_zoom(camera_zoom: f32) -> u8 {
    let min_zoom = BWIKI_TILE_ZOOMS
        .iter()
        .map(|item| item.zoom)
        .min()
        .unwrap_or(BWIKI_WORLD_ZOOM);
    let max_zoom = BWIKI_TILE_ZOOMS
        .iter()
        .map(|item| item.zoom)
        .max()
        .unwrap_or(BWIKI_WORLD_ZOOM);
    let target = BWIKI_WORLD_ZOOM as i32 + camera_zoom.max(0.0001).log2().round() as i32;
    target.clamp(i32::from(min_zoom), i32::from(max_zoom)) as u8
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BwikiVisibleTile {
    pub zoom: u8,
    pub x: i32,
    pub y: i32,
    pub world_size: u32,
}

#[derive(Debug, Clone)]
pub struct BwikiVisibleTileLayer {
    pub zoom: u8,
    pub tiles: Vec<BwikiVisibleTile>,
}

#[must_use]
pub fn visible_tile_layers(
    camera: crate::domain::geometry::MapCamera,
    viewport: crate::domain::geometry::ViewportSize,
    prefetch_tiles: i32,
) -> Vec<BwikiVisibleTileLayer> {
    if !viewport.is_valid() {
        return Vec::new();
    }

    let preferred_zoom = preferred_display_tile_zoom(camera.zoom);
    let mut zooms = vec![preferred_zoom];
    if let Some(fallback_zoom) = previous_supported_zoom(preferred_zoom) {
        zooms.insert(0, fallback_zoom);
    }

    zooms
        .into_iter()
        .filter_map(|zoom| visible_tiles_for_zoom(camera, viewport, zoom, prefetch_tiles))
        .collect()
}

#[must_use]
pub fn raw_coordinate_to_world(raw_lat: i32, raw_lng: i32) -> WorldPoint {
    let scale = (1u32 << (BWIKI_WORLD_ZOOM - 7)) as f32;
    let origin = world_origin();
    WorldPoint::new(
        raw_lng as f32 * scale + origin.x,
        raw_lat as f32 * scale + origin.y,
    )
}

pub fn ensure_tile_blocking(cache_root: &Path, zoom: u8, x: i32, y: i32) -> Result<PathBuf> {
    let cache = BwikiCachePaths::new(cache_root.to_path_buf());
    cache.ensure_directories()?;
    download_tile(&cache, zoom, x, y)
}

pub fn load_logic_map_scaled_image(cache_root: &Path, scale: u32) -> Result<GrayImage> {
    let cache = BwikiCachePaths::new(cache_root.to_path_buf());
    cache.ensure_directories()?;
    let range = preferred_logic_zoom(scale);
    let output_tile_size = (range.world_tile_size() / scale.max(1)).max(1);
    assemble_gray_map(&cache, range, output_tile_size)
}

pub fn load_logic_map_with_tracking_poi_scaled_image(
    cache_root: &Path,
    scale: u32,
    view_size: u32,
) -> Result<GrayImage> {
    let mut image = load_logic_map_scaled_image(cache_root, scale)?;
    overlay_tracking_poi_icons(cache_root, scale, view_size, &mut image)?;
    Ok(image)
}

fn overlay_tracking_poi_icons(
    cache_root: &Path,
    scale: u32,
    view_size: u32,
    image: &mut GrayImage,
) -> Result<()> {
    if view_size == 0 {
        return Ok(());
    }

    let manager = BwikiResourceManager::new(cache_root.to_path_buf())
        .context("failed to create BWiki resource manager for tracking POI overlays")?;
    let dataset = wait_for_tracking_dataset(&manager)?;
    let icon_diameter = (((view_size as f32) / 7.0) / scale.max(1) as f32)
        .round()
        .max(6.0) as u32;
    let mut icons = HashMap::<u32, RgbaImage>::new();

    for mark_type in TRACKING_POI_MARK_TYPES {
        let Some(definition) = dataset.type_by_mark_type(mark_type).cloned() else {
            bail!("missing BWiki tracking POI definition for mark type {mark_type}");
        };
        let Some(points) = dataset.points_by_type.get(&mark_type) else {
            continue;
        };
        if points.is_empty() {
            continue;
        }

        let path = wait_for_tracking_icon_path(&manager, &definition)?;
        let icon = if let Some(icon) = icons.get(&mark_type) {
            icon
        } else {
            let loaded = image::open(&path)
                .with_context(|| format!("failed to open cached BWiki icon {}", path.display()))?
                .to_rgba8();
            icons.entry(mark_type).or_insert(loaded)
        };

        for point in points {
            overlay_tracking_icon(image, icon, point.world, scale, icon_diameter);
        }
    }

    Ok(())
}

fn wait_for_tracking_dataset(manager: &BwikiResourceManager) -> Result<Arc<BwikiDataset>> {
    let started = SystemTime::now();
    loop {
        manager.ensure_dataset_loaded();
        if let Some(dataset) = manager.dataset_snapshot() {
            return Ok(dataset);
        }
        let elapsed = SystemTime::now()
            .duration_since(started)
            .unwrap_or_default();
        if elapsed >= TRACKING_POI_ICON_WAIT_TIMEOUT {
            let last_error = manager
                .last_error()
                .unwrap_or_else(|| "no bwiki worker error reported".to_owned());
            bail!("timed out waiting for BWiki dataset; last_error={last_error}");
        }
        thread::sleep(TRACKING_POI_ICON_POLL_INTERVAL);
    }
}

fn wait_for_tracking_icon_path(
    manager: &BwikiResourceManager,
    definition: &BwikiTypeDefinition,
) -> Result<PathBuf> {
    let started = SystemTime::now();
    loop {
        if let Some(path) = manager.ensure_icon_path(definition.mark_type, &definition.icon_url) {
            return Ok(path);
        }
        let elapsed = SystemTime::now()
            .duration_since(started)
            .unwrap_or_default();
        if elapsed >= TRACKING_POI_ICON_WAIT_TIMEOUT {
            let last_error = manager
                .last_error()
                .unwrap_or_else(|| "no bwiki worker error reported".to_owned());
            bail!(
                "timed out waiting for cached tracking POI icon {} ({}) at mark type {}; last_error={last_error}",
                definition.name,
                definition.icon_url,
                definition.mark_type
            );
        }
        thread::sleep(TRACKING_POI_ICON_POLL_INTERVAL);
    }
}

fn overlay_tracking_icon(
    image: &mut GrayImage,
    icon: &RgbaImage,
    world: WorldPoint,
    scale: u32,
    diameter: u32,
) {
    if diameter == 0 {
        return;
    }

    let resized = resize(icon, diameter, diameter, FilterType::CatmullRom);
    let center_x = (world.x / scale.max(1) as f32).round() as i32;
    let center_y = (world.y / scale.max(1) as f32).round() as i32;
    let left = center_x - diameter as i32 / 2;
    let top = center_y - diameter as i32 / 2;

    for (x, y, pixel) in resized.enumerate_pixels() {
        let dst_x = left + x as i32;
        let dst_y = top + y as i32;
        if dst_x < 0 || dst_y < 0 || dst_x >= image.width() as i32 || dst_y >= image.height() as i32
        {
            continue;
        }

        let alpha = f32::from(pixel.0[3]) / 255.0;
        if alpha <= 0.0 {
            continue;
        }

        let source = tracking_icon_luma(pixel.0[0], pixel.0[1], pixel.0[2]);
        let base = f32::from(image.get_pixel(dst_x as u32, dst_y as u32).0[0]);
        let blended = (base * (1.0 - alpha) + f32::from(source) * alpha)
            .round()
            .clamp(0.0, 255.0) as u8;
        image.put_pixel(dst_x as u32, dst_y as u32, Luma([blended]));
    }
}

fn tracking_icon_luma(r: u8, g: u8, b: u8) -> u8 {
    let original = ((u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) / 256) as u8;
    let muted = (108u32 + ((u32::from(original) * 72) / 255)).min(u32::from(u8::MAX)) as u8;
    ((u16::from(original) + u16::from(muted)) / 2) as u8
}

fn run_bwiki_worker(inner: Arc<BwikiManagerInner>, job_rx: Receiver<BwikiJob>) {
    while let Ok(job) = job_rx.recv() {
        let key = job.key();
        debug!(job = ?key, "running BWiki background job");
        let result = match job {
            BwikiJob::RefreshDataset => refresh_dataset_job(&inner),
            BwikiJob::DownloadIcon {
                mark_type,
                icon_url,
            } => download_icon_job(&inner.cache, mark_type, &icon_url),
            BwikiJob::DownloadTile { zoom, x, y } => download_tile_job(&inner.cache, zoom, x, y),
        };

        let mut state = inner.state.lock();
        state.queued_jobs.remove(&key);
        match result {
            Ok(dataset) => {
                if let Some(dataset) = dataset {
                    info!(
                        type_count = dataset.types.len(),
                        point_count = dataset.total_point_count(),
                        "BWiki dataset refresh completed"
                    );
                    state.dataset = Some(Arc::new(dataset));
                }
                state.last_error = None;
                inner.version.fetch_add(1, Ordering::SeqCst);
            }
            Err(error) => {
                warn!(job = ?key, error = %error, "BWiki background job failed");
                state.last_error = Some(format!("{error:#}"));
                inner.version.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

fn refresh_dataset_job(inner: &BwikiManagerInner) -> Result<Option<BwikiDataset>> {
    info!("refreshing remote BWiki dataset");
    let dataset = fetch_remote_dataset()?;
    let cache_path = dataset_cache_path(&inner.cache);
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create BWiki data cache directory {}",
                parent.display()
            )
        })?;
    }
    let body =
        serde_json::to_string_pretty(&dataset).context("failed to serialize BWiki dataset")?;
    fs::write(&cache_path, body).with_context(|| {
        format!(
            "failed to write BWiki dataset cache {}",
            cache_path.display()
        )
    })?;
    info!(
        cache_path = %cache_path.display(),
        type_count = dataset.types.len(),
        point_count = dataset.total_point_count(),
        "persisted refreshed BWiki dataset"
    );
    Ok(Some(dataset))
}

fn download_icon_job(
    cache: &BwikiCachePaths,
    mark_type: u32,
    icon_url: &str,
) -> Result<Option<BwikiDataset>> {
    download_icon(cache, mark_type, icon_url)?;
    Ok(None)
}

fn download_tile_job(
    cache: &BwikiCachePaths,
    zoom: u8,
    x: i32,
    y: i32,
) -> Result<Option<BwikiDataset>> {
    download_tile(cache, zoom, x, y)?;
    Ok(None)
}

fn fetch_remote_dataset() -> Result<BwikiDataset> {
    let type_html = fetch_remote_text(TYPE_PARSE_URL)?;
    let point_html = fetch_remote_text(POINT_PARSE_URL)?;

    let type_text = decode_html_entities(&strip_html_tags(&type_html));
    let raw_types = serde_json::from_str::<RawTypeCollection>(&type_text)
        .context("failed to parse BWiki type catalog")?;

    let mut point_text = decode_html_entities(&strip_html_tags(&point_html));
    point_text = scrub_data_page_references(&point_text);
    point_text = quote_numeric_object_keys(&point_text);

    let points_raw = serde_json::from_str::<BTreeMap<String, Vec<RawPointDefinition>>>(&point_text)
        .context("failed to parse BWiki point catalog")?;

    let mut points_by_type = BTreeMap::new();
    for (mark_type, entries) in points_raw {
        let mark_type = mark_type
            .parse::<u32>()
            .with_context(|| format!("invalid point mark type key {mark_type}"))?;
        let normalized = entries
            .into_iter()
            .map(|entry| BwikiPointRecord {
                mark_type: entry.mark_type,
                title: entry.title,
                id: entry.id,
                raw_lat: entry.point.lat,
                raw_lng: entry.point.lng,
                world: raw_coordinate_to_world(entry.point.lat, entry.point.lng),
                uid: entry.uid,
                layer: entry.layer,
                time: entry.time,
                version: entry.version,
            })
            .collect::<Vec<_>>();
        points_by_type.insert(mark_type, normalized);
    }

    let mut types = raw_types
        .data
        .into_iter()
        .map(|entry| BwikiTypeDefinition {
            category: entry.category,
            mark_type: entry.mark_type,
            name: entry.name,
            icon_url: normalize_icon_url(&entry.icon),
            point_count: points_by_type.get(&entry.mark_type).map_or(0, Vec::len),
            type_known: true,
        })
        .collect::<Vec<_>>();

    let known = types
        .iter()
        .map(|item| item.mark_type)
        .collect::<HashSet<_>>();
    for (mark_type, points) in &points_by_type {
        if known.contains(mark_type) {
            continue;
        }
        types.push(BwikiTypeDefinition {
            category: "未分类".to_owned(),
            mark_type: *mark_type,
            name: format!("未登记类型 {}", mark_type),
            icon_url: String::new(),
            point_count: points.len(),
            type_known: false,
        });
    }

    types.sort_by(|left, right| {
        left.category
            .cmp(&right.category)
            .then(left.mark_type.cmp(&right.mark_type))
    });

    Ok(BwikiDataset {
        fetched_at_epoch_secs: unix_now_secs(),
        types,
        points_by_type,
    })
}

fn fetch_remote_text(url: &str) -> Result<String> {
    let agent = Agent::new();
    let response = agent
        .get(url)
        .set("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("failed to fetch BWiki endpoint {url}"))?;
    let mut reader = response.into_reader();
    let mut body = String::new();
    reader
        .read_to_string(&mut body)
        .with_context(|| format!("failed to read BWiki response body {url}"))?;
    let parsed = serde_json::from_str::<ParseResponse>(&body)
        .with_context(|| format!("failed to decode BWiki parse payload {url}"))?;
    Ok(parsed.parse.text)
}

fn load_dataset_from_cache(path: &Path) -> Result<BwikiDataset> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read cached BWiki dataset {}", path.display()))?;
    let mut dataset = serde_json::from_str::<BwikiDataset>(&raw)
        .with_context(|| format!("failed to parse cached BWiki dataset {}", path.display()))?;
    recalculate_dataset_world_points(&mut dataset);
    Ok(dataset)
}

fn cache_is_stale(path: &Path, ttl: Duration) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return true;
    };
    let Ok(modified) = metadata.modified() else {
        return true;
    };
    let Ok(age) = SystemTime::now().duration_since(modified) else {
        return true;
    };
    age > ttl
}

fn dataset_cache_path(cache: &BwikiCachePaths) -> PathBuf {
    cache.data_dir.join("dataset.json")
}

fn icon_cache_path(cache: &BwikiCachePaths, mark_type: u32, icon_url: &str) -> PathBuf {
    let extension = Path::new(icon_url)
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.trim().is_empty())
        .unwrap_or("png");
    cache.icons_dir.join(format!("{mark_type}.{extension}"))
}

fn tile_cache_path(cache: &BwikiCachePaths, zoom: u8, x: i32, y: i32) -> PathBuf {
    cache
        .tiles_dir
        .join(format!("z{zoom}"))
        .join(format!("tile-{x}_{y}.png"))
}

fn tile_url(zoom: u8, x: i32, y: i32) -> String {
    TILE_URL_TEMPLATE
        .replace("{z}", &zoom.to_string())
        .replace("{x}", &x.to_string())
        .replace("{y}", &y.to_string())
}

fn tile_is_inside_bounds(zoom: u8, x: i32, y: i32) -> bool {
    zoom_world_bounds(zoom).is_some_and(|range| {
        x >= range.min_x && x <= range.max_x && y >= range.min_y && y <= range.max_y
    })
}

fn download_tile(cache: &BwikiCachePaths, zoom: u8, x: i32, y: i32) -> Result<PathBuf> {
    if !tile_is_inside_bounds(zoom, x, y) {
        bail!("requested tile z={zoom} x={x} y={y} is outside configured bounds");
    }

    let target = tile_cache_path(cache, zoom, x, y);
    if target.is_file() {
        return Ok(target);
    }
    debug!(zoom, x, y, target = %target.display(), "downloading BWiki tile");

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create BWiki tile cache directory {}",
                parent.display()
            )
        })?;
    }

    let bytes = download_binary(&tile_url(zoom, x, y))?;
    let temp = target.with_extension(format!("{}.part", std::process::id()));
    fs::write(&temp, bytes)
        .with_context(|| format!("failed to write BWiki tile temp file {}", temp.display()))?;
    match fs::rename(&temp, &target) {
        Ok(()) => {}
        Err(error) if target.is_file() => {
            let _ = fs::remove_file(&temp);
            let _ = error;
        }
        Err(error) => {
            let _ = fs::remove_file(&temp);
            return Err(error).with_context(|| {
                format!(
                    "failed to finalize BWiki tile {} from {}",
                    target.display(),
                    temp.display()
                )
            });
        }
    }
    debug!(zoom, x, y, target = %target.display(), "downloaded BWiki tile");
    Ok(target)
}

fn download_icon(cache: &BwikiCachePaths, mark_type: u32, icon_url: &str) -> Result<PathBuf> {
    if icon_url.trim().is_empty() {
        bail!("icon URL is empty for mark type {mark_type}");
    }

    let target = icon_cache_path(cache, mark_type, icon_url);
    if target.is_file() {
        return Ok(target);
    }
    debug!(
        mark_type,
        icon_url,
        target = %target.display(),
        "downloading BWiki icon"
    );

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create icon cache directory {}", parent.display())
        })?;
    }

    let bytes = download_binary(icon_url)?;
    fs::write(&target, bytes)
        .with_context(|| format!("failed to write icon cache {}", target.display()))?;
    debug!(mark_type, target = %target.display(), "downloaded BWiki icon");
    Ok(target)
}

fn download_binary(url: &str) -> Result<Vec<u8>> {
    let agent = Agent::new();
    let response = agent
        .get(url)
        .set("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("failed to download remote asset {url}"))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read remote asset body {url}"))?;
    Ok(bytes)
}

fn assemble_gray_map(
    cache: &BwikiCachePaths,
    range: BwikiTileZoom,
    output_tile_size: u32,
) -> Result<GrayImage> {
    let width = range.width_tiles() * output_tile_size.max(1);
    let height = range.height_tiles() * output_tile_size.max(1);
    let mut canvas = GrayImage::new(width, height);

    for tile_y in range.min_y..=range.max_y {
        for tile_x in range.min_x..=range.max_x {
            let tile_path = download_tile(cache, range.zoom, tile_x, tile_y)?;
            let tile = image::open(&tile_path)
                .with_context(|| format!("failed to open BWiki tile {}", tile_path.display()))?
                .into_luma8();
            let tile = if output_tile_size == TILE_SIZE {
                tile
            } else {
                resize(
                    &tile,
                    output_tile_size.max(1),
                    output_tile_size.max(1),
                    FilterType::Triangle,
                )
            };
            let dx = (tile_x - range.min_x) as i64 * output_tile_size as i64;
            let dy = (tile_y - range.min_y) as i64 * output_tile_size as i64;
            replace(&mut canvas, &tile, dx, dy);
        }
    }

    Ok(canvas)
}

fn preferred_logic_zoom(scale: u32) -> BwikiTileZoom {
    let scale = scale.max(1);
    BWIKI_TILE_ZOOMS
        .iter()
        .copied()
        .filter(|item| item.source_world_pixel_size() <= scale)
        .max_by_key(|item| item.zoom)
        .unwrap_or_else(|| BWIKI_TILE_ZOOMS[0])
}

fn previous_supported_zoom(zoom: u8) -> Option<u8> {
    BWIKI_TILE_ZOOMS
        .iter()
        .filter_map(|item| (item.zoom < zoom).then_some(item.zoom))
        .max()
}

fn visible_tiles_for_zoom(
    camera: crate::domain::geometry::MapCamera,
    viewport: crate::domain::geometry::ViewportSize,
    zoom: u8,
    prefetch_tiles: i32,
) -> Option<BwikiVisibleTileLayer> {
    let range = zoom_world_bounds(zoom)?;
    let top_left = camera.screen_to_world(WorldPoint::new(0.0, 0.0));
    let bottom_right = camera.screen_to_world(WorldPoint::new(viewport.width, viewport.height));
    let min_world = WorldPoint::new(
        top_left.x.min(bottom_right.x),
        top_left.y.min(bottom_right.y),
    );
    let max_world = WorldPoint::new(
        top_left.x.max(bottom_right.x),
        top_left.y.max(bottom_right.y),
    );
    let (mut min_x, mut min_y) = world_to_tile_coordinate(zoom, min_world)?;
    let (mut max_x, mut max_y) = world_to_tile_coordinate(zoom, max_world)?;
    min_x = (min_x - prefetch_tiles).clamp(range.min_x, range.max_x);
    max_x = (max_x + prefetch_tiles).clamp(range.min_x, range.max_x);
    min_y = (min_y - prefetch_tiles).clamp(range.min_y, range.max_y);
    max_y = (max_y + prefetch_tiles).clamp(range.min_y, range.max_y);

    let mut tiles = Vec::with_capacity(((max_x - min_x + 1) * (max_y - min_y + 1)) as usize);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            tiles.push(BwikiVisibleTile {
                zoom,
                x,
                y,
                world_size: range.world_tile_size(),
            });
        }
    }

    Some(BwikiVisibleTileLayer { zoom, tiles })
}

fn strip_html_tags(input: &str) -> String {
    static TAG_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex = TAG_RE.get_or_init(|| Regex::new(r"<[^>]+>").expect("valid html strip regex"));
    regex.replace_all(input, "").trim().to_owned()
}

fn decode_html_entities(input: &str) -> String {
    input
        .replace("&quot;", "\"")
        .replace("&#34;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
}

fn scrub_data_page_references(input: &str) -> String {
    static DATA_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex =
        DATA_RE.get_or_init(|| Regex::new(r":Data:[\s\S]{0,30}?/json").expect("valid data regex"));
    regex.replace_all(input, ":[]").to_string()
}

fn quote_numeric_object_keys(input: &str) -> String {
    static KEY_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex = KEY_RE
        .get_or_init(|| Regex::new(r#"([\{,]\s*)(\d+)(\s*:)"#).expect("valid numeric key regex"));
    regex.replace_all(input, "$1\"$2\"$3").to_string()
}

fn normalize_icon_url(input: &str) -> String {
    static URL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex =
        URL_RE.get_or_init(|| Regex::new(r#"https?://[^\s"]+"#).expect("valid icon url regex"));
    regex
        .find(input.trim())
        .map(|value| value.as_str().trim_end_matches('"').to_owned())
        .unwrap_or_else(|| input.trim().to_owned())
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn recalculate_dataset_world_points(dataset: &mut BwikiDataset) {
    for points in dataset.points_by_type.values_mut() {
        for point in points {
            point.world = raw_coordinate_to_world(point.raw_lat, point.raw_lng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BWIKI_WORLD_ZOOM, default_map_dimensions, preferred_display_tile_zoom,
        raw_coordinate_to_world, tile_coordinate_to_world_origin, world_to_tile_coordinate,
    };

    #[test]
    fn raw_coordinate_to_world_matches_leaflet_simple_projection() {
        let world = raw_coordinate_to_world(1387, -1457);
        assert_eq!(world.x, 3230.0);
        assert_eq!(world.y, 7382.0);
    }

    #[test]
    fn origin_matches_world_tile_anchor() {
        let world = raw_coordinate_to_world(0, 0);
        assert_eq!(world.x, 6144.0);
        assert_eq!(world.y, 4608.0);
    }

    #[test]
    fn coarse_tiles_stay_anchored_to_same_world_origin() {
        let world = tile_coordinate_to_world_origin(4, -2, -2).expect("z4 tile origin");
        assert_eq!(world.x, -2048.0);
        assert_eq!(world.y, -3584.0);

        let z8_top_left =
            tile_coordinate_to_world_origin(BWIKI_WORLD_ZOOM, -24, -18).expect("z8 tile origin");
        assert_eq!(z8_top_left.x, 0.0);
        assert_eq!(z8_top_left.y, 0.0);
    }

    #[test]
    fn point_stays_in_same_leaflet_tile_cell_as_browser_projection() {
        let point = raw_coordinate_to_world(1387, -1457);
        let tile =
            tile_coordinate_to_world_origin(BWIKI_WORLD_ZOOM, -12, 10).expect("z8 tile origin");
        assert_eq!(point.x - tile.x, 158.0);
        assert_eq!(point.y - tile.y, 214.0);
        assert_eq!(
            world_to_tile_coordinate(BWIKI_WORLD_ZOOM, point),
            Some((-12, 10))
        );
    }

    #[test]
    fn known_dataset_bounds_stay_inside_world() {
        let dims = default_map_dimensions();
        for (lat, lng) in [(-2132, -3015), (-2132, 2817), (2061, -3015), (2061, 2817)] {
            let world = raw_coordinate_to_world(lat, lng);
            assert!(world.x >= 0.0 && world.x <= dims.width as f32);
            assert!(world.y >= 0.0 && world.y <= dims.height as f32);
        }
    }

    #[test]
    fn preferred_display_zoom_tracks_screen_density() {
        assert_eq!(preferred_display_tile_zoom(1.0), 8);
        assert_eq!(preferred_display_tile_zoom(0.5), 7);
        assert_eq!(preferred_display_tile_zoom(0.25), 6);
        assert_eq!(preferred_display_tile_zoom(0.125), 5);
    }
}
