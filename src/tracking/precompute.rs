use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, Weak},
    time::UNIX_EPOCH,
};

use image::{GrayImage, RgbaImage};
use imageproc::contrast::equalize_histogram;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    error::{ContextExt as _, Result},
    resources::{
        BWIKI_WORLD_ZOOM, BwikiCachePaths, WorkspaceSnapshot,
        load_logic_map_with_tracking_poi_scaled_image,
        load_logic_map_with_tracking_poi_scaled_rgba_image, zoom_world_bounds,
    },
    tracking::vision::{
        ColorMapPyramid, MapPyramid, ScaledColorMap, ScaledMap, build_match_representation,
        coarse_global_downscale, downscale_gray, downscale_rgba,
    },
};

const MATCH_PYRAMID_CACHE_VERSION: u32 = 3;
const COLOR_PYRAMID_CACHE_VERSION: u32 = 1;
const TENSOR_CACHE_FORMAT_VERSION: u32 = 2;
const TRACKER_SOURCE_HASH_CACHE_VERSION: u32 = 1;
const TENSOR_CACHE_MAGIC: [u8; 8] = *b"GMTRTC01";
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

static COLOR_PYRAMID_MEMORY_CACHE: OnceLock<Mutex<HashMap<String, Weak<ColorMapPyramid>>>> =
    OnceLock::new();
static TRACKER_SOURCE_HASH_MEMORY_CACHE: OnceLock<Mutex<HashMap<String, TrackerSourceHashes>>> =
    OnceLock::new();

#[derive(Debug, Clone)]
pub struct PreparedMatchPyramid {
    pub cache_key: String,
    pub pyramid: MapPyramid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackerSourceHashes {
    fast_fingerprint: String,
    tile_hash: String,
    poi_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedTrackerSourceHashes {
    version: u32,
    fast_fingerprint: String,
    tile_hash: String,
    poi_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedTensorCache {
    pub width: u32,
    pub height: u32,
    pub primary_channels: usize,
    pub secondary_channels: usize,
    pub primary: Vec<f32>,
    pub secondary: Vec<f32>,
}

impl PersistedTensorCache {
    pub fn from_parts(
        width: u32,
        height: u32,
        primary_channels: usize,
        secondary_channels: usize,
        primary: Vec<f32>,
        secondary: Vec<f32>,
    ) -> Result<Self> {
        let expected_primary = element_count(width, height, primary_channels)?;
        if primary.len() != expected_primary {
            crate::bail!(
                "persisted tensor primary buffer length {} does not match expected {}",
                primary.len(),
                expected_primary
            );
        }
        let expected_secondary = element_count(width, height, secondary_channels)?;
        if secondary.len() != expected_secondary {
            crate::bail!(
                "persisted tensor secondary buffer length {} does not match expected {}",
                secondary.len(),
                expected_secondary
            );
        }

        Ok(Self {
            width,
            height,
            primary_channels,
            secondary_channels,
            primary,
            secondary,
        })
    }
}

pub fn load_or_build_match_pyramid(workspace: &WorkspaceSnapshot) -> Result<PreparedMatchPyramid> {
    let local_scale = workspace.config.template.local_downscale.max(1);
    let global_scale = workspace.config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(&workspace.config);
    let initial_key = match_pyramid_cache_key(
        &workspace.assets.bwiki_cache_dir,
        workspace.config.view_size,
        local_scale,
        global_scale,
        coarse_scale,
    )?;

    if let Ok(Some(pyramid)) = load_cached_match_pyramid(
        workspace,
        &initial_key,
        local_scale,
        global_scale,
        coarse_scale,
    ) {
        return Ok(PreparedMatchPyramid {
            cache_key: initial_key,
            pyramid,
        });
    }

    let base_map = load_logic_map_with_tracking_poi_scaled_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        workspace.config.view_size,
    )
    .with_context(|| {
        format!(
            "failed to load augmented BWiki logic tiles from {}",
            workspace.assets.bwiki_cache_dir.display()
        )
    })?;
    let base_map = equalize_histogram(&base_map);
    let local_map = build_match_representation(&downscale_gray(&base_map, local_scale));
    let global_map = if global_scale == local_scale {
        local_map.clone()
    } else {
        build_match_representation(&downscale_gray(&base_map, global_scale))
    };
    let coarse_map = if coarse_scale == global_scale {
        global_map.clone()
    } else {
        build_match_representation(&downscale_gray(&base_map, coarse_scale))
    };

    let final_key = match_pyramid_cache_key(
        &workspace.assets.bwiki_cache_dir,
        workspace.config.view_size,
        local_scale,
        global_scale,
        coarse_scale,
    )?;
    let pyramid = MapPyramid {
        local: ScaledMap {
            scale: local_scale,
            image: local_map,
        },
        global: ScaledMap {
            scale: global_scale,
            image: global_map,
        },
        coarse: ScaledMap {
            scale: coarse_scale,
            image: coarse_map,
        },
    };

    let _ = persist_match_pyramid(workspace, &final_key, &pyramid);

    Ok(PreparedMatchPyramid {
        cache_key: final_key,
        pyramid,
    })
}

pub fn tracker_map_cache_key(workspace: &WorkspaceSnapshot) -> Result<String> {
    let local_scale = workspace.config.template.local_downscale.max(1);
    let global_scale = workspace.config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(&workspace.config);
    match_pyramid_cache_key(
        &workspace.assets.bwiki_cache_dir,
        workspace.config.view_size,
        local_scale,
        global_scale,
        coarse_scale,
    )
}

pub fn color_pyramid_cache_key(map_cache_key: &str) -> String {
    format!("cv{COLOR_PYRAMID_CACHE_VERSION}-{map_cache_key}")
}

pub fn load_or_build_color_map_pyramid(
    workspace: &WorkspaceSnapshot,
    map_cache_key: &str,
) -> Result<Arc<ColorMapPyramid>> {
    let local_scale = workspace.config.template.local_downscale.max(1);
    let global_scale = workspace.config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(&workspace.config);
    let cache_key = color_pyramid_cache_key(map_cache_key);
    let memory_key = workspace_memory_cache_key(workspace, "color-pyramids", &cache_key);

    if let Some(pyramid) = {
        let mut cache = color_pyramid_memory_cache().lock();
        match cache
            .get(&memory_key)
            .cloned()
            .and_then(|entry| entry.upgrade())
        {
            Some(pyramid) => Some(pyramid),
            None => {
                cache.remove(&memory_key);
                None
            }
        }
    } {
        return Ok(pyramid);
    }

    if let Ok(Some(pyramid)) = load_cached_color_map_pyramid(
        workspace,
        &cache_key,
        local_scale,
        global_scale,
        coarse_scale,
    ) {
        let pyramid = Arc::new(pyramid);
        color_pyramid_memory_cache()
            .lock()
            .insert(memory_key, Arc::downgrade(&pyramid));
        return Ok(pyramid);
    }

    let base_map = load_logic_map_with_tracking_poi_scaled_rgba_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        workspace.config.view_size,
    )
    .with_context(|| {
        format!(
            "failed to load augmented color BWiki logic tiles from {}",
            workspace.assets.bwiki_cache_dir.display()
        )
    })?;
    let local_map = downscale_rgba(&base_map, local_scale);
    let global_map = if global_scale == local_scale {
        local_map.clone()
    } else {
        downscale_rgba(&base_map, global_scale)
    };
    let coarse_map = if coarse_scale == global_scale {
        global_map.clone()
    } else {
        downscale_rgba(&base_map, coarse_scale)
    };

    let pyramid = ColorMapPyramid {
        local: ScaledColorMap {
            scale: local_scale,
            image: local_map,
        },
        global: ScaledColorMap {
            scale: global_scale,
            image: global_map,
        },
        coarse: ScaledColorMap {
            scale: coarse_scale,
            image: coarse_map,
        },
    };

    let _ = persist_color_map_pyramid(workspace, &cache_key, &pyramid);
    let pyramid = Arc::new(pyramid);
    color_pyramid_memory_cache()
        .lock()
        .insert(memory_key, Arc::downgrade(&pyramid));
    Ok(pyramid)
}

pub fn tracker_tensor_cache_path(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    cache_key: &str,
) -> PathBuf {
    tracker_tensor_cache_path_for_version(workspace, prefix, cache_key, TENSOR_CACHE_FORMAT_VERSION)
}

pub fn tracker_legacy_tensor_cache_path(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    cache_key: &str,
) -> PathBuf {
    tracker_tensor_cache_path_for_version(workspace, prefix, cache_key, 1)
}

fn tracker_tensor_cache_path_for_version(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    cache_key: &str,
    format_version: u32,
) -> PathBuf {
    tracker_cache_root(workspace)
        .join("tensors")
        .join(format!("{prefix}-tv{format_version}-{cache_key}.bin"))
}

pub fn load_tensor_cache(path: &Path) -> Result<Option<PersistedTensorCache>> {
    if !path.is_file() {
        return Ok(None);
    }

    let file = File::open(path)
        .with_context(|| format!("failed to open tracker tensor cache {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).with_context(|| {
        format!(
            "failed to read tracker tensor cache header {}",
            path.display()
        )
    })?;
    if magic != TENSOR_CACHE_MAGIC {
        crate::bail!(
            "unsupported tracker tensor cache magic in {}",
            path.display()
        );
    }

    let version = read_u32(&mut reader)?;
    let cache = match version {
        1 => {
            let width = read_u32(&mut reader)?;
            let height = read_u32(&mut reader)?;
            let channels = read_u32(&mut reader)? as usize;
            let primary_len = read_u64(&mut reader)? as usize;
            let secondary_len = read_u64(&mut reader)? as usize;
            let expected = element_count(width, height, channels)?;
            if primary_len != expected || secondary_len != expected {
                crate::bail!(
                    "tracker tensor cache payload length mismatch in {}",
                    path.display()
                );
            }

            let primary = read_f32_vec(&mut reader, primary_len)?;
            let secondary = read_f32_vec(&mut reader, secondary_len)?;
            PersistedTensorCache::from_parts(width, height, channels, channels, primary, secondary)?
        }
        2 => {
            let width = read_u32(&mut reader)?;
            let height = read_u32(&mut reader)?;
            let primary_channels = read_u32(&mut reader)? as usize;
            let secondary_channels = read_u32(&mut reader)? as usize;
            let primary_len = read_u64(&mut reader)? as usize;
            let secondary_len = read_u64(&mut reader)? as usize;
            let expected_primary = element_count(width, height, primary_channels)?;
            let expected_secondary = element_count(width, height, secondary_channels)?;
            if primary_len != expected_primary || secondary_len != expected_secondary {
                crate::bail!(
                    "tracker tensor cache payload length mismatch in {}",
                    path.display()
                );
            }

            let primary = read_f32_vec(&mut reader, primary_len)?;
            let secondary = read_f32_vec(&mut reader, secondary_len)?;
            PersistedTensorCache::from_parts(
                width,
                height,
                primary_channels,
                secondary_channels,
                primary,
                secondary,
            )?
        }
        _ => {
            crate::bail!(
                "unsupported tracker tensor cache version {} in {}",
                version,
                path.display()
            );
        }
    };

    Ok(Some(cache))
}

pub fn save_tensor_cache(path: &Path, cache: &PersistedTensorCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create tracker tensor cache directory {}",
                parent.display()
            )
        })?;
    }

    let temp_path = cache_temp_path(path);
    let file = File::create(&temp_path).with_context(|| {
        format!(
            "failed to create tracker tensor temp cache {}",
            temp_path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&TENSOR_CACHE_MAGIC)?;
    write_u32(&mut writer, TENSOR_CACHE_FORMAT_VERSION)?;
    write_u32(&mut writer, cache.width)?;
    write_u32(&mut writer, cache.height)?;
    write_u32(&mut writer, cache.primary_channels as u32)?;
    write_u32(&mut writer, cache.secondary_channels as u32)?;
    write_u64(&mut writer, cache.primary.len() as u64)?;
    write_u64(&mut writer, cache.secondary.len() as u64)?;
    write_f32_slice(&mut writer, &cache.primary)?;
    write_f32_slice(&mut writer, &cache.secondary)?;
    writer.flush()?;
    drop(writer);

    replace_cache_file(&temp_path, path)?;
    Ok(())
}

pub fn metadata_fingerprint(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    let mut hash = FNV_OFFSET_BASIS;
    update_hash_bytes(&mut hash, path.to_string_lossy().as_bytes());
    update_hash_u64(&mut hash, metadata.len());
    if let Ok(modified) = metadata.modified() {
        if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
            update_hash_u64(&mut hash, duration.as_secs());
            update_hash_u64(&mut hash, duration.subsec_nanos() as u64);
        }
    }
    Ok(format!("{hash:016x}"))
}

pub fn clear_match_pyramid_caches(workspace: &WorkspaceSnapshot) -> Result<()> {
    let path = tracker_cache_root(workspace).join("pyramids");
    remove_cache_dir_if_exists(&path)
}

pub fn clear_color_pyramid_caches(workspace: &WorkspaceSnapshot) -> Result<()> {
    prune_color_pyramid_memory_cache(workspace);
    let path = tracker_cache_root(workspace).join("color-pyramids");
    remove_cache_dir_if_exists(&path)
}

pub fn clear_tracker_source_hash_caches(workspace: &WorkspaceSnapshot) -> Result<()> {
    prune_tracker_source_hash_memory_cache(&workspace.assets.bwiki_cache_dir);
    let path = tracker_cache_root(workspace).join("source-hashes");
    remove_cache_dir_if_exists(&path)
}

pub fn clear_tensor_caches_by_prefix(workspace: &WorkspaceSnapshot, prefix: &str) -> Result<usize> {
    let directory = tracker_cache_root(workspace).join("tensors");
    if !directory.is_dir() {
        return Ok(0);
    }

    let mut removed = 0usize;
    for entry in fs::read_dir(&directory).with_context(|| {
        format!(
            "failed to enumerate tracker tensor cache directory {}",
            directory.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "failed to read entry in tracker tensor cache directory {}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !file_name.starts_with(prefix) {
            continue;
        }

        fs::remove_file(&path)
            .with_context(|| format!("failed to remove tracker tensor cache {}", path.display()))?;
        removed += 1;
    }

    Ok(removed)
}

fn load_cached_match_pyramid(
    workspace: &WorkspaceSnapshot,
    cache_key: &str,
    local_scale: u32,
    global_scale: u32,
    coarse_scale: u32,
) -> Result<Option<MapPyramid>> {
    let cache_dir = match_pyramid_cache_dir(workspace, cache_key);
    let local_path = cache_dir.join("local.png");
    let global_path = cache_dir.join("global.png");
    let coarse_path = cache_dir.join("coarse.png");
    if !local_path.is_file() || !global_path.is_file() || !coarse_path.is_file() {
        return Ok(None);
    }

    let local_image = image::open(&local_path)
        .with_context(|| {
            format!(
                "failed to open tracker local pyramid {}",
                local_path.display()
            )
        })?
        .into_luma8();
    let global_image = image::open(&global_path)
        .with_context(|| {
            format!(
                "failed to open tracker global pyramid {}",
                global_path.display()
            )
        })?
        .into_luma8();
    let coarse_image = image::open(&coarse_path)
        .with_context(|| {
            format!(
                "failed to open tracker coarse pyramid {}",
                coarse_path.display()
            )
        })?
        .into_luma8();

    Ok(Some(MapPyramid {
        local: ScaledMap {
            scale: local_scale,
            image: local_image,
        },
        global: ScaledMap {
            scale: global_scale,
            image: global_image,
        },
        coarse: ScaledMap {
            scale: coarse_scale,
            image: coarse_image,
        },
    }))
}

fn load_cached_color_map_pyramid(
    workspace: &WorkspaceSnapshot,
    cache_key: &str,
    local_scale: u32,
    global_scale: u32,
    coarse_scale: u32,
) -> Result<Option<ColorMapPyramid>> {
    let cache_dir = color_pyramid_cache_dir(workspace, cache_key);
    let local_path = cache_dir.join("local.png");
    let global_path = cache_dir.join("global.png");
    let coarse_path = cache_dir.join("coarse.png");
    if !local_path.is_file() || !global_path.is_file() || !coarse_path.is_file() {
        return Ok(None);
    }

    let local_image = image::open(&local_path)
        .with_context(|| {
            format!(
                "failed to open tracker local color pyramid {}",
                local_path.display()
            )
        })?
        .to_rgba8();
    let global_image = image::open(&global_path)
        .with_context(|| {
            format!(
                "failed to open tracker global color pyramid {}",
                global_path.display()
            )
        })?
        .to_rgba8();
    let coarse_image = image::open(&coarse_path)
        .with_context(|| {
            format!(
                "failed to open tracker coarse color pyramid {}",
                coarse_path.display()
            )
        })?
        .to_rgba8();

    Ok(Some(ColorMapPyramid {
        local: ScaledColorMap {
            scale: local_scale,
            image: local_image,
        },
        global: ScaledColorMap {
            scale: global_scale,
            image: global_image,
        },
        coarse: ScaledColorMap {
            scale: coarse_scale,
            image: coarse_image,
        },
    }))
}

fn persist_match_pyramid(
    workspace: &WorkspaceSnapshot,
    cache_key: &str,
    pyramid: &MapPyramid,
) -> Result<()> {
    let cache_dir = match_pyramid_cache_dir(workspace, cache_key);
    fs::create_dir_all(&cache_dir).with_context(|| {
        format!(
            "failed to create tracker pyramid cache directory {}",
            cache_dir.display()
        )
    })?;
    save_gray_image(&cache_dir.join("local.png"), &pyramid.local.image)?;
    save_gray_image(&cache_dir.join("global.png"), &pyramid.global.image)?;
    save_gray_image(&cache_dir.join("coarse.png"), &pyramid.coarse.image)?;
    Ok(())
}

fn persist_color_map_pyramid(
    workspace: &WorkspaceSnapshot,
    cache_key: &str,
    pyramid: &ColorMapPyramid,
) -> Result<()> {
    let cache_dir = color_pyramid_cache_dir(workspace, cache_key);
    fs::create_dir_all(&cache_dir).with_context(|| {
        format!(
            "failed to create tracker color pyramid cache directory {}",
            cache_dir.display()
        )
    })?;
    save_rgba_image(&cache_dir.join("local.png"), &pyramid.local.image)?;
    save_rgba_image(&cache_dir.join("global.png"), &pyramid.global.image)?;
    save_rgba_image(&cache_dir.join("coarse.png"), &pyramid.coarse.image)?;
    Ok(())
}

fn save_gray_image(path: &Path, image: &GrayImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create tracker cache directory {}",
                parent.display()
            )
        })?;
    }

    let temp_path = cache_temp_path(path);
    image.save(&temp_path).with_context(|| {
        format!(
            "failed to write tracker cache image {}",
            temp_path.display()
        )
    })?;
    replace_cache_file(&temp_path, path)?;
    Ok(())
}

fn save_rgba_image(path: &Path, image: &RgbaImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create tracker cache directory {}",
                parent.display()
            )
        })?;
    }

    let temp_path = cache_temp_path(path);
    image.save(&temp_path).with_context(|| {
        format!(
            "failed to write tracker cache image {}",
            temp_path.display()
        )
    })?;
    replace_cache_file(&temp_path, path)?;
    Ok(())
}

fn replace_cache_file(temp_path: &Path, target_path: &Path) -> Result<()> {
    if target_path.exists() {
        fs::remove_file(target_path).with_context(|| {
            format!(
                "failed to remove stale tracker cache file {}",
                target_path.display()
            )
        })?;
    }
    fs::rename(temp_path, target_path).with_context(|| {
        format!(
            "failed to move tracker cache file {} -> {}",
            temp_path.display(),
            target_path.display()
        )
    })?;
    Ok(())
}

fn cache_temp_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("cache");
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin");
    path.with_file_name(format!("{stem}.tmp.{ext}"))
}

fn tracker_cache_root(workspace: &WorkspaceSnapshot) -> PathBuf {
    workspace.project_root.join("cache").join("tracking")
}

fn remove_cache_dir_if_exists(path: &Path) -> Result<()> {
    if !path.is_dir() {
        return Ok(());
    }

    fs::remove_dir_all(path).with_context(|| {
        format!(
            "failed to remove tracker cache directory {}",
            path.display()
        )
    })
}

fn match_pyramid_cache_dir(workspace: &WorkspaceSnapshot, cache_key: &str) -> PathBuf {
    tracker_cache_root(workspace)
        .join("pyramids")
        .join(cache_key)
}

fn color_pyramid_cache_dir(workspace: &WorkspaceSnapshot, cache_key: &str) -> PathBuf {
    tracker_cache_root(workspace)
        .join("color-pyramids")
        .join(cache_key)
}

fn match_pyramid_cache_key(
    bwiki_cache_dir: &Path,
    view_size: u32,
    local_scale: u32,
    global_scale: u32,
    coarse_scale: u32,
) -> Result<String> {
    let range = zoom_world_bounds(BWIKI_WORLD_ZOOM)
        .ok_or_else(|| crate::app_error!("missing BWiki zoom metadata for tracker precompute"))?;
    let source_hashes = load_or_compute_tracker_source_hashes(bwiki_cache_dir, range.zoom)?;
    Ok(format!(
        "mv{MATCH_PYRAMID_CACHE_VERSION}-z{}-{}-poi{}-v{}-l{}-g{}-c{}",
        range.zoom,
        source_hashes.tile_hash,
        source_hashes.poi_hash,
        view_size,
        local_scale,
        global_scale,
        coarse_scale
    ))
}

fn load_or_compute_tracker_source_hashes(
    bwiki_cache_dir: &Path,
    zoom: u8,
) -> Result<TrackerSourceHashes> {
    let fast_fingerprint = tracker_source_fast_fingerprint(bwiki_cache_dir, zoom)?;
    let memory_key = tracker_source_hash_memory_key(bwiki_cache_dir, zoom);

    if let Some(cached) = tracker_source_hash_memory_cache()
        .lock()
        .get(&memory_key)
        .cloned()
        .filter(|cached| cached.fast_fingerprint == fast_fingerprint)
    {
        return Ok(cached);
    }

    let cache_path = tracker_source_hash_cache_path(bwiki_cache_dir, zoom);
    if let Ok(Some(cached)) = load_tracker_source_hash_cache(&cache_path) {
        if cached.fast_fingerprint == fast_fingerprint {
            tracker_source_hash_memory_cache()
                .lock()
                .insert(memory_key, cached.clone());
            return Ok(cached);
        }
    }

    let range = zoom_world_bounds(zoom)
        .ok_or_else(|| crate::app_error!("missing BWiki zoom metadata for tracker precompute"))?;
    let computed = TrackerSourceHashes {
        fast_fingerprint,
        tile_hash: logic_tile_revision_hash(
            bwiki_cache_dir,
            range.zoom,
            range.min_x,
            range.max_x,
            range.min_y,
            range.max_y,
        )?,
        poi_hash: tracking_poi_revision_hash(bwiki_cache_dir)?,
    };
    let _ = save_tracker_source_hash_cache(&cache_path, &computed);
    tracker_source_hash_memory_cache()
        .lock()
        .insert(memory_key, computed.clone());
    Ok(computed)
}

fn tracker_source_fast_fingerprint(bwiki_cache_dir: &Path, zoom: u8) -> Result<String> {
    let cache = BwikiCachePaths::new(bwiki_cache_dir.to_path_buf());
    cache.ensure_directories()?;

    let mut hash = FNV_OFFSET_BASIS;
    update_hash_u64(&mut hash, u64::from(TRACKER_SOURCE_HASH_CACHE_VERSION));
    update_hash_u64(&mut hash, u64::from(zoom));
    update_hash_bytes(&mut hash, bwiki_cache_dir.to_string_lossy().as_bytes());

    for path in [
        cache.data_dir.join("dataset.json"),
        cache.icons_dir.clone(),
        cache.tiles_dir.clone(),
        cache.tiles_dir.join(zoom.to_string()),
    ] {
        update_hash_bytes(&mut hash, path.to_string_lossy().as_bytes());
        update_hash_bytes(&mut hash, maybe_metadata_fingerprint(&path).as_bytes());
    }

    Ok(format!("{hash:016x}"))
}

fn tracker_source_hash_cache_path(bwiki_cache_dir: &Path, zoom: u8) -> PathBuf {
    tracker_cache_root_from_bwiki_cache_dir(bwiki_cache_dir)
        .join("source-hashes")
        .join(format!(
            "z{zoom}-sv{TRACKER_SOURCE_HASH_CACHE_VERSION}.json"
        ))
}

fn tracker_cache_root_from_bwiki_cache_dir(bwiki_cache_dir: &Path) -> PathBuf {
    bwiki_cache_dir.parent().map_or_else(
        || PathBuf::from("cache").join("tracking"),
        |cache_root| cache_root.join("tracking"),
    )
}

fn load_tracker_source_hash_cache(path: &Path) -> Result<Option<TrackerSourceHashes>> {
    if !path.is_file() {
        return Ok(None);
    }

    let file = File::open(path).with_context(|| {
        format!(
            "failed to open tracker source hash cache {}",
            path.display()
        )
    })?;
    let reader = BufReader::new(file);
    let Ok(persisted) = serde_json::from_reader::<_, PersistedTrackerSourceHashes>(reader) else {
        return Ok(None);
    };
    if persisted.version != TRACKER_SOURCE_HASH_CACHE_VERSION {
        return Ok(None);
    }

    Ok(Some(TrackerSourceHashes {
        fast_fingerprint: persisted.fast_fingerprint,
        tile_hash: persisted.tile_hash,
        poi_hash: persisted.poi_hash,
    }))
}

fn save_tracker_source_hash_cache(path: &Path, hashes: &TrackerSourceHashes) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create tracker source hash cache directory {}",
                parent.display()
            )
        })?;
    }

    let temp_path = cache_temp_path(path);
    let file = File::create(&temp_path).with_context(|| {
        format!(
            "failed to create tracker source hash temp cache {}",
            temp_path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(
        &mut writer,
        &PersistedTrackerSourceHashes {
            version: TRACKER_SOURCE_HASH_CACHE_VERSION,
            fast_fingerprint: hashes.fast_fingerprint.clone(),
            tile_hash: hashes.tile_hash.clone(),
            poi_hash: hashes.poi_hash.clone(),
        },
    )?;
    writer.flush()?;
    drop(writer);

    replace_cache_file(&temp_path, path)?;
    Ok(())
}

fn maybe_metadata_fingerprint(path: &Path) -> String {
    metadata_fingerprint(path).unwrap_or_else(|_| "missing".to_owned())
}

fn logic_tile_revision_hash(
    bwiki_cache_dir: &Path,
    zoom: u8,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> Result<String> {
    let cache = BwikiCachePaths::new(bwiki_cache_dir.to_path_buf());
    cache.ensure_directories()?;

    let mut hash = FNV_OFFSET_BASIS;
    update_hash_u64(&mut hash, zoom as u64);
    for tile_y in min_y..=max_y {
        for tile_x in min_x..=max_x {
            let relative = format!("{zoom}/tile-{tile_x}_{tile_y}.png");
            update_hash_bytes(&mut hash, relative.as_bytes());

            let path = cache.tiles_dir.join(&relative);
            match fs::metadata(&path) {
                Ok(metadata) => {
                    update_hash_u64(&mut hash, metadata.len());
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                            update_hash_u64(&mut hash, duration.as_secs());
                            update_hash_u64(&mut hash, duration.subsec_nanos() as u64);
                        }
                    }
                }
                Err(_) => update_hash_u64(&mut hash, u64::MAX),
            }
        }
    }

    Ok(format!("{hash:016x}"))
}

fn tracking_poi_revision_hash(bwiki_cache_dir: &Path) -> Result<String> {
    const TRACKING_POI_MARK_TYPES: [u32; 7] = [201, 202, 203, 204, 205, 206, 210];

    let cache = BwikiCachePaths::new(bwiki_cache_dir.to_path_buf());
    cache.ensure_directories()?;

    let mut hash = FNV_OFFSET_BASIS;
    let dataset_path = cache.data_dir.join("dataset.json");
    update_hash_bytes(&mut hash, b"dataset.json");
    hash_file_metadata(&mut hash, &dataset_path);

    if let Ok(entries) = fs::read_dir(&cache.icons_dir) {
        let paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        for mark_type in TRACKING_POI_MARK_TYPES {
            let prefix = format!("{mark_type}.");
            if let Some(path) = paths.iter().find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&prefix))
            }) {
                update_hash_bytes(&mut hash, prefix.as_bytes());
                hash_file_metadata(&mut hash, path);
            }
        }
    }

    Ok(format!("{hash:016x}"))
}

fn hash_file_metadata(hash: &mut u64, path: &Path) {
    match fs::metadata(path) {
        Ok(metadata) => {
            update_hash_u64(hash, metadata.len());
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                    update_hash_u64(hash, duration.as_secs());
                    update_hash_u64(hash, u64::from(duration.subsec_nanos()));
                }
            }
        }
        Err(_) => update_hash_u64(hash, 0),
    }
}

fn workspace_memory_cache_key(
    workspace: &WorkspaceSnapshot,
    category: &str,
    cache_key: &str,
) -> String {
    format!(
        "{}::{category}::{cache_key}",
        workspace.project_root.display()
    )
}

fn tracker_source_hash_memory_key(bwiki_cache_dir: &Path, zoom: u8) -> String {
    format!("{}::z{zoom}", bwiki_cache_dir.display())
}

fn color_pyramid_memory_cache() -> &'static Mutex<HashMap<String, Weak<ColorMapPyramid>>> {
    COLOR_PYRAMID_MEMORY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tracker_source_hash_memory_cache() -> &'static Mutex<HashMap<String, TrackerSourceHashes>> {
    TRACKER_SOURCE_HASH_MEMORY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn prune_color_pyramid_memory_cache(workspace: &WorkspaceSnapshot) {
    let prefix = format!("{}::", workspace.project_root.display());
    color_pyramid_memory_cache()
        .lock()
        .retain(|key, _| !key.starts_with(&prefix));
}

fn prune_tracker_source_hash_memory_cache(bwiki_cache_dir: &Path) {
    let prefix = format!("{}::", bwiki_cache_dir.display());
    tracker_source_hash_memory_cache()
        .lock()
        .retain(|key, _| !key.starts_with(&prefix));
}

fn element_count(width: u32, height: u32, channels: usize) -> Result<usize> {
    let channels = channels.max(1);
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| crate::app_error!("tracker tensor cache dimensions overflow"))
}

fn update_hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn update_hash_u64(hash: &mut u64, value: u64) {
    update_hash_bytes(hash, &value.to_le_bytes());
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f32_slice(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32_vec(reader: &mut impl Read, len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use std::{env, fs};

    use super::*;
    use uuid::Uuid;

    #[test]
    fn tensor_cache_roundtrip_preserves_shape_and_values() -> Result<()> {
        let root =
            env::temp_dir().join(format!("game-map-tracker-rs-precompute-{}", Uuid::new_v4()));
        let path = root.join("tensor-cache.bin");
        let cache = PersistedTensorCache::from_parts(
            3,
            2,
            2,
            1,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            vec![11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        )?;

        save_tensor_cache(&path, &cache)?;
        let loaded = load_tensor_cache(&path)?.expect("tensor cache should exist");
        assert_eq!(loaded, cache);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn tensor_cache_loads_legacy_v1_layout() -> Result<()> {
        let root = env::temp_dir().join(format!(
            "game-map-tracker-rs-precompute-legacy-{}",
            Uuid::new_v4()
        ));
        let path = root.join("tensor-cache-v1.bin");
        fs::create_dir_all(path.parent().expect("temp parent should exist"))?;

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&TENSOR_CACHE_MAGIC)?;
        write_u32(&mut writer, 1)?;
        write_u32(&mut writer, 2)?;
        write_u32(&mut writer, 2)?;
        write_u32(&mut writer, 3)?;
        write_u64(&mut writer, 12)?;
        write_u64(&mut writer, 12)?;
        write_f32_slice(
            &mut writer,
            &[
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
            ],
        )?;
        write_f32_slice(
            &mut writer,
            &[
                1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0,
            ],
        )?;
        writer.flush()?;
        drop(writer);

        let loaded = load_tensor_cache(&path)?.expect("legacy tensor cache should exist");
        assert_eq!(loaded.width, 2);
        assert_eq!(loaded.height, 2);
        assert_eq!(loaded.primary_channels, 3);
        assert_eq!(loaded.secondary_channels, 3);
        assert_eq!(
            loaded.secondary,
            vec![
                1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0,
            ]
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn tracker_source_hash_cache_roundtrip_preserves_values() -> Result<()> {
        let root = env::temp_dir().join(format!(
            "game-map-tracker-rs-source-hash-{}",
            Uuid::new_v4()
        ));
        let path = root.join("source-hashes").join("z8-sv1.json");
        let hashes = TrackerSourceHashes {
            fast_fingerprint: "fast-fingerprint".to_owned(),
            tile_hash: "tile-hash".to_owned(),
            poi_hash: "poi-hash".to_owned(),
        };

        save_tracker_source_hash_cache(&path, &hashes)?;
        let loaded =
            load_tracker_source_hash_cache(&path)?.expect("source hash cache should exist");
        assert_eq!(loaded, hashes);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
