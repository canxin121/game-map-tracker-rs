use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context as _, Result, anyhow, bail};
use image::GrayImage;
use imageproc::contrast::equalize_histogram;

use crate::{
    resources::{
        BWIKI_WORLD_ZOOM, BwikiCachePaths, WorkspaceSnapshot,
        load_logic_map_with_tracking_poi_scaled_image, zoom_world_bounds,
    },
    tracking::vision::{
        MapPyramid, ScaledMap, build_match_representation, coarse_global_downscale, downscale_gray,
    },
};

const MATCH_PYRAMID_CACHE_VERSION: u32 = 3;
const TENSOR_CACHE_FORMAT_VERSION: u32 = 1;
const TENSOR_CACHE_MAGIC: [u8; 8] = *b"GMTRTC01";
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Debug, Clone)]
pub struct PreparedMatchPyramid {
    pub cache_key: String,
    pub pyramid: MapPyramid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedTensorCache {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub primary: Vec<f32>,
    pub secondary: Vec<f32>,
}

impl PersistedTensorCache {
    pub fn from_parts(
        width: u32,
        height: u32,
        channels: usize,
        primary: Vec<f32>,
        secondary: Vec<f32>,
    ) -> Result<Self> {
        let expected = element_count(width, height, channels)?;
        if primary.len() != expected {
            bail!(
                "persisted tensor primary buffer length {} does not match expected {}",
                primary.len(),
                expected
            );
        }
        if secondary.len() != expected {
            bail!(
                "persisted tensor secondary buffer length {} does not match expected {}",
                secondary.len(),
                expected
            );
        }

        Ok(Self {
            width,
            height,
            channels,
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

pub fn tracker_tensor_cache_path(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    cache_key: &str,
) -> PathBuf {
    tracker_cache_root(workspace).join("tensors").join(format!(
        "{prefix}-tv{TENSOR_CACHE_FORMAT_VERSION}-{cache_key}.bin"
    ))
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
        bail!(
            "unsupported tracker tensor cache magic in {}",
            path.display()
        );
    }

    let version = read_u32(&mut reader)?;
    if version != TENSOR_CACHE_FORMAT_VERSION {
        bail!(
            "unsupported tracker tensor cache version {} in {}",
            version,
            path.display()
        );
    }

    let width = read_u32(&mut reader)?;
    let height = read_u32(&mut reader)?;
    let channels = read_u32(&mut reader)? as usize;
    let primary_len = read_u64(&mut reader)? as usize;
    let secondary_len = read_u64(&mut reader)? as usize;
    let expected = element_count(width, height, channels)?;
    if primary_len != expected || secondary_len != expected {
        bail!(
            "tracker tensor cache payload length mismatch in {}",
            path.display()
        );
    }

    let primary = read_f32_vec(&mut reader, primary_len)?;
    let secondary = read_f32_vec(&mut reader, secondary_len)?;
    Ok(Some(PersistedTensorCache::from_parts(
        width, height, channels, primary, secondary,
    )?))
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
    write_u32(&mut writer, cache.channels as u32)?;
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

fn match_pyramid_cache_key(
    bwiki_cache_dir: &Path,
    view_size: u32,
    local_scale: u32,
    global_scale: u32,
    coarse_scale: u32,
) -> Result<String> {
    let range = zoom_world_bounds(BWIKI_WORLD_ZOOM)
        .ok_or_else(|| anyhow!("missing BWiki zoom metadata for tracker precompute"))?;
    let tile_hash = logic_tile_revision_hash(
        bwiki_cache_dir,
        range.zoom,
        range.min_x,
        range.max_x,
        range.min_y,
        range.max_y,
    )?;
    let poi_hash = tracking_poi_revision_hash(bwiki_cache_dir)?;
    Ok(format!(
        "mv{MATCH_PYRAMID_CACHE_VERSION}-z{}-{tile_hash}-poi{poi_hash}-v{}-l{}-g{}-c{}",
        range.zoom, view_size, local_scale, global_scale, coarse_scale
    ))
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

fn element_count(width: u32, height: u32, channels: usize) -> Result<usize> {
    let channels = channels.max(1);
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| anyhow!("tracker tensor cache dimensions overflow"))
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
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            vec![11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        )?;

        save_tensor_cache(&path, &cache)?;
        let loaded = load_tensor_cache(&path)?.expect("tensor cache should exist");
        assert_eq!(loaded, cache);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
