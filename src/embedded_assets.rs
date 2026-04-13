use anyhow::{Context as _, Result, anyhow};
use image::{GenericImageView as _, GrayImage};

pub const LOGIC_MAP_ASSET_PATH: &str = "assets/map/logic_map.png";
pub const DISPLAY_MAP_ASSET_PATH: &str = "assets/map/display_map.png";
pub const POINT_ICON_ASSET_DIR: &str = "assets/points";

include!(concat!(env!("OUT_DIR"), "/embedded_assets.rs"));

#[must_use]
pub fn asset_bytes(path: &str) -> Option<&'static [u8]> {
    EMBEDDED_ASSET_FILES
        .binary_search_by_key(&path, |(candidate, _)| *candidate)
        .ok()
        .map(|index| EMBEDDED_ASSET_FILES[index].1)
}

pub fn required_asset_bytes(path: &str) -> Result<&'static [u8]> {
    asset_bytes(path).ok_or_else(|| anyhow!("missing embedded asset {path}"))
}

pub fn image_dimensions(path: &str) -> Result<(u32, u32)> {
    let image = image::load_from_memory(required_asset_bytes(path)?)
        .with_context(|| format!("failed to decode embedded image {path}"))?;
    Ok(image.dimensions())
}

pub fn load_luma_image(path: &str) -> Result<GrayImage> {
    image::load_from_memory(required_asset_bytes(path)?)
        .with_context(|| format!("failed to decode embedded image {path}"))
        .map(|image| image.into_luma8())
}

pub fn runtime_asset_paths(path: &str) -> Vec<&'static str> {
    let normalized = path.trim_matches('/');
    let prefix = if normalized.is_empty() {
        None
    } else {
        Some(format!("{normalized}/"))
    };

    EMBEDDED_ASSET_FILES
        .iter()
        .filter_map(|(candidate, _)| {
            if normalized.is_empty()
                || *candidate == normalized
                || prefix
                    .as_ref()
                    .is_some_and(|prefix| candidate.starts_with(prefix))
            {
                Some(*candidate)
            } else {
                None
            }
        })
        .collect()
}
