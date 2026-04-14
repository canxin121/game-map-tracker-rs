include!(concat!(env!("OUT_DIR"), "/embedded_assets.rs"));

#[must_use]
pub fn asset_bytes(path: &str) -> Option<&'static [u8]> {
    EMBEDDED_ASSET_FILES
        .binary_search_by_key(&path, |(candidate, _)| *candidate)
        .ok()
        .map(|index| EMBEDDED_ASSET_FILES[index].1)
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
