use std::{
    env,
    path::{Path, PathBuf},
};

pub(crate) fn repository_root() -> Option<PathBuf> {
    env::current_dir()
        .ok()
        .and_then(|path| find_repository_root(&path))
        .or_else(|| find_repository_root(Path::new(env!("CARGO_MANIFEST_DIR"))))
}

pub(crate) fn repository_tracker_encoder_path() -> Option<PathBuf> {
    repository_root().map(|root| root.join("models").join("tracker_encoder.safetensors"))
}

pub(crate) fn embedded_tracker_encoder_bytes() -> &'static [u8] {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/models/tracker_encoder.safetensors"
    ))
}

fn find_repository_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(path) = current {
        if path.join(".git").exists() {
            return Some(path.to_path_buf());
        }
        current = path.parent();
    }
    None
}
