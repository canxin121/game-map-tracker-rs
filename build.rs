use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));
    let generated_path = out_dir.join("embedded_assets.rs");
    let icon_assets_root = manifest_dir.join("assets").join("icons");
    let tracker_encoder_path = manifest_dir
        .join("models")
        .join("tracker_encoder.safetensors");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let ai_burn_enabled = env::var_os("CARGO_FEATURE_AI_BURN").is_some();

    for root in [&icon_assets_root] {
        println!("cargo:rerun-if-changed={}", root.display());
    }
    println!("cargo:rerun-if-changed={}", tracker_encoder_path.display());

    println!("cargo:rustc-check-cfg=cfg(burn_cuda_backend)");
    println!("cargo:rustc-check-cfg=cfg(burn_vulkan_backend)");
    println!("cargo:rustc-check-cfg=cfg(burn_metal_backend)");

    if ai_burn_enabled && target_os == "windows" {
        println!("cargo:rustc-cfg=burn_cuda_backend");
        println!("cargo:rustc-cfg=burn_vulkan_backend");
    }
    if ai_burn_enabled && target_os == "macos" {
        println!("cargo:rustc-cfg=burn_metal_backend");
    }
    if ai_burn_enabled && !matches!(target_os.as_str(), "windows" | "macos") {
        println!("cargo:rustc-cfg=burn_vulkan_backend");
    }

    let mut files = Vec::new();
    collect_files(
        &icon_assets_root,
        &icon_assets_root,
        Path::new("assets").join("icons").as_path(),
        &mut files,
    );
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut generated = String::from("pub static EMBEDDED_ASSET_FILES: &[(&str, &[u8])] = &[\n");

    for (relative, absolute) in files {
        let absolute = absolute.to_string_lossy().to_string();

        generated.push_str(&format!(
            "    ({relative:?}, include_bytes!({absolute:?}) as &[u8]),\n"
        ));
    }

    generated.push_str("];\n");
    fs::write(generated_path, generated).expect("failed to write generated embedded assets");
}

fn collect_files(
    source_root: &Path,
    current_dir: &Path,
    target_prefix: &Path,
    files: &mut Vec<(String, PathBuf)>,
) {
    for entry in fs::read_dir(current_dir).expect("failed to scan embedded workspace") {
        let entry = entry.expect("failed to read embedded entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files(source_root, &path, target_prefix, files);
        } else if path.is_file() {
            let relative = path
                .strip_prefix(source_root)
                .expect("embedded file should stay under source root")
                .to_path_buf();
            let relative = if target_prefix.as_os_str().is_empty() {
                relative
            } else {
                target_prefix.join(relative)
            };
            files.push((normalize_path(&relative), path));
        }
    }
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
