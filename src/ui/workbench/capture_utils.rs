use std::path::Path;

use image::RgbaImage;

use crate::{
    config::CaptureRegion,
    error::{ContextExt as _, Result},
    tracking::capture::DesktopCapture,
};

pub(super) fn capture_region_rgba(region: &CaptureRegion) -> Result<RgbaImage> {
    DesktopCapture::from_absolute_region(region)?.capture_rgba()
}

pub(super) fn save_capture_region_png(region: &CaptureRegion, output_path: &Path) -> Result<()> {
    capture_region_rgba(region)?
        .save(output_path)
        .with_context(|| format!("failed to save capture image {}", output_path.display()))?;
    Ok(())
}
