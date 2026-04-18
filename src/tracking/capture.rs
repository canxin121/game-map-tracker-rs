use anyhow::{Context as _, Result, anyhow, bail};
use image::{DynamicImage, GrayImage, RgbaImage};
use imageproc::contrast::equalize_histogram;
use screenshots::{Screen, display_info::DisplayInfo};
use serde::{Deserialize, Serialize};

use crate::config::CaptureRegion;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScreenCaptureRegion {
    pub screen_origin_x: i32,
    pub screen_origin_y: i32,
    pub relative_left: i32,
    pub relative_top: i32,
    pub width: u32,
    pub height: u32,
}

impl ScreenCaptureRegion {
    #[must_use]
    pub fn from_capture_region(region: &CaptureRegion, display: DisplayInfo) -> Self {
        Self {
            screen_origin_x: display.x,
            screen_origin_y: display.y,
            relative_left: region.left - display.x,
            relative_top: region.top - display.y,
            width: region.width,
            height: region.height,
        }
    }
}

pub trait CaptureSource: Send {
    fn capture_gray(&self) -> Result<GrayImage>;
}

#[derive(Debug, Clone)]
pub struct DesktopCapture {
    screen: Screen,
    region: ScreenCaptureRegion,
}

impl DesktopCapture {
    pub fn from_absolute_region(region: &CaptureRegion) -> Result<Self> {
        let screen = Screen::all()?
            .into_iter()
            .find(|screen| contains_point(screen.display_info, region.left, region.top))
            .ok_or_else(|| {
                anyhow!(
                    "no display contains capture point ({}, {})",
                    region.left,
                    region.top
                )
            })?;

        let capture_region = ScreenCaptureRegion::from_capture_region(region, screen.display_info);
        validate_region(capture_region, screen.display_info)?;

        Ok(Self {
            screen,
            region: capture_region,
        })
    }

    pub fn capture_rgba(&self) -> Result<RgbaImage> {
        let rgba = self
            .screen
            .capture_area(
                self.region.relative_left,
                self.region.relative_top,
                self.region.width,
                self.region.height,
            )
            .context("failed to capture raw minimap region")?;

        let (width, height) = rgba.dimensions();
        screenshot_buffer_to_rgba(rgba.into_raw(), width, height)
    }
}

impl CaptureSource for DesktopCapture {
    fn capture_gray(&self) -> Result<GrayImage> {
        Ok(preprocess_capture(self.capture_rgba()?))
    }
}

fn contains_point(display: DisplayInfo, x: i32, y: i32) -> bool {
    let x2 = display.x + display.width as i32;
    let y2 = display.y + display.height as i32;
    x >= display.x && x < x2 && y >= display.y && y < y2
}

fn validate_region(region: ScreenCaptureRegion, display: DisplayInfo) -> Result<()> {
    if region.relative_left < 0 || region.relative_top < 0 {
        bail!(
            "capture region ({}, {}) falls outside the selected display origin ({}, {})",
            region.relative_left,
            region.relative_top,
            region.screen_origin_x,
            region.screen_origin_y
        );
    }

    let right = region.relative_left + region.width as i32;
    let bottom = region.relative_top + region.height as i32;
    if right > display.width as i32 || bottom > display.height as i32 {
        bail!(
            "capture region {}x{} exceeds display bounds {}x{}",
            region.width,
            region.height,
            display.width,
            display.height
        );
    }

    Ok(())
}

fn screenshot_buffer_to_rgba(buffer: Vec<u8>, width: u32, height: u32) -> Result<RgbaImage> {
    RgbaImage::from_raw(width, height, buffer)
        .context("failed to normalize screenshot image buffer into the primary image crate")
}

#[must_use]
pub fn preprocess_capture(rgba: RgbaImage) -> GrayImage {
    let gray = DynamicImage::ImageRgba8(rgba).into_luma8();
    equalize_histogram(&gray)
}
