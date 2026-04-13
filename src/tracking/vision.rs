use std::cmp;

use anyhow::{Context as _, Result, bail};
use image::{
    GrayImage, ImageBuffer, Luma, Rgba, RgbaImage,
    imageops::{FilterType, crop_imm, resize},
};
use imageproc::{
    contrast::equalize_histogram,
    drawing::{draw_hollow_rect_mut, draw_line_segment_mut},
    rect::Rect,
};
use serde::{Deserialize, Serialize};
use strum::Display;

use crate::{
    domain::{geometry::WorldPoint, tracker::TrackerEngineKind},
    embedded_assets,
    resources::WorkspaceSnapshot,
    tracking::debug::{DebugField, DebugImage, DebugImageKind, TrackingDebugSnapshot},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
pub enum SearchStage {
    #[strum(to_string = "GlobalRelocate")]
    GlobalRelocate,
    #[strum(to_string = "LocalTrack")]
    LocalTrack,
}

#[derive(Debug, Clone)]
pub struct TrackerState {
    pub stage: SearchStage,
    pub last_world: Option<WorldPoint>,
    pub lost_frames: u32,
    pub local_fail_streak: u32,
    pub frame_index: u64,
}

impl Default for TrackerState {
    fn default() -> Self {
        Self {
            stage: SearchStage::GlobalRelocate,
            last_world: None,
            lost_frames: 0,
            local_fail_streak: 0,
            frame_index: 0,
        }
    }
}

impl TrackerState {
    pub fn begin_frame(&mut self) -> u64 {
        self.frame_index += 1;
        self.frame_index
    }

    pub fn mark_success(&mut self, world: WorldPoint) {
        self.stage = SearchStage::LocalTrack;
        self.last_world = Some(world);
        self.lost_frames = 0;
        self.local_fail_streak = 0;
    }

    pub fn increment_local_fail(&mut self, threshold: u32) -> bool {
        self.local_fail_streak += 1;
        if self.local_fail_streak >= threshold {
            self.stage = SearchStage::GlobalRelocate;
            true
        } else {
            false
        }
    }

    pub fn next_inertial_position(&mut self, max_lost_frames: u32) -> Option<WorldPoint> {
        let world = self.last_world?;
        self.lost_frames += 1;
        if self.lost_frames > max_lost_frames {
            self.stage = SearchStage::GlobalRelocate;
            return None;
        }
        Some(world)
    }
}

#[derive(Debug, Clone)]
pub struct ScaledMap {
    pub scale: u32,
    pub image: GrayImage,
}

#[derive(Debug, Clone)]
pub struct MapPyramid {
    pub local: ScaledMap,
    pub global: ScaledMap,
}

#[derive(Debug, Clone)]
pub struct MaskSet {
    pub local: GrayImage,
    pub global: GrayImage,
}

#[derive(Debug, Clone)]
pub struct MatchCandidate {
    pub world: WorldPoint,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchCrop {
    pub image: GrayImage,
    pub origin_x: u32,
    pub origin_y: u32,
}

#[derive(Debug, Clone)]
pub enum DebugOverlay {
    Crosshair {
        x: u32,
        y: u32,
    },
    Rect {
        left: u32,
        top: u32,
        width: u32,
        height: u32,
    },
}

pub fn load_logic_map_pyramid(workspace: &WorkspaceSnapshot) -> Result<(MapPyramid, MaskSet)> {
    let config = &workspace.config;
    let logic_map = embedded_assets::load_luma_image(workspace.assets.logic_map_asset_path)
        .with_context(|| {
            format!(
                "failed to load logic map from embedded asset {}",
                workspace.assets.logic_map_asset_path
            )
        })?;
    let logic_map = equalize_histogram(&logic_map);

    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);

    let pyramid = MapPyramid {
        local: ScaledMap {
            scale: local_scale,
            image: downscale_gray(&logic_map, local_scale),
        },
        global: ScaledMap {
            scale: global_scale,
            image: downscale_gray(&logic_map, global_scale),
        },
    };

    let masks = MaskSet {
        local: build_mask(
            scaled_dimension(config.minimap.width, local_scale),
            scaled_dimension(config.minimap.height, local_scale),
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
        global: build_mask(
            scaled_dimension(config.minimap.width, global_scale),
            scaled_dimension(config.minimap.height, global_scale),
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
    };

    Ok((pyramid, masks))
}

pub fn downscale_gray(image: &GrayImage, scale: u32) -> GrayImage {
    if scale <= 1 {
        return image.clone();
    }

    resize(
        image,
        scaled_dimension(image.width(), scale),
        scaled_dimension(image.height(), scale),
        FilterType::Triangle,
    )
}

pub fn scaled_dimension(dimension: u32, scale: u32) -> u32 {
    cmp::max(8, dimension / scale.max(1))
}

pub fn build_mask(width: u32, height: u32, inner_radius: f32, outer_radius: f32) -> GrayImage {
    let center_x = (width.saturating_sub(1)) as f32 * 0.5;
    let center_y = (height.saturating_sub(1)) as f32 * 0.5;
    let radius_x = width.max(1) as f32 * 0.5;
    let radius_y = height.max(1) as f32 * 0.5;

    ImageBuffer::from_fn(width, height, |x, y| {
        let dx = (x as f32 - center_x) / radius_x.max(1.0);
        let dy = (y as f32 - center_y) / radius_y.max(1.0);
        let distance = (dx * dx + dy * dy).sqrt();
        let enabled = distance <= outer_radius && distance >= inner_radius;
        Luma([if enabled { 255 } else { 0 }])
    })
}

pub fn crop_around_center(
    image: &GrayImage,
    center: (u32, u32),
    radius: u32,
    template_width: u32,
    template_height: u32,
) -> Result<SearchCrop> {
    if image.width() <= template_width || image.height() <= template_height {
        bail!("search image is smaller than template");
    }

    let half_w = template_width / 2;
    let half_h = template_height / 2;
    let desired_x = center.0.saturating_sub(radius + half_w);
    let desired_y = center.1.saturating_sub(radius + half_h);
    let max_x = image.width().saturating_sub(template_width + 1);
    let max_y = image.height().saturating_sub(template_height + 1);
    let origin_x = desired_x.min(max_x);
    let origin_y = desired_y.min(max_y);
    let width = cmp::min(
        image.width().saturating_sub(origin_x),
        radius.saturating_mul(2) + template_width + 1,
    );
    let height = cmp::min(
        image.height().saturating_sub(origin_y),
        radius.saturating_mul(2) + template_height + 1,
    );

    let image = crop_imm(image, origin_x, origin_y, width, height).to_image();
    Ok(SearchCrop {
        image,
        origin_x,
        origin_y,
    })
}

#[must_use]
pub fn center_to_scaled(world: WorldPoint, scale: u32) -> (u32, u32) {
    (
        (world.x.max(0.0) as u32) / scale.max(1),
        (world.y.max(0.0) as u32) / scale.max(1),
    )
}

#[must_use]
pub fn build_debug_snapshot(
    engine: TrackerEngineKind,
    frame_index: u64,
    stage: SearchStage,
    images: Vec<DebugImage>,
    fields: Vec<DebugField>,
) -> TrackingDebugSnapshot {
    TrackingDebugSnapshot {
        engine,
        frame_index,
        stage_label: stage.to_string(),
        images,
        fields,
    }
}

#[must_use]
pub fn preview_image(
    label: impl Into<String>,
    image: &GrayImage,
    overlays: &[DebugOverlay],
    max_side: u32,
) -> DebugImage {
    let (target_width, target_height) =
        fit_preview_dimensions(image.width(), image.height(), max_side);
    let mut preview = if target_width == image.width() && target_height == image.height() {
        image.clone()
    } else {
        resize(image, target_width, target_height, FilterType::Nearest)
    };

    let scale_x = preview.width() as f32 / image.width().max(1) as f32;
    let scale_y = preview.height() as f32 / image.height().max(1) as f32;
    for overlay in overlays {
        draw_overlay(&mut preview, overlay, scale_x, scale_y);
    }

    DebugImage::new(label, preview.width(), preview.height(), preview.into_raw())
        .with_kind(DebugImageKind::Snapshot)
}

#[must_use]
pub fn preview_mask_image(
    label: impl Into<String>,
    image: &GrayImage,
    max_side: u32,
) -> DebugImage {
    let (target_width, target_height) =
        fit_preview_dimensions(image.width(), image.height(), max_side);
    let preview = if target_width == image.width() && target_height == image.height() {
        image.clone()
    } else {
        resize(image, target_width, target_height, FilterType::Nearest)
    };

    let rgba = RgbaImage::from_fn(preview.width(), preview.height(), |x, y| {
        let intensity = preview.get_pixel(x, y).0[0];
        if intensity > 0 {
            Rgba([78, 205, 196, 255])
        } else {
            Rgba([12, 18, 28, 255])
        }
    });

    DebugImage::rgba(
        label,
        rgba.width(),
        rgba.height(),
        DebugImageKind::Mask,
        rgba.into_raw(),
    )
}

#[must_use]
pub fn preview_heatmap(
    label: impl Into<String>,
    width: u32,
    height: u32,
    scores: &[f32],
    peak: Option<(u32, u32)>,
    max_side: u32,
) -> DebugImage {
    if width == 0 || height == 0 || scores.is_empty() {
        return DebugImage::rgba(label, 1, 1, DebugImageKind::Heatmap, vec![0, 0, 0, 255]);
    }

    let (mut min_score, mut max_score) = (f32::INFINITY, f32::NEG_INFINITY);
    for score in scores.iter().copied().filter(|score| score.is_finite()) {
        min_score = min_score.min(score);
        max_score = max_score.max(score);
    }
    if !min_score.is_finite() || !max_score.is_finite() {
        min_score = 0.0;
        max_score = 1.0;
    }

    let range = (max_score - min_score).max(1e-6);
    let mut rgba = RgbaImage::from_fn(width, height, |x, y| {
        let index = (y * width + x) as usize;
        let score = scores.get(index).copied().unwrap_or(min_score);
        let normalized = ((score - min_score) / range).clamp(0.0, 1.0);
        Rgba(heatmap_color(normalized))
    });

    if let Some((x, y)) = peak {
        draw_crosshair_rgba(
            &mut rgba,
            x as f32,
            y as f32,
            4.0,
            Rgba([255, 255, 255, 255]),
        );
    }

    let (target_width, target_height) = fit_preview_dimensions(width, height, max_side);
    let rgba = if target_width == width && target_height == height {
        rgba
    } else {
        resize(&rgba, target_width, target_height, FilterType::Nearest)
    };

    DebugImage::rgba(
        label,
        rgba.width(),
        rgba.height(),
        DebugImageKind::Heatmap,
        rgba.into_raw(),
    )
}

#[must_use]
pub fn mask_as_unit_vec(mask: &GrayImage, channels: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(mask.width() as usize * mask.height() as usize * channels);
    for _ in 0..channels {
        values.extend(mask.pixels().map(|pixel| f32::from(pixel.0[0]) / 255.0));
    }
    values
}

#[must_use]
pub fn gray_image_as_unit_vec(image: &GrayImage) -> Vec<f32> {
    image
        .pixels()
        .map(|pixel| f32::from(pixel.0[0]) / 255.0)
        .collect()
}

fn fit_preview_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if width <= max_side && height <= max_side {
        return (width.max(1), height.max(1));
    }

    let scale = (max_side as f32 / width.max(height) as f32).clamp(0.1, 1.0);
    (
        (width as f32 * scale).round().max(1.0) as u32,
        (height as f32 * scale).round().max(1.0) as u32,
    )
}

fn draw_overlay(image: &mut GrayImage, overlay: &DebugOverlay, scale_x: f32, scale_y: f32) {
    match overlay {
        DebugOverlay::Crosshair { x, y } => {
            let x = (*x as f32 * scale_x).round();
            let y = (*y as f32 * scale_y).round();
            draw_crosshair(image, x, y, 8.0);
        }
        DebugOverlay::Rect {
            left,
            top,
            width,
            height,
        } => {
            let rect = Rect::at(
                ((*left as f32) * scale_x).round() as i32,
                ((*top as f32) * scale_y).round() as i32,
            )
            .of_size(
                ((*width as f32) * scale_x).round().max(1.0) as u32,
                ((*height as f32) * scale_y).round().max(1.0) as u32,
            );
            draw_hollow_rect_mut(image, rect, Luma([255]));
        }
    }
}

fn draw_crosshair(image: &mut GrayImage, x: f32, y: f32, radius: f32) {
    draw_line_segment_mut(image, (x - radius, y), (x + radius, y), Luma([255]));
    draw_line_segment_mut(image, (x, y - radius), (x, y + radius), Luma([255]));
}

fn draw_crosshair_rgba(image: &mut RgbaImage, x: f32, y: f32, radius: f32, color: Rgba<u8>) {
    draw_line_segment_mut(image, (x - radius, y), (x + radius, y), color);
    draw_line_segment_mut(image, (x, y - radius), (x, y + radius), color);
}

fn heatmap_color(value: f32) -> [u8; 4] {
    let value = value.clamp(0.0, 1.0);
    let (r, g, b) = if value < 0.25 {
        let t = value / 0.25;
        (0.0, t * 180.0, 120.0 + t * 135.0)
    } else if value < 0.5 {
        let t = (value - 0.25) / 0.25;
        (t * 80.0, 180.0 + t * 60.0, 255.0 - t * 180.0)
    } else if value < 0.75 {
        let t = (value - 0.5) / 0.25;
        (80.0 + t * 175.0, 240.0 - t * 60.0, 75.0 - t * 75.0)
    } else {
        let t = (value - 0.75) / 0.25;
        (255.0, 180.0 - t * 120.0, 0.0)
    };

    [r.round() as u8, g.round() as u8, b.round() as u8, 255]
}
