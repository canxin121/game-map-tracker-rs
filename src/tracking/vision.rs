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
    resources::{WorkspaceSnapshot, load_logic_map_with_tracking_poi_scaled_image},
    tracking::debug::{DebugField, DebugImage, DebugImageKind, TrackingDebugSnapshot},
};

const INSCRIBED_SQUARE_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

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
    pub reacquire_anchor: Option<WorldPoint>,
    pub lost_frames: u32,
    pub local_fail_streak: u32,
    pub frame_index: u64,
}

impl Default for TrackerState {
    fn default() -> Self {
        Self {
            stage: SearchStage::GlobalRelocate,
            last_world: None,
            reacquire_anchor: None,
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
        self.reacquire_anchor = None;
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
        self.reacquire_anchor.get_or_insert(world);
        self.lost_frames += 1;
        if self.lost_frames > max_lost_frames {
            self.stage = SearchStage::GlobalRelocate;
            return None;
        }
        Some(world)
    }

    pub fn force_global_relocate(&mut self) {
        self.stage = SearchStage::GlobalRelocate;
        self.local_fail_streak = 0;
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
    pub coarse: ScaledMap,
}

#[derive(Debug, Clone)]
pub struct MaskSet {
    pub local: GrayImage,
    pub global: GrayImage,
    pub coarse: GrayImage,
}

#[derive(Debug, Clone)]
pub struct MatchCandidate {
    pub world: WorldPoint,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalCandidateDecision {
    Accept,
    Reject,
    ForceGlobalRelocate { jump: f32, anchor: WorldPoint },
}

#[derive(Debug, Clone)]
pub struct SearchCrop {
    pub image: GrayImage,
    pub origin_x: u32,
    pub origin_y: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchRegion {
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u32,
    pub height: u32,
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
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(config);
    let base_map = load_logic_map_with_tracking_poi_scaled_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        config.view_size,
    )
    .with_context(|| {
        format!(
            "failed to load augmented BWiki logic tiles from {}",
            workspace.assets.bwiki_cache_dir.display()
        )
    })?;
    let base_map = equalize_histogram(&base_map);
    let local_map = downscale_gray(&base_map, local_scale);
    let global_map = if global_scale == local_scale {
        local_map.clone()
    } else {
        downscale_gray(&base_map, global_scale)
    };
    let coarse_map = if coarse_scale == global_scale {
        global_map.clone()
    } else {
        downscale_gray(&base_map, coarse_scale)
    };

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
        coarse: build_mask(
            scaled_dimension(config.minimap.width, coarse_scale),
            scaled_dimension(config.minimap.height, coarse_scale),
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
    };

    Ok((pyramid, masks))
}

pub fn coarse_global_downscale(config: &crate::config::AppConfig) -> u32 {
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    global_scale.saturating_mul(2).max(global_scale)
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

#[must_use]
pub fn inscribed_square_dimension(dimension: u32) -> u32 {
    ((dimension.max(1) as f32) * INSCRIBED_SQUARE_SCALE)
        .round()
        .max(1.0) as u32
}

#[must_use]
pub fn scaled_template_dimension(view_size: u32, scale: u32) -> u32 {
    inscribed_square_dimension(scaled_dimension(view_size.max(1), scale.max(1)))
}

pub fn build_inner_square_mask(
    width: u32,
    height: u32,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    let normalized_inner = normalized_inner_radius(inner_radius, outer_radius);
    ImageBuffer::from_fn(width, height, |x, y| {
        let enabled = inscribed_square_circle_ratio(width, height, x, y) >= normalized_inner;
        Luma([if enabled { 255 } else { 0 }])
    })
}

pub fn capture_template_annulus(
    captured: &GrayImage,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    let diameter_px = ((captured.width().min(captured.height()) as f32) * outer_radius)
        .round()
        .max(1.0) as u32;
    let offset_x = captured.width().saturating_sub(diameter_px) / 2;
    let offset_y = captured.height().saturating_sub(diameter_px) / 2;
    let square = crop_imm(captured, offset_x, offset_y, diameter_px, diameter_px).to_image();
    soften_capture_to_annulus(&square, inner_radius, outer_radius)
}

pub fn capture_template_inner_square(
    captured: &GrayImage,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    let diameter_px = ((captured.width().min(captured.height()) as f32) * outer_radius)
        .round()
        .max(1.0) as u32;
    let square_side =
        inscribed_square_dimension(diameter_px).min(captured.width().min(captured.height()).max(1));
    let offset_x = captured.width().saturating_sub(square_side) / 2;
    let offset_y = captured.height().saturating_sub(square_side) / 2;
    let square = crop_imm(captured, offset_x, offset_y, square_side, square_side).to_image();
    soften_capture_center_hole(&square, normalized_inner_radius(inner_radius, outer_radius))
}

fn soften_capture_to_annulus(image: &GrayImage, inner_radius: f32, outer_radius: f32) -> GrayImage {
    let feather = annulus_feather(image.width(), image.height());
    let valid_inner = (inner_radius + feather).clamp(0.0, 1.5);
    let valid_outer = (outer_radius - feather).clamp(valid_inner, 1.5);
    let mut ring_sum = 0u64;
    let mut ring_count = 0u64;
    let mut weighted_sum = 0.0f32;
    let mut weighted_count = 0.0f32;
    let mut fallback_sum = 0u64;

    for (x, y, pixel) in image.enumerate_pixels() {
        let distance = annulus_distance(image.width(), image.height(), x, y);
        let alpha = annulus_alpha(distance, inner_radius, outer_radius, feather);
        if distance >= valid_inner && distance <= valid_outer {
            ring_sum += u64::from(pixel.0[0]);
            ring_count += 1;
        }
        weighted_sum += f32::from(pixel.0[0]) * alpha;
        weighted_count += alpha;
        fallback_sum += u64::from(pixel.0[0]);
    }

    let neutral = if ring_count > 0 {
        (ring_sum / ring_count) as u8
    } else if weighted_count > 1e-3 {
        (weighted_sum / weighted_count).round().clamp(0.0, 255.0) as u8
    } else {
        let pixel_count = u64::from(image.width().max(1)) * u64::from(image.height().max(1));
        (fallback_sum / pixel_count.max(1)) as u8
    };

    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let source = f32::from(image.get_pixel(x, y).0[0]);
        let alpha = annulus_alpha(
            annulus_distance(image.width(), image.height(), x, y),
            inner_radius,
            outer_radius,
            feather,
        );
        let value = (f32::from(neutral) * (1.0 - alpha) + source * alpha)
            .round()
            .clamp(0.0, 255.0) as u8;
        Luma([value])
    })
}

fn soften_capture_center_hole(image: &GrayImage, inner_radius: f32) -> GrayImage {
    let feather = annulus_feather(image.width(), image.height());
    let valid_inner = (inner_radius + feather).clamp(0.0, 1.0);
    let mut ring_sum = 0u64;
    let mut ring_count = 0u64;
    let mut weighted_sum = 0.0f32;
    let mut weighted_count = 0.0f32;
    let mut fallback_sum = 0u64;

    for (x, y, pixel) in image.enumerate_pixels() {
        let distance = inscribed_square_circle_ratio(image.width(), image.height(), x, y);
        let alpha = inner_hole_alpha(distance, inner_radius, feather);
        if distance >= valid_inner {
            ring_sum += u64::from(pixel.0[0]);
            ring_count += 1;
        }
        weighted_sum += f32::from(pixel.0[0]) * alpha;
        weighted_count += alpha;
        fallback_sum += u64::from(pixel.0[0]);
    }

    let neutral = if ring_count > 0 {
        (ring_sum / ring_count) as u8
    } else if weighted_count > 1e-3 {
        (weighted_sum / weighted_count).round().clamp(0.0, 255.0) as u8
    } else {
        let pixel_count = u64::from(image.width().max(1)) * u64::from(image.height().max(1));
        (fallback_sum / pixel_count.max(1)) as u8
    };

    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let source = f32::from(image.get_pixel(x, y).0[0]);
        let alpha = inner_hole_alpha(
            inscribed_square_circle_ratio(image.width(), image.height(), x, y),
            inner_radius,
            feather,
        );
        let value = (f32::from(neutral) * (1.0 - alpha) + source * alpha)
            .round()
            .clamp(0.0, 255.0) as u8;
        Luma([value])
    })
}

fn annulus_alpha(distance: f32, inner_radius: f32, outer_radius: f32, feather: f32) -> f32 {
    let inner = inner_radius.clamp(0.0, 1.0);
    let outer = outer_radius.clamp(inner + f32::EPSILON, 1.5);

    let inner_alpha = if inner <= 0.0 {
        1.0
    } else {
        smoothstep(inner, inner + feather, distance)
    };
    let outer_alpha = if outer >= 1.5 {
        1.0
    } else {
        1.0 - smoothstep((outer - feather).max(0.0), outer, distance)
    };

    (inner_alpha * outer_alpha).clamp(0.0, 1.0)
}

fn inner_hole_alpha(distance: f32, inner_radius: f32, feather: f32) -> f32 {
    let inner = inner_radius.clamp(0.0, 1.0);
    if inner <= 0.0 {
        return 1.0;
    }
    if inner >= 1.0 {
        return 0.0;
    }

    smoothstep(inner, (inner + feather).min(1.0), distance)
}

fn annulus_feather(width: u32, height: u32) -> f32 {
    let diameter = width.min(height).max(1) as f32;
    (8.0 / diameter).clamp(0.02, 0.06)
}

fn annulus_distance(width: u32, height: u32, x: u32, y: u32) -> f32 {
    let center_x = (width.saturating_sub(1)) as f32 * 0.5;
    let center_y = (height.saturating_sub(1)) as f32 * 0.5;
    let radius_x = width.max(1) as f32 * 0.5;
    let radius_y = height.max(1) as f32 * 0.5;
    let dx = (x as f32 - center_x) / radius_x.max(1.0);
    let dy = (y as f32 - center_y) / radius_y.max(1.0);
    (dx * dx + dy * dy).sqrt()
}

fn inscribed_square_circle_ratio(width: u32, height: u32, x: u32, y: u32) -> f32 {
    annulus_distance(width, height, x, y) * INSCRIBED_SQUARE_SCALE
}

fn normalized_inner_radius(inner_radius: f32, outer_radius: f32) -> f32 {
    if outer_radius <= f32::EPSILON {
        0.0
    } else {
        (inner_radius / outer_radius).clamp(0.0, 1.0)
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }

    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn build_match_representation(image: &GrayImage) -> GrayImage {
    let edge = edge_enhance_gray(image);
    let blended = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let base = u16::from(image.get_pixel(x, y).0[0]);
        let edge = u16::from(edge.get_pixel(x, y).0[0]);
        let value = ((base * 3) + (edge * 5)) / 8;
        Luma([value.min(u16::from(u8::MAX)) as u8])
    });

    equalize_histogram(&blended)
}

fn edge_enhance_gray(image: &GrayImage) -> GrayImage {
    let grad_x = imageproc::gradients::horizontal_sobel(image);
    let grad_y = imageproc::gradients::vertical_sobel(image);

    let edge = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let gx = u32::from(grad_x.get_pixel(x, y).0[0].unsigned_abs());
        let gy = u32::from(grad_y.get_pixel(x, y).0[0].unsigned_abs());
        let magnitude = ((gx + gy) / 2).min(u32::from(u8::MAX)) as u8;
        Luma([magnitude])
    });

    equalize_histogram(&edge)
}

pub fn search_region_around_center(
    image_width: u32,
    image_height: u32,
    center: (u32, u32),
    radius: u32,
    template_width: u32,
    template_height: u32,
) -> Result<SearchRegion> {
    if image_width <= template_width || image_height <= template_height {
        bail!("search image is smaller than template");
    }

    let half_w = template_width / 2;
    let half_h = template_height / 2;
    let desired_x = center.0.saturating_sub(radius + half_w);
    let desired_y = center.1.saturating_sub(radius + half_h);
    let max_x = image_width.saturating_sub(template_width + 1);
    let max_y = image_height.saturating_sub(template_height + 1);
    let origin_x = desired_x.min(max_x);
    let origin_y = desired_y.min(max_y);
    let width = cmp::min(
        image_width.saturating_sub(origin_x),
        radius.saturating_mul(2) + template_width + 1,
    );
    let height = cmp::min(
        image_height.saturating_sub(origin_y),
        radius.saturating_mul(2) + template_height + 1,
    );

    Ok(SearchRegion {
        origin_x,
        origin_y,
        width,
        height,
    })
}

pub fn crop_search_region(image: &GrayImage, region: SearchRegion) -> Result<SearchCrop> {
    if image.width() < region.origin_x.saturating_add(region.width)
        || image.height() < region.origin_y.saturating_add(region.height)
    {
        bail!("search region is outside image bounds");
    }

    let image = crop_imm(
        image,
        region.origin_x,
        region.origin_y,
        region.width,
        region.height,
    )
    .to_image();
    Ok(SearchCrop {
        image,
        origin_x: region.origin_x,
        origin_y: region.origin_y,
    })
}

pub fn crop_around_center(
    image: &GrayImage,
    center: (u32, u32),
    radius: u32,
    template_width: u32,
    template_height: u32,
) -> Result<SearchCrop> {
    let region = search_region_around_center(
        image.width(),
        image.height(),
        center,
        radius,
        template_width,
        template_height,
    )?;
    crop_search_region(image, region)
}

#[must_use]
pub fn local_candidate_decision(
    last_world: WorldPoint,
    candidate_world: WorldPoint,
    max_accepted_jump_px: u32,
    reacquire_anchor: Option<WorldPoint>,
    reacquire_jump_threshold_px: u32,
) -> LocalCandidateDecision {
    if let Some(anchor) = reacquire_anchor {
        let jump = world_jump_distance(candidate_world, anchor);
        if jump > reacquire_jump_threshold_px as f32 {
            return LocalCandidateDecision::ForceGlobalRelocate { jump, anchor };
        }
    }

    let jump = world_jump_distance(candidate_world, last_world);
    if jump <= max_accepted_jump_px as f32 {
        return LocalCandidateDecision::Accept;
    }

    LocalCandidateDecision::Reject
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

fn world_jump_distance(lhs: WorldPoint, rhs: WorldPoint) -> f32 {
    (lhs.x - rhs.x).abs() + (lhs.y - rhs.y).abs()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_template_annulus_neutralizes_center_arrow_region() {
        let image = GrayImage::from_fn(101, 101, |x, y| {
            let distance = annulus_distance(101, 101, x, y);
            let value = if distance <= 0.18 {
                245
            } else if distance <= 0.94 {
                122
            } else {
                18
            };
            Luma([value])
        });

        let annulus = capture_template_annulus(&image, 0.22, 1.0);
        let center = annulus.get_pixel(50, 50).0[0];
        let ring = annulus.get_pixel(50, 28).0[0];
        let corner = annulus.get_pixel(0, 0).0[0];

        assert!(
            (i32::from(center) - 122).abs() <= 12,
            "center should be neutralized toward ring intensity, got {center}"
        );
        assert!(
            (i32::from(corner) - 122).abs() <= 16,
            "outer UI should be neutralized toward ring intensity, got {corner}"
        );
        assert!(
            (i32::from(ring) - 122).abs() <= 10,
            "ring texture should stay intact, got {ring}"
        );
    }

    #[test]
    fn capture_template_inner_square_preserves_real_corners_and_neutralizes_center() {
        let image = GrayImage::from_fn(121, 121, |x, y| {
            let distance = annulus_distance(121, 121, x, y);
            let value = if distance <= 0.16 {
                245
            } else if distance <= 1.0 {
                ((x * 3 + y * 5) % 251) as u8
            } else {
                7
            };
            Luma([value])
        });

        let square = capture_template_inner_square(&image, 0.16, 1.0);
        let side = inscribed_square_dimension(121);
        let offset = (121 - side) / 2;
        let raw_square = crop_imm(&image, offset, offset, side, side).to_image();
        let expected_corner = raw_square.get_pixel(0, 0).0[0];
        let outer_sample = raw_square.get_pixel(side / 2, 6).0[0];
        let center = square.get_pixel(square.width() / 2, square.height() / 2).0[0];
        let (retained_sum, retained_count) = raw_square
            .enumerate_pixels()
            .filter(|(x, y, _)| inscribed_square_circle_ratio(side, side, *x, *y) >= 0.24)
            .fold((0u64, 0u64), |(sum, count), (_, _, pixel)| {
                (sum + u64::from(pixel.0[0]), count + 1)
            });
        let retained_mean = retained_sum as f32 / retained_count.max(1) as f32;

        assert_eq!(square.width(), side);
        assert_eq!(square.height(), side);
        assert_eq!(square.get_pixel(0, 0).0[0], expected_corner);
        assert_eq!(square.get_pixel(side / 2, 6).0[0], outer_sample);
        assert!(
            (f32::from(center) - retained_mean).abs() <= 18.0,
            "center should be neutralized toward the retained map texture, got center {center} vs mean {retained_mean:.1}"
        );
        assert!(
            (i32::from(center) - 245).abs() >= 40,
            "center should not keep the original hole highlight, got {center}"
        );
    }

    #[test]
    fn inner_square_mask_keeps_corners_and_cuts_center_hole() {
        let mask = build_inner_square_mask(71, 71, 0.22, 1.0);

        assert_eq!(mask.get_pixel(0, 0).0[0], 255);
        assert_eq!(mask.get_pixel(mask.width() / 2, mask.height() / 2).0[0], 0);
        assert_eq!(mask.get_pixel(mask.width() / 2, 0).0[0], 255);
    }

    #[test]
    fn local_candidate_accepts_normal_local_jump() {
        let decision = local_candidate_decision(
            WorldPoint::new(100.0, 100.0),
            WorldPoint::new(120.0, 130.0),
            60,
            None,
            40,
        );

        assert_eq!(decision, LocalCandidateDecision::Accept);
    }

    #[test]
    fn local_candidate_forces_global_when_reacquire_jump_is_too_large() {
        let decision = local_candidate_decision(
            WorldPoint::new(100.0, 100.0),
            WorldPoint::new(180.0, 180.0),
            200,
            Some(WorldPoint::new(100.0, 100.0)),
            80,
        );

        assert_eq!(
            decision,
            LocalCandidateDecision::ForceGlobalRelocate {
                jump: 160.0,
                anchor: WorldPoint::new(100.0, 100.0),
            }
        );
    }

    #[test]
    fn tracker_state_records_and_clears_reacquire_anchor() {
        let world = WorldPoint::new(240.0, 320.0);
        let mut state = TrackerState::default();
        state.mark_success(world);

        assert_eq!(state.reacquire_anchor, None);
        assert_eq!(state.next_inertial_position(5), Some(world));
        assert_eq!(state.reacquire_anchor, Some(world));

        state.mark_success(WorldPoint::new(260.0, 340.0));
        assert_eq!(state.reacquire_anchor, None);
    }
}
