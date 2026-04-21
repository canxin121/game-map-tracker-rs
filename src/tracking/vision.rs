use std::cmp;

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
    config::AppConfig,
    domain::{geometry::WorldPoint, tracker::TrackerEngineKind},
    error::{ContextExt as _, Result},
    resources::{
        WorkspaceSnapshot, load_logic_map_with_tracking_poi_scaled_image,
        load_logic_map_with_tracking_poi_scaled_rgba_image,
    },
    tracking::debug::{DebugField, DebugImage, DebugImageKind, TrackingDebugSnapshot},
};

const INSCRIBED_SQUARE_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;
const SYNTHETIC_CAPTURE_MAP_RADIUS_RATIO: f32 = 0.84;
const SYNTHETIC_CAPTURE_FRAME_RADIUS_RATIO: f32 = 0.93;
const SYNTHETIC_CAPTURE_BORDER_RADIUS_RATIO: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
struct CaptureSelectionLayout {
    square_side_px: u32,
    source_offset_x: u32,
    source_offset_y: u32,
    circle_diameter_px: u32,
    circle_offset_x: u32,
    circle_offset_y: u32,
    normalized_inner_radius: f32,
}

#[derive(Debug, Clone, Copy)]
struct SyntheticCaptureLayout {
    circle_diameter_px: u32,
    circle_offset_x: u32,
    circle_offset_y: u32,
    map_diameter_px: u32,
    map_inset_px: u32,
    normalized_inner_radius: f32,
}

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
pub struct ScaledColorMap {
    pub scale: u32,
    pub image: RgbaImage,
}

#[derive(Debug, Clone)]
pub struct ColorMapPyramid {
    pub local: ScaledColorMap,
    pub global: ScaledColorMap,
    pub coarse: ScaledColorMap,
}

#[derive(Debug, Clone)]
pub struct MaskSet {
    pub local: GrayImage,
    pub global: GrayImage,
    pub coarse: GrayImage,
}

#[derive(Debug, Clone)]
pub struct ColorCaptureTemplates {
    pub local: RgbaImage,
    pub global: RgbaImage,
    pub coarse: RgbaImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTemplateShape {
    Annulus,
    InnerSquare,
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

#[derive(Debug, Clone)]
pub struct ColorSearchCrop {
    pub image: RgbaImage,
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

#[derive(Debug, Clone, Copy)]
pub struct ScorePeak {
    pub left: u32,
    pub top: u32,
    pub score: f32,
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
    let annulus_inner = normalized_inner_radius(
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );
    let local_size = scaled_template_dimension(config.view_size.max(1), local_scale);
    let global_size = scaled_dimension(config.view_size.max(1), global_scale);
    let coarse_size = scaled_dimension(config.view_size.max(1), coarse_scale);
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
        local: build_inner_square_mask(
            local_size,
            local_size,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
        global: build_mask(global_size, global_size, annulus_inner, 1.0),
        coarse: build_mask(coarse_size, coarse_size, annulus_inner, 1.0),
    };

    Ok((pyramid, masks))
}

pub fn load_logic_color_map_pyramid(workspace: &WorkspaceSnapshot) -> Result<ColorMapPyramid> {
    let config = &workspace.config;
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(config);
    let base_map = load_logic_map_with_tracking_poi_scaled_rgba_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        config.view_size,
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

    Ok(ColorMapPyramid {
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
    })
}

pub fn coarse_global_downscale(config: &crate::config::AppConfig) -> u32 {
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    global_scale
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

pub fn downscale_rgba(image: &RgbaImage, scale: u32) -> RgbaImage {
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

fn capture_selection_layout(
    width: u32,
    height: u32,
    inner_radius: f32,
    outer_radius: f32,
) -> CaptureSelectionLayout {
    let square_side_px = width.max(height).max(1);
    let short_side = width.min(height).max(1);
    let circle_diameter_px = ((short_side as f32) * outer_radius.clamp(0.1, 1.5))
        .round()
        .clamp(1.0, short_side as f32) as u32;

    CaptureSelectionLayout {
        square_side_px,
        source_offset_x: square_side_px.saturating_sub(width) / 2,
        source_offset_y: square_side_px.saturating_sub(height) / 2,
        circle_diameter_px,
        circle_offset_x: square_side_px.saturating_sub(circle_diameter_px) / 2,
        circle_offset_y: square_side_px.saturating_sub(circle_diameter_px) / 2,
        normalized_inner_radius: normalized_inner_radius(inner_radius, outer_radius),
    }
}

fn capture_selection_distance(layout: CaptureSelectionLayout, x: u32, y: u32) -> Option<f32> {
    let local_x = x.checked_sub(layout.circle_offset_x)?;
    let local_y = y.checked_sub(layout.circle_offset_y)?;
    if local_x >= layout.circle_diameter_px || local_y >= layout.circle_diameter_px {
        return None;
    }

    Some(annulus_distance(
        layout.circle_diameter_px,
        layout.circle_diameter_px,
        local_x,
        local_y,
    ))
}

pub(crate) fn capture_selection_outer_square(
    captured: &GrayImage,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    let layout = capture_selection_layout(
        captured.width(),
        captured.height(),
        inner_radius,
        outer_radius,
    );
    GrayImage::from_fn(layout.square_side_px, layout.square_side_px, |x, y| {
        let Some(source_x) = x.checked_sub(layout.source_offset_x) else {
            return Luma([0]);
        };
        let Some(source_y) = y.checked_sub(layout.source_offset_y) else {
            return Luma([0]);
        };
        if source_x >= captured.width() || source_y >= captured.height() {
            return Luma([0]);
        }
        let Some(distance) = capture_selection_distance(layout, x, y) else {
            return Luma([0]);
        };
        if distance > 1.0 || distance < layout.normalized_inner_radius {
            return Luma([0]);
        }

        *captured.get_pixel(source_x, source_y)
    })
}

pub(crate) fn capture_selection_outer_square_rgba(
    captured: &RgbaImage,
    inner_radius: f32,
    outer_radius: f32,
) -> RgbaImage {
    let layout = capture_selection_layout(
        captured.width(),
        captured.height(),
        inner_radius,
        outer_radius,
    );
    RgbaImage::from_fn(layout.square_side_px, layout.square_side_px, |x, y| {
        let Some(source_x) = x.checked_sub(layout.source_offset_x) else {
            return Rgba([0, 0, 0, 0]);
        };
        let Some(source_y) = y.checked_sub(layout.source_offset_y) else {
            return Rgba([0, 0, 0, 0]);
        };
        if source_x >= captured.width() || source_y >= captured.height() {
            return Rgba([0, 0, 0, 0]);
        }
        let Some(distance) = capture_selection_distance(layout, x, y) else {
            return Rgba([0, 0, 0, 0]);
        };
        if distance > 1.0 || distance < layout.normalized_inner_radius {
            return Rgba([0, 0, 0, 0]);
        }

        *captured.get_pixel(source_x, source_y)
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn crop_view_square(image: &GrayImage, view_size: u32, center: (u32, u32)) -> GrayImage {
    let side = view_size.min(image.width()).min(image.height()).max(1);
    let left = center.0.saturating_sub(side / 2).min(image.width() - side);
    let top = center.1.saturating_sub(side / 2).min(image.height() - side);
    crop_imm(image, left, top, side, side).to_image()
}

fn crop_view_square_rgba(image: &RgbaImage, view_size: u32, center: (u32, u32)) -> RgbaImage {
    let side = view_size.min(image.width()).min(image.height()).max(1);
    let left = center.0.saturating_sub(side / 2).min(image.width() - side);
    let top = center.1.saturating_sub(side / 2).min(image.height() - side);
    crop_imm(image, left, top, side, side).to_image()
}

fn synthetic_capture_layout(config: &AppConfig) -> SyntheticCaptureLayout {
    let capture_width = config.minimap.width.max(1);
    let capture_height = config.minimap.height.max(1);
    let capture_short_side = capture_width.min(capture_height).max(1);
    let circle_diameter_px = ((capture_short_side as f32)
        * config.template.mask_outer_radius.clamp(0.1, 1.5))
    .round()
    .clamp(1.0, capture_short_side as f32) as u32;
    let map_diameter_px = ((circle_diameter_px as f32) * SYNTHETIC_CAPTURE_MAP_RADIUS_RATIO)
        .round()
        .clamp(1.0, circle_diameter_px as f32) as u32;

    SyntheticCaptureLayout {
        circle_diameter_px,
        circle_offset_x: capture_width.saturating_sub(circle_diameter_px) / 2,
        circle_offset_y: capture_height.saturating_sub(circle_diameter_px) / 2,
        map_diameter_px,
        map_inset_px: circle_diameter_px.saturating_sub(map_diameter_px) / 2,
        normalized_inner_radius: normalized_inner_radius(
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn mean_gray_value(image: &GrayImage) -> u8 {
    let (sum, count) = image.pixels().fold((0u64, 0u64), |(sum, count), pixel| {
        (sum + u64::from(pixel.0[0]), count + 1)
    });
    if count == 0 { 0 } else { (sum / count) as u8 }
}

fn mean_rgb_value(image: &RgbaImage) -> [u8; 3] {
    let (sum, count) = image
        .pixels()
        .fold(([0u64; 3], 0u64), |(mut sum, count), pixel| {
            sum[0] += u64::from(pixel.0[0]);
            sum[1] += u64::from(pixel.0[1]);
            sum[2] += u64::from(pixel.0[2]);
            (sum, count + 1)
        });
    if count == 0 {
        [0, 0, 0]
    } else {
        [
            (sum[0] / count) as u8,
            (sum[1] / count) as u8,
            (sum[2] / count) as u8,
        ]
    }
}

fn shade_channel(value: u8, factor: f32, lift: f32) -> u8 {
    (f32::from(value) * factor + lift).round().clamp(0.0, 255.0) as u8
}

fn shade_rgb(value: [u8; 3], factor: f32, lift: f32) -> [u8; 3] {
    [
        shade_channel(value[0], factor, lift),
        shade_channel(value[1], factor, lift),
        shade_channel(value[2], factor, lift),
    ]
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn synthesize_runtime_capture_from_map(
    map: &GrayImage,
    config: &AppConfig,
    center: (u32, u32),
) -> GrayImage {
    let crop = crop_view_square(map, config.view_size.max(1), center);
    let layout = synthetic_capture_layout(config);
    let map_square = resize(
        &crop,
        layout.map_diameter_px,
        layout.map_diameter_px,
        FilterType::Triangle,
    );
    let mean = mean_gray_value(&map_square);
    let background = shade_channel(mean, 0.08, 6.0);
    let border = shade_channel(mean, 0.18, 12.0);
    let frame = shade_channel(mean, 0.32, 20.0);
    let hole = shade_channel(mean, 0.60, 72.0);
    let mut canvas = GrayImage::from_pixel(
        config.minimap.width,
        config.minimap.height,
        Luma([background]),
    );

    for local_y in 0..layout.circle_diameter_px {
        for local_x in 0..layout.circle_diameter_px {
            let distance = annulus_distance(
                layout.circle_diameter_px,
                layout.circle_diameter_px,
                local_x,
                local_y,
            );
            if distance > SYNTHETIC_CAPTURE_BORDER_RADIUS_RATIO {
                continue;
            }

            let value = if distance > SYNTHETIC_CAPTURE_FRAME_RADIUS_RATIO {
                border
            } else if distance > SYNTHETIC_CAPTURE_MAP_RADIUS_RATIO {
                frame
            } else {
                let map_x = local_x
                    .saturating_sub(layout.map_inset_px)
                    .min(map_square.width().saturating_sub(1));
                let map_y = local_y
                    .saturating_sub(layout.map_inset_px)
                    .min(map_square.height().saturating_sub(1));
                let source = map_square.get_pixel(map_x, map_y).0[0];
                if distance <= layout.normalized_inner_radius {
                    blend_channel(hole, source, 0.22)
                } else {
                    source
                }
            };

            canvas.put_pixel(
                layout.circle_offset_x + local_x,
                layout.circle_offset_y + local_y,
                Luma([value]),
            );
        }
    }

    canvas
}

pub(crate) fn synthesize_runtime_capture_rgba_from_map(
    map: &RgbaImage,
    config: &AppConfig,
    center: (u32, u32),
) -> RgbaImage {
    let crop = crop_view_square_rgba(map, config.view_size.max(1), center);
    let layout = synthetic_capture_layout(config);
    let map_square = resize(
        &crop,
        layout.map_diameter_px,
        layout.map_diameter_px,
        FilterType::Triangle,
    );
    let mean = mean_rgb_value(&map_square);
    let border = shade_rgb(mean, 0.18, 12.0);
    let frame = shade_rgb(mean, 0.32, 20.0);
    let mut canvas = RgbaImage::from_pixel(
        config.minimap.width,
        config.minimap.height,
        Rgba([0, 0, 0, 0]),
    );

    for local_y in 0..layout.circle_diameter_px {
        for local_x in 0..layout.circle_diameter_px {
            let distance = annulus_distance(
                layout.circle_diameter_px,
                layout.circle_diameter_px,
                local_x,
                local_y,
            );
            if distance > SYNTHETIC_CAPTURE_BORDER_RADIUS_RATIO {
                continue;
            }

            let pixel = if distance > SYNTHETIC_CAPTURE_FRAME_RADIUS_RATIO {
                Rgba([border[0], border[1], border[2], 255])
            } else if distance > SYNTHETIC_CAPTURE_MAP_RADIUS_RATIO {
                Rgba([frame[0], frame[1], frame[2], 255])
            } else {
                let map_x = local_x
                    .saturating_sub(layout.map_inset_px)
                    .min(map_square.width().saturating_sub(1));
                let map_y = local_y
                    .saturating_sub(layout.map_inset_px)
                    .min(map_square.height().saturating_sub(1));
                if distance <= layout.normalized_inner_radius {
                    Rgba([0, 0, 0, 0])
                } else {
                    let source = map_square.get_pixel(map_x, map_y).0;
                    Rgba([source[0], source[1], source[2], 255])
                }
            };

            canvas.put_pixel(
                layout.circle_offset_x + local_x,
                layout.circle_offset_y + local_y,
                pixel,
            );
        }
    }

    canvas
}

pub fn capture_template_annulus(
    captured: &GrayImage,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    capture_selection_outer_square(captured, inner_radius, outer_radius)
}

pub fn capture_template_inner_square(
    captured: &GrayImage,
    inner_radius: f32,
    outer_radius: f32,
) -> GrayImage {
    capture_selection_outer_square(captured, inner_radius, outer_radius)
}

pub fn capture_template_annulus_rgba(
    captured: &RgbaImage,
    inner_radius: f32,
    outer_radius: f32,
) -> RgbaImage {
    capture_selection_outer_square_rgba(captured, inner_radius, outer_radius)
}

pub fn capture_template_inner_square_rgba(
    captured: &RgbaImage,
    inner_radius: f32,
    outer_radius: f32,
) -> RgbaImage {
    capture_selection_outer_square_rgba(captured, inner_radius, outer_radius)
}

fn blend_channel(neutral: u8, source: u8, alpha: f32) -> u8 {
    (f32::from(neutral) * (1.0 - alpha) + f32::from(source) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
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

pub fn normalized_inner_radius(inner_radius: f32, outer_radius: f32) -> f32 {
    if outer_radius <= f32::EPSILON {
        0.0
    } else {
        (inner_radius / outer_radius).clamp(0.0, 1.0)
    }
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
        crate::bail!("search image is smaller than template");
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

pub fn top_score_peaks(
    score_map: &[f32],
    score_width: u32,
    score_height: u32,
    threshold: f32,
    suppression_radius: u32,
    limit: usize,
) -> Vec<ScorePeak> {
    if score_map.is_empty() || score_width == 0 || score_height == 0 || limit == 0 {
        return Vec::new();
    }

    let expected_len = score_width as usize * score_height as usize;
    if score_map.len() < expected_len {
        return Vec::new();
    }

    let mut ranked = score_map
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| *score >= threshold)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let suppression_radius = suppression_radius.max(1) as i64;
    let suppression_distance_sq = suppression_radius * suppression_radius;
    let mut peaks = Vec::with_capacity(limit.min(ranked.len()));

    for (index, score) in ranked {
        let left = index as u32 % score_width;
        let top = index as u32 / score_width;
        let suppressed = peaks.iter().any(|peak: &ScorePeak| {
            let dx = i64::from(left) - i64::from(peak.left);
            let dy = i64::from(top) - i64::from(peak.top);
            dx * dx + dy * dy <= suppression_distance_sq
        });
        if suppressed {
            continue;
        }

        peaks.push(ScorePeak { left, top, score });
        if peaks.len() >= limit {
            break;
        }
    }

    peaks
}

pub fn crop_search_region(image: &GrayImage, region: SearchRegion) -> Result<SearchCrop> {
    if image.width() < region.origin_x.saturating_add(region.width)
        || image.height() < region.origin_y.saturating_add(region.height)
    {
        crate::bail!("search region is outside image bounds");
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

pub fn crop_search_region_rgba(image: &RgbaImage, region: SearchRegion) -> Result<ColorSearchCrop> {
    if image.width() < region.origin_x.saturating_add(region.width)
        || image.height() < region.origin_y.saturating_add(region.height)
    {
        crate::bail!("search region is outside image bounds");
    }

    let image = crop_imm(
        image,
        region.origin_x,
        region.origin_y,
        region.width,
        region.height,
    )
    .to_image();
    Ok(ColorSearchCrop {
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

pub fn crop_around_center_rgba(
    image: &RgbaImage,
    center: (u32, u32),
    radius: u32,
    template_width: u32,
    template_height: u32,
) -> Result<ColorSearchCrop> {
    let region = search_region_around_center(
        image.width(),
        image.height(),
        center,
        radius,
        template_width,
        template_height,
    )?;
    crop_search_region_rgba(image, region)
}

pub fn prepare_color_capture_template(
    captured: &RgbaImage,
    view_size: u32,
    scale: u32,
    inner_radius: f32,
    outer_radius: f32,
    shape: ColorTemplateShape,
) -> RgbaImage {
    let square = match shape {
        ColorTemplateShape::Annulus => {
            capture_template_annulus_rgba(captured, inner_radius, outer_radius)
        }
        ColorTemplateShape::InnerSquare => {
            capture_template_inner_square_rgba(captured, inner_radius, outer_radius)
        }
    };
    let template_size = scaled_dimension(view_size.max(1), scale.max(1));
    if square.width() == template_size && square.height() == template_size {
        square
    } else {
        resize(&square, template_size, template_size, FilterType::Triangle)
    }
}

pub fn scaled_color_score(
    map: &ScaledColorMap,
    world: WorldPoint,
    template: &RgbaImage,
    mask: &GrayImage,
) -> Option<f32> {
    let (left, top) = centered_crop_origin(
        &map.image,
        center_to_scaled(world, map.scale),
        template.width(),
        template.height(),
    )
    .ok()?;
    Some(masked_chroma_similarity_region(
        &map.image, left, top, template, mask,
    ))
}

pub fn scaled_luma_score(
    map: &ScaledColorMap,
    world: WorldPoint,
    template: &RgbaImage,
    mask: &GrayImage,
) -> Option<f32> {
    let (left, top) = centered_crop_origin(
        &map.image,
        center_to_scaled(world, map.scale),
        template.width(),
        template.height(),
    )
    .ok()?;
    Some(masked_luma_similarity_region(
        &map.image, left, top, template, mask,
    ))
}

pub fn masked_chroma_similarity(search: &RgbaImage, template: &RgbaImage, mask: &GrayImage) -> f32 {
    if search.dimensions() != template.dimensions() || search.dimensions() != mask.dimensions() {
        return 0.0;
    }
    masked_chroma_similarity_region(search, 0, 0, template, mask)
}

fn masked_chroma_similarity_region(
    search: &RgbaImage,
    left: u32,
    top: u32,
    template: &RgbaImage,
    mask: &GrayImage,
) -> f32 {
    let mut rgb_dot = 0.0f32;
    let mut rgb_search_norm = 0.0f32;
    let mut rgb_template_norm = 0.0f32;
    let mut chroma_dot = 0.0f32;
    let mut chroma_search_norm = 0.0f32;
    let mut chroma_template_norm = 0.0f32;
    for y in 0..template.height() {
        for x in 0..template.width() {
            let mask_weight = f32::from(mask.get_pixel(x, y).0[0]) / 255.0;
            let search_pixel = search.get_pixel(left + x, top + y).0;
            let template_pixel = template.get_pixel(x, y).0;
            let alpha_weight =
                (f32::from(search_pixel[3]) / 255.0) * (f32::from(template_pixel[3]) / 255.0);
            let effective_weight = mask_weight * alpha_weight;
            if effective_weight <= 0.0 {
                continue;
            }

            let search_rgb = normalized_rgb_linear(search_pixel);
            let template_rgb = normalized_rgb_linear(template_pixel);
            let search_chroma = normalized_rgb(search_pixel);
            let template_chroma = normalized_rgb(template_pixel);
            let chroma_weight = 0.1
                + 0.9
                    * ((pixel_chroma(search_pixel) + pixel_chroma(template_pixel)) * 0.5)
                        .clamp(0.0, 1.0);
            rgb_dot += effective_weight
                * (search_rgb[0] * template_rgb[0]
                    + search_rgb[1] * template_rgb[1]
                    + search_rgb[2] * template_rgb[2]);
            rgb_search_norm += effective_weight
                * (search_rgb[0] * search_rgb[0]
                    + search_rgb[1] * search_rgb[1]
                    + search_rgb[2] * search_rgb[2]);
            rgb_template_norm += effective_weight
                * (template_rgb[0] * template_rgb[0]
                    + template_rgb[1] * template_rgb[1]
                    + template_rgb[2] * template_rgb[2]);

            let weight = effective_weight * chroma_weight;
            chroma_dot += weight
                * (search_chroma[0] * template_chroma[0]
                    + search_chroma[1] * template_chroma[1]
                    + search_chroma[2] * template_chroma[2]);
            chroma_search_norm += weight
                * (search_chroma[0] * search_chroma[0]
                    + search_chroma[1] * search_chroma[1]
                    + search_chroma[2] * search_chroma[2]);
            chroma_template_norm += weight
                * (template_chroma[0] * template_chroma[0]
                    + template_chroma[1] * template_chroma[1]
                    + template_chroma[2] * template_chroma[2]);
        }
    }

    let rgb_score = cosine_score(rgb_dot, rgb_search_norm, rgb_template_norm);
    let chroma_score = cosine_score(chroma_dot, chroma_search_norm, chroma_template_norm);
    if rgb_score <= 1e-6 {
        chroma_score
    } else if chroma_score <= 1e-6 {
        rgb_score
    } else {
        (rgb_score * 0.65 + chroma_score * 0.35).clamp(0.0, 1.0)
    }
}

pub fn masked_luma_similarity(search: &RgbaImage, template: &RgbaImage, mask: &GrayImage) -> f32 {
    if search.dimensions() != template.dimensions() || search.dimensions() != mask.dimensions() {
        return 0.0;
    }
    masked_luma_similarity_region(search, 0, 0, template, mask)
}

fn masked_luma_similarity_region(
    search: &RgbaImage,
    left: u32,
    top: u32,
    template: &RgbaImage,
    mask: &GrayImage,
) -> f32 {
    let mut total_weight = 0.0f32;
    let mut search_sum = 0.0f32;
    let mut template_sum = 0.0f32;
    for y in 0..template.height() {
        for x in 0..template.width() {
            let mask_weight = f32::from(mask.get_pixel(x, y).0[0]) / 255.0;
            let search_pixel = search.get_pixel(left + x, top + y).0;
            let template_pixel = template.get_pixel(x, y).0;
            let alpha_weight =
                (f32::from(search_pixel[3]) / 255.0) * (f32::from(template_pixel[3]) / 255.0);
            let effective_weight = mask_weight * alpha_weight;
            if effective_weight <= 0.0 {
                continue;
            }

            total_weight += effective_weight;
            search_sum += effective_weight * pixel_luma(search_pixel);
            template_sum += effective_weight * pixel_luma(template_pixel);
        }
    }

    if total_weight <= 1e-6 {
        return 0.0;
    }

    let search_mean = search_sum / total_weight;
    let template_mean = template_sum / total_weight;
    let mut dot = 0.0f32;
    let mut search_norm = 0.0f32;
    let mut template_norm = 0.0f32;
    for y in 0..template.height() {
        for x in 0..template.width() {
            let mask_weight = f32::from(mask.get_pixel(x, y).0[0]) / 255.0;
            let search_pixel = search.get_pixel(left + x, top + y).0;
            let template_pixel = template.get_pixel(x, y).0;
            let alpha_weight =
                (f32::from(search_pixel[3]) / 255.0) * (f32::from(template_pixel[3]) / 255.0);
            let effective_weight = mask_weight * alpha_weight;
            if effective_weight <= 0.0 {
                continue;
            }

            let search_value = pixel_luma(search_pixel) - search_mean;
            let template_value = pixel_luma(template_pixel) - template_mean;
            dot += effective_weight * search_value * template_value;
            search_norm += effective_weight * search_value * search_value;
            template_norm += effective_weight * template_value * template_value;
        }
    }

    cosine_score(dot, search_norm, template_norm)
}

fn centered_crop_origin(
    image: &RgbaImage,
    center: (u32, u32),
    width: u32,
    height: u32,
) -> Result<(u32, u32)> {
    if image.width() < width || image.height() < height {
        crate::bail!("color image is smaller than requested crop");
    }

    let left = center
        .0
        .saturating_sub(width / 2)
        .min(image.width() - width);
    let top = center
        .1
        .saturating_sub(height / 2)
        .min(image.height() - height);
    Ok((left, top))
}

fn normalized_rgb(pixel: [u8; 4]) -> [f32; 3] {
    let r = f32::from(pixel[0]) / 255.0;
    let g = f32::from(pixel[1]) / 255.0;
    let b = f32::from(pixel[2]) / 255.0;
    let sum = r + g + b;
    if sum <= 1e-6 {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    } else {
        [r / sum, g / sum, b / sum]
    }
}

fn normalized_rgb_linear(pixel: [u8; 4]) -> [f32; 3] {
    [
        f32::from(pixel[0]) / 255.0,
        f32::from(pixel[1]) / 255.0,
        f32::from(pixel[2]) / 255.0,
    ]
}

fn pixel_luma(pixel: [u8; 4]) -> f32 {
    (f32::from(pixel[0]) * 77.0 + f32::from(pixel[1]) * 150.0 + f32::from(pixel[2]) * 29.0)
        / (255.0 * 256.0)
}

fn pixel_chroma(pixel: [u8; 4]) -> f32 {
    let max = pixel[0].max(pixel[1]).max(pixel[2]) as f32;
    let min = pixel[0].min(pixel[1]).min(pixel[2]) as f32;
    (max - min) / 255.0
}

fn cosine_score(dot: f32, left_norm: f32, right_norm: f32) -> f32 {
    if left_norm <= 1e-6 || right_norm <= 1e-6 {
        0.0
    } else {
        (dot / (left_norm * right_norm).sqrt()).clamp(0.0, 1.0)
    }
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

#[must_use]
pub fn rgba_image_as_unit_vec(image: &RgbaImage) -> Vec<f32> {
    let plane_len = image.width() as usize * image.height() as usize;
    let mut values = Vec::with_capacity(plane_len * 3);
    for channel in 0..3 {
        values.extend(image.pixels().map(|pixel| {
            let alpha = f32::from(pixel.0[3]) / 255.0;
            (f32::from(pixel.0[channel]) / 255.0) * alpha
        }));
    }
    values
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

        assert!(center <= 6, "center hole should stay blank, got {center}");
        assert!(corner <= 6, "outer corner should stay blank, got {corner}");
        assert!(
            (i32::from(ring) - 122).abs() <= 10,
            "ring texture should stay intact, got {ring}"
        );
    }

    #[test]
    fn capture_template_inner_square_returns_selection_outer_square() {
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
        let expected = capture_selection_outer_square(&image, 0.16, 1.0);
        let center = square.get_pixel(square.width() / 2, square.height() / 2).0[0];

        assert_eq!(square.width(), 121);
        assert_eq!(square.height(), 121);
        assert_eq!(square, expected);
        assert_eq!(square.get_pixel(0, 0).0[0], 0);
        assert!(center <= 6, "center hole should stay blank, got {center}");
    }

    #[test]
    fn inner_square_mask_keeps_corners_and_cuts_center_hole() {
        let mask = build_inner_square_mask(71, 71, 0.22, 1.0);

        assert_eq!(mask.get_pixel(0, 0).0[0], 255);
        assert_eq!(mask.get_pixel(mask.width() / 2, mask.height() / 2).0[0], 0);
        assert_eq!(mask.get_pixel(mask.width() / 2, 0).0[0], 255);
    }

    #[test]
    fn capture_selection_outer_square_pads_to_square_and_blanks_exterior() {
        let image = GrayImage::from_pixel(12, 12, Luma([91]));
        let square = capture_selection_outer_square(&image, 0.25, 1.0);

        assert_eq!(square.width(), 12);
        assert_eq!(square.height(), 12);
        assert_eq!(square.get_pixel(0, 0).0[0], 0);
        assert_eq!(
            square.get_pixel(square.width() / 2, square.height() / 2).0[0],
            0
        );
        assert_ne!(
            square.get_pixel(square.width() / 2, 2).0[0],
            0,
            "ring pixels should survive the approximate outer-square preprocessing",
        );
    }

    #[test]
    fn capture_selection_outer_square_rgba_keeps_outer_and_center_transparent() {
        let image = RgbaImage::from_pixel(128, 128, Rgba([90, 120, 160, 255]));
        let square = capture_selection_outer_square_rgba(&image, 0.18, 1.0);

        assert_eq!(square.width(), 128);
        assert_eq!(square.height(), 128);
        assert_eq!(square.get_pixel(0, 0).0[3], 0);
        assert_eq!(
            square.get_pixel(square.width() / 2, square.height() / 2).0[3],
            0
        );
        assert_eq!(
            square.get_pixel(square.width() / 2, square.height() / 4).0[3],
            255
        );
    }

    #[test]
    fn synthetic_runtime_capture_rgba_keeps_outer_and_center_transparent() {
        let config = AppConfig {
            view_size: 96,
            minimap: crate::config::CaptureRegion {
                top: 0,
                left: 0,
                width: 160,
                height: 120,
            },
            template: crate::config::TemplateTrackingConfig {
                mask_inner_radius: 0.16,
                mask_outer_radius: 80.0 / 120.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let map = RgbaImage::from_fn(256, 256, |x, y| {
            Rgba([
                ((x * 11 + y * 7) % 251) as u8,
                ((x * 13 + y * 5) % 251) as u8,
                ((x * 17 + y * 3) % 251) as u8,
                255,
            ])
        });
        let capture = synthesize_runtime_capture_rgba_from_map(&map, &config, (128, 128));
        let layout = synthetic_capture_layout(&config);
        let center_x = layout.circle_offset_x + layout.circle_diameter_px / 2;
        let center_y = layout.circle_offset_y + layout.circle_diameter_px / 2;
        let ring_x = center_x;
        let ring_y = layout.circle_offset_y + layout.circle_diameter_px / 4;

        assert_eq!(capture.get_pixel(0, 0).0[3], 0);
        assert_eq!(capture.get_pixel(center_x, center_y).0[3], 0);
        assert_eq!(capture.get_pixel(ring_x, ring_y).0[3], 255);
    }

    #[test]
    fn capture_template_inner_square_uses_selection_outer_square_geometry() {
        let config = AppConfig {
            view_size: 96,
            minimap: crate::config::CaptureRegion {
                top: 0,
                left: 0,
                width: 160,
                height: 120,
            },
            template: crate::config::TemplateTrackingConfig {
                mask_inner_radius: 0.16,
                mask_outer_radius: 80.0 / 120.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let map = GrayImage::from_fn(256, 256, |x, y| Luma([((x * 11 + y * 7) % 251) as u8]));
        let capture = synthesize_runtime_capture_from_map(&map, &config, (128, 128));
        let square = capture_template_inner_square(
            &capture,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        );

        assert_eq!(square.width(), 160);
        assert_eq!(square.height(), 160);
        assert_eq!(square.get_pixel(0, 0).0[0], 0);
        assert_eq!(
            square.get_pixel(square.width() / 2, square.height() / 2).0[0],
            0
        );
    }

    #[test]
    fn capture_template_inner_square_keeps_off_center_selection_square() {
        let circle_left = 4u32;
        let circle_top = 6u32;
        let circle_diameter = 104u32;
        let capture = GrayImage::from_fn(120, 116, |x, y| {
            let local_x = x as i32 - circle_left as i32;
            let local_y = y as i32 - circle_top as i32;
            if local_x < 0
                || local_y < 0
                || local_x >= circle_diameter as i32
                || local_y >= circle_diameter as i32
            {
                return Luma([8]);
            }

            let distance = annulus_distance(
                circle_diameter,
                circle_diameter,
                local_x as u32,
                local_y as u32,
            );
            let value = if distance <= 0.16 {
                244
            } else if distance <= 0.86 {
                (((local_x as u32) * 9 + (local_y as u32) * 7) % 251) as u8
            } else if distance <= 0.93 {
                68
            } else if distance <= 1.0 {
                26
            } else {
                8
            };
            Luma([value])
        });

        let square = capture_template_inner_square(&capture, 0.16, 1.0);
        let expected = capture_selection_outer_square(&capture, 0.16, 1.0);

        assert_eq!(square.width(), 120);
        assert_eq!(square.height(), 120);
        assert_eq!(square, expected);
    }

    #[test]
    fn prepare_color_capture_template_scales_inner_square_to_template_dimension() {
        let image = RgbaImage::from_pixel(128, 128, Rgba([90, 120, 160, 255]));
        let template = prepare_color_capture_template(
            &image,
            120,
            2,
            0.18,
            1.0,
            ColorTemplateShape::InnerSquare,
        );

        assert_eq!(template.width(), scaled_dimension(120, 2));
        assert_eq!(template.height(), scaled_dimension(120, 2));
    }

    #[test]
    fn masked_luma_similarity_prefers_structurally_matching_patch() {
        let mask = GrayImage::from_pixel(4, 4, Luma([255]));
        let template = RgbaImage::from_fn(4, 4, |x, y| {
            let value = ((x * 41 + y * 67) % 255) as u8;
            Rgba([value, value, value, 255])
        });
        let shifted = RgbaImage::from_fn(4, 4, |x, y| {
            let value = (((x * 41 + y * 67) % 255) as u16 + 18).min(255) as u8;
            Rgba([value, value, value, 255])
        });
        let mismatched = RgbaImage::from_fn(4, 4, |x, y| {
            let value = ((x * 19 + y * 11 + 127) % 255) as u8;
            Rgba([value, value, value, 255])
        });

        let shifted_score = masked_luma_similarity(&shifted, &template, &mask);
        let mismatched_score = masked_luma_similarity(&mismatched, &template, &mask);

        assert!(
            shifted_score > 0.98,
            "expected high luma similarity for shifted match, got {shifted_score}"
        );
        assert!(
            mismatched_score + 0.10 < shifted_score,
            "expected structure mismatch to score lower, shifted={shifted_score} mismatched={mismatched_score}"
        );
    }

    #[test]
    fn scaled_luma_score_crops_centered_patch() {
        let map = ScaledColorMap {
            scale: 1,
            image: RgbaImage::from_fn(8, 8, |x, y| {
                let value = ((x * 21 + y * 17) % 255) as u8;
                Rgba([value, value, value, 255])
            }),
        };
        let template = image::imageops::crop_imm(&map.image, 2, 2, 4, 4).to_image();
        let mask = GrayImage::from_pixel(4, 4, Luma([255]));

        let score = scaled_luma_score(&map, WorldPoint::new(4.0, 4.0), &template, &mask)
            .expect("expected centered luma score");

        assert!(
            score > 0.99,
            "expected exact centered crop match, got {score}"
        );
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
