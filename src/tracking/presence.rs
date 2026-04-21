use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use image::{
    DynamicImage, GrayImage, Luma, Rgba, RgbaImage,
    imageops::{FilterType, crop_imm, resize},
};
use imageproc::{
    contrast::equalize_histogram,
    distance_transform::Norm,
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut},
    morphology::{close, open},
    rect::Rect,
    region_labelling::{Connectivity, connected_components},
};
use serde::{Deserialize, Serialize};

use crate::{
    config::CaptureRegion,
    error::{ContextExt as _, Result},
    resources::WorkspaceSnapshot,
    tracking::{
        capture::DesktopCapture,
        debug::{DebugField, DebugImage},
    },
};

const EXPECTED_LABEL_COUNT: usize = 6;
const TAG_NAMES: [&str; EXPECTED_LABEL_COUNT] = ["F1", "F2", "F3", "F4", "J", "P"];
const RAW_TEMPLATE_WIDTH: u32 = 48;
const RAW_TEMPLATE_HEIGHT: u32 = 32;
const RAW_TEMPLATE_AREA: usize = (RAW_TEMPLATE_WIDTH * RAW_TEMPLATE_HEIGHT) as usize;
const SLOT_RAW_THRESHOLD: f32 = 0.90;
const PRESENT_MIN_RAW: f32 = 0.90;
pub const DEFAULT_MEAN_RAW_THRESHOLD: f32 = 0.96;
pub const MODEL_FILE_NAME: &str = "f1p_presence_model.json";
const CANONICAL_TARGET_WIDTH: u32 = 587;
const CANONICAL_TARGET_HEIGHT: u32 = 36;
const CANONICAL_ANCHORS: [(u32, u32, u32, u32); EXPECTED_LABEL_COUNT] = [
    (9, 3, 44, 30),
    (122, 3, 44, 30),
    (228, 3, 44, 30),
    (336, 3, 44, 30),
    (443, 3, 39, 30),
    (542, 3, 38, 30),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimapPresenceModel {
    pub target_width: u32,
    pub target_height: u32,
    pub anchors: Vec<ProbeBox>,
    pub raw_templates: Vec<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl ProbeBox {
    fn x2(&self) -> u32 {
        self.x + self.w
    }

    fn y2(&self) -> u32 {
        self.y + self.h
    }

    fn center_x(&self) -> f32 {
        self.x as f32 + self.w as f32 * 0.5
    }

    fn center_y(&self) -> f32 {
        self.y as f32 + self.h as f32 * 0.5
    }

    fn validate_in_bounds(&self, image_width: u32, image_height: u32) -> Result<()> {
        if self.w == 0 || self.h == 0 {
            crate::bail!("检测槽位尺寸不能为 0");
        }
        if self.x2() > image_width || self.y2() > image_height {
            crate::bail!(
                "检测槽位 ({}, {}, {}, {}) 超出了目标图尺寸 {}x{}",
                self.x,
                self.y,
                self.w,
                self.h,
                image_width,
                image_height
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MinimapPresenceSample {
    pub present: bool,
    pub score: f32,
    pub mean_raw_score: f32,
    pub min_raw_score: f32,
    pub threshold: f32,
    pub current_raw_preview: RgbaImage,
    pub current_modeled_preview: RgbaImage,
    pub slot_scores: Vec<f32>,
    pub slot_found: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct MinimapPresenceModelBuild {
    pub model: MinimapPresenceModel,
    pub sample: MinimapPresenceSample,
}

pub struct MinimapPresenceDetector {
    capture: DesktopCapture,
    prepared: PreparedMinimapPresenceDetector,
}

#[derive(Debug, Clone)]
struct PreparedMinimapPresenceDetector {
    model: MinimapPresenceModel,
    prepared_templates: Vec<Vec<Vec<f32>>>,
    threshold: f32,
}

#[derive(Debug, Clone, Copy)]
struct ProbePalette {
    background: [f32; 3],
    background_luma: f32,
    support: [f32; 3],
    support_luma: f32,
    support_gap: f32,
    support_radius: f32,
}

#[derive(Debug, Clone)]
struct ComponentStats {
    area: u32,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl ComponentStats {
    fn width(&self) -> u32 {
        self.right.saturating_sub(self.left) + 1
    }

    fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top) + 1
    }

    fn fill_ratio(&self) -> f32 {
        self.area as f32 / (self.width().max(1) * self.height().max(1)) as f32
    }

    fn into_probe_box(self) -> ProbeBox {
        ProbeBox {
            x: self.left,
            y: self.top,
            w: self.width(),
            h: self.height(),
        }
    }
}

impl PreparedMinimapPresenceDetector {
    fn from_model(model: MinimapPresenceModel, threshold: f32) -> Result<Self> {
        validate_model(&model)?;
        let prepared_templates = model
            .raw_templates
            .iter()
            .map(|bank| {
                bank.iter()
                    .map(|raw| prepare_centered_template(raw))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            model,
            prepared_templates,
            threshold,
        })
    }

    fn sample_image(&self, image: &RgbaImage) -> Result<MinimapPresenceSample> {
        let normalized = resize_to_target(
            image,
            self.model.target_width.max(1),
            self.model.target_height.max(1),
        );
        let normalized_gray = DynamicImage::ImageRgba8(normalized.clone()).into_luma8();
        let mut slot_scores = Vec::with_capacity(EXPECTED_LABEL_COUNT);
        for (anchor, prepared_bank) in self.model.anchors.iter().zip(&self.prepared_templates) {
            let raw_crop =
                crop_imm(&normalized_gray, anchor.x, anchor.y, anchor.w, anchor.h).to_image();
            let processed = preprocess_raw_crop(&raw_crop);
            let score = prepared_bank
                .iter()
                .map(|prepared_template| {
                    grayscale_similarity_score_prepared(&processed, prepared_template)
                })
                .fold(f32::NEG_INFINITY, f32::max);
            let score = if score.is_finite() { score } else { 0.0 };
            slot_scores.push(score);
        }

        let mean_raw_score = if slot_scores.is_empty() {
            0.0
        } else {
            slot_scores.iter().sum::<f32>() / slot_scores.len() as f32
        };
        let min_raw_score = slot_scores.iter().copied().fold(f32::INFINITY, f32::min);
        let min_raw_score = if min_raw_score.is_finite() {
            min_raw_score
        } else {
            0.0
        };
        let score = 0.65 * mean_raw_score + 0.35 * min_raw_score;
        let slot_found = slot_scores
            .iter()
            .map(|score| *score >= SLOT_RAW_THRESHOLD)
            .collect::<Vec<_>>();
        let present = min_raw_score >= PRESENT_MIN_RAW && mean_raw_score >= self.threshold;
        let current_modeled_preview =
            draw_modeled_preview(&normalized, &self.model.anchors, &slot_scores, present);

        Ok(MinimapPresenceSample {
            present,
            score,
            mean_raw_score,
            min_raw_score,
            threshold: self.threshold,
            current_raw_preview: normalized,
            current_modeled_preview,
            slot_scores,
            slot_found,
        })
    }
}

impl MinimapPresenceDetector {
    pub fn new(workspace: &WorkspaceSnapshot) -> Result<Option<Self>> {
        let probe = &workspace.config.minimap_presence_probe;
        if !probe.enabled {
            return Ok(None);
        }

        let region = probe.capture_region().ok_or_else(|| {
            crate::app_error!(
                "minimap_presence_probe 已启用，但 top/left/width/height 还没有完整配置"
            )
        })?;
        let capture = DesktopCapture::from_absolute_region(&region)?;
        let model = load_minimap_presence_model(&workspace.project_root)?;
        let prepared = PreparedMinimapPresenceDetector::from_model(model, probe.match_threshold)?;

        Ok(Some(Self { capture, prepared }))
    }

    pub fn sample(&self) -> Result<MinimapPresenceSample> {
        let captured = self.capture.capture_rgba()?;
        self.prepared.sample_image(&captured)
    }

    #[must_use]
    pub fn debug_images(&self, sample: &MinimapPresenceSample) -> Vec<DebugImage> {
        vec![
            preview_rgba_image("Probe Live", &sample.current_raw_preview, 196),
            preview_rgba_image("Probe Modeled", &sample.current_modeled_preview, 196),
        ]
    }

    #[must_use]
    pub fn debug_fields(&self, sample: &MinimapPresenceSample) -> Vec<DebugField> {
        vec![
            DebugField::new(
                "探针状态",
                if sample.present { "Present" } else { "Missing" },
            ),
            DebugField::new("探针得分", format!("{:.3}", sample.score)),
            DebugField::new("均值得分", format!("{:.3}", sample.mean_raw_score)),
            DebugField::new("最低得分", format!("{:.3}", sample.min_raw_score)),
            DebugField::new("均值阈值", format!("{:.3}", sample.threshold)),
            DebugField::new("槽位命中", format!("{}/6", count_true(&sample.slot_found))),
            DebugField::new("槽位分数", format_slot_scores(&sample.slot_scores)),
        ]
    }
}

pub fn build_minimap_presence_probe_model(
    region: &CaptureRegion,
) -> Result<MinimapPresenceModelBuild> {
    if region.width < 12 || region.height < 12 {
        crate::bail!("F1-P 标签区域过小，至少需要 12x12");
    }

    let capture = DesktopCapture::from_absolute_region(region)?;
    let image = capture.capture_rgba()?;
    build_minimap_presence_model_from_image(&image)
}

pub fn build_minimap_presence_model_from_image(
    image: &RgbaImage,
) -> Result<MinimapPresenceModelBuild> {
    if image.width() < 12 || image.height() < 12 {
        crate::bail!("F1-P 标签建模输入过小，至少需要 12x12");
    }

    let anchors = canonical_anchor_boxes(image.width(), image.height());
    let gray = DynamicImage::ImageRgba8(image.clone()).into_luma8();
    let raw_templates = anchors
        .iter()
        .map(|anchor| {
            let crop = crop_imm(&gray, anchor.x, anchor.y, anchor.w, anchor.h).to_image();
            Ok(vec![preprocess_raw_crop(&crop).into_raw()])
        })
        .collect::<Result<Vec<_>>>()?;

    let model = MinimapPresenceModel {
        target_width: image.width(),
        target_height: image.height(),
        anchors,
        raw_templates,
    };
    let prepared =
        PreparedMinimapPresenceDetector::from_model(model.clone(), DEFAULT_MEAN_RAW_THRESHOLD)?;
    let sample = prepared.sample_image(image)?;

    if !sample.present {
        crate::bail!(
            "当前选区未通过 F1-P 模型自检，均值 {:.3}，最低 {:.3}",
            sample.mean_raw_score,
            sample.min_raw_score
        );
    }

    Ok(MinimapPresenceModelBuild { model, sample })
}

pub fn build_minimap_presence_model_from_images(
    images: &[RgbaImage],
) -> Result<MinimapPresenceModelBuild> {
    if images.is_empty() {
        crate::bail!("F1-P 标签建模输入不能为空");
    }

    let target_width = images[0].width();
    let target_height = images[0].height();
    if target_width < 12 || target_height < 12 {
        crate::bail!("F1-P 标签建模输入过小，至少需要 12x12");
    }

    let normalized_images = images
        .iter()
        .map(|image| resize_to_target(image, target_width.max(1), target_height.max(1)))
        .collect::<Vec<_>>();
    let anchors = anchor_boxes_from_seed_images(&normalized_images)?;
    let mut raw_templates = vec![Vec::<Vec<u8>>::new(); EXPECTED_LABEL_COUNT];
    for image in &normalized_images {
        let gray = DynamicImage::ImageRgba8(image.clone()).into_luma8();
        for (slot_index, anchor) in anchors.iter().enumerate() {
            let crop = crop_imm(&gray, anchor.x, anchor.y, anchor.w, anchor.h).to_image();
            raw_templates[slot_index].push(preprocess_raw_crop(&crop).into_raw());
        }
    }

    let model = MinimapPresenceModel {
        target_width,
        target_height,
        anchors,
        raw_templates,
    };
    let prepared =
        PreparedMinimapPresenceDetector::from_model(model.clone(), DEFAULT_MEAN_RAW_THRESHOLD)?;
    let sample = prepared.sample_image(&normalized_images[0])?;

    if !sample.present {
        crate::bail!(
            "当前选区未通过 F1-P 模型自检，均值 {:.3}，最低 {:.3}",
            sample.mean_raw_score,
            sample.min_raw_score
        );
    }

    Ok(MinimapPresenceModelBuild { model, sample })
}

pub fn validate_minimap_presence_probe_region(region: &CaptureRegion) -> Result<()> {
    build_minimap_presence_probe_model(region).map(|_| ())
}

#[must_use]
pub fn minimap_presence_model_path(project_root: &Path) -> PathBuf {
    project_root
        .join("cache")
        .join("tracking")
        .join(MODEL_FILE_NAME)
}

pub fn save_minimap_presence_model(
    project_root: &Path,
    model: &MinimapPresenceModel,
) -> Result<PathBuf> {
    validate_model(model)?;
    let final_path = minimap_presence_model_path(project_root);
    let temp_path = final_path.with_extension("json.tmp");
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create model parent {}", parent.display()))?;
    }

    let raw =
        serde_json::to_vec_pretty(model).context("failed to serialize F1-P presence model")?;
    fs::write(&temp_path, raw)
        .with_context(|| format!("failed to stage F1-P model at {}", temp_path.display()))?;

    if final_path.exists() {
        fs::remove_file(&final_path).with_context(|| {
            format!(
                "failed to replace previous F1-P model {}",
                final_path.display()
            )
        })?;
    }

    fs::rename(&temp_path, &final_path)
        .with_context(|| format!("failed to install F1-P model {}", final_path.display()))?;
    Ok(final_path)
}

pub fn load_minimap_presence_model(project_root: &Path) -> Result<MinimapPresenceModel> {
    let path = minimap_presence_model_path(project_root);
    let raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let model = serde_json::from_str::<MinimapPresenceModel>(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    validate_model(&model)?;
    Ok(model)
}

pub fn delete_minimap_presence_model(project_root: &Path) -> Result<bool> {
    let path = minimap_presence_model_path(project_root);
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path)
        .with_context(|| format!("failed to delete F1-P model {}", path.display()))?;
    Ok(true)
}

fn validate_model(model: &MinimapPresenceModel) -> Result<()> {
    if model.target_width == 0 || model.target_height == 0 {
        crate::bail!("F1-P 模型尺寸无效");
    }
    if model.anchors.len() != EXPECTED_LABEL_COUNT {
        crate::bail!(
            "F1-P 模型槽位数量无效，期望 {}，实际 {}",
            EXPECTED_LABEL_COUNT,
            model.anchors.len()
        );
    }
    if model.raw_templates.len() != EXPECTED_LABEL_COUNT {
        crate::bail!(
            "F1-P 模板数量无效，期望 {}，实际 {}",
            EXPECTED_LABEL_COUNT,
            model.raw_templates.len()
        );
    }
    for anchor in &model.anchors {
        anchor.validate_in_bounds(model.target_width, model.target_height)?;
    }
    for (index, bank) in model.raw_templates.iter().enumerate() {
        if bank.is_empty() {
            crate::bail!(
                "F1-P 模板 {} 为空，至少需要一个槽位模板",
                TAG_NAMES.get(index).copied().unwrap_or("?")
            );
        }
        for raw in bank {
            if raw.len() != RAW_TEMPLATE_AREA {
                crate::bail!(
                    "F1-P 模板 {} 大小无效，期望 {} 字节，实际 {} 字节",
                    TAG_NAMES.get(index).copied().unwrap_or("?"),
                    RAW_TEMPLATE_AREA,
                    raw.len()
                );
            }
        }
    }
    Ok(())
}

fn anchor_boxes_from_seed_images(images: &[RgbaImage]) -> Result<Vec<ProbeBox>> {
    let seed_sets = images
        .iter()
        .filter_map(|image| seed_boxes_from_image(image).ok())
        .collect::<Vec<_>>();
    if seed_sets.is_empty() {
        crate::bail!("当前样本中没有任何一张成功提取到稳定的 F1-P 锚框");
    }

    let mut anchors = Vec::with_capacity(EXPECTED_LABEL_COUNT);
    for slot_index in 0..EXPECTED_LABEL_COUNT {
        anchors.push(ProbeBox {
            x: median_u32(
                &seed_sets
                    .iter()
                    .map(|seed| seed[slot_index].x)
                    .collect::<Vec<_>>(),
            ),
            y: median_u32(
                &seed_sets
                    .iter()
                    .map(|seed| seed[slot_index].y)
                    .collect::<Vec<_>>(),
            ),
            w: median_u32(
                &seed_sets
                    .iter()
                    .map(|seed| seed[slot_index].w)
                    .collect::<Vec<_>>(),
            ),
            h: median_u32(
                &seed_sets
                    .iter()
                    .map(|seed| seed[slot_index].h)
                    .collect::<Vec<_>>(),
            ),
        });
    }
    Ok(anchors)
}

fn canonical_anchor_boxes(target_width: u32, target_height: u32) -> Vec<ProbeBox> {
    let scale_x = target_width.max(1) as f32 / CANONICAL_TARGET_WIDTH as f32;
    let scale_y = target_height.max(1) as f32 / CANONICAL_TARGET_HEIGHT as f32;
    CANONICAL_ANCHORS
        .iter()
        .map(|(x, y, w, h)| {
            let width = ((*w as f32 * scale_x).round() as u32).max(1);
            let height = ((*h as f32 * scale_y).round() as u32).max(1);
            let left =
                ((*x as f32 * scale_x).round() as u32).min(target_width.saturating_sub(width));
            let top =
                ((*y as f32 * scale_y).round() as u32).min(target_height.saturating_sub(height));
            ProbeBox {
                x: left,
                y: top,
                w: width,
                h: height,
            }
        })
        .collect()
}

fn seed_boxes_from_image(image: &RgbaImage) -> Result<Vec<ProbeBox>> {
    let palette = derive_palette(image);
    let support_mask = detect_support_mask(image, &palette);
    let components = filter_components(&support_mask);
    if components.len() != EXPECTED_LABEL_COUNT {
        crate::bail!(
            "当前选区只提取到了 {} 个标签候选，需要准确提取到 6 个标签；请只框住 F1 到 P 这排标签，不要包含上方图标",
            components.len()
        );
    }
    validate_anchor_layout(&components, image.width(), image.height())?;
    Ok(components)
}

fn derive_palette(image: &RgbaImage) -> ProbePalette {
    let border = border_rgb_samples(image);
    let background = mean_rgb_lower_luma_fraction(&border, 0.55).unwrap_or([60.0, 110.0, 188.0]);
    let background_luma = rgb_luma(background);

    let mut rough_mask = GrayImage::new(image.width(), image.height());
    for (x, y, pixel) in image.enumerate_pixels() {
        let rgb = rgb_channels(pixel);
        let luma = rgb_luma(rgb);
        let background_distance = color_distance(rgb, background);
        let spread = rgb_channel_spread(rgb);
        let keep = ((luma >= background_luma + 32.0) && (background_distance >= 18.0))
            || ((luma >= background_luma + 20.0) && (spread <= 26.0));
        if keep {
            rough_mask.put_pixel(x, y, Luma([255]));
        }
    }
    let rough_mask = morphological_cleanup(&rough_mask);
    let support_samples = mask_rgb_samples(image, &rough_mask);
    let support = mean_rgb(&support_samples).unwrap_or([244.0, 236.0, 220.0]);
    let support_luma = rgb_luma(support);
    let support_gap = color_distance(support, background).max(1.0);
    let support_distances = color_distances(&support_samples, support);
    let support_radius = percentile_f32(&support_distances, 0.93)
        .max(support_gap * 0.18)
        .max(14.0)
        + 6.0;

    ProbePalette {
        background,
        background_luma,
        support,
        support_luma,
        support_gap,
        support_radius,
    }
}

fn detect_support_mask(image: &RgbaImage, palette: &ProbePalette) -> GrayImage {
    let min_luma =
        palette.background_luma + (palette.support_luma - palette.background_luma) * 0.28;
    let min_background_distance = (palette.support_gap * 0.12).max(10.0);
    let mut mask = GrayImage::new(image.width(), image.height());

    for (x, y, pixel) in image.enumerate_pixels() {
        let rgb = rgb_channels(pixel);
        let luma = rgb_luma(rgb);
        let support_distance = color_distance(rgb, palette.support);
        let background_distance = color_distance(rgb, palette.background);
        let spread = rgb_channel_spread(rgb);
        let blue_excess = rgb[2] - (rgb[0] + rgb[1]) * 0.5;
        let keep = (luma >= min_luma)
            && (background_distance >= min_background_distance * 0.60)
            && (((support_distance <= palette.support_radius * 2.10)
                && (support_distance <= background_distance * 1.45 + 18.0))
                || (blue_excess <= 10.0)
                || ((luma >= min_luma - 10.0) && (spread <= 26.0)));
        if keep {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    morphological_cleanup(&mask)
}

fn morphological_cleanup(mask: &GrayImage) -> GrayImage {
    open(&close(mask, Norm::LInf, 1), Norm::LInf, 1)
}

fn filter_components(mask: &GrayImage) -> Vec<ProbeBox> {
    let labels = connected_components(mask, Connectivity::Eight, Luma([0]));
    let image_area = (mask.width().max(1) * mask.height().max(1)) as f32;
    let mut components = HashMap::<u32, ComponentStats>::new();

    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel.0[0];
        if label == 0 {
            continue;
        }
        components
            .entry(label)
            .and_modify(|component| {
                component.area += 1;
                component.left = component.left.min(x);
                component.top = component.top.min(y);
                component.right = component.right.max(x);
                component.bottom = component.bottom.max(y);
            })
            .or_insert(ComponentStats {
                area: 1,
                left: x,
                top: y,
                right: x,
                bottom: y,
            });
    }

    let min_width = mask.width() as f32 * 0.018;
    let max_width = mask.width() as f32 * 0.26;
    let min_height = mask.height() as f32 * 0.18;
    let max_height = mask.height() as f32;

    let mut filtered = components
        .into_values()
        .filter(|component| {
            let width = component.width() as f32;
            let height = component.height() as f32;
            let area = component.area as f32;
            area >= image_area * 0.0025
                && area <= image_area * 0.24
                && width >= min_width
                && width <= max_width
                && height >= min_height
                && height <= max_height
                && component.fill_ratio() >= 0.10
        })
        .map(ComponentStats::into_probe_box)
        .collect::<Vec<_>>();

    filtered.sort_by(|left, right| left.center_x().total_cmp(&right.center_x()));
    filtered
}

fn validate_anchor_layout(boxes: &[ProbeBox], image_width: u32, image_height: u32) -> Result<()> {
    if boxes.len() != EXPECTED_LABEL_COUNT {
        crate::bail!("F1-P 标签槽位数量不正确");
    }

    let y_centers = boxes.iter().map(ProbeBox::center_y).collect::<Vec<_>>();
    let y_center_mean = y_centers.iter().sum::<f32>() / y_centers.len() as f32;
    let y_center_variance = y_centers
        .iter()
        .map(|value| {
            let delta = *value - y_center_mean;
            delta * delta
        })
        .sum::<f32>()
        / y_centers.len() as f32;
    let y_center_deviation = y_center_variance.sqrt() / image_height.max(1) as f32;
    if y_center_deviation > 0.08 {
        crate::bail!("当前选区中的标签没有稳定落在同一行，请只框住 F1 到 P 标签带");
    }

    let gaps = boxes
        .windows(2)
        .map(|pair| pair[1].x as f32 - pair[0].x2() as f32)
        .collect::<Vec<_>>();
    if gaps.iter().any(|gap| *gap < 4.0) {
        crate::bail!("当前选区中的标签间隔异常，疑似框入了错误区域");
    }

    let gap_cv = coefficient_of_variation(&gaps);
    if gap_cv > 0.45 {
        crate::bail!("当前选区中的标签间隔不稳定，请重新框选仅包含整排标签的区域");
    }

    let widths = boxes.iter().map(|item| item.w as f32).collect::<Vec<_>>();
    if coefficient_of_variation(&widths) > 0.25 {
        crate::bail!("当前选区中的标签宽度波动过大，疑似不是标准 F1-P 标签带");
    }

    let span_ratio = (boxes.last().map(ProbeBox::x2).unwrap_or(0)
        - boxes.first().map(|item| item.x).unwrap_or(0)) as f32
        / image_width.max(1) as f32;
    if !(0.45..=0.99).contains(&span_ratio) {
        crate::bail!("当前选区中的标签横向覆盖范围异常，请重新框选");
    }

    Ok(())
}

fn preprocess_raw_crop(gray_crop: &GrayImage) -> GrayImage {
    let resized = resize(
        gray_crop,
        RAW_TEMPLATE_WIDTH,
        RAW_TEMPLATE_HEIGHT,
        FilterType::Triangle,
    );
    equalize_histogram(&resized)
}

fn prepare_centered_template(raw: &[u8]) -> Result<Vec<f32>> {
    if raw.len() != RAW_TEMPLATE_AREA {
        crate::bail!(
            "F1-P 槽位模板大小不正确，期望 {} 字节，实际 {} 字节",
            RAW_TEMPLATE_AREA,
            raw.len()
        );
    }

    let mut normalized = raw
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect::<Vec<_>>();
    let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
    for value in &mut normalized {
        *value -= mean;
    }
    let norm = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= 1e-6 {
        crate::bail!("F1-P 槽位模板对比度过低，无法建立稳定模板");
    }
    for value in &mut normalized {
        *value /= norm;
    }
    Ok(normalized)
}

fn grayscale_similarity_score_prepared(image_gray: &GrayImage, prepared_template: &[f32]) -> f32 {
    let mut normalized = image_gray
        .pixels()
        .map(|pixel| pixel.0[0] as f32 / 255.0)
        .collect::<Vec<_>>();
    let mean = normalized.iter().sum::<f32>() / normalized.len().max(1) as f32;
    for value in &mut normalized {
        *value -= mean;
    }
    let norm = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= 1e-6 {
        return 0.0;
    }

    normalized
        .iter()
        .zip(prepared_template)
        .map(|(value, template)| (*value / norm) * template)
        .sum::<f32>()
}

fn resize_to_target(image: &RgbaImage, target_width: u32, target_height: u32) -> RgbaImage {
    if image.width() == target_width && image.height() == target_height {
        return image.clone();
    }

    resize(
        image,
        target_width.max(1),
        target_height.max(1),
        FilterType::Triangle,
    )
}

fn draw_modeled_preview(
    image: &RgbaImage,
    anchors: &[ProbeBox],
    slot_scores: &[f32],
    present: bool,
) -> RgbaImage {
    let mut render = image.clone();
    for (anchor, score) in anchors.iter().zip(slot_scores) {
        let color = if *score >= SLOT_RAW_THRESHOLD {
            Rgba([72, 220, 96, 255])
        } else {
            Rgba([240, 70, 70, 255])
        };
        draw_hollow_rect_mut(
            &mut render,
            Rect::at(anchor.x as i32, anchor.y as i32).of_size(anchor.w, anchor.h),
            color,
        );
        let marker_width = anchor.w.clamp(4, 10);
        draw_filled_rect_mut(
            &mut render,
            Rect::at(anchor.x as i32, anchor.y as i32).of_size(marker_width, 3),
            color,
        );
    }

    if render.height() > 0 {
        let render_width = render.width();
        let render_height = render.height();
        let band_top = render.height().saturating_sub(3);
        let band_color = if present {
            Rgba([72, 220, 96, 255])
        } else {
            Rgba([240, 70, 70, 255])
        };
        draw_filled_rect_mut(
            &mut render,
            Rect::at(0, band_top as i32).of_size(render_width, render_height - band_top),
            band_color,
        );
    }

    render
}

fn count_true(values: &[bool]) -> usize {
    values.iter().filter(|value| **value).count()
}

fn format_slot_scores(scores: &[f32]) -> String {
    TAG_NAMES
        .iter()
        .zip(scores.iter().copied().chain(std::iter::repeat(0.0)))
        .take(EXPECTED_LABEL_COUNT)
        .map(|(tag, score)| format!("{tag}:{score:.2}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn rgb_channels(pixel: &Rgba<u8>) -> [f32; 3] {
    [pixel.0[0] as f32, pixel.0[1] as f32, pixel.0[2] as f32]
}

fn rgb_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
}

fn rgb_channel_spread(rgb: [f32; 3]) -> f32 {
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    max - min
}

fn color_distance(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    let dr = lhs[0] - rhs[0];
    let dg = lhs[1] - rhs[1];
    let db = lhs[2] - rhs[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn color_distances(values: &[[f32; 3]], target: [f32; 3]) -> Vec<f32> {
    values
        .iter()
        .map(|value| color_distance(*value, target))
        .collect()
}

fn mean_rgb(values: &[[f32; 3]]) -> Option<[f32; 3]> {
    if values.is_empty() {
        return None;
    }
    let mut sum = [0.0f32; 3];
    for value in values {
        sum[0] += value[0];
        sum[1] += value[1];
        sum[2] += value[2];
    }
    let count = values.len() as f32;
    Some([sum[0] / count, sum[1] / count, sum[2] / count])
}

fn mean_rgb_lower_luma_fraction(values: &[[f32; 3]], ratio: f32) -> Option<[f32; 3]> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| rgb_luma(*left).total_cmp(&rgb_luma(*right)));
    let take = ((sorted.len() as f32 * ratio.clamp(0.1, 1.0)).round() as usize)
        .max(1)
        .min(sorted.len());
    mean_rgb(&sorted[..take])
}

fn percentile_f32(values: &[f32], ratio: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index = ((sorted.len() - 1) as f32 * ratio.clamp(0.0, 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn median_u32(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn coefficient_of_variation(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if mean.abs() <= f32::EPSILON {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt() / mean.abs()
}

fn border_rgb_samples(image: &RgbaImage) -> Vec<[f32; 3]> {
    let edge_width = 2u32
        .min(image.width().saturating_sub(1))
        .min(image.height().saturating_sub(1));
    let mut samples = Vec::new();
    for (x, y, pixel) in image.enumerate_pixels() {
        if x <= edge_width
            || y <= edge_width
            || x >= image.width().saturating_sub(edge_width + 1)
            || y >= image.height().saturating_sub(edge_width + 1)
        {
            samples.push(rgb_channels(pixel));
        }
    }
    samples
}

fn mask_rgb_samples(image: &RgbaImage, mask: &GrayImage) -> Vec<[f32; 3]> {
    let mut samples = Vec::new();
    for (x, y, pixel) in image.enumerate_pixels() {
        if mask.get_pixel(x, y).0[0] > 0 {
            samples.push(rgb_channels(pixel));
        }
    }
    samples
}

fn preview_rgba_image(label: impl Into<String>, image: &RgbaImage, max_side: u32) -> DebugImage {
    let (target_width, target_height) =
        fit_preview_dimensions(image.width(), image.height(), max_side);
    let preview = if target_width == image.width() && target_height == image.height() {
        image.clone()
    } else {
        resize(image, target_width, target_height, FilterType::Nearest)
    };

    DebugImage::rgba(
        label,
        preview.width(),
        preview.height(),
        crate::tracking::debug::DebugImageKind::Snapshot,
        preview.into_raw(),
    )
}

fn fit_preview_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if width == 0 || height == 0 || max_side == 0 {
        return (1, 1);
    }
    if width.max(height) <= max_side {
        return (width, height);
    }

    let scale = max_side as f32 / width.max(height) as f32;
    (
        (width as f32 * scale).round().max(1.0) as u32,
        (height as f32 * scale).round().max(1.0) as u32,
    )
}

#[cfg(test)]
fn asset_test_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("test")
}

#[cfg(test)]
fn asset_paths(prefix: &str) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(asset_test_root())
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(|entry| entry.ok()))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".png"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[cfg(test)]
fn load_test_image(path: &Path) -> RgbaImage {
    image::open(path)
        .unwrap_or_else(|error| panic!("failed to load test image {}: {error:#}", path.display()))
        .into_rgba8()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_image_build_succeeds_for_all_real_positive_assets() {
        let positive_paths = asset_paths("has_map_");
        assert!(
            !positive_paths.is_empty(),
            "expected positive assets/test samples"
        );

        for path in &positive_paths {
            let image = load_test_image(path);
            let build = build_minimap_presence_model_from_image(&image).unwrap_or_else(|error| {
                panic!(
                    "single-image build failed for {}: {error:#}",
                    path.display()
                )
            });
            assert!(
                build.sample.present,
                "single-image build sample should pass self-check for {}",
                path.display()
            );
        }
    }

    #[test]
    fn model_round_trip_covers_all_real_assets_test_samples() {
        let positive_paths = asset_paths("has_map_");
        let negative_paths = asset_paths("no_map_");
        assert!(
            !positive_paths.is_empty(),
            "expected positive assets/test samples"
        );
        assert!(
            !negative_paths.is_empty(),
            "expected negative assets/test samples"
        );

        let positive_images = positive_paths
            .iter()
            .map(|path| load_test_image(path))
            .collect::<Vec<_>>();
        let build =
            build_minimap_presence_model_from_images(&positive_images).expect("build model");
        assert!(build.sample.present, "seed capture should pass self-check");

        let detector = PreparedMinimapPresenceDetector::from_model(
            build.model.clone(),
            DEFAULT_MEAN_RAW_THRESHOLD,
        )
        .expect("prepared detector");

        for path in &positive_paths {
            let image = load_test_image(path);
            let sample = detector.sample_image(&image).unwrap_or_else(|error| {
                panic!("positive sample {} failed: {error:#}", path.display())
            });
            assert!(
                sample.present,
                "expected positive sample {} to be present, got mean {:.3}, min {:.3}",
                path.display(),
                sample.mean_raw_score,
                sample.min_raw_score
            );
        }

        for path in &negative_paths {
            let image = load_test_image(path);
            let sample = detector.sample_image(&image).unwrap_or_else(|error| {
                panic!("negative sample {} failed: {error:#}", path.display())
            });
            assert!(
                !sample.present,
                "expected negative sample {} to be missing, got mean {:.3}, min {:.3}",
                path.display(),
                sample.mean_raw_score,
                sample.min_raw_score
            );
        }
    }
}
