use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
#[cfg(feature = "ai-burn")]
use burn::tensor::{
    Tensor, TensorData,
    backend::Backend,
    module::{conv2d, interpolate, max_pool2d},
    ops::{ConvOptions, InterpolateMode, InterpolateOptions},
};
use image::{
    GrayImage, Luma,
    imageops::{FilterType, blur, resize},
};
use imageproc::{
    contrast::otsu_level,
    distance_transform::Norm,
    morphology::{close, open},
    region_labelling::{Connectivity, connected_components},
};

#[cfg(test)]
use crate::config::load_existing_config;
use crate::{
    config::{AiDevicePreference, CaptureRegion},
    resources::WorkspaceSnapshot,
    tracking::{
        capture::{CaptureSource, DesktopCapture},
        debug::{DebugField, DebugImage},
        vision::{gray_image_as_unit_vec, preview_image, preview_mask_image},
    },
};
#[cfg(test)]
use image::imageops::crop_imm;

#[cfg(feature = "ai-burn")]
use crate::tracking::burn_support::{
    BurnDeviceConfig, BurnDeviceSelection, burn_device_label, select_burn_device,
};
#[cfg(test)]
use directories::ProjectDirs;

const TEMPLATE_FILE_NAME: &str = "minimap-presence-probe.png";
const MAX_TEMPLATE_SIDE: u32 = 640;
const SIGNATURE_WIDTH: u32 = 256;
const SIGNATURE_HEIGHT: u32 = 32;
const MIN_REQUIRED_LABEL_COMPONENTS: usize = 4;
const MIN_COMPONENT_AREA_RATIO: f32 = 0.0025;
const MAX_COMPONENT_AREA_RATIO: f32 = 0.24;
const MIN_COMPONENT_WIDTH_RATIO: f32 = 0.018;
const MAX_COMPONENT_WIDTH_RATIO: f32 = 0.26;
const MIN_COMPONENT_HEIGHT_RATIO: f32 = 0.18;
const MAX_COMPONENT_HEIGHT_RATIO: f32 = 0.92;
const MIN_COMPONENT_FILL_RATIO: f32 = 0.10;
const MAX_SELECTED_COMPONENTS: usize = 8;
const MIN_SUPPORT_COVERAGE: f32 = 0.08;
pub struct MinimapPresenceDetector {
    capture: DesktopCapture,
    template_preview: GrayImage,
    template_signature: ProbeSignature,
    template_units: Vec<f32>,
    weight_units: Vec<f32>,
    mask_weight_sum: f32,
    template_mean: f32,
    template_norm: f32,
    threshold: f32,
    backend: ProbeBackend,
}

#[derive(Debug, Clone)]
pub struct MinimapPresenceSample {
    pub present: bool,
    pub score: f32,
    pub structure_score: f32,
    pub layout_score: f32,
    pub label_count_score: f32,
    pub ink_label_score: f32,
    pub geometry_score: f32,
    pub threshold: f32,
    pub current_preview: GrayImage,
    pub current_support_mask: GrayImage,
    pub current_ink_mask: GrayImage,
    pub mask_coverage: f32,
    pub label_count: usize,
    pub inked_label_count: usize,
    pub expected_label_count: usize,
}

enum ProbeBackend {
    Cpu,
    #[cfg(feature = "ai-burn")]
    Burn(Box<dyn ProbeTensorBackend>),
}

#[cfg(feature = "ai-burn")]
trait ProbeTensorBackend: Send + Sync {
    fn signature(&self, image: &GrayImage) -> Result<ProbeSignature>;
    fn score(&self, current_units: &[f32]) -> Result<f32>;
    fn label(&self) -> String;
}

#[cfg(feature = "ai-burn")]
struct BurnProbeBackend<B: Backend>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    device: B::Device,
    device_label: String,
    box3_kernel: Tensor<B, 4>,
    box7_kernel: Tensor<B, 4>,
    centered_template: Tensor<B, 1>,
    mask: Tensor<B, 1>,
    mask_weight_sum: f32,
    template_norm: f32,
}

#[derive(Debug, Clone, Copy)]
struct ProbeDeviceConfig {
    device: AiDevicePreference,
    device_index: usize,
}

#[cfg(feature = "ai-burn")]
impl BurnDeviceConfig for ProbeDeviceConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[derive(Debug, Clone)]
struct ProbeSignature {
    support_mask: GrayImage,
    ink_mask: GrayImage,
    feature_units: Vec<f32>,
    weight_units: Vec<f32>,
    bright_profile: Vec<f32>,
    ink_profile: Vec<f32>,
    label_count: usize,
    inked_label_count: usize,
    geometry_score: f32,
    support_coverage: f32,
}

#[derive(Debug, Clone)]
struct ProbeScoreBreakdown {
    structure_score: f32,
    layout_score: f32,
    label_count_score: f32,
    ink_label_score: f32,
    geometry_score: f32,
    final_score: f32,
}

#[derive(Debug, Clone)]
struct ProbeComponent {
    area: u32,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl ProbeComponent {
    fn width(&self) -> u32 {
        self.right.saturating_sub(self.left) + 1
    }

    fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top) + 1
    }

    fn center_x(&self) -> f32 {
        (self.left + self.right) as f32 * 0.5
    }

    fn center_y(&self) -> f32 {
        (self.top + self.bottom) as f32 * 0.5
    }

    fn fill_ratio(&self) -> f32 {
        self.area as f32 / (self.width().max(1) * self.height().max(1)) as f32
    }
}

impl MinimapPresenceDetector {
    pub fn new(workspace: &WorkspaceSnapshot) -> Result<Option<Self>> {
        let probe = &workspace.config.minimap_presence_probe;
        if !probe.enabled {
            return Ok(None);
        }

        let region = probe.capture_region().ok_or_else(|| {
            anyhow::anyhow!(
                "minimap_presence_probe 已启用，但 top/left/width/height 还没有完整配置"
            )
        })?;

        let capture = DesktopCapture::from_absolute_region(&region)?;
        let template_path = minimap_presence_probe_template_path(&workspace.project_root);
        if !template_path.is_file() {
            bail!(
                "小地图存在探针已启用，但模板文件不存在：{}。请重新使用“F1-P 标签带取区”完成一次取区。",
                template_path.display()
            );
        }

        let template_preview = image::open(&template_path)
            .with_context(|| {
                format!(
                    "failed to load minimap presence probe template {}",
                    template_path.display()
                )
            })?
            .into_luma8();
        let template_preview = normalize_probe_capture(&template_preview);
        let template_signature = build_probe_signature(&template_preview);
        validate_probe_signature(&template_signature)?;

        let template_units = template_signature.feature_units.clone();
        let weight_units = template_signature.weight_units.clone();
        let mask_weight_sum = weight_units.iter().sum::<f32>().max(f32::EPSILON);
        let template_mean = weighted_mean(&template_units, &weight_units, mask_weight_sum);
        let template_norm = centered_weighted_norm(&template_units, &weight_units, template_mean);
        let backend = build_backend(
            probe.device,
            probe.device_index,
            &template_preview,
            &template_units,
            &weight_units,
            mask_weight_sum,
            template_mean,
            template_norm,
        )?;

        Ok(Some(Self {
            capture,
            template_preview,
            template_signature,
            template_units,
            weight_units,
            mask_weight_sum,
            template_mean,
            template_norm,
            threshold: probe.match_threshold,
            backend,
        }))
    }

    pub fn sample(&self) -> Result<MinimapPresenceSample> {
        let current_preview = normalize_probe_capture(&self.capture.capture_gray()?);
        let current_preview = resize_to_match(&current_preview, &self.template_preview);
        let current_signature = self.sample_signature(&current_preview)?;
        let structure_score = self.score_current(&current_signature.feature_units)?;
        let scores = build_probe_score_breakdown(
            &self.template_signature,
            &current_signature,
            structure_score,
        );

        Ok(MinimapPresenceSample {
            present: scores.final_score >= self.threshold,
            score: scores.final_score,
            structure_score: scores.structure_score,
            layout_score: scores.layout_score,
            label_count_score: scores.label_count_score,
            ink_label_score: scores.ink_label_score,
            geometry_score: scores.geometry_score,
            threshold: self.threshold,
            current_preview,
            current_support_mask: current_signature.support_mask,
            current_ink_mask: current_signature.ink_mask,
            mask_coverage: self.template_signature.support_coverage,
            label_count: current_signature.label_count,
            inked_label_count: current_signature.inked_label_count,
            expected_label_count: self.template_signature.label_count,
        })
    }

    #[must_use]
    pub fn debug_images(&self, sample: &MinimapPresenceSample) -> Vec<DebugImage> {
        vec![
            preview_image("Probe Live", &sample.current_preview, &[], 196),
            preview_image("Probe Template", &self.template_preview, &[], 196),
            preview_mask_image("Probe Live Labels", &sample.current_support_mask, 196),
            preview_mask_image(
                "Probe Template Labels",
                &self.template_signature.support_mask,
                196,
            ),
            preview_mask_image("Probe Live Ink", &sample.current_ink_mask, 196),
            preview_mask_image("Probe Template Ink", &self.template_signature.ink_mask, 196),
        ]
    }

    #[must_use]
    pub fn debug_fields(&self, sample: &MinimapPresenceSample) -> Vec<DebugField> {
        vec![
            DebugField::new(
                "探针状态",
                if sample.present { "Present" } else { "Missing" },
            ),
            DebugField::new("探针设备", self.backend_label()),
            DebugField::new("探针得分", format!("{:.3}", sample.score)),
            DebugField::new("结构得分", format!("{:.3}", sample.structure_score)),
            DebugField::new("布局得分", format!("{:.3}", sample.layout_score)),
            DebugField::new(
                "标签数量",
                format!("{}/{}", sample.label_count, sample.expected_label_count),
            ),
            DebugField::new(
                "黑字标签",
                format!(
                    "{}/{}",
                    sample.inked_label_count, sample.expected_label_count
                ),
            ),
            DebugField::new("黑字得分", format!("{:.3}", sample.ink_label_score)),
            DebugField::new("几何得分", format!("{:.3}", sample.geometry_score)),
            DebugField::new("探针阈值", format!("{:.3}", sample.threshold)),
            DebugField::new("探针遮罩", format!("{:.0}%", sample.mask_coverage * 100.0)),
        ]
    }

    fn score_current(&self, current_units: &[f32]) -> Result<f32> {
        match &self.backend {
            ProbeBackend::Cpu => Ok(weighted_pearson_correlation_units(
                current_units,
                &self.template_units,
                &self.weight_units,
                self.mask_weight_sum,
                self.template_mean,
                self.template_norm,
            )),
            #[cfg(feature = "ai-burn")]
            ProbeBackend::Burn(backend) => backend.score(current_units),
        }
    }

    fn sample_signature(&self, image: &GrayImage) -> Result<ProbeSignature> {
        match &self.backend {
            ProbeBackend::Cpu => Ok(build_probe_signature(image)),
            #[cfg(feature = "ai-burn")]
            ProbeBackend::Burn(backend) => backend.signature(image),
        }
    }

    fn backend_label(&self) -> String {
        match &self.backend {
            ProbeBackend::Cpu => "CPU".to_owned(),
            #[cfg(feature = "ai-burn")]
            ProbeBackend::Burn(backend) => backend.label(),
        }
    }
}

pub fn calibrate_minimap_presence_probe(
    project_root: &Path,
    region: &CaptureRegion,
) -> Result<PathBuf> {
    if region.width < 12 || region.height < 12 {
        bail!("小地图存在探针区域过小，至少需要 12x12");
    }

    let capture = DesktopCapture::from_absolute_region(region)?;
    let preview = normalize_probe_capture(&capture.capture_gray()?);
    let signature = build_probe_signature(&preview);
    validate_probe_signature(&signature).with_context(
        || "当前取区没有提取到稳定的 F1-P 标签结构，请只框住标签带本身，不要包含上方图标",
    )?;

    let path = minimap_presence_probe_template_path(project_root);
    save_gray_image(&path, &preview)?;
    Ok(path)
}

#[must_use]
pub fn minimap_presence_probe_template_path(project_root: &Path) -> PathBuf {
    project_root
        .join("cache")
        .join("tracking")
        .join(TEMPLATE_FILE_NAME)
}

fn build_probe_signature(image: &GrayImage) -> ProbeSignature {
    let bright_mask = detect_bright_label_mask(image);
    let components = select_label_components(&bright_mask);
    let support_mask = build_support_mask(image.width(), image.height(), &components);
    let ink_mask = build_ink_mask(image, &components, &support_mask);
    let (aligned_support_mask, aligned_ink_mask) = align_signature_masks(&support_mask, &ink_mask);
    let bright_profile = smoothed_column_profile(&aligned_support_mask);
    let ink_profile = smoothed_column_profile(&aligned_ink_mask);
    let feature_units = build_feature_units(&aligned_support_mask, &aligned_ink_mask);
    let weight_units = build_weight_units(&aligned_support_mask);
    let inked_label_count = count_inked_labels(&ink_mask, &components);
    let geometry_score = component_geometry_quality(&components);
    let support_coverage = mask_coverage(&support_mask);

    ProbeSignature {
        support_mask,
        ink_mask,
        feature_units,
        weight_units,
        bright_profile,
        ink_profile,
        label_count: components.len(),
        inked_label_count,
        geometry_score,
        support_coverage,
    }
}

fn align_signature_masks(support_mask: &GrayImage, ink_mask: &GrayImage) -> (GrayImage, GrayImage) {
    let Some((left, top, right, bottom)) = mask_bounds(support_mask) else {
        let empty = GrayImage::from_pixel(SIGNATURE_WIDTH, SIGNATURE_HEIGHT, Luma([0]));
        return (empty.clone(), empty);
    };
    let left = left.saturating_sub(2);
    let top = top.saturating_sub(1);
    let right = (right + 2).min(support_mask.width().saturating_sub(1));
    let bottom = (bottom + 1).min(support_mask.height().saturating_sub(1));
    let width = right.saturating_sub(left) + 1;
    let height = bottom.saturating_sub(top) + 1;

    let support_crop = image::imageops::crop_imm(support_mask, left, top, width, height).to_image();
    let ink_crop = image::imageops::crop_imm(ink_mask, left, top, width, height).to_image();
    let aligned_support = resize(
        &support_crop,
        SIGNATURE_WIDTH,
        SIGNATURE_HEIGHT,
        FilterType::Triangle,
    );
    let aligned_ink = resize(
        &ink_crop,
        SIGNATURE_WIDTH,
        SIGNATURE_HEIGHT,
        FilterType::Triangle,
    );
    (aligned_support, aligned_ink)
}

fn mask_bounds(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let mut left = mask.width();
    let mut top = mask.height();
    let mut right = 0u32;
    let mut bottom = 0u32;
    let mut found = false;

    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] == 0 {
            continue;
        }
        found = true;
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }

    found.then_some((left, top, right, bottom))
}

fn validate_probe_signature(signature: &ProbeSignature) -> Result<()> {
    if signature.label_count < MIN_REQUIRED_LABEL_COMPONENTS {
        bail!(
            "标签组件不足：检测到 {} 个，需要至少 {} 个",
            signature.label_count,
            MIN_REQUIRED_LABEL_COMPONENTS
        );
    }
    if signature.support_coverage < MIN_SUPPORT_COVERAGE {
        bail!(
            "标签覆盖率不足：{:.0}%，请缩小到只包含标签带本身",
            signature.support_coverage * 100.0
        );
    }
    Ok(())
}

fn build_probe_score_breakdown(
    template: &ProbeSignature,
    current: &ProbeSignature,
    structure_score: f32,
) -> ProbeScoreBreakdown {
    let bright_layout = profile_similarity(&template.bright_profile, &current.bright_profile);
    let ink_layout = profile_similarity(&template.ink_profile, &current.ink_profile);
    let layout_score = (bright_layout * 0.72 + ink_layout * 0.28).clamp(0.0, 1.0);
    let label_count_score = count_similarity(template.label_count, current.label_count);
    let ink_label_score = count_similarity(template.inked_label_count, current.inked_label_count);
    let geometry_score =
        (current.geometry_score / template.geometry_score.max(0.35)).clamp(0.0, 1.0);
    let raw_score = (structure_score * 0.45
        + layout_score * 0.15
        + label_count_score * 0.10
        + ink_label_score * 0.15
        + geometry_score * 0.15)
        .clamp(0.0, 1.0);
    let stability_gate =
        harmonic_mean(&[label_count_score, ink_label_score, geometry_score]).clamp(0.0, 1.0);
    let final_score = (raw_score * (0.20 + 0.80 * stability_gate)).clamp(0.0, 1.0);

    ProbeScoreBreakdown {
        structure_score,
        layout_score,
        label_count_score,
        ink_label_score,
        geometry_score,
        final_score,
    }
}

fn detect_bright_label_mask(image: &GrayImage) -> GrayImage {
    let background = blur(image, 3.2);
    let local_bright = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let value = image.get_pixel(x, y).0[0];
        let baseline = background.get_pixel(x, y).0[0];
        Luma([value.saturating_sub(baseline)])
    });

    let mut residual_values = local_bright
        .pixels()
        .map(|pixel| pixel.0[0])
        .collect::<Vec<_>>();
    residual_values.sort_unstable();
    let mut raw_values = image.pixels().map(|pixel| pixel.0[0]).collect::<Vec<_>>();
    raw_values.sort_unstable();

    let residual_cutoff = percentile(&residual_values, 0.84).max(18);
    let raw_floor = percentile(&raw_values, 0.56).max(otsu_level(image));

    let mask = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let value = image.get_pixel(x, y).0[0];
        let residual = local_bright.get_pixel(x, y).0[0];
        Luma([if residual >= residual_cutoff && value >= raw_floor {
            255
        } else {
            0
        }])
    });

    open(&close(&mask, Norm::LInf, 1), Norm::LInf, 1)
}

fn select_label_components(mask: &GrayImage) -> Vec<ProbeComponent> {
    let mut components =
        merge_adjacent_components(select_label_components_from_connected_regions(mask));
    if !components.is_empty() {
        components = refine_label_components(mask, &components);
    }
    if components.len() < MIN_REQUIRED_LABEL_COMPONENTS {
        components = select_label_components_from_profile(mask);
    }
    if components.is_empty() {
        return components;
    }

    let image_area = (mask.width().max(1) * mask.height().max(1)) as f32;
    let min_area = image_area * MIN_COMPONENT_AREA_RATIO;
    let max_area = image_area * MAX_COMPONENT_AREA_RATIO;
    let min_width = mask.width().max(1) as f32 * MIN_COMPONENT_WIDTH_RATIO;
    let max_width = mask.width().max(1) as f32 * MAX_COMPONENT_WIDTH_RATIO;
    let min_height = mask.height().max(1) as f32 * MIN_COMPONENT_HEIGHT_RATIO;
    let max_height = mask.height().max(1) as f32 * MAX_COMPONENT_HEIGHT_RATIO;

    let mut components = components
        .into_iter()
        .filter(|component| {
            let width = component.width() as f32;
            let height = component.height() as f32;
            let area = component.area as f32;
            area >= min_area
                && area <= max_area
                && width >= min_width
                && width <= max_width
                && height >= min_height
                && height <= max_height
                && component.fill_ratio() >= MIN_COMPONENT_FILL_RATIO
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        return components;
    }

    let row_center = weighted_component_center_y(&components);
    components.retain(|component| {
        (component.center_y() - row_center).abs() <= mask.height().max(1) as f32 * 0.22
    });
    if components.len() > MAX_SELECTED_COMPONENTS {
        components.sort_by(|left, right| right.area.cmp(&left.area));
        components.truncate(MAX_SELECTED_COMPONENTS);
    }
    components.sort_by(|left, right| left.center_x().total_cmp(&right.center_x()));
    components
}

fn select_label_components_from_connected_regions(mask: &GrayImage) -> Vec<ProbeComponent> {
    let connected_mask = close(mask, Norm::LInf, 2);
    let labels = connected_components(&connected_mask, Connectivity::Eight, Luma([0]));
    let mut components = HashMap::<u32, ProbeComponent>::new();

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
            .or_insert(ProbeComponent {
                area: 1,
                left: x,
                top: y,
                right: x,
                bottom: y,
            });
    }

    let mut components = components.into_values().collect::<Vec<_>>();
    components.sort_by(|left, right| left.center_x().total_cmp(&right.center_x()));
    components
}

fn refine_label_components(mask: &GrayImage, components: &[ProbeComponent]) -> Vec<ProbeComponent> {
    if components.is_empty() {
        return Vec::new();
    }

    let profile = smoothed_column_profile(mask);
    let row_top = median_u32(components.iter().map(|component| component.top));
    let row_bottom = median_u32(components.iter().map(|component| component.bottom));
    let boundary_cutoff = (profile.iter().copied().fold(0.0f32, f32::max) * 0.10).clamp(0.03, 0.08);
    let centers = components
        .iter()
        .map(|component| component.center_x().round() as u32)
        .collect::<Vec<_>>();

    components
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let left_limit = if index > 0 {
                (centers[index - 1] + centers[index]) / 2
            } else {
                0
            };
            let right_limit = if index + 1 < centers.len() {
                (centers[index] + centers[index + 1]) / 2
            } else {
                mask.width().saturating_sub(1)
            };
            let center = centers[index].clamp(left_limit, right_limit);
            let left = expand_component_left(&profile, left_limit, center, boundary_cutoff);
            let right = expand_component_right(&profile, center, right_limit, boundary_cutoff);
            let top = row_top.min(mask.height().saturating_sub(1));
            let bottom = row_bottom.min(mask.height().saturating_sub(1)).max(top);
            let area = count_active_pixels(mask, left, top, right, bottom);

            ProbeComponent {
                area: area.max(component.area),
                left,
                top,
                right,
                bottom,
            }
        })
        .collect()
}

fn merge_adjacent_components(mut components: Vec<ProbeComponent>) -> Vec<ProbeComponent> {
    if components.len() < 2 {
        return components;
    }

    components.sort_by(|left, right| left.left.cmp(&right.left));
    let mut merged = Vec::with_capacity(components.len());
    let mut current = components[0].clone();

    for next in components.into_iter().skip(1) {
        if next.left <= current.right.saturating_add(3) {
            current.area += next.area;
            current.left = current.left.min(next.left);
            current.top = current.top.min(next.top);
            current.right = current.right.max(next.right);
            current.bottom = current.bottom.max(next.bottom);
            continue;
        }

        merged.push(current);
        current = next;
    }

    merged.push(current);
    merged
}

fn select_label_components_from_profile(mask: &GrayImage) -> Vec<ProbeComponent> {
    let profile = smoothed_column_profile(mask);
    if profile.is_empty() {
        return Vec::new();
    }

    let max_profile = profile.iter().copied().fold(0.0f32, f32::max);
    let column_cutoff = (max_profile * 0.26).clamp(0.08, 0.22);
    let min_run_width = (mask.width().max(1) as f32 * MIN_COMPONENT_WIDTH_RATIO)
        .round()
        .max(1.0) as u32;

    let mut components = Vec::new();
    let mut x = 0u32;
    while x < mask.width() {
        while x < mask.width() && profile[x as usize] < column_cutoff {
            x += 1;
        }
        if x >= mask.width() {
            break;
        }

        let start = x;
        while x < mask.width() && profile[x as usize] >= column_cutoff {
            x += 1;
        }
        let end = x.saturating_sub(1);
        if end.saturating_sub(start) + 1 < min_run_width {
            continue;
        }

        let mut top = mask.height();
        let mut bottom = 0u32;
        let mut area = 0u32;
        for scan_x in start..=end {
            for scan_y in 0..mask.height() {
                if mask.get_pixel(scan_x, scan_y).0[0] == 0 {
                    continue;
                }
                area += 1;
                top = top.min(scan_y);
                bottom = bottom.max(scan_y);
            }
        }
        if area == 0 || top >= mask.height() {
            continue;
        }

        components.push(ProbeComponent {
            area,
            left: start,
            top,
            right: end,
            bottom,
        });
    }

    components
}

fn expand_component_left(profile: &[f32], limit: u32, center: u32, cutoff: f32) -> u32 {
    let mut x = center.min((profile.len().saturating_sub(1)) as u32);
    let limit = limit.min(x);
    while x > limit && profile[x as usize] >= cutoff {
        x -= 1;
    }
    if profile[x as usize] < cutoff && x < center {
        x + 1
    } else {
        x
    }
}

fn expand_component_right(profile: &[f32], center: u32, limit: u32, cutoff: f32) -> u32 {
    let max_index = (profile.len().saturating_sub(1)) as u32;
    let mut x = center.min(max_index);
    let limit = limit.min(max_index).max(x);
    while x < limit && profile[x as usize] >= cutoff {
        x += 1;
    }
    if profile[x as usize] < cutoff && x > center {
        x.saturating_sub(1)
    } else {
        x
    }
}

fn count_active_pixels(mask: &GrayImage, left: u32, top: u32, right: u32, bottom: u32) -> u32 {
    let mut area = 0u32;
    for y in top..=bottom {
        for x in left..=right {
            if mask.get_pixel(x, y).0[0] > 0 {
                area += 1;
            }
        }
    }
    area
}

fn median_u32(values: impl Iterator<Item = u32>) -> u32 {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn weighted_component_center_y(components: &[ProbeComponent]) -> f32 {
    let weight_sum = components
        .iter()
        .map(|component| component.area as f32)
        .sum::<f32>();
    if weight_sum <= f32::EPSILON {
        return 0.0;
    }

    components
        .iter()
        .map(|component| component.center_y() * component.area as f32)
        .sum::<f32>()
        / weight_sum
}

fn build_support_mask(width: u32, height: u32, components: &[ProbeComponent]) -> GrayImage {
    let mut mask = GrayImage::from_pixel(width, height, Luma([0]));
    for component in components {
        let pad_x = 1;
        let pad_y = 1;
        let left = component.left.saturating_sub(pad_x);
        let top = component.top.saturating_sub(pad_y);
        let right = (component.right + pad_x).min(width.saturating_sub(1));
        let bottom = (component.bottom + pad_y).min(height.saturating_sub(1));
        fill_rect(&mut mask, left, top, right, bottom, 255);
    }
    mask
}

fn build_ink_mask(
    image: &GrayImage,
    components: &[ProbeComponent],
    support_mask: &GrayImage,
) -> GrayImage {
    let mut mask = GrayImage::from_pixel(image.width(), image.height(), Luma([0]));
    for component in components {
        let width = component.width();
        let height = component.height();
        let inset_x = ((width as f32 * 0.12).round() as u32).clamp(1, 4);
        let inset_y = ((height as f32 * 0.18).round() as u32).clamp(1, 4);
        let left = (component.left + inset_x).min(component.right);
        let top = (component.top + inset_y).min(component.bottom);
        let right = component.right.saturating_sub(inset_x).max(left);
        let bottom = component.bottom.saturating_sub(inset_y).max(top);
        let bright_mean = mean_in_rect(
            image,
            component.left,
            component.top,
            component.right,
            component.bottom,
        );
        let cutoff = (bright_mean.round() as i32 - 88).clamp(24, 160) as u8;

        for y in top..=bottom {
            for x in left..=right {
                if support_mask.get_pixel(x, y).0[0] == 0 {
                    continue;
                }
                if image.get_pixel(x, y).0[0] <= cutoff {
                    mask.put_pixel(x, y, Luma([255]));
                }
            }
        }
    }
    mask
}

fn count_inked_labels(ink_mask: &GrayImage, components: &[ProbeComponent]) -> usize {
    components
        .iter()
        .filter(|component| {
            let ink_pixels = count_active_pixels(
                ink_mask,
                component.left,
                component.top,
                component.right,
                component.bottom,
            );
            let bbox_area = component.width().max(1) * component.height().max(1);
            ink_pixels >= 6 && ink_pixels as f32 / bbox_area as f32 >= 0.008
        })
        .count()
}

fn component_geometry_quality(components: &[ProbeComponent]) -> f32 {
    if components.len() < 2 {
        return 0.0;
    }

    let widths = components
        .iter()
        .map(|component| component.width() as f32)
        .collect::<Vec<_>>();
    let gaps = components
        .windows(2)
        .map(|pair| pair[1].left.saturating_sub(pair[0].right) as f32)
        .collect::<Vec<_>>();
    let width_score = consistency_score(&widths, 0.45);
    let gap_score = consistency_score(&gaps, 0.60);
    (width_score * 0.45 + gap_score * 0.55).clamp(0.0, 1.0)
}

fn consistency_score(values: &[f32], tolerance: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return 1.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if mean <= f32::EPSILON {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    let coefficient = variance.sqrt() / mean;
    (1.0 - coefficient / tolerance.max(f32::EPSILON)).clamp(0.0, 1.0)
}

fn harmonic_mean(values: &[f32]) -> f32 {
    if values.is_empty() || values.iter().any(|value| *value <= f32::EPSILON) {
        return 0.0;
    }

    values.len() as f32 / values.iter().map(|value| 1.0 / value).sum::<f32>()
}

fn mean_in_rect(image: &GrayImage, left: u32, top: u32, right: u32, bottom: u32) -> f32 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for y in top..=bottom {
        for x in left..=right {
            sum += u64::from(image.get_pixel(x, y).0[0]);
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

fn build_feature_units(support_mask: &GrayImage, ink_mask: &GrayImage) -> Vec<f32> {
    let mut units = gray_image_as_unit_vec(support_mask);
    units.extend(gray_image_as_unit_vec(ink_mask));
    units
}

fn build_weight_units(support_mask: &GrayImage) -> Vec<f32> {
    let weights = gray_image_as_unit_vec(support_mask);
    let mut repeated = Vec::with_capacity(weights.len() * 2);
    repeated.extend(weights.iter().copied());
    repeated.extend(weights);
    repeated
}

fn smoothed_column_profile(mask: &GrayImage) -> Vec<f32> {
    let profile = (0..mask.width())
        .map(|x| {
            let active = (0..mask.height())
                .filter(|&y| mask.get_pixel(x, y).0[0] > 0)
                .count() as f32;
            active / mask.height().max(1) as f32
        })
        .collect::<Vec<_>>();
    smooth_profile(&profile, 2)
}

fn profile_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let mut rhs_norm = 0.0f32;
    for (lhs_value, rhs_value) in lhs.iter().zip(rhs).take(len) {
        dot += lhs_value * rhs_value;
        lhs_norm += lhs_value * lhs_value;
        rhs_norm += rhs_value * rhs_value;
    }

    if lhs_norm <= f32::EPSILON || rhs_norm <= f32::EPSILON {
        return 0.0;
    }

    (dot / (lhs_norm.sqrt() * rhs_norm.sqrt())).clamp(0.0, 1.0)
}

fn count_similarity(expected: usize, actual: usize) -> f32 {
    if expected == 0 {
        return if actual == 0 { 1.0 } else { 0.0 };
    }

    let difference = expected.abs_diff(actual) as f32;
    (1.0 - difference / expected.max(1) as f32).clamp(0.0, 1.0)
}

fn fill_rect(image: &mut GrayImage, left: u32, top: u32, right: u32, bottom: u32, value: u8) {
    for y in top..=bottom {
        for x in left..=right {
            image.put_pixel(x, y, Luma([value]));
        }
    }
}

fn build_backend(
    device: AiDevicePreference,
    device_index: usize,
    template_preview: &GrayImage,
    _template_units: &[f32],
    _weight_units: &[f32],
    _mask_weight_sum: f32,
    _template_mean: f32,
    _template_norm: f32,
) -> Result<ProbeBackend> {
    if device == AiDevicePreference::Cpu {
        return Ok(ProbeBackend::Cpu);
    }

    #[cfg(feature = "ai-burn")]
    {
        let device = select_burn_device(&ProbeDeviceConfig {
            device,
            device_index,
        })?;
        return Ok(match device {
            BurnDeviceSelection::Cpu => ProbeBackend::Cpu,
            #[cfg(burn_cuda_backend)]
            BurnDeviceSelection::Cuda(device) => {
                ProbeBackend::Burn(Box::new(BurnProbeBackend::<burn::backend::Cuda>::new(
                    device.clone(),
                    burn_device_label(&BurnDeviceSelection::Cuda(device)),
                    template_preview,
                )?))
            }
            #[cfg(burn_vulkan_backend)]
            BurnDeviceSelection::Vulkan(device) => {
                ProbeBackend::Burn(Box::new(BurnProbeBackend::<burn::backend::Vulkan>::new(
                    device.clone(),
                    burn_device_label(&BurnDeviceSelection::Vulkan(device)),
                    template_preview,
                )?))
            }
            #[cfg(burn_metal_backend)]
            BurnDeviceSelection::Metal(device) => {
                ProbeBackend::Burn(Box::new(BurnProbeBackend::<burn::backend::Metal>::new(
                    device.clone(),
                    burn_device_label(&BurnDeviceSelection::Metal(device)),
                    template_preview,
                )?))
            }
        });
    }

    #[cfg(not(feature = "ai-burn"))]
    {
        let _ = template_preview;
        let _ = device_index;
        bail!(
            "小地图存在探针配置选择了 {} 设备，但当前二进制未启用 `ai-burn` 特性；请改回 cpu 或重新构建带 Burn 后端的版本",
            device
        );
    }
}

#[cfg(feature = "ai-burn")]
impl<B> BurnProbeBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn new(device: B::Device, device_label: String, template_preview: &GrayImage) -> Result<Self> {
        let box3_kernel = box_kernel_tensor::<B>(&device, 3);
        let box7_kernel = box_kernel_tensor::<B>(&device, 7);
        let template_signature = extract_burn_probe_signature::<B>(
            &device,
            &box3_kernel,
            &box7_kernel,
            template_preview,
        )?;
        let template_units = template_signature.feature_units;
        let weight_units = template_signature.weight_units;
        let len = template_units.len();
        let mask_weight_sum = weight_units.iter().sum::<f32>().max(f32::EPSILON);
        let template_mean = weighted_mean(&template_units, &weight_units, mask_weight_sum);
        let template_norm = centered_weighted_norm(&template_units, &weight_units, template_mean);
        let template = Tensor::<B, 1>::from_data(TensorData::new(template_units, [len]), &device);
        let mask = Tensor::<B, 1>::from_data(TensorData::new(weight_units, [len]), &device);
        let centered_template = (template - template_mean) * mask.clone();

        Ok(Self {
            device,
            device_label,
            box3_kernel,
            box7_kernel,
            centered_template,
            mask,
            mask_weight_sum,
            template_norm,
        })
    }
}

#[cfg(feature = "ai-burn")]
impl<B> ProbeTensorBackend for BurnProbeBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn signature(&self, image: &GrayImage) -> Result<ProbeSignature> {
        extract_burn_probe_signature::<B>(&self.device, &self.box3_kernel, &self.box7_kernel, image)
    }

    fn score(&self, current_units: &[f32]) -> Result<f32> {
        let current = Tensor::<B, 1>::from_data(
            TensorData::new(current_units.to_vec(), [current_units.len()]),
            &self.device,
        );
        let masked_sum = (current.clone() * self.mask.clone()).sum().into_scalar();
        let current_mean = masked_sum / self.mask_weight_sum.max(f32::EPSILON);
        let current_centered = (current - current_mean) * self.mask.clone();
        let dot = (current_centered.clone() * self.centered_template.clone())
            .sum()
            .into_scalar();
        let current_norm = current_centered.powi_scalar(2).sum().into_scalar().sqrt();
        if current_norm <= f32::EPSILON || self.template_norm <= f32::EPSILON {
            return Ok(0.0);
        }

        Ok((dot / (current_norm * self.template_norm)).clamp(0.0, 1.0))
    }

    fn label(&self) -> String {
        self.device_label.clone()
    }
}

#[cfg(feature = "ai-burn")]
fn extract_burn_probe_signature<B>(
    device: &B::Device,
    box3_kernel: &Tensor<B, 4>,
    box7_kernel: &Tensor<B, 4>,
    image: &GrayImage,
) -> Result<ProbeSignature>
where
    B: Backend<FloatElem = f32>,
{
    let image = gray_image_tensor::<B>(image, device);
    let background = conv2d(
        image.clone(),
        box7_kernel.clone(),
        None,
        ConvOptions::new([1, 1], [3, 3], [1, 1], 1),
    );
    let bright_local = ((image.clone() - background.clone()) * 5.2).clamp(0.0, 1.0);
    let bright_raw = (image.clone() * 3.0 - 1.50).clamp(0.0, 1.0);
    let support_seed = bright_local * (bright_raw * 0.55 + 0.45);
    let support_dilated = max_pool2d(support_seed, [3, 3], [1, 1], [1, 1], [1, 1], false);
    let support_blurred = conv2d(
        support_dilated,
        box3_kernel.clone(),
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    );
    let row_gate = (support_blurred.clone().mean_dim(3) * 3.6 - 0.10)
        .clamp(0.0, 1.0)
        .unsqueeze_dim(3);
    let support_soft = ((support_blurred * row_gate) * 1.8 - 0.08).clamp(0.0, 1.0);

    let local_dark = ((background - image.clone()) * 5.0).clamp(0.0, 1.0);
    let dark_raw = (image * -3.2 + 1.70).clamp(0.0, 1.0);
    let ink_support = (conv2d(
        support_soft.clone(),
        box3_kernel.clone(),
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    ) * 1.5
        - 0.05)
        .clamp(0.0, 1.0);
    let ink_soft = (local_dark * (dark_raw * 0.55 + 0.45) * ink_support).clamp(0.0, 1.0);

    let support_small = interpolate(
        support_soft,
        [SIGNATURE_HEIGHT as usize, SIGNATURE_WIDTH as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    )
    .clamp(0.0, 1.0);
    let ink_small = interpolate(
        ink_soft,
        [SIGNATURE_HEIGHT as usize, SIGNATURE_WIDTH as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    )
    .clamp(0.0, 1.0);
    let support_rows = tensor_to_rows::<B>(support_small)?;
    let ink_rows = tensor_to_rows::<B>(ink_small)?;
    Ok(build_probe_signature_from_soft_rows(
        &support_rows,
        &ink_rows,
    ))
}

#[cfg(feature = "ai-burn")]
fn gray_image_tensor<B>(image: &GrayImage, device: &B::Device) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    Tensor::<B, 4>::from_data(
        TensorData::new(
            gray_image_as_unit_vec(image),
            [1, 1, image.height() as usize, image.width() as usize],
        ),
        device,
    )
}

#[cfg(feature = "ai-burn")]
fn box_kernel_tensor<B>(device: &B::Device, size: usize) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    let weight = 1.0f32 / (size * size).max(1) as f32;
    Tensor::<B, 4>::from_data(
        TensorData::new(vec![weight; size * size], [1, 1, size, size]),
        device,
    )
}

#[cfg(feature = "ai-burn")]
fn tensor_to_rows<B>(tensor: Tensor<B, 4>) -> Result<Vec<Vec<f32>>>
where
    B: Backend<FloatElem = f32>,
{
    let flat = tensor
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(flat
        .chunks(SIGNATURE_WIDTH as usize)
        .map(|row| row.to_vec())
        .collect())
}

#[cfg(feature = "ai-burn")]
fn build_probe_signature_from_soft_rows(
    support_rows: &[Vec<f32>],
    ink_rows: &[Vec<f32>],
) -> ProbeSignature {
    let height = support_rows.len() as u32;
    let width = support_rows.first().map_or(0, |row| row.len() as u32);
    let bright_profile = smoothed_soft_column_profile(support_rows);
    let ink_profile = smoothed_soft_column_profile(ink_rows);
    let row_profile = smoothed_soft_row_profile(support_rows);
    let components = select_soft_label_components(&bright_profile, &row_profile, width, height);
    let support_mask = build_support_mask(width, height, &components);
    let ink_mask = build_ink_mask_from_soft_rows(ink_rows, &components, &support_mask);
    let feature_units = build_feature_units(&blur(&support_mask, 0.9), &blur(&ink_mask, 0.8));
    let weight_units = build_weight_units(&support_mask);
    let inked_label_count = count_inked_labels(&ink_mask, &components);
    let geometry_score = component_geometry_quality(&components);
    let support_coverage = mask_coverage(&support_mask);

    ProbeSignature {
        support_mask,
        ink_mask,
        feature_units,
        weight_units,
        bright_profile,
        ink_profile,
        label_count: components.len(),
        inked_label_count,
        geometry_score,
        support_coverage,
    }
}

#[cfg(feature = "ai-burn")]
fn smoothed_soft_column_profile(rows: &[Vec<f32>]) -> Vec<f32> {
    let width = rows.first().map_or(0, Vec::len);
    let height = rows.len().max(1) as f32;
    let profile = (0..width)
        .map(|x| rows.iter().map(|row| row[x]).sum::<f32>() / height)
        .collect::<Vec<_>>();
    smooth_profile(&profile, 2)
}

#[cfg(feature = "ai-burn")]
fn smoothed_soft_row_profile(rows: &[Vec<f32>]) -> Vec<f32> {
    let width = rows.first().map_or(0, Vec::len).max(1) as f32;
    let profile = rows
        .iter()
        .map(|row| row.iter().sum::<f32>() / width)
        .collect::<Vec<_>>();
    smooth_profile(&profile, 1)
}

#[cfg(feature = "ai-burn")]
fn select_soft_label_components(
    bright_profile: &[f32],
    row_profile: &[f32],
    width: u32,
    height: u32,
) -> Vec<ProbeComponent> {
    if bright_profile.is_empty() || row_profile.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    let max_profile = bright_profile.iter().copied().fold(0.0f32, f32::max);
    let max_row = row_profile.iter().copied().fold(0.0f32, f32::max);
    if max_profile <= f32::EPSILON || max_row <= f32::EPSILON {
        return Vec::new();
    }

    let column_cutoff = (max_profile * 0.62).clamp(0.10, 0.52);
    let row_cutoff = (max_row * 0.58).clamp(0.08, 0.42);
    let min_run_width = (width.max(1) as f32 * MIN_COMPONENT_WIDTH_RATIO)
        .round()
        .max(2.0) as u32;

    let mut top = 0u32;
    while top + 1 < height && row_profile[top as usize] < row_cutoff {
        top += 1;
    }
    let mut bottom = height.saturating_sub(1);
    while bottom > top && row_profile[bottom as usize] < row_cutoff {
        bottom = bottom.saturating_sub(1);
    }
    top = top.saturating_sub(1);
    bottom = (bottom + 1).min(height.saturating_sub(1));

    let mut components = Vec::new();
    let mut x = 0u32;
    while x < width {
        while x < width && bright_profile[x as usize] < column_cutoff {
            x += 1;
        }
        if x >= width {
            break;
        }

        let start = x;
        while x < width && bright_profile[x as usize] >= column_cutoff {
            x += 1;
        }
        let end = x.saturating_sub(1);
        let run_width = end.saturating_sub(start) + 1;
        if run_width < min_run_width {
            continue;
        }

        components.push(ProbeComponent {
            area: run_width * (bottom.saturating_sub(top) + 1),
            left: start,
            top,
            right: end,
            bottom,
        });
    }

    components
}

#[cfg(feature = "ai-burn")]
fn build_ink_mask_from_soft_rows(
    ink_rows: &[Vec<f32>],
    components: &[ProbeComponent],
    support_mask: &GrayImage,
) -> GrayImage {
    let width = support_mask.width();
    let height = support_mask.height();
    let mut mask = GrayImage::from_pixel(width, height, Luma([0]));
    for component in components {
        let left = component.left.min(width.saturating_sub(1));
        let top = component.top.min(height.saturating_sub(1));
        let right = component.right.min(width.saturating_sub(1)).max(left);
        let bottom = component.bottom.min(height.saturating_sub(1)).max(top);
        let mut peak = 0.0f32;
        for y in top..=bottom {
            for x in left..=right {
                peak = peak.max(ink_rows[y as usize][x as usize]);
            }
        }
        let threshold = (peak * 0.52).clamp(0.05, 0.42);
        for y in top..=bottom {
            for x in left..=right {
                if support_mask.get_pixel(x, y).0[0] == 0 {
                    continue;
                }
                if ink_rows[y as usize][x as usize] >= threshold {
                    mask.put_pixel(x, y, Luma([255]));
                }
            }
        }
    }
    mask
}

fn smooth_profile(profile: &[f32], radius: usize) -> Vec<f32> {
    if profile.is_empty() {
        return Vec::new();
    }

    let mut smoothed = profile.to_vec();
    for index in 0..profile.len() {
        let left = index.saturating_sub(radius);
        let right = (index + radius).min(profile.len() - 1);
        let window = &profile[left..=right];
        smoothed[index] = window.iter().sum::<f32>() / window.len() as f32;
    }
    smoothed
}

fn normalize_probe_capture(image: &GrayImage) -> GrayImage {
    let width = image.width().max(1);
    let height = image.height().max(1);
    if width.max(height) <= MAX_TEMPLATE_SIDE {
        return image.clone();
    }

    let scale = MAX_TEMPLATE_SIDE as f32 / width.max(height) as f32;
    resize(
        image,
        (width as f32 * scale).round().max(1.0) as u32,
        (height as f32 * scale).round().max(1.0) as u32,
        FilterType::Triangle,
    )
}

fn percentile(values: &[u8], ratio: f32) -> u8 {
    if values.is_empty() {
        return 255;
    }

    let ratio = ratio.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f32 * ratio).round() as usize;
    values[index.min(values.len() - 1)]
}

fn mask_coverage(mask: &GrayImage) -> f32 {
    let active = mask.pixels().filter(|pixel| pixel.0[0] > 0).count() as f32;
    let total = (mask.width().max(1) * mask.height().max(1)) as f32;
    active / total
}

fn resize_to_match(image: &GrayImage, target: &GrayImage) -> GrayImage {
    if image.width() == target.width() && image.height() == target.height() {
        return image.clone();
    }

    resize(
        image,
        target.width().max(1),
        target.height().max(1),
        FilterType::Triangle,
    )
}

fn weighted_mean(values: &[f32], mask: &[f32], mask_weight_sum: f32) -> f32 {
    values
        .iter()
        .zip(mask)
        .map(|(value, weight)| value * weight)
        .sum::<f32>()
        / mask_weight_sum.max(f32::EPSILON)
}

fn centered_weighted_norm(values: &[f32], mask: &[f32], mean: f32) -> f32 {
    values
        .iter()
        .zip(mask)
        .map(|(value, weight)| (value - mean) * weight)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn weighted_pearson_correlation_units(
    lhs: &[f32],
    rhs: &[f32],
    mask: &[f32],
    mask_weight_sum: f32,
    rhs_mean: f32,
    rhs_norm: f32,
) -> f32 {
    let mut dot = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let lhs_mean = weighted_mean(lhs, mask, mask_weight_sum);

    for ((lhs_value, rhs_value), weight) in lhs.iter().zip(rhs).zip(mask) {
        if *weight <= 0.0 {
            continue;
        }

        let lhs_weighted = (lhs_value - lhs_mean) * weight;
        let rhs_weighted = (rhs_value - rhs_mean) * weight;
        dot += lhs_weighted * rhs_weighted;
        lhs_norm += lhs_weighted * lhs_weighted;
    }

    let lhs_norm = lhs_norm.sqrt();
    if lhs_norm <= f32::EPSILON || rhs_norm <= f32::EPSILON {
        return 0.0;
    }

    (dot / (lhs_norm * rhs_norm)).clamp(0.0, 1.0)
}

fn save_gray_image(path: &Path, image: &GrayImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create minimap presence probe directory {}",
                parent.display()
            )
        })?;
    }

    let temp_path = path.with_extension("tmp");
    image.save(&temp_path).with_context(|| {
        format!(
            "failed to write minimap presence probe image {}",
            temp_path.display()
        )
    })?;
    if path.exists() {
        fs::remove_file(path).with_context(|| {
            format!(
                "failed to replace existing minimap presence probe image {}",
                path.display()
            )
        })?;
    }
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to finalize minimap presence probe image {}",
            path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
fn score_probe_images_cpu(template: &GrayImage, current: &GrayImage) -> ProbeScoreBreakdown {
    let template = normalize_probe_capture(template);
    let current = resize_to_match(&normalize_probe_capture(current), &template);
    let template_signature = build_probe_signature(&template);
    let current_signature = build_probe_signature(&current);
    let mask_weight_sum = template_signature
        .weight_units
        .iter()
        .sum::<f32>()
        .max(f32::EPSILON);
    let template_mean = weighted_mean(
        &template_signature.feature_units,
        &template_signature.weight_units,
        mask_weight_sum,
    );
    let structure_score = weighted_pearson_correlation_units(
        &current_signature.feature_units,
        &template_signature.feature_units,
        &template_signature.weight_units,
        mask_weight_sum,
        template_mean,
        centered_weighted_norm(
            &template_signature.feature_units,
            &template_signature.weight_units,
            template_mean,
        ),
    );
    build_probe_score_breakdown(&template_signature, &current_signature, structure_score)
}

#[cfg(test)]
fn asset_test_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("test")
}

#[cfg(test)]
fn load_test_image(name: &str) -> GrayImage {
    image::open(asset_test_root().join(name))
        .unwrap_or_else(|error| panic!("failed to load test image {name}: {error:#}"))
        .into_luma8()
}

#[cfg(test)]
fn default_test_probe_region() -> CaptureRegion {
    CaptureRegion {
        top: 116,
        left: 1660,
        width: 590,
        height: 38,
    }
}

#[cfg(test)]
fn configured_test_probe_region() -> Option<CaptureRegion> {
    let workspace_root = ProjectDirs::from("io", "rocom", "game-map-tracker-rs")
        .map(|dirs| dirs.data_local_dir().to_path_buf())?;
    let config = load_existing_config(&workspace_root).ok()?;
    config.minimap_presence_probe.capture_region()
}

#[cfg(test)]
fn effective_test_probe_region() -> CaptureRegion {
    configured_test_probe_region().unwrap_or_else(default_test_probe_region)
}

#[cfg(test)]
fn crop_test_region(image: &GrayImage, region: &CaptureRegion) -> GrayImage {
    let left = region.left.max(0) as u32;
    let top = region.top.max(0) as u32;
    assert!(
        left + region.width <= image.width() && top + region.height <= image.height(),
        "probe region {:?} is outside image {}x{}",
        region,
        image.width(),
        image.height()
    );
    crop_imm(image, left, top, region.width, region.height).to_image()
}

#[cfg(test)]
fn synthetic_probe(
    include_labels: bool,
    alternate_background: bool,
    add_clutter: bool,
) -> GrayImage {
    let width = 240;
    let height = 44;
    let mut image = GrayImage::from_fn(width, height, |x, y| {
        let base = if alternate_background {
            24 + (((x * 7 + y * 13) % 46) as u8)
        } else {
            18 + (((x * 3 + y * 5) % 30) as u8)
        };
        Luma([base])
    });

    if add_clutter {
        for stripe in [18, 92, 171] {
            for y in 4..14 {
                for x in stripe..(stripe + 16).min(width) {
                    image.put_pixel(x, y, Luma([210]));
                }
            }
        }
    }

    if include_labels {
        for index in 0..6 {
            let left = 8 + index * 38;
            draw_synthetic_label(&mut image, left, 11, 28, 20);
        }
    }

    image
}

#[cfg(test)]
fn draw_synthetic_label(image: &mut GrayImage, left: u32, top: u32, width: u32, height: u32) {
    let radius = 5i32;
    let right = left + width - 1;
    let bottom = top + height - 1;
    for y in top..=bottom {
        for x in left..=right {
            let dx = if x < left + radius as u32 {
                left as i32 + radius - x as i32
            } else if x > right.saturating_sub(radius as u32) {
                x as i32 - (right as i32 - radius)
            } else {
                0
            };
            let dy = if y < top + radius as u32 {
                top as i32 + radius - y as i32
            } else if y > bottom.saturating_sub(radius as u32) {
                y as i32 - (bottom as i32 - radius)
            } else {
                0
            };
            if dx * dx + dy * dy > radius * radius {
                continue;
            }

            let is_border = x == left || x == right || y == top || y == bottom;
            image.put_pixel(x, y, Luma([if is_border { 246 } else { 226 }]));
        }
    }

    let mid_y = top + height / 2;
    for x in left + 8..left + width - 8 {
        image.put_pixel(x, mid_y, Luma([28]));
    }
    for y in top + 5..bottom - 4 {
        image.put_pixel(left + 10, y, Luma([28]));
        image.put_pixel(left + width - 11, y, Luma([28]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::test_support::{benchmark_repeated, print_perf_per_op};

    #[test]
    fn probe_score_stays_high_when_labels_remain_visible() {
        let (score, elapsed) = benchmark_repeated(1, 5, || {
            score_probe_images_cpu(
                &synthetic_probe(true, false, false),
                &synthetic_probe(true, true, false),
            )
        });
        print_perf_per_op("presence/cpu", "synthetic_visible_score", 5, elapsed);

        assert!(
            score.final_score >= 0.78,
            "expected visible labels to keep a high probe score, got {:.3}",
            score.final_score
        );
    }

    #[test]
    fn probe_score_drops_when_labels_disappear_into_unrelated_background() {
        let (score, elapsed) = benchmark_repeated(1, 5, || {
            score_probe_images_cpu(
                &synthetic_probe(true, false, false),
                &synthetic_probe(false, true, true),
            )
        });
        print_perf_per_op("presence/cpu", "synthetic_missing_score", 5, elapsed);

        assert!(
            score.final_score <= 0.32,
            "expected missing labels to lower the probe score, got {:.3}",
            score.final_score
        );
    }

    #[test]
    fn badge_only_template_extracts_multiple_label_components() {
        let template = load_test_image("badge_only.png");
        let (signature, elapsed) = benchmark_repeated(1, 5, || build_probe_signature(&template));
        print_perf_per_op("presence/cpu", "badge_signature_build", 5, elapsed);
        assert!(
            signature.label_count >= 6,
            "expected badge template to yield 6 labels, got {}",
            signature.label_count
        );
    }

    #[test]
    fn real_has_map_regions_score_high() {
        let region = effective_test_probe_region();
        let template = load_test_image("badge_only.png");
        for name in ["has_map_1.png", "has_map_2.png"] {
            let image = load_test_image(name);
            let crop = crop_test_region(&image, &region);
            let (score, elapsed) =
                benchmark_repeated(1, 5, || score_probe_images_cpu(&template, &crop));
            print_perf_per_op("presence/cpu", &format!("{name}_score"), 5, elapsed);
            assert!(
                score.final_score >= 0.70,
                "expected {name} to keep a high probe score, got {:.3}",
                score.final_score
            );
        }
    }

    #[test]
    fn real_no_map_regions_score_low() {
        let region = effective_test_probe_region();
        let template = load_test_image("badge_only.png");
        for name in [
            "no_map_1.png",
            "no_map_2.png",
            "no_map_3.png",
            "no_map_4.png",
            "no_map_5.png",
            "no_map_6.png",
        ] {
            let image = load_test_image(name);
            let crop = crop_test_region(&image, &region);
            let (score, elapsed) =
                benchmark_repeated(1, 5, || score_probe_images_cpu(&template, &crop));
            print_perf_per_op("presence/cpu", &format!("{name}_score"), 5, elapsed);
            assert!(
                score.final_score <= 0.35,
                "expected {name} to stay low without badge labels, got {:.3}",
                score.final_score
            );
        }
    }
}
