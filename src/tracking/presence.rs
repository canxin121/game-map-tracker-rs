use std::{sync::mpsc, thread, time::Duration};

use anyhow::Result;
#[cfg(feature = "ai-burn")]
use burn::tensor::{
    Tensor, TensorData,
    backend::Backend,
    module::{conv2d, max_pool2d},
    ops::ConvOptions,
};
#[cfg(test)]
use directories::ProjectDirs;
use image::{
    DynamicImage, GrayImage, Luma, RgbaImage,
    imageops::{FilterType, blur, crop_imm, resize},
};
#[cfg(test)]
use std::path::PathBuf;
use tracing::warn;

#[cfg(test)]
use crate::config::CaptureRegion;
#[cfg(test)]
use crate::config::load_existing_config;
use crate::{
    config::AiDevicePreference,
    resources::WorkspaceSnapshot,
    tracking::{
        capture::DesktopCapture,
        debug::{DebugField, DebugImage},
        vision::{capture_template_annulus, gray_image_as_unit_vec, preview_image},
    },
};

#[cfg(feature = "ai-burn")]
use crate::tracking::burn_support::{
    BurnDeviceConfig, BurnDeviceSelection, burn_device_label, select_burn_device,
};
#[cfg(all(test, feature = "ai-burn"))]
use crate::tracking::burn_support::{
    available_burn_backend_preferences, available_burn_device_descriptors,
};

const MAX_CAPTURE_SIDE: u32 = 384;
const GPU_BACKEND_WARMUP_TIMEOUT: Duration = Duration::from_millis(1500);
const CIRCLE_ANGLE_SAMPLES: usize = 144;
const CIRCLE_SEARCH_MIN_RATIO: f32 = 0.72;
const CIRCLE_SEARCH_MAX_RATIO: f32 = 0.92;
const MAP_RING_INSET_PX: u32 = 2;
const SUPPORT_THRESHOLD: f32 = 0.45;
const CIRCULARITY_SUPPORT_MIN: f32 = 0.25;
const CIRCULARITY_SUPPORT_IDEAL: f32 = 0.50;

pub struct MinimapPresenceDetector {
    capture: DesktopCapture,
    threshold: f32,
    arrow_hole_ratio: f32,
    backend: ProbeBackend,
}

#[derive(Debug, Clone)]
pub struct MinimapPresenceSample {
    pub present: bool,
    pub score: f32,
    pub border_score: f32,
    pub circularity_score: f32,
    pub contrast_score: f32,
    pub radius_score: f32,
    pub threshold: f32,
    pub current_preview: GrayImage,
    pub current_dark_response: GrayImage,
    pub current_border_mask: GrayImage,
    pub current_map_ring: GrayImage,
    pub radius_ratio: f32,
    pub arc_coverage: f32,
    pub quadrant_coverage: f32,
    pub border_contrast: f32,
}

enum ProbeBackend {
    Cpu,
    #[cfg(feature = "ai-burn")]
    Burn(Box<dyn ProbeTensorBackend>),
}

#[cfg(feature = "ai-burn")]
trait ProbeTensorBackend: Send + Sync {
    fn signature(&self, image: &GrayImage, arrow_hole_ratio: f32) -> Result<ProbeSignature>;
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
    dark_response: GrayImage,
    border_mask: GrayImage,
    map_ring: GrayImage,
    radius_ratio: f32,
    arc_coverage: f32,
    quadrant_coverage: f32,
    mean_strength: f32,
    border_contrast: f32,
    darkness: f32,
}

#[derive(Debug, Clone)]
struct ProbeScoreBreakdown {
    border_score: f32,
    circularity_score: f32,
    contrast_score: f32,
    radius_score: f32,
    final_score: f32,
}

#[derive(Debug, Clone)]
struct CandidateCircle {
    radius_px: u32,
    radius_ratio: f32,
    arc_coverage: f32,
    quadrant_coverage: f32,
    mean_strength: f32,
    border_contrast: f32,
    darkness: f32,
    final_score: f32,
}

impl MinimapPresenceDetector {
    pub fn new(workspace: &WorkspaceSnapshot) -> Result<Option<Self>> {
        let region = workspace.config.minimap.clone();
        if region.width < 48 || region.height < 48 {
            anyhow::bail!(
                "小地图圆形区域过小，当前配置为 top {} / left {} / {}x{}",
                region.top,
                region.left,
                region.width,
                region.height
            );
        }

        let probe = &workspace.config.minimap_presence_probe;
        let capture = DesktopCapture::from_absolute_region(&region)?;
        let arrow_hole_ratio = normalized_hole_ratio(
            workspace.config.template.mask_inner_radius,
            workspace.config.template.mask_outer_radius,
        );
        let backend = build_backend_with_warmup(
            probe.device,
            probe.device_index,
            region.width,
            region.height,
        );

        Ok(Some(Self {
            capture,
            threshold: probe.match_threshold,
            arrow_hole_ratio,
            backend,
        }))
    }

    pub fn sample(&self) -> Result<MinimapPresenceSample> {
        let current_preview = prepare_probe_capture(self.capture.capture_rgba()?);
        let (current_signature, scores) = analyze_probe_input_with_backend(
            &self.backend,
            &current_preview,
            self.arrow_hole_ratio,
        )?;

        Ok(MinimapPresenceSample {
            present: scores.final_score >= self.threshold,
            score: scores.final_score,
            border_score: scores.border_score,
            circularity_score: scores.circularity_score,
            contrast_score: scores.contrast_score,
            radius_score: scores.radius_score,
            threshold: self.threshold,
            current_preview,
            current_dark_response: current_signature.dark_response,
            current_border_mask: current_signature.border_mask,
            current_map_ring: current_signature.map_ring,
            radius_ratio: current_signature.radius_ratio,
            arc_coverage: current_signature.arc_coverage,
            quadrant_coverage: current_signature.quadrant_coverage,
            border_contrast: current_signature.border_contrast,
        })
    }

    #[must_use]
    pub fn debug_images(&self, sample: &MinimapPresenceSample) -> Vec<DebugImage> {
        vec![
            preview_image("Minimap Circle Input", &sample.current_preview, &[], 196),
            preview_image(
                "Minimap Circle Dark Response",
                &sample.current_dark_response,
                &[],
                196,
            ),
            preview_image("Minimap Ring", &sample.current_map_ring, &[], 196),
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
            DebugField::new("圆形得分", format!("{:.3}", sample.circularity_score)),
            DebugField::new("边框得分", format!("{:.3}", sample.border_score)),
            DebugField::new("对比得分", format!("{:.3}", sample.contrast_score)),
            DebugField::new("半径得分", format!("{:.3}", sample.radius_score)),
            DebugField::new("圆弧覆盖", format!("{:.0}%", sample.arc_coverage * 100.0)),
            DebugField::new(
                "象限覆盖",
                format!("{:.0}%", sample.quadrant_coverage * 100.0),
            ),
            DebugField::new("边框对比", format!("{:.1}", sample.border_contrast)),
            DebugField::new("半径比例", format!("{:.3}", sample.radius_ratio)),
            DebugField::new("探针阈值", format!("{:.3}", sample.threshold)),
        ]
    }

    fn backend_label(&self) -> String {
        match &self.backend {
            ProbeBackend::Cpu => "CPU".to_owned(),
            #[cfg(feature = "ai-burn")]
            ProbeBackend::Burn(backend) => backend.label(),
        }
    }
}

fn build_probe_signature(image: &GrayImage, arrow_hole_ratio: f32) -> ProbeSignature {
    let dark_response = detect_dark_border_response(image);
    build_probe_signature_from_dark_response(image, dark_response, arrow_hole_ratio)
}

fn build_probe_signature_from_dark_response(
    image: &GrayImage,
    dark_response: GrayImage,
    arrow_hole_ratio: f32,
) -> ProbeSignature {
    let candidate = detect_best_circle_candidate(image, &dark_response);
    let border_mask = build_border_mask(
        image.width(),
        image.height(),
        candidate.radius_px,
        &dark_response,
    );
    let map_ring = extract_map_ring(image, candidate.radius_px, arrow_hole_ratio);

    ProbeSignature {
        dark_response,
        border_mask,
        map_ring,
        radius_ratio: candidate.radius_ratio,
        arc_coverage: candidate.arc_coverage,
        quadrant_coverage: candidate.quadrant_coverage,
        mean_strength: candidate.mean_strength,
        border_contrast: candidate.border_contrast,
        darkness: candidate.darkness,
    }
}

fn detect_dark_border_response(image: &GrayImage) -> GrayImage {
    let background = blur(image, 4.0);
    let raw = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let baseline = background.get_pixel(x, y).0[0];
        let value = image.get_pixel(x, y).0[0];
        Luma([baseline.saturating_sub(value)])
    });
    blur(&raw, 1.2)
}

fn detect_best_circle_candidate(image: &GrayImage, dark_response: &GrayImage) -> CandidateCircle {
    let max_radius = image.width().min(image.height()).max(2) as f32 * 0.5 - 1.0;
    let start = (max_radius * CIRCLE_SEARCH_MIN_RATIO).floor().max(12.0) as u32;
    let end = (max_radius * CIRCLE_SEARCH_MAX_RATIO)
        .ceil()
        .max(start as f32) as u32;

    let mut best = CandidateCircle {
        radius_px: start,
        radius_ratio: start as f32 / max_radius.max(1.0),
        arc_coverage: 0.0,
        quadrant_coverage: 0.0,
        mean_strength: 0.0,
        border_contrast: 0.0,
        darkness: 0.0,
        final_score: 0.0,
    };
    for radius_px in start..=end {
        let candidate = score_circle_candidate(image, dark_response, radius_px, max_radius);
        if candidate.final_score > best.final_score {
            best = candidate;
        }
    }
    best
}

fn score_circle_candidate(
    image: &GrayImage,
    dark_response: &GrayImage,
    radius_px: u32,
    max_radius: f32,
) -> CandidateCircle {
    let center_x = (image.width().saturating_sub(1)) as f32 * 0.5;
    let center_y = (image.height().saturating_sub(1)) as f32 * 0.5;
    let mut supported = 0usize;
    let mut quadrant_supported = [0usize; 4];
    let mut strength_sum = 0.0f32;
    let mut frame_contrast_sum = 0.0f32;
    let mut border_contrast_sum = 0.0f32;
    let mut dark_sum = 0.0f32;

    for index in 0..CIRCLE_ANGLE_SAMPLES {
        let angle = index as f32 / CIRCLE_ANGLE_SAMPLES as f32 * std::f32::consts::TAU;
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let radius_px = radius_px as i32;
        let interior_mean = sample_band_mean(
            image,
            center_x,
            center_y,
            cos_theta,
            sin_theta,
            radius_px - 6,
            radius_px - 3,
        );
        let gray_frame_mean = sample_band_mean(
            image,
            center_x,
            center_y,
            cos_theta,
            sin_theta,
            radius_px + 1,
            radius_px + 3,
        );
        let outer_border_mean = sample_band_mean(
            image,
            center_x,
            center_y,
            cos_theta,
            sin_theta,
            radius_px + 5,
            radius_px + 9,
        );
        let dark_mean = sample_band_mean(
            dark_response,
            center_x,
            center_y,
            cos_theta,
            sin_theta,
            radius_px + 4,
            radius_px + 9,
        );

        let frame_contrast = (interior_mean - gray_frame_mean).max(0.0);
        let border_contrast = (interior_mean - outer_border_mean).max(0.0);
        let frame_step = lower_bound_score(frame_contrast, 6.0, 22.0);
        let border_step = lower_bound_score(border_contrast, 12.0, 34.0);
        let dark_step = lower_bound_score(dark_mean, 4.0, 18.0);
        let strength = (frame_step * 0.35 + border_step * 0.40 + dark_step * 0.25).clamp(0.0, 1.0);

        if strength >= SUPPORT_THRESHOLD {
            supported += 1;
            quadrant_supported[(index * 4) / CIRCLE_ANGLE_SAMPLES] += 1;
        }

        strength_sum += strength;
        frame_contrast_sum += frame_contrast;
        border_contrast_sum += border_contrast;
        dark_sum += dark_mean;
    }

    let sample_count = CIRCLE_ANGLE_SAMPLES as f32;
    let arc_coverage = supported as f32 / sample_count;
    let quadrant_size = sample_count / 4.0;
    let quadrant_coverage = quadrant_supported
        .into_iter()
        .min()
        .map_or(0.0, |supported| supported as f32 / quadrant_size.max(1.0));
    let mean_strength = strength_sum / sample_count;
    let border_contrast = ((frame_contrast_sum / sample_count) * 0.40
        + (border_contrast_sum / sample_count) * 0.60)
        .max(0.0);
    let darkness = dark_sum / sample_count;
    let radius_ratio = radius_px as f32 / max_radius.max(1.0);

    let border_score =
        (mean_strength * 0.60 + lower_bound_score(darkness, 4.0, 18.0) * 0.40).clamp(0.0, 1.0);
    let circularity_score = (arc_coverage * 0.70 + quadrant_coverage * 0.30).clamp(0.0, 1.0);
    let circularity_support = lower_bound_score(
        circularity_score,
        CIRCULARITY_SUPPORT_MIN,
        CIRCULARITY_SUPPORT_IDEAL,
    );
    let contrast_score = lower_bound_score(border_contrast, 10.0, 28.0);
    let radius_score = preferred_range_score(radius_ratio, 0.78, 0.90, 0.95);
    let raw_score = (circularity_score * 0.38
        + contrast_score * 0.28 * circularity_support
        + border_score * 0.24 * circularity_support
        + radius_score * 0.10)
        .clamp(0.0, 1.0);
    let final_score = (raw_score * 1.35).clamp(0.0, 1.0);

    CandidateCircle {
        radius_px,
        radius_ratio,
        arc_coverage,
        quadrant_coverage,
        mean_strength,
        border_contrast,
        darkness,
        final_score,
    }
}

fn build_border_mask(
    width: u32,
    height: u32,
    radius_px: u32,
    dark_response: &GrayImage,
) -> GrayImage {
    let center_x = (width.saturating_sub(1)) as f32 * 0.5;
    let center_y = (height.saturating_sub(1)) as f32 * 0.5;
    let inner_radius = radius_px as f32;
    let outer_radius = radius_px as f32 + border_band_thickness(width, height);

    GrayImage::from_fn(width, height, |x, y| {
        let distance = radial_distance(width, height, center_x, center_y, x, y);
        let dark = dark_response.get_pixel(x, y).0[0];
        Luma([
            if distance >= inner_radius && distance <= outer_radius && dark >= 10 {
                255
            } else {
                0
            },
        ])
    })
}

fn extract_map_ring(image: &GrayImage, radius_px: u32, arrow_hole_ratio: f32) -> GrayImage {
    let radius_px = radius_px.saturating_sub(MAP_RING_INSET_PX).max(1);
    let diameter = radius_px
        .saturating_mul(2)
        .max(16)
        .min(image.width().min(image.height()).max(16));
    let left = image.width().saturating_sub(diameter) / 2;
    let top = image.height().saturating_sub(diameter) / 2;
    let square = crop_imm(image, left, top, diameter, diameter).to_image();
    capture_template_annulus(&square, arrow_hole_ratio, 1.0)
}

fn border_band_thickness(width: u32, height: u32) -> f32 {
    (width.min(height) as f32 * 0.055).clamp(6.0, 14.0)
}

fn sample_band_mean(
    image: &GrayImage,
    center_x: f32,
    center_y: f32,
    cos_theta: f32,
    sin_theta: f32,
    start_radius: i32,
    end_radius: i32,
) -> f32 {
    if end_radius < start_radius {
        return 0.0;
    }

    let mut sum = 0u64;
    let mut count = 0u64;
    for radius in start_radius..=end_radius {
        let radius = radius.max(0) as f32;
        let x = (center_x + cos_theta * radius)
            .round()
            .clamp(0.0, image.width().saturating_sub(1) as f32) as u32;
        let y = (center_y + sin_theta * radius)
            .round()
            .clamp(0.0, image.height().saturating_sub(1) as f32) as u32;
        sum += u64::from(image.get_pixel(x, y).0[0]);
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

fn radial_distance(width: u32, height: u32, center_x: f32, center_y: f32, x: u32, y: u32) -> f32 {
    let radius_x = width.max(1) as f32 * 0.5;
    let radius_y = height.max(1) as f32 * 0.5;
    let dx = (x as f32 - center_x) / radius_x.max(1.0);
    let dy = (y as f32 - center_y) / radius_y.max(1.0);
    (dx * dx + dy * dy).sqrt() * radius_x.min(radius_y)
}

fn score_probe_signature(signature: &ProbeSignature) -> ProbeScoreBreakdown {
    let border_score = (signature.mean_strength * 0.60
        + lower_bound_score(signature.darkness, 4.0, 18.0) * 0.40)
        .clamp(0.0, 1.0);
    let circularity_score =
        (signature.arc_coverage * 0.70 + signature.quadrant_coverage * 0.30).clamp(0.0, 1.0);
    let circularity_support = lower_bound_score(
        circularity_score,
        CIRCULARITY_SUPPORT_MIN,
        CIRCULARITY_SUPPORT_IDEAL,
    );
    let contrast_score = lower_bound_score(signature.border_contrast, 10.0, 28.0);
    let radius_score = preferred_range_score(signature.radius_ratio, 0.78, 0.90, 0.95);
    let raw_score = (circularity_score * 0.38
        + contrast_score * 0.28 * circularity_support
        + border_score * 0.24 * circularity_support
        + radius_score * 0.10)
        .clamp(0.0, 1.0);
    let final_score = (raw_score * 1.35).clamp(0.0, 1.0);

    ProbeScoreBreakdown {
        border_score,
        circularity_score,
        contrast_score,
        radius_score,
        final_score,
    }
}

fn lower_bound_score(actual: f32, minimum: f32, ideal: f32) -> f32 {
    if ideal <= minimum {
        return (actual >= minimum) as u8 as f32;
    }
    ((actual - minimum) / (ideal - minimum)).clamp(0.0, 1.0)
}

fn preferred_range_score(actual: f32, minimum: f32, ideal: f32, maximum: f32) -> f32 {
    if actual <= ideal {
        return lower_bound_score(actual, minimum, ideal);
    }
    if maximum <= ideal {
        return 1.0;
    }
    (1.0 - (actual - ideal) / (maximum - ideal)).clamp(0.0, 1.0)
}

fn normalized_hole_ratio(inner_radius: f32, outer_radius: f32) -> f32 {
    if outer_radius <= f32::EPSILON {
        0.0
    } else {
        (inner_radius / outer_radius).clamp(0.0, 1.0)
    }
}

fn prepare_probe_capture(rgba: RgbaImage) -> GrayImage {
    normalize_minimap_capture(&DynamicImage::ImageRgba8(rgba).into_luma8())
}

fn analyze_probe_input_with_backend(
    backend: &ProbeBackend,
    image: &GrayImage,
    arrow_hole_ratio: f32,
) -> Result<(ProbeSignature, ProbeScoreBreakdown)> {
    let signature = match backend {
        ProbeBackend::Cpu => build_probe_signature(image, arrow_hole_ratio),
        #[cfg(feature = "ai-burn")]
        ProbeBackend::Burn(backend) => backend.signature(image, arrow_hole_ratio)?,
    };
    let scores = score_probe_signature(&signature);
    Ok((signature, scores))
}

fn normalize_minimap_capture(image: &GrayImage) -> GrayImage {
    let side = image.width().min(image.height()).max(1);
    let left = image.width().saturating_sub(side) / 2;
    let top = image.height().saturating_sub(side) / 2;
    let square = crop_imm(image, left, top, side, side).to_image();
    if side <= MAX_CAPTURE_SIDE {
        return square;
    }

    resize(
        &square,
        MAX_CAPTURE_SIDE,
        MAX_CAPTURE_SIDE,
        FilterType::Triangle,
    )
}

fn build_backend(device: AiDevicePreference, device_index: usize) -> Result<ProbeBackend> {
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
                )?))
            }
            #[cfg(burn_vulkan_backend)]
            BurnDeviceSelection::Vulkan(device) => {
                ProbeBackend::Burn(Box::new(BurnProbeBackend::<burn::backend::Vulkan>::new(
                    device.clone(),
                    burn_device_label(&BurnDeviceSelection::Vulkan(device)),
                )?))
            }
            #[cfg(burn_metal_backend)]
            BurnDeviceSelection::Metal(device) => {
                ProbeBackend::Burn(Box::new(BurnProbeBackend::<burn::backend::Metal>::new(
                    device.clone(),
                    burn_device_label(&BurnDeviceSelection::Metal(device)),
                )?))
            }
        });
    }

    #[cfg(not(feature = "ai-burn"))]
    {
        let _ = device_index;
        anyhow::bail!(
            "小地图圆形探针配置选择了 {} 设备，但当前二进制未启用 `ai-burn` 特性；请改回 cpu 或重新构建带 Burn 后端的版本",
            device
        );
    }
}

fn build_backend_with_warmup(
    device: AiDevicePreference,
    device_index: usize,
    probe_width: u32,
    probe_height: u32,
) -> ProbeBackend {
    if device == AiDevicePreference::Cpu {
        return ProbeBackend::Cpu;
    }

    let thread_name = format!("minimap-circle-warmup-{device}-{device_index}");
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let spawn_result = thread::Builder::new().name(thread_name).spawn(move || {
        let result = try_build_backend_with_warmup(device, device_index, probe_width, probe_height);
        let _ = result_tx.send(result);
    });
    if let Err(error) = spawn_result {
        warn!(
            preferred_device = %device,
            device_index,
            error = %format!("{error:#}"),
            "failed to spawn minimap-circle backend warmup thread; falling back to CPU"
        );
        return ProbeBackend::Cpu;
    }

    match result_rx.recv_timeout(GPU_BACKEND_WARMUP_TIMEOUT) {
        Ok(Ok(backend)) => backend,
        Ok(Err(error)) => {
            warn!(
                preferred_device = %device,
                device_index,
                error = %format!("{error:#}"),
                "minimap-circle selected backend failed warmup; falling back to CPU"
            );
            ProbeBackend::Cpu
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            warn!(
                preferred_device = %device,
                device_index,
                timeout_ms = GPU_BACKEND_WARMUP_TIMEOUT.as_millis(),
                "minimap-circle selected backend warmup timed out; falling back to CPU"
            );
            ProbeBackend::Cpu
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            warn!(
                preferred_device = %device,
                device_index,
                "minimap-circle selected backend warmup disconnected; falling back to CPU"
            );
            ProbeBackend::Cpu
        }
    }
}

fn try_build_backend_with_warmup(
    device: AiDevicePreference,
    device_index: usize,
    probe_width: u32,
    probe_height: u32,
) -> Result<ProbeBackend> {
    let backend = build_backend(device, device_index)?;
    warmup_backend(&backend, probe_width, probe_height)?;
    Ok(backend)
}

fn warmup_backend(backend: &ProbeBackend, probe_width: u32, probe_height: u32) -> Result<()> {
    match backend {
        ProbeBackend::Cpu => Ok(()),
        #[cfg(feature = "ai-burn")]
        ProbeBackend::Burn(backend) => {
            let warmup = warmup_probe_frame(probe_width, probe_height);
            let _ = backend.signature(&warmup, 0.22)?;
            Ok(())
        }
    }
}

fn warmup_probe_frame(width: u32, height: u32) -> GrayImage {
    let side = width.min(height).max(128);
    let center = (side.saturating_sub(1)) as f32 * 0.5;
    let map_radius = side as f32 * 0.44;
    let frame_radius = side as f32 * 0.47;
    let border_radius = side as f32 * 0.53;
    GrayImage::from_fn(side, side, |x, y| {
        let dx = x as f32 - center;
        let dy = y as f32 - center;
        let distance = (dx * dx + dy * dy).sqrt();
        let value = if distance <= map_radius {
            184 + ((x * 7 + y * 11) % 42) as u8
        } else if distance <= frame_radius {
            118
        } else if distance <= border_radius {
            54
        } else {
            30 + ((x * 3 + y * 5) % 18) as u8
        };
        Luma([value])
    })
}

#[cfg(feature = "ai-burn")]
impl<B> BurnProbeBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn new(device: B::Device, device_label: String) -> Result<Self> {
        let box3_kernel = box_kernel_tensor::<B>(&device, 3);
        let box7_kernel = box_kernel_tensor::<B>(&device, 7);

        Ok(Self {
            device,
            device_label,
            box3_kernel,
            box7_kernel,
        })
    }
}

#[cfg(feature = "ai-burn")]
impl<B> ProbeTensorBackend for BurnProbeBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn signature(&self, image: &GrayImage, arrow_hole_ratio: f32) -> Result<ProbeSignature> {
        extract_burn_probe_signature::<B>(
            &self.device,
            &self.box3_kernel,
            &self.box7_kernel,
            image,
            arrow_hole_ratio,
        )
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
    arrow_hole_ratio: f32,
) -> Result<ProbeSignature>
where
    B: Backend<FloatElem = f32>,
{
    let image_tensor = gray_image_tensor::<B>(image, device);
    let background = conv2d(
        image_tensor.clone(),
        box7_kernel.clone(),
        None,
        ConvOptions::new([1, 1], [3, 3], [1, 1], 1),
    );
    let dark_soft = (conv2d(
        max_pool2d(
            (background - image_tensor).clamp(0.0, 1.0),
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            false,
        ),
        box3_kernel.clone(),
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    ) * 1.6)
        .clamp(0.0, 1.0);
    let dark_response = tensor_to_gray_image::<B>(dark_soft)?;
    Ok(build_probe_signature_from_dark_response(
        image,
        dark_response,
        arrow_hole_ratio,
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
fn tensor_to_gray_image<B>(tensor: Tensor<B, 4>) -> Result<GrayImage>
where
    B: Backend<FloatElem = f32>,
{
    let tensor = tensor.squeeze::<2>();
    let dims: [usize; 2] = tensor.shape().dims();
    let width = dims[1].max(1) as u32;
    let height = dims[0].max(1) as u32;
    let flat = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    Ok(GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let value = flat.get(index).copied().unwrap_or_default().clamp(0.0, 1.0);
        Luma([(value * 255.0).round().clamp(0.0, 255.0) as u8])
    }))
}

#[cfg(test)]
fn analyze_probe_capture_cpu(
    image: &RgbaImage,
    arrow_hole_ratio: f32,
) -> Result<(GrayImage, ProbeSignature, ProbeScoreBreakdown)> {
    let prepared = prepare_probe_capture(image.clone());
    let backend = ProbeBackend::Cpu;
    let (signature, scores) =
        analyze_probe_input_with_backend(&backend, &prepared, arrow_hole_ratio)?;
    Ok((prepared, signature, scores))
}

#[cfg(test)]
fn analyze_probe_capture_with_backend(
    backend: &ProbeBackend,
    image: &RgbaImage,
    arrow_hole_ratio: f32,
) -> Result<(GrayImage, ProbeSignature, ProbeScoreBreakdown)> {
    let prepared = prepare_probe_capture(image.clone());
    let (signature, scores) =
        analyze_probe_input_with_backend(backend, &prepared, arrow_hole_ratio)?;
    Ok((prepared, signature, scores))
}

#[cfg(test)]
fn asset_test_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("test")
}

#[cfg(test)]
fn load_test_capture(name: &str) -> RgbaImage {
    image::open(asset_test_root().join(name))
        .unwrap_or_else(|error| panic!("failed to load test image {name}: {error:#}"))
        .into_rgba8()
}

#[cfg(test)]
fn default_test_minimap_region() -> CaptureRegion {
    CaptureRegion {
        top: 56,
        left: 2287,
        width: 240,
        height: 240,
    }
}

#[cfg(test)]
fn configured_test_minimap_region() -> Option<CaptureRegion> {
    let workspace_root = ProjectDirs::from("io", "rocom", "game-map-tracker-rs")
        .map(|dirs| dirs.data_local_dir().to_path_buf())?;
    let config = load_existing_config(&workspace_root).ok()?;
    Some(config.minimap)
}

#[cfg(test)]
fn effective_test_minimap_region() -> CaptureRegion {
    configured_test_minimap_region().unwrap_or_else(default_test_minimap_region)
}

#[cfg(test)]
fn default_test_hole_ratio() -> f32 {
    0.22
}

#[cfg(test)]
fn configured_test_hole_ratio() -> Option<f32> {
    let workspace_root = ProjectDirs::from("io", "rocom", "game-map-tracker-rs")
        .map(|dirs| dirs.data_local_dir().to_path_buf())?;
    let config = load_existing_config(&workspace_root).ok()?;
    Some(normalized_hole_ratio(
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    ))
}

#[cfg(test)]
fn effective_test_hole_ratio() -> f32 {
    configured_test_hole_ratio().unwrap_or_else(default_test_hole_ratio)
}

#[cfg(all(test, feature = "ai-burn"))]
fn configured_test_backend() -> Option<(AiDevicePreference, usize)> {
    let workspace_root = ProjectDirs::from("io", "rocom", "game-map-tracker-rs")
        .map(|dirs| dirs.data_local_dir().to_path_buf())?;
    let config = load_existing_config(&workspace_root).ok()?;
    Some((
        config.minimap_presence_probe.device,
        config.minimap_presence_probe.device_index,
    ))
}

#[cfg(all(test, feature = "ai-burn"))]
fn available_test_backends() -> Vec<(AiDevicePreference, usize, String)> {
    let mut backends = Vec::new();
    for preference in available_burn_backend_preferences() {
        for descriptor in available_burn_device_descriptors(preference) {
            backends.push((preference, descriptor.ordinal, descriptor.name));
        }
    }
    backends
}

#[cfg(test)]
fn crop_test_region(image: &RgbaImage, region: &CaptureRegion) -> RgbaImage {
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
fn load_test_probe_capture(name: &str, region: &CaptureRegion) -> RgbaImage {
    let image = load_test_capture(name);
    if image.width() == region.width && image.height() == region.height {
        return image;
    }

    crop_test_region(&image, region)
}

#[cfg(test)]
fn list_test_image_names(prefix: &str) -> Vec<String> {
    let mut names = std::fs::read_dir(asset_test_root())
        .unwrap_or_else(|error| {
            panic!("failed to list test assets with prefix {prefix}: {error:#}")
        })
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with(prefix) && name.ends_with(".png"))
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[cfg(test)]
fn ring_mean(image: &GrayImage, inner_ratio: f32) -> f32 {
    let center_x = (image.width().saturating_sub(1)) as f32 * 0.5;
    let center_y = (image.height().saturating_sub(1)) as f32 * 0.5;
    let radius_x = image.width().max(1) as f32 * 0.5;
    let radius_y = image.height().max(1) as f32 * 0.5;
    let mut sum = 0u64;
    let mut count = 0u64;
    for (x, y, pixel) in image.enumerate_pixels() {
        let dx = (x as f32 - center_x) / radius_x.max(1.0);
        let dy = (y as f32 - center_y) / radius_y.max(1.0);
        let distance = (dx * dx + dy * dy).sqrt();
        if distance >= (inner_ratio + 0.04).clamp(0.0, 1.0) && distance <= 0.94 {
            sum += u64::from(pixel.0[0]);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::test_support::{benchmark_repeated, print_perf_per_op};

    #[test]
    fn real_has_map_regions_score_high_and_extract_ring() {
        let region = effective_test_minimap_region();
        let hole_ratio = effective_test_hole_ratio();
        let names = list_test_image_names("has_map_");
        assert!(
            !names.is_empty(),
            "expected at least one has_map_*.png sample in assets/test"
        );
        for name in names {
            let probe = load_test_probe_capture(&name, &region);
            let ((_, signature, score), elapsed) = benchmark_repeated(1, 5, || {
                analyze_probe_capture_cpu(&probe, hole_ratio).unwrap()
            });
            print_perf_per_op("presence/cpu", &format!("{name}_score"), 5, elapsed);
            println!(
                "{name}: final={:.3} arc={:.3} border={:.3} circularity={:.3} contrast={:.3} radius={:.3}",
                score.final_score,
                signature.arc_coverage,
                score.border_score,
                score.circularity_score,
                score.contrast_score,
                score.radius_score
            );

            assert!(
                score.final_score >= 0.75,
                "expected {name} to keep a high minimap-circle score, got {:.3}",
                score.final_score
            );
            assert!(
                signature.arc_coverage >= 0.68,
                "expected {name} to keep stable circular arc coverage, got {:.3}",
                signature.arc_coverage
            );

            let ring_mean = ring_mean(&signature.map_ring, hole_ratio);
            let center = signature
                .map_ring
                .get_pixel(
                    signature.map_ring.width() / 2,
                    signature.map_ring.height() / 2,
                )
                .0[0] as f32;
            let corner = signature.map_ring.get_pixel(0, 0).0[0] as f32;
            assert!(
                (center - ring_mean).abs() <= 14.0,
                "expected extracted ring center to be neutralized, got center {center:.1} vs ring {ring_mean:.1}",
            );
            assert!(
                (corner - ring_mean).abs() <= 14.0,
                "expected extracted ring outside corner to be neutralized, got corner {corner:.1} vs ring {ring_mean:.1}",
            );
        }
    }

    #[test]
    fn real_no_map_regions_score_low() {
        let region = effective_test_minimap_region();
        let hole_ratio = effective_test_hole_ratio();
        let names = list_test_image_names("no_map_");
        assert!(
            !names.is_empty(),
            "expected at least one no_map_*.png sample in assets/test"
        );
        for name in names {
            let probe = load_test_probe_capture(&name, &region);
            let ((_, signature, score), elapsed) = benchmark_repeated(1, 5, || {
                analyze_probe_capture_cpu(&probe, hole_ratio).unwrap()
            });
            print_perf_per_op("presence/cpu", &format!("{name}_score"), 5, elapsed);
            println!(
                "{name}: final={:.3} arc={:.3} border={:.3} circularity={:.3} contrast={:.3} radius={:.3}",
                score.final_score,
                signature.arc_coverage,
                score.border_score,
                score.circularity_score,
                score.contrast_score,
                score.radius_score
            );

            assert!(
                score.final_score <= 0.30,
                "expected {name} to stay low without the minimap circle, got {:.3}",
                score.final_score
            );
            assert!(
                signature.arc_coverage <= 0.24,
                "expected {name} to keep low arc coverage, got {:.3}",
                signature.arc_coverage
            );
        }
    }

    #[cfg(feature = "ai-burn")]
    #[test]
    fn configured_gpu_backend_scores_real_samples_when_available() -> Result<()> {
        let Some((device, device_index)) = configured_test_backend() else {
            return Ok(());
        };
        if device == AiDevicePreference::Cpu {
            return Ok(());
        }

        assert!(
            available_burn_device_descriptors(device)
                .iter()
                .any(|descriptor| descriptor.ordinal == device_index),
            "configured minimap-circle device {device}:{device_index} is not currently available"
        );

        let region = effective_test_minimap_region();
        let hole_ratio = effective_test_hole_ratio();
        let backend =
            try_build_backend_with_warmup(device, device_index, region.width, region.height)?;
        let label = match &backend {
            ProbeBackend::Cpu => {
                panic!("expected selected backend {device}:{device_index} to remain GPU-capable")
            }
            ProbeBackend::Burn(backend) => backend.label(),
        };

        let has_map = load_test_probe_capture("has_map_1.png", &region);
        let (_, _, has_score) = analyze_probe_capture_with_backend(&backend, &has_map, hole_ratio)?;

        assert!(
            has_score.final_score >= 0.75,
            "expected {label} to keep a high minimap-circle score for has_map_1.png, got {:.3}",
            has_score.final_score
        );
        for name in list_test_image_names("no_map_") {
            let no_map = load_test_probe_capture(&name, &region);
            let (_, _, no_score) =
                analyze_probe_capture_with_backend(&backend, &no_map, hole_ratio)?;
            assert!(
                no_score.final_score <= 0.30,
                "expected {label} to reject {name}, got {:.3}",
                no_score.final_score
            );
        }
        Ok(())
    }

    #[cfg(feature = "ai-burn")]
    #[test]
    fn all_available_backends_score_real_samples() -> Result<()> {
        let region = effective_test_minimap_region();
        let hole_ratio = effective_test_hole_ratio();
        let has_map_names = list_test_image_names("has_map_");
        let no_map_names = list_test_image_names("no_map_");
        let backends = available_test_backends();
        let mut failures = Vec::new();

        assert!(
            !has_map_names.is_empty(),
            "expected at least one has_map_*.png sample in assets/test"
        );
        assert!(
            !no_map_names.is_empty(),
            "expected at least one no_map_*.png sample in assets/test"
        );
        assert!(
            !backends.is_empty(),
            "expected at least one available minimap-circle backend"
        );

        for (device, device_index, device_name) in backends {
            let backend = match try_build_backend_with_warmup(
                device,
                device_index,
                region.width,
                region.height,
            ) {
                Ok(backend) => backend,
                Err(error) => {
                    failures.push(format!(
                        "failed to initialize device={device} index={device_index} name={device_name}: {error:#}"
                    ));
                    continue;
                }
            };
            let label = match &backend {
                ProbeBackend::Cpu => "CPU".to_owned(),
                ProbeBackend::Burn(backend) => backend.label(),
            };
            println!(
                "testing minimap-circle backend device={} index={} name={} label={}",
                device, device_index, device_name, label
            );

            for name in &has_map_names {
                let has_map = load_test_probe_capture(name, &region);
                let (_, signature, score) =
                    analyze_probe_capture_with_backend(&backend, &has_map, hole_ratio)?;
                println!(
                    "[{label}] {name}: final={:.3} arc={:.3} border={:.3} circularity={:.3} contrast={:.3} radius={:.3}",
                    score.final_score,
                    signature.arc_coverage,
                    score.border_score,
                    score.circularity_score,
                    score.contrast_score,
                    score.radius_score
                );
                if score.final_score < 0.75 {
                    failures.push(format!(
                        "expected {label} to keep a high minimap-circle score for {name}, got {:.3}",
                        score.final_score
                    ));
                }
            }

            for name in &no_map_names {
                let no_map = load_test_probe_capture(name, &region);
                let (_, signature, score) =
                    analyze_probe_capture_with_backend(&backend, &no_map, hole_ratio)?;
                println!(
                    "[{label}] {name}: final={:.3} arc={:.3} border={:.3} circularity={:.3} contrast={:.3} radius={:.3}",
                    score.final_score,
                    signature.arc_coverage,
                    score.border_score,
                    score.circularity_score,
                    score.contrast_score,
                    score.radius_score
                );
                if score.final_score > 0.30 {
                    failures.push(format!(
                        "expected {label} to reject {name}, got {:.3}",
                        score.final_score
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "minimap-circle backend validation failures:\n{}",
            failures.join("\n")
        );
        Ok(())
    }
}
