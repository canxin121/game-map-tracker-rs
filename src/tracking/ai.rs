use std::{sync::Arc, time::Duration};

use anyhow::Result;

#[cfg(all(
    feature = "ai-candle",
    any(feature = "ai-candle-cuda", feature = "ai-candle-metal")
))]
use anyhow::Context as _;
#[cfg(feature = "ai-candle")]
use image::GrayImage;
#[cfg(feature = "ai-candle")]
use std::path::PathBuf;

#[cfg(feature = "ai-candle")]
use crate::{
    config::{AiDevicePreference, AiTrackingConfig, AppConfig},
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackingSource},
    },
    tracking::{
        capture::{CaptureSource, DesktopCapture},
        debug::{DebugField, TrackingDebugSnapshot},
        vision::{
            DebugOverlay, MapPyramid, MaskSet, MatchCandidate, SearchCrop, SearchStage,
            TrackerState, build_debug_snapshot, center_to_scaled, crop_around_center,
            downscale_gray, load_logic_map_pyramid, preview_heatmap, preview_image,
            preview_mask_image,
        },
    },
};
use crate::{
    domain::tracker::TrackerEngineKind,
    resources::WorkspaceSnapshot,
    tracking::runtime::{TrackingStatus, TrackingTick, TrackingWorker},
};

#[cfg(feature = "ai-candle")]
use candle_core::{DType, Device, DeviceLocation, Tensor};
#[cfg(feature = "ai-candle")]
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct LocateResult {
    best_left: u32,
    best_top: u32,
    best_score: f32,
    score_width: u32,
    score_height: u32,
    score_map: Vec<f32>,
    accepted: Option<MatchCandidate>,
}

#[derive(Debug, Clone)]
pub struct CandleTrackerWorker {
    #[cfg(feature = "ai-candle")]
    inner: CandleTrackerInner,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct CandleTrackerInner {
    config: AppConfig,
    capture: DesktopCapture,
    pyramid: MapPyramid,
    masks: MaskSet,
    state: TrackerState,
    encoder: FixedFeatureEncoder,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct FixedFeatureEncoder {
    device: Device,
    edge_bank: Conv2d,
    source: EncoderSource,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
enum EncoderSource {
    Safetensors(PathBuf),
    BuiltIn,
}

#[cfg(feature = "ai-candle")]
impl EncoderSource {
    fn label(&self) -> String {
        match self {
            Self::Safetensors(path) => format!("Safetensors ({})", path.display()),
            Self::BuiltIn => "Built-in Conv Edge Bank".to_owned(),
        }
    }
}

impl CandleTrackerWorker {
    pub fn new(workspace: Arc<WorkspaceSnapshot>) -> Result<Self> {
        #[cfg(feature = "ai-candle")]
        {
            let config = workspace.config.clone();
            let capture = DesktopCapture::from_absolute_region(&config.minimap)?;
            let (pyramid, masks) = load_logic_map_pyramid(workspace.as_ref())?;
            let encoder = FixedFeatureEncoder::new(workspace.as_ref(), &config.ai)?;

            Ok(Self {
                inner: CandleTrackerInner {
                    config,
                    capture,
                    pyramid,
                    masks,
                    state: TrackerState::default(),
                    encoder,
                },
            })
        }

        #[cfg(not(feature = "ai-candle"))]
        {
            let _ = workspace;
            Ok(Self {})
        }
    }
}

#[cfg(feature = "ai-candle")]
impl FixedFeatureEncoder {
    fn new(workspace: &WorkspaceSnapshot, config: &AiTrackingConfig) -> Result<Self> {
        let device = select_candle_device(config)?;
        if let Some(encoder) = Self::load_from_safetensors(workspace, &device)? {
            return Ok(encoder);
        }

        let kernels = vec![
            -1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, // sobel x
            -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, // sobel y
            0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0, // laplacian
            -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0, // diag 1
            2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, // diag 2
            1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, // blur
        ];
        let weight = Tensor::from_vec(kernels, (6, 1, 3, 3), &device)?;
        let edge_bank = Conv2d::new(
            weight,
            None,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
        );

        Ok(Self {
            device,
            edge_bank,
            source: EncoderSource::BuiltIn,
        })
    }

    fn source_label(&self) -> String {
        self.source.label()
    }

    fn device_label(&self) -> String {
        candle_device_label(&self.device)
    }

    fn load_from_safetensors(
        workspace: &WorkspaceSnapshot,
        device: &Device,
    ) -> Result<Option<Self>> {
        for candidate in encoder_weight_candidates(workspace) {
            if !candidate.exists() {
                continue;
            }

            if let Ok(encoder) = Self::load_single_safetensors(&candidate, device) {
                return Ok(Some(encoder));
            }
        }

        Ok(None)
    }

    fn load_single_safetensors(path: &PathBuf, device: &Device) -> Result<Self> {
        let builder = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        let weight_name = tensor_name(
            &builder,
            &[
                "edge_bank.weight",
                "encoder.edge_bank.weight",
                "conv.weight",
                "weight",
            ],
        )?;
        let bias_name = [
            "edge_bank.bias",
            "encoder.edge_bank.bias",
            "conv.bias",
            "bias",
        ]
        .iter()
        .copied()
        .find(|name| builder.contains_tensor(name));
        let weight = builder.get_unchecked(weight_name)?;
        let (_, in_channels, kernel_h, kernel_w) = weight.dims4()?;
        if in_channels != 1 || kernel_h != kernel_w {
            anyhow::bail!(
                "unsupported encoder kernel shape {:?} in {}",
                weight.shape(),
                path.display()
            );
        }
        let bias = bias_name
            .map(|name| builder.get_unchecked(name))
            .transpose()?;
        let edge_bank = Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                padding: kernel_h / 2,
                ..Default::default()
            },
        );

        Ok(Self {
            device: device.clone(),
            edge_bank,
            source: EncoderSource::Safetensors(path.clone()),
        })
    }

    fn encode(&self, image: &GrayImage) -> Result<Tensor> {
        let height = image.height() as usize;
        let width = image.width() as usize;
        let base = Tensor::from_vec(
            image
                .pixels()
                .map(|pixel| f32::from(pixel.0[0]) / 255.0)
                .collect::<Vec<_>>(),
            (1, 1, height, width),
            &self.device,
        )?;
        let edges = self.edge_bank.forward(&base)?;
        let merged = Tensor::cat(&[&base, &edges], 1)?;
        Ok(merged)
    }
}

#[cfg(feature = "ai-candle")]
fn encoder_weight_candidates(workspace: &WorkspaceSnapshot) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = workspace
        .config
        .ai
        .weights_path
        .as_ref()
        .filter(|path| !path.trim().is_empty())
    {
        let candidate = PathBuf::from(path);
        if !candidate.is_absolute() {
            candidates.push(workspace.project_root.join(candidate));
        }
    }

    candidates.extend([
        workspace
            .project_root
            .join("models")
            .join("candle_edge_bank.safetensors"),
        workspace
            .project_root
            .join("models")
            .join("tracker_encoder.safetensors"),
    ]);
    candidates
}

#[cfg(feature = "ai-candle")]
fn tensor_name<'a>(builder: &VarBuilder<'a>, names: &[&'a str]) -> Result<&'a str> {
    names
        .iter()
        .copied()
        .find(|name| builder.contains_tensor(name))
        .ok_or_else(|| anyhow::anyhow!("failed to resolve encoder tensor names: {:?}", names))
}

#[cfg(feature = "ai-candle")]
impl CandleTrackerInner {
    fn run_frame(&mut self) -> Result<TrackingTick> {
        self.state.begin_frame();
        let captured = self.capture.capture_gray()?;
        let local_template = downscale_gray(&captured, self.pyramid.local.scale);
        let global_template = downscale_gray(&captured, self.pyramid.global.scale);

        let mut status = self.base_status();
        let mut estimate = None;
        let mut global_result = None;
        let mut refine_crop = None;
        let mut refine_result = None;
        let mut local_result = None;

        if self.config.local_search.enabled && matches!(self.state.stage, SearchStage::LocalTrack) {
            if let Some(last_world) = self.state.last_world {
                let crop = crop_around_center(
                    &self.pyramid.local.image,
                    center_to_scaled(last_world, self.pyramid.local.scale),
                    self.config.local_search.radius_px / self.pyramid.local.scale.max(1),
                    local_template.width(),
                    local_template.height(),
                )?;
                let result = self.locate_with_candle(
                    &crop.image,
                    &local_template,
                    &self.masks.local,
                    self.config.template.local_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.local.scale,
                )?;
                refine_crop = Some(crop.clone());
                local_result = Some(result.clone());

                if let (Some(candidate), Some(last_world)) =
                    (result.accepted.clone(), self.state.last_world)
                {
                    let jump = (candidate.world.x - last_world.x).abs()
                        + (candidate.world.y - last_world.y).abs();
                    if jump <= self.config.local_search.max_accepted_jump_px as f32 {
                        status.source = Some(TrackingSource::CandleEmbedding);
                        status.match_score = Some(candidate.score);
                        status.message = format!(
                            "卷积特征匹配局部锁定成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                            candidate.score, candidate.world.x, candidate.world.y
                        );
                        estimate =
                            Some(self.commit_success(candidate, TrackingSource::CandleEmbedding));
                    }
                }

                if estimate.is_none() {
                    let switched = self
                        .state
                        .increment_local_fail(self.config.local_search.lock_fail_threshold);
                    status.message = format!(
                        "卷积特征匹配局部锁定失败，第 {} 次重试。",
                        self.state.local_fail_streak
                    );
                    if switched {
                        status.message =
                            "卷积特征匹配局部锁定连续失败，切回全局重定位。".to_owned();
                    }
                }
            }
        }

        if estimate.is_none() {
            let result = self.locate_with_candle(
                &self.pyramid.global.image,
                &global_template,
                &self.masks.global,
                self.config.template.global_match_threshold,
                0,
                0,
                self.pyramid.global.scale,
            )?;
            global_result = Some(result.clone());

            if let Some(coarse) = result.accepted {
                let crop = crop_around_center(
                    &self.pyramid.local.image,
                    center_to_scaled(coarse.world, self.pyramid.local.scale),
                    self.config.template.global_refine_radius_px / self.pyramid.local.scale.max(1),
                    local_template.width(),
                    local_template.height(),
                )?;
                let refine = self.locate_with_candle(
                    &crop.image,
                    &local_template,
                    &self.masks.local,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.local.scale,
                )?;

                refine_crop = Some(crop.clone());
                refine_result = Some(refine.clone());
                if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                    status.source = Some(TrackingSource::CandleEmbedding);
                    status.match_score = Some(candidate.score);
                    status.message = format!(
                        "卷积特征匹配全局重定位成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                        candidate.score, candidate.world.x, candidate.world.y
                    );
                    estimate =
                        Some(self.commit_success(candidate, TrackingSource::CandleEmbedding));
                }
            }
        }

        if estimate.is_none() {
            estimate = self.apply_inertial_fallback(&mut status);
        }

        if estimate.is_none() {
            status.source = None;
            status.match_score = None;
            status.message = "卷积特征匹配当前帧未找到可靠匹配，等待下一帧。".to_owned();
        }

        let debug = Some(self.build_debug_snapshot(
            &captured,
            &global_template,
            global_result.as_ref(),
            refine_crop.as_ref(),
            refine_result.as_ref().or(local_result.as_ref()),
            estimate.as_ref(),
        ));

        Ok(TrackingTick {
            status,
            estimate,
            debug,
        })
    }

    fn base_status(&self) -> TrackingStatus {
        TrackingStatus {
            engine: TrackerEngineKind::ConvolutionFeatureMatch,
            frame_index: self.state.frame_index,
            lifecycle: crate::domain::tracker::TrackerLifecycle::Running,
            message: String::new(),
            source: None,
            match_score: None,
        }
    }

    fn commit_success(
        &mut self,
        candidate: MatchCandidate,
        source: TrackingSource,
    ) -> PositionEstimate {
        self.state.mark_success(candidate.world);
        PositionEstimate::tracked(candidate.world, source, Some(candidate.score), false)
    }

    fn apply_inertial_fallback(&mut self, status: &mut TrackingStatus) -> Option<PositionEstimate> {
        let last_world = self
            .state
            .next_inertial_position(self.config.max_lost_frames)?;
        status.source = Some(TrackingSource::InertialHold);
        status.match_score = None;
        status.message = format!(
            "卷积特征匹配进入惯性保位，第 {} / {} 帧。",
            self.state.lost_frames, self.config.max_lost_frames
        );

        Some(PositionEstimate::tracked(
            last_world,
            TrackingSource::InertialHold,
            None,
            true,
        ))
    }

    fn build_debug_snapshot(
        &self,
        captured: &GrayImage,
        global_template: &GrayImage,
        global_result: Option<&LocateResult>,
        refine_crop: Option<&SearchCrop>,
        refine_result: Option<&LocateResult>,
        estimate: Option<&PositionEstimate>,
    ) -> TrackingDebugSnapshot {
        let minimap = preview_image(
            "Minimap",
            captured,
            &[DebugOverlay::Crosshair {
                x: captured.width() / 2,
                y: captured.height() / 2,
            }],
            196,
        );

        let mut global_overlays = Vec::new();
        if let Some(global) = global_result {
            global_overlays.push(DebugOverlay::Rect {
                left: global.best_left,
                top: global.best_top,
                width: global_template.width(),
                height: global_template.height(),
            });
        }
        if let Some(position) = estimate {
            let (x, y) = center_to_scaled(position.world, self.pyramid.global.scale);
            global_overlays.push(DebugOverlay::Crosshair { x, y });
        }
        let global = preview_image(
            "Tensor Coarse",
            &self.pyramid.global.image,
            &global_overlays,
            196,
        );
        let global_mask = preview_mask_image("Global Mask", &self.masks.global, 196);
        let global_heatmap = global_result.map(|result| {
            preview_heatmap(
                "Tensor Coarse Heatmap",
                result.score_width,
                result.score_height,
                &result.score_map,
                Some((result.best_left, result.best_top)),
                196,
            )
        });

        let refine = if let Some(crop) = refine_crop {
            let mut overlays = Vec::new();
            if let Some(result) = refine_result {
                overlays.push(DebugOverlay::Rect {
                    left: result.best_left.saturating_sub(crop.origin_x),
                    top: result.best_top.saturating_sub(crop.origin_y),
                    width: self.masks.local.width(),
                    height: self.masks.local.height(),
                });
            }
            if let Some(position) = estimate {
                let x = (position.world.x as u32 / self.pyramid.local.scale.max(1))
                    .saturating_sub(crop.origin_x);
                let y = (position.world.y as u32 / self.pyramid.local.scale.max(1))
                    .saturating_sub(crop.origin_y);
                overlays.push(DebugOverlay::Crosshair { x, y });
            }
            preview_image("Tensor Refine", &crop.image, &overlays, 196)
        } else {
            preview_image("Tensor Refine", &self.pyramid.local.image, &[], 196)
        };
        let local_mask = preview_mask_image("Local Mask", &self.masks.local, 196);
        let refine_heatmap = refine_result.map(|result| {
            preview_heatmap(
                "Tensor Refine Heatmap",
                result.score_width,
                result.score_height,
                &result.score_map,
                Some((result.best_left, result.best_top)),
                196,
            )
        });

        let mut fields = vec![
            DebugField::new("阶段", self.state.stage.to_string()),
            DebugField::new("设备", self.encoder.device_label()),
            DebugField::new("编码器", self.encoder.source_label()),
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
        ];
        if let Some(result) = global_result {
            fields.push(DebugField::new(
                "粗匹配得分",
                format!("{:.3}", result.best_score),
            ));
        }
        if let Some(result) = refine_result {
            fields.push(DebugField::new(
                "精修得分",
                format!("{:.3}", result.best_score),
            ));
        }
        if let Some(position) = estimate {
            fields.push(DebugField::new(
                "输出坐标",
                format!("{:.0}, {:.0}", position.world.x, position.world.y),
            ));
        }

        build_debug_snapshot(
            TrackerEngineKind::ConvolutionFeatureMatch,
            self.state.frame_index,
            self.state.stage,
            vec![
                minimap,
                global_mask,
                global_heatmap.unwrap_or_else(|| {
                    preview_mask_image("Tensor Coarse Heatmap", &self.masks.global, 196)
                }),
                global,
                local_mask,
                refine_heatmap.unwrap_or_else(|| {
                    preview_mask_image("Tensor Refine Heatmap", &self.masks.local, 196)
                }),
                refine,
            ],
            fields,
        )
    }

    fn locate_with_candle(
        &self,
        image: &GrayImage,
        template: &GrayImage,
        mask: &GrayImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        if image.width() <= template.width() || image.height() <= template.height() {
            return Ok(LocateResult {
                best_left: 0,
                best_top: 0,
                best_score: f32::MIN,
                score_width: 0,
                score_height: 0,
                score_map: Vec::new(),
                accepted: None,
            });
        }

        let search = self.encoder.encode(image)?;
        let template = self.encoder.encode(template)?;
        let (_, channels, _, _) = search.dims4()?;
        let mask = Tensor::from_vec(
            repeated_mask(mask, channels),
            (1, channels, mask.height() as usize, mask.width() as usize),
            &self.encoder.device,
        )?;

        let masked_template = template.broadcast_mul(&mask)?;
        let numerator = search.conv2d(&masked_template, 0, 1, 1, 1)?;
        let search_patch_energy = search.sqr()?.conv2d(&mask, 0, 1, 1, 1)?;
        let template_energy = masked_template.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let denominator = ((&search_patch_energy * template_energy as f64)? + 1e-6)?.sqrt()?;
        let score_map = numerator.broadcast_div(&denominator)?;

        let scores = score_map.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        let score_height = scores.len() as u32;
        let score_width = scores.first().map_or(0, |row| row.len() as u32);
        let score_map = scores
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let mut best_left = 0usize;
        let mut best_top = 0usize;
        let mut best_score = f32::MIN;
        for (row_index, row) in scores.iter().enumerate() {
            for (column_index, score) in row.iter().enumerate() {
                if *score > best_score {
                    best_score = *score;
                    best_left = column_index;
                    best_top = row_index;
                }
            }
        }

        let accepted = if best_score >= threshold {
            Some(MatchCandidate {
                world: WorldPoint::new(
                    (origin_x + best_left as u32 + template.dims4()?.3 as u32 / 2) as f32
                        * scale as f32,
                    (origin_y + best_top as u32 + template.dims4()?.2 as u32 / 2) as f32
                        * scale as f32,
                ),
                score: best_score,
            })
        } else {
            None
        };

        Ok(LocateResult {
            best_left: best_left as u32,
            best_top: best_top as u32,
            best_score,
            score_width,
            score_height,
            score_map,
            accepted,
        })
    }
}

#[cfg(feature = "ai-candle")]
fn repeated_mask(mask: &GrayImage, channels: usize) -> Vec<f32> {
    let single = mask
        .pixels()
        .map(|pixel| f32::from(pixel.0[0]) / 255.0)
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(single.len() * channels);
    for _ in 0..channels {
        values.extend(single.iter().copied());
    }
    values
}

#[cfg(feature = "ai-candle")]
fn select_candle_device(config: &AiTrackingConfig) -> Result<Device> {
    match config.device {
        AiDevicePreference::Cpu => Ok(Device::Cpu),
        AiDevicePreference::Cuda => build_cuda_device(config.device_index),
        AiDevicePreference::Metal => build_metal_device(config.device_index),
    }
}

#[cfg(feature = "ai-candle")]
fn candle_device_label(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "CPU".to_owned(),
        DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("Metal:{gpu_id}"),
    }
}

#[cfg(feature = "ai-candle")]
fn available_candle_backends() -> &'static str {
    #[cfg(all(feature = "ai-candle-cuda", feature = "ai-candle-metal"))]
    {
        return "CPU / CUDA / Metal";
    }

    #[cfg(all(feature = "ai-candle-cuda", not(feature = "ai-candle-metal")))]
    {
        return "CPU / CUDA";
    }

    #[cfg(all(not(feature = "ai-candle-cuda"), feature = "ai-candle-metal"))]
    {
        return "CPU / Metal";
    }

    #[cfg(all(not(feature = "ai-candle-cuda"), not(feature = "ai-candle-metal")))]
    {
        "CPU"
    }
}

#[cfg(all(feature = "ai-candle", feature = "ai-candle-cuda"))]
fn build_cuda_device(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal)
        .with_context(|| format!("无法初始化 CUDA 设备 {ordinal}，请检查驱动、运行库和显卡状态"))
}

#[cfg(all(feature = "ai-candle", not(feature = "ai-candle-cuda")))]
fn build_cuda_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 CUDA 设备，但当前二进制未启用 `ai-candle-cuda` 特性；请使用 `cargo run --features ai-candle-cuda` 重新构建"
    )
}

#[cfg(all(feature = "ai-candle", feature = "ai-candle-metal"))]
fn build_metal_device(ordinal: usize) -> Result<Device> {
    Device::new_metal(ordinal)
        .with_context(|| format!("无法初始化 Metal 设备 {ordinal}，请检查系统和 GPU 支持状态"))
}

#[cfg(all(feature = "ai-candle", not(feature = "ai-candle-metal")))]
fn build_metal_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 Metal 设备，但当前二进制未启用 `ai-candle-metal` 特性；请使用 `cargo run --features ai-candle-metal` 重新构建"
    )
}

impl TrackingWorker for CandleTrackerWorker {
    fn refresh_interval(&self) -> Duration {
        #[cfg(feature = "ai-candle")]
        {
            return Duration::from_millis(self.inner.config.ai.refresh_rate_ms);
        }

        #[cfg(not(feature = "ai-candle"))]
        {
            Duration::from_millis(250)
        }
    }

    fn tick(&mut self) -> Result<TrackingTick> {
        #[cfg(feature = "ai-candle")]
        {
            return self.inner.run_frame();
        }

        #[cfg(not(feature = "ai-candle"))]
        {
            anyhow::bail!("卷积特征匹配后端被选中，但当前二进制未启用 `ai-candle` 特性")
        }
    }

    fn initial_status(&self) -> TrackingStatus {
        #[cfg(feature = "ai-candle")]
        let message = format!(
            "卷积特征匹配引擎已就绪：设备 {}，可用后端 {}，{} + 张量相似度搜索。",
            self.inner.encoder.device_label(),
            available_candle_backends(),
            self.inner.encoder.source_label()
        );

        #[cfg(not(feature = "ai-candle"))]
        let message =
            "卷积特征匹配引擎已就绪：设备 CPU，固定卷积特征组 + 张量相似度搜索。".to_owned();

        TrackingStatus::new(TrackerEngineKind::ConvolutionFeatureMatch, message)
    }

    fn engine_kind(&self) -> TrackerEngineKind {
        TrackerEngineKind::ConvolutionFeatureMatch
    }
}
