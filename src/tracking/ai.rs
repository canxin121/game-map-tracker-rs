use std::{sync::Arc, time::Duration};

use anyhow::Result;
#[cfg(feature = "ai-candle")]
use image::GrayImage;
#[cfg(feature = "ai-candle")]
use std::path::PathBuf;

#[cfg(feature = "ai-candle")]
use crate::{
    config::{AiTrackingConfig, AppConfig},
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackingSource},
    },
    tracking::{
        candle_support::{available_candle_backends, candle_device_label, select_candle_device},
        capture::{CaptureSource, DesktopCapture},
        debug::{DebugField, TrackingDebugSnapshot},
        precompute::{
            PersistedTensorCache, clear_match_pyramid_caches, clear_tensor_caches_by_prefix,
            load_or_build_match_pyramid, load_tensor_cache, metadata_fingerprint,
            save_tensor_cache, tracker_tensor_cache_path,
        },
        presence::{MinimapPresenceDetector, MinimapPresenceSample},
        vision::{
            DebugOverlay, MapPyramid, MaskSet, MatchCandidate, SearchCrop, SearchStage,
            TrackerState, build_debug_snapshot, build_mask, build_match_representation,
            center_to_scaled, crop_around_center, mask_as_unit_vec, preview_heatmap, preview_image,
            preview_mask_image, scaled_dimension,
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
    presence_detector: Option<MinimapPresenceDetector>,
    pyramid: MapPyramid,
    masks: MaskSet,
    state: TrackerState,
    matcher: CandleFeatureMatcher,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct CandleFeatureMatcher {
    encoder: FixedFeatureEncoder,
    global_search: SearchTensorCache,
    global_mask: Tensor,
    local_mask: Tensor,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct FixedFeatureEncoder {
    device: Device,
    edge_bank: Conv2d,
    source: EncoderSource,
    output_channels: usize,
}

#[derive(Debug, Clone)]
#[cfg(feature = "ai-candle")]
struct SearchTensorCache {
    features: Tensor,
    squared: Tensor,
    width: u32,
    height: u32,
    channels: usize,
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
            let presence_detector = MinimapPresenceDetector::new(workspace.as_ref())?;
            let prepared_pyramid = load_or_build_match_pyramid(workspace.as_ref())?;
            let map_cache_key = prepared_pyramid.cache_key;
            let pyramid = prepared_pyramid.pyramid;
            let masks = build_template_masks(&config);
            let matcher = CandleFeatureMatcher::new(
                workspace.as_ref(),
                &config.ai,
                &pyramid,
                &masks,
                &map_cache_key,
            )?;

            Ok(Self {
                inner: CandleTrackerInner {
                    config,
                    capture,
                    presence_detector,
                    pyramid,
                    masks,
                    state: TrackerState::default(),
                    matcher,
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

pub fn rebuild_convolution_engine_cache(workspace: &WorkspaceSnapshot) -> Result<()> {
    #[cfg(feature = "ai-candle")]
    {
        clear_match_pyramid_caches(workspace)?;
        clear_tensor_caches_by_prefix(workspace, "feature-global-search")?;

        let prepared_pyramid = load_or_build_match_pyramid(workspace)?;
        let masks = build_template_masks(&workspace.config);
        let _ = CandleFeatureMatcher::new(
            workspace,
            &workspace.config.ai,
            &prepared_pyramid.pyramid,
            &masks,
            &prepared_pyramid.cache_key,
        )?;
        return Ok(());
    }

    #[cfg(not(feature = "ai-candle"))]
    {
        let _ = workspace;
        anyhow::bail!("卷积特征匹配后端被选中，但当前二进制未启用 `ai-candle` 特性")
    }
}

#[cfg(feature = "ai-candle")]
const CUDA_CONV_IM2COL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

#[cfg(feature = "ai-candle")]
const METAL_CONV_IM2COL_BUDGET_BYTES: usize = 128 * 1024 * 1024;

#[cfg(feature = "ai-candle")]
const FEATURE_ENCODER_CACHE_VERSION: u32 = 1;

#[cfg(feature = "ai-candle")]
impl FixedFeatureEncoder {
    fn new(workspace: &WorkspaceSnapshot, config: &AiTrackingConfig) -> Result<Self> {
        let device = select_candle_device(config)?;
        if let Some(encoder) = Self::load_from_safetensors(workspace, &device)? {
            return Ok(encoder);
        }

        Self::built_in_with_device(device)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn built_in(config: &AiTrackingConfig) -> Result<Self> {
        Self::built_in_with_device(select_candle_device(config)?)
    }

    fn built_in_with_device(device: Device) -> Result<Self> {
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
            output_channels: 7,
        })
    }

    fn source_label(&self) -> String {
        self.source.label()
    }

    fn device_label(&self) -> String {
        candle_device_label(&self.device)
    }

    fn output_channels(&self) -> usize {
        self.output_channels
    }

    fn cache_key(&self) -> Result<String> {
        let key = match &self.source {
            EncoderSource::BuiltIn => {
                format!(
                    "ev{FEATURE_ENCODER_CACHE_VERSION}-builtin-c{}",
                    self.output_channels
                )
            }
            EncoderSource::Safetensors(path) => format!(
                "ev{FEATURE_ENCODER_CACHE_VERSION}-sf-{}-c{}",
                metadata_fingerprint(path)?,
                self.output_channels
            ),
        };
        Ok(key)
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
        let (out_channels, in_channels, kernel_h, kernel_w) = weight.dims4()?;
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
            output_channels: out_channels + 1,
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
fn prepare_capture_templates(
    captured: &GrayImage,
    config: &AppConfig,
    pyramid: &MapPyramid,
) -> (GrayImage, GrayImage) {
    (
        prepare_capture_template(
            captured,
            config.view_size,
            pyramid.local.scale,
            config.template.mask_outer_radius,
        ),
        prepare_capture_template(
            captured,
            config.view_size,
            pyramid.global.scale,
            config.template.mask_outer_radius,
        ),
    )
}

#[cfg(feature = "ai-candle")]
fn prepare_capture_template(
    captured: &GrayImage,
    view_size: u32,
    scale: u32,
    mask_outer_radius: f32,
) -> GrayImage {
    let diameter_px = ((captured.width().min(captured.height()) as f32) * mask_outer_radius)
        .round()
        .max(1.0) as u32;
    let offset_x = captured.width().saturating_sub(diameter_px) / 2;
    let offset_y = captured.height().saturating_sub(diameter_px) / 2;
    let square = image::imageops::crop_imm(captured, offset_x, offset_y, diameter_px, diameter_px)
        .to_image();
    let template_size = scaled_dimension(view_size.max(1), scale.max(1));
    let resized = if square.width() == template_size && square.height() == template_size {
        square
    } else {
        image::imageops::resize(
            &square,
            template_size,
            template_size,
            image::imageops::FilterType::Triangle,
        )
    };

    build_match_representation(&resized)
}

#[cfg(feature = "ai-candle")]
fn build_template_masks(config: &AppConfig) -> MaskSet {
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    let local_size = scaled_dimension(config.view_size.max(1), local_scale);
    let global_size = scaled_dimension(config.view_size.max(1), global_scale);

    MaskSet {
        local: build_mask(
            local_size,
            local_size,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
        global: build_mask(
            global_size,
            global_size,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
    }
}

#[cfg(feature = "ai-candle")]
impl CandleFeatureMatcher {
    fn new(
        workspace: &WorkspaceSnapshot,
        config: &AiTrackingConfig,
        pyramid: &MapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let encoder = FixedFeatureEncoder::new(workspace, config)?;
        Self::with_encoder_cached(workspace, map_cache_key, encoder, pyramid, masks)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn with_encoder(
        encoder: FixedFeatureEncoder,
        pyramid: &MapPyramid,
        masks: &MaskSet,
    ) -> Result<Self> {
        let global_search = SearchTensorCache::from_gray_image(&pyramid.global.image, &encoder)?;
        let global_mask = mask_tensor(&masks.global, encoder.output_channels(), &encoder.device)?;
        let local_mask = mask_tensor(&masks.local, encoder.output_channels(), &encoder.device)?;

        Ok(Self {
            encoder,
            global_search,
            global_mask,
            local_mask,
        })
    }

    fn with_encoder_cached(
        workspace: &WorkspaceSnapshot,
        map_cache_key: &str,
        encoder: FixedFeatureEncoder,
        pyramid: &MapPyramid,
        masks: &MaskSet,
    ) -> Result<Self> {
        let global_search = load_or_build_feature_global_search(
            workspace,
            map_cache_key,
            &pyramid.global.image,
            &encoder,
        )?;
        let global_mask = mask_tensor(&masks.global, encoder.output_channels(), &encoder.device)?;
        let local_mask = mask_tensor(&masks.local, encoder.output_channels(), &encoder.device)?;

        Ok(Self {
            encoder,
            global_search,
            global_mask,
            local_mask,
        })
    }

    fn device_label(&self) -> String {
        self.encoder.device_label()
    }

    fn source_label(&self) -> String {
        self.encoder.source_label()
    }

    fn locate_dynamic(
        &self,
        image: &GrayImage,
        template: &GrayImage,
        mask: &Tensor,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::from_gray_image(image, &self.encoder)?;
        self.locate_cached(
            &search, template, mask, threshold, origin_x, origin_y, scale,
        )
    }

    fn locate_cached(
        &self,
        search: &SearchTensorCache,
        template: &GrayImage,
        mask: &Tensor,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        if search.width <= template.width() || search.height <= template.height() {
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

        let template_features = self.encoder.encode(template)?;
        let weighted_template = template_features.broadcast_mul(mask)?;
        let template_energy = weighted_template.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let chunk_rows = candle_match_chunk_rows(
            &self.encoder.device,
            search.width,
            search.height,
            template.width(),
            template.height(),
            search.channels,
        );
        if chunk_rows
            < search
                .height
                .saturating_sub(template.height())
                .saturating_add(1)
        {
            return locate_cached_in_chunks(
                search,
                &weighted_template,
                mask,
                template_energy,
                threshold,
                origin_x,
                origin_y,
                scale,
                template.width(),
                template.height(),
                chunk_rows,
            );
        }

        let numerator = search.features.conv2d(&weighted_template, 0, 1, 1, 1)?;
        let search_patch_energy = search.squared.conv2d(mask, 0, 1, 1, 1)?;
        let denominator = ((&search_patch_energy * template_energy as f64)? + 1e-6)?.sqrt()?;
        let score_map = numerator.broadcast_div(&denominator)?;

        locate_result_from_score_map(
            score_map,
            threshold,
            origin_x,
            origin_y,
            scale,
            template.width(),
            template.height(),
        )
    }
}

#[cfg(feature = "ai-candle")]
impl SearchTensorCache {
    fn from_gray_image(image: &GrayImage, encoder: &FixedFeatureEncoder) -> Result<Self> {
        let features = encoder.encode(image)?;
        let (_, channels, height, width) = features.dims4()?;
        let squared = features.sqr()?;
        Ok(Self {
            features,
            squared,
            width: width as u32,
            height: height as u32,
            channels,
        })
    }

    fn from_persisted(cache: PersistedTensorCache, encoder: &FixedFeatureEncoder) -> Result<Self> {
        if cache.channels != encoder.output_channels() {
            anyhow::bail!(
                "feature search tensor cache channel count {} does not match encoder output {}",
                cache.channels,
                encoder.output_channels()
            );
        }

        let features = Tensor::from_vec(
            cache.primary,
            (
                1,
                cache.channels,
                cache.height as usize,
                cache.width as usize,
            ),
            &encoder.device,
        )?;
        let squared = Tensor::from_vec(
            cache.secondary,
            (
                1,
                cache.channels,
                cache.height as usize,
                cache.width as usize,
            ),
            &encoder.device,
        )?;
        Ok(Self {
            features,
            squared,
            width: cache.width,
            height: cache.height,
            channels: cache.channels,
        })
    }

    fn to_persisted(&self) -> Result<PersistedTensorCache> {
        let features = self
            .features
            .squeeze(0)?
            .to_vec3::<f32>()?
            .into_iter()
            .flat_map(|rows| rows.into_iter().flat_map(|row| row.into_iter()))
            .collect::<Vec<_>>();
        let squared = self
            .squared
            .squeeze(0)?
            .to_vec3::<f32>()?
            .into_iter()
            .flat_map(|rows| rows.into_iter().flat_map(|row| row.into_iter()))
            .collect::<Vec<_>>();
        PersistedTensorCache::from_parts(self.width, self.height, self.channels, features, squared)
    }
}

#[cfg(feature = "ai-candle")]
fn load_or_build_feature_global_search(
    workspace: &WorkspaceSnapshot,
    map_cache_key: &str,
    image: &GrayImage,
    encoder: &FixedFeatureEncoder,
) -> Result<SearchTensorCache> {
    let cache_key = format!("{map_cache_key}-{}", encoder.cache_key()?);
    let cache_path = tracker_tensor_cache_path(workspace, "feature-global-search", &cache_key);
    if let Ok(Some(cache)) = load_tensor_cache(&cache_path) {
        if let Ok(search) = SearchTensorCache::from_persisted(cache, encoder) {
            return Ok(search);
        }
    }

    let search = SearchTensorCache::from_gray_image(image, encoder)?;
    if let Ok(persisted) = search.to_persisted() {
        let _ = save_tensor_cache(&cache_path, &persisted);
    }
    Ok(search)
}

#[cfg(feature = "ai-candle")]
fn mask_tensor(mask: &GrayImage, channels: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        mask_as_unit_vec(mask, channels),
        (1, channels, mask.height() as usize, mask.width() as usize),
        device,
    )?)
}

#[cfg(feature = "ai-candle")]
fn candle_match_chunk_rows(
    device: &Device,
    search_width: u32,
    search_height: u32,
    template_width: u32,
    template_height: u32,
    channels: usize,
) -> u32 {
    let output_height = search_height
        .saturating_sub(template_height)
        .saturating_add(1);
    let output_width = search_width
        .saturating_sub(template_width)
        .saturating_add(1);
    if output_height == 0 || output_width == 0 {
        return 0;
    }

    let budget_bytes = match device.location() {
        DeviceLocation::Cpu => return output_height,
        DeviceLocation::Cuda { .. } => CUDA_CONV_IM2COL_BUDGET_BYTES,
        DeviceLocation::Metal { .. } => METAL_CONV_IM2COL_BUDGET_BYTES,
    };

    let per_output_row_bytes = output_width as usize
        * template_width as usize
        * template_height as usize
        * channels.max(1)
        * std::mem::size_of::<f32>();
    if per_output_row_bytes == 0 {
        return output_height;
    }

    ((budget_bytes / per_output_row_bytes).max(1) as u32).min(output_height)
}

#[cfg(feature = "ai-candle")]
fn locate_cached_in_chunks(
    search: &SearchTensorCache,
    weighted_template: &Tensor,
    mask: &Tensor,
    template_energy: f32,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
    chunk_rows: u32,
) -> Result<LocateResult> {
    let score_width = search.width - template_width + 1;
    let score_height = search.height - template_height + 1;
    let mut score_map = Vec::with_capacity(score_width as usize * score_height as usize);
    let mut output_row = 0u32;

    while output_row < score_height {
        let rows = chunk_rows.min(score_height - output_row).max(1);
        let slice_height = rows + template_height - 1;
        let features_chunk =
            search
                .features
                .narrow(2, output_row as usize, slice_height as usize)?;
        let squared_chunk = search
            .squared
            .narrow(2, output_row as usize, slice_height as usize)?;
        let numerator = features_chunk.conv2d(weighted_template, 0, 1, 1, 1)?;
        let search_patch_energy = squared_chunk.conv2d(mask, 0, 1, 1, 1)?;
        let denominator = ((&search_patch_energy * template_energy as f64)? + 1e-6)?.sqrt()?;
        let chunk_scores = numerator.broadcast_div(&denominator)?;
        let rows = chunk_scores.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        let produced_rows = rows.len() as u32;
        for row in rows {
            score_map.extend(row);
        }
        output_row += produced_rows;
    }

    Ok(locate_result_from_flat_scores(
        score_map,
        score_width,
        score_height,
        threshold,
        origin_x,
        origin_y,
        scale,
        template_width,
        template_height,
    ))
}

#[cfg(feature = "ai-candle")]
fn locate_result_from_score_map(
    score_map: Tensor,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
) -> Result<LocateResult> {
    let scores = score_map.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
    let score_height = scores.len() as u32;
    let score_width = scores.first().map_or(0, |row| row.len() as u32);
    let score_map = scores.into_iter().flatten().collect::<Vec<_>>();
    Ok(locate_result_from_flat_scores(
        score_map,
        score_width,
        score_height,
        threshold,
        origin_x,
        origin_y,
        scale,
        template_width,
        template_height,
    ))
}

#[cfg(feature = "ai-candle")]
fn locate_result_from_flat_scores(
    score_map: Vec<f32>,
    score_width: u32,
    score_height: u32,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
) -> LocateResult {
    let mut best_left = 0u32;
    let mut best_top = 0u32;
    let mut best_score = f32::MIN;

    for (index, score) in score_map.iter().enumerate() {
        if *score > best_score {
            best_score = *score;
            best_left = (index as u32) % score_width.max(1);
            best_top = (index as u32) / score_width.max(1);
        }
    }

    let accepted = if best_score >= threshold {
        Some(MatchCandidate {
            world: WorldPoint::new(
                (origin_x + best_left + template_width / 2) as f32 * scale as f32,
                (origin_y + best_top + template_height / 2) as f32 * scale as f32,
            ),
            score: best_score,
        })
    } else {
        None
    };

    LocateResult {
        best_left,
        best_top,
        best_score,
        score_width,
        score_height,
        score_map,
        accepted,
    }
}

#[cfg(feature = "ai-candle")]
impl CandleTrackerInner {
    fn run_frame(&mut self) -> Result<TrackingTick> {
        self.state.begin_frame();
        let probe_sample = self
            .presence_detector
            .as_ref()
            .map(MinimapPresenceDetector::sample)
            .transpose()?;
        if let Some(sample) = probe_sample.as_ref().filter(|sample| !sample.present) {
            let mut status = self.base_status();
            let estimate = self.apply_probe_absent_fallback(&mut status);
            let debug = Some(self.build_probe_miss_debug_snapshot(sample, estimate.as_ref()));
            return Ok(TrackingTick {
                status,
                estimate,
                debug,
            });
        }

        let captured = self.capture.capture_gray()?;
        let (local_template, global_template) =
            prepare_capture_templates(&captured, &self.config, &self.pyramid);

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
                let result = self.matcher.locate_dynamic(
                    &crop.image,
                    &local_template,
                    &self.matcher.local_mask,
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
            let result = self.matcher.locate_cached(
                &self.matcher.global_search,
                &global_template,
                &self.matcher.global_mask,
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
                let refine = self.matcher.locate_dynamic(
                    &crop.image,
                    &local_template,
                    &self.matcher.local_mask,
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
            probe_sample.as_ref(),
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

    fn apply_probe_absent_fallback(
        &mut self,
        status: &mut TrackingStatus,
    ) -> Option<PositionEstimate> {
        let estimate = self.apply_inertial_fallback(status);
        if estimate.is_some() {
            status.message = format!("F1-P 标签探针未命中，小地图疑似被遮挡，{}", status.message);
            return estimate;
        }

        status.source = None;
        status.match_score = None;
        status.message = "F1-P 标签探针未命中，小地图疑似被遮挡，等待界面恢复。".to_owned();
        None
    }

    fn build_debug_snapshot(
        &self,
        captured: &GrayImage,
        global_template: &GrayImage,
        global_result: Option<&LocateResult>,
        refine_crop: Option<&SearchCrop>,
        refine_result: Option<&LocateResult>,
        estimate: Option<&PositionEstimate>,
        probe_sample: Option<&MinimapPresenceSample>,
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
            DebugField::new("设备", self.matcher.device_label()),
            DebugField::new("编码器", self.matcher.source_label()),
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
        if let (Some(detector), Some(sample)) = (self.presence_detector.as_ref(), probe_sample) {
            fields.extend(detector.debug_fields(sample));
        }

        let mut images = vec![
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
        ];
        if let (Some(detector), Some(sample)) = (self.presence_detector.as_ref(), probe_sample) {
            images.extend(detector.debug_images(sample));
        }

        build_debug_snapshot(
            TrackerEngineKind::ConvolutionFeatureMatch,
            self.state.frame_index,
            self.state.stage,
            images,
            fields,
        )
    }

    fn build_probe_miss_debug_snapshot(
        &self,
        sample: &MinimapPresenceSample,
        estimate: Option<&PositionEstimate>,
    ) -> TrackingDebugSnapshot {
        let mut fields = vec![
            DebugField::new("阶段", self.state.stage.to_string()),
            DebugField::new("设备", self.matcher.device_label()),
            DebugField::new("编码器", self.matcher.source_label()),
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
        ];
        let mut images = Vec::new();

        if let Some(detector) = self.presence_detector.as_ref() {
            fields.extend(detector.debug_fields(sample));
            images.extend(detector.debug_images(sample));
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
            images,
            fields,
        )
    }
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
            self.inner.matcher.device_label(),
            available_candle_backends(),
            self.inner.matcher.source_label()
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

#[cfg(all(test, feature = "ai-candle"))]
mod tests {
    use std::{fs, path::PathBuf, sync::OnceLock};

    use super::*;
    use crate::{
        config::{AiDevicePreference, AppConfig, CaptureRegion},
        resources::{load_logic_map_scaled_image, raw_coordinate_to_world},
        tracking::vision::{ScaledMap, downscale_gray},
    };
    use anyhow::Result;
    use image::{
        GrayImage, Luma,
        imageops::{FilterType, crop_imm, replace, resize},
    };
    use serde::Deserialize;

    struct TestFixture {
        config: AppConfig,
        map: GrayImage,
        pyramid: MapPyramid,
        masks: MaskSet,
    }

    static FIXTURE: OnceLock<TestFixture> = OnceLock::new();

    #[derive(Debug, Deserialize)]
    struct FlatPointRecord {
        point: FlatPointCoordinate,
    }

    #[derive(Debug, Deserialize)]
    struct FlatPointCoordinate {
        lat: i32,
        lng: i32,
    }

    fn fixture() -> &'static TestFixture {
        FIXTURE.get_or_init(|| {
            let cache_root = bwiki_tiles_root();
            let raw_map = load_logic_map_scaled_image(&cache_root, 1)
                .expect("failed to assemble z8 logic map");
            let map = imageproc::contrast::equalize_histogram(&raw_map);

            let mut config = AppConfig::default();
            config.minimap = CaptureRegion {
                top: 0,
                left: 0,
                width: 255,
                height: 235,
            };
            config.view_size = 400;
            config.template.local_downscale = 4;
            config.template.global_downscale = 8;
            config.template.local_match_threshold = 0.45;
            config.template.global_match_threshold = 0.40;
            config.ai.weights_path = None;

            let local_scale = config.template.local_downscale.max(1);
            let global_scale = config.template.global_downscale.max(local_scale);
            let pyramid = MapPyramid {
                local: ScaledMap {
                    scale: local_scale,
                    image: build_match_representation(&downscale_gray(&map, local_scale)),
                },
                global: ScaledMap {
                    scale: global_scale,
                    image: build_match_representation(&downscale_gray(&map, global_scale)),
                },
            };
            let masks = build_template_masks(&config);

            TestFixture {
                config,
                map,
                pyramid,
                masks,
            }
        })
    }

    fn bwiki_tiles_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp-bwiki-rocom")
            .join("tiles-z8-test")
    }

    fn flat_points_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp-bwiki-rocom")
            .join("flat-points.json")
    }

    fn sample_positions(image: &GrayImage, view_size: u32) -> Vec<(u32, u32)> {
        let min_center = align_to(view_size / 2 + 32, 4);
        let max_x = align_to(image.width().saturating_sub(view_size / 2 + 32), 4);
        let max_y = align_to(image.height().saturating_sub(view_size / 2 + 32), 4);

        let raw = fs::read_to_string(flat_points_path()).expect("failed to read flat-points.json");
        let records: Vec<FlatPointRecord> =
            serde_json::from_str(&raw).expect("failed to parse flat-points.json");

        let mut points = Vec::new();
        for record in records {
            let world = raw_coordinate_to_world(record.point.lat, record.point.lng);
            let x = align_to(world.x.max(0.0) as u32, 4).clamp(min_center, max_x);
            let y = align_to(world.y.max(0.0) as u32, 4).clamp(min_center, max_y);
            if points
                .iter()
                .all(|(px, py)| x.abs_diff(*px) + y.abs_diff(*py) >= 1_500)
            {
                points.push((x, y));
            }
            if points.len() >= 6 {
                break;
            }
        }
        points
    }

    fn align_to(value: u32, step: u32) -> u32 {
        (value / step.max(1)) * step.max(1)
    }

    fn synthetic_capture(fixture: &TestFixture, center: (u32, u32)) -> GrayImage {
        let half = fixture.config.view_size / 2;
        let left = center.0.saturating_sub(half);
        let top = center.1.saturating_sub(half);
        let crop = crop_imm(
            &fixture.map,
            left,
            top,
            fixture.config.view_size,
            fixture.config.view_size,
        )
        .to_image();

        let diameter_px = fixture
            .config
            .minimap
            .width
            .min(fixture.config.minimap.height)
            .max(1);
        let diameter_px =
            ((diameter_px as f32) * fixture.config.template.mask_outer_radius).round() as u32;
        let minimap = resize(&crop, diameter_px, diameter_px, FilterType::Triangle);
        let mut canvas = GrayImage::from_pixel(
            fixture.config.minimap.width,
            fixture.config.minimap.height,
            Luma([0]),
        );
        let offset_x = i64::from((fixture.config.minimap.width - diameter_px) / 2);
        let offset_y = i64::from((fixture.config.minimap.height - diameter_px) / 2);
        replace(&mut canvas, &minimap, offset_x, offset_y);
        canvas
    }

    fn assert_world_close(actual: WorldPoint, expected: (u32, u32), tolerance: f32) {
        let dx = (actual.x - expected.0 as f32).abs();
        let dy = (actual.y - expected.1 as f32).abs();
        assert!(
            dx <= tolerance && dy <= tolerance,
            "expected ({}, {}), got ({:.1}, {:.1}), dx={dx:.1}, dy={dy:.1}",
            expected.0,
            expected.1,
            actual.x,
            actual.y
        );
    }

    fn matcher_for_device(
        fixture: &TestFixture,
        device: AiDevicePreference,
    ) -> CandleFeatureMatcher {
        let mut config = fixture.config.ai.clone();
        config.device = device;
        let encoder =
            FixedFeatureEncoder::built_in(&config).expect("failed to create built-in encoder");
        CandleFeatureMatcher::with_encoder(encoder, &fixture.pyramid, &fixture.masks)
            .expect("failed to create feature matcher")
    }

    fn locate_fixture_candle(
        fixture: &TestFixture,
        matcher: &CandleFeatureMatcher,
        capture: &GrayImage,
    ) -> Result<MatchCandidate> {
        let (local_template, global_template) =
            prepare_capture_templates(capture, &fixture.config, &fixture.pyramid);
        let coarse = matcher.locate_cached(
            &matcher.global_search,
            &global_template,
            &matcher.global_mask,
            fixture.config.template.global_match_threshold,
            0,
            0,
            fixture.pyramid.global.scale,
        )?;
        let coarse = coarse.accepted.unwrap_or_else(|| {
            panic!(
                "global locate should accept, best_score={:.3}, best_left={}, best_top={}",
                coarse.best_score, coarse.best_left, coarse.best_top
            )
        });
        let crop = crop_around_center(
            &fixture.pyramid.local.image,
            center_to_scaled(coarse.world, fixture.pyramid.local.scale),
            fixture.config.template.global_refine_radius_px / fixture.pyramid.local.scale.max(1),
            local_template.width(),
            local_template.height(),
        )?;
        let refine = matcher.locate_dynamic(
            &crop.image,
            &local_template,
            &matcher.local_mask,
            fixture.config.template.global_match_threshold,
            crop.origin_x,
            crop.origin_y,
            fixture.pyramid.local.scale,
        )?;

        Ok(refine
            .accepted
            .or(Some(coarse))
            .expect("refine locate should accept"))
    }

    #[test]
    fn prepared_template_uses_world_view_diameter() {
        let fixture = fixture();
        let capture = synthetic_capture(
            fixture,
            sample_positions(&fixture.map, fixture.config.view_size)[0],
        );
        let local = prepare_capture_template(
            &capture,
            fixture.config.view_size,
            fixture.pyramid.local.scale,
            fixture.config.template.mask_outer_radius,
        );
        let global = prepare_capture_template(
            &capture,
            fixture.config.view_size,
            fixture.pyramid.global.scale,
            fixture.config.template.mask_outer_radius,
        );

        assert_eq!(
            local.width(),
            scaled_dimension(fixture.config.view_size, fixture.pyramid.local.scale)
        );
        assert_eq!(
            global.width(),
            scaled_dimension(fixture.config.view_size, fixture.pyramid.global.scale)
        );
        assert_eq!(local.width(), fixture.masks.local.width());
        assert_eq!(global.width(), fixture.masks.global.width());
    }

    #[test]
    fn synthetic_captures_locate_many_positions_on_candle_cpu() -> Result<()> {
        let fixture = fixture();
        let matcher = matcher_for_device(fixture, AiDevicePreference::Cpu);
        for point in sample_positions(&fixture.map, fixture.config.view_size) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_candle(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 4.0);
        }
        Ok(())
    }

    #[test]
    fn local_track_sequence_stays_locked_on_candle_cpu() -> Result<()> {
        let fixture = fixture();
        let matcher = matcher_for_device(fixture, AiDevicePreference::Cpu);
        let path = [
            (
                align_to(fixture.map.width() / 3, 4),
                align_to(fixture.map.height() / 3, 4),
            ),
            (
                align_to(fixture.map.width() / 3 + 96, 4),
                align_to(fixture.map.height() / 3 + 48, 4),
            ),
            (
                align_to(fixture.map.width() / 3 + 176, 4),
                align_to(fixture.map.height() / 3 + 104, 4),
            ),
            (
                align_to(fixture.map.width() / 3 + 236, 4),
                align_to(fixture.map.height() / 3 + 144, 4),
            ),
        ];

        let first = locate_fixture_candle(fixture, &matcher, &synthetic_capture(fixture, path[0]))?;
        assert_world_close(first.world, path[0], 4.0);

        for window in path.windows(2) {
            let previous = WorldPoint::new(window[0].0 as f32, window[0].1 as f32);
            let capture = synthetic_capture(fixture, window[1]);
            let (local_template, _) =
                prepare_capture_templates(&capture, &fixture.config, &fixture.pyramid);
            let crop = crop_around_center(
                &fixture.pyramid.local.image,
                center_to_scaled(previous, fixture.pyramid.local.scale),
                fixture.config.local_search.radius_px / fixture.pyramid.local.scale.max(1),
                local_template.width(),
                local_template.height(),
            )?;
            let locate = matcher.locate_dynamic(
                &crop.image,
                &local_template,
                &matcher.local_mask,
                fixture.config.template.local_match_threshold,
                crop.origin_x,
                crop.origin_y,
                fixture.pyramid.local.scale,
            )?;
            let candidate = locate.accepted.expect("local track should accept");
            assert_world_close(candidate.world, window[1], 4.0);
        }

        Ok(())
    }

    #[cfg(candle_cuda_backend)]
    #[test]
    fn synthetic_captures_locate_many_positions_on_candle_cuda() -> Result<()> {
        let fixture = fixture();
        let matcher = match std::panic::catch_unwind(|| {
            matcher_for_device(fixture, AiDevicePreference::Cuda)
        }) {
            Ok(matcher) => matcher,
            Err(_) => return Ok(()),
        };

        for point in sample_positions(&fixture.map, fixture.config.view_size) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_candle(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 4.0);
        }
        Ok(())
    }
}
