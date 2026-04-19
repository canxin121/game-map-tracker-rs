use std::{fs, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Result;
#[cfg(feature = "ai-burn")]
use burn::{
    backend::ndarray::NdArrayDevice,
    tensor::{
        Tensor, TensorData, backend::Backend, cast::ToElement, module::conv2d, ops::ConvOptions,
    },
};
#[cfg(feature = "ai-burn")]
use image::{GrayImage, RgbaImage};
#[cfg(feature = "ai-burn")]
use safetensors::SafeTensors;
use tracing::{info, warn};

use crate::{
    config::TemplateTrackingConfig,
    domain::tracker::TrackerEngineKind,
    resources::WorkspaceSnapshot,
    tracking::runtime::{TrackingStatus, TrackingTick, TrackingWorker},
};
#[cfg(feature = "ai-burn")]
use crate::{
    config::{AiTrackingConfig, AppConfig},
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackingSource},
    },
    tracking::{
        burn_support::{
            BurnDeviceSelection, available_burn_backends, burn_device_label,
            burn_score_map_capture_enabled, select_burn_device,
        },
        capture::{CaptureSource, DesktopCapture, preprocess_capture},
        debug::{DebugField, TrackingDebugSnapshot},
        precompute::{
            PersistedTensorCache, clear_match_pyramid_caches, clear_tensor_caches_by_prefix,
            load_tensor_cache, metadata_fingerprint, save_tensor_cache, tracker_map_cache_key,
            tracker_tensor_cache_path,
        },
        presence::{MinimapPresenceDetector, MinimapPresenceSample},
        vision::{
            ColorCaptureTemplates, ColorMapPyramid, ColorTemplateShape, LocalCandidateDecision,
            MaskSet, MatchCandidate, SearchCrop, SearchStage, TrackerState, build_debug_snapshot,
            build_mask, capture_template_inner_square, center_to_scaled,
            coarse_global_downscale, crop_search_region_rgba, load_logic_color_map_pyramid,
            local_candidate_decision, mask_as_unit_vec, prepare_color_capture_template,
            preview_image, rgba_image_as_unit_vec, scaled_dimension, search_region_around_center,
            top_score_peaks,
        },
    },
};

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
struct LocateResult {
    best_left: u32,
    best_top: u32,
    best_score: f32,
    score_width: u32,
    score_height: u32,
    score_map: Option<Vec<f32>>,
    accepted: Option<MatchCandidate>,
}

#[cfg(feature = "ai-burn")]
const MAX_GLOBAL_COARSE_CANDIDATES: usize = 12;

pub struct BurnTrackerWorker {
    #[cfg(feature = "ai-burn")]
    inner: BurnTrackerInner,
}

#[cfg(feature = "ai-burn")]
struct BurnTrackerInner {
    config: AppConfig,
    capture: DesktopCapture,
    presence_detector: Option<MinimapPresenceDetector>,
    color_pyramid: ColorMapPyramid,
    state: TrackerState,
    debug_enabled: bool,
    matcher: BurnFeatureMatcher,
}

#[cfg(feature = "ai-burn")]
enum BurnFeatureMatcher {
    NdArray(BurnFeatureMatcherBackend<burn::backend::NdArray>),
    #[cfg(burn_cuda_backend)]
    Cuda(BurnFeatureMatcherBackend<burn::backend::Cuda>),
    #[cfg(burn_vulkan_backend)]
    Vulkan(BurnFeatureMatcherBackend<burn::backend::Vulkan>),
    #[cfg(burn_metal_backend)]
    Metal(BurnFeatureMatcherBackend<burn::backend::Metal>),
}

#[cfg(feature = "ai-burn")]
struct BurnFeatureMatcherBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    encoder: FixedFeatureEncoder<B>,
    coarse_search: SearchTensorCache<B>,
    global_mask: Tensor<B, 4>,
    coarse_mask: Tensor<B, 4>,
    coarse_search_patch_energy: Tensor<B, 4>,
    local_mask: Tensor<B, 4>,
    chunk_budget_bytes: Option<usize>,
}

#[cfg(feature = "ai-burn")]
struct FixedFeatureEncoder<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    device: B::Device,
    device_label: String,
    edge_kernels: Tensor<B, 4>,
    edge_bias: Option<Tensor<B, 1>>,
    source: EncoderSource,
    padding: usize,
    output_channels: usize,
}

#[cfg(feature = "ai-burn")]
struct SearchTensorCache<B>
where
    B: Backend<FloatElem = f32>,
{
    features: Tensor<B, 4>,
    squared: Tensor<B, 4>,
    width: u32,
    height: u32,
    channels: usize,
}

#[cfg(feature = "ai-burn")]
struct PreparedCaptureTemplates {
    local: PreparedTemplate,
    global: PreparedTemplate,
    coarse: PreparedTemplate,
}

#[cfg(feature = "ai-burn")]
enum PreparedTemplate {
    NdArray(BurnPreparedTemplate<burn::backend::NdArray>),
    #[cfg(burn_cuda_backend)]
    Cuda(BurnPreparedTemplate<burn::backend::Cuda>),
    #[cfg(burn_vulkan_backend)]
    Vulkan(BurnPreparedTemplate<burn::backend::Vulkan>),
    #[cfg(burn_metal_backend)]
    Metal(BurnPreparedTemplate<burn::backend::Metal>),
}

#[cfg(feature = "ai-burn")]
struct BurnPreparedTemplate<B>
where
    B: Backend<FloatElem = f32>,
{
    image: RgbaImage,
    mask: Tensor<B, 4>,
    weighted_template: Tensor<B, 4>,
    template_energy: f32,
}

#[cfg(feature = "ai-burn")]
impl<B> BurnPreparedTemplate<B>
where
    B: Backend<FloatElem = f32>,
{
    fn width(&self) -> u32 {
        self.image.width()
    }

    fn height(&self) -> u32 {
        self.image.height()
    }
}

#[derive(Debug, Clone)]
enum EncoderSource {
    Safetensors(PathBuf),
    BuiltIn,
}

impl EncoderSource {
    fn label(&self) -> String {
        match self {
            Self::Safetensors(path) => format!("Safetensors ({})", path.display()),
            Self::BuiltIn => "Built-in Conv Edge Bank".to_owned(),
        }
    }
}

#[cfg(feature = "ai-burn")]
const CUDA_CONV_IM2COL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

#[cfg(feature = "ai-burn")]
const WGPU_CONV_IM2COL_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[cfg(feature = "ai-burn")]
const FEATURE_ENCODER_CACHE_VERSION: u32 = 2;

impl BurnTrackerWorker {
    pub fn new(workspace: Arc<WorkspaceSnapshot>) -> Result<Self> {
        info!(
            cache_root = %workspace.assets.bwiki_cache_dir.display(),
            view_size = workspace.config.view_size,
            device = %workspace.config.ai.device,
            device_index = workspace.config.ai.device_index,
            "initializing convolution tracker worker"
        );
        #[cfg(feature = "ai-burn")]
        {
            let config = workspace.config.clone();
            let capture = DesktopCapture::from_absolute_region(&config.minimap)?;
            let presence_detector = MinimapPresenceDetector::new(workspace.as_ref())?;
            let map_cache_key = tracker_map_cache_key(workspace.as_ref())?;
            let masks = build_template_masks(&config);
            let color_pyramid = load_logic_color_map_pyramid(workspace.as_ref())?;
            let matcher = BurnFeatureMatcher::new(
                workspace.as_ref(),
                &config.ai,
                &color_pyramid,
                &masks,
                &map_cache_key,
            )?;

            Ok(Self {
                inner: BurnTrackerInner {
                    config,
                    capture,
                    presence_detector,
                    color_pyramid,
                    state: TrackerState::default(),
                    debug_enabled: false,
                    matcher,
                },
            })
        }

        #[cfg(not(feature = "ai-burn"))]
        {
            let _ = workspace;
            Ok(Self {})
        }
    }
}

pub fn rebuild_convolution_engine_cache(workspace: &WorkspaceSnapshot) -> Result<()> {
    info!(
        cache_root = %workspace.assets.bwiki_cache_dir.display(),
        device = %workspace.config.ai.device,
        device_index = workspace.config.ai.device_index,
        "rebuilding convolution tracker cache"
    );
    #[cfg(feature = "ai-burn")]
    {
        clear_match_pyramid_caches(workspace)?;
        clear_tensor_caches_by_prefix(workspace, "feature-local-search")?;
        clear_tensor_caches_by_prefix(workspace, "feature-global-search")?;
        clear_tensor_caches_by_prefix(workspace, "feature-coarse-search")?;

        let map_cache_key = tracker_map_cache_key(workspace)?;
        let color_pyramid = load_logic_color_map_pyramid(workspace)?;
        let masks = build_template_masks(&workspace.config);
        let _ = BurnFeatureMatcher::new(
            workspace,
            &workspace.config.ai,
            &color_pyramid,
            &masks,
            &map_cache_key,
        )?;
        info!("rebuild of convolution tracker cache completed");
        return Ok(());
    }

    #[cfg(not(feature = "ai-burn"))]
    {
        let _ = workspace;
        anyhow::bail!("卷积特征匹配后端被选中，但当前二进制未启用 `ai-burn` 特性")
    }
}

#[cfg(feature = "ai-burn")]
impl<B> FixedFeatureEncoder<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn new(workspace: &WorkspaceSnapshot, device: B::Device, device_label: String) -> Result<Self> {
        if let Some(encoder) =
            Self::load_from_safetensors(workspace, device.clone(), device_label.clone())?
        {
            return Ok(encoder);
        }

        Self::built_in_with_device(device, device_label)
    }

    fn built_in_with_device(device: B::Device, device_label: String) -> Result<Self> {
        let mono_kernels = vec![
            -1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -1.0, -2.0, -1.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 2.0, -1.0,
            2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, 1.0,
            2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0,
        ];
        let mut kernels = Vec::with_capacity(18 * 3 * 3 * 3);
        for channel in 0..3 {
            for kernel in mono_kernels.chunks_exact(9) {
                for input_channel in 0..3 {
                    if input_channel == channel {
                        kernels.extend_from_slice(kernel);
                    } else {
                        kernels.extend(std::iter::repeat(0.0f32).take(9));
                    }
                }
            }
        }
        let edge_kernels =
            Tensor::<B, 4>::from_data(TensorData::new(kernels, [18, 3, 3, 3]), &device);

        Ok(Self {
            device,
            device_label,
            edge_kernels,
            edge_bias: None,
            source: EncoderSource::BuiltIn,
            padding: 1,
            output_channels: 21,
        })
    }

    fn load_from_safetensors(
        workspace: &WorkspaceSnapshot,
        device: B::Device,
        device_label: String,
    ) -> Result<Option<Self>> {
        for candidate in encoder_weight_candidates(workspace) {
            if !candidate.exists() {
                continue;
            }

            if let Ok(encoder) =
                Self::load_single_safetensors(&candidate, device.clone(), device_label.clone())
            {
                return Ok(Some(encoder));
            }
        }

        Ok(None)
    }

    fn load_single_safetensors(
        path: &PathBuf,
        device: B::Device,
        device_label: String,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let weight_name = find_safetensor_name(
            &tensors,
            &[
                "edge_bank.weight",
                "encoder.edge_bank.weight",
                "conv.weight",
                "weight",
            ],
        )?;
        let weight_view = tensors.tensor(weight_name)?;
        let shape = weight_view.shape();
        if shape.len() != 4 || shape[1] != 3 || shape[2] != shape[3] {
            anyhow::bail!(
                "unsupported encoder kernel shape {:?} in {}",
                shape,
                path.display()
            );
        }

        let out_channels = shape[0];
        let kernel_h = shape[2];
        let weight = Tensor::<B, 4>::from_data(
            TensorData::new(
                safetensor_view_f32_vec(&weight_view)?,
                [shape[0], shape[1], shape[2], shape[3]],
            ),
            &device,
        );

        let bias = [
            "edge_bank.bias",
            "encoder.edge_bank.bias",
            "conv.bias",
            "bias",
        ]
        .iter()
        .copied()
        .find(|name| tensors.tensor(name).is_ok())
        .map(|name| {
            let view = tensors.tensor(name)?;
            if view.shape().len() != 1 || view.shape()[0] != out_channels {
                anyhow::bail!(
                    "unsupported encoder bias shape {:?} in {}",
                    view.shape(),
                    path.display()
                );
            }
            Ok(Tensor::<B, 1>::from_data(
                TensorData::new(safetensor_view_f32_vec(&view)?, [out_channels]),
                &device,
            ))
        })
        .transpose()?;

        Ok(Self {
            device,
            device_label,
            edge_kernels: weight,
            edge_bias: bias,
            source: EncoderSource::Safetensors(path.clone()),
            padding: kernel_h / 2,
            output_channels: out_channels + 3,
        })
    }

    fn source_label(&self) -> String {
        self.source.label()
    }

    fn device_label(&self) -> String {
        self.device_label.clone()
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

    fn encode(&self, image: &RgbaImage) -> Result<Tensor<B, 4>> {
        let base = Tensor::<B, 4>::from_data(
            TensorData::new(
                rgba_image_as_unit_vec(image),
                [1, 3, image.height() as usize, image.width() as usize],
            ),
            &self.device,
        );
        let edges = conv2d(
            base.clone(),
            self.edge_kernels.clone(),
            self.edge_bias.clone(),
            ConvOptions::new([1, 1], [self.padding, self.padding], [1, 1], 1),
        );
        Ok(Tensor::<B, 4>::cat(vec![base, edges], 1))
    }
}

#[cfg(feature = "ai-burn")]
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

#[cfg(feature = "ai-burn")]
fn find_safetensor_name<'a>(tensors: &SafeTensors<'a>, names: &[&'a str]) -> Result<&'a str> {
    names
        .iter()
        .copied()
        .find(|name| tensors.tensor(name).is_ok())
        .ok_or_else(|| anyhow::anyhow!("failed to resolve encoder tensor names: {:?}", names))
}

#[cfg(feature = "ai-burn")]
fn safetensor_view_f32_vec(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != safetensors::Dtype::F32 {
        anyhow::bail!(
            "unsupported safetensors dtype {:?}, only f32 is supported",
            view.dtype()
        );
    }

    let data = view.data();
    if data.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!("invalid safetensors payload length {}", data.len());
    }

    Ok(data
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(feature = "ai-burn")]
fn prepare_color_capture_templates(
    captured: &RgbaImage,
    config: &AppConfig,
    pyramid: &ColorMapPyramid,
) -> ColorCaptureTemplates {
    ColorCaptureTemplates {
        local: prepare_color_capture_template(
            captured,
            config.view_size,
            pyramid.local.scale,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
            ColorTemplateShape::Annulus,
        ),
        global: prepare_color_capture_template(
            captured,
            config.view_size,
            pyramid.global.scale,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
            ColorTemplateShape::Annulus,
        ),
        coarse: prepare_color_capture_template(
            captured,
            config.view_size,
            pyramid.coarse.scale,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
            ColorTemplateShape::Annulus,
        ),
    }
}

#[cfg(feature = "ai-burn")]
fn build_template_masks(config: &AppConfig) -> MaskSet {
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(config);
    let local_size = scaled_dimension(config.view_size.max(1), local_scale);
    let global_size = scaled_dimension(config.view_size.max(1), global_scale);
    let coarse_size = scaled_dimension(config.view_size.max(1), coarse_scale);

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
        coarse: build_mask(
            coarse_size,
            coarse_size,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
        ),
    }
}

#[cfg(feature = "ai-burn")]
impl BurnFeatureMatcher {
    fn new(
        workspace: &WorkspaceSnapshot,
        config: &AiTrackingConfig,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let selection = select_burn_device(config)?;
        Self::from_selection(workspace, selection, pyramid, masks, map_cache_key)
    }

    fn from_selection(
        workspace: &WorkspaceSnapshot,
        selection: BurnDeviceSelection,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        match selection {
            BurnDeviceSelection::Cpu => Ok(Self::NdArray(BurnFeatureMatcherBackend::<
                burn::backend::NdArray,
            >::new(
                workspace,
                NdArrayDevice::Cpu,
                "CPU".to_owned(),
                None,
                pyramid,
                masks,
                map_cache_key,
            )?)),
            #[cfg(burn_cuda_backend)]
            BurnDeviceSelection::Cuda(device) => Ok(Self::Cuda(BurnFeatureMatcherBackend::<
                burn::backend::Cuda,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Cuda(device)),
                Some(CUDA_CONV_IM2COL_BUDGET_BYTES),
                pyramid,
                masks,
                map_cache_key,
            )?)),
            #[cfg(burn_vulkan_backend)]
            BurnDeviceSelection::Vulkan(device) => Ok(Self::Vulkan(BurnFeatureMatcherBackend::<
                burn::backend::Vulkan,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Vulkan(device)),
                Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                pyramid,
                masks,
                map_cache_key,
            )?)),
            #[cfg(burn_metal_backend)]
            BurnDeviceSelection::Metal(device) => Ok(Self::Metal(BurnFeatureMatcherBackend::<
                burn::backend::Metal,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Metal(device)),
                Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                pyramid,
                masks,
                map_cache_key,
            )?)),
        }
    }

    fn device_label(&self) -> String {
        match self {
            Self::NdArray(matcher) => matcher.device_label(),
            #[cfg(burn_cuda_backend)]
            Self::Cuda(matcher) => matcher.device_label(),
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(matcher) => matcher.device_label(),
            #[cfg(burn_metal_backend)]
            Self::Metal(matcher) => matcher.device_label(),
        }
    }

    fn source_label(&self) -> String {
        match self {
            Self::NdArray(matcher) => matcher.source_label(),
            #[cfg(burn_cuda_backend)]
            Self::Cuda(matcher) => matcher.source_label(),
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(matcher) => matcher.source_label(),
            #[cfg(burn_metal_backend)]
            Self::Metal(matcher) => matcher.source_label(),
        }
    }

    fn prepare_capture_templates(
        &self,
        captured: &RgbaImage,
        config: &AppConfig,
        pyramid: &ColorMapPyramid,
    ) -> Result<PreparedCaptureTemplates> {
        let ColorCaptureTemplates {
            local,
            global,
            coarse,
        } = prepare_color_capture_templates(captured, config, pyramid);
        Ok(PreparedCaptureTemplates {
            local: match self {
                Self::NdArray(matcher) => {
                    PreparedTemplate::NdArray(matcher.prepare_template(local, &matcher.local_mask)?)
                }
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => {
                    PreparedTemplate::Cuda(matcher.prepare_template(local, &matcher.local_mask)?)
                }
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => {
                    PreparedTemplate::Vulkan(matcher.prepare_template(local, &matcher.local_mask)?)
                }
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => {
                    PreparedTemplate::Metal(matcher.prepare_template(local, &matcher.local_mask)?)
                }
            },
            global: match self {
                Self::NdArray(matcher) => PreparedTemplate::NdArray(
                    matcher.prepare_template(global, &matcher.global_mask)?,
                ),
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => {
                    PreparedTemplate::Cuda(matcher.prepare_template(global, &matcher.global_mask)?)
                }
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => PreparedTemplate::Vulkan(
                    matcher.prepare_template(global, &matcher.global_mask)?,
                ),
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => {
                    PreparedTemplate::Metal(matcher.prepare_template(global, &matcher.global_mask)?)
                }
            },
            coarse: match self {
                Self::NdArray(matcher) => PreparedTemplate::NdArray(
                    matcher.prepare_template(coarse, &matcher.coarse_mask)?,
                ),
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => {
                    PreparedTemplate::Cuda(matcher.prepare_template(coarse, &matcher.coarse_mask)?)
                }
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => PreparedTemplate::Vulkan(
                    matcher.prepare_template(coarse, &matcher.coarse_mask)?,
                ),
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => {
                    PreparedTemplate::Metal(matcher.prepare_template(coarse, &matcher.coarse_mask)?)
                }
            },
        })
    }

    #[allow(dead_code)]
    fn locate_coarse(
        &self,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match self {
            Self::NdArray(matcher) => {
                matcher.locate_coarse(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_cuda_backend)]
            Self::Cuda(matcher) => {
                matcher.locate_coarse(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(matcher) => {
                matcher.locate_coarse(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_metal_backend)]
            Self::Metal(matcher) => {
                matcher.locate_coarse(template, threshold, origin_x, origin_y, scale)
            }
        }
    }

    fn locate_coarse_prepared(
        &self,
        template: &PreparedTemplate,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match (self, template) {
            (Self::NdArray(matcher), PreparedTemplate::NdArray(template)) => {
                matcher.locate_coarse_prepared(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_cuda_backend)]
            (Self::Cuda(matcher), PreparedTemplate::Cuda(template)) => {
                matcher.locate_coarse_prepared(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_vulkan_backend)]
            (Self::Vulkan(matcher), PreparedTemplate::Vulkan(template)) => {
                matcher.locate_coarse_prepared(template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_metal_backend)]
            (Self::Metal(matcher), PreparedTemplate::Metal(template)) => {
                matcher.locate_coarse_prepared(template, threshold, origin_x, origin_y, scale)
            }
            _ => anyhow::bail!("prepared template backend does not match matcher backend"),
        }
    }

    #[allow(dead_code)]
    fn locate_global_crop(
        &self,
        image: &RgbaImage,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match self {
            Self::NdArray(matcher) => {
                matcher.locate_global_crop(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_cuda_backend)]
            Self::Cuda(matcher) => {
                matcher.locate_global_crop(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(matcher) => {
                matcher.locate_global_crop(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_metal_backend)]
            Self::Metal(matcher) => {
                matcher.locate_global_crop(image, template, threshold, origin_x, origin_y, scale)
            }
        }
    }

    fn locate_global_crop_prepared(
        &self,
        image: &RgbaImage,
        template: &PreparedTemplate,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match (self, template) {
            (Self::NdArray(matcher), PreparedTemplate::NdArray(template)) => matcher
                .locate_global_crop_prepared(image, template, threshold, origin_x, origin_y, scale),
            #[cfg(burn_cuda_backend)]
            (Self::Cuda(matcher), PreparedTemplate::Cuda(template)) => matcher
                .locate_global_crop_prepared(image, template, threshold, origin_x, origin_y, scale),
            #[cfg(burn_vulkan_backend)]
            (Self::Vulkan(matcher), PreparedTemplate::Vulkan(template)) => matcher
                .locate_global_crop_prepared(image, template, threshold, origin_x, origin_y, scale),
            #[cfg(burn_metal_backend)]
            (Self::Metal(matcher), PreparedTemplate::Metal(template)) => matcher
                .locate_global_crop_prepared(image, template, threshold, origin_x, origin_y, scale),
            _ => anyhow::bail!("prepared template backend does not match matcher backend"),
        }
    }

    #[allow(dead_code)]
    fn locate_dynamic(
        &self,
        image: &RgbaImage,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match self {
            Self::NdArray(matcher) => {
                matcher.locate_local(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_cuda_backend)]
            Self::Cuda(matcher) => {
                matcher.locate_local(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(matcher) => {
                matcher.locate_local(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_metal_backend)]
            Self::Metal(matcher) => {
                matcher.locate_local(image, template, threshold, origin_x, origin_y, scale)
            }
        }
    }

    fn locate_dynamic_prepared(
        &self,
        image: &RgbaImage,
        template: &PreparedTemplate,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        match (self, template) {
            (Self::NdArray(matcher), PreparedTemplate::NdArray(template)) => {
                matcher.locate_local_prepared(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_cuda_backend)]
            (Self::Cuda(matcher), PreparedTemplate::Cuda(template)) => {
                matcher.locate_local_prepared(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_vulkan_backend)]
            (Self::Vulkan(matcher), PreparedTemplate::Vulkan(template)) => {
                matcher.locate_local_prepared(image, template, threshold, origin_x, origin_y, scale)
            }
            #[cfg(burn_metal_backend)]
            (Self::Metal(matcher), PreparedTemplate::Metal(template)) => {
                matcher.locate_local_prepared(image, template, threshold, origin_x, origin_y, scale)
            }
            _ => anyhow::bail!("prepared template backend does not match matcher backend"),
        }
    }
}

#[cfg(feature = "ai-burn")]
impl PreparedTemplate {
    fn image(&self) -> &RgbaImage {
        match self {
            Self::NdArray(template) => &template.image,
            #[cfg(burn_cuda_backend)]
            Self::Cuda(template) => &template.image,
            #[cfg(burn_vulkan_backend)]
            Self::Vulkan(template) => &template.image,
            #[cfg(burn_metal_backend)]
            Self::Metal(template) => &template.image,
        }
    }

    fn width(&self) -> u32 {
        self.image().width()
    }

    fn height(&self) -> u32 {
        self.image().height()
    }
}

#[cfg(feature = "ai-burn")]
impl<B> BurnFeatureMatcherBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn new(
        workspace: &WorkspaceSnapshot,
        device: B::Device,
        device_label: String,
        chunk_budget_bytes: Option<usize>,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let encoder = FixedFeatureEncoder::<B>::new(workspace, device, device_label)?;
        let coarse_search = load_or_build_feature_search::<B>(
            workspace,
            "feature-coarse-search",
            map_cache_key,
            &pyramid.coarse.image,
            &encoder,
        )?;
        let global_mask =
            mask_tensor::<B>(&masks.global, encoder.output_channels(), &encoder.device);
        let coarse_mask =
            mask_tensor::<B>(&masks.coarse, encoder.output_channels(), &encoder.device);
        let local_mask = mask_tensor::<B>(&masks.local, encoder.output_channels(), &encoder.device);
        let coarse_search_patch_energy = conv2d(
            coarse_search.squared.clone(),
            coarse_mask.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );

        Ok(Self {
            encoder,
            coarse_search,
            global_mask,
            coarse_mask,
            coarse_search_patch_energy,
            local_mask,
            chunk_budget_bytes,
        })
    }

    fn device_label(&self) -> String {
        self.encoder.device_label()
    }

    fn source_label(&self) -> String {
        self.encoder.source_label()
    }

    fn prepare_template(
        &self,
        image: RgbaImage,
        mask: &Tensor<B, 4>,
    ) -> Result<BurnPreparedTemplate<B>> {
        let template_features = self.encoder.encode(&image)?;
        let weighted_template = template_features * mask.clone();
        let template_energy = weighted_template.clone().powi_scalar(2).sum().into_scalar();
        Ok(BurnPreparedTemplate {
            image,
            mask: mask.clone(),
            weighted_template,
            template_energy,
        })
    }

    #[allow(dead_code)]
    fn locate_coarse(
        &self,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let prepared = self.prepare_template(template.clone(), &self.coarse_mask)?;
        self.locate_cached(
            &self.coarse_search,
            &prepared,
            Some(self.coarse_search_patch_energy.clone()),
            true,
            threshold,
            origin_x,
            origin_y,
            scale,
        )
    }

    fn locate_coarse_prepared(
        &self,
        template: &BurnPreparedTemplate<B>,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        self.locate_cached(
            &self.coarse_search,
            template,
            Some(self.coarse_search_patch_energy.clone()),
            true,
            threshold,
            origin_x,
            origin_y,
            scale,
        )
    }

    #[allow(dead_code)]
    fn locate_global_crop(
        &self,
        image: &RgbaImage,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let prepared = self.prepare_template(template.clone(), &self.global_mask)?;
        self.locate_global_crop_prepared(image, &prepared, threshold, origin_x, origin_y, scale)
    }

    fn locate_global_crop_prepared(
        &self,
        image: &RgbaImage,
        template: &BurnPreparedTemplate<B>,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::<B>::from_rgba_image(image, &self.encoder)?;
        self.locate_cached(
            &search, template, None, false, threshold, origin_x, origin_y, scale,
        )
    }

    #[allow(dead_code)]
    fn locate_local(
        &self,
        image: &RgbaImage,
        template: &RgbaImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let prepared = self.prepare_template(template.clone(), &self.local_mask)?;
        self.locate_local_prepared(image, &prepared, threshold, origin_x, origin_y, scale)
    }

    fn locate_local_prepared(
        &self,
        image: &RgbaImage,
        template: &BurnPreparedTemplate<B>,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::<B>::from_rgba_image(image, &self.encoder)?;
        self.locate_cached(
            &search, template, None, false, threshold, origin_x, origin_y, scale,
        )
    }

    fn locate_cached(
        &self,
        search: &SearchTensorCache<B>,
        template: &BurnPreparedTemplate<B>,
        precomputed_patch_energy: Option<Tensor<B, 4>>,
        capture_score_map: bool,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        if search.width <= template.width() || search.height <= template.height() {
            return Ok(empty_locate_result());
        }

        let score_width = search.width - template.width() + 1;
        let score_height = search.height - template.height() + 1;
        let chunk_rows = burn_match_chunk_rows(
            self.chunk_budget_bytes,
            search.width,
            search.height,
            template.width(),
            template.height(),
            search.channels,
        );

        if chunk_rows < score_height {
            return locate_cached_in_chunks(
                search,
                &template.weighted_template,
                &template.mask,
                precomputed_patch_energy,
                template.template_energy,
                capture_score_map,
                threshold,
                origin_x,
                origin_y,
                scale,
                template.width(),
                template.height(),
                chunk_rows,
            );
        }

        let numerator = conv2d(
            search.features.clone(),
            template.weighted_template.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
        let search_patch_energy = match precomputed_patch_energy {
            Some(patch_energy) => patch_energy,
            None => conv2d(
                search.squared.clone(),
                template.mask.clone(),
                None::<Tensor<B, 1>>,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            ),
        };
        let normalized = numerator / (search_patch_energy * template.template_energy + 1e-6).sqrt();

        if capture_score_map || burn_score_map_capture_enabled() {
            let score_map = tensor4_to_flat_f32(normalized)?;
            return Ok(locate_result_from_flat_scores(
                score_map,
                score_width,
                score_height,
                threshold,
                origin_x,
                origin_y,
                scale,
                template.width(),
                template.height(),
            ));
        }

        let (best_score, best_left, best_top) =
            tensor_best_match(normalized, score_width, score_height)?;
        Ok(locate_result_from_best(
            best_score,
            best_left,
            best_top,
            score_width,
            score_height,
            threshold,
            origin_x,
            origin_y,
            scale,
            template.width(),
            template.height(),
            None,
        ))
    }
}

#[cfg(feature = "ai-burn")]
impl<B> SearchTensorCache<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    fn from_rgba_image(image: &RgbaImage, encoder: &FixedFeatureEncoder<B>) -> Result<Self> {
        let features = encoder.encode(image)?;
        let dims: [usize; 4] = features.shape().dims();
        let squared = features.clone().powi_scalar(2);
        Ok(Self {
            features,
            squared,
            width: dims[3] as u32,
            height: dims[2] as u32,
            channels: dims[1],
        })
    }

    fn from_persisted(
        cache: PersistedTensorCache,
        encoder: &FixedFeatureEncoder<B>,
    ) -> Result<Self> {
        if cache.channels != encoder.output_channels() {
            anyhow::bail!(
                "feature search tensor cache channel count {} does not match encoder output {}",
                cache.channels,
                encoder.output_channels()
            );
        }

        let features = Tensor::<B, 4>::from_data(
            TensorData::new(
                cache.primary,
                [
                    1,
                    cache.channels,
                    cache.height as usize,
                    cache.width as usize,
                ],
            ),
            &encoder.device,
        );
        let squared = Tensor::<B, 4>::from_data(
            TensorData::new(
                cache.secondary,
                [
                    1,
                    cache.channels,
                    cache.height as usize,
                    cache.width as usize,
                ],
            ),
            &encoder.device,
        );
        Ok(Self {
            features,
            squared,
            width: cache.width,
            height: cache.height,
            channels: cache.channels,
        })
    }

    fn to_persisted(&self) -> Result<PersistedTensorCache> {
        let features = tensor4_to_flat_f32(self.features.clone())?;
        let squared = tensor4_to_flat_f32(self.squared.clone())?;
        PersistedTensorCache::from_parts(self.width, self.height, self.channels, features, squared)
    }
}

#[cfg(feature = "ai-burn")]
fn load_or_build_feature_search<B>(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    map_cache_key: &str,
    image: &RgbaImage,
    encoder: &FixedFeatureEncoder<B>,
) -> Result<SearchTensorCache<B>>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    let cache_key = format!("{map_cache_key}-{}", encoder.cache_key()?);
    let cache_path = tracker_tensor_cache_path(workspace, prefix, &cache_key);
    if let Ok(Some(cache)) = load_tensor_cache(&cache_path) {
        if let Ok(search) = SearchTensorCache::<B>::from_persisted(cache, encoder) {
            return Ok(search);
        }
    }

    let search = SearchTensorCache::<B>::from_rgba_image(image, encoder)?;
    if let Ok(persisted) = search.to_persisted() {
        let _ = save_tensor_cache(&cache_path, &persisted);
    }
    Ok(search)
}

#[cfg(feature = "ai-burn")]
fn mask_tensor<B>(mask: &GrayImage, channels: usize, device: &B::Device) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    Tensor::<B, 4>::from_data(
        TensorData::new(
            mask_as_unit_vec(mask, channels),
            [1, channels, mask.height() as usize, mask.width() as usize],
        ),
        device,
    )
}

#[cfg(feature = "ai-burn")]
fn burn_match_chunk_rows(
    chunk_budget_bytes: Option<usize>,
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

    let Some(budget_bytes) = chunk_budget_bytes else {
        return output_height;
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

#[cfg(feature = "ai-burn")]
fn locate_cached_in_chunks<B>(
    search: &SearchTensorCache<B>,
    weighted_template: &Tensor<B, 4>,
    mask: &Tensor<B, 4>,
    precomputed_patch_energy: Option<Tensor<B, 4>>,
    template_energy: f32,
    capture_score_map: bool,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
    chunk_rows: u32,
) -> Result<LocateResult>
where
    B: Backend<FloatElem = f32>,
{
    let score_width = search.width - template_width + 1;
    let score_height = search.height - template_height + 1;
    let capture_score_map = capture_score_map || burn_score_map_capture_enabled();
    let mut score_map =
        capture_score_map.then(|| Vec::with_capacity(score_width as usize * score_height as usize));
    let mut output_row = 0u32;
    let mut best_score = f32::MIN;
    let mut best_left = 0u32;
    let mut best_top = 0u32;

    while output_row < score_height {
        let rows = chunk_rows.min(score_height - output_row).max(1);
        let slice_height = rows + template_height - 1;
        let features_chunk =
            search
                .features
                .clone()
                .narrow(2, output_row as usize, slice_height as usize);
        let squared_chunk =
            search
                .squared
                .clone()
                .narrow(2, output_row as usize, slice_height as usize);
        let numerator = conv2d(
            features_chunk,
            weighted_template.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
        let search_patch_energy = match precomputed_patch_energy.as_ref() {
            Some(cached) => (*cached)
                .clone()
                .narrow(2, output_row as usize, rows as usize),
            None => conv2d(
                squared_chunk,
                mask.clone(),
                None::<Tensor<B, 1>>,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            ),
        };
        let normalized = numerator / (search_patch_energy * template_energy + 1e-6).sqrt();
        let chunk_scores = tensor4_to_flat_f32(normalized)?;
        if let Some((chunk_best_score, chunk_best_left, chunk_best_top)) =
            best_match_in_flat_scores(&chunk_scores, score_width)
        {
            if chunk_best_score > best_score {
                best_score = chunk_best_score;
                best_left = chunk_best_left;
                best_top = output_row + chunk_best_top;
            }
        }
        let produced_rows = (chunk_scores.len() / score_width.max(1) as usize) as u32;
        if let Some(score_map) = score_map.as_mut() {
            score_map.extend(chunk_scores);
        }
        output_row += produced_rows.max(1);
    }

    Ok(locate_result_from_best(
        best_score,
        best_left,
        best_top,
        score_width,
        score_height,
        threshold,
        origin_x,
        origin_y,
        scale,
        template_width,
        template_height,
        score_map,
    ))
}

#[cfg(feature = "ai-burn")]
fn tensor4_to_flat_f32<B>(tensor: Tensor<B, 4>) -> Result<Vec<f32>>
where
    B: Backend<FloatElem = f32>,
{
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(feature = "ai-burn")]
fn tensor_best_match<B>(
    tensor: Tensor<B, 4>,
    score_width: u32,
    score_height: u32,
) -> Result<(f32, u32, u32)>
where
    B: Backend<FloatElem = f32>,
{
    let flat_len = score_width.max(1) as usize * score_height.max(1) as usize;
    let flat: Tensor<B, 1> = tensor.reshape([flat_len]);
    let (best_score, best_index) = flat.max_dim_with_indices(0);
    let best_score = best_score.into_scalar();
    let best_index = best_index.into_scalar().to_usize() as u32;
    Ok((
        best_score,
        best_index % score_width.max(1),
        best_index / score_width.max(1),
    ))
}

fn best_match_in_flat_scores(score_map: &[f32], score_width: u32) -> Option<(f32, u32, u32)> {
    if score_map.is_empty() {
        return None;
    }

    let mut best_index = 0usize;
    let mut best_score = f32::MIN;
    for (index, score) in score_map.iter().copied().enumerate() {
        if score > best_score {
            best_score = score;
            best_index = index;
        }
    }

    let score_width = score_width.max(1);
    Some((
        best_score,
        (best_index as u32) % score_width,
        (best_index as u32) / score_width,
    ))
}

fn empty_locate_result() -> LocateResult {
    LocateResult {
        best_left: 0,
        best_top: 0,
        best_score: f32::MIN,
        score_width: 0,
        score_height: 0,
        score_map: None,
        accepted: None,
    }
}

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
    let (best_score, best_left, best_top) =
        best_match_in_flat_scores(&score_map, score_width).unwrap_or((f32::MIN, 0, 0));

    locate_result_from_best(
        best_score,
        best_left,
        best_top,
        score_width,
        score_height,
        threshold,
        origin_x,
        origin_y,
        scale,
        template_width,
        template_height,
        Some(score_map),
    )
}

fn locate_result_from_best(
    best_score: f32,
    best_left: u32,
    best_top: u32,
    score_width: u32,
    score_height: u32,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
    score_map: Option<Vec<f32>>,
) -> LocateResult {
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

#[cfg(feature = "ai-burn")]
impl BurnTrackerInner {
    fn run_frame(&mut self) -> Result<TrackingTick> {
        self.state.begin_frame();
        if self.presence_detector.is_none() {
            let mut status = self.base_status();
            status.probe_summary = self.probe_summary(None);
            let estimate = self.apply_probe_unavailable_fallback(&mut status);
            status.locate_summary =
                Self::blocked_locate_summary("小地图圆环未启用", estimate.as_ref());
            return Ok(TrackingTick {
                status,
                estimate,
                debug: None,
            });
        }

        let probe_sample = self
            .presence_detector
            .as_ref()
            .map(MinimapPresenceDetector::sample)
            .transpose()?;
        if let Some(sample) = probe_sample.as_ref().filter(|sample| !sample.present) {
            let mut status = self.base_status();
            status.probe_summary = self.probe_summary(probe_sample.as_ref());
            let estimate = self.apply_probe_absent_fallback(&mut status);
            status.locate_summary =
                Self::blocked_locate_summary("小地图圆环未命中", estimate.as_ref());
            let debug_captured = if self.debug_enabled {
                match self.capture.capture_gray() {
                    Ok(captured) => Some(captured),
                    Err(error) => {
                        warn!(
                            error = %format!("{error:#}"),
                            "failed to capture minimap input for probe-miss debug snapshot"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let debug = self.debug_enabled.then(|| {
                self.build_probe_miss_debug_snapshot(
                    debug_captured.as_ref(),
                    sample,
                    estimate.as_ref(),
                )
            });
            return Ok(TrackingTick {
                status,
                estimate,
                debug,
            });
        }

        let captured_rgba = self.capture.capture_rgba()?;
        let captured = preprocess_capture(captured_rgba.clone());
        let templates = self.matcher.prepare_capture_templates(
            &captured_rgba,
            &self.config,
            &self.color_pyramid,
        )?;

        let mut status = self.base_status();
        status.probe_summary = self.probe_summary(probe_sample.as_ref());
        let mut estimate = None;
        let mut global_result = None;
        let mut refine_result = None;
        let mut local_result = None;
        let mut forced_global_jump: Option<f32> = None;
        let mut locate_summary = "等待定位".to_owned();

        if self.config.local_search.enabled && matches!(self.state.stage, SearchStage::LocalTrack) {
            if let Some(last_world) = self.state.last_world {
                let region = search_region_around_center(
                    self.color_pyramid.local.image.width(),
                    self.color_pyramid.local.image.height(),
                    center_to_scaled(last_world, self.color_pyramid.local.scale),
                    self.config.local_search.radius_px / self.color_pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region_rgba(&self.color_pyramid.local.image, region)?;
                let result = self.matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    self.config.template.local_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.color_pyramid.local.scale,
                )?;
                local_result = Some(result.clone());

                if let (Some(candidate), Some(last_world)) =
                    (result.accepted.clone(), self.state.last_world)
                {
                    match local_candidate_decision(
                        last_world,
                        candidate.world,
                        self.config.local_search.max_accepted_jump_px,
                        self.state.reacquire_anchor,
                        self.config.local_search.reacquire_jump_threshold_px,
                    ) {
                        LocalCandidateDecision::Accept => {
                            status.source = Some(TrackingSource::FeatureEmbedding);
                            status.match_score = Some(candidate.score);
                            status.message = format!(
                                "卷积特征匹配局部锁定成功，RGB 特征得分 {:.3}，坐标 {:.0}, {:.0}。",
                                candidate.score, candidate.world.x, candidate.world.y
                            );
                            locate_summary = Self::locate_success_summary("局部", &candidate);
                            estimate =
                                Some(self.commit_success(candidate, TrackingSource::FeatureEmbedding));
                        }
                        LocalCandidateDecision::ForceGlobalRelocate { jump, .. } => {
                            self.state.force_global_relocate();
                            forced_global_jump = Some(jump);
                        }
                        LocalCandidateDecision::Reject => {}
                    }
                }

                if estimate.is_none() {
                    if let Some(jump) = forced_global_jump {
                        status.message = format!(
                            "小地图恢复后卷积特征局部候选跳变 {:.0}，超过阈值 {}，切回全局重定位。",
                            jump, self.config.local_search.reacquire_jump_threshold_px
                        );
                    } else {
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
        }

        if estimate.is_none() {
            let result = locate_global_relocate_runtime(
                &self.matcher,
                &self.color_pyramid,
                &self.config.template,
                &templates.coarse,
                &templates.global,
            )?;
            global_result = Some(result.clone());

            if let Some(coarse) = result.accepted {
                let region = search_region_around_center(
                    self.color_pyramid.local.image.width(),
                    self.color_pyramid.local.image.height(),
                    center_to_scaled(coarse.world, self.color_pyramid.local.scale),
                    self.config.template.global_refine_radius_px
                        / self.color_pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region_rgba(&self.color_pyramid.local.image, region)?;
                let refine = self.matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.color_pyramid.local.scale,
                )?;
                refine_result = Some(refine.clone());
                if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                    status.source = Some(TrackingSource::FeatureEmbedding);
                    status.match_score = Some(candidate.score);
                    status.message = format!(
                        "卷积特征匹配全局重定位成功，RGB 特征得分 {:.3}，坐标 {:.0}, {:.0}。",
                        candidate.score, candidate.world.x, candidate.world.y
                    );
                    locate_summary = Self::locate_success_summary("全局", &candidate);
                    estimate =
                        Some(self.commit_success(candidate, TrackingSource::FeatureEmbedding));
                }
            }
        }

        if estimate.is_none() {
            locate_summary = Self::locate_failure_summary(
                "全局",
                refine_result.as_ref().or(global_result.as_ref()),
            );
            estimate = self.apply_inertial_fallback(&mut status);
            if let Some(position) = estimate.as_ref() {
                locate_summary = Self::with_inertial_suffix(&locate_summary, position);
            }
        }

        if estimate.is_none() {
            status.source = None;
            status.match_score = None;
            status.message = "卷积特征匹配当前帧未找到可靠匹配，等待下一帧。".to_owned();
        }
        status.locate_summary = locate_summary;

        let debug = self.debug_enabled.then(|| {
            self.build_debug_snapshot(
                &captured,
                &captured,
                global_result.as_ref(),
                None,
                refine_result.as_ref().or(local_result.as_ref()),
                estimate.as_ref(),
                probe_sample.as_ref(),
            )
        });

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
            probe_summary: String::new(),
            locate_summary: String::new(),
        }
    }

    fn probe_summary(&self, sample: Option<&MinimapPresenceSample>) -> String {
        match sample {
            Some(sample) => format!(
                "{} {:.3}/{:.3}",
                if sample.present {
                    "存在"
                } else {
                    "不存在"
                },
                sample.score,
                sample.threshold
            ),
            None if self.presence_detector.is_some() => "等待判断".to_owned(),
            None => "未启用".to_owned(),
        }
    }

    fn locate_success_summary(scope: &str, candidate: &MatchCandidate) -> String {
        format!(
            "{scope}成功 {:.3} @ {:.0}, {:.0}",
            candidate.score, candidate.world.x, candidate.world.y
        )
    }

    fn locate_failure_summary(scope: &str, result: Option<&LocateResult>) -> String {
        result.map_or_else(
            || format!("{scope}未命中"),
            |result| format!("{scope}未命中 {:.3}", result.best_score),
        )
    }

    fn blocked_locate_summary(reason: &str, estimate: Option<&PositionEstimate>) -> String {
        estimate.map_or_else(
            || format!("已阻止定位，{reason}"),
            |estimate| {
                format!(
                    "已阻止定位，{reason}，惯性保位 @ {:.0}, {:.0}",
                    estimate.world.x, estimate.world.y
                )
            },
        )
    }

    fn with_inertial_suffix(summary: &str, estimate: &PositionEstimate) -> String {
        format!(
            "{summary}，惯性保位 @ {:.0}, {:.0}",
            estimate.world.x, estimate.world.y
        )
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
            status.message = format!(
                "小地图圆环探针未命中，小地图疑似被遮挡，已阻止定位，{}",
                status.message
            );
            return estimate;
        }

        status.source = None;
        status.match_score = None;
        status.message =
            "小地图圆环探针未命中，小地图疑似被遮挡，已阻止定位，等待界面恢复。".to_owned();
        None
    }

    fn apply_probe_unavailable_fallback(
        &mut self,
        status: &mut TrackingStatus,
    ) -> Option<PositionEstimate> {
        let estimate = self.apply_inertial_fallback(status);
        if estimate.is_some() {
            status.message = format!("小地图圆环探针未启用，已阻止定位，{}", status.message);
            return estimate;
        }

        status.source = None;
        status.match_score = None;
        status.message =
            "小地图圆环探针未启用，已阻止定位；请先完成小地图圆形取区并启用探针。".to_owned();
        None
    }

    fn build_debug_snapshot(
        &self,
        captured: &GrayImage,
        _global_template: &GrayImage,
        global_result: Option<&LocateResult>,
        _refine_crop: Option<&SearchCrop>,
        refine_result: Option<&LocateResult>,
        estimate: Option<&PositionEstimate>,
        probe_sample: Option<&MinimapPresenceSample>,
    ) -> TrackingDebugSnapshot {
        let minimap_input = preview_image(
            "Minimap Input",
            &capture_template_inner_square(
                captured,
                self.config.template.mask_inner_radius,
                self.config.template.mask_outer_radius,
            ),
            &[],
            196,
        );

        let mut fields = vec![
            DebugField::new("阶段", self.state.stage.to_string()),
            DebugField::new("设备", self.matcher.device_label()),
            DebugField::new("编码器", self.matcher.source_label()),
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
            DebugField::new(
                "重获锚点",
                self.state.reacquire_anchor.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
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

        let mut images = vec![minimap_input];
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
        captured: Option<&GrayImage>,
        sample: &MinimapPresenceSample,
        estimate: Option<&PositionEstimate>,
    ) -> TrackingDebugSnapshot {
        let mut fields = vec![
            DebugField::new("阶段", self.state.stage.to_string()),
            DebugField::new("设备", self.matcher.device_label()),
            DebugField::new("编码器", self.matcher.source_label()),
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
            DebugField::new(
                "重获锚点",
                self.state.reacquire_anchor.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
        ];
        let mut images = Vec::new();
        if let Some(captured) = captured {
            images.push(preview_image(
                "Minimap Input",
                &capture_template_inner_square(
                    captured,
                    self.config.template.mask_inner_radius,
                    self.config.template.mask_outer_radius,
                ),
                &[],
                196,
            ));
        }

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

fn coarse_refine_radius_px(config: &TemplateTrackingConfig) -> u32 {
    config.global_refine_radius_px.max(384)
}

#[cfg(feature = "ai-burn")]
fn coarse_peak_suppression_radius(
    config: &TemplateTrackingConfig,
    coarse_scale: u32,
    template_width: u32,
    template_height: u32,
) -> u32 {
    (coarse_refine_radius_px(config) / coarse_scale.max(1))
        .max(template_width.max(template_height) / 2)
        .max(1)
}

#[cfg(feature = "ai-burn")]
fn coarse_peak_candidates(
    result: &LocateResult,
    config: &TemplateTrackingConfig,
    template_width: u32,
    template_height: u32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
) -> Vec<MatchCandidate> {
    let mut candidates = Vec::new();
    if let Some(score_map) = result.score_map.as_ref() {
        let peaks = top_score_peaks(
            score_map,
            result.score_width,
            result.score_height,
            config.global_match_threshold,
            coarse_peak_suppression_radius(config, scale, template_width, template_height),
            MAX_GLOBAL_COARSE_CANDIDATES,
        );
        candidates.extend(peaks.into_iter().map(|peak| MatchCandidate {
            world: WorldPoint::new(
                (origin_x + peak.left + template_width / 2) as f32 * scale as f32,
                (origin_y + peak.top + template_height / 2) as f32 * scale as f32,
            ),
            score: peak.score,
        }));
    }

    if candidates.is_empty() {
        if let Some(candidate) = result.accepted.clone() {
            candidates.push(candidate);
        }
    }

    candidates
}

#[cfg(feature = "ai-burn")]
fn locate_global_relocate_runtime(
    matcher: &BurnFeatureMatcher,
    color_pyramid: &ColorMapPyramid,
    config: &TemplateTrackingConfig,
    coarse_template: &PreparedTemplate,
    global_template: &PreparedTemplate,
) -> Result<LocateResult> {
    let coarse = matcher.locate_coarse_prepared(
        coarse_template,
        config.global_match_threshold,
        0,
        0,
        color_pyramid.coarse.scale,
    )?;
    if coarse.accepted.is_none() {
        return Ok(coarse);
    }

    let mut best_refined = None;
    let mut best_score = f32::MIN;
    for candidate in coarse_peak_candidates(
        &coarse,
        config,
        coarse_template.width(),
        coarse_template.height(),
        0,
        0,
        color_pyramid.coarse.scale,
    ) {
        let region = search_region_around_center(
            color_pyramid.global.image.width(),
            color_pyramid.global.image.height(),
            center_to_scaled(candidate.world, color_pyramid.global.scale),
            coarse_refine_radius_px(config) / color_pyramid.global.scale.max(1),
            global_template.width(),
            global_template.height(),
        )?;
        let crop = crop_search_region_rgba(&color_pyramid.global.image, region)?;
        let refined = matcher.locate_global_crop_prepared(
            &crop.image,
            global_template,
            config.global_match_threshold,
            crop.origin_x,
            crop.origin_y,
            color_pyramid.global.scale,
        )?;
        if refined.accepted.is_some() && refined.best_score > best_score {
            best_score = refined.best_score;
            best_refined = Some(refined);
        }
    }

    Ok(best_refined.unwrap_or(coarse))
}

impl TrackingWorker for BurnTrackerWorker {
    fn refresh_interval(&self) -> Duration {
        #[cfg(feature = "ai-burn")]
        {
            return Duration::from_millis(self.inner.config.ai.refresh_rate_ms);
        }

        #[cfg(not(feature = "ai-burn"))]
        {
            Duration::from_millis(250)
        }
    }

    fn tick(&mut self) -> Result<TrackingTick> {
        #[cfg(feature = "ai-burn")]
        {
            return self.inner.run_frame();
        }

        #[cfg(not(feature = "ai-burn"))]
        {
            anyhow::bail!("卷积特征匹配后端被选中，但当前二进制未启用 `ai-burn` 特性")
        }
    }

    fn set_debug_enabled(&mut self, enabled: bool) {
        #[cfg(feature = "ai-burn")]
        {
            self.inner.debug_enabled = enabled;
        }

        #[cfg(not(feature = "ai-burn"))]
        {
            let _ = enabled;
        }
    }

    fn initial_status(&self) -> TrackingStatus {
        #[cfg(feature = "ai-burn")]
        let message = format!(
            "卷积特征匹配引擎已就绪：设备 {}，可用后端 {}，{} + Burn 张量相似度搜索。",
            self.inner.matcher.device_label(),
            available_burn_backends(),
            self.inner.matcher.source_label()
        );

        #[cfg(not(feature = "ai-burn"))]
        let message = "卷积特征匹配引擎当前二进制未启用 `ai-burn` 特性。".to_owned();

        TrackingStatus::new(TrackerEngineKind::ConvolutionFeatureMatch, message)
    }

    fn engine_kind(&self) -> TrackerEngineKind {
        TrackerEngineKind::ConvolutionFeatureMatch
    }
}

#[cfg(all(test, feature = "ai-burn", burn_vulkan_backend))]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::{
        config::{AiDevicePreference, AppConfig},
        domain::tracker::TrackingSource,
        resources::{
            WorkspaceSnapshot, load_logic_map_with_tracking_poi_scaled_image,
            load_logic_map_with_tracking_poi_scaled_rgba_image,
        },
        tracking::test_support::{
            StressFailure, StressRoundStats, build_test_workspace, print_perf_ms,
            random_stress_paths, require_vulkan_discrete_ordinal, runtime_config_or_default,
            stress_env_u32, stress_env_usize, synthetic_capture_rgba_from_map, timed,
            write_stress_report,
        },
    };
    use anyhow::{Result, bail};
    use image::{GrayImage, RgbaImage};

    const MAX_ROUNDS: usize = 6;
    const GLOBAL_CASES_PER_ROUND: usize = 40;
    const LOCAL_STEPS_PER_CASE: usize = 6;
    const LOCAL_STEP_MIN: u32 = 28;
    const LOCAL_STEP_MAX: u32 = 112;
    const GLOBAL_TOLERANCE: f32 = 16.0;
    const LOCAL_TOLERANCE: f32 = 12.0;
    const TARGET_ACCURACY: f32 = 0.90;

    struct TestFixture {
        config: AppConfig,
        workspace: WorkspaceSnapshot,
        map_cache_key: String,
        map: GrayImage,
        color_map: RgbaImage,
        color_pyramid: ColorMapPyramid,
        masks: MaskSet,
    }

    #[derive(Debug, Clone)]
    struct FrameResult {
        world: Option<WorldPoint>,
        score: Option<f32>,
        color_score: Option<f32>,
        source: Option<TrackingSource>,
        note: String,
    }

    static FIXTURE: OnceLock<TestFixture> = OnceLock::new();

    fn max_rounds() -> usize {
        stress_env_usize("GAME_MAP_TRACKER_STRESS_ROUNDS", MAX_ROUNDS)
    }

    fn global_cases_per_round() -> usize {
        stress_env_usize(
            "GAME_MAP_TRACKER_STRESS_GLOBAL_CASES",
            GLOBAL_CASES_PER_ROUND,
        )
    }

    fn local_steps_per_case() -> usize {
        stress_env_usize("GAME_MAP_TRACKER_STRESS_LOCAL_STEPS", LOCAL_STEPS_PER_CASE)
    }

    fn local_step_min() -> u32 {
        stress_env_u32("GAME_MAP_TRACKER_STRESS_LOCAL_STEP_MIN", LOCAL_STEP_MIN)
    }

    fn local_step_max() -> u32 {
        stress_env_u32("GAME_MAP_TRACKER_STRESS_LOCAL_STEP_MAX", LOCAL_STEP_MAX)
    }

    fn fixture() -> &'static TestFixture {
        FIXTURE.get_or_init(|| {
            let (fixture, elapsed) = timed(|| {
                let mut config = runtime_config_or_default();
                config.ai.weights_path = None;
                let workspace = build_test_workspace(config.clone(), "ai-vulkan");
                let raw_map = load_logic_map_with_tracking_poi_scaled_image(
                    &workspace.assets.bwiki_cache_dir,
                    1,
                    config.view_size,
                )
                .expect("failed to assemble augmented z8 logic map");
                let color_map = load_logic_map_with_tracking_poi_scaled_rgba_image(
                    &workspace.assets.bwiki_cache_dir,
                    1,
                    config.view_size,
                )
                .expect("failed to assemble augmented z8 color logic map");
                let map = imageproc::contrast::equalize_histogram(&raw_map);
                let color_pyramid = load_logic_color_map_pyramid(&workspace)
                    .expect("failed to build color pyramid");
                let map_cache_key =
                    tracker_map_cache_key(&workspace).expect("failed to compute map cache key");
                let masks = build_template_masks(&config);

                TestFixture {
                    config,
                    workspace,
                    map_cache_key,
                    map,
                    color_map,
                    color_pyramid,
                    masks,
                }
            });
            print_perf_ms("ai/vulkan", "fixture_prepare", elapsed);
            fixture
        })
    }

    fn matcher_for_vulkan(fixture: &TestFixture, ordinal: usize) -> BurnFeatureMatcher {
        let mut config = fixture.config.ai.clone();
        config.device = AiDevicePreference::Vulkan;
        config.device_index = ordinal;
        BurnFeatureMatcher::new(
            &fixture.workspace,
            &config,
            &fixture.color_pyramid,
            &fixture.masks,
            &fixture.map_cache_key,
        )
        .expect("failed to create Vulkan feature matcher")
    }

    fn within_tolerance(actual: Option<WorldPoint>, expected: (u32, u32), tolerance: f32) -> bool {
        actual.is_some_and(|actual| {
            (actual.x - expected.0 as f32).abs() <= tolerance
                && (actual.y - expected.1 as f32).abs() <= tolerance
        })
    }

    fn locate_global_runtime(
        fixture: &TestFixture,
        matcher: &BurnFeatureMatcher,
        templates: &PreparedCaptureTemplates,
    ) -> Result<LocateResult> {
        locate_global_relocate_runtime(
            matcher,
            &fixture.color_pyramid,
            &fixture.config.template,
            &templates.coarse,
            &templates.global,
        )
    }

    fn simulate_runtime_frame(
        fixture: &TestFixture,
        matcher: &BurnFeatureMatcher,
        capture: &RgbaImage,
        state: &mut TrackerState,
    ) -> Result<FrameResult> {
        let templates =
            matcher.prepare_capture_templates(capture, &fixture.config, &fixture.color_pyramid)?;
        let mut best_score = None;
        let mut best_color = None;
        let mut note = String::new();

        if fixture.config.local_search.enabled && matches!(state.stage, SearchStage::LocalTrack) {
            if let Some(last_world) = state.last_world {
                let region = search_region_around_center(
                    fixture.color_pyramid.local.image.width(),
                    fixture.color_pyramid.local.image.height(),
                    center_to_scaled(last_world, fixture.color_pyramid.local.scale),
                    fixture.config.local_search.radius_px
                        / fixture.color_pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region_rgba(&fixture.color_pyramid.local.image, region)?;
                let result = matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    fixture.config.template.local_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    fixture.color_pyramid.local.scale,
                )?;
                best_score = Some(result.best_score);
                best_color = Some(result.best_score);
                if let Some(candidate) = result.accepted.clone() {
                    match local_candidate_decision(
                        last_world,
                        candidate.world,
                        fixture.config.local_search.max_accepted_jump_px,
                        state.reacquire_anchor,
                        fixture.config.local_search.reacquire_jump_threshold_px,
                    ) {
                        LocalCandidateDecision::Accept => {
                            state.mark_success(candidate.world);
                            return Ok(FrameResult {
                                world: Some(candidate.world),
                                score: Some(candidate.score),
                                color_score: Some(candidate.score),
                                source: Some(TrackingSource::FeatureEmbedding),
                                note: "local_accept".to_owned(),
                            });
                        }
                        LocalCandidateDecision::ForceGlobalRelocate { jump, .. } => {
                            state.force_global_relocate();
                            note = format!("forced_global_jump={jump:.1}");
                        }
                        LocalCandidateDecision::Reject => {}
                    }
                }
                if note.is_empty() {
                    let switched =
                        state.increment_local_fail(fixture.config.local_search.lock_fail_threshold);
                    note = if switched {
                        "local_fail_switched_to_global".to_owned()
                    } else {
                        format!("local_fail_streak={}", state.local_fail_streak)
                    };
                }
            }
        }

        let global = locate_global_runtime(fixture, matcher, &templates)?;
        best_score = Some(global.best_score);
        best_color = Some(global.best_score);
        if let Some(coarse) = global.accepted {
            let region = search_region_around_center(
                fixture.color_pyramid.local.image.width(),
                fixture.color_pyramid.local.image.height(),
                center_to_scaled(coarse.world, fixture.color_pyramid.local.scale),
                fixture.config.template.global_refine_radius_px
                    / fixture.color_pyramid.local.scale.max(1),
                templates.local.width(),
                templates.local.height(),
            )?;
            let crop = crop_search_region_rgba(&fixture.color_pyramid.local.image, region)?;
            let refine = matcher.locate_dynamic_prepared(
                &crop.image,
                &templates.local,
                fixture.config.template.global_match_threshold,
                crop.origin_x,
                crop.origin_y,
                fixture.color_pyramid.local.scale,
            )?;
            best_score = Some(refine.best_score);
            best_color = Some(refine.best_score);
            if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                state.mark_success(candidate.world);
                return Ok(FrameResult {
                    world: Some(candidate.world),
                    score: Some(candidate.score),
                    color_score: Some(candidate.score),
                    source: Some(TrackingSource::FeatureEmbedding),
                    note: "global_accept".to_owned(),
                });
            }
        }

        let inertial = state.next_inertial_position(fixture.config.max_lost_frames);
        if note.is_empty() {
            note = "global_fail".to_owned();
        }
        Ok(FrameResult {
            world: inertial,
            score: best_score,
            color_score: best_color,
            source: inertial.map(|_| TrackingSource::InertialHold),
            note,
        })
    }

    fn run_round(
        fixture: &TestFixture,
        matcher: &BurnFeatureMatcher,
        seed: u64,
    ) -> Result<StressRoundStats> {
        let cases = random_stress_paths(
            &fixture.map,
            fixture.config.view_size,
            global_cases_per_round(),
            local_steps_per_case(),
            local_step_min(),
            local_step_max(),
            seed,
        );

        let mut stats = StressRoundStats::default();
        for (case_index, case) in cases.iter().enumerate() {
            let mut state = TrackerState::default();

            let capture =
                synthetic_capture_rgba_from_map(&fixture.color_map, &fixture.config, case.start);
            let frame = simulate_runtime_frame(fixture, matcher, &capture, &mut state)?;
            stats.global_total += 1;
            if within_tolerance(frame.world, case.start, GLOBAL_TOLERANCE) {
                stats.global_success += 1;
            } else {
                stats.failures.push(StressFailure {
                    case_index,
                    step_index: 0,
                    stage: "global",
                    expected: case.start,
                    actual: frame.world.map(|world| (world.x, world.y)),
                    score: frame.score,
                    color_score: frame.color_score,
                    source: frame.source.map(|source| source.to_string()),
                    note: frame.note,
                });
            }

            for (step_index, target) in case.locals.iter().copied().enumerate() {
                let capture =
                    synthetic_capture_rgba_from_map(&fixture.color_map, &fixture.config, target);
                let frame = simulate_runtime_frame(fixture, matcher, &capture, &mut state)?;
                stats.local_total += 1;
                if within_tolerance(frame.world, target, LOCAL_TOLERANCE) {
                    stats.local_success += 1;
                } else {
                    stats.failures.push(StressFailure {
                        case_index,
                        step_index: step_index + 1,
                        stage: "local",
                        expected: target,
                        actual: frame.world.map(|world| (world.x, world.y)),
                        score: frame.score,
                        color_score: frame.color_score,
                        source: frame.source.map(|source| source.to_string()),
                        note: frame.note,
                    });
                }
            }
        }

        Ok(stats)
    }

    #[test]
    fn vulkan_discrete_randomized_runtime_regression() -> Result<()> {
        let fixture = fixture();
        let (ordinal, device_name) = require_vulkan_discrete_ordinal();
        let (matcher, init_elapsed) = timed(|| matcher_for_vulkan(fixture, ordinal));
        print_perf_ms("ai/vulkan", "matcher_init", init_elapsed);

        let mut best_global = 0.0f32;
        let mut best_local = 0.0f32;
        let mut best_overall = 0.0f32;

        let max_rounds = max_rounds();
        for round in 0..max_rounds {
            let seed = 0x4149_5655_4c4b_414eu64.wrapping_add(round as u64 * 0x9e37_79b9);
            let (stats, elapsed) = timed(|| run_round(fixture, &matcher, seed));
            let stats = stats?;
            let report_path = write_stress_report(
                "ai-vulkan",
                round + 1,
                &stats,
                &format!("seed={seed} device={} ({device_name})", ordinal),
            );
            let global_accuracy = stats.global_accuracy();
            let local_accuracy = stats.local_accuracy();
            let overall_accuracy = stats.overall_accuracy();
            best_global = best_global.max(global_accuracy);
            best_local = best_local.max(local_accuracy);
            best_overall = best_overall.max(overall_accuracy);

            println!(
                "[ai/vulkan][round={}] global={:.2}% local={:.2}% overall={:.2}% elapsed_ms={:.0} failures={} report={}",
                round + 1,
                global_accuracy * 100.0,
                local_accuracy * 100.0,
                overall_accuracy * 100.0,
                elapsed.as_secs_f64() * 1000.0,
                stats.failures.len(),
                report_path.display()
            );

            if global_accuracy >= TARGET_ACCURACY && local_accuracy >= TARGET_ACCURACY {
                return Ok(());
            }
        }

        bail!(
            "AI Vulkan stress accuracy stayed below target after {} rounds; best global {:.2}%, best local {:.2}%, best overall {:.2}%",
            max_rounds,
            best_global * 100.0,
            best_local * 100.0,
            best_overall * 100.0
        )
    }
}
