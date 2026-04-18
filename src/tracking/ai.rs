use std::{fs, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Result;
use tracing::info;
#[cfg(feature = "ai-burn")]
use burn::{
    backend::ndarray::NdArrayDevice,
    tensor::{
        Tensor, TensorData, backend::Backend, cast::ToElement, module::conv2d, ops::ConvOptions,
    },
};
#[cfg(feature = "ai-burn")]
use image::GrayImage;
#[cfg(feature = "ai-burn")]
use safetensors::SafeTensors;

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
        capture::{CaptureSource, DesktopCapture},
        debug::{DebugField, TrackingDebugSnapshot},
        precompute::{
            PersistedTensorCache, clear_match_pyramid_caches, clear_tensor_caches_by_prefix,
            load_or_build_match_pyramid, load_tensor_cache, metadata_fingerprint,
            save_tensor_cache, tracker_tensor_cache_path,
        },
        presence::{MinimapPresenceDetector, MinimapPresenceSample},
        vision::{
            DebugOverlay, LocalCandidateDecision, MapPyramid, MaskSet, MatchCandidate, SearchCrop,
            SearchStage, TrackerState, build_debug_snapshot, build_mask,
            build_match_representation, capture_template_annulus, center_to_scaled,
            coarse_global_downscale, crop_search_region, local_candidate_decision,
            mask_as_unit_vec, preview_heatmap, preview_image, preview_mask_image, scaled_dimension,
            search_region_around_center,
        },
    },
};

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

pub struct BurnTrackerWorker {
    #[cfg(feature = "ai-burn")]
    inner: BurnTrackerInner,
}

#[cfg(feature = "ai-burn")]
struct BurnTrackerInner {
    config: AppConfig,
    capture: DesktopCapture,
    presence_detector: Option<MinimapPresenceDetector>,
    pyramid: MapPyramid,
    masks: MaskSet,
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
    image: GrayImage,
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
const FEATURE_ENCODER_CACHE_VERSION: u32 = 1;

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
            let prepared_pyramid = load_or_build_match_pyramid(workspace.as_ref())?;
            let map_cache_key = prepared_pyramid.cache_key;
            let pyramid = prepared_pyramid.pyramid;
            let masks = build_template_masks(&config);
            let matcher = BurnFeatureMatcher::new(
                workspace.as_ref(),
                &config.ai,
                &pyramid,
                &masks,
                &map_cache_key,
            )?;

            Ok(Self {
                inner: BurnTrackerInner {
                    config,
                    capture,
                    presence_detector,
                    pyramid,
                    masks,
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

        let prepared_pyramid = load_or_build_match_pyramid(workspace)?;
        let masks = build_template_masks(&workspace.config);
        let _ = BurnFeatureMatcher::new(
            workspace,
            &workspace.config.ai,
            &prepared_pyramid.pyramid,
            &masks,
            &prepared_pyramid.cache_key,
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
        let kernels = vec![
            -1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -1.0, -2.0, -1.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 2.0, -1.0,
            2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, 1.0,
            2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0,
        ];
        let edge_kernels =
            Tensor::<B, 4>::from_data(TensorData::new(kernels, [6, 1, 3, 3]), &device);

        Ok(Self {
            device,
            device_label,
            edge_kernels,
            edge_bias: None,
            source: EncoderSource::BuiltIn,
            padding: 1,
            output_channels: 7,
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
        if shape.len() != 4 || shape[1] != 1 || shape[2] != shape[3] {
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
            output_channels: out_channels + 1,
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

    fn encode(&self, image: &GrayImage) -> Result<Tensor<B, 4>> {
        let base = Tensor::<B, 4>::from_data(
            TensorData::new(
                image
                    .pixels()
                    .map(|pixel| f32::from(pixel.0[0]) / 255.0)
                    .collect::<Vec<_>>(),
                [1, 1, image.height() as usize, image.width() as usize],
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
fn prepare_capture_templates(
    captured: &GrayImage,
    config: &AppConfig,
    pyramid: &MapPyramid,
) -> (GrayImage, GrayImage, GrayImage) {
    let square = capture_template_annulus(
        captured,
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );
    (
        prepare_square_template(&square, config.view_size, pyramid.local.scale),
        prepare_square_template(&square, config.view_size, pyramid.global.scale),
        prepare_square_template(&square, config.view_size, pyramid.coarse.scale),
    )
}

#[cfg(feature = "ai-burn")]
#[cfg_attr(not(test), allow(dead_code))]
fn prepare_capture_template(
    captured: &GrayImage,
    view_size: u32,
    scale: u32,
    mask_inner_radius: f32,
    mask_outer_radius: f32,
) -> GrayImage {
    let square = capture_template_annulus(captured, mask_inner_radius, mask_outer_radius);
    prepare_square_template(&square, view_size, scale)
}

#[cfg(feature = "ai-burn")]
fn prepare_square_template(square: &GrayImage, view_size: u32, scale: u32) -> GrayImage {
    let template_size = scaled_dimension(view_size.max(1), scale.max(1));
    let resized = if square.width() == template_size && square.height() == template_size {
        square.clone()
    } else {
        image::imageops::resize(
            square,
            template_size,
            template_size,
            image::imageops::FilterType::Triangle,
        )
    };

    build_match_representation(&resized)
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
        pyramid: &MapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let selection = select_burn_device(config)?;
        Self::from_selection(workspace, selection, pyramid, masks, map_cache_key)
    }

    fn from_selection(
        workspace: &WorkspaceSnapshot,
        selection: BurnDeviceSelection,
        pyramid: &MapPyramid,
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
        captured: &GrayImage,
        config: &AppConfig,
        pyramid: &MapPyramid,
    ) -> Result<PreparedCaptureTemplates> {
        let (local, global, coarse) = prepare_capture_templates(captured, config, pyramid);
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
        template: &GrayImage,
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
        image: &GrayImage,
        template: &GrayImage,
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
        image: &GrayImage,
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
        image: &GrayImage,
        template: &GrayImage,
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
        image: &GrayImage,
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
    fn image(&self) -> &GrayImage {
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
        pyramid: &MapPyramid,
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
        image: GrayImage,
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
        template: &GrayImage,
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
            threshold,
            origin_x,
            origin_y,
            scale,
        )
    }

    #[allow(dead_code)]
    fn locate_global_crop(
        &self,
        image: &GrayImage,
        template: &GrayImage,
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
        image: &GrayImage,
        template: &BurnPreparedTemplate<B>,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::<B>::from_gray_image(image, &self.encoder)?;
        self.locate_cached(
            &search, template, None, threshold, origin_x, origin_y, scale,
        )
    }

    #[allow(dead_code)]
    fn locate_local(
        &self,
        image: &GrayImage,
        template: &GrayImage,
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
        image: &GrayImage,
        template: &BurnPreparedTemplate<B>,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::<B>::from_gray_image(image, &self.encoder)?;
        self.locate_cached(
            &search, template, None, threshold, origin_x, origin_y, scale,
        )
    }

    fn locate_cached(
        &self,
        search: &SearchTensorCache<B>,
        template: &BurnPreparedTemplate<B>,
        precomputed_patch_energy: Option<Tensor<B, 4>>,
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

        if burn_score_map_capture_enabled() {
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
    fn from_gray_image(image: &GrayImage, encoder: &FixedFeatureEncoder<B>) -> Result<Self> {
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
    image: &GrayImage,
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

    let search = SearchTensorCache::<B>::from_gray_image(image, encoder)?;
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
    let capture_score_map = burn_score_map_capture_enabled();
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
        let probe_sample = self
            .presence_detector
            .as_ref()
            .map(MinimapPresenceDetector::sample)
            .transpose()?;
        if let Some(sample) = probe_sample.as_ref().filter(|sample| !sample.present) {
            let mut status = self.base_status();
            let estimate = self.apply_probe_absent_fallback(&mut status);
            let debug = self
                .debug_enabled
                .then(|| self.build_probe_miss_debug_snapshot(sample, estimate.as_ref()));
            return Ok(TrackingTick {
                status,
                estimate,
                debug,
            });
        }

        let captured = self.capture.capture_gray()?;
        let templates =
            self.matcher
                .prepare_capture_templates(&captured, &self.config, &self.pyramid)?;

        let mut status = self.base_status();
        let mut estimate = None;
        let mut global_result = None;
        let mut refine_crop = None;
        let mut refine_result = None;
        let mut local_result = None;
        let mut forced_global_jump: Option<f32> = None;

        if self.config.local_search.enabled && matches!(self.state.stage, SearchStage::LocalTrack) {
            if let Some(last_world) = self.state.last_world {
                let region = search_region_around_center(
                    self.pyramid.local.image.width(),
                    self.pyramid.local.image.height(),
                    center_to_scaled(last_world, self.pyramid.local.scale),
                    self.config.local_search.radius_px / self.pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region(&self.pyramid.local.image, region)?;
                let result = self.matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    self.config.template.local_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.local.scale,
                )?;
                refine_crop = self.debug_enabled.then_some(crop);
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
                                "卷积特征匹配局部锁定成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                                candidate.score, candidate.world.x, candidate.world.y
                            );
                            estimate = Some(
                                self.commit_success(candidate, TrackingSource::FeatureEmbedding),
                            );
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
            let coarse = self.matcher.locate_coarse_prepared(
                &templates.coarse,
                self.config.template.global_match_threshold,
                0,
                0,
                self.pyramid.coarse.scale,
            )?;
            let result = if let Some(candidate) = coarse.accepted.clone() {
                let region = search_region_around_center(
                    self.pyramid.global.image.width(),
                    self.pyramid.global.image.height(),
                    center_to_scaled(candidate.world, self.pyramid.global.scale),
                    coarse_refine_radius_px(&self.config.template)
                        / self.pyramid.global.scale.max(1),
                    templates.global.width(),
                    templates.global.height(),
                )?;
                let crop = crop_search_region(&self.pyramid.global.image, region)?;
                let refined = self.matcher.locate_global_crop_prepared(
                    &crop.image,
                    &templates.global,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.global.scale,
                )?;
                refined.accepted.clone().map_or(coarse, |_| refined)
            } else {
                coarse
            };
            global_result = Some(result.clone());

            if let Some(coarse) = result.accepted {
                let region = search_region_around_center(
                    self.pyramid.local.image.width(),
                    self.pyramid.local.image.height(),
                    center_to_scaled(coarse.world, self.pyramid.local.scale),
                    self.config.template.global_refine_radius_px / self.pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region(&self.pyramid.local.image, region)?;
                let refine = self.matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.local.scale,
                )?;
                refine_crop = self.debug_enabled.then_some(crop);
                refine_result = Some(refine.clone());
                if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                    status.source = Some(TrackingSource::FeatureEmbedding);
                    status.match_score = Some(candidate.score);
                    status.message = format!(
                        "卷积特征匹配全局重定位成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                        candidate.score, candidate.world.x, candidate.world.y
                    );
                    estimate =
                        Some(self.commit_success(candidate, TrackingSource::FeatureEmbedding));
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

        let debug = self.debug_enabled.then(|| {
            self.build_debug_snapshot(
                &captured,
                templates.global.image(),
                global_result.as_ref(),
                refine_crop.as_ref(),
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
        let global_heatmap = global_result.and_then(|result| {
            let score_map = result.score_map.as_ref()?;
            Some(preview_heatmap(
                "Tensor Coarse Heatmap",
                result.score_width,
                result.score_height,
                score_map,
                Some((result.best_left, result.best_top)),
                196,
            ))
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
        let refine_heatmap = refine_result.and_then(|result| {
            let score_map = result.score_map.as_ref()?;
            Some(preview_heatmap(
                "Tensor Refine Heatmap",
                result.score_width,
                result.score_height,
                score_map,
                Some((result.best_left, result.best_top)),
                196,
            ))
        });

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
            DebugField::new(
                "重获锚点",
                self.state.reacquire_anchor.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
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

fn coarse_refine_radius_px(config: &TemplateTrackingConfig) -> u32 {
    config.global_refine_radius_px.max(384)
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

#[cfg(all(test, feature = "ai-burn"))]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::{
        config::{AiDevicePreference, AppConfig, CaptureRegion},
        resources::{WorkspaceSnapshot, load_logic_map_with_tracking_poi_scaled_image},
        tracking::{
            precompute::load_or_build_match_pyramid,
            test_support::{
                benchmark_repeated, build_test_workspace, print_perf_ms, print_perf_per_op,
                sample_world_positions, timed,
            },
        },
    };
    use anyhow::Result;
    use image::{
        GrayImage, Luma,
        imageops::{FilterType, crop_imm, replace, resize},
    };

    struct TestFixture {
        config: AppConfig,
        workspace: WorkspaceSnapshot,
        map_cache_key: String,
        map: GrayImage,
        pyramid: MapPyramid,
        masks: MaskSet,
    }

    static FIXTURE: OnceLock<TestFixture> = OnceLock::new();

    fn fixture() -> &'static TestFixture {
        FIXTURE.get_or_init(|| {
            let (fixture, elapsed) = timed(|| {
                let mut config = AppConfig::default();
                config.minimap = CaptureRegion {
                    top: 0,
                    left: 0,
                    width: 255,
                    height: 235,
                };
                // Keep the AI regression fixture representative of the production path,
                // but use a lighter pyramid so CPU smoke tests finish in practical time.
                config.view_size = 320;
                config.template.local_downscale = 6;
                config.template.global_downscale = 12;
                config.template.local_match_threshold = 0.45;
                config.template.global_match_threshold = 0.40;
                config.ai.weights_path = None;
                let workspace = build_test_workspace(config.clone(), "ai");
                let prepared_pyramid =
                    load_or_build_match_pyramid(&workspace).expect("failed to build match pyramid");
                let raw_map = load_logic_map_with_tracking_poi_scaled_image(
                    &workspace.assets.bwiki_cache_dir,
                    1,
                    config.view_size,
                )
                .expect("failed to assemble augmented z8 logic map");
                let map = imageproc::contrast::equalize_histogram(&raw_map);
                let pyramid = prepared_pyramid.pyramid;
                let masks = build_template_masks(&config);

                TestFixture {
                    config,
                    workspace,
                    map_cache_key: prepared_pyramid.cache_key,
                    map,
                    pyramid,
                    masks,
                }
            });
            print_perf_ms("ai", "fixture_prepare", elapsed);
            fixture
        })
    }

    fn sample_positions(image: &GrayImage, view_size: u32) -> Vec<(u32, u32)> {
        sample_world_positions(image, view_size, 1)
    }

    fn stress_positions(image: &GrayImage, view_size: u32) -> Vec<(u32, u32)> {
        sample_world_positions(image, view_size, 2)
    }

    fn align_to(value: u32, step: u32) -> u32 {
        (value / step.max(1)) * step.max(1)
    }

    fn bounded_world_point(fixture: &TestFixture, x: i32, y: i32) -> (u32, u32) {
        let min_center = align_to(fixture.config.view_size / 2 + 32, 4);
        let max_x = align_to(
            fixture
                .map
                .width()
                .saturating_sub(fixture.config.view_size / 2 + 32),
            4,
        );
        let max_y = align_to(
            fixture
                .map
                .height()
                .saturating_sub(fixture.config.view_size / 2 + 32),
            4,
        );
        (
            align_to(x.clamp(min_center as i32, max_x as i32) as u32, 4),
            align_to(y.clamp(min_center as i32, max_y as i32) as u32, 4),
        )
    }

    fn stress_paths(fixture: &TestFixture) -> Vec<[(u32, u32); 4]> {
        sample_world_positions(&fixture.map, fixture.config.view_size, 1)
            .into_iter()
            .map(|start| {
                let dir_x = if start.0 > fixture.map.width() / 2 {
                    -1
                } else {
                    1
                };
                let dir_y = if start.1 > fixture.map.height() / 2 {
                    -1
                } else {
                    1
                };
                [
                    bounded_world_point(fixture, start.0 as i32, start.1 as i32),
                    bounded_world_point(
                        fixture,
                        start.0 as i32 + dir_x * 96,
                        start.1 as i32 + dir_y * 48,
                    ),
                    bounded_world_point(
                        fixture,
                        start.0 as i32 + dir_x * 176,
                        start.1 as i32 + dir_y * 104,
                    ),
                    bounded_world_point(
                        fixture,
                        start.0 as i32 + dir_x * 236,
                        start.1 as i32 + dir_y * 144,
                    ),
                ]
            })
            .collect()
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

    fn matcher_for_device(fixture: &TestFixture, device: AiDevicePreference) -> BurnFeatureMatcher {
        let mut config = fixture.config.ai.clone();
        config.device = device;
        BurnFeatureMatcher::new(
            &fixture.workspace,
            &config,
            &fixture.pyramid,
            &fixture.masks,
            &fixture.map_cache_key,
        )
        .expect("failed to create feature matcher")
    }

    fn locate_fixture_burn(
        fixture: &TestFixture,
        matcher: &BurnFeatureMatcher,
        capture: &GrayImage,
    ) -> Result<MatchCandidate> {
        let templates =
            matcher.prepare_capture_templates(capture, &fixture.config, &fixture.pyramid)?;
        let coarse = matcher.locate_coarse_prepared(
            &templates.coarse,
            fixture.config.template.global_match_threshold,
            0,
            0,
            fixture.pyramid.coarse.scale,
        )?;
        let coarse = coarse.accepted.unwrap_or_else(|| {
            panic!(
                "global locate should accept, best_score={:.3}, best_left={}, best_top={}",
                coarse.best_score, coarse.best_left, coarse.best_top
            )
        });
        let global_region = search_region_around_center(
            fixture.pyramid.global.image.width(),
            fixture.pyramid.global.image.height(),
            center_to_scaled(coarse.world, fixture.pyramid.global.scale),
            coarse_refine_radius_px(&fixture.config.template) / fixture.pyramid.global.scale.max(1),
            templates.global.width(),
            templates.global.height(),
        )?;
        let global_crop = crop_search_region(&fixture.pyramid.global.image, global_region)?;
        let global = matcher.locate_global_crop_prepared(
            &global_crop.image,
            &templates.global,
            fixture.config.template.global_match_threshold,
            global_crop.origin_x,
            global_crop.origin_y,
            fixture.pyramid.global.scale,
        )?;
        let global = global
            .accepted
            .or(Some(coarse))
            .expect("global refine should accept");
        let region = search_region_around_center(
            fixture.pyramid.local.image.width(),
            fixture.pyramid.local.image.height(),
            center_to_scaled(global.world, fixture.pyramid.local.scale),
            fixture.config.template.global_refine_radius_px / fixture.pyramid.local.scale.max(1),
            templates.local.width(),
            templates.local.height(),
        )?;
        let crop = crop_search_region(&fixture.pyramid.local.image, region)?;
        let refine = matcher.locate_dynamic_prepared(
            &crop.image,
            &templates.local,
            fixture.config.template.global_match_threshold,
            crop.origin_x,
            crop.origin_y,
            fixture.pyramid.local.scale,
        )?;

        Ok(refine
            .accepted
            .or(Some(global))
            .expect("refine locate should accept"))
    }

    #[test]
    fn prepared_template_uses_world_view_diameter() {
        let fixture = fixture();
        let capture = synthetic_capture(
            fixture,
            sample_positions(&fixture.map, fixture.config.view_size)[0],
        );
        let (local, local_elapsed) = timed(|| {
            prepare_capture_template(
                &capture,
                fixture.config.view_size,
                fixture.pyramid.local.scale,
                fixture.config.template.mask_inner_radius,
                fixture.config.template.mask_outer_radius,
            )
        });
        let (global, global_elapsed) = timed(|| {
            prepare_capture_template(
                &capture,
                fixture.config.view_size,
                fixture.pyramid.global.scale,
                fixture.config.template.mask_inner_radius,
                fixture.config.template.mask_outer_radius,
            )
        });
        let (coarse, coarse_elapsed) = timed(|| {
            prepare_capture_template(
                &capture,
                fixture.config.view_size,
                fixture.pyramid.coarse.scale,
                fixture.config.template.mask_inner_radius,
                fixture.config.template.mask_outer_radius,
            )
        });
        print_perf_ms("ai", "prepare_local_template", local_elapsed);
        print_perf_ms("ai", "prepare_global_template", global_elapsed);
        print_perf_ms("ai", "prepare_coarse_template", coarse_elapsed);

        assert_eq!(
            local.width(),
            scaled_dimension(fixture.config.view_size, fixture.pyramid.local.scale)
        );
        assert_eq!(
            global.width(),
            scaled_dimension(fixture.config.view_size, fixture.pyramid.global.scale)
        );
        assert_eq!(
            coarse.width(),
            scaled_dimension(fixture.config.view_size, fixture.pyramid.coarse.scale)
        );
        assert_eq!(local.width(), fixture.masks.local.width());
        assert_eq!(global.width(), fixture.masks.global.width());
        assert_eq!(coarse.width(), fixture.masks.coarse.width());
    }

    #[cfg(burn_cuda_backend)]
    #[test]
    fn synthetic_captures_locate_many_positions_on_burn_cuda() -> Result<()> {
        let fixture = fixture();
        let (matcher, init_elapsed) = match std::panic::catch_unwind(|| {
            timed(|| matcher_for_device(fixture, AiDevicePreference::Cuda))
        }) {
            Ok(result) => result,
            Err(_) => return Ok(()),
        };
        print_perf_ms("ai/cuda", "matcher_init", init_elapsed);

        let points = sample_positions(&fixture.map, fixture.config.view_size);
        if let Some(point) = points.first().copied() {
            let capture = synthetic_capture(fixture, point);
            let (candidate, locate_elapsed) =
                benchmark_repeated(0, 1, || locate_fixture_burn(fixture, &matcher, &capture));
            print_perf_per_op("ai/cuda", "reference_locate", 1, locate_elapsed);
            let candidate = candidate?;
            assert_world_close(candidate.world, point, 4.0);
        }
        for point in points.into_iter().skip(1) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_burn(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 4.0);
        }
        Ok(())
    }

    #[cfg(burn_vulkan_backend)]
    #[test]
    fn synthetic_captures_locate_many_positions_on_burn_vulkan() -> Result<()> {
        let fixture = fixture();
        let (matcher, init_elapsed) = match std::panic::catch_unwind(|| {
            timed(|| matcher_for_device(fixture, AiDevicePreference::Vulkan))
        }) {
            Ok(result) => result,
            Err(_) => return Ok(()),
        };
        print_perf_ms("ai/vulkan", "matcher_init", init_elapsed);

        let points = sample_positions(&fixture.map, fixture.config.view_size);
        if let Some(point) = points.first().copied() {
            let capture = synthetic_capture(fixture, point);
            let (candidate, locate_elapsed) =
                benchmark_repeated(0, 1, || locate_fixture_burn(fixture, &matcher, &capture));
            print_perf_per_op("ai/vulkan", "reference_locate", 1, locate_elapsed);
            let candidate = candidate?;
            assert_world_close(candidate.world, point, 4.0);
        }
        for point in points.into_iter().skip(1) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_burn(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 4.0);
        }
        Ok(())
    }

    #[cfg(burn_metal_backend)]
    #[test]
    fn synthetic_captures_locate_many_positions_on_burn_metal() -> Result<()> {
        let fixture = fixture();
        let (matcher, init_elapsed) = match std::panic::catch_unwind(|| {
            timed(|| matcher_for_device(fixture, AiDevicePreference::Metal))
        }) {
            Ok(result) => result,
            Err(_) => return Ok(()),
        };
        print_perf_ms("ai/metal", "matcher_init", init_elapsed);

        let points = sample_positions(&fixture.map, fixture.config.view_size);
        if let Some(point) = points.first().copied() {
            let capture = synthetic_capture(fixture, point);
            let (candidate, locate_elapsed) =
                benchmark_repeated(0, 1, || locate_fixture_burn(fixture, &matcher, &capture));
            print_perf_per_op("ai/metal", "reference_locate", 1, locate_elapsed);
            let candidate = candidate?;
            assert_world_close(candidate.world, point, 4.0);
        }
        for point in points.into_iter().skip(1) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_burn(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 4.0);
        }
        Ok(())
    }

    #[test]
    fn synthetic_captures_locate_many_positions_on_burn_cpu() -> Result<()> {
        let fixture = fixture();
        let (matcher, init_elapsed) =
            timed(|| matcher_for_device(fixture, AiDevicePreference::Cpu));
        print_perf_ms("ai/cpu", "matcher_init", init_elapsed);

        let points = stress_positions(&fixture.map, fixture.config.view_size);
        if let Some(point) = points.first().copied() {
            let capture = synthetic_capture(fixture, point);
            let (candidate, locate_elapsed) =
                benchmark_repeated(0, 1, || locate_fixture_burn(fixture, &matcher, &capture));
            print_perf_per_op("ai/cpu", "reference_locate", 1, locate_elapsed);
            let candidate = candidate?;
            assert_world_close(candidate.world, point, 10.0);
        }
        for point in points.into_iter().skip(1) {
            let capture = synthetic_capture(fixture, point);
            let candidate = locate_fixture_burn(fixture, &matcher, &capture)?;
            assert_world_close(candidate.world, point, 10.0);
        }

        Ok(())
    }

    #[test]
    fn local_track_sequence_stays_locked_on_burn_cpu() -> Result<()> {
        let fixture = fixture();
        let matcher = matcher_for_device(fixture, AiDevicePreference::Cpu);
        for path in stress_paths(fixture) {
            let first_capture = synthetic_capture(fixture, path[0]);
            let first = locate_fixture_burn(fixture, &matcher, &first_capture)?;
            assert_world_close(first.world, path[0], 10.0);

            for window in path.windows(2) {
                let previous = WorldPoint::new(window[0].0 as f32, window[0].1 as f32);
                let capture = synthetic_capture(fixture, window[1]);
                let templates = matcher.prepare_capture_templates(
                    &capture,
                    &fixture.config,
                    &fixture.pyramid,
                )?;
                let region = search_region_around_center(
                    fixture.pyramid.local.image.width(),
                    fixture.pyramid.local.image.height(),
                    center_to_scaled(previous, fixture.pyramid.local.scale),
                    fixture.config.local_search.radius_px / fixture.pyramid.local.scale.max(1),
                    templates.local.width(),
                    templates.local.height(),
                )?;
                let crop = crop_search_region(&fixture.pyramid.local.image, region)?;
                let locate = matcher.locate_dynamic_prepared(
                    &crop.image,
                    &templates.local,
                    fixture.config.template.local_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    fixture.pyramid.local.scale,
                )?;
                let candidate = locate.accepted.expect("local track should accept");
                assert_world_close(candidate.world, window[1], 12.0);
            }
        }

        Ok(())
    }
}
