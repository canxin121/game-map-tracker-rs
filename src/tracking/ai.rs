use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use crate::error::{ContextExt as _, Result};

use burn::{
    backend::{Autodiff, ndarray::NdArrayDevice},
    tensor::{
        Tensor, TensorData,
        activation::relu,
        backend::{AutodiffBackend, Backend},
        cast::ToElement,
        module::conv2d,
        ops::ConvOptions,
    },
};

use image::{GrayImage, RgbaImage, imageops::crop_imm};

use safetensors::{Dtype, SafeTensors, serialize_to_file, tensor::TensorView};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    domain::tracker::TrackerEngineKind,
    resources::WorkspaceSnapshot,
    tracking::{
        runtime::{TrackingStatus, TrackingTick, TrackingWorker},
        template::TemplateGlobalLocator,
    },
};

#[cfg(any(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
use crate::tracking::burn_support::burn_device_label;
use crate::{
    config::{AiTrackingConfig, AppConfig},
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackingSource},
    },
    model_assets::{embedded_tracker_encoder_bytes, repository_root},
    resources::{
        WorkspaceBootstrap, load_logic_map_with_tracking_poi_scaled_image,
        load_logic_map_with_tracking_poi_scaled_rgba_image,
    },
    tracking::{
        burn_support::{BurnDeviceSelection, available_burn_backends, select_burn_device},
        capture::{DesktopCapture, preprocess_capture},
        debug::{DebugField, TrackingDebugSnapshot},
        precompute::{
            clear_match_pyramid_caches, clear_tensor_caches_by_prefix, tracker_map_cache_key,
        },
        presence::{MinimapPresenceDetector, MinimapPresenceSample},
        vision::{
            ColorCaptureTemplates, ColorMapPyramid, ColorTemplateShape, LocalCandidateDecision,
            MaskSet, MatchCandidate, SearchCrop, SearchStage, TrackerState, build_debug_snapshot,
            build_mask, capture_template_inner_square, center_to_scaled, coarse_global_downscale,
            crop_search_region_rgba, load_logic_color_map_pyramid, local_candidate_decision,
            mask_as_unit_vec, normalized_inner_radius, prepare_color_capture_template,
            preview_image, rgba_image_as_unit_vec, scaled_color_score, scaled_dimension,
            search_region_around_center, synthesize_runtime_capture_rgba_from_map,
        },
    },
};

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
struct LocateResult {
    best_score: f32,
    accepted: Option<MatchCandidate>,
}

pub struct BurnTrackerWorker {
    inner: BurnTrackerInner,
}

struct BurnTrackerInner {
    config: AppConfig,
    capture: DesktopCapture,
    presence_detector: Option<MinimapPresenceDetector>,
    color_pyramid: ColorMapPyramid,
    state: TrackerState,
    debug_enabled: bool,
    matcher: BurnFeatureMatcher,
    template_global_locator: TemplateGlobalLocator,
}

enum BurnFeatureMatcher {
    NdArray(BurnFeatureMatcherBackend<burn::backend::NdArray>),
    #[cfg(burn_cuda_backend)]
    Cuda(BurnFeatureMatcherBackend<burn::backend::Cuda>),
    #[cfg(burn_vulkan_backend)]
    Vulkan(BurnFeatureMatcherBackend<burn::backend::Vulkan>),
    #[cfg(burn_metal_backend)]
    Metal(BurnFeatureMatcherBackend<burn::backend::Metal>),
}

struct BurnFeatureMatcherBackend<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    encoder: FixedFeatureEncoder<B>,
    local_mask: Tensor<B, 4>,
    chunk_budget_bytes: Option<usize>,
}

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

struct PreparedCaptureTemplates {
    local: PreparedTemplate,
}

enum PreparedTemplate {
    NdArray(BurnPreparedTemplate<burn::backend::NdArray>),
    #[cfg(burn_cuda_backend)]
    Cuda(BurnPreparedTemplate<burn::backend::Cuda>),
    #[cfg(burn_vulkan_backend)]
    Vulkan(BurnPreparedTemplate<burn::backend::Vulkan>),
    #[cfg(burn_metal_backend)]
    Metal(BurnPreparedTemplate<burn::backend::Metal>),
}

struct BurnPreparedTemplate<B>
where
    B: Backend<FloatElem = f32>,
{
    image: RgbaImage,
    mask: Tensor<B, 4>,
    weighted_template: Tensor<B, 4>,
    template_energy: f32,
}

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
    EmbeddedPretrained,
    BuiltIn,
}

impl EncoderSource {
    fn label(&self) -> String {
        match self {
            Self::Safetensors(path) => format!("Safetensors ({})", path.display()),
            Self::EmbeddedPretrained => {
                "Embedded Pretrained Encoder (models/tracker_encoder.safetensors)".to_owned()
            }
            Self::BuiltIn => "Built-in Conv Edge Bank".to_owned(),
        }
    }
}

#[cfg(burn_cuda_backend)]
const CUDA_CONV_IM2COL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
const WGPU_CONV_IM2COL_BUDGET_BYTES: usize = 256 * 1024 * 1024;

impl BurnTrackerWorker {
    pub fn new(workspace: Arc<WorkspaceSnapshot>) -> Result<Self> {
        info!(
            cache_root = %workspace.assets.bwiki_cache_dir.display(),
            view_size = workspace.config.view_size,
            device = %workspace.config.ai.device,
            device_index = workspace.config.ai.device_index,
            "initializing convolution tracker worker"
        );
        let config = workspace.config.clone();
        if !config.minimap.is_configured() {
            crate::bail!("小地图区域尚未配置，请先完成小地图取区");
        }
        let capture = DesktopCapture::from_absolute_region(&config.minimap)?;
        let presence_detector = MinimapPresenceDetector::new(workspace.as_ref())?;
        let map_cache_key = tracker_map_cache_key(workspace.as_ref())?;
        let masks = build_template_masks(&config);
        let color_pyramid = load_logic_color_map_pyramid(workspace.as_ref())?;
        let matcher = BurnFeatureMatcher::new(workspace.as_ref(), &config.ai, &masks)?;
        let template_global_locator = TemplateGlobalLocator::new_cached(
            workspace.as_ref(),
            &config,
            &color_pyramid,
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
                template_global_locator,
            },
        })
    }
}

pub fn rebuild_convolution_engine_cache(workspace: &WorkspaceSnapshot) -> Result<()> {
    info!(
        cache_root = %workspace.assets.bwiki_cache_dir.display(),
        device = %workspace.config.ai.device,
        device_index = workspace.config.ai.device_index,
        "rebuilding convolution tracker cache"
    );
    clear_match_pyramid_caches(workspace)?;
    clear_tensor_caches_by_prefix(workspace, "feature-local-search")?;
    clear_tensor_caches_by_prefix(workspace, "feature-global-search")?;
    clear_tensor_caches_by_prefix(workspace, "feature-coarse-search")?;

    let masks = build_template_masks(&workspace.config);
    let _ = BurnFeatureMatcher::new(workspace, &workspace.config.ai, &masks)?;
    info!("rebuild of convolution tracker cache completed");
    Ok(())
}

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

        if let Ok(encoder) = Self::load_embedded_safetensors(device, device_label) {
            return Ok(Some(encoder));
        }

        Ok(None)
    }

    fn load_single_safetensors(
        path: &Path,
        device: B::Device,
        device_label: String,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        Self::load_safetensors_bytes(
            &bytes,
            EncoderSource::Safetensors(path.to_path_buf()),
            device,
            device_label,
        )
    }

    fn load_embedded_safetensors(device: B::Device, device_label: String) -> Result<Self> {
        Self::load_safetensors_bytes(
            embedded_tracker_encoder_bytes(),
            EncoderSource::EmbeddedPretrained,
            device,
            device_label,
        )
    }

    fn load_safetensors_bytes(
        bytes: &[u8],
        source: EncoderSource,
        device: B::Device,
        device_label: String,
    ) -> Result<Self> {
        let source_label = source.label();
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
            crate::bail!(
                "unsupported encoder kernel shape {:?} in {}",
                shape,
                source_label
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
                crate::bail!(
                    "unsupported encoder bias shape {:?} in {}",
                    view.shape(),
                    source_label
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
            source,
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
        candidates.push(resolve_workspace_relative_path(
            &workspace.project_root,
            &candidate,
        ));
    }

    dedupe_paths(&mut candidates);
    candidates
}

fn resolve_workspace_relative_path(workspace_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    }
}

fn dedupe_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::new();
    paths.retain(|path| seen.insert(path.clone()));
}

fn find_safetensor_name<'a>(tensors: &SafeTensors<'a>, names: &[&'a str]) -> Result<&'a str> {
    names
        .iter()
        .copied()
        .find(|name| tensors.tensor(name).is_ok())
        .ok_or_else(|| crate::app_error!("failed to resolve encoder tensor names: {:?}", names))
}

fn safetensor_view_f32_vec(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != safetensors::Dtype::F32 {
        crate::bail!(
            "unsupported safetensors dtype {:?}, only f32 is supported",
            view.dtype()
        );
    }

    let data = view.data();
    if data.len() % std::mem::size_of::<f32>() != 0 {
        crate::bail!("invalid safetensors payload length {}", data.len());
    }

    Ok(data
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

type TrainingBackend = Autodiff<burn::backend::NdArray>;

const TRAINING_LOCAL_STEP_MIN: u32 = 28;

const TRAINING_LOCAL_STEP_MAX: u32 = 112;

const TRAINING_GLOBAL_NEGATIVE_DISTANCE: u32 = 640;

const TRAINING_LOCAL_NEGATIVE_MIN_DISTANCE: u32 = 56;

const TRAINING_POINT_ATTEMPTS: usize = 24;

#[derive(Debug, Clone)]
struct EncoderTrainingConfig {
    workspace_root: Option<PathBuf>,
    output_path: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    epochs: usize,
    steps_per_epoch: usize,
    samples_per_step: usize,
    checkpoint_every: usize,
    learning_rate: f32,
    margin: f32,
    weight_decay: f32,
    hard_negative_candidates: usize,
    seed: u64,
    resume: bool,
}

impl Default for EncoderTrainingConfig {
    fn default() -> Self {
        Self {
            workspace_root: None,
            output_path: None,
            checkpoint_dir: None,
            epochs: 6,
            steps_per_epoch: 120,
            samples_per_step: 6,
            checkpoint_every: 12,
            learning_rate: 0.008,
            margin: 0.12,
            weight_decay: 0.0001,
            hard_negative_candidates: 12,
            seed: 0x434f_4e56_5452_4149,
            resume: true,
        }
    }
}

#[derive(Debug, Clone)]
struct EncoderTrainingPaths {
    workspace_root: PathBuf,
    output_weights: PathBuf,
    checkpoint_dir: PathBuf,
    checkpoint_weights: PathBuf,
    checkpoint_state: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncoderTrainingState {
    epoch: usize,
    step_in_epoch: usize,
    global_step: usize,
    seed: u64,
    last_loss: Option<f32>,
    best_loss: Option<f32>,
}

impl EncoderTrainingState {
    fn fresh(seed: u64) -> Self {
        Self {
            epoch: 0,
            step_in_epoch: 0,
            global_step: 0,
            seed,
            last_loss: None,
            best_loss: None,
        }
    }
}

#[derive(Debug)]
struct EncoderTrainingFixture {
    workspace: WorkspaceSnapshot,
    config: AppConfig,
    map: GrayImage,
    color_map: RgbaImage,
    color_pyramid: ColorMapPyramid,
    masks: MaskSet,
    min_center: u32,
    max_x: u32,
    max_y: u32,
}

#[derive(Debug)]
struct PreparedTrainingTemplate<B>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    mask: Tensor<B, 4>,
    weighted_template: Tensor<B, 4>,
    template_energy: Tensor<B, 1>,
}

#[derive(Debug, Clone, Default)]
struct TrainingSampleStats {
    global_positive: f32,
    global_negative: f32,
    coarse_positive: f32,
    coarse_negative: f32,
    local_positive: f32,
    local_negative: f32,
}

#[derive(Debug, Clone, Default)]
struct TrainingStepAverages {
    samples: usize,
    global_positive: f32,
    global_negative: f32,
    coarse_positive: f32,
    coarse_negative: f32,
    local_positive: f32,
    local_negative: f32,
}

impl TrainingStepAverages {
    fn push(&mut self, sample: &TrainingSampleStats) {
        self.samples += 1;
        self.global_positive += sample.global_positive;
        self.global_negative += sample.global_negative;
        self.coarse_positive += sample.coarse_positive;
        self.coarse_negative += sample.coarse_negative;
        self.local_positive += sample.local_positive;
        self.local_negative += sample.local_negative;
    }

    fn scale(&self, value: f32) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            value / self.samples as f32
        }
    }
}

#[derive(Debug, Clone)]
struct TrainingRng {
    state: u64,
}

impl TrainingRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn range_u32(&mut self, start: u32, end: u32) -> u32 {
        if end <= start {
            return start;
        }
        start + self.next_u32() % (end - start + 1)
    }

    fn range_i32(&mut self, start: i32, end: i32) -> i32 {
        if end <= start {
            return start;
        }
        start + (self.next_u32() % (end - start + 1) as u32) as i32
    }
}

pub fn run_encoder_training_cli(args: Vec<OsString>) -> Result<()> {
    if args
        .iter()
        .any(|arg| matches!(arg.to_string_lossy().as_ref(), "--help" | "-h"))
    {
        print_encoder_training_usage();
        return Ok(());
    }

    let config = parse_encoder_training_args(args)?;
    train_convolution_encoder(config)
}

fn print_encoder_training_usage() {
    println!(
        "用法: game-map-tracker-rs train-encoder [workspace_root] [选项]\n\
         选项:\n\
         \t--workspace <path>                指定训练数据工作区根目录，默认使用 app 运行目录\n\
         \t--output <path>                   最终/持续导出的 safetensors，默认写入仓库 models/tracker_encoder.safetensors\n\
         \t--checkpoint-dir <path>           checkpoint 目录，默认写入仓库 models/checkpoints/tracker-encoder\n\
         \t--epochs <n>                      训练轮数，默认 6\n\
         \t--steps-per-epoch <n>             每轮步数，默认 120\n\
         \t--samples-per-step <n>            每步样本数，默认 6\n\
         \t--checkpoint-every <n>            每多少步持续落盘一次，默认 12\n\
         \t--learning-rate <f32>             学习率，默认 0.008\n\
         \t--margin <f32>                    排序间隔损失边距，默认 0.12\n\
         \t--weight-decay <f32>              L2 正则权重，默认 0.0001\n\
         \t--hard-negative-candidates <n>    每次挑选的硬负样本候选数，默认 12\n\
         \t--seed <u64>                      随机种子\n\
         \t--fresh                           忽略现有 checkpoint，从当前编码器起点重新训练"
    );
}

fn parse_encoder_training_args(args: Vec<OsString>) -> Result<EncoderTrainingConfig> {
    let mut config = EncoderTrainingConfig::default();
    let mut index = 0usize;

    while index < args.len() {
        let arg = args[index].to_string_lossy().into_owned();
        let mut next_value = |name: &str| -> Result<String> {
            let value = args
                .get(index + 1)
                .map(|value| value.to_string_lossy().into_owned())
                .with_context(|| format!("missing value for {name}"))?;
            index += 1;
            Ok(value)
        };

        match arg.as_str() {
            "--workspace" => {
                config.workspace_root = Some(PathBuf::from(next_value("--workspace")?))
            }
            "--output" => config.output_path = Some(PathBuf::from(next_value("--output")?)),
            "--checkpoint-dir" => {
                config.checkpoint_dir = Some(PathBuf::from(next_value("--checkpoint-dir")?))
            }
            "--epochs" => config.epochs = next_value("--epochs")?.parse()?,
            "--steps-per-epoch" => {
                config.steps_per_epoch = next_value("--steps-per-epoch")?.parse()?
            }
            "--samples-per-step" => {
                config.samples_per_step = next_value("--samples-per-step")?.parse()?
            }
            "--checkpoint-every" => {
                config.checkpoint_every = next_value("--checkpoint-every")?.parse()?
            }
            "--learning-rate" => config.learning_rate = next_value("--learning-rate")?.parse()?,
            "--margin" => config.margin = next_value("--margin")?.parse()?,
            "--weight-decay" => config.weight_decay = next_value("--weight-decay")?.parse()?,
            "--hard-negative-candidates" => {
                config.hard_negative_candidates =
                    next_value("--hard-negative-candidates")?.parse()?
            }
            "--seed" => config.seed = next_value("--seed")?.parse()?,
            "--fresh" => config.resume = false,
            value if value.starts_with("--") => {
                crate::bail!("unknown train-encoder option: {value}")
            }
            value => {
                if config.workspace_root.is_some() {
                    crate::bail!("unexpected positional argument: {value}");
                }
                config.workspace_root = Some(PathBuf::from(value));
            }
        }

        index += 1;
    }

    if config.epochs == 0
        || config.steps_per_epoch == 0
        || config.samples_per_step == 0
        || config.checkpoint_every == 0
        || config.hard_negative_candidates == 0
    {
        crate::bail!("train-encoder numeric options must be > 0");
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        crate::bail!("--learning-rate must be a finite positive number");
    }
    if !config.margin.is_finite() || config.margin <= 0.0 {
        crate::bail!("--margin must be a finite positive number");
    }
    if !config.weight_decay.is_finite() || config.weight_decay < 0.0 {
        crate::bail!("--weight-decay must be a finite non-negative number");
    }

    Ok(config)
}

fn resolve_encoder_training_paths(config: &EncoderTrainingConfig) -> Result<EncoderTrainingPaths> {
    let workspace_root = match config.workspace_root.clone() {
        Some(path) if path.is_absolute() => path,
        Some(path) => std::env::current_dir()?.join(path),
        None => WorkspaceBootstrap::prepare()?.workspace_root,
    };
    let output_root = repository_root().unwrap_or(std::env::current_dir()?);
    let output_weights = config
        .output_path
        .clone()
        .map(|path| normalize_training_path(&output_root, path))
        .unwrap_or_else(|| {
            output_root
                .join("models")
                .join("tracker_encoder.safetensors")
        });
    let checkpoint_dir = config
        .checkpoint_dir
        .clone()
        .map(|path| normalize_training_path(&output_root, path))
        .unwrap_or_else(|| {
            output_root
                .join("models")
                .join("checkpoints")
                .join("tracker-encoder")
        });

    Ok(EncoderTrainingPaths {
        workspace_root,
        checkpoint_weights: checkpoint_dir.join("checkpoint-latest.safetensors"),
        checkpoint_state: checkpoint_dir.join("checkpoint-state.json"),
        checkpoint_dir,
        output_weights,
    })
}

fn normalize_training_path(workspace_root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
}

fn build_encoder_training_fixture(workspace_root: &Path) -> Result<EncoderTrainingFixture> {
    let workspace = WorkspaceSnapshot::load(workspace_root.to_path_buf())?;
    let config = workspace.config.clone();
    let raw_map = load_logic_map_with_tracking_poi_scaled_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        config.view_size,
    )?;
    let color_map = load_logic_map_with_tracking_poi_scaled_rgba_image(
        &workspace.assets.bwiki_cache_dir,
        1,
        config.view_size,
    )?;
    let color_pyramid = load_logic_color_map_pyramid(&workspace)?;
    let map = imageproc::contrast::equalize_histogram(&raw_map);
    let min_center = align_training_point(config.view_size / 2 + 32, 4);
    let max_x = align_training_point(map.width().saturating_sub(config.view_size / 2 + 32), 4);
    let max_y = align_training_point(map.height().saturating_sub(config.view_size / 2 + 32), 4);
    let masks = build_template_masks(&config);

    Ok(EncoderTrainingFixture {
        workspace,
        config,
        map,
        color_map,
        color_pyramid,
        masks,
        min_center,
        max_x,
        max_y,
    })
}

fn align_training_point(value: u32, step: u32) -> u32 {
    value / step.max(1) * step.max(1)
}

fn local_texture_radius(view_size: u32) -> u32 {
    (view_size / 2).clamp(96, 180)
}

fn local_texture_score(image: &GrayImage, center_x: u32, center_y: u32, radius: u32) -> u64 {
    let left = center_x.saturating_sub(radius);
    let top = center_y.saturating_sub(radius);
    let right = (center_x + radius).min(image.width().saturating_sub(1));
    let bottom = (center_y + radius).min(image.height().saturating_sub(1));
    let mut total = 0u64;

    for y in top..=bottom {
        for x in left..=right {
            let center = i32::from(image.get_pixel(x, y).0[0]);
            let right_value = i32::from(image.get_pixel((x + 1).min(right), y).0[0]);
            let down_value = i32::from(image.get_pixel(x, (y + 1).min(bottom)).0[0]);
            total += (center - right_value).unsigned_abs() as u64;
            total += (center - down_value).unsigned_abs() as u64;
        }
    }

    total
}

fn sample_textured_world_point(
    fixture: &EncoderTrainingFixture,
    rng: &mut TrainingRng,
) -> (u32, u32) {
    let radius = local_texture_radius(fixture.config.view_size);
    let mut best = (fixture.min_center, fixture.min_center);
    let mut best_score = 0u64;
    for _ in 0..TRAINING_POINT_ATTEMPTS {
        let candidate = (
            align_training_point(rng.range_u32(fixture.min_center, fixture.max_x), 4),
            align_training_point(rng.range_u32(fixture.min_center, fixture.max_y), 4),
        );
        let score = local_texture_score(&fixture.map, candidate.0, candidate.1, radius);
        if score >= best_score {
            best = candidate;
            best_score = score;
        }
    }
    best
}

fn clamp_world_point(fixture: &EncoderTrainingFixture, point: (i32, i32)) -> (u32, u32) {
    (
        align_training_point(
            point
                .0
                .clamp(fixture.min_center as i32, fixture.max_x as i32) as u32,
            4,
        ),
        align_training_point(
            point
                .1
                .clamp(fixture.min_center as i32, fixture.max_y as i32) as u32,
            4,
        ),
    )
}

fn sample_local_world_point(
    fixture: &EncoderTrainingFixture,
    rng: &mut TrainingRng,
    anchor: (u32, u32),
) -> (u32, u32) {
    let radius = local_texture_radius(fixture.config.view_size);
    let mut best = anchor;
    let mut best_score = local_texture_score(&fixture.map, anchor.0, anchor.1, radius);
    for _ in 0..TRAINING_POINT_ATTEMPTS {
        let dx = rng.range_i32(
            -(TRAINING_LOCAL_STEP_MAX as i32),
            TRAINING_LOCAL_STEP_MAX as i32,
        );
        let dy = rng.range_i32(
            -(TRAINING_LOCAL_STEP_MAX as i32),
            TRAINING_LOCAL_STEP_MAX as i32,
        );
        if dx.unsigned_abs().max(dy.unsigned_abs()) < TRAINING_LOCAL_STEP_MIN {
            continue;
        }
        let candidate = clamp_world_point(fixture, (anchor.0 as i32 + dx, anchor.1 as i32 + dy));
        let score = local_texture_score(&fixture.map, candidate.0, candidate.1, radius);
        if score >= best_score {
            best = candidate;
            best_score = score;
        }
    }
    best
}

fn crop_centered_training_patch(
    image: &RgbaImage,
    center: (u32, u32),
    width: u32,
    height: u32,
) -> Result<RgbaImage> {
    if image.width() < width || image.height() < height {
        crate::bail!("search image is smaller than the requested patch");
    }

    let left = center
        .0
        .saturating_sub(width / 2)
        .min(image.width() - width);
    let top = center
        .1
        .saturating_sub(height / 2)
        .min(image.height() - height);
    Ok(crop_imm(image, left, top, width, height).to_image())
}

fn crop_map_patch_at_world(
    map: &crate::tracking::vision::ScaledColorMap,
    world: (u32, u32),
    width: u32,
    height: u32,
) -> Result<RgbaImage> {
    let center = center_to_scaled(WorldPoint::new(world.0 as f32, world.1 as f32), map.scale);
    crop_centered_training_patch(&map.image, center, width, height)
}

fn world_distance(a: (u32, u32), b: (u32, u32)) -> u32 {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

fn random_far_world_point(
    fixture: &EncoderTrainingFixture,
    rng: &mut TrainingRng,
    avoid: (u32, u32),
    min_distance: u32,
) -> (u32, u32) {
    for _ in 0..TRAINING_POINT_ATTEMPTS.saturating_mul(4) {
        let candidate = sample_textured_world_point(fixture, rng);
        if world_distance(candidate, avoid) >= min_distance {
            return candidate;
        }
    }
    sample_textured_world_point(fixture, rng)
}

fn global_negative_candidates(
    fixture: &EncoderTrainingFixture,
    rng: &mut TrainingRng,
    positive: (u32, u32),
    count: usize,
) -> Vec<(u32, u32)> {
    let mut candidates = Vec::with_capacity(count);
    while candidates.len() < count {
        let candidate = match candidates.len() % 3 {
            0 => (
                align_training_point(rng.range_u32(fixture.min_center, fixture.max_x), 4),
                positive.1,
            ),
            1 => (
                positive.0,
                align_training_point(rng.range_u32(fixture.min_center, fixture.max_y), 4),
            ),
            _ => random_far_world_point(fixture, rng, positive, TRAINING_GLOBAL_NEGATIVE_DISTANCE),
        };
        if world_distance(candidate, positive) < TRAINING_GLOBAL_NEGATIVE_DISTANCE {
            continue;
        }
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn local_negative_candidates(
    fixture: &EncoderTrainingFixture,
    rng: &mut TrainingRng,
    positive: (u32, u32),
    count: usize,
) -> Vec<(u32, u32)> {
    let mut candidates = Vec::with_capacity(count);
    let radius = fixture
        .config
        .local_search
        .radius_px
        .max(TRAINING_LOCAL_STEP_MAX);
    while candidates.len() < count {
        let candidate = if candidates.len() % 4 == 3 {
            random_far_world_point(
                fixture,
                rng,
                positive,
                radius + TRAINING_GLOBAL_NEGATIVE_DISTANCE,
            )
        } else {
            let dx = rng.range_i32(-(radius as i32), radius as i32);
            let dy = rng.range_i32(-(radius as i32), radius as i32);
            clamp_world_point(fixture, (positive.0 as i32 + dx, positive.1 as i32 + dy))
        };
        let distance = world_distance(candidate, positive);
        if distance < TRAINING_LOCAL_NEGATIVE_MIN_DISTANCE {
            continue;
        }
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn select_hard_negative_world(
    map: &crate::tracking::vision::ScaledColorMap,
    template: &RgbaImage,
    mask: &GrayImage,
    candidates: &[(u32, u32)],
) -> Option<(u32, u32)> {
    candidates
        .iter()
        .copied()
        .filter_map(|world| {
            let score = scaled_color_score(
                map,
                WorldPoint::new(world.0 as f32, world.1 as f32),
                template,
                mask,
            )?;
            Some((score, world))
        })
        .max_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.0.cmp(&right.1.0))
                .then_with(|| left.1.1.cmp(&right.1.1))
        })
        .map(|(_, world)| world)
}

fn prepare_training_template<B>(
    encoder: &FixedFeatureEncoder<B>,
    image: &RgbaImage,
    mask: &GrayImage,
) -> Result<PreparedTrainingTemplate<B>>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    let mask = mask_tensor::<B>(mask, encoder.output_channels(), &encoder.device);
    let template_features = encoder.encode(image)?;
    let weighted_template = template_features * mask.clone();
    let template_energy = weighted_template.clone().powi_scalar(2).sum();
    Ok(PreparedTrainingTemplate {
        mask,
        weighted_template,
        template_energy,
    })
}

fn score_patch_against_template<B>(
    encoder: &FixedFeatureEncoder<B>,
    prepared: &PreparedTrainingTemplate<B>,
    patch: &RgbaImage,
) -> Result<Tensor<B, 1>>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    let patch_features = encoder.encode(patch)?;
    let numerator = (patch_features.clone() * prepared.weighted_template.clone()).sum();
    let patch_energy = (patch_features.powi_scalar(2) * prepared.mask.clone()).sum();
    Ok(numerator / (patch_energy * prepared.template_energy.clone() + 1e-6).sqrt())
}

fn tensor_scalar_f32<B>(tensor: Tensor<B, 1>) -> Result<f32>
where
    B: Backend<FloatElem = f32>,
{
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| crate::app_error!("failed to convert tensor into f32 scalar: {error}"))?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| crate::app_error!("tensor scalar conversion returned an empty buffer"))
}

fn ranking_loss<B>(
    device: &B::Device,
    positive: Tensor<B, 1>,
    negative: Tensor<B, 1>,
    margin: f32,
) -> Tensor<B, 1>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    relu(Tensor::<B, 1>::from_data([margin], device) + negative - positive)
}

fn positive_pull_loss<B>(device: &B::Device, positive: Tensor<B, 1>) -> Tensor<B, 1>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    relu(Tensor::<B, 1>::ones([1], device) - positive)
}

fn compute_training_sample_loss(
    fixture: &EncoderTrainingFixture,
    encoder: &FixedFeatureEncoder<TrainingBackend>,
    rng: &mut TrainingRng,
    margin: f32,
    hard_negative_candidates: usize,
) -> Result<(Tensor<TrainingBackend, 1>, TrainingSampleStats)> {
    let global_world = sample_textured_world_point(fixture, rng);
    let local_world = sample_local_world_point(fixture, rng, global_world);

    let global_capture =
        synthesize_runtime_capture_rgba_from_map(&fixture.color_map, &fixture.config, global_world);
    let global_templates =
        prepare_color_capture_templates(&global_capture, &fixture.config, &fixture.color_pyramid);
    let local_capture =
        synthesize_runtime_capture_rgba_from_map(&fixture.color_map, &fixture.config, local_world);
    let local_templates =
        prepare_color_capture_templates(&local_capture, &fixture.config, &fixture.color_pyramid);

    let global_negatives =
        global_negative_candidates(fixture, rng, global_world, hard_negative_candidates);
    let local_negatives =
        local_negative_candidates(fixture, rng, local_world, hard_negative_candidates);
    let global_negative = select_hard_negative_world(
        &fixture.color_pyramid.global,
        &global_templates.global,
        &fixture.masks.global,
        &global_negatives,
    )
    .unwrap_or_else(|| {
        random_far_world_point(
            fixture,
            rng,
            global_world,
            TRAINING_GLOBAL_NEGATIVE_DISTANCE,
        )
    });
    let coarse_negative = select_hard_negative_world(
        &fixture.color_pyramid.coarse,
        &global_templates.coarse,
        &fixture.masks.coarse,
        &global_negatives,
    )
    .unwrap_or(global_negative);
    let local_negative = select_hard_negative_world(
        &fixture.color_pyramid.local,
        &local_templates.local,
        &fixture.masks.local,
        &local_negatives,
    )
    .unwrap_or_else(|| {
        random_far_world_point(fixture, rng, local_world, TRAINING_GLOBAL_NEGATIVE_DISTANCE)
    });

    let prepared_global =
        prepare_training_template(encoder, &global_templates.global, &fixture.masks.global)?;
    let prepared_coarse =
        prepare_training_template(encoder, &global_templates.coarse, &fixture.masks.coarse)?;
    let prepared_local =
        prepare_training_template(encoder, &local_templates.local, &fixture.masks.local)?;

    let positive_global_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.global,
        global_world,
        global_templates.global.width(),
        global_templates.global.height(),
    )?;
    let negative_global_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.global,
        global_negative,
        global_templates.global.width(),
        global_templates.global.height(),
    )?;
    let positive_coarse_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.coarse,
        global_world,
        global_templates.coarse.width(),
        global_templates.coarse.height(),
    )?;
    let negative_coarse_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.coarse,
        coarse_negative,
        global_templates.coarse.width(),
        global_templates.coarse.height(),
    )?;
    let positive_local_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.local,
        local_world,
        local_templates.local.width(),
        local_templates.local.height(),
    )?;
    let negative_local_patch = crop_map_patch_at_world(
        &fixture.color_pyramid.local,
        local_negative,
        local_templates.local.width(),
        local_templates.local.height(),
    )?;

    let positive_global_score =
        score_patch_against_template(encoder, &prepared_global, &positive_global_patch)?;
    let negative_global_score =
        score_patch_against_template(encoder, &prepared_global, &negative_global_patch)?;
    let positive_coarse_score =
        score_patch_against_template(encoder, &prepared_coarse, &positive_coarse_patch)?;
    let negative_coarse_score =
        score_patch_against_template(encoder, &prepared_coarse, &negative_coarse_patch)?;
    let positive_local_score =
        score_patch_against_template(encoder, &prepared_local, &positive_local_patch)?;
    let negative_local_score =
        score_patch_against_template(encoder, &prepared_local, &negative_local_patch)?;

    let global_loss = ranking_loss(
        &encoder.device,
        positive_global_score.clone(),
        negative_global_score.clone(),
        margin,
    ) + positive_pull_loss(&encoder.device, positive_global_score.clone()) * 0.20;
    let coarse_loss = ranking_loss(
        &encoder.device,
        positive_coarse_score.clone(),
        negative_coarse_score.clone(),
        margin,
    ) + positive_pull_loss(&encoder.device, positive_coarse_score.clone()) * 0.15;
    let local_loss = ranking_loss(
        &encoder.device,
        positive_local_score.clone(),
        negative_local_score.clone(),
        margin,
    ) + positive_pull_loss(&encoder.device, positive_local_score.clone()) * 0.25;
    let loss = global_loss * 0.35 + coarse_loss * 0.20 + local_loss * 0.45;

    let stats = TrainingSampleStats {
        global_positive: tensor_scalar_f32(positive_global_score.inner())?,
        global_negative: tensor_scalar_f32(negative_global_score.inner())?,
        coarse_positive: tensor_scalar_f32(positive_coarse_score.inner())?,
        coarse_negative: tensor_scalar_f32(negative_coarse_score.inner())?,
        local_positive: tensor_scalar_f32(positive_local_score.inner())?,
        local_negative: tensor_scalar_f32(negative_local_score.inner())?,
    };

    Ok((loss, stats))
}

fn apply_training_step(
    encoder: &mut FixedFeatureEncoder<TrainingBackend>,
    grads: &<TrainingBackend as AutodiffBackend>::Gradients,
    learning_rate: f32,
) -> Result<()> {
    let kernel_grad = encoder
        .edge_kernels
        .grad(grads)
        .ok_or_else(|| crate::app_error!("failed to read encoder kernel gradients"))?;
    let updated_kernel = encoder.edge_kernels.clone().inner() - kernel_grad * learning_rate;
    encoder.edge_kernels = Tensor::<TrainingBackend, 4>::from_inner(updated_kernel).require_grad();

    if let Some(bias) = encoder.edge_bias.take() {
        let updated_bias = if let Some(grad) = bias.grad(grads) {
            bias.clone().inner() - grad * learning_rate
        } else {
            bias.inner()
        };
        encoder.edge_bias =
            Some(Tensor::<TrainingBackend, 1>::from_inner(updated_bias).require_grad());
    }

    Ok(())
}

fn load_or_initialize_training_state(
    fixture: &EncoderTrainingFixture,
    paths: &EncoderTrainingPaths,
    config: &EncoderTrainingConfig,
) -> Result<(EncoderTrainingState, FixedFeatureEncoder<TrainingBackend>)> {
    let device = NdArrayDevice::default();
    let device_label = "NdArray Autodiff".to_owned();
    let (state, mut encoder) = if config.resume
        && paths.checkpoint_weights.is_file()
        && paths.checkpoint_state.is_file()
    {
        let state = serde_json::from_str::<EncoderTrainingState>(&fs::read_to_string(
            &paths.checkpoint_state,
        )?)
        .with_context(|| {
            format!(
                "failed to parse encoder checkpoint state {}",
                paths.checkpoint_state.display()
            )
        })?;
        let encoder = FixedFeatureEncoder::<TrainingBackend>::load_single_safetensors(
            &paths.checkpoint_weights,
            device,
            device_label,
        )?;
        (state, encoder)
    } else {
        (
            EncoderTrainingState::fresh(config.seed),
            FixedFeatureEncoder::<TrainingBackend>::new(&fixture.workspace, device, device_label)?,
        )
    };

    encoder.edge_kernels = encoder.edge_kernels.require_grad();
    if let Some(bias) = encoder.edge_bias.take() {
        encoder.edge_bias = Some(bias.require_grad());
    }

    Ok((state, encoder))
}

fn serialize_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn save_encoder_weights<B>(encoder: &FixedFeatureEncoder<B>, path: &Path) -> Result<()>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    let Some(parent) = path.parent() else {
        crate::bail!("invalid encoder weight path {}", path.display());
    };
    fs::create_dir_all(parent)?;

    let weight_shape: [usize; 4] = encoder.edge_kernels.shape().dims();
    let weight_values = encoder
        .edge_kernels
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| crate::app_error!("failed to serialize encoder kernels: {error}"))?;
    let weight_bytes = serialize_f32(&weight_values);
    let weight_view = TensorView::new(Dtype::F32, weight_shape.to_vec(), &weight_bytes)?;

    if let Some(bias) = encoder.edge_bias.as_ref() {
        let bias_shape: [usize; 1] = bias.shape().dims();
        let bias_values = bias
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| crate::app_error!("failed to serialize encoder bias: {error}"))?;
        let bias_bytes = serialize_f32(&bias_values);
        let bias_view = TensorView::new(Dtype::F32, bias_shape.to_vec(), &bias_bytes)?;
        serialize_to_file(
            vec![
                ("edge_bank.bias", bias_view),
                ("edge_bank.weight", weight_view),
            ],
            None,
            path,
        )?;
        return Ok(());
    }

    serialize_to_file(vec![("edge_bank.weight", weight_view)], None, path)?;
    Ok(())
}

fn save_training_checkpoint(
    paths: &EncoderTrainingPaths,
    state: &EncoderTrainingState,
    encoder: &FixedFeatureEncoder<TrainingBackend>,
) -> Result<()> {
    fs::create_dir_all(&paths.checkpoint_dir)?;
    save_encoder_weights(encoder, &paths.checkpoint_weights)?;
    save_encoder_weights(encoder, &paths.output_weights)?;
    fs::write(
        &paths.checkpoint_state,
        serde_json::to_string_pretty(state)?,
    )?;
    Ok(())
}

fn train_convolution_encoder(config: EncoderTrainingConfig) -> Result<()> {
    let paths = resolve_encoder_training_paths(&config)?;
    let fixture = build_encoder_training_fixture(&paths.workspace_root)?;
    let (mut state, mut encoder) = load_or_initialize_training_state(&fixture, &paths, &config)?;

    info!(
        workspace_root = %paths.workspace_root.display(),
        checkpoint = %paths.checkpoint_weights.display(),
        output = %paths.output_weights.display(),
        epochs = config.epochs,
        steps_per_epoch = config.steps_per_epoch,
        samples_per_step = config.samples_per_step,
        learning_rate = config.learning_rate,
        margin = config.margin,
        weight_decay = config.weight_decay,
        source = %encoder.source_label(),
        "starting convolution encoder training"
    );

    if state.epoch >= config.epochs {
        save_training_checkpoint(&paths, &state, &encoder)?;
        info!("encoder training already reached the requested epoch budget");
        return Ok(());
    }

    let mut rng = TrainingRng::new(
        state.seed ^ (state.global_step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    for epoch in state.epoch..config.epochs {
        let start_step = if epoch == state.epoch {
            state.step_in_epoch.min(config.steps_per_epoch)
        } else {
            0
        };

        for step in start_step..config.steps_per_epoch {
            let mut average = TrainingStepAverages::default();
            let mut total_loss = None;
            for _ in 0..config.samples_per_step {
                let (sample_loss, sample_stats) = compute_training_sample_loss(
                    &fixture,
                    &encoder,
                    &mut rng,
                    config.margin,
                    config.hard_negative_candidates,
                )?;
                average.push(&sample_stats);
                total_loss = Some(match total_loss {
                    Some(loss) => loss + sample_loss,
                    None => sample_loss,
                });
            }

            let mut loss =
                total_loss.expect("samples_per_step must be > 0") / config.samples_per_step as f32;
            if config.weight_decay > 0.0 {
                loss =
                    loss + encoder.edge_kernels.clone().powi_scalar(2).mean() * config.weight_decay;
                if let Some(bias) = encoder.edge_bias.as_ref() {
                    loss = loss + bias.clone().powi_scalar(2).mean() * config.weight_decay;
                }
            }

            let loss_scalar = tensor_scalar_f32(loss.clone().inner())?;
            let grads = loss.backward();
            apply_training_step(&mut encoder, &grads, config.learning_rate)?;

            state.epoch = epoch;
            state.step_in_epoch = step + 1;
            state.global_step += 1;
            state.last_loss = Some(loss_scalar);
            state.best_loss = Some(
                state
                    .best_loss
                    .map_or(loss_scalar, |best| best.min(loss_scalar)),
            );

            if state.global_step % config.checkpoint_every == 0 {
                save_training_checkpoint(&paths, &state, &encoder)?;
                info!(
                    epoch = epoch + 1,
                    step = step + 1,
                    global_step = state.global_step,
                    loss = loss_scalar,
                    global_pos = average.scale(average.global_positive),
                    global_neg = average.scale(average.global_negative),
                    coarse_pos = average.scale(average.coarse_positive),
                    coarse_neg = average.scale(average.coarse_negative),
                    local_pos = average.scale(average.local_positive),
                    local_neg = average.scale(average.local_negative),
                    checkpoint = %paths.checkpoint_weights.display(),
                    "encoder training checkpoint saved"
                );
            }
        }

        state.epoch = epoch + 1;
        state.step_in_epoch = 0;
        save_training_checkpoint(&paths, &state, &encoder)?;
        info!(
            epoch = epoch + 1,
            loss = state.last_loss.unwrap_or_default(),
            best_loss = state.best_loss.unwrap_or_default(),
            output = %paths.output_weights.display(),
            "encoder training epoch completed"
        );
    }

    info!(
        output = %paths.output_weights.display(),
        best_loss = state.best_loss.unwrap_or_default(),
        "convolution encoder training finished"
    );
    Ok(())
}

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
            ColorTemplateShape::InnerSquare,
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

fn build_template_masks(config: &AppConfig) -> MaskSet {
    let local_scale = config.template.local_downscale.max(1);
    let global_scale = config.template.global_downscale.max(local_scale);
    let coarse_scale = coarse_global_downscale(config);
    let local_size = scaled_dimension(config.view_size.max(1), local_scale);
    let global_size = scaled_dimension(config.view_size.max(1), global_scale);
    let coarse_size = scaled_dimension(config.view_size.max(1), coarse_scale);
    let annulus_inner = normalized_inner_radius(
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );

    MaskSet {
        local: build_mask(local_size, local_size, annulus_inner, 1.0),
        global: build_mask(global_size, global_size, annulus_inner, 1.0),
        coarse: build_mask(coarse_size, coarse_size, annulus_inner, 1.0),
    }
}

impl BurnFeatureMatcher {
    fn new(
        workspace: &WorkspaceSnapshot,
        config: &AiTrackingConfig,
        masks: &MaskSet,
    ) -> Result<Self> {
        let selection = select_burn_device(config)?;
        Self::from_selection(workspace, selection, masks)
    }

    fn from_selection(
        workspace: &WorkspaceSnapshot,
        selection: BurnDeviceSelection,
        masks: &MaskSet,
    ) -> Result<Self> {
        match selection {
            BurnDeviceSelection::Cpu => Ok(Self::NdArray(BurnFeatureMatcherBackend::<
                burn::backend::NdArray,
            >::new(
                workspace,
                NdArrayDevice::Cpu,
                "CPU".to_owned(),
                None,
                masks,
            )?)),
            #[cfg(burn_cuda_backend)]
            BurnDeviceSelection::Cuda(device) => Ok(Self::Cuda(BurnFeatureMatcherBackend::<
                burn::backend::Cuda,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Cuda(device)),
                Some(CUDA_CONV_IM2COL_BUDGET_BYTES),
                masks,
            )?)),
            #[cfg(burn_vulkan_backend)]
            BurnDeviceSelection::Vulkan(device) => Ok(Self::Vulkan(BurnFeatureMatcherBackend::<
                burn::backend::Vulkan,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Vulkan(device)),
                Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                masks,
            )?)),
            #[cfg(burn_metal_backend)]
            BurnDeviceSelection::Metal(device) => Ok(Self::Metal(BurnFeatureMatcherBackend::<
                burn::backend::Metal,
            >::new(
                workspace,
                device.clone(),
                burn_device_label(&BurnDeviceSelection::Metal(device)),
                Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                masks,
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
        let local = prepare_color_capture_template(
            captured,
            config.view_size,
            pyramid.local.scale,
            config.template.mask_inner_radius,
            config.template.mask_outer_radius,
            ColorTemplateShape::InnerSquare,
        );
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
        })
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

    #[allow(unreachable_patterns)]
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
            _ => crate::bail!("prepared template backend does not match matcher backend"),
        }
    }
}

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
        masks: &MaskSet,
    ) -> Result<Self> {
        let encoder = FixedFeatureEncoder::<B>::new(workspace, device, device_label)?;
        let local_mask = mask_tensor::<B>(&masks.local, encoder.output_channels(), &encoder.device);

        Ok(Self {
            encoder,
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
        self.locate_cached(&search, template, threshold, origin_x, origin_y, scale)
    }

    fn locate_cached(
        &self,
        search: &SearchTensorCache<B>,
        template: &BurnPreparedTemplate<B>,
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
        let search_patch_energy = conv2d(
            search.squared.clone(),
            template.mask.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
        let normalized = numerator / (search_patch_energy * template.template_energy + 1e-6).sqrt();

        let (best_score, best_left, best_top) =
            tensor_best_match(normalized, score_width, score_height)?;
        Ok(locate_result_from_best(
            best_score,
            best_left,
            best_top,
            threshold,
            origin_x,
            origin_y,
            scale,
            template.width(),
            template.height(),
        ))
    }
}

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
}

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

fn locate_cached_in_chunks<B>(
    search: &SearchTensorCache<B>,
    weighted_template: &Tensor<B, 4>,
    mask: &Tensor<B, 4>,
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
        let search_patch_energy = conv2d(
            squared_chunk,
            mask.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
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
        output_row += produced_rows.max(1);
    }

    Ok(locate_result_from_best(
        best_score,
        best_left,
        best_top,
        threshold,
        origin_x,
        origin_y,
        scale,
        template_width,
        template_height,
    ))
}

fn tensor4_to_flat_f32<B>(tensor: Tensor<B, 4>) -> Result<Vec<f32>>
where
    B: Backend<FloatElem = f32>,
{
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| crate::app_error!(error.to_string()))
}

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
        best_score: f32::MIN,
        accepted: None,
    }
}

fn locate_result_from_best(
    best_score: f32,
    best_left: u32,
    best_top: u32,
    threshold: f32,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
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
        best_score,
        accepted,
    }
}

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
            status.probe_summary = self.probe_summary(Some(sample));
            let estimate = self.apply_probe_absent_fallback(&mut status);
            status.locate_summary = estimate.as_ref().map_or_else(
                || "F1-P 标签探针未命中，已阻止定位".to_owned(),
                |estimate| {
                    format!(
                        "F1-P 标签探针未命中，已阻止定位，惯性保位 @ {:.0}, {:.0}",
                        estimate.world.x, estimate.world.y
                    )
                },
            );
            let debug = self
                .debug_enabled
                .then(|| self.build_probe_miss_debug_snapshot(sample, estimate.as_ref()));
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
            let template_candidate = self.template_global_locator.locate_capture(
                &captured_rgba,
                &self.config,
                &self.color_pyramid,
            )?;
            if let Some(candidate) = template_candidate.clone() {
                global_result = Some(LocateResult {
                    best_score: candidate.score,
                    accepted: Some(candidate.clone()),
                });
                refine_result = global_result.clone();
                status.source = Some(TrackingSource::FeatureEmbedding);
                status.match_score = Some(candidate.score);
                status.message = format!(
                    "卷积特征匹配全局重定位成功，RGB 特征得分 {:.3}，坐标 {:.0}, {:.0}。",
                    candidate.score, candidate.world.x, candidate.world.y
                );
                locate_summary = Self::locate_success_summary("全局", &candidate);
                estimate = Some(self.commit_success(candidate, TrackingSource::FeatureEmbedding));
            } else {
                global_result = Some(LocateResult {
                    best_score: f32::MIN,
                    accepted: None,
                });
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
                "{} m{:.3}/{:.3} n{:.3}",
                if sample.present {
                    "存在"
                } else {
                    "不存在"
                },
                sample.mean_raw_score,
                sample.threshold,
                sample.min_raw_score
            ),
            None => "探针未启用".to_owned(),
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

        if let Some(detector) = self.presence_detector.as_ref() {
            fields.extend(detector.debug_fields(sample));
        }
        if let Some(position) = estimate {
            fields.push(DebugField::new(
                "输出坐标",
                format!("{:.0}, {:.0}", position.world.x, position.world.y),
            ));
        }

        let images = self
            .presence_detector
            .as_ref()
            .map(|detector| detector.debug_images(sample))
            .unwrap_or_default();

        build_debug_snapshot(
            TrackerEngineKind::ConvolutionFeatureMatch,
            self.state.frame_index,
            self.state.stage,
            images,
            fields,
        )
    }
}

impl TrackingWorker for BurnTrackerWorker {
    fn refresh_interval(&self) -> Duration {
        Duration::from_millis(self.inner.config.ai.refresh_rate_ms)
    }

    fn tick(&mut self) -> Result<TrackingTick> {
        self.inner.run_frame()
    }

    fn set_debug_enabled(&mut self, enabled: bool) {
        self.inner.debug_enabled = enabled;
    }

    fn initial_status(&self) -> TrackingStatus {
        let message = format!(
            "卷积特征匹配引擎已就绪：设备 {}，可用后端 {}，{} + Burn 张量相似度搜索。",
            self.inner.matcher.device_label(),
            available_burn_backends(),
            self.inner.matcher.source_label()
        );

        TrackingStatus::new(TrackerEngineKind::ConvolutionFeatureMatch, message)
    }

    fn engine_kind(&self) -> TrackerEngineKind {
        TrackerEngineKind::ConvolutionFeatureMatch
    }
}

#[cfg(all(test, burn_vulkan_backend))]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::error::Result;
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
            stress_env_u32, stress_env_usize, timed, write_stress_report,
        },
    };
    use image::{GrayImage, RgbaImage};

    const MAX_ROUNDS: usize = 6;
    const GLOBAL_CASES_PER_ROUND: usize = 40;
    const LOCAL_STEPS_PER_CASE: usize = 6;
    const LOCAL_STEP_MIN: u32 = 28;
    const LOCAL_STEP_MAX: u32 = 112;
    const GLOBAL_TOLERANCE: f32 = 24.0;
    const LOCAL_TOLERANCE: f32 = 24.0;
    const TARGET_GLOBAL_ACCURACY: f32 = 0.95;

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

    fn min_rounds_before_success(max_rounds: usize) -> usize {
        stress_env_usize("GAME_MAP_TRACKER_STRESS_MIN_ROUNDS", 1).min(max_rounds.max(1))
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
        BurnFeatureMatcher::new(&fixture.workspace, &config, &fixture.masks)
            .expect("failed to create Vulkan feature matcher")
    }

    fn template_global_locator_for_vulkan(
        fixture: &TestFixture,
        ordinal: usize,
    ) -> TemplateGlobalLocator {
        let mut config = fixture.config.clone();
        config.template.device = AiDevicePreference::Vulkan;
        config.template.device_index = ordinal;
        TemplateGlobalLocator::new_cached(
            &fixture.workspace,
            &config,
            &fixture.color_pyramid,
            &fixture.map_cache_key,
        )
        .expect("failed to create template global locator")
    }

    fn within_tolerance(actual: Option<WorldPoint>, expected: (u32, u32), tolerance: f32) -> bool {
        actual.is_some_and(|actual| {
            (actual.x - expected.0 as f32).abs() <= tolerance
                && (actual.y - expected.1 as f32).abs() <= tolerance
        })
    }

    fn locate_global_runtime(
        fixture: &TestFixture,
        global_locator: &TemplateGlobalLocator,
        capture: &RgbaImage,
    ) -> Result<Option<MatchCandidate>> {
        global_locator.locate_capture(capture, &fixture.config, &fixture.color_pyramid)
    }

    fn simulate_runtime_frame(
        fixture: &TestFixture,
        matcher: &BurnFeatureMatcher,
        global_locator: &TemplateGlobalLocator,
        capture: &RgbaImage,
        state: &mut TrackerState,
    ) -> Result<FrameResult> {
        let templates =
            matcher.prepare_capture_templates(capture, &fixture.config, &fixture.color_pyramid)?;
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

        let global = locate_global_runtime(fixture, global_locator, capture)?;
        let best_score = global.as_ref().map(|candidate| candidate.score);
        let best_color = best_score;
        if let Some(candidate) = global {
            state.mark_success(candidate.world);
            return Ok(FrameResult {
                world: Some(candidate.world),
                score: Some(candidate.score),
                color_score: Some(candidate.score),
                source: Some(TrackingSource::FeatureEmbedding),
                note: "global_accept".to_owned(),
            });
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
        global_locator: &TemplateGlobalLocator,
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

            let capture = synthesize_runtime_capture_rgba_from_map(
                &fixture.color_map,
                &fixture.config,
                case.start,
            );
            let frame =
                simulate_runtime_frame(fixture, matcher, global_locator, &capture, &mut state)?;
            stats.global_total += 1;
            let global_aligned = within_tolerance(frame.world, case.start, GLOBAL_TOLERANCE);
            if global_aligned {
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
                stats.local_skipped += case.locals.len();
                continue;
            }

            for (step_index, target) in case.locals.iter().copied().enumerate() {
                let capture = synthesize_runtime_capture_rgba_from_map(
                    &fixture.color_map,
                    &fixture.config,
                    target,
                );
                let frame =
                    simulate_runtime_frame(fixture, matcher, global_locator, &capture, &mut state)?;
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
                    stats.local_skipped += case.locals.len().saturating_sub(step_index + 1);
                    break;
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
        let (global_locator, global_init_elapsed) =
            timed(|| template_global_locator_for_vulkan(fixture, ordinal));
        print_perf_ms("ai/vulkan", "template_global_init", global_init_elapsed);

        let mut best_global = 0.0f32;
        let mut best_local = 0.0f32;
        let mut best_overall = 0.0f32;
        let mut aggregate = StressRoundStats::default();

        let max_rounds = max_rounds();
        let min_rounds = min_rounds_before_success(max_rounds);
        for round in 0..max_rounds {
            let seed = 0x4149_5655_4c4b_414eu64.wrapping_add(round as u64 * 0x9e37_79b9);
            let (stats, elapsed) = timed(|| run_round(fixture, &matcher, &global_locator, seed));
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
            aggregate.global_total += stats.global_total;
            aggregate.global_success += stats.global_success;
            aggregate.local_total += stats.local_total;
            aggregate.local_success += stats.local_success;
            aggregate.local_skipped += stats.local_skipped;
            let aggregate_global = aggregate.global_accuracy();
            let aggregate_local = aggregate.local_accuracy();
            let aggregate_overall = aggregate.overall_accuracy();

            println!(
                "[ai/vulkan][round={}] global={:.2}% local={:.2}% overall={:.2}% agg_global={:.2}% agg_local={:.2}% agg_overall={:.2}% local_skipped={} elapsed_ms={:.0} failures={} report={}",
                round + 1,
                global_accuracy * 100.0,
                local_accuracy * 100.0,
                overall_accuracy * 100.0,
                aggregate_global * 100.0,
                aggregate_local * 100.0,
                aggregate_overall * 100.0,
                stats.local_skipped,
                elapsed.as_secs_f64() * 1000.0,
                stats.failures.len(),
                report_path.display()
            );

            if round + 1 >= min_rounds && aggregate_global >= TARGET_GLOBAL_ACCURACY {
                return Ok(());
            }
        }

        let aggregate_global = aggregate.global_accuracy();
        let aggregate_local = aggregate.local_accuracy();
        let aggregate_overall = aggregate.overall_accuracy();
        crate::bail!(
            "AI Vulkan global accuracy stayed below target after {} rounds (required minimum rounds before success: {}); target global >= {:.0}%, aggregate global/local/overall {:.2}%/{:.2}%/{:.2}%, best global {:.2}%, best local {:.2}%, best overall {:.2}%",
            max_rounds,
            min_rounds,
            TARGET_GLOBAL_ACCURACY * 100.0,
            aggregate_global * 100.0,
            aggregate_local * 100.0,
            aggregate_overall * 100.0,
            best_global * 100.0,
            best_local * 100.0,
            best_overall * 100.0
        )
    }
}
