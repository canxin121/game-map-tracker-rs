use std::{sync::Arc, time::Duration};

use tracing::info;

#[cfg(feature = "ai-burn")]
use std::cmp::Ordering;

use crate::config::TemplateTrackingConfig;
#[cfg(not(feature = "ai-burn"))]
use crate::tracking::vision::crop_around_center;
#[cfg(feature = "ai-burn")]
use crate::tracking::vision::{mask_as_unit_vec, rgba_image_as_unit_vec};
#[cfg(not(feature = "ai-burn"))]
use crate::tracking::{
    precompute::load_or_build_match_pyramid,
    vision::{
        MapPyramid, build_match_representation, capture_template_annulus, crop_search_region,
        load_logic_map_pyramid,
    },
};
use crate::{
    config::AppConfig,
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackerEngineKind, TrackingSource},
    },
    resources::WorkspaceSnapshot,
    tracking::{
        capture::{DesktopCapture, preprocess_capture},
        debug::{DebugField, TrackingDebugSnapshot},
        precompute::{clear_match_pyramid_caches, tracker_map_cache_key},
        presence::{MinimapPresenceDetector, MinimapPresenceSample},
        runtime::{TrackingStatus, TrackingTick, TrackingWorker},
        vision::{
            ColorCaptureTemplates, ColorMapPyramid, ColorTemplateShape, LocalCandidateDecision,
            MaskSet, MatchCandidate, SearchCrop, SearchStage, TrackerState, build_debug_snapshot,
            build_mask, capture_template_inner_square, capture_template_inner_square_rgba,
            center_to_scaled, coarse_global_downscale, crop_search_region_rgba,
            load_logic_color_map_pyramid, local_candidate_decision, normalized_inner_radius,
            prepare_color_capture_template, preview_image, scaled_color_score, scaled_dimension,
            search_region_around_center, top_score_peaks,
        },
    },
};
use anyhow::Result;
#[cfg(feature = "ai-burn")]
use burn::{
    backend::ndarray::NdArrayDevice,
    tensor::{
        Tensor, TensorData, backend::Backend, cast::ToElement, module::conv2d, ops::ConvOptions,
    },
};
use image::{GrayImage, RgbaImage};

#[cfg(feature = "ai-burn")]
use crate::tracking::precompute::{
    PersistedTensorCache, clear_tensor_caches_by_prefix, load_tensor_cache, save_tensor_cache,
    tracker_tensor_cache_path,
};

#[cfg(feature = "ai-burn")]
use crate::tracking::burn_support::{
    BurnDeviceSelection, available_burn_backends, burn_device_label,
    burn_score_map_capture_enabled, select_burn_device,
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

#[cfg(feature = "ai-burn")]
const TEMPLATE_COARSE_THRESHOLD_RELAX_MARGIN: f32 = 0.12;
#[cfg(feature = "ai-burn")]
const TEMPLATE_COARSE_PEAK_MARGIN: f32 = 0.03;
#[cfg(feature = "ai-burn")]
const TEMPLATE_MIN_COARSE_CANDIDATE_THRESHOLD: f32 = 0.18;
#[cfg(feature = "ai-burn")]
const TEMPLATE_RERANK_FINAL_SCORE_WEIGHT: f32 = 0.35;
#[cfg(feature = "ai-burn")]
const TEMPLATE_RERANK_GLOBAL_SCORE_WEIGHT: f32 = 0.10;
#[cfg(feature = "ai-burn")]
const TEMPLATE_RERANK_GLOBAL_COLOR_WEIGHT: f32 = 0.20;
#[cfg(feature = "ai-burn")]
const TEMPLATE_RERANK_LOCAL_COLOR_WEIGHT: f32 = 0.30;
#[cfg(feature = "ai-burn")]
const TEMPLATE_RERANK_COARSE_WEIGHT: f32 = 0.05;

#[cfg(feature = "ai-burn")]
#[derive(Debug, Clone)]
struct TemplateRefinedCandidate {
    result: LocateResult,
    coarse_score: f32,
    global_score: f32,
    global_color_score: f32,
    local_color_score: f32,
    rerank_score: f32,
}

pub struct TemplateTrackerWorker {
    config: AppConfig,
    capture: DesktopCapture,
    presence_detector: Option<MinimapPresenceDetector>,
    #[cfg(not(feature = "ai-burn"))]
    pyramid: MapPyramid,
    #[cfg(feature = "ai-burn")]
    color_pyramid: ColorMapPyramid,
    state: TrackerState,
    debug_enabled: bool,
    #[cfg(feature = "ai-burn")]
    matcher: TemplateMatcher,
}

#[cfg(feature = "ai-burn")]
enum TemplateMatcher {
    NdArray(BurnTemplateMatcher<burn::backend::NdArray>),
    #[cfg(burn_cuda_backend)]
    Cuda(BurnTemplateMatcher<burn::backend::Cuda>),
    #[cfg(burn_vulkan_backend)]
    Vulkan(BurnTemplateMatcher<burn::backend::Vulkan>),
    #[cfg(burn_metal_backend)]
    Metal(BurnTemplateMatcher<burn::backend::Metal>),
}

#[cfg(feature = "ai-burn")]
struct BurnTemplateMatcher<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    device: B::Device,
    device_label: String,
    coarse_search: SearchTensorCache<B>,
    global_mask_squared: Tensor<B, 4>,
    coarse_mask_squared: Tensor<B, 4>,
    coarse_search_patch_energy: Tensor<B, 4>,
    local_mask_squared: Tensor<B, 4>,
    chunk_budget_bytes: Option<usize>,
}

#[cfg(feature = "ai-burn")]
struct SearchTensorCache<B>
where
    B: Backend<FloatElem = f32>,
{
    image: Tensor<B, 4>,
    squared: Tensor<B, 4>,
    width: u32,
    height: u32,
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
    mask_squared: Tensor<B, 4>,
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

#[cfg(feature = "ai-burn")]
const CUDA_CONV_IM2COL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

#[cfg(feature = "ai-burn")]
const WGPU_CONV_IM2COL_BUDGET_BYTES: usize = 128 * 1024 * 1024;

impl TemplateTrackerWorker {
    pub fn new(workspace: Arc<WorkspaceSnapshot>) -> Result<Self> {
        info!(
            cache_root = %workspace.assets.bwiki_cache_dir.display(),
            view_size = workspace.config.view_size,
            device = %workspace.config.template.device,
            device_index = workspace.config.template.device_index,
            "initializing template tracker worker"
        );
        let config = workspace.config.clone();
        if !config.minimap.is_configured() {
            anyhow::bail!("小地图区域尚未配置，请先完成小地图取区");
        }
        let capture = DesktopCapture::from_absolute_region(&config.minimap)?;
        let presence_detector = MinimapPresenceDetector::new(workspace.as_ref())?;
        #[cfg(feature = "ai-burn")]
        let cache_key = tracker_map_cache_key(workspace.as_ref())?;
        #[cfg(not(feature = "ai-burn"))]
        let pyramid = load_logic_map_pyramid(workspace.as_ref())?.0;
        #[cfg(feature = "ai-burn")]
        let color_pyramid = load_logic_color_map_pyramid(workspace.as_ref())?;
        let masks = build_template_masks(&config);

        #[cfg(feature = "ai-burn")]
        let matcher = TemplateMatcher::new_cached(
            workspace.as_ref(),
            &config.template,
            &color_pyramid,
            &masks,
            &cache_key,
        )?;

        Ok(Self {
            config,
            capture,
            presence_detector,
            #[cfg(not(feature = "ai-burn"))]
            pyramid,
            #[cfg(feature = "ai-burn")]
            color_pyramid,
            state: TrackerState::default(),
            debug_enabled: false,
            #[cfg(feature = "ai-burn")]
            matcher,
        })
    }

    #[cfg(feature = "ai-burn")]
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
                let result = self.matcher.locate_local_prepared(
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
                            status.source = Some(TrackingSource::LocalTrack);
                            status.match_score = Some(candidate.score);
                            status.message = format!(
                                "局部模板锁定成功，RGB 得分 {:.3}，坐标 {:.0}, {:.0}。",
                                candidate.score, candidate.world.x, candidate.world.y
                            );
                            locate_summary = Self::locate_success_summary("局部", &candidate);
                            estimate =
                                Some(self.commit_success(candidate, TrackingSource::LocalTrack));
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
                            "小地图恢复后局部模板候选跳变 {:.0}，超过阈值 {}，切回全局重定位。",
                            jump, self.config.local_search.reacquire_jump_threshold_px
                        );
                    } else {
                        let switched = self
                            .state
                            .increment_local_fail(self.config.local_search.lock_fail_threshold);
                        status.message = format!(
                            "局部模板锁定失败，第 {} 次重试。",
                            self.state.local_fail_streak
                        );
                        if switched {
                            status.message = "局部锁定连续失败，切回全局重定位。".to_owned();
                        }
                    }
                }
            }
        }

        if estimate.is_none() {
            let result = self.locate_global_template(
                &templates.coarse,
                &templates.global,
                &templates.local,
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
                let refine = self.matcher.locate_local_prepared(
                    &crop.image,
                    &templates.local,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.color_pyramid.local.scale,
                )?;
                refine_result = Some(refine.clone());
                if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                    status.source = Some(TrackingSource::GlobalRelocate);
                    status.match_score = Some(candidate.score);
                    status.message = format!(
                        "全局重定位成功，RGB 得分 {:.3}，坐标 {:.0}, {:.0}。",
                        candidate.score, candidate.world.x, candidate.world.y
                    );
                    locate_summary = Self::locate_success_summary("全局", &candidate);
                    estimate = Some(self.commit_success(candidate, TrackingSource::GlobalRelocate));
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
            status.message = "当前帧未找到可靠匹配，等待下一帧。".to_owned();
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

    #[cfg(not(feature = "ai-burn"))]
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

        let captured = self.capture.capture_gray()?;
        let (local_template, global_template, coarse_template) =
            prepare_capture_templates(&captured, &self.config, &self.pyramid);

        let mut status = self.base_status();
        status.probe_summary = self.probe_summary(probe_sample.as_ref());
        let mut estimate = None;
        let mut global_result = None;
        let mut refine_crop = None;
        let mut refine_result = None;
        let mut local_result = None;
        let mut locate_summary = "等待定位".to_owned();

        if self.config.local_search.enabled && matches!(self.state.stage, SearchStage::LocalTrack) {
            if let Some(last_world) = self.state.last_world {
                let crop = crop_around_center(
                    &self.pyramid.local.image,
                    center_to_scaled(last_world, self.pyramid.local.scale),
                    self.config.local_search.radius_px / self.pyramid.local.scale.max(1),
                    local_template.width(),
                    local_template.height(),
                )?;
                let result = self.locate_local_template(
                    &crop.image,
                    &local_template,
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
                    match local_candidate_decision(
                        last_world,
                        candidate.world,
                        self.config.local_search.max_accepted_jump_px,
                        self.state.reacquire_anchor,
                        self.config.local_search.reacquire_jump_threshold_px,
                    ) {
                        LocalCandidateDecision::Accept => {
                            status.source = Some(TrackingSource::LocalTrack);
                            status.match_score = Some(candidate.score);
                            status.message = format!(
                                "局部模板锁定成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                                candidate.score, candidate.world.x, candidate.world.y
                            );
                            locate_summary = Self::locate_success_summary("局部", &candidate);
                            estimate =
                                Some(self.commit_success(candidate, TrackingSource::LocalTrack));
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
                            "小地图恢复后局部模板候选跳变 {:.0}，超过阈值 {}，切回全局重定位。",
                            jump, self.config.local_search.reacquire_jump_threshold_px
                        );
                    } else {
                        let switched = self
                            .state
                            .increment_local_fail(self.config.local_search.lock_fail_threshold);
                        status.message = format!(
                            "局部模板锁定失败，第 {} 次重试。",
                            self.state.local_fail_streak
                        );
                        if switched {
                            status.message = "局部锁定连续失败，切回全局重定位。".to_owned();
                        }
                    }
                }
            }
        }

        if estimate.is_none() {
            let result =
                self.locate_global_template(&coarse_template, &global_template, &local_template)?;
            global_result = Some(result.clone());

            if let Some(coarse) = result.accepted {
                let crop = crop_around_center(
                    &self.pyramid.local.image,
                    center_to_scaled(coarse.world, self.pyramid.local.scale),
                    self.config.template.global_refine_radius_px / self.pyramid.local.scale.max(1),
                    local_template.width(),
                    local_template.height(),
                )?;
                let refine = self.locate_local_template(
                    &crop.image,
                    &local_template,
                    self.config.template.global_match_threshold,
                    crop.origin_x,
                    crop.origin_y,
                    self.pyramid.local.scale,
                )?;

                refine_crop = Some(crop.clone());
                refine_result = Some(refine.clone());
                if let Some(candidate) = refine.accepted.or(Some(coarse)) {
                    status.source = Some(TrackingSource::GlobalRelocate);
                    status.match_score = Some(candidate.score);
                    status.message = format!(
                        "全局重定位成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                        candidate.score, candidate.world.x, candidate.world.y
                    );
                    locate_summary = Self::locate_success_summary("全局", &candidate);
                    estimate = Some(self.commit_success(candidate, TrackingSource::GlobalRelocate));
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
            status.message = "当前帧未找到可靠匹配，等待下一帧。".to_owned();
        }
        status.locate_summary = locate_summary;

        let debug = self.debug_enabled.then(|| {
            self.build_debug_snapshot(
                &captured,
                &global_template,
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

    #[cfg(feature = "ai-burn")]
    fn locate_global_template(
        &self,
        coarse_template: &PreparedTemplate,
        global_template: &PreparedTemplate,
        local_template: &PreparedTemplate,
    ) -> Result<LocateResult> {
        locate_global_template_runtime(
            &self.matcher,
            &self.color_pyramid,
            &self.config.template,
            coarse_template,
            global_template,
            local_template,
        )
    }

    #[cfg(not(feature = "ai-burn"))]
    fn locate_global_template(
        &self,
        coarse_template: &GrayImage,
        global_template: &GrayImage,
        local_template: &GrayImage,
    ) -> Result<LocateResult> {
        let _ = (coarse_template, global_template, local_template);
        anyhow::bail!("模板匹配引擎当前二进制未启用 `ai-burn` 特性")
    }

    #[cfg(not(feature = "ai-burn"))]
    fn locate_local_template(
        &self,
        image: &GrayImage,
        template: &GrayImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let _ = (image, template, threshold, origin_x, origin_y, scale);
        anyhow::bail!("模板匹配引擎当前二进制未启用 `ai-burn` 特性")
    }

    fn base_status(&self) -> TrackingStatus {
        TrackingStatus {
            engine: TrackerEngineKind::MultiScaleTemplateMatch,
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
            "进入惯性保位，第 {} / {} 帧。",
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
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
            DebugField::new(
                "最后坐标",
                self.state.last_world.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
            DebugField::new(
                "重获锚点",
                self.state.reacquire_anchor.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
        ];

        #[cfg(feature = "ai-burn")]
        fields.push(DebugField::new("设备", self.matcher.device_label()));

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
                "输出来源",
                format!("{} / {}", position.source, self.engine_kind()),
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
            self.engine_kind(),
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
            DebugField::new("局部失败", self.state.local_fail_streak.to_string()),
            DebugField::new("丢失帧", self.state.lost_frames.to_string()),
            DebugField::new(
                "最后坐标",
                self.state.last_world.map_or_else(
                    || "--".to_owned(),
                    |world| format!("{:.0}, {:.0}", world.x, world.y),
                ),
            ),
        ];

        #[cfg(feature = "ai-burn")]
        fields.push(DebugField::new("设备", self.matcher.device_label()));

        if let Some(detector) = self.presence_detector.as_ref() {
            fields.extend(detector.debug_fields(sample));
        }
        if let Some(position) = estimate {
            fields.push(DebugField::new(
                "输出来源",
                format!("{} / {}", position.source, self.engine_kind()),
            ));
        }

        let images = self
            .presence_detector
            .as_ref()
            .map(|detector| detector.debug_images(sample))
            .unwrap_or_default();

        build_debug_snapshot(
            self.engine_kind(),
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
fn coarse_peak_threshold(result: &LocateResult, config: &TemplateTrackingConfig) -> f32 {
    let relaxed_floor = (config.global_match_threshold - TEMPLATE_COARSE_THRESHOLD_RELAX_MARGIN)
        .max(TEMPLATE_MIN_COARSE_CANDIDATE_THRESHOLD);
    (result.best_score - TEMPLATE_COARSE_PEAK_MARGIN)
        .clamp(relaxed_floor, config.global_match_threshold)
}

#[cfg(feature = "ai-burn")]
fn match_candidate_from_result(
    result: &LocateResult,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    template_width: u32,
    template_height: u32,
) -> Option<MatchCandidate> {
    (result.score_width > 0 && result.score_height > 0).then(|| MatchCandidate {
        world: WorldPoint::new(
            (origin_x + result.best_left + template_width / 2) as f32 * scale as f32,
            (origin_y + result.best_top + template_height / 2) as f32 * scale as f32,
        ),
        score: result.best_score,
    })
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
            coarse_peak_threshold(result, config),
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

    for candidate in [
        result.accepted.clone(),
        match_candidate_from_result(
            result,
            origin_x,
            origin_y,
            scale,
            template_width,
            template_height,
        ),
    ]
    .into_iter()
    .flatten()
    {
        let already_present = candidates.iter().any(|existing| {
            existing.world.x == candidate.world.x && existing.world.y == candidate.world.y
        });
        if !already_present {
            candidates.push(candidate);
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.world.x.total_cmp(&right.world.x))
            .then_with(|| left.world.y.total_cmp(&right.world.y))
    });
    candidates.truncate(MAX_GLOBAL_COARSE_CANDIDATES);
    candidates
}

#[cfg(feature = "ai-burn")]
fn locate_global_template_runtime(
    matcher: &TemplateMatcher,
    color_pyramid: &ColorMapPyramid,
    config: &TemplateTrackingConfig,
    coarse_template: &PreparedTemplate,
    global_template: &PreparedTemplate,
    local_template: &PreparedTemplate,
) -> Result<LocateResult> {
    let coarse = matcher.locate_coarse_prepared(
        coarse_template,
        config.global_match_threshold,
        0,
        0,
        color_pyramid.coarse.scale,
    )?;

    let coarse_candidates = coarse_peak_candidates(
        &coarse,
        config,
        coarse_template.width(),
        coarse_template.height(),
        0,
        0,
        color_pyramid.coarse.scale,
    );
    if coarse_candidates.is_empty() {
        return Ok(coarse);
    }

    let global_color_mask = build_mask(
        global_template.width(),
        global_template.height(),
        normalized_inner_radius(config.mask_inner_radius, config.mask_outer_radius),
        1.0,
    );
    let local_color_mask = build_mask(
        local_template.width(),
        local_template.height(),
        normalized_inner_radius(config.mask_inner_radius, config.mask_outer_radius),
        1.0,
    );
    let mut best_refined = None;
    let mut best_refined_score = f32::MIN;
    let mut refined_candidates = Vec::new();
    for candidate in coarse_candidates {
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
        if refined.best_score > best_refined_score {
            best_refined_score = refined.best_score;
            best_refined = Some(refined.clone());
        }

        let Some(accepted) = refined.accepted.as_ref() else {
            continue;
        };

        let local_region = search_region_around_center(
            color_pyramid.local.image.width(),
            color_pyramid.local.image.height(),
            center_to_scaled(accepted.world, color_pyramid.local.scale),
            config.global_refine_radius_px / color_pyramid.local.scale.max(1),
            local_template.width(),
            local_template.height(),
        )?;
        let local_crop = crop_search_region_rgba(&color_pyramid.local.image, local_region)?;
        let local_refined = matcher.locate_local_prepared(
            &local_crop.image,
            local_template,
            config.global_match_threshold,
            local_crop.origin_x,
            local_crop.origin_y,
            color_pyramid.local.scale,
        )?;
        let final_result = if local_refined.accepted.is_some() {
            local_refined
        } else {
            refined.clone()
        };
        if final_result.best_score > best_refined_score {
            best_refined_score = final_result.best_score;
            best_refined = Some(final_result.clone());
        }
        let final_world = final_result
            .accepted
            .as_ref()
            .map(|candidate| candidate.world)
            .unwrap_or(accepted.world);

        let global_color_score = scaled_color_score(
            &color_pyramid.global,
            final_world,
            global_template.image(),
            &global_color_mask,
        )
        .unwrap_or(final_result.best_score);
        let local_color_score = scaled_color_score(
            &color_pyramid.local,
            final_world,
            local_template.image(),
            &local_color_mask,
        )
        .unwrap_or(final_result.best_score);
        let rerank_score = (final_result.best_score * TEMPLATE_RERANK_FINAL_SCORE_WEIGHT
            + refined.best_score * TEMPLATE_RERANK_GLOBAL_SCORE_WEIGHT
            + global_color_score * TEMPLATE_RERANK_GLOBAL_COLOR_WEIGHT
            + local_color_score * TEMPLATE_RERANK_LOCAL_COLOR_WEIGHT
            + candidate.score * TEMPLATE_RERANK_COARSE_WEIGHT)
            .clamp(0.0, 1.0);
        refined_candidates.push(TemplateRefinedCandidate {
            result: final_result,
            coarse_score: candidate.score,
            global_score: refined.best_score,
            global_color_score,
            local_color_score,
            rerank_score,
        });
    }

    if let Some(best) = refined_candidates.into_iter().max_by(|left, right| {
        left.rerank_score
            .total_cmp(&right.rerank_score)
            .then_with(|| left.local_color_score.total_cmp(&right.local_color_score))
            .then_with(|| left.global_color_score.total_cmp(&right.global_color_score))
            .then_with(|| left.result.best_score.total_cmp(&right.result.best_score))
            .then_with(|| left.global_score.total_cmp(&right.global_score))
            .then_with(|| left.coarse_score.total_cmp(&right.coarse_score))
            .then(Ordering::Equal)
    }) {
        return Ok(best.result);
    }

    if let Some(best_refined) = best_refined {
        if best_refined.best_score > coarse.best_score {
            return Ok(best_refined);
        }
    }

    Ok(coarse)
}

pub fn rebuild_template_engine_cache(workspace: &WorkspaceSnapshot) -> Result<()> {
    info!(
        cache_root = %workspace.assets.bwiki_cache_dir.display(),
        device = %workspace.config.template.device,
        device_index = workspace.config.template.device_index,
        "rebuilding template tracker cache"
    );
    clear_match_pyramid_caches(workspace)?;

    #[cfg(feature = "ai-burn")]
    clear_tensor_caches_by_prefix(workspace, "template-local-search")?;
    #[cfg(feature = "ai-burn")]
    clear_tensor_caches_by_prefix(workspace, "template-global-search")?;
    #[cfg(feature = "ai-burn")]
    clear_tensor_caches_by_prefix(workspace, "template-coarse-search")?;

    #[cfg(feature = "ai-burn")]
    let cache_key = tracker_map_cache_key(workspace)?;
    #[cfg(feature = "ai-burn")]
    let color_pyramid = load_logic_color_map_pyramid(workspace)?;

    #[cfg(not(feature = "ai-burn"))]
    let _ = load_or_build_match_pyramid(workspace)?;

    #[cfg(feature = "ai-burn")]
    {
        let masks = build_template_masks(&workspace.config);
        let _ = TemplateMatcher::new_cached(
            workspace,
            &workspace.config.template,
            &color_pyramid,
            &masks,
            &cache_key,
        )?;
    }

    info!("rebuild of template tracker cache completed");
    Ok(())
}

#[cfg(not(feature = "ai-burn"))]
fn prepare_capture_templates(
    captured: &GrayImage,
    config: &AppConfig,
    pyramid: &MapPyramid,
) -> (GrayImage, GrayImage, GrayImage) {
    let local_square = capture_template_inner_square(
        captured,
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );
    let annulus = capture_template_annulus(
        captured,
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );
    (
        prepare_square_template(&local_square, config.view_size, pyramid.local.scale),
        prepare_annulus_template(&annulus, config.view_size, pyramid.global.scale),
        prepare_annulus_template(&annulus, config.view_size, pyramid.coarse.scale),
    )
}

fn prepare_color_capture_templates(
    captured: &RgbaImage,
    config: &AppConfig,
    pyramid: &ColorMapPyramid,
) -> ColorCaptureTemplates {
    let local_square = capture_template_inner_square_rgba(
        captured,
        config.template.mask_inner_radius,
        config.template.mask_outer_radius,
    );
    let local_size = scaled_dimension(config.view_size.max(1), pyramid.local.scale);
    ColorCaptureTemplates {
        local: if local_square.width() == local_size && local_square.height() == local_size {
            local_square
        } else {
            image::imageops::resize(
                &local_square,
                local_size,
                local_size,
                image::imageops::FilterType::Triangle,
            )
        },
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

#[cfg_attr(not(test), allow(dead_code))]
#[cfg(not(feature = "ai-burn"))]
fn prepare_capture_template(
    captured: &GrayImage,
    view_size: u32,
    scale: u32,
    mask_inner_radius: f32,
    mask_outer_radius: f32,
) -> GrayImage {
    let square = capture_template_inner_square(captured, mask_inner_radius, mask_outer_radius);
    prepare_square_template(&square, view_size, scale)
}

#[cfg(not(feature = "ai-burn"))]
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

#[cfg(not(feature = "ai-burn"))]
fn prepare_annulus_template(square: &GrayImage, view_size: u32, scale: u32) -> GrayImage {
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

#[cfg(feature = "ai-burn")]
impl TemplateMatcher {
    #[allow(dead_code)]
    fn new(
        config: &TemplateTrackingConfig,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
    ) -> Result<Self> {
        let selection = select_burn_device(config)?;
        Self::from_selection(selection, pyramid, masks, None)
    }

    fn new_cached(
        workspace: &WorkspaceSnapshot,
        config: &TemplateTrackingConfig,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let selection = select_burn_device(config)?;
        Self::from_selection(selection, pyramid, masks, Some((workspace, map_cache_key)))
    }

    fn from_selection(
        selection: BurnDeviceSelection,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        cache: Option<(&WorkspaceSnapshot, &str)>,
    ) -> Result<Self> {
        match selection {
            BurnDeviceSelection::Cpu => {
                let matcher = match cache {
                    Some((workspace, map_cache_key)) => {
                        BurnTemplateMatcher::<burn::backend::NdArray>::new_cached(
                            workspace,
                            NdArrayDevice::Cpu,
                            "CPU".to_owned(),
                            None,
                            pyramid,
                            masks,
                            map_cache_key,
                        )?
                    }
                    None => BurnTemplateMatcher::<burn::backend::NdArray>::new(
                        NdArrayDevice::Cpu,
                        "CPU".to_owned(),
                        None,
                        pyramid,
                        masks,
                    )?,
                };
                Ok(Self::NdArray(matcher))
            }
            #[cfg(burn_cuda_backend)]
            BurnDeviceSelection::Cuda(device) => {
                let label = burn_device_label(&BurnDeviceSelection::Cuda(device.clone()));
                let matcher = match cache {
                    Some((workspace, map_cache_key)) => {
                        BurnTemplateMatcher::<burn::backend::Cuda>::new_cached(
                            workspace,
                            device,
                            label,
                            Some(CUDA_CONV_IM2COL_BUDGET_BYTES),
                            pyramid,
                            masks,
                            map_cache_key,
                        )?
                    }
                    None => BurnTemplateMatcher::<burn::backend::Cuda>::new(
                        device,
                        label,
                        Some(CUDA_CONV_IM2COL_BUDGET_BYTES),
                        pyramid,
                        masks,
                    )?,
                };
                Ok(Self::Cuda(matcher))
            }
            #[cfg(burn_vulkan_backend)]
            BurnDeviceSelection::Vulkan(device) => {
                let label = burn_device_label(&BurnDeviceSelection::Vulkan(device.clone()));
                let matcher = match cache {
                    Some((workspace, map_cache_key)) => {
                        BurnTemplateMatcher::<burn::backend::Vulkan>::new_cached(
                            workspace,
                            device,
                            label,
                            Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                            pyramid,
                            masks,
                            map_cache_key,
                        )?
                    }
                    None => BurnTemplateMatcher::<burn::backend::Vulkan>::new(
                        device,
                        label,
                        Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                        pyramid,
                        masks,
                    )?,
                };
                Ok(Self::Vulkan(matcher))
            }
            #[cfg(burn_metal_backend)]
            BurnDeviceSelection::Metal(device) => {
                let label = burn_device_label(&BurnDeviceSelection::Metal(device.clone()));
                let matcher = match cache {
                    Some((workspace, map_cache_key)) => {
                        BurnTemplateMatcher::<burn::backend::Metal>::new_cached(
                            workspace,
                            device,
                            label,
                            Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                            pyramid,
                            masks,
                            map_cache_key,
                        )?
                    }
                    None => BurnTemplateMatcher::<burn::backend::Metal>::new(
                        device,
                        label,
                        Some(WGPU_CONV_IM2COL_BUDGET_BYTES),
                        pyramid,
                        masks,
                    )?,
                };
                Ok(Self::Metal(matcher))
            }
        }
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
                Self::NdArray(matcher) => PreparedTemplate::NdArray(
                    matcher.prepare_template(local, &matcher.local_mask_squared),
                ),
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => PreparedTemplate::Cuda(
                    matcher.prepare_template(local, &matcher.local_mask_squared),
                ),
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => PreparedTemplate::Vulkan(
                    matcher.prepare_template(local, &matcher.local_mask_squared),
                ),
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => PreparedTemplate::Metal(
                    matcher.prepare_template(local, &matcher.local_mask_squared),
                ),
            },
            global: match self {
                Self::NdArray(matcher) => PreparedTemplate::NdArray(
                    matcher.prepare_template(global, &matcher.global_mask_squared),
                ),
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => PreparedTemplate::Cuda(
                    matcher.prepare_template(global, &matcher.global_mask_squared),
                ),
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => PreparedTemplate::Vulkan(
                    matcher.prepare_template(global, &matcher.global_mask_squared),
                ),
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => PreparedTemplate::Metal(
                    matcher.prepare_template(global, &matcher.global_mask_squared),
                ),
            },
            coarse: match self {
                Self::NdArray(matcher) => PreparedTemplate::NdArray(
                    matcher.prepare_template(coarse, &matcher.coarse_mask_squared),
                ),
                #[cfg(burn_cuda_backend)]
                Self::Cuda(matcher) => PreparedTemplate::Cuda(
                    matcher.prepare_template(coarse, &matcher.coarse_mask_squared),
                ),
                #[cfg(burn_vulkan_backend)]
                Self::Vulkan(matcher) => PreparedTemplate::Vulkan(
                    matcher.prepare_template(coarse, &matcher.coarse_mask_squared),
                ),
                #[cfg(burn_metal_backend)]
                Self::Metal(matcher) => PreparedTemplate::Metal(
                    matcher.prepare_template(coarse, &matcher.coarse_mask_squared),
                ),
            },
        })
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

    fn locate_local_prepared(
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
impl<B> BurnTemplateMatcher<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Clone + Send + Sync + 'static,
{
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(
        device: B::Device,
        device_label: String,
        chunk_budget_bytes: Option<usize>,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
    ) -> Result<Self> {
        let coarse_search =
            SearchTensorCache::<B>::from_rgba_image(&pyramid.coarse.image, &device)?;
        Self::from_parts(
            device,
            device_label,
            chunk_budget_bytes,
            coarse_search,
            masks,
        )
    }

    fn new_cached(
        workspace: &WorkspaceSnapshot,
        device: B::Device,
        device_label: String,
        chunk_budget_bytes: Option<usize>,
        pyramid: &ColorMapPyramid,
        masks: &MaskSet,
        map_cache_key: &str,
    ) -> Result<Self> {
        let coarse_search = load_or_build_template_search::<B>(
            workspace,
            "template-coarse-search",
            map_cache_key,
            &pyramid.coarse.image,
            &device,
        )?;
        Self::from_parts(
            device,
            device_label,
            chunk_budget_bytes,
            coarse_search,
            masks,
        )
    }

    fn from_parts(
        device: B::Device,
        device_label: String,
        chunk_budget_bytes: Option<usize>,
        coarse_search: SearchTensorCache<B>,
        masks: &MaskSet,
    ) -> Result<Self> {
        let global_mask_squared = mask_squared_tensor::<B>(&masks.global, &device);
        let coarse_mask_squared = mask_squared_tensor::<B>(&masks.coarse, &device);
        let local_mask_squared = mask_squared_tensor::<B>(&masks.local, &device);
        let coarse_search_patch_energy = conv2d(
            coarse_search.squared.clone(),
            coarse_mask_squared.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );

        Ok(Self {
            device,
            device_label,
            coarse_search,
            global_mask_squared,
            coarse_mask_squared,
            coarse_search_patch_energy,
            local_mask_squared,
            chunk_budget_bytes,
        })
    }

    fn device_label(&self) -> String {
        self.device_label.clone()
    }

    fn prepare_template(
        &self,
        image: RgbaImage,
        mask_squared: &Tensor<B, 4>,
    ) -> BurnPreparedTemplate<B> {
        let template_tensor = rgba_image_tensor::<B>(&image, &self.device);
        let weighted_template = template_tensor.clone() * mask_squared.clone();
        let template_energy = (template_tensor.powi_scalar(2) * mask_squared.clone())
            .sum()
            .into_scalar();
        BurnPreparedTemplate {
            image,
            mask_squared: mask_squared.clone(),
            weighted_template,
            template_energy,
        }
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
        let prepared = self.prepare_template(template.clone(), &self.coarse_mask_squared);
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
        let prepared = self.prepare_template(template.clone(), &self.global_mask_squared);
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
        let search = SearchTensorCache::<B>::from_rgba_image(image, &self.device)?;
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
        let prepared = self.prepare_template(template.clone(), &self.local_mask_squared);
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
        let search = SearchTensorCache::<B>::from_rgba_image(image, &self.device)?;
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
            3,
        );

        if chunk_rows < score_height {
            return locate_cached_in_chunks(
                search,
                &template.weighted_template,
                &template.mask_squared,
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
            search.image.clone(),
            template.weighted_template.clone(),
            None::<Tensor<B, 1>>,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
        let search_patch_energy = match precomputed_patch_energy {
            Some(patch_energy) => patch_energy,
            None => conv2d(
                search.squared.clone(),
                template.mask_squared.clone(),
                None::<Tensor<B, 1>>,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            ),
        };
        let normalized = numerator / (search_patch_energy * template.template_energy + 1e-6).sqrt();

        if capture_score_map || burn_score_map_capture_enabled() {
            let score_map = tensor_to_flat_f32(normalized)?;
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
{
    fn from_rgba_image(image: &RgbaImage, device: &B::Device) -> Result<Self> {
        let image_tensor = rgba_image_tensor::<B>(image, device);
        let squared = image_tensor.clone().powi_scalar(2);
        Ok(Self {
            image: image_tensor,
            squared,
            width: image.width(),
            height: image.height(),
        })
    }

    fn from_persisted(cache: PersistedTensorCache, device: &B::Device) -> Result<Self> {
        if cache.channels != 3 {
            anyhow::bail!(
                "template search tensor cache channel count {} is invalid",
                cache.channels
            );
        }

        let image = Tensor::<B, 4>::from_data(
            TensorData::new(
                cache.primary,
                [1, 3, cache.height as usize, cache.width as usize],
            ),
            device,
        );
        let squared = Tensor::<B, 4>::from_data(
            TensorData::new(
                cache.secondary,
                [1, 3, cache.height as usize, cache.width as usize],
            ),
            device,
        );
        Ok(Self {
            image,
            squared,
            width: cache.width,
            height: cache.height,
        })
    }

    fn to_persisted(&self) -> Result<PersistedTensorCache> {
        let image = tensor_to_flat_f32(self.image.clone())?;
        let squared = tensor_to_flat_f32(self.squared.clone())?;
        PersistedTensorCache::from_parts(self.width, self.height, 3, image, squared)
    }
}

#[cfg(feature = "ai-burn")]
fn load_or_build_template_search<B>(
    workspace: &WorkspaceSnapshot,
    prefix: &str,
    map_cache_key: &str,
    image: &RgbaImage,
    device: &B::Device,
) -> Result<SearchTensorCache<B>>
where
    B: Backend<FloatElem = f32>,
{
    let cache_path = tracker_tensor_cache_path(workspace, prefix, map_cache_key);
    if let Ok(Some(cache)) = load_tensor_cache(&cache_path) {
        if let Ok(search) = SearchTensorCache::<B>::from_persisted(cache, device) {
            return Ok(search);
        }
    }

    let search = SearchTensorCache::<B>::from_rgba_image(image, device)?;
    if let Ok(persisted) = search.to_persisted() {
        let _ = save_tensor_cache(&cache_path, &persisted);
    }
    Ok(search)
}

impl TrackingWorker for TemplateTrackerWorker {
    fn refresh_interval(&self) -> Duration {
        Duration::from_millis(self.config.template.refresh_rate_ms)
    }

    fn tick(&mut self) -> Result<TrackingTick> {
        self.run_frame()
    }

    fn set_debug_enabled(&mut self, enabled: bool) {
        self.debug_enabled = enabled;
    }

    fn initial_status(&self) -> TrackingStatus {
        #[cfg(feature = "ai-burn")]
        let message = format!(
            "多尺度模板匹配引擎已启动：设备 {}，可用后端 {}，Burn masked NCC 张量匹配 + 局部锁定 / 全局重定位 / 惯性保位。",
            self.matcher.device_label(),
            available_burn_backends(),
        );

        #[cfg(not(feature = "ai-burn"))]
        let message = "多尺度模板匹配引擎当前二进制未启用 `ai-burn` 特性。".to_owned();

        TrackingStatus::new(TrackerEngineKind::MultiScaleTemplateMatch, message)
    }

    fn engine_kind(&self) -> TrackerEngineKind {
        TrackerEngineKind::MultiScaleTemplateMatch
    }
}

#[cfg(feature = "ai-burn")]
fn rgba_image_tensor<B>(image: &RgbaImage, device: &B::Device) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    Tensor::<B, 4>::from_data(
        TensorData::new(
            rgba_image_as_unit_vec(image),
            [1, 3, image.height() as usize, image.width() as usize],
        ),
        device,
    )
}

#[cfg(feature = "ai-burn")]
fn mask_squared_tensor<B>(mask: &GrayImage, device: &B::Device) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    let values = mask_as_unit_vec(mask, 3)
        .into_iter()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    Tensor::<B, 4>::from_data(
        TensorData::new(
            values,
            [1, 3, mask.height() as usize, mask.width() as usize],
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
    mask_squared: &Tensor<B, 4>,
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
        let image_chunk =
            search
                .image
                .clone()
                .narrow(2, output_row as usize, slice_height as usize);
        let squared_chunk =
            search
                .squared
                .clone()
                .narrow(2, output_row as usize, slice_height as usize);
        let numerator = conv2d(
            image_chunk.clone(),
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
                mask_squared.clone(),
                None::<Tensor<B, 1>>,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            ),
        };
        let normalized = numerator / (search_patch_energy * template_energy + 1e-6).sqrt();
        let chunk_scores = tensor_to_flat_f32(normalized)?;
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
fn tensor_to_flat_f32<B>(tensor: Tensor<B, 4>) -> Result<Vec<f32>>
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
    const GLOBAL_TOLERANCE: f32 = 12.0;
    const LOCAL_TOLERANCE: f32 = 10.0;
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

    fn template_env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(default)
    }

    fn fixture() -> &'static TestFixture {
        FIXTURE.get_or_init(|| {
            let (fixture, elapsed) = timed(|| {
                let mut config = runtime_config_or_default();
                config.template.global_match_threshold = template_env_f32(
                    "GAME_MAP_TRACKER_TEMPLATE_GLOBAL_THRESHOLD",
                    config.template.global_match_threshold,
                );
                config.template.local_match_threshold = template_env_f32(
                    "GAME_MAP_TRACKER_TEMPLATE_LOCAL_THRESHOLD",
                    config.template.local_match_threshold,
                );
                config.template.global_downscale = stress_env_u32(
                    "GAME_MAP_TRACKER_TEMPLATE_GLOBAL_DOWNSCALE",
                    config.template.global_downscale,
                );
                config.template.local_downscale = stress_env_u32(
                    "GAME_MAP_TRACKER_TEMPLATE_LOCAL_DOWNSCALE",
                    config.template.local_downscale,
                );
                let workspace = build_test_workspace(config.clone(), "template-vulkan");
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
            print_perf_ms("template/vulkan", "fixture_prepare", elapsed);
            fixture
        })
    }

    fn matcher_for_vulkan(fixture: &TestFixture, ordinal: usize) -> TemplateMatcher {
        let mut config = fixture.config.template.clone();
        config.device = AiDevicePreference::Vulkan;
        config.device_index = ordinal;
        TemplateMatcher::new_cached(
            &fixture.workspace,
            &config,
            &fixture.color_pyramid,
            &fixture.masks,
            &fixture.map_cache_key,
        )
        .expect("failed to create Vulkan template matcher")
    }

    fn within_tolerance(actual: Option<WorldPoint>, expected: (u32, u32), tolerance: f32) -> bool {
        actual.is_some_and(|actual| {
            (actual.x - expected.0 as f32).abs() <= tolerance
                && (actual.y - expected.1 as f32).abs() <= tolerance
        })
    }

    fn locate_global_runtime(
        fixture: &TestFixture,
        matcher: &TemplateMatcher,
        templates: &PreparedCaptureTemplates,
    ) -> Result<LocateResult> {
        locate_global_template_runtime(
            matcher,
            &fixture.color_pyramid,
            &fixture.config.template,
            &templates.coarse,
            &templates.global,
            &templates.local,
        )
    }

    fn simulate_runtime_frame(
        fixture: &TestFixture,
        matcher: &TemplateMatcher,
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
                let result = matcher.locate_local_prepared(
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
                                source: Some(TrackingSource::LocalTrack),
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
        let mut best_score = Some(global.best_score);
        let mut best_color = Some(global.best_score);
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
            let refine = matcher.locate_local_prepared(
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
                    source: Some(TrackingSource::GlobalRelocate),
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
        matcher: &TemplateMatcher,
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
        print_perf_ms("template/vulkan", "matcher_init", init_elapsed);

        let mut best_global = 0.0f32;
        let mut best_local = 0.0f32;
        let mut best_overall = 0.0f32;

        let max_rounds = max_rounds();
        for round in 0..max_rounds {
            let seed = 0x5445_4d50_4c41_5445u64.wrapping_add(round as u64 * 0x9e37_79b9);
            let (stats, elapsed) = timed(|| run_round(fixture, &matcher, seed));
            let stats = stats?;
            let report_path = write_stress_report(
                "template-vulkan",
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
                "[template/vulkan][round={}] global={:.2}% local={:.2}% overall={:.2}% elapsed_ms={:.0} failures={} report={}",
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
            "template Vulkan stress accuracy stayed below target after {} rounds; best global {:.2}%, best local {:.2}%, best overall {:.2}%",
            max_rounds,
            best_global * 100.0,
            best_local * 100.0,
            best_overall * 100.0
        )
    }
}
