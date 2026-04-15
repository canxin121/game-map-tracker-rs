use std::{sync::Arc, time::Duration};

use anyhow::Result;
use image::GrayImage;
#[cfg(not(feature = "ai-candle"))]
use image::{ImageBuffer, Luma};
#[cfg(not(feature = "ai-candle"))]
use imageproc::template_matching::{MatchTemplateMethod, match_template_with_mask_parallel};

#[cfg(not(feature = "ai-candle"))]
use crate::config::AiDevicePreference;
#[cfg(feature = "ai-candle")]
use crate::config::TemplateTrackingConfig;
#[cfg(feature = "ai-candle")]
use crate::tracking::vision::{gray_image_as_unit_vec, mask_as_unit_vec};
use crate::{
    config::AppConfig,
    domain::{
        geometry::WorldPoint,
        tracker::{PositionEstimate, TrackerEngineKind, TrackingSource},
    },
    resources::WorkspaceSnapshot,
    tracking::{
        capture::{CaptureSource, DesktopCapture},
        debug::{DebugField, TrackingDebugSnapshot},
        runtime::{TrackingStatus, TrackingTick, TrackingWorker},
        vision::{
            DebugOverlay, MapPyramid, MaskSet, MatchCandidate, SearchCrop, SearchStage,
            TrackerState, build_debug_snapshot, center_to_scaled, crop_around_center,
            downscale_gray, load_logic_map_pyramid, preview_heatmap, preview_image,
            preview_mask_image,
        },
    },
};

#[cfg(feature = "ai-candle")]
use crate::tracking::candle_support::{
    available_candle_backends, candle_device_label, select_candle_device,
};
#[cfg(feature = "ai-candle")]
use candle_core::{Device, Tensor};

#[derive(Debug, Clone)]
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
pub struct TemplateTrackerWorker {
    config: AppConfig,
    capture: DesktopCapture,
    pyramid: MapPyramid,
    masks: MaskSet,
    state: TrackerState,
    #[cfg(feature = "ai-candle")]
    matcher: CandleTemplateMatcher,
}

#[cfg(feature = "ai-candle")]
#[derive(Debug, Clone)]
struct CandleTemplateMatcher {
    device: Device,
    global_search: SearchTensorCache,
    global_mask_squared: Tensor,
    local_mask_squared: Tensor,
}

#[cfg(feature = "ai-candle")]
#[derive(Debug, Clone)]
struct SearchTensorCache {
    image: Tensor,
    squared: Tensor,
    width: u32,
    height: u32,
}

impl TemplateTrackerWorker {
    pub fn new(workspace: Arc<WorkspaceSnapshot>) -> Result<Self> {
        let config = workspace.config.clone();
        let capture = DesktopCapture::from_absolute_region(&config.minimap)?;
        let (pyramid, masks) = load_logic_map_pyramid(workspace.as_ref())?;

        #[cfg(feature = "ai-candle")]
        let matcher = CandleTemplateMatcher::new(&config.template, &pyramid, &masks)?;

        #[cfg(not(feature = "ai-candle"))]
        if config.template.device != AiDevicePreference::Cpu {
            anyhow::bail!(
                "模板匹配引擎配置选择了 {} 设备，但当前二进制未启用 `ai-candle` 特性；请先使用默认构建或显式启用 Candle 后端重新构建",
                config.template.device
            );
        }

        Ok(Self {
            config,
            capture,
            pyramid,
            masks,
            state: TrackerState::default(),
            #[cfg(feature = "ai-candle")]
            matcher,
        })
    }

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
                    let jump = (candidate.world.x - last_world.x).abs()
                        + (candidate.world.y - last_world.y).abs();
                    if jump <= self.config.local_search.max_accepted_jump_px as f32 {
                        status.source = Some(TrackingSource::LocalTrack);
                        status.match_score = Some(candidate.score);
                        status.message = format!(
                            "局部模板锁定成功，得分 {:.3}，坐标 {:.0}, {:.0}。",
                            candidate.score, candidate.world.x, candidate.world.y
                        );
                        estimate = Some(self.commit_success(candidate, TrackingSource::LocalTrack));
                    }
                }

                if estimate.is_none() {
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

        if estimate.is_none() {
            let result = self.locate_global_template(&global_template)?;
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
                    estimate = Some(self.commit_success(candidate, TrackingSource::GlobalRelocate));
                }
            }
        }

        if estimate.is_none() {
            estimate = self.apply_inertial_fallback(&mut status);
        }

        if estimate.is_none() {
            status.source = None;
            status.match_score = None;
            status.message = "当前帧未找到可靠匹配，等待下一帧。".to_owned();
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

    fn locate_global_template(&self, template: &GrayImage) -> Result<LocateResult> {
        #[cfg(feature = "ai-candle")]
        {
            return self.matcher.locate_cached(
                &self.matcher.global_search,
                template,
                &self.matcher.global_mask_squared,
                self.config.template.global_match_threshold,
                0,
                0,
                self.pyramid.global.scale,
            );
        }

        #[cfg(not(feature = "ai-candle"))]
        {
            locate_template_cpu(
                &self.pyramid.global.image,
                template,
                &self.masks.global,
                self.config.template.global_match_threshold,
                0,
                0,
                self.pyramid.global.scale,
            )
        }
    }

    fn locate_local_template(
        &self,
        image: &GrayImage,
        template: &GrayImage,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        #[cfg(feature = "ai-candle")]
        {
            return self.matcher.locate_dynamic(
                image,
                template,
                &self.matcher.local_mask_squared,
                threshold,
                origin_x,
                origin_y,
                scale,
            );
        }

        #[cfg(not(feature = "ai-candle"))]
        {
            locate_template_cpu(
                image,
                template,
                &self.masks.local,
                threshold,
                origin_x,
                origin_y,
                scale,
            )
        }
    }

    fn base_status(&self) -> TrackingStatus {
        TrackingStatus {
            engine: TrackerEngineKind::MultiScaleTemplateMatch,
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
            "Coarse Match",
            &self.pyramid.global.image,
            &global_overlays,
            196,
        );
        let global_mask = preview_mask_image("Global Mask", &self.masks.global, 196);
        let global_heatmap = global_result.map(|result| {
            preview_heatmap(
                "Coarse Heatmap",
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

            preview_image("Refine / Local", &crop.image, &overlays, 196)
        } else {
            preview_image("Refine / Local", &self.pyramid.local.image, &[], 196)
        };
        let local_mask = preview_mask_image("Local Mask", &self.masks.local, 196);
        let refine_heatmap = refine_result.map(|result| {
            preview_heatmap(
                "Refine Heatmap",
                result.score_width,
                result.score_height,
                &result.score_map,
                Some((result.best_left, result.best_top)),
                196,
            )
        });

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

        #[cfg(feature = "ai-candle")]
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

        build_debug_snapshot(
            self.engine_kind(),
            self.state.frame_index,
            self.state.stage,
            vec![
                minimap,
                global_mask,
                global_heatmap.unwrap_or_else(|| {
                    preview_mask_image("Coarse Heatmap", &self.masks.global, 196)
                }),
                global,
                local_mask,
                refine_heatmap.unwrap_or_else(|| {
                    preview_mask_image("Refine Heatmap", &self.masks.local, 196)
                }),
                refine,
            ],
            fields,
        )
    }
}

#[cfg(feature = "ai-candle")]
impl CandleTemplateMatcher {
    fn new(config: &TemplateTrackingConfig, pyramid: &MapPyramid, masks: &MaskSet) -> Result<Self> {
        let device = select_candle_device(config)?;
        let global_search = SearchTensorCache::from_gray_image(&pyramid.global.image, &device)?;
        let global_mask_squared = mask_squared_tensor(&masks.global, &device)?;
        let local_mask_squared = mask_squared_tensor(&masks.local, &device)?;

        Ok(Self {
            device,
            global_search,
            global_mask_squared,
            local_mask_squared,
        })
    }

    fn device_label(&self) -> String {
        candle_device_label(&self.device)
    }

    fn locate_dynamic(
        &self,
        image: &GrayImage,
        template: &GrayImage,
        mask_squared: &Tensor,
        threshold: f32,
        origin_x: u32,
        origin_y: u32,
        scale: u32,
    ) -> Result<LocateResult> {
        let search = SearchTensorCache::from_gray_image(image, &self.device)?;
        self.locate_cached(
            &search,
            template,
            mask_squared,
            threshold,
            origin_x,
            origin_y,
            scale,
        )
    }

    fn locate_cached(
        &self,
        search: &SearchTensorCache,
        template: &GrayImage,
        mask_squared: &Tensor,
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

        let template_tensor = gray_image_tensor(template, &self.device)?;
        let weighted_template = template_tensor.broadcast_mul(mask_squared)?;
        let numerator = search.image.conv2d(&weighted_template, 0, 1, 1, 1)?;
        let search_patch_energy = search.squared.conv2d(mask_squared, 0, 1, 1, 1)?;
        let template_energy = template_tensor
            .sqr()?
            .broadcast_mul(mask_squared)?
            .sum_all()?
            .to_scalar::<f32>()?;
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
    fn from_gray_image(image: &GrayImage, device: &Device) -> Result<Self> {
        let image_tensor = gray_image_tensor(image, device)?;
        let squared = image_tensor.sqr()?;
        Ok(Self {
            image: image_tensor,
            squared,
            width: image.width(),
            height: image.height(),
        })
    }
}

impl TrackingWorker for TemplateTrackerWorker {
    fn refresh_interval(&self) -> Duration {
        Duration::from_millis(self.config.template.refresh_rate_ms)
    }

    fn tick(&mut self) -> Result<TrackingTick> {
        self.run_frame()
    }

    fn initial_status(&self) -> TrackingStatus {
        #[cfg(feature = "ai-candle")]
        let message = format!(
            "多尺度模板匹配引擎已启动：设备 {}，可用后端 {}，masked NCC 张量匹配 + 局部锁定 / 全局重定位 / 惯性保位。",
            self.matcher.device_label(),
            available_candle_backends(),
        );

        #[cfg(not(feature = "ai-candle"))]
        let message =
            "多尺度模板匹配引擎已启动：设备 CPU，局部锁定 + 全局重定位 + 惯性保位。".to_owned();

        TrackingStatus::new(TrackerEngineKind::MultiScaleTemplateMatch, message)
    }

    fn engine_kind(&self) -> TrackerEngineKind {
        TrackerEngineKind::MultiScaleTemplateMatch
    }
}

#[cfg(feature = "ai-candle")]
fn gray_image_tensor(image: &GrayImage, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        gray_image_as_unit_vec(image),
        (1, 1, image.height() as usize, image.width() as usize),
        device,
    )?)
}

#[cfg(feature = "ai-candle")]
fn mask_squared_tensor(mask: &GrayImage, device: &Device) -> Result<Tensor> {
    let values = mask_as_unit_vec(mask, 1)
        .into_iter()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(
        values,
        (1, 1, mask.height() as usize, mask.width() as usize),
        device,
    )?)
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
                (origin_x + best_left as u32 + template_width / 2) as f32 * scale as f32,
                (origin_y + best_top as u32 + template_height / 2) as f32 * scale as f32,
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

#[cfg(not(feature = "ai-candle"))]
fn locate_template_cpu(
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

    let result = match_template_with_mask_parallel(
        image,
        template,
        MatchTemplateMethod::CrossCorrelationNormalized,
        mask,
    );
    let (best_left, best_top, best_score) = best_match_location(&result);
    let score_width = result.width();
    let score_height = result.height();
    let score_map = result.pixels().map(|pixel| pixel.0[0]).collect::<Vec<_>>();

    let accepted = if best_score >= threshold {
        Some(MatchCandidate {
            world: WorldPoint::new(
                (origin_x + best_left + template.width() / 2) as f32 * scale as f32,
                (origin_y + best_top + template.height() / 2) as f32 * scale as f32,
            ),
            score: best_score,
        })
    } else {
        None
    };

    Ok(LocateResult {
        best_left,
        best_top,
        best_score,
        score_width,
        score_height,
        score_map,
        accepted,
    })
}

#[cfg(not(feature = "ai-candle"))]
fn best_match_location(result: &GrayImageF32) -> (u32, u32, f32) {
    let mut best_left = 0;
    let mut best_top = 0;
    let mut best_score = f32::MIN;

    for (x, y, pixel) in result.enumerate_pixels() {
        let score = pixel.0[0];
        if score > best_score {
            best_left = x;
            best_top = y;
            best_score = score;
        }
    }

    (best_left, best_top, best_score)
}

#[cfg(not(feature = "ai-candle"))]
type GrayImageF32 = ImageBuffer<Luma<f32>, Vec<f32>>;
