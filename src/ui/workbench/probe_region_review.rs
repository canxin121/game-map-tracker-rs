use std::sync::Arc;

use gpui::{
    AnyWindowHandle, ClickEvent, Context, InteractiveElement as _, IntoElement, MouseButton,
    ParentElement as _, Render, RenderImage, Styled, Window, canvas, div, fill, hsla, px,
};

use crate::{
    config::CaptureRegion,
    tracking::{
        debug::{DebugImage, DebugImageFormat, DebugImageKind},
        presence::{MinimapPresenceModel, MinimapPresenceModelBuild, MinimapPresenceSample},
    },
};

use super::{
    TrackerWorkbench,
    debug_images::{contained_image_bounds, render_image_from_debug_image},
    picker_shared::{close_picker, commit_picker_result, picker_control_button},
};

pub(super) struct MinimapPresenceProbeReviewWindow {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    region: CaptureRegion,
    model: MinimapPresenceModel,
    sample: MinimapPresenceSample,
    raw_render_image: Option<Arc<RenderImage>>,
    modeled_render_image: Option<Arc<RenderImage>>,
}

impl MinimapPresenceProbeReviewWindow {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        main_window_handle: AnyWindowHandle,
        region: CaptureRegion,
        build: MinimapPresenceModelBuild,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Self {
        let raw_render_image = render_rgba_image("F1-P Raw", &build.sample.current_raw_preview);
        let modeled_render_image =
            render_rgba_image("F1-P Modeled", &build.sample.current_modeled_preview);
        Self {
            workbench,
            main_window_handle,
            region,
            model: build.model,
            sample: build.sample,
            raw_render_image,
            modeled_render_image,
        }
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.release_images(cx);
        close_picker(&self.workbench, window, cx, |this, _| {
            this.handle_minimap_presence_probe_review_cancelled();
        });
    }

    fn confirm(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let region = self.region.clone();
        let model = self.model.clone();
        self.release_images(cx);
        commit_picker_result(
            &self.workbench,
            self.main_window_handle,
            window,
            cx,
            move |this, main_window, cx| {
                this.finish_minimap_presence_probe_pick(region, model, main_window, cx);
            },
        );
    }

    fn release_images(&mut self, cx: &mut Context<Self>) {
        if let Some(image) = self.raw_render_image.take() {
            cx.drop_image(image, None);
        }
        if let Some(image) = self.modeled_render_image.take() {
            cx.drop_image(image, None);
        }
    }

    fn status_line(&self) -> String {
        format!(
            "{}  final {:.3}  mean {:.3} / {:.3}  min {:.3}",
            if self.sample.present {
                "自检通过"
            } else {
                "自检未通过"
            },
            self.sample.score,
            self.sample.mean_raw_score,
            self.sample.threshold,
            self.sample.min_raw_score
        )
    }

    fn slot_line(&self) -> String {
        ["F1", "F2", "F3", "F4", "J", "P"]
            .iter()
            .zip(self.sample.slot_scores.iter().copied())
            .map(|(tag, score)| format!("{tag}:{score:.2}"))
            .collect::<Vec<_>>()
            .join("  ")
    }
}

impl Drop for MinimapPresenceProbeReviewWindow {
    fn drop(&mut self) {
        self.raw_render_image = None;
        self.modeled_render_image = None;
    }
}

impl Render for MinimapPresenceProbeReviewWindow {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(hsla(0.60, 0.10, 0.10, 0.98))
            .p_4()
            .child(
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(hsla(0.0, 0.0, 1.0, 0.96))
                                    .child("F1-P 建模预览"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .line_height(px(18.0))
                                    .text_color(hsla(0.0, 0.0, 1.0, 0.80))
                                    .child("先确认当前建模结果确实框住了 F1 F2 F3 F4 J P 六个标签，再点击“确认保存”。"),
                            ),
                    )
                    .child(
                        div()
                            .rounded_lg()
                            .bg(hsla(
                                if self.sample.present { 0.36 } else { 0.0 },
                                0.52,
                                0.26,
                                0.88,
                            ))
                            .border_1()
                            .border_color(hsla(0.0, 0.0, 1.0, 0.12))
                            .p_3()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(hsla(0.0, 0.0, 1.0, 0.94))
                                    .child(self.status_line()),
                            )
                            .child(
                                div()
                                    .mt_2()
                                    .text_xs()
                                    .text_color(hsla(0.0, 0.0, 1.0, 0.76))
                                    .child(self.slot_line()),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h(px(0.0))
                            .flex()
                            .gap_3()
                            .child(preview_panel(
                                "原始捕获",
                                self.sample.current_raw_preview.width(),
                                self.sample.current_raw_preview.height(),
                                self.raw_render_image.clone(),
                            ))
                            .child(preview_panel(
                                "建模结果",
                                self.sample.current_modeled_preview.width(),
                                self.sample.current_modeled_preview.height(),
                                self.modeled_render_image.clone(),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(hsla(0.0, 0.0, 1.0, 0.70))
                                    .child(format!(
                                        "区域 top {} / left {} / {}x{}",
                                        self.region.top,
                                        self.region.left,
                                        self.region.width,
                                        self.region.height
                                    )),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(picker_control_button(
                                        "probe-review-cancel",
                                        "取消",
                                        hsla(0.0, 0.0, 0.16, 0.88),
                                        hsla(0.0, 0.0, 0.24, 0.96),
                                        hsla(0.0, 0.0, 1.0, 0.18),
                                        true,
                                        cx.listener(|this, _: &ClickEvent, window, cx| {
                                            this.cancel(window, cx);
                                        }),
                                    ))
                                    .child(picker_control_button(
                                        "probe-review-confirm",
                                        "确认保存",
                                        hsla(0.36, 0.68, 0.48, 0.92),
                                        hsla(0.36, 0.72, 0.56, 0.98),
                                        hsla(0.36, 0.80, 0.70, 0.34),
                                        true,
                                        cx.listener(|this, _: &ClickEvent, window, cx| {
                                            this.confirm(window, cx);
                                        }),
                                    )),
                            ),
                    ),
            )
    }
}

fn render_rgba_image(label: &str, image: &image::RgbaImage) -> Option<Arc<RenderImage>> {
    let debug_image = DebugImage::rgba(
        label,
        image.width(),
        image.height(),
        DebugImageKind::Snapshot,
        image.clone().into_raw(),
    );
    render_image_from_debug_image(&debug_image)
}

fn preview_panel(
    title: &'static str,
    image_width: u32,
    image_height: u32,
    render_image: Option<Arc<RenderImage>>,
) -> impl IntoElement {
    div()
        .flex_1()
        .min_w(px(0.0))
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(hsla(0.0, 0.0, 1.0, 0.88))
                .child(title),
        )
        .child(
            div()
                .flex_1()
                .min_h(px(0.0))
                .rounded_lg()
                .bg(hsla(0.0, 0.0, 0.08, 0.88))
                .border_1()
                .border_color(hsla(0.0, 0.0, 1.0, 0.14))
                .overflow_hidden()
                .child(preview_canvas(
                    image_width,
                    image_height,
                    render_image,
                    DebugImageFormat::Rgba8,
                )),
        )
}

fn preview_canvas(
    image_width: u32,
    image_height: u32,
    render_image: Option<Arc<RenderImage>>,
    format: DebugImageFormat,
) -> impl IntoElement {
    div()
        .size_full()
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .child(
            canvas(
                move |_, _, _| (image_width, image_height, render_image.clone(), format),
                move |bounds, state, window, _| {
                    window.paint_quad(fill(bounds, hsla(0.0, 0.0, 0.08, 0.88)));
                    let (image_width, image_height, render_image, format) = state;
                    if image_width == 0 || image_height == 0 {
                        return;
                    }
                    let Some(render_image) = render_image.as_ref() else {
                        return;
                    };
                    let image_bounds = contained_image_bounds(bounds, image_width, image_height);
                    let _ = window.paint_image(
                        image_bounds,
                        0.0.into(),
                        render_image.clone(),
                        0,
                        matches!(format, DebugImageFormat::Gray8),
                    );
                },
            )
            .size_full(),
        )
}
