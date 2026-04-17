use gpui::{
    AnyWindowHandle, App, Bounds, ClickEvent, Context, CursorStyle, InteractiveElement as _,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _,
    PathBuilder, Pixels, Point, Render, SharedString, StatefulInteractiveElement as _, Styled,
    Window, canvas, div, hsla, point, px, size, transparent_black,
};

use crate::config::CaptureRegion;

use super::TrackerWorkbench;

const MIN_SELECTION_SIZE: f32 = 18.0;
const OUTLINE_STROKE_WIDTH: f32 = 3.0;

pub(super) struct MinimapPresenceProbePicker {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    display_bounds: Bounds<Pixels>,
    selection: Option<Bounds<Pixels>>,
    drag_anchor: Option<Point<Pixels>>,
}

impl MinimapPresenceProbePicker {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        main_window_handle: AnyWindowHandle,
        display_bounds: Bounds<Pixels>,
    ) -> Self {
        Self {
            workbench,
            main_window_handle,
            display_bounds,
            selection: None,
            drag_anchor: None,
        }
    }

    fn begin_drag(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        self.drag_anchor = Some(event.position);
        self.selection = Some(rect_from_drag(event.position, event.position));
        cx.notify();
    }

    fn update_drag(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        let Some(anchor) = self.drag_anchor else {
            return;
        };

        self.selection = Some(clamp_rect_to_bounds(
            rect_from_drag(anchor, event.position),
            self.local_bounds(),
        ));
        cx.notify();
    }

    fn finish_drag(&mut self, event: &MouseUpEvent, cx: &mut Context<Self>) {
        let Some(anchor) = self.drag_anchor.take() else {
            return;
        };

        let selection =
            clamp_rect_to_bounds(rect_from_drag(anchor, event.position), self.local_bounds());
        let width = f32::from(selection.size.width);
        let height = f32::from(selection.size.height);
        self.selection =
            (width >= MIN_SELECTION_SIZE && height >= MIN_SELECTION_SIZE).then_some(selection);
        cx.notify();
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.handle_minimap_presence_probe_picker_cancelled();
            });
        }
        window.remove_window();
    }

    fn reset_selection(&mut self, cx: &mut Context<Self>) {
        self.selection = None;
        self.drag_anchor = None;
        cx.notify();
    }

    fn confirm_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.selection else {
            return;
        };

        let region = self.capture_region_for(selection);
        self.commit_region(region, window, cx);
    }

    fn commit_region(
        &mut self,
        region: CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let workbench = self.workbench.clone();
        let main_window_handle = self.main_window_handle;
        let _ = main_window_handle.update(cx, move |_, main_window, cx| {
            if let Some(workbench) = workbench.upgrade() {
                let _ = workbench.update(cx, |this, cx| {
                    this.finish_minimap_presence_probe_pick(region, main_window, cx);
                });
            }
        });

        window.remove_window();
    }

    fn capture_region_for(&self, selection: Bounds<Pixels>) -> CaptureRegion {
        CaptureRegion {
            top: (f32::from(self.display_bounds.origin.y) + f32::from(selection.origin.y)).round()
                as i32,
            left: (f32::from(self.display_bounds.origin.x) + f32::from(selection.origin.x)).round()
                as i32,
            width: f32::from(selection.size.width).round().max(1.0) as u32,
            height: f32::from(selection.size.height).round().max(1.0) as u32,
        }
    }

    fn local_bounds(&self) -> Bounds<Pixels> {
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: self.display_bounds.size,
        }
    }
}

impl Render for MinimapPresenceProbePicker {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let has_selection = self.selection.is_some();

        div()
            .relative()
            .size_full()
            .bg(transparent_black())
            .cursor(CursorStyle::Crosshair)
            .child(
                canvas(
                    |_, _, _| {},
                    move |bounds, _, window, cx| {
                        let selection = entity.read(cx).selection.map(|selection| Bounds {
                            origin: point(
                                bounds.origin.x + selection.origin.x,
                                bounds.origin.y + selection.origin.y,
                            ),
                            size: selection.size,
                        });

                        if let Some(selection) = selection {
                            paint_selection(window, selection);
                        }
                    },
                )
                .size_full()
                .bg(transparent_black()),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    this.begin_drag(event, cx);
                }),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                this.update_drag(event, cx);
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, _, cx| {
                    this.finish_drag(event, cx);
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    this.cancel(window, cx);
                }),
            )
            .child(picker_controls(has_selection, cx))
    }
}

fn picker_controls(
    has_selection: bool,
    cx: &mut Context<MinimapPresenceProbePicker>,
) -> impl IntoElement {
    div()
        .absolute()
        .right(px(24.0))
        .bottom(px(24.0))
        .w(px(360.0))
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .rounded_xl()
        .bg(hsla(0.0, 0.0, 0.08, 0.72))
        .border_1()
        .border_color(hsla(0.0, 0.0, 1.0, 0.18))
        .shadow_lg()
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .on_mouse_up(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(hsla(0.0, 0.0, 1.0, 0.95))
                .child("F1-P 标签带取区"),
        )
        .child(
            div()
                .text_xs()
                .line_height(px(18.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.82))
                .child(if has_selection {
                    "确认后会把当前矩形区域直接抓成探针模板。请只框住 F1 到 P 这排标签，尽量不要带上方图标。"
                } else {
                    "按住左键拖出一个矩形，只框住 F1 到 P 这排标签。右键或取消可退出，不满意可点重画。"
                }),
        )
        .child(
            div()
                .flex()
                .items_center()
                .justify_end()
                .gap_2()
                .child(picker_control_button(
                    "probe-picker-reset",
                    "重画",
                    hsla(0.58, 0.10, 0.32, 0.90),
                    hsla(0.58, 0.12, 0.42, 0.96),
                    hsla(0.0, 0.0, 1.0, 0.18),
                    has_selection,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.reset_selection(cx);
                    }),
                ))
                .child(picker_control_button(
                    "probe-picker-cancel",
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
                    "probe-picker-confirm",
                    "确认",
                    hsla(0.36, 0.68, 0.48, 0.92),
                    hsla(0.36, 0.72, 0.56, 0.98),
                    hsla(0.36, 0.80, 0.70, 0.34),
                    has_selection,
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.confirm_selection(window, cx);
                    }),
                )),
        )
}

fn picker_control_button(
    id: impl Into<SharedString>,
    label: &'static str,
    background: gpui::Hsla,
    hover_background: gpui::Hsla,
    border_color: gpui::Hsla,
    enabled: bool,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    let button = div()
        .id(id.into())
        .h(px(34.0))
        .px_3()
        .min_w(px(72.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .on_mouse_up(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(hsla(0.0, 0.0, 1.0, 0.96))
                .child(label),
        );

    if enabled {
        button
            .cursor(CursorStyle::PointingHand)
            .hover(move |style| style.bg(hover_background))
            .active(|style| style.opacity(0.92))
            .on_click(
                move |event: &ClickEvent, window: &mut Window, cx: &mut App| {
                    cx.stop_propagation();
                    on_click(event, window, cx);
                },
            )
    } else {
        button.opacity(0.42)
    }
}

fn rect_from_drag(start: Point<Pixels>, end: Point<Pixels>) -> Bounds<Pixels> {
    let left = f32::from(start.x).min(f32::from(end.x));
    let top = f32::from(start.y).min(f32::from(end.y));
    let right = f32::from(start.x).max(f32::from(end.x));
    let bottom = f32::from(start.y).max(f32::from(end.y));

    Bounds {
        origin: point(px(left), px(top)),
        size: size(px((right - left).max(1.0)), px((bottom - top).max(1.0))),
    }
}

fn clamp_rect_to_bounds(selection: Bounds<Pixels>, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    let left =
        f32::from(selection.left()).clamp(f32::from(bounds.left()), f32::from(bounds.right()));
    let top = f32::from(selection.top()).clamp(f32::from(bounds.top()), f32::from(bounds.bottom()));
    let right =
        f32::from(selection.right()).clamp(f32::from(bounds.left()), f32::from(bounds.right()));
    let bottom =
        f32::from(selection.bottom()).clamp(f32::from(bounds.top()), f32::from(bounds.bottom()));

    Bounds {
        origin: point(px(left.min(right)), px(top.min(bottom))),
        size: size(
            px((right - left).abs().max(1.0)),
            px((bottom - top).abs().max(1.0)),
        ),
    }
}

fn paint_selection(window: &mut Window, selection: Bounds<Pixels>) {
    if let Some(path) = rect_path(PathBuilder::stroke(px(OUTLINE_STROKE_WIDTH)), selection) {
        window.paint_path(path, hsla(0.0, 0.0, 1.0, 0.98));
    }
}

fn rect_path(mut builder: PathBuilder, bounds: Bounds<Pixels>) -> Option<gpui::Path<Pixels>> {
    builder.move_to(bounds.origin);
    builder.line_to(point(bounds.right(), bounds.top()));
    builder.line_to(point(bounds.right(), bounds.bottom()));
    builder.line_to(point(bounds.left(), bounds.bottom()));
    builder.close();
    builder.build().ok()
}
