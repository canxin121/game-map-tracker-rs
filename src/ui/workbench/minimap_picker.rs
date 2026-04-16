use gpui::{
    AnyWindowHandle, App, Bounds, ClickEvent, Context, CursorStyle, InteractiveElement as _,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _,
    PathBuilder, Pixels, Point, Render, SharedString, StatefulInteractiveElement as _, Styled,
    Window, canvas, div, hsla, point, px, size, transparent_black,
};

use crate::config::CaptureRegion;

use super::TrackerWorkbench;

const MIN_SELECTION_SIZE: f32 = 24.0;
const OUTLINE_STROKE_WIDTH: f32 = 3.0;
const MOVE_HANDLE_RADIUS: f32 = 7.0;
const MOVE_HANDLE_HIT_RADIUS: f32 = 18.0;
const RESIZE_HIT_TOLERANCE: f32 = 14.0;

#[derive(Clone, Copy)]
struct CircleSelection {
    center: Point<Pixels>,
    radius: Pixels,
}

#[derive(Clone, Copy)]
enum DragMode {
    Drawing { anchor: Point<Pixels> },
    Moving { pointer_offset: Point<Pixels> },
    Resizing,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HitTarget {
    None,
    Move,
    Resize,
}

pub(super) struct MinimapRegionPicker {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    display_bounds: Bounds<Pixels>,
    selection: Option<CircleSelection>,
    drag_mode: Option<DragMode>,
    pointer_position: Option<Point<Pixels>>,
}

impl MinimapRegionPicker {
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
            drag_mode: None,
            pointer_position: None,
        }
    }

    fn begin_drag(&mut self, event: &MouseDownEvent) {
        self.pointer_position = Some(event.position);

        let Some(selection) = self.selection else {
            self.drag_mode = Some(DragMode::Drawing {
                anchor: event.position,
            });
            return;
        };

        self.drag_mode = match hit_target(selection, event.position) {
            HitTarget::Move => Some(DragMode::Moving {
                pointer_offset: point(
                    event.position.x - selection.center.x,
                    event.position.y - selection.center.y,
                ),
            }),
            HitTarget::Resize => Some(DragMode::Resizing),
            HitTarget::None => None,
        };
    }

    fn update_drag(&mut self, event: &MouseMoveEvent) {
        self.pointer_position = Some(event.position);

        match self.drag_mode {
            Some(DragMode::Drawing { anchor }) => {
                self.selection = Some(circle_from_drag(anchor, event.position));
            }
            Some(DragMode::Moving { pointer_offset }) => {
                if let Some(selection) = self.selection {
                    self.selection = Some(clamp_circle_to_bounds(
                        CircleSelection {
                            center: point(
                                event.position.x - pointer_offset.x,
                                event.position.y - pointer_offset.y,
                            ),
                            radius: selection.radius,
                        },
                        self.local_bounds(),
                    ));
                }
            }
            Some(DragMode::Resizing) => {
                if let Some(selection) = self.selection {
                    let next_radius = distance(selection.center, event.position).clamp(
                        MIN_SELECTION_SIZE / 2.0,
                        max_radius_for_center(selection.center, self.local_bounds()),
                    );
                    self.selection = Some(CircleSelection {
                        center: selection.center,
                        radius: px(next_radius),
                    });
                }
            }
            None => {}
        }
    }

    fn finish_drag(&mut self, event: &MouseUpEvent) {
        self.pointer_position = Some(event.position);

        if let Some(DragMode::Drawing { .. }) = self.drag_mode {
            if self
                .selection
                .is_some_and(|selection| f32::from(selection.radius) * 2.0 < MIN_SELECTION_SIZE)
            {
                self.selection = None;
            }
        }

        self.drag_mode = None;
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.handle_minimap_region_picker_cancelled();
            });
        }
        window.remove_window();
    }

    fn reset_selection(&mut self, cx: &mut Context<Self>) {
        self.selection = None;
        self.drag_mode = None;
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
                    this.finish_minimap_region_pick(region, main_window, cx);
                });
            }
        });

        window.remove_window();
    }

    fn capture_region_for(&self, selection: CircleSelection) -> CaptureRegion {
        let bounds = circle_selection_bounds(selection);
        CaptureRegion {
            top: (f32::from(self.display_bounds.origin.y) + f32::from(bounds.origin.y)).round()
                as i32,
            left: (f32::from(self.display_bounds.origin.x) + f32::from(bounds.origin.x)).round()
                as i32,
            width: f32::from(bounds.size.width).round().max(1.0) as u32,
            height: f32::from(bounds.size.height).round().max(1.0) as u32,
        }
    }

    fn local_bounds(&self) -> Bounds<Pixels> {
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: self.display_bounds.size,
        }
    }

    fn cursor_style(&self) -> CursorStyle {
        match self.drag_mode {
            Some(DragMode::Drawing { .. }) => CursorStyle::Crosshair,
            Some(DragMode::Moving { .. }) => CursorStyle::ClosedHand,
            Some(DragMode::Resizing) => CursorStyle::ResizeLeftRight,
            None => {
                let Some(selection) = self.selection else {
                    return CursorStyle::Crosshair;
                };

                match self
                    .pointer_position
                    .map(|position| hit_target(selection, position))
                    .unwrap_or(HitTarget::None)
                {
                    HitTarget::Move => CursorStyle::OpenHand,
                    HitTarget::Resize => CursorStyle::ResizeLeftRight,
                    HitTarget::None => CursorStyle::Crosshair,
                }
            }
        }
    }
}

impl Render for MinimapRegionPicker {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let has_selection = self.selection.is_some();

        div()
            .relative()
            .size_full()
            .bg(transparent_black())
            .cursor(self.cursor_style())
            .child(
                canvas(
                    |_, _, _| {},
                    move |bounds, _, window, cx| {
                        let selection = entity
                            .read(cx)
                            .selection
                            .map(|selection| translate_circle(selection, bounds.origin));

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
                    this.begin_drag(event);
                    cx.notify();
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    this.cancel(window, cx);
                }),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                this.update_drag(event);
                cx.notify();
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, _, cx| {
                    this.finish_drag(event);
                    cx.notify();
                }),
            )
            .child(picker_controls(has_selection, cx))
    }
}

fn picker_controls(has_selection: bool, cx: &mut Context<MinimapRegionPicker>) -> impl IntoElement {
    div()
        .absolute()
        .right(px(24.0))
        .bottom(px(24.0))
        .w(px(320.0))
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
                .child("小地图圆形取区"),
        )
        .child(
            div()
                .text_xs()
                .line_height(px(18.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.82))
                .child(if has_selection {
                    "拖动圆心可整体移动，拖动圆边可调整半径。确认后会保存为圆的外接正方形截图范围。"
                } else {
                    "先按住左键拖出一个圆，再继续微调位置和半径。右键或取消按钮可退出。"
                }),
        )
        .child(
            div()
                .flex()
                .items_center()
                .justify_end()
                .gap_2()
                .child(picker_control_button(
                    "minimap-picker-reset",
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
                    "minimap-picker-cancel",
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
                    "minimap-picker-confirm",
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

fn circular_selection_bounds(start: Point<Pixels>, end: Point<Pixels>) -> Bounds<Pixels> {
    let start_x = f32::from(start.x);
    let start_y = f32::from(start.y);
    let end_x = f32::from(end.x);
    let end_y = f32::from(end.y);
    let delta_x = end_x - start_x;
    let delta_y = end_y - start_y;
    let diameter = delta_x.abs().min(delta_y.abs()).max(1.0);

    Bounds {
        origin: point(
            px(if delta_x >= 0.0 {
                start_x
            } else {
                start_x - diameter
            }),
            px(if delta_y >= 0.0 {
                start_y
            } else {
                start_y - diameter
            }),
        ),
        size: size(px(diameter), px(diameter)),
    }
}

fn circle_selection_bounds(selection: CircleSelection) -> Bounds<Pixels> {
    Bounds {
        origin: point(
            selection.center.x - selection.radius,
            selection.center.y - selection.radius,
        ),
        size: size(selection.radius * 2.0, selection.radius * 2.0),
    }
}

fn circle_from_drag(start: Point<Pixels>, end: Point<Pixels>) -> CircleSelection {
    let bounds = circular_selection_bounds(start, end);
    CircleSelection {
        center: point(
            bounds.origin.x + bounds.size.width / 2.0,
            bounds.origin.y + bounds.size.height / 2.0,
        ),
        radius: px(f32::from(bounds.size.width).min(f32::from(bounds.size.height)) / 2.0),
    }
}

fn translate_circle(selection: CircleSelection, origin: Point<Pixels>) -> CircleSelection {
    CircleSelection {
        center: point(origin.x + selection.center.x, origin.y + selection.center.y),
        radius: selection.radius,
    }
}

fn hit_target(selection: CircleSelection, position: Point<Pixels>) -> HitTarget {
    let distance = distance(selection.center, position);
    let radius = f32::from(selection.radius);
    if (distance - radius).abs() <= RESIZE_HIT_TOLERANCE {
        return HitTarget::Resize;
    }

    let move_radius = (radius * 0.35).max(MOVE_HANDLE_HIT_RADIUS).min(radius);
    if distance <= move_radius {
        HitTarget::Move
    } else {
        HitTarget::None
    }
}

fn distance(a: Point<Pixels>, b: Point<Pixels>) -> f32 {
    let dx = f32::from(a.x - b.x);
    let dy = f32::from(a.y - b.y);
    dx.hypot(dy)
}

fn max_radius_for_center(center: Point<Pixels>, bounds: Bounds<Pixels>) -> f32 {
    let center_x = f32::from(center.x);
    let center_y = f32::from(center.y);
    let left = center_x - f32::from(bounds.left());
    let right = f32::from(bounds.right()) - center_x;
    let top = center_y - f32::from(bounds.top());
    let bottom = f32::from(bounds.bottom()) - center_y;
    left.min(right)
        .min(top)
        .min(bottom)
        .max(MIN_SELECTION_SIZE / 2.0)
}

fn clamp_circle_to_bounds(selection: CircleSelection, bounds: Bounds<Pixels>) -> CircleSelection {
    let max_supported_radius = (f32::from(bounds.size.width).min(f32::from(bounds.size.height))
        / 2.0)
        .max(MIN_SELECTION_SIZE / 2.0);
    let radius = f32::from(selection.radius).clamp(MIN_SELECTION_SIZE / 2.0, max_supported_radius);

    let min_x = f32::from(bounds.left()) + radius;
    let max_x = f32::from(bounds.right()) - radius;
    let min_y = f32::from(bounds.top()) + radius;
    let max_y = f32::from(bounds.bottom()) - radius;

    CircleSelection {
        center: point(
            px(f32::from(selection.center.x).clamp(min_x, max_x)),
            px(f32::from(selection.center.y).clamp(min_y, max_y)),
        ),
        radius: px(radius),
    }
}

fn paint_selection(window: &mut Window, selection: CircleSelection) {
    let outline_radius =
        px((f32::from(selection.radius) - OUTLINE_STROKE_WIDTH / 2.0)
            .max(MOVE_HANDLE_RADIUS + 2.0));
    if let Some(path) = circle_path(
        PathBuilder::stroke(px(OUTLINE_STROKE_WIDTH)),
        selection.center,
        outline_radius,
    ) {
        window.paint_path(path, hsla(0.0, 0.0, 1.0, 0.98));
    }

    if let Some(path) = circle_path(
        PathBuilder::fill(),
        selection.center,
        px(MOVE_HANDLE_RADIUS - 2.0),
    ) {
        window.paint_path(path, hsla(0.56, 0.76, 0.86, 0.90));
    }

    if let Some(path) = circle_path(
        PathBuilder::stroke(px(2.0)),
        selection.center,
        px(MOVE_HANDLE_RADIUS),
    ) {
        window.paint_path(path, hsla(0.0, 0.0, 1.0, 0.98));
    }
}

fn circle_path(
    mut builder: PathBuilder,
    center: Point<Pixels>,
    radius: Pixels,
) -> Option<gpui::Path<Pixels>> {
    if f32::from(radius) <= 0.0 {
        return None;
    }

    builder.move_to(point(center.x + radius, center.y));
    builder.arc_to(
        point(radius, radius),
        px(0.0),
        false,
        false,
        point(center.x - radius, center.y),
    );
    builder.arc_to(
        point(radius, radius),
        px(0.0),
        false,
        false,
        point(center.x + radius, center.y),
    );
    builder.close();
    builder.build().ok()
}
