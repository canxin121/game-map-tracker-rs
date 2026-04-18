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
const INNER_OUTLINE_STROKE_WIDTH: f32 = 2.0;
const MIN_RING_THICKNESS: f32 = 12.0;
const MOVE_HANDLE_RADIUS: f32 = 7.0;
const MOVE_HANDLE_HIT_RADIUS: f32 = 12.0;
const OUTER_RESIZE_HIT_TOLERANCE: f32 = 14.0;
const INNER_RESIZE_HIT_TOLERANCE: f32 = 10.0;
const MIN_INNER_RESIZE_HIT_TOLERANCE: f32 = 4.0;
const MIN_OUTER_SELECTION_RATIO: f32 = 0.1;

#[derive(Clone, Copy, Debug)]
struct CircleSelection {
    center: Point<Pixels>,
    radius: Pixels,
    inner_ratio: f32,
}

impl CircleSelection {
    fn inner_radius(self) -> Pixels {
        px(f32::from(self.radius) * self.inner_ratio)
    }
}

#[derive(Clone, Copy)]
enum DragMode {
    Drawing {
        anchor: Point<Pixels>,
        inner_ratio: f32,
    },
    Moving {
        pointer_offset: Point<Pixels>,
    },
    ResizingOuter,
    ResizingInner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HitTarget {
    None,
    Move,
    ResizeOuter,
    ResizeInner,
}

#[derive(Clone)]
pub(super) struct MinimapRegionPickResult {
    pub(super) region: CaptureRegion,
    pub(super) mask_inner_radius: f32,
    pub(super) mask_outer_radius: f32,
}

pub(super) struct MinimapRegionPicker {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    display_bounds: Bounds<Pixels>,
    selection: Option<CircleSelection>,
    drag_mode: Option<DragMode>,
    pointer_position: Option<Point<Pixels>>,
    default_inner_ratio: f32,
}

impl MinimapRegionPicker {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        main_window_handle: AnyWindowHandle,
        display_bounds: Bounds<Pixels>,
        minimap_region: CaptureRegion,
        mask_inner_radius: f32,
        mask_outer_radius: f32,
    ) -> Self {
        let default_inner_ratio = normalized_inner_ratio(mask_inner_radius, mask_outer_radius);
        let selection = selection_from_existing_region(
            display_bounds,
            &minimap_region,
            mask_inner_radius,
            mask_outer_radius,
        );
        Self {
            workbench,
            main_window_handle,
            display_bounds,
            selection,
            drag_mode: None,
            pointer_position: None,
            default_inner_ratio,
        }
    }

    fn begin_drag(&mut self, event: &MouseDownEvent) {
        self.pointer_position = Some(event.position);

        let Some(selection) = self.selection else {
            self.drag_mode = Some(DragMode::Drawing {
                anchor: event.position,
                inner_ratio: self.default_inner_ratio,
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
            HitTarget::ResizeOuter => Some(DragMode::ResizingOuter),
            HitTarget::ResizeInner => Some(DragMode::ResizingInner),
            HitTarget::None => Some(DragMode::Drawing {
                anchor: event.position,
                inner_ratio: self.default_inner_ratio,
            }),
        };
    }

    fn update_drag(&mut self, event: &MouseMoveEvent) {
        self.pointer_position = Some(event.position);

        match self.drag_mode {
            Some(DragMode::Drawing {
                anchor,
                inner_ratio,
            }) => {
                self.selection = Some(clamp_circle_to_bounds(
                    circle_from_drag(anchor, event.position, inner_ratio),
                    self.local_bounds(),
                ));
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
                            inner_ratio: selection.inner_ratio,
                        },
                        self.local_bounds(),
                    ));
                }
            }
            Some(DragMode::ResizingOuter) => {
                if let Some(selection) = self.selection {
                    let next_radius = distance(selection.center, event.position).clamp(
                        MIN_SELECTION_SIZE / 2.0,
                        max_radius_for_center(selection.center, self.local_bounds()),
                    );
                    self.selection = Some(CircleSelection {
                        center: selection.center,
                        radius: px(next_radius),
                        inner_ratio: clamp_inner_ratio(selection.inner_ratio, px(next_radius)),
                    });
                }
            }
            Some(DragMode::ResizingInner) => {
                if let Some(selection) = self.selection {
                    let next_inner_ratio = inner_ratio_for_pointer(selection, event.position);
                    self.selection = Some(CircleSelection {
                        center: selection.center,
                        radius: selection.radius,
                        inner_ratio: next_inner_ratio,
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

        let result = self.pick_result_for(selection);
        self.commit_region(result, window, cx);
    }

    fn commit_region(
        &mut self,
        result: MinimapRegionPickResult,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let workbench = self.workbench.clone();
        let main_window_handle = self.main_window_handle;

        let _ = main_window_handle.update(cx, move |_, main_window, cx| {
            if let Some(workbench) = workbench.upgrade() {
                let _ = workbench.update(cx, |this, cx| {
                    this.finish_minimap_region_pick(result, main_window, cx);
                });
            }
        });

        window.remove_window();
    }

    fn pick_result_for(&self, selection: CircleSelection) -> MinimapRegionPickResult {
        pick_result_for_selection(self.display_bounds, selection)
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
            Some(DragMode::ResizingOuter | DragMode::ResizingInner) => CursorStyle::ResizeLeftRight,
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
                    HitTarget::ResizeOuter | HitTarget::ResizeInner => CursorStyle::ResizeLeftRight,
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
                .child("小地图环形取区"),
        )
        .child(
            div()
                .text_xs()
                .line_height(px(18.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.82))
                .child(if has_selection {
                    "拖动圆心可整体移动，拖动外圈改截图范围，拖动内圈改中心挖空。确认后会保存外圈截图和环形遮罩。"
                } else {
                    "先按住左键拖出外圈，再继续微调外圈和内圈。右键或取消按钮可退出。"
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

fn circle_from_drag(start: Point<Pixels>, end: Point<Pixels>, inner_ratio: f32) -> CircleSelection {
    let bounds = circular_selection_bounds(start, end);
    let radius = px(f32::from(bounds.size.width).min(f32::from(bounds.size.height)) / 2.0);
    CircleSelection {
        center: point(
            bounds.origin.x + bounds.size.width / 2.0,
            bounds.origin.y + bounds.size.height / 2.0,
        ),
        radius,
        inner_ratio: clamp_inner_ratio(inner_ratio, radius),
    }
}

fn translate_circle(selection: CircleSelection, origin: Point<Pixels>) -> CircleSelection {
    CircleSelection {
        center: point(origin.x + selection.center.x, origin.y + selection.center.y),
        radius: selection.radius,
        inner_ratio: selection.inner_ratio,
    }
}

fn hit_target(selection: CircleSelection, position: Point<Pixels>) -> HitTarget {
    let distance = distance(selection.center, position);
    let radius = f32::from(selection.radius);
    let inner_radius = f32::from(selection.inner_radius());

    if inner_radius > 0.0
        && (distance - inner_radius).abs() <= inner_resize_hit_tolerance(inner_radius)
    {
        return HitTarget::ResizeInner;
    }

    if (distance - radius).abs() <= OUTER_RESIZE_HIT_TOLERANCE {
        return HitTarget::ResizeOuter;
    }

    let move_radius = MOVE_HANDLE_HIT_RADIUS.min(radius);
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

fn inner_resize_hit_tolerance(inner_radius: f32) -> f32 {
    INNER_RESIZE_HIT_TOLERANCE
        .min((inner_radius - MOVE_HANDLE_HIT_RADIUS - 2.0).max(MIN_INNER_RESIZE_HIT_TOLERANCE))
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

fn max_inner_ratio_for_radius(radius: Pixels) -> f32 {
    let radius = f32::from(radius).max(1.0);
    ((radius - MIN_RING_THICKNESS).max(0.0) / radius).clamp(0.0, 1.0)
}

fn clamp_inner_ratio(inner_ratio: f32, radius: Pixels) -> f32 {
    inner_ratio.clamp(0.0, max_inner_ratio_for_radius(radius))
}

fn normalized_inner_ratio(mask_inner_radius: f32, mask_outer_radius: f32) -> f32 {
    if mask_outer_radius <= f32::EPSILON {
        0.0
    } else {
        (mask_inner_radius / mask_outer_radius).clamp(0.0, 1.0)
    }
}

fn inner_ratio_for_pointer(selection: CircleSelection, position: Point<Pixels>) -> f32 {
    let normalized = distance(selection.center, position) / f32::from(selection.radius).max(1.0);
    clamp_inner_ratio(normalized, selection.radius)
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
        inner_ratio: clamp_inner_ratio(selection.inner_ratio, px(radius)),
    }
}

fn paint_selection(window: &mut Window, selection: CircleSelection) {
    if let Some(path) = circle_path(PathBuilder::fill(), selection.center, selection.radius) {
        window.paint_path(path, hsla(0.56, 0.76, 0.82, 0.14));
    }

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

    let inner_radius = selection.inner_radius();
    if f32::from(inner_radius) > 1.0 {
        if let Some(path) = circle_path(
            PathBuilder::fill(),
            selection.center,
            px((f32::from(inner_radius) - INNER_OUTLINE_STROKE_WIDTH).max(1.0)),
        ) {
            window.paint_path(path, hsla(0.0, 0.0, 0.0, 0.54));
        }
        if let Some(path) = circle_path(
            PathBuilder::stroke(px(INNER_OUTLINE_STROKE_WIDTH)),
            selection.center,
            px((f32::from(inner_radius) - INNER_OUTLINE_STROKE_WIDTH / 2.0).max(1.0)),
        ) {
            window.paint_path(path, hsla(0.12, 0.86, 0.60, 0.98));
        }
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

fn pick_result_for_selection(
    display_bounds: Bounds<Pixels>,
    selection: CircleSelection,
) -> MinimapRegionPickResult {
    let bounds = circle_selection_bounds(selection);
    MinimapRegionPickResult {
        region: CaptureRegion {
            top: (f32::from(display_bounds.origin.y) + f32::from(bounds.origin.y)).round() as i32,
            left: (f32::from(display_bounds.origin.x) + f32::from(bounds.origin.x)).round() as i32,
            width: f32::from(bounds.size.width).round().max(1.0) as u32,
            height: f32::from(bounds.size.height).round().max(1.0) as u32,
        },
        mask_inner_radius: selection.inner_ratio,
        mask_outer_radius: 1.0,
    }
}

fn selection_from_existing_region(
    display_bounds: Bounds<Pixels>,
    region: &CaptureRegion,
    mask_inner_radius: f32,
    mask_outer_radius: f32,
) -> Option<CircleSelection> {
    if region.width == 0 || region.height == 0 {
        return None;
    }

    let base_radius = region.width.min(region.height) as f32 * 0.5;
    if base_radius <= 0.0 {
        return None;
    }

    let outer_ratio = if mask_outer_radius <= f32::EPSILON {
        1.0
    } else {
        mask_outer_radius.clamp(MIN_OUTER_SELECTION_RATIO, 1.0)
    };
    let selection = CircleSelection {
        center: point(
            px(region.left as f32 - f32::from(display_bounds.origin.x) + region.width as f32 * 0.5),
            px(region.top as f32 - f32::from(display_bounds.origin.y) + region.height as f32 * 0.5),
        ),
        radius: px((base_radius * outer_ratio).max(MIN_SELECTION_SIZE * 0.5)),
        inner_ratio: normalized_inner_ratio(mask_inner_radius, outer_ratio),
    };

    Some(clamp_circle_to_bounds(
        selection,
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: display_bounds.size,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_selection() -> CircleSelection {
        CircleSelection {
            center: point(px(120.0), px(80.0)),
            radius: px(100.0),
            inner_ratio: 0.2,
        }
    }

    #[test]
    fn existing_region_is_migrated_to_direct_annulus_selection() {
        let display_bounds = Bounds {
            origin: point(px(0.0), px(0.0)),
            size: size(px(500.0), px(400.0)),
        };
        let selection = selection_from_existing_region(
            display_bounds,
            &CaptureRegion {
                top: 40,
                left: 60,
                width: 200,
                height: 200,
            },
            0.16,
            0.8,
        )
        .expect("selection");

        assert!((f32::from(selection.radius) - 80.0).abs() < 0.1);
        assert!((selection.inner_ratio - 0.2).abs() < 0.001);
    }

    #[test]
    fn pick_result_saves_outer_circle_as_capture_box() {
        let result = pick_result_for_selection(
            Bounds {
                origin: point(px(100.0), px(50.0)),
                size: size(px(600.0), px(400.0)),
            },
            test_selection(),
        );

        assert_eq!(
            result.region,
            CaptureRegion {
                top: 30,
                left: 120,
                width: 200,
                height: 200,
            }
        );
        assert!((result.mask_inner_radius - 0.2).abs() < 0.001);
        assert!((result.mask_outer_radius - 1.0).abs() < 0.001);
    }

    #[test]
    fn inner_ring_hover_wins_over_move_handle() {
        let selection = CircleSelection {
            center: point(px(100.0), px(100.0)),
            radius: px(90.0),
            inner_ratio: 0.18,
        };

        let target = hit_target(selection, point(px(116.0), px(100.0)));
        assert_eq!(target, HitTarget::ResizeInner);
        assert_eq!(
            hit_target(selection, point(px(100.0), px(100.0))),
            HitTarget::Move
        );
    }

    #[test]
    fn shrinking_outer_radius_clamps_inner_ratio_to_leave_ring_thickness() {
        let selection = clamp_circle_to_bounds(
            CircleSelection {
                center: point(px(40.0), px(40.0)),
                radius: px(20.0),
                inner_ratio: 0.85,
            },
            Bounds {
                origin: point(px(0.0), px(0.0)),
                size: size(px(80.0), px(80.0)),
            },
        );

        assert!(selection.inner_ratio <= max_inner_ratio_for_radius(selection.radius));
        assert!(
            f32::from(selection.radius) * (1.0 - selection.inner_ratio)
                >= MIN_RING_THICKNESS - 0.01
        );
    }
}
