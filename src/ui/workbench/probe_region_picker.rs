use gpui::{
    Bounds, ClickEvent, Context, CursorStyle, InteractiveElement as _, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Point,
    Render, Styled, Window, canvas, div, fill, hsla, point, px, size, transparent_black,
};

use crate::config::CaptureRegion;

use super::{
    TrackerWorkbench,
    picker_geometry::{capture_region_from_selection_bounds, selection_bounds_from_capture_region},
    picker_shared::{close_picker, picker_control_button},
};

const MIN_SELECTION_SIZE: f32 = 18.0;
const EDGE_HIT_TOLERANCE: f32 = 10.0;
const MOVE_HANDLE_SIZE: f32 = 10.0;
const OUTLINE_STROKE_WIDTH: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResizeEdges {
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
}

impl ResizeEdges {
    const fn any(self) -> bool {
        self.left || self.right || self.top || self.bottom
    }
}

#[derive(Clone, Copy, Debug)]
enum DragMode {
    Drawing {
        anchor: Point<Pixels>,
    },
    Moving {
        pointer_offset: Point<Pixels>,
    },
    Resizing {
        anchor: Point<Pixels>,
        initial_bounds: Bounds<Pixels>,
        edges: ResizeEdges,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HitTarget {
    None,
    Move,
    Resize(ResizeEdges),
}

pub(super) struct MinimapPresenceProbePicker {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    picker_bounds: Bounds<Pixels>,
    selection: Option<Bounds<Pixels>>,
    drag_mode: Option<DragMode>,
    pointer_position: Option<Point<Pixels>>,
}

impl MinimapPresenceProbePicker {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        picker_bounds: Bounds<Pixels>,
        probe_region: Option<CaptureRegion>,
    ) -> Self {
        Self {
            workbench,
            picker_bounds,
            selection: probe_region
                .and_then(|region| selection_from_existing_region(picker_bounds, &region)),
            drag_mode: None,
            pointer_position: None,
        }
    }

    fn begin_drag(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        self.pointer_position = Some(event.position);

        let next_drag_mode = match self.selection {
            Some(selection) => match hit_target(selection, event.position) {
                HitTarget::Move => Some(DragMode::Moving {
                    pointer_offset: point(
                        event.position.x - selection.origin.x,
                        event.position.y - selection.origin.y,
                    ),
                }),
                HitTarget::Resize(edges) => Some(DragMode::Resizing {
                    anchor: event.position,
                    initial_bounds: selection,
                    edges,
                }),
                HitTarget::None => Some(DragMode::Drawing {
                    anchor: event.position,
                }),
            },
            None => Some(DragMode::Drawing {
                anchor: event.position,
            }),
        };

        self.drag_mode = next_drag_mode;
        if let Some(DragMode::Drawing { anchor }) = self.drag_mode {
            self.selection = Some(clamp_rect_to_bounds(
                rect_from_drag(anchor, anchor),
                self.local_bounds(),
            ));
        }
        cx.notify();
    }

    fn update_drag(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        self.pointer_position = Some(event.position);

        match self.drag_mode {
            Some(DragMode::Drawing { anchor }) => {
                self.selection = Some(clamp_rect_to_bounds(
                    rect_from_drag(anchor, event.position),
                    self.local_bounds(),
                ));
            }
            Some(DragMode::Moving { pointer_offset }) => {
                if let Some(selection) = self.selection {
                    self.selection = Some(move_selection(
                        selection,
                        pointer_offset,
                        event.position,
                        self.local_bounds(),
                    ));
                }
            }
            Some(DragMode::Resizing {
                anchor,
                initial_bounds,
                edges,
            }) => {
                self.selection = Some(resize_selection(
                    initial_bounds,
                    anchor,
                    event.position,
                    edges,
                    self.local_bounds(),
                ));
            }
            None => {}
        }

        cx.notify();
    }

    fn finish_drag(&mut self, event: &MouseUpEvent, cx: &mut Context<Self>) {
        self.pointer_position = Some(event.position);
        if let Some(selection) = self.selection {
            if selection_too_small(selection) {
                self.selection = None;
            }
        }
        self.drag_mode = None;
        cx.notify();
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        close_picker(&self.workbench, window, cx, |this, _| {
            this.handle_minimap_presence_probe_picker_cancelled();
        });
    }

    fn reset_selection(&mut self, cx: &mut Context<Self>) {
        self.selection = None;
        self.drag_mode = None;
        self.pointer_position = None;
        cx.notify();
    }

    fn confirm_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.selection else {
            return;
        };

        let region = capture_region_for_selection(self.picker_bounds, selection);
        self.build_preview(region, window, cx);
    }

    fn build_preview(
        &mut self,
        region: CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.prepare_minimap_presence_probe_model_preview();
            });
        }

        let workbench = self.workbench.clone();
        window.defer(cx, move |window, cx| {
            window.remove_window();
            if let Some(workbench) = workbench.upgrade() {
                let _ = workbench.update(cx, |this, cx| {
                    this.begin_minimap_presence_probe_model_preview(region.clone(), cx);
                });
            }
        });
    }

    fn local_bounds(&self) -> Bounds<Pixels> {
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: self.picker_bounds.size,
        }
    }

    fn cursor_style(&self) -> CursorStyle {
        match self.drag_mode {
            Some(DragMode::Drawing { .. }) => CursorStyle::Crosshair,
            Some(DragMode::Moving { .. }) => CursorStyle::ClosedHand,
            Some(DragMode::Resizing { .. }) => CursorStyle::ResizeLeftRight,
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
                    HitTarget::Resize(_) => CursorStyle::ResizeLeftRight,
                    HitTarget::None => CursorStyle::Crosshair,
                }
            }
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
            .cursor(self.cursor_style())
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
                    "拖动边框可改尺寸，拖动中间可整体平移。确认后会先执行建模预览，再由你手动确认是否保存，请尽量不要带上方图标。"
                } else {
                    "按住左键拖出一个矩形，只框住 F1 到 P 这排标签。再次打开会显示上次保存的选区。"
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

fn selection_from_existing_region(
    display_bounds: Bounds<Pixels>,
    region: &CaptureRegion,
) -> Option<Bounds<Pixels>> {
    Some(clamp_rect_to_bounds(
        selection_bounds_from_capture_region(display_bounds, region)?,
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: display_bounds.size,
        },
    ))
}

fn capture_region_for_selection(
    display_bounds: Bounds<Pixels>,
    selection: Bounds<Pixels>,
) -> CaptureRegion {
    capture_region_from_selection_bounds(display_bounds, selection)
}

fn selection_too_small(selection: Bounds<Pixels>) -> bool {
    f32::from(selection.size.width) < MIN_SELECTION_SIZE
        || f32::from(selection.size.height) < MIN_SELECTION_SIZE
}

fn rect_from_drag(anchor: Point<Pixels>, current: Point<Pixels>) -> Bounds<Pixels> {
    let left = f32::from(anchor.x).min(f32::from(current.x));
    let top = f32::from(anchor.y).min(f32::from(current.y));
    let right = f32::from(anchor.x).max(f32::from(current.x));
    let bottom = f32::from(anchor.y).max(f32::from(current.y));
    bounds_from_edges(left, top, right, bottom)
}

fn move_selection(
    selection: Bounds<Pixels>,
    pointer_offset: Point<Pixels>,
    pointer: Point<Pixels>,
    bounds: Bounds<Pixels>,
) -> Bounds<Pixels> {
    let width = f32::from(selection.size.width);
    let height = f32::from(selection.size.height);
    let max_left = (f32::from(bounds.size.width) - width).max(0.0);
    let max_top = (f32::from(bounds.size.height) - height).max(0.0);
    let left = (f32::from(pointer.x) - f32::from(pointer_offset.x)).clamp(0.0, max_left);
    let top = (f32::from(pointer.y) - f32::from(pointer_offset.y)).clamp(0.0, max_top);

    Bounds {
        origin: point(px(left), px(top)),
        size: selection.size,
    }
}

fn resize_selection(
    initial_bounds: Bounds<Pixels>,
    anchor: Point<Pixels>,
    pointer: Point<Pixels>,
    edges: ResizeEdges,
    local_bounds: Bounds<Pixels>,
) -> Bounds<Pixels> {
    let dx = f32::from(pointer.x) - f32::from(anchor.x);
    let dy = f32::from(pointer.y) - f32::from(anchor.y);
    let local_right_limit = f32::from(local_bounds.size.width);
    let local_bottom_limit = f32::from(local_bounds.size.height);

    let mut left = rect_left(initial_bounds);
    let mut top = rect_top(initial_bounds);
    let mut right = rect_right(initial_bounds);
    let mut bottom = rect_bottom(initial_bounds);

    if edges.left {
        left = (rect_left(initial_bounds) + dx).clamp(0.0, right - MIN_SELECTION_SIZE);
    }
    if edges.right {
        right =
            (rect_right(initial_bounds) + dx).clamp(left + MIN_SELECTION_SIZE, local_right_limit);
    }
    if edges.top {
        top = (rect_top(initial_bounds) + dy).clamp(0.0, bottom - MIN_SELECTION_SIZE);
    }
    if edges.bottom {
        bottom =
            (rect_bottom(initial_bounds) + dy).clamp(top + MIN_SELECTION_SIZE, local_bottom_limit);
    }

    clamp_rect_to_bounds(bounds_from_edges(left, top, right, bottom), local_bounds)
}

fn clamp_rect_to_bounds(rect: Bounds<Pixels>, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    let left = rect_left(rect).clamp(0.0, f32::from(bounds.size.width));
    let top = rect_top(rect).clamp(0.0, f32::from(bounds.size.height));
    let right = rect_right(rect).clamp(0.0, f32::from(bounds.size.width));
    let bottom = rect_bottom(rect).clamp(0.0, f32::from(bounds.size.height));
    bounds_from_edges(left, top, right.max(left), bottom.max(top))
}

fn hit_target(selection: Bounds<Pixels>, pointer: Point<Pixels>) -> HitTarget {
    let left = rect_left(selection);
    let top = rect_top(selection);
    let right = rect_right(selection);
    let bottom = rect_bottom(selection);
    let x = f32::from(pointer.x);
    let y = f32::from(pointer.y);

    if x < left - EDGE_HIT_TOLERANCE
        || x > right + EDGE_HIT_TOLERANCE
        || y < top - EDGE_HIT_TOLERANCE
        || y > bottom + EDGE_HIT_TOLERANCE
    {
        return HitTarget::None;
    }

    let edges = ResizeEdges {
        left: (x - left).abs() <= EDGE_HIT_TOLERANCE,
        right: (x - right).abs() <= EDGE_HIT_TOLERANCE,
        top: (y - top).abs() <= EDGE_HIT_TOLERANCE,
        bottom: (y - bottom).abs() <= EDGE_HIT_TOLERANCE,
    };
    if edges.any() {
        return HitTarget::Resize(edges);
    }

    if x >= left && x <= right && y >= top && y <= bottom {
        HitTarget::Move
    } else {
        HitTarget::None
    }
}

fn paint_selection(window: &mut Window, bounds: Bounds<Pixels>) {
    window.paint_quad(fill(bounds, hsla(0.56, 0.76, 0.82, 0.14)));

    if let Some(path) = rect_path(PathBuilder::stroke(px(OUTLINE_STROKE_WIDTH)), bounds) {
        window.paint_path(path, hsla(0.0, 0.0, 1.0, 0.98));
    }

    let handle_size = px(MOVE_HANDLE_SIZE);
    let half_handle = px(MOVE_HANDLE_SIZE * 0.5);
    let handles = [
        point(bounds.origin.x, bounds.origin.y),
        point(bounds.origin.x + bounds.size.width, bounds.origin.y),
        point(bounds.origin.x, bounds.origin.y + bounds.size.height),
        point(
            bounds.origin.x + bounds.size.width,
            bounds.origin.y + bounds.size.height,
        ),
        point(
            bounds.origin.x + px(f32::from(bounds.size.width) * 0.5),
            bounds.origin.y + px(f32::from(bounds.size.height) * 0.5),
        ),
    ];

    for center in handles {
        let handle_bounds = Bounds {
            origin: point(center.x - half_handle, center.y - half_handle),
            size: size(handle_size, handle_size),
        };
        window.paint_quad(fill(handle_bounds, hsla(0.56, 0.76, 0.86, 0.90)));
        if let Some(path) = rect_path(PathBuilder::stroke(px(1.5)), handle_bounds) {
            window.paint_path(path, hsla(0.0, 0.0, 1.0, 0.98));
        }
    }
}

fn rect_path(mut builder: PathBuilder, bounds: Bounds<Pixels>) -> Option<gpui::Path<Pixels>> {
    builder.move_to(bounds.origin);
    builder.line_to(point(bounds.origin.x + bounds.size.width, bounds.origin.y));
    builder.line_to(point(
        bounds.origin.x + bounds.size.width,
        bounds.origin.y + bounds.size.height,
    ));
    builder.line_to(point(bounds.origin.x, bounds.origin.y + bounds.size.height));
    builder.close();
    builder.build().ok()
}

fn bounds_from_edges(left: f32, top: f32, right: f32, bottom: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(px(left), px(top)),
        size: size(px((right - left).max(0.0)), px((bottom - top).max(0.0))),
    }
}

fn rect_left(bounds: Bounds<Pixels>) -> f32 {
    f32::from(bounds.origin.x)
}

fn rect_top(bounds: Bounds<Pixels>) -> f32 {
    f32::from(bounds.origin.y)
}

fn rect_right(bounds: Bounds<Pixels>) -> f32 {
    f32::from(bounds.origin.x) + f32::from(bounds.size.width)
}

fn rect_bottom(bounds: Bounds<Pixels>) -> f32 {
    f32::from(bounds.origin.y) + f32::from(bounds.size.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_probe_region_round_trips_to_local_selection() {
        let display_bounds = Bounds {
            origin: point(px(1200.0), px(80.0)),
            size: size(px(1600.0), px(900.0)),
        };
        let region = CaptureRegion {
            top: 116,
            left: 1660,
            width: 590,
            height: 38,
        };

        let selection = selection_from_existing_region(display_bounds, &region).expect("selection");
        let round_trip = capture_region_for_selection(display_bounds, selection);

        assert_eq!(round_trip, region);
    }

    #[test]
    fn resizing_left_edge_keeps_minimum_width() {
        let selection = Bounds {
            origin: point(px(40.0), px(50.0)),
            size: size(px(80.0), px(32.0)),
        };

        let resized = resize_selection(
            selection,
            point(px(40.0), px(60.0)),
            point(px(200.0), px(60.0)),
            ResizeEdges {
                left: true,
                ..ResizeEdges::default()
            },
            Bounds {
                origin: point(px(0.0), px(0.0)),
                size: size(px(300.0), px(200.0)),
            },
        );

        assert!((f32::from(resized.size.width) - MIN_SELECTION_SIZE).abs() < 0.1);
    }
}
