use gpui::{
    AnyWindowHandle, Bounds, Context, CursorStyle, InteractiveElement as _, IntoElement,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder,
    Pixels, Point, Render, Styled, Window, canvas, div, fill, hsla, point, px, rgb, size,
};

use crate::config::CaptureRegion;

use super::TrackerWorkbench;

const MIN_SELECTION_SIZE: f32 = 24.0;

pub(super) struct MinimapRegionPicker {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    display_bounds: Bounds<Pixels>,
    drag_origin: Option<Point<Pixels>>,
    drag_current: Option<Point<Pixels>>,
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
            drag_origin: None,
            drag_current: None,
        }
    }

    fn begin_drag(&mut self, event: &MouseDownEvent) {
        self.drag_origin = Some(event.position);
        self.drag_current = Some(event.position);
    }

    fn update_drag(&mut self, event: &MouseMoveEvent) {
        if self.drag_origin.is_some() {
            self.drag_current = Some(event.position);
        }
    }

    fn finish_drag(&mut self, event: &MouseUpEvent, window: &mut Window, cx: &mut Context<Self>) {
        let Some(start) = self.drag_origin else {
            return;
        };
        let end = event.position;
        self.drag_current = Some(end);

        let selection = normalized_bounds(start, end);
        if selection.size.width < px(MIN_SELECTION_SIZE)
            || selection.size.height < px(MIN_SELECTION_SIZE)
        {
            self.drag_origin = None;
            self.drag_current = None;
            cx.notify();
            return;
        }

        let region = self.capture_region_for(selection);
        self.commit_region(region, window, cx);
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.handle_minimap_region_picker_cancelled();
            });
        }
        window.remove_window();
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

    fn selection_bounds(&self) -> Option<Bounds<Pixels>> {
        Some(normalized_bounds(self.drag_origin?, self.drag_current?))
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
}

impl Render for MinimapRegionPicker {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();

        div()
            .size_full()
            .cursor(CursorStyle::Crosshair)
            .child(
                canvas(
                    |_, _, _| {},
                    move |bounds, _, window, cx| {
                        let selection = entity
                            .read(cx)
                            .selection_bounds()
                            .map(|selection| translate_bounds(selection, bounds.origin));

                        if let Some(selection) = selection {
                            paint_outside_selection(window, bounds, selection);
                            paint_selection(window, selection);
                        } else {
                            window.paint_quad(
                                fill(bounds, hsla(0.0, 0.0, 0.0, 0.38)).corner_radii(px(0.0)),
                            );
                        }
                    },
                )
                .size_full(),
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
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                this.update_drag(event);
                cx.notify();
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, window, cx| {
                    this.finish_drag(event, window, cx);
                    cx.notify();
                }),
            )
    }
}

fn normalized_bounds(start: Point<Pixels>, end: Point<Pixels>) -> Bounds<Pixels> {
    let start_x = f32::from(start.x);
    let start_y = f32::from(start.y);
    let end_x = f32::from(end.x);
    let end_y = f32::from(end.y);

    Bounds {
        origin: point(px(start_x.min(end_x)), px(start_y.min(end_y))),
        size: size(
            px((start_x - end_x).abs().max(1.0)),
            px((start_y - end_y).abs().max(1.0)),
        ),
    }
}

fn translate_bounds(bounds: Bounds<Pixels>, origin: Point<Pixels>) -> Bounds<Pixels> {
    Bounds {
        origin: point(origin.x + bounds.origin.x, origin.y + bounds.origin.y),
        size: bounds.size,
    }
}

fn paint_outside_selection(window: &mut Window, bounds: Bounds<Pixels>, selection: Bounds<Pixels>) {
    let overlay = hsla(0.0, 0.0, 0.0, 0.42);
    let top_height = selection.origin.y - bounds.origin.y;
    if top_height > px(0.0) {
        window.paint_quad(fill(
            Bounds {
                origin: bounds.origin,
                size: size(bounds.size.width, top_height),
            },
            overlay,
        ));
    }

    let bottom_origin = selection.origin.y + selection.size.height;
    let bottom_height = bounds.bottom() - bottom_origin;
    if bottom_height > px(0.0) {
        window.paint_quad(fill(
            Bounds {
                origin: point(bounds.origin.x, bottom_origin),
                size: size(bounds.size.width, bottom_height),
            },
            overlay,
        ));
    }

    let left_width = selection.origin.x - bounds.origin.x;
    if left_width > px(0.0) {
        window.paint_quad(fill(
            Bounds {
                origin: point(bounds.origin.x, selection.origin.y),
                size: size(left_width, selection.size.height),
            },
            overlay,
        ));
    }

    let right_origin = selection.origin.x + selection.size.width;
    let right_width = bounds.right() - right_origin;
    if right_width > px(0.0) {
        window.paint_quad(fill(
            Bounds {
                origin: point(right_origin, selection.origin.y),
                size: size(right_width, selection.size.height),
            },
            overlay,
        ));
    }
}

fn paint_selection(window: &mut Window, selection: Bounds<Pixels>) {
    window.paint_quad(fill(selection, hsla(0.58, 0.12, 0.92, 0.06)).corner_radii(px(18.0)));

    let border_bounds = expand_bounds(selection, 1.0);
    if let Some(path) = rectangle_path(border_bounds, 2.0) {
        window.paint_path(path, rgb(0xFFFFFF));
    }

    if let Some(path) = rectangle_path(selection, 1.0) {
        window.paint_path(path, hsla(0.58, 0.72, 0.82, 0.95));
    }
}

fn rectangle_path(bounds: Bounds<Pixels>, stroke_width: f32) -> Option<gpui::Path<Pixels>> {
    let mut builder = PathBuilder::stroke(px(stroke_width));
    builder.move_to(bounds.origin);
    builder.line_to(point(bounds.right(), bounds.origin.y));
    builder.line_to(point(bounds.right(), bounds.bottom()));
    builder.line_to(point(bounds.origin.x, bounds.bottom()));
    builder.line_to(bounds.origin);
    builder.build().ok()
}

fn expand_bounds(bounds: Bounds<Pixels>, amount: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(bounds.origin.x - px(amount), bounds.origin.y - px(amount)),
        size: size(
            bounds.size.width + px(amount * 2.0),
            bounds.size.height + px(amount * 2.0),
        ),
    }
}
