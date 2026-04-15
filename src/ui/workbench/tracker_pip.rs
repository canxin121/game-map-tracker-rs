use gpui::{
    Bounds, Context, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    ParentElement as _, Pixels, Render, ScrollDelta, ScrollWheelEvent, Styled as _, Window, canvas,
    div,
};
use gpui_component::ActiveTheme as _;

use crate::{
    domain::geometry::{MapCamera, WorldPoint},
    ui::map_canvas::MapViewportState,
};

use super::{
    TrackerWorkbench,
    panels::{paint_bwiki_tile_layers, paint_tracker_map_overlay},
    theme::WorkbenchThemeTokens,
};

const MAP_CLICK_DRAG_THRESHOLD: f32 = 4.0;

pub(super) struct TrackerPipWindow {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    viewport: MapViewportState,
}

impl TrackerPipWindow {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        initial_camera: MapCamera,
        initial_focus: Option<WorldPoint>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Self {
        Self {
            workbench,
            viewport: MapViewportState {
                camera: initial_camera,
                pending_center: initial_focus,
                needs_fit: false,
                ..Default::default()
            },
        }
    }

    fn sync_view_state(&mut self, width: f32, height: f32, follow_point: Option<WorldPoint>) {
        self.viewport.update_viewport(width, height);
        if let Some(point) = follow_point {
            self.viewport.center_on_or_queue(point);
        }
        self.viewport.apply_pending_center();
    }

    fn begin_map_drag(&mut self, screen_x: f32, screen_y: f32) {
        let screen = WorldPoint::new(screen_x, screen_y);
        self.viewport.dragging_from = Some(screen);
        self.viewport.drag_origin = Some(screen);
        self.viewport.drag_moved = false;
        self.viewport.reset_interaction_redraw();
    }

    fn update_map_drag(&mut self, screen_x: f32, screen_y: f32) -> bool {
        let Some(from) = self.viewport.dragging_from.take() else {
            return false;
        };
        let current = WorldPoint::new(screen_x, screen_y);

        if !self.viewport.drag_moved {
            let origin = self.viewport.drag_origin.unwrap_or(from);
            let total_dx = current.x - origin.x;
            let total_dy = current.y - origin.y;
            if total_dx.hypot(total_dy) < MAP_CLICK_DRAG_THRESHOLD {
                self.viewport.dragging_from = Some(current);
                return false;
            }

            self.viewport.drag_moved = true;
            self.viewport.camera.pan_by(total_dx, total_dy);
            self.viewport.dragging_from = Some(current);
            return true;
        }

        let dx = current.x - from.x;
        let dy = current.y - from.y;
        self.viewport.camera.pan_by(dx, dy);
        self.viewport.dragging_from = Some(current);
        true
    }

    fn end_map_drag(&mut self) {
        self.viewport.dragging_from = None;
        self.viewport.drag_origin = None;
        self.viewport.drag_moved = false;
        self.viewport.reset_interaction_redraw();
    }

    fn zoom_map(&mut self, anchor_x: f32, anchor_y: f32, delta: f32) {
        self.viewport.reset_interaction_redraw();
        self.viewport.camera.zoom_at(anchor_x, anchor_y, delta);
    }
}

impl Render for TrackerPipWindow {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let tokens = WorkbenchThemeTokens::from_theme(cx.theme());

        div().size_full().bg(tokens.panel_deep_bg).child(
            canvas(
                move |_, _, _| (),
                move |bounds, _, window, cx| {
                    let bounds_width = f32::from(bounds.size.width);
                    let bounds_height = f32::from(bounds.size.height);
                    let (camera, workbench, bwiki_resources, bwiki_tile_cache) =
                        entity.update(cx, |this, cx| {
                            let mut follow_point = None;
                            let workbench = this.workbench.upgrade();
                            let (bwiki_resources, bwiki_tile_cache) =
                                if let Some(workbench_entity) = workbench.as_ref() {
                                    let workbench_state = workbench_entity.read(cx);
                                    if workbench_state.is_auto_focus_enabled() {
                                        follow_point = workbench_state
                                            .preview_position
                                            .as_ref()
                                            .map(|position| position.world);
                                    }
                                    (
                                        Some(workbench_state.bwiki_resources.clone()),
                                        Some(workbench_state.bwiki_tile_cache.clone()),
                                    )
                                } else {
                                    (None, None)
                                };
                            this.sync_view_state(bounds_width, bounds_height, follow_point);
                            (
                                this.viewport.camera,
                                workbench,
                                bwiki_resources,
                                bwiki_tile_cache,
                            )
                        });

                    if let (Some(workbench), Some(bwiki_resources), Some(bwiki_tile_cache)) =
                        (workbench, bwiki_resources, bwiki_tile_cache)
                    {
                        paint_bwiki_tile_layers(
                            window,
                            bounds,
                            cx,
                            camera,
                            &bwiki_resources,
                            &bwiki_tile_cache,
                            tokens.map_canvas_backdrop,
                        );
                        paint_tracker_map_overlay(
                            &workbench,
                            window,
                            bounds,
                            cx,
                            bounds_width,
                            bounds_height,
                            camera,
                            tokens,
                        );
                    }

                    install_tracker_pip_navigation_handlers(window, entity.clone(), bounds);
                },
            )
            .size_full(),
        )
    }
}

fn install_tracker_pip_navigation_handlers(
    window: &mut Window,
    entity: gpui::Entity<TrackerPipWindow>,
    bounds: Bounds<Pixels>,
) {
    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseDownEvent, _, _, cx| {
            if event.button != MouseButton::Left || !bounds.contains(&event.position) {
                return;
            }

            let _ = entity.update(cx, |this, _| {
                this.begin_map_drag(
                    f32::from(event.position.x) - f32::from(bounds.origin.x),
                    f32::from(event.position.y) - f32::from(bounds.origin.y),
                );
            });
        }
    });

    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseMoveEvent, _, _, cx| {
            let _ = entity.update(cx, |this, cx| {
                if this.update_map_drag(
                    f32::from(event.position.x) - f32::from(bounds.origin.x),
                    f32::from(event.position.y) - f32::from(bounds.origin.y),
                ) {
                    cx.notify();
                }
            });
        }
    });

    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseUpEvent, _, _, cx| {
            if event.button != MouseButton::Left {
                return;
            }

            let _ = entity.update(cx, |this, _| {
                this.end_map_drag();
            });
        }
    });

    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &ScrollWheelEvent, _, _, cx| {
            if !bounds.contains(&event.position) {
                return;
            }

            let delta = match event.delta {
                ScrollDelta::Pixels(delta) => (f32::from(delta.y) / 320.0).clamp(-0.35, 0.35),
                ScrollDelta::Lines(delta) => (delta.y / 8.0).clamp(-0.35, 0.35),
            };
            let anchor_x = f32::from(event.position.x) - f32::from(bounds.origin.x);
            let anchor_y = f32::from(event.position.y) - f32::from(bounds.origin.y);
            let _ = entity.update(cx, |this, cx| {
                this.zoom_map(anchor_x, anchor_y, delta);
                cx.notify();
            });
        }
    });
}

pub(super) fn apply_window_topmost(window: &mut Window, always_on_top: bool) -> anyhow::Result<()> {
    #[cfg(target_os = "windows")]
    {
        use anyhow::{anyhow, ensure};
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            HWND_NOTOPMOST, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos,
        };

        let handle = window
            .window_handle()
            .map_err(|error| anyhow!("无法获取追踪画中画窗口句柄：{error:?}"))?;
        let hwnd = match handle.as_raw() {
            RawWindowHandle::Win32(handle) => handle.hwnd.get() as _,
            other => {
                return Err(anyhow!("当前平台窗口句柄不支持置顶切换：{other:?}"));
            }
        };
        let insert_after = if always_on_top {
            HWND_TOPMOST
        } else {
            HWND_NOTOPMOST
        };
        let flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE;
        let result = unsafe { SetWindowPos(hwnd, insert_after, 0, 0, 0, 0, flags) };
        ensure!(
            result != 0,
            "SetWindowPos 失败：{}",
            std::io::Error::last_os_error()
        );
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = (window, always_on_top);
    }

    Ok(())
}
