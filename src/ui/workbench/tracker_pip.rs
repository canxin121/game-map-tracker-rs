use gpui::{
    Bounds, ClickEvent, Context, InteractiveElement as _, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, ScrollDelta,
    ScrollWheelEvent, SharedString, StatefulInteractiveElement as _, Styled as _, Window,
    WindowControlArea, canvas, div, px,
};
use gpui_component::{ActiveTheme as _, tooltip::Tooltip};

use crate::{
    domain::geometry::{MapCamera, WorldPoint},
    resources::BwikiResourceManager,
    ui::{map_canvas::MapViewportState, tile_cache::TileImageCache},
};

use super::{
    TrackerMapRenderSnapshot, TrackerWorkbench,
    panels::{paint_bwiki_tile_layers, paint_tracker_map_overlay_snapshot},
    theme::WorkbenchThemeTokens,
};

const MAP_CLICK_DRAG_THRESHOLD: f32 = 4.0;

pub(super) struct TrackerPipWindow {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    viewport: MapViewportState,
    snapshot: TrackerMapRenderSnapshot,
    always_on_top: bool,
    bwiki_resources: BwikiResourceManager,
    bwiki_tile_cache: gpui::Entity<TileImageCache>,
}

impl TrackerPipWindow {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        initial_camera: MapCamera,
        initial_focus: Option<WorldPoint>,
        snapshot: TrackerMapRenderSnapshot,
        bwiki_resources: BwikiResourceManager,
        bwiki_tile_cache: gpui::Entity<TileImageCache>,
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
            always_on_top: snapshot.pip_always_on_top,
            snapshot,
            bwiki_resources,
            bwiki_tile_cache,
        }
    }

    pub(super) fn update_snapshot(&mut self, snapshot: TrackerMapRenderSnapshot) {
        if let Some(point) = snapshot.follow_point {
            self.viewport.center_on_or_queue(point);
        }
        self.always_on_top = snapshot.pip_always_on_top;
        self.snapshot = snapshot;
    }

    fn sync_view_state(&mut self, width: f32, height: f32) {
        self.viewport.update_viewport(width, height);
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

    fn toggle_always_on_top(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let next = !self.always_on_top;
        match apply_window_topmost(window, next) {
            Ok(()) => {
                self.always_on_top = next;
                if let Some(workbench) = self.workbench.upgrade() {
                    let _ = workbench.update(cx, |this, _| {
                        this.set_tracker_pip_always_on_top_from_pip(next);
                    });
                }
                cx.notify();
            }
            Err(error) => {
                if let Some(workbench) = self.workbench.upgrade() {
                    let _ = workbench.update(cx, |this, _| {
                        this.status_text = format!("切换追踪画中画置顶失败：{error:#}").into();
                    });
                }
            }
        }
    }

    fn close_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.handle_tracker_pip_window_closed();
            });
        }
        window.remove_window();
    }
}

impl Render for TrackerPipWindow {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let tokens = WorkbenchThemeTokens::from_theme(cx.theme());

        div()
            .size_full()
            .bg(tokens.panel_deep_bg)
            .flex()
            .flex_col()
            .child(pip_titlebar(self, cx, tokens))
            .child(
                div()
                    .flex_1()
                    .relative()
                    .child(
                        canvas(
                            move |_, _, _| (),
                            move |bounds, _, window, cx| {
                                let (camera, snapshot, bwiki_resources, bwiki_tile_cache) =
                                    entity.update(cx, |this, _| {
                                        this.sync_view_state(
                                            f32::from(bounds.size.width),
                                            f32::from(bounds.size.height),
                                        );
                                        (
                                            this.viewport.camera,
                                            this.snapshot.clone(),
                                            this.bwiki_resources.clone(),
                                            this.bwiki_tile_cache.clone(),
                                        )
                                    });

                                paint_bwiki_tile_layers(
                                    window,
                                    bounds,
                                    cx,
                                    camera,
                                    &bwiki_resources,
                                    &bwiki_tile_cache,
                                    tokens.map_canvas_backdrop,
                                );
                                paint_tracker_map_overlay_snapshot(
                                    window,
                                    bounds,
                                    cx,
                                    camera,
                                    tokens,
                                    &snapshot,
                                    &bwiki_resources,
                                );
                                install_tracker_pip_navigation_handlers(
                                    window,
                                    entity.clone(),
                                    bounds,
                                );
                            },
                        )
                        .size_full(),
                    ),
            )
    }
}

fn pip_titlebar(
    this: &TrackerPipWindow,
    cx: &mut Context<TrackerPipWindow>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .h(px(44.0))
        .flex()
        .items_center()
        .justify_between()
        .gap_3()
        .bg(tokens.panel_bg.opacity(0.97))
        .border_b_1()
        .border_color(tokens.border_strong)
        .px_4()
        .shadow_xs()
        .child(
            div()
                .flex_1()
                .h_full()
                .flex()
                .items_center()
                .window_control_area(WindowControlArea::Drag)
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .child("追踪画中画"),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .on_mouse_down(MouseButton::Left, |_, _, cx| {
                    cx.stop_propagation();
                })
                .on_mouse_up(MouseButton::Left, |_, _, cx| {
                    cx.stop_propagation();
                })
                .child(pip_control_button(
                    "tracker-pip-topmost-local",
                    tokens,
                    "置顶",
                    if this.always_on_top {
                        "当前已置顶"
                    } else {
                        "当前未置顶"
                    },
                    PipControlTone::Toggle(this.always_on_top),
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.toggle_always_on_top(window, cx);
                    }),
                ))
                .child(pip_control_button(
                    "tracker-pip-close-local",
                    tokens,
                    "关闭",
                    "关闭追踪画中画",
                    PipControlTone::Danger,
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.close_window(window, cx);
                    }),
                )),
        )
}

#[derive(Debug, Clone, Copy)]
enum PipControlTone {
    Toggle(bool),
    Danger,
}

fn pip_control_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    tooltip: impl Into<SharedString>,
    tone: PipControlTone,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        PipControlTone::Toggle(true) => (
            tokens.toolbar_button_primary_bg,
            tokens.toolbar_button_primary_hover_bg,
            tokens.border_strong,
        ),
        PipControlTone::Toggle(false) => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
        PipControlTone::Danger => (
            tokens.toolbar_button_danger_bg,
            tokens.toolbar_button_danger_hover_bg,
            tokens.border_strong,
        ),
    };

    div()
        .id(id.into())
        .h(px(32.0))
        .px_3()
        .min_w(px(56.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .cursor_pointer()
        .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
        .hover(move |style| style.bg(hover_background))
        .active(|style| style.opacity(0.92))
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .on_mouse_up(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .on_click(move |event, window, cx| {
            cx.stop_propagation();
            on_click(event, window, cx);
        })
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .child(label),
        )
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
