use gpui::{
    AppContext, Bounds, ClickEvent, Context, InteractiveElement as _, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, ScrollDelta,
    ScrollWheelEvent, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription,
    Window, WindowControlArea, canvas, div, fill, point, prelude::FluentBuilder as _, px, size,
};
use gpui_component::{
    ActiveTheme as _, Icon, Size, Sizable as _, select::SelectItem, tooltip::Tooltip,
};

use crate::{
    config::CaptureRegion,
    domain::geometry::{MapCamera, WorldPoint},
    resources::BwikiResourceManager,
    tracking::{
        capture::DesktopCapture,
        debug::{DebugImage, DebugImageFormat, DebugImageKind},
    },
    ui::{map_canvas::MapViewportState, tile_cache::TileImageCache},
};

use super::{
    TestCaseLabel, TrackerMapRenderSnapshot, TrackerWorkbench,
    forms::{PipCapturePickerItem, PipCapturePickerTarget},
    panels::{paint_bwiki_tile_layers, paint_tracker_map_overlay_snapshot},
    select::{ActionMenu, ActionMenuEvent, ActionMenuState},
    theme::WorkbenchThemeTokens,
};

const MAP_CLICK_DRAG_THRESHOLD: f32 = 4.0;

pub(super) struct TrackerPipWindow {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    viewport: MapViewportState,
    snapshot: TrackerMapRenderSnapshot,
    always_on_top: bool,
    minimap_region: CaptureRegion,
    last_window_bounds: Option<Bounds<Pixels>>,
    bwiki_resources: BwikiResourceManager,
    bwiki_tile_cache: gpui::Entity<TileImageCache>,
    capture_menu: gpui::Entity<ActionMenuState<PipCapturePickerItem>>,
    _subscriptions: Vec<Subscription>,
}

impl TrackerPipWindow {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        initial_camera: MapCamera,
        initial_focus: Option<WorldPoint>,
        snapshot: TrackerMapRenderSnapshot,
        minimap_region: CaptureRegion,
        bwiki_resources: BwikiResourceManager,
        bwiki_tile_cache: gpui::Entity<TileImageCache>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let capture_menu =
            cx.new(|cx| ActionMenuState::new(Self::capture_picker_items(), 6, window, cx));
        let mut subscriptions = Vec::new();
        let capture_menu_subscription = capture_menu.clone();
        subscriptions.push(cx.subscribe_in(
            &capture_menu_subscription,
            window,
            |this, _, event: &ActionMenuEvent<PipCapturePickerItem>, _, cx| {
                let ActionMenuEvent::Confirm(target) = event;
                this.open_capture_picker(*target, cx);
                cx.notify();
            },
        ));

        Self {
            workbench,
            viewport: MapViewportState {
                camera: initial_camera,
                pending_center: initial_focus,
                needs_fit: false,
                ..Default::default()
            },
            always_on_top: snapshot.pip_always_on_top,
            minimap_region,
            last_window_bounds: None,
            snapshot,
            bwiki_resources,
            bwiki_tile_cache,
            capture_menu,
            _subscriptions: subscriptions,
        }
    }

    pub(super) fn update_snapshot(
        &mut self,
        snapshot: TrackerMapRenderSnapshot,
        minimap_region: CaptureRegion,
        _cx: &mut Context<Self>,
    ) {
        if let Some(point) = snapshot.follow_point {
            self.viewport.center_on_or_queue(point);
        }
        self.always_on_top = snapshot.pip_always_on_top;
        self.minimap_region = minimap_region;
        self.snapshot = snapshot;
    }

    fn sync_view_state(&mut self, width: f32, height: f32) {
        self.viewport.update_viewport(width, height);
        self.viewport.apply_pending_center();
    }

    fn sync_capture_panel_position(&mut self, bounds: Bounds<Pixels>, cx: &mut Context<Self>) {
        if self.last_window_bounds == Some(bounds) {
            return;
        }
        self.last_window_bounds = Some(bounds);
        let workbench = self.workbench.clone();
        cx.defer(move |cx| {
            if let Some(workbench) = workbench.upgrade() {
                let _ = workbench.update(cx, |this, cx| {
                    this.sync_tracker_pip_capture_panel_with_bounds(bounds, cx);
                });
            }
        });
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
            let _ = workbench.update(cx, |this, cx| {
                this.handle_tracker_pip_window_closed(cx);
            });
        }
        window.remove_window();
    }

    fn open_capture_picker(&mut self, target: PipCapturePickerTarget, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, cx| match target {
                PipCapturePickerTarget::Minimap => this.toggle_minimap_region_picker_from_pip(cx),
            });
        }
    }

    fn toggle_capture_panel(&mut self, window: &Window, cx: &mut Context<Self>) {
        let bounds = window.window_bounds().get_bounds();
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, cx| {
                this.toggle_tracker_pip_capture_panel_from_pip(bounds, cx);
            });
        }
    }

    fn capture_picker_items() -> Vec<PipCapturePickerItem> {
        vec![PipCapturePickerItem::new(
            PipCapturePickerTarget::Minimap,
            "小地图",
            "圆形外接截图区域",
            "minimap capture picker 小地图 截图 取区",
        )]
    }
}

impl Render for TrackerPipWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let tokens = WorkbenchThemeTokens::from_theme(cx.theme());
        self.sync_capture_panel_position(window.window_bounds().get_bounds(), cx);

        div()
            .size_full()
            .bg(tokens.panel_deep_bg)
            .flex()
            .flex_col()
            .child(pip_titlebar(self, cx, tokens))
            .child(
                div().flex_1().relative().child(
                    canvas(
                        move |_, _, _| (),
                        move |bounds, _, window, cx| {
                            let (camera, snapshot, bwiki_resources, bwiki_tile_cache) = entity
                                .update(cx, |this, _| {
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
                            install_tracker_pip_navigation_handlers(window, entity.clone(), bounds);
                        },
                    )
                    .size_full(),
                ),
            )
            .child(pip_bottom_panel(self, cx, tokens))
    }
}

pub(super) struct TrackerPipCapturePanelWindow {
    workbench: gpui::WeakEntity<TrackerWorkbench>,
    minimap_region: CaptureRegion,
    capture_label_menu: gpui::Entity<ActionMenuState<PipTestCaseLabelItem>>,
    capture_label: TestCaseLabel,
    capture_preview: Option<DebugImage>,
    capture_preview_error: Option<SharedString>,
    _subscriptions: Vec<Subscription>,
}

impl TrackerPipCapturePanelWindow {
    pub(super) fn new(
        workbench: gpui::WeakEntity<TrackerWorkbench>,
        minimap_region: CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let capture_label_menu =
            cx.new(|cx| ActionMenuState::new(Self::capture_label_items(), 4, window, cx));
        let mut subscriptions = Vec::new();
        let capture_label_menu_subscription = capture_label_menu.clone();
        subscriptions.push(cx.subscribe_in(
            &capture_label_menu_subscription,
            window,
            |this, _, event: &ActionMenuEvent<PipTestCaseLabelItem>, _, cx| {
                let ActionMenuEvent::Confirm(label) = event;
                this.capture_label = *label;
                cx.notify();
            },
        ));

        let mut this = Self {
            workbench,
            minimap_region,
            capture_label_menu,
            capture_label: TestCaseLabel::HasMap,
            capture_preview: None,
            capture_preview_error: None,
            _subscriptions: subscriptions,
        };
        this.refresh_capture_preview();
        this
    }

    pub(super) fn update_minimap_region(
        &mut self,
        minimap_region: CaptureRegion,
        _cx: &mut Context<Self>,
    ) {
        self.minimap_region = minimap_region;
        self.refresh_capture_preview();
    }

    fn capture_test_case(&mut self, label: TestCaseLabel, cx: &mut Context<Self>) {
        if let Some(workbench) = self.workbench.upgrade() {
            let _ = workbench.update(cx, |this, _| {
                this.capture_test_case(label);
            });
        }
        self.capture_label = label;
        self.refresh_capture_preview();
        cx.notify();
    }

    fn capture_label_items() -> Vec<PipTestCaseLabelItem> {
        vec![
            PipTestCaseLabelItem::new(TestCaseLabel::HasMap, "有图", "has_map 有地图 正样本"),
            PipTestCaseLabelItem::new(TestCaseLabel::NoMap, "无图", "no_map 无地图 负样本"),
        ]
    }

    fn refresh_capture_preview(&mut self) {
        match DesktopCapture::from_absolute_region(&self.minimap_region)
            .and_then(|capture| capture.capture_rgba())
        {
            Ok(image) => {
                let (width, height) = image.dimensions();
                self.capture_preview = Some(DebugImage::rgba(
                    "Minimap Capture",
                    width,
                    height,
                    DebugImageKind::Snapshot,
                    image.into_raw(),
                ));
                self.capture_preview_error = None;
            }
            Err(error) => {
                self.capture_preview = None;
                self.capture_preview_error = Some(format!("刷新捕获图失败：{error:#}").into());
            }
        }
    }
}

impl Render for TrackerPipCapturePanelWindow {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let tokens = WorkbenchThemeTokens::from_theme(cx.theme());

        div()
            .size_full()
            .bg(tokens.panel_bg.opacity(0.99))
            .border_1()
            .border_color(tokens.border_strong)
            .p_3()
            .child(
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(tokens.app_fg)
                                    .child("捕获调试图"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(tokens.text_muted)
                                    .child("仅捕获当前 minimap"),
                            ),
                    )
                    .child(pip_capture_preview(
                        self.capture_preview.clone(),
                        self.capture_preview_error.clone(),
                        tokens,
                    ))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(pip_capture_label_menu(self, tokens))
                            .child(pip_control_text_button(
                                "tracker-pip-capture-run",
                                tokens,
                                "捕获",
                                "按当前标签保存 minimap 测试样本",
                                PipControlTone::Neutral,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    let label = this.capture_label;
                                    this.capture_test_case(label, cx);
                                }),
                            )),
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
                .child(pip_capture_menu(this, tokens))
                .child(pip_control_icon_button(
                    "tracker-pip-topmost-local",
                    tokens,
                    Icon::default().path("assets/icons/arrow-up.svg"),
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
                .child(pip_control_icon_button(
                    "tracker-pip-close-local",
                    tokens,
                    Icon::default().path("assets/icons/close.svg"),
                    "关闭追踪画中画",
                    PipControlTone::Danger,
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.close_window(window, cx);
                    }),
                )),
        )
}

fn pip_capture_menu(this: &TrackerPipWindow, tokens: WorkbenchThemeTokens) -> impl IntoElement {
    ActionMenu::new(&this.capture_menu)
        .icon(Icon::default().path("assets/icons/arrow-down.svg"))
        .center_label()
        .with_size(Size::Small)
        .h(px(32.0))
        .min_w(px(56.0))
        .px_2()
        .rounded_lg()
        .bg(tokens.toolbar_button_bg)
        .border_1()
        .border_color(tokens.border)
        .menu_width(px(280.0))
        .label("取区")
        .search_placeholder("搜索 小地图")
        .empty_message("当前没有可用的取区目标。")
}

fn pip_bottom_panel(
    this: &TrackerPipWindow,
    cx: &mut Context<TrackerPipWindow>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .h(px(34.0))
        .flex()
        .items_center()
        .gap_3()
        .bg(tokens.panel_bg.opacity(0.98))
        .border_t_1()
        .border_color(tokens.border_strong)
        .px_3()
        .when(this.snapshot.pip_test_case_capture_enabled, |row| {
            row.child(pip_control_icon_button(
                "tracker-pip-capture-panel-toggle",
                tokens,
                if this.snapshot.pip_capture_panel_expanded {
                    Icon::default().path("assets/icons/arrow-right.svg")
                } else {
                    Icon::default().path("assets/icons/arrow-left.svg")
                },
                if this.snapshot.pip_capture_panel_expanded {
                    "收起画中画外置捕获面板"
                } else {
                    "展开画中画外置捕获面板"
                },
                PipControlTone::Toggle(this.snapshot.pip_capture_panel_expanded),
                cx.listener(|this, _: &ClickEvent, window, cx| {
                    this.toggle_capture_panel(window, cx);
                }),
            ))
            .child(div().w(px(1.0)).h(px(14.0)).bg(tokens.border))
        })
        .child(pip_debug_footer_item(
            "地图",
            this.snapshot.pip_probe_summary.clone(),
            tokens,
        ))
        .child(div().w(px(1.0)).h(px(14.0)).bg(tokens.border))
        .child(pip_debug_footer_item(
            "定位",
            this.snapshot.pip_locate_summary.clone(),
            tokens,
        ))
}

fn pip_debug_footer_item(
    label: &'static str,
    value: SharedString,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .flex_1()
        .min_w(px(0.0))
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .text_xs()
                .font_weight(gpui::FontWeight::MEDIUM)
                .text_color(tokens.text_muted)
                .child(label),
        )
        .child(
            div()
                .flex_1()
                .min_w(px(0.0))
                .text_xs()
                .text_color(tokens.text_soft)
                .overflow_hidden()
                .whitespace_nowrap()
                .truncate()
                .child(value),
        )
}

#[derive(Debug, Clone, Copy)]
enum PipControlTone {
    Neutral,
    Toggle(bool),
    Danger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipTestCaseLabelItem {
    value: TestCaseLabel,
    title: &'static str,
    searchable_text: &'static str,
}

impl PipTestCaseLabelItem {
    const fn new(value: TestCaseLabel, title: &'static str, searchable_text: &'static str) -> Self {
        Self {
            value,
            title,
            searchable_text,
        }
    }
}

impl SelectItem for PipTestCaseLabelItem {
    type Value = TestCaseLabel;

    fn title(&self) -> SharedString {
        self.title.into()
    }

    fn value(&self) -> &Self::Value {
        &self.value
    }

    fn matches(&self, query: &str) -> bool {
        self.searchable_text
            .to_ascii_lowercase()
            .contains(&query.to_ascii_lowercase())
    }
}

fn pip_control_icon_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: Icon,
    tooltip: impl Into<SharedString>,
    tone: PipControlTone,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        PipControlTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
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
        .w(px(32.0))
        .h(px(32.0))
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
        .child(icon.small().text_color(tokens.app_fg))
}

fn pip_control_text_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    tooltip: impl Into<SharedString>,
    tone: PipControlTone,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        PipControlTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
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
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .child(label),
        )
}

fn pip_capture_label_menu(
    this: &TrackerPipCapturePanelWindow,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    ActionMenu::new(&this.capture_label_menu)
        .icon(Icon::default().path("assets/icons/arrow-down.svg"))
        .center_label()
        .with_size(Size::Small)
        .h(px(32.0))
        .min_w(px(92.0))
        .px_2()
        .rounded_lg()
        .bg(tokens.toolbar_button_bg)
        .border_1()
        .border_color(tokens.border)
        .menu_width(px(200.0))
        .label(this.capture_label.display_name())
        .search_placeholder("搜索 有图 / 无图")
        .empty_message("当前没有可用标签。")
}

fn pip_capture_preview(
    image: Option<DebugImage>,
    error: Option<SharedString>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let placeholder = error.unwrap_or_else(|| "打开面板后会自动刷新当前 minimap 捕获图。".into());
    div()
        .flex_1()
        .min_h(px(0.0))
        .rounded_lg()
        .bg(tokens.debug_canvas_bg)
        .border_1()
        .border_color(tokens.border_strong)
        .overflow_hidden()
        .child(if let Some(image) = image {
            pip_debug_image_canvas(image, tokens).into_any_element()
        } else {
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .px_3()
                .text_xs()
                .text_color(tokens.text_muted)
                .child(placeholder)
                .into_any_element()
        })
}

fn pip_debug_image_canvas(image: DebugImage, tokens: WorkbenchThemeTokens) -> impl IntoElement {
    canvas(
        move |_, _, _| image.clone(),
        move |bounds, image, window, _| {
            window.paint_quad(fill(bounds, tokens.debug_canvas_bg));
            if image.width == 0 || image.height == 0 || image.pixels.is_empty() {
                return;
            }

            let scale_x = f32::from(bounds.size.width) / image.width as f32;
            let scale_y = f32::from(bounds.size.height) / image.height as f32;
            match image.format {
                DebugImageFormat::Gray8 => {
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let intensity = image.pixels[(y * image.width + x) as usize] as u32;
                            let rgb = (intensity << 16) | (intensity << 8) | intensity;
                            let pixel_bounds = Bounds {
                                origin: point(
                                    bounds.origin.x + px(x as f32 * scale_x),
                                    bounds.origin.y + px(y as f32 * scale_y),
                                ),
                                size: size(px(scale_x.max(1.0)), px(scale_y.max(1.0))),
                            };
                            window.paint_quad(fill(pixel_bounds, gpui::rgb(rgb)));
                        }
                    }
                }
                DebugImageFormat::Rgba8 => {
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let index = ((y * image.width + x) * 4) as usize;
                            if index + 3 >= image.pixels.len() {
                                continue;
                            }
                            let r = image.pixels[index] as u32;
                            let g = image.pixels[index + 1] as u32;
                            let b = image.pixels[index + 2] as u32;
                            let rgb = (r << 16) | (g << 8) | b;
                            let pixel_bounds = Bounds {
                                origin: point(
                                    bounds.origin.x + px(x as f32 * scale_x),
                                    bounds.origin.y + px(y as f32 * scale_y),
                                ),
                                size: size(px(scale_x.max(1.0)), px(scale_y.max(1.0))),
                            };
                            window.paint_quad(fill(pixel_bounds, gpui::rgb(rgb)));
                        }
                    }
                }
            }
        },
    )
    .size_full()
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

pub(super) fn apply_window_bounds(
    window: &mut Window,
    bounds: Bounds<Pixels>,
) -> anyhow::Result<()> {
    #[cfg(target_os = "windows")]
    {
        use anyhow::{anyhow, ensure};
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        use windows_sys::Win32::UI::WindowsAndMessaging::{SWP_NOACTIVATE, SetWindowPos};

        let handle = window
            .window_handle()
            .map_err(|error| anyhow!("无法获取窗口句柄：{error:?}"))?;
        let hwnd = match handle.as_raw() {
            RawWindowHandle::Win32(handle) => handle.hwnd.get() as _,
            other => {
                return Err(anyhow!("当前平台窗口句柄不支持移动窗口：{other:?}"));
            }
        };
        let result = unsafe {
            SetWindowPos(
                hwnd,
                std::ptr::null_mut(),
                f32::from(bounds.origin.x).round() as i32,
                f32::from(bounds.origin.y).round() as i32,
                f32::from(bounds.size.width).round() as i32,
                f32::from(bounds.size.height).round() as i32,
                SWP_NOACTIVATE,
            )
        };
        ensure!(
            result != 0,
            "SetWindowPos 失败：{}",
            std::io::Error::last_os_error()
        );
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = (window, bounds);
    }

    Ok(())
}
