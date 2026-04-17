use std::collections::{BTreeMap, HashMap, HashSet};

use gpui::{
    AnyElement, Bounds, ClickEvent, ClipboardItem, ContentMask, Context, ImgResourceLoader,
    InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, PathBuilder, ScrollDelta, ScrollWheelEvent, SharedString,
    StatefulInteractiveElement as _, Styled as _, canvas, div, fill, point,
    prelude::FluentBuilder as _, px, size,
};
use gpui_component::{
    ActiveTheme as _, Icon, IconName, Selectable, Sizable as _,
    button::{Button, ButtonGroup},
    input::Input,
    scroll::ScrollableElement as _,
    tooltip::Tooltip,
};
use strum::IntoEnumIterator;

use crate::{
    domain::{theme::ThemePreference, tracker::TrackingSource},
    resources::{
        BwikiResourceManager, tile_coordinate_to_world_origin, visible_tile_layers,
        zoom_world_bounds,
    },
    ui::map_canvas::{
        BWIKI_MARKER_ICON_HEIGHT, BWIKI_MARKER_ICON_WIDTH, bounds_corner_radius,
        bwiki_marker_image_bounds, inflate_bounds, parse_hex_color, screen_points,
    },
};

use super::{
    MapCanvasKind, TrackerCacheKind, TrackerMapRenderSnapshot, TrackerWorkbench,
    forms::read_input_value,
    page::{MapPage, SettingsPage, WorkbenchPage},
    select::Select,
    theme::WorkbenchThemeTokens,
};

pub(super) fn render_workbench(
    this: &mut TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
) -> impl IntoElement {
    this.bwiki_resources.ensure_dataset_loaded();
    this.sync_bwiki_visibility_defaults();
    let tokens = WorkbenchThemeTokens::from_theme(cx.theme());
    let page_content = match this.active_page {
        WorkbenchPage::Map => map_page(this, cx, tokens).into_any_element(),
        WorkbenchPage::Markers => markers_page(this, cx, tokens).into_any_element(),
        WorkbenchPage::Settings => settings_page(this, cx, tokens).into_any_element(),
    };

    div()
        .size_full()
        .bg(tokens.app_bg)
        .text_color(tokens.app_fg)
        .p_5()
        .child(
            div()
                .flex()
                .gap_4()
                .size_full()
                .overflow_hidden()
                .child(navigation_sidebar(this, cx, tokens))
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap_4()
                        .min_h(px(0.0))
                        .overflow_hidden()
                        .child(page_header(this, cx, tokens))
                        .child(page_content),
                ),
        )
}

fn navigation_sidebar(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let tracker_is_active =
        this.active_page == WorkbenchPage::Map && this.map_page == MapPage::Tracker;
    let bwiki_is_active = this.active_page == WorkbenchPage::Map && this.map_page == MapPage::Bwiki;
    let markers_is_active = this.active_page == WorkbenchPage::Markers;
    let settings_is_active = this.active_page == WorkbenchPage::Settings;
    let settings_is_open = this.settings_nav_expanded;

    div()
        .w(px(252.0))
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .gap_4()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(section_title("导航栏"))
        .child(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(sidebar_nav_item(
                    "sidebar-nav-bwiki",
                    tokens,
                    "节点图鉴",
                    bwiki_is_active,
                    tokens.nav_item_active_bg,
                    None,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.select_map_page(MapPage::Bwiki);
                        cx.notify();
                    }),
                ))
                .child(sidebar_nav_item(
                    "sidebar-nav-tracker",
                    tokens,
                    "路线追踪",
                    tracker_is_active,
                    tokens.nav_item_active_bg,
                    None,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.select_map_page(MapPage::Tracker);
                        cx.notify();
                    }),
                ))
                .child(sidebar_nav_item(
                    "sidebar-nav-markers",
                    tokens,
                    "路线管理",
                    markers_is_active,
                    tokens.nav_item_active_bg,
                    None,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.select_routes_page();
                        cx.notify();
                    }),
                ))
                .child(sidebar_nav_item(
                    "sidebar-nav-settings",
                    tokens,
                    "设置",
                    settings_is_active,
                    tokens.nav_item_active_bg,
                    Some(if settings_is_open { "v" } else { ">" }),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_settings_navigation();
                        cx.notify();
                    }),
                ))
                .when(settings_is_open, |column| {
                    column.child(
                        div()
                            .ml_4()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .rounded_lg()
                            .bg(tokens.nav_branch_bg)
                            .border_1()
                            .border_color(tokens.border)
                            .p_2()
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-runtime",
                                tokens,
                                "界面与运行",
                                settings_is_active && this.settings_page == SettingsPage::Runtime,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Runtime);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-capture",
                                tokens,
                                "截图与局部搜索",
                                settings_is_active && this.settings_page == SettingsPage::Capture,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Capture);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-convolution",
                                tokens,
                                "卷积特征匹配",
                                settings_is_active
                                    && this.settings_page == SettingsPage::Convolution,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Convolution);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-template",
                                tokens,
                                "多尺度模板匹配",
                                settings_is_active && this.settings_page == SettingsPage::Template,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Template);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-debug",
                                tokens,
                                "调试",
                                settings_is_active && this.settings_page == SettingsPage::Debug,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Debug);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-settings-resources",
                                tokens,
                                "资源",
                                settings_is_active && this.settings_page == SettingsPage::Resources,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Resources);
                                    cx.notify();
                                }),
                            )),
                    )
                }),
        )
}

fn sidebar_nav_item(
    id: &'static str,
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    active: bool,
    active_bg: gpui::Hsla,
    trailing: Option<&'static str>,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let text_color = if active {
        tokens.app_fg
    } else {
        tokens.text_soft
    };
    let border_color = if active {
        tokens.border_strong
    } else {
        tokens.border
    };
    let indicator_color = if active {
        tokens.border_strong
    } else {
        tokens.border
    };

    div()
        .id(id)
        .w_full()
        .min_h(px(42.0))
        .flex()
        .items_center()
        .justify_between()
        .gap_3()
        .px_3()
        .py_2()
        .rounded_lg()
        .bg(if active {
            active_bg
        } else {
            tokens.nav_item_bg
        })
        .border_1()
        .border_color(border_color)
        .cursor_pointer()
        .hover(|style| style.bg(tokens.nav_item_hover_bg))
        .active(|style| style.opacity(0.92))
        .on_click(on_click)
        .child(
            div()
                .flex_1()
                .flex()
                .items_center()
                .gap_3()
                .child(
                    div()
                        .w(px(4.0))
                        .h(px(if active { 18.0 } else { 8.0 }))
                        .rounded_full()
                        .bg(indicator_color),
                )
                .child(
                    div()
                        .flex_1()
                        .text_sm()
                        .font_weight(if active {
                            gpui::FontWeight::SEMIBOLD
                        } else {
                            gpui::FontWeight::NORMAL
                        })
                        .text_color(text_color)
                        .child(label),
                ),
        )
        .when_some(trailing, |row, trailing| {
            row.child(
                div()
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(if active {
                        tokens.app_fg
                    } else {
                        tokens.text_muted
                    })
                    .child(trailing),
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct PaginationMetrics {
    page: usize,
    page_count: usize,
}

fn normalized_query(
    input: &gpui::Entity<gpui_component::input::InputState>,
    cx: &mut Context<TrackerWorkbench>,
) -> String {
    read_input_value(input, cx).trim().to_lowercase()
}

fn matches_query<I, S>(query: &str, values: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    query.is_empty()
        || values
            .into_iter()
            .any(|value| value.as_ref().to_lowercase().contains(query))
}

fn paginate_items<T>(items: Vec<T>, page: usize, page_size: usize) -> (Vec<T>, PaginationMetrics) {
    let filtered_items = items.len();
    let page_size = page_size.max(1);
    let page_count = filtered_items.max(1).div_ceil(page_size);
    let page = page.min(page_count.saturating_sub(1));
    let start = page.saturating_mul(page_size);
    let visible_items = items
        .into_iter()
        .skip(start)
        .take(page_size)
        .collect::<Vec<_>>();

    (visible_items, PaginationMetrics { page, page_count })
}

fn paginated_list(
    id_prefix: &'static str,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
    title: Option<SharedString>,
    header_content: Vec<AnyElement>,
    search_input: &gpui::Entity<gpui_component::input::InputState>,
    search_actions: Vec<AnyElement>,
    page_input: &gpui::Entity<gpui_component::input::InputState>,
    metrics: PaginationMetrics,
    empty_text: &'static str,
    rows: Vec<AnyElement>,
    set_page: fn(&mut TrackerWorkbench, usize, &mut gpui::Window, &mut Context<TrackerWorkbench>),
) -> impl IntoElement {
    let first_page = 0usize;
    let last_page = metrics.page_count.saturating_sub(1);
    let has_results = !rows.is_empty();
    let has_header_content = !header_content.is_empty();
    let has_search_actions = !search_actions.is_empty();

    let mut controls = Vec::new();
    controls.push(
        pager_icon_button(
            format!("{id_prefix}-first"),
            tokens,
            "<<",
            metrics.page == first_page,
            cx.listener(move |this, _: &ClickEvent, window, cx| {
                set_page(this, first_page, window, cx);
                cx.notify();
            }),
        )
        .into_any_element(),
    );
    controls.push(
        pager_icon_button(
            format!("{id_prefix}-prev"),
            tokens,
            "<",
            metrics.page == first_page,
            cx.listener(move |this, _: &ClickEvent, window, cx| {
                set_page(this, metrics.page.saturating_sub(1), window, cx);
                cx.notify();
            }),
        )
        .into_any_element(),
    );
    controls.push(
        pager_jump_control(
            format!("{id_prefix}-jump"),
            tokens,
            page_input,
            metrics.page_count,
        )
        .into_any_element(),
    );
    controls.push(
        pager_icon_button(
            format!("{id_prefix}-next"),
            tokens,
            ">",
            metrics.page >= last_page,
            cx.listener(move |this, _: &ClickEvent, window, cx| {
                set_page(this, (metrics.page + 1).min(last_page), window, cx);
                cx.notify();
            }),
        )
        .into_any_element(),
    );
    controls.push(
        pager_icon_button(
            format!("{id_prefix}-last"),
            tokens,
            ">>",
            metrics.page >= last_page,
            cx.listener(move |this, _: &ClickEvent, window, cx| {
                set_page(this, last_page, window, cx);
                cx.notify();
            }),
        )
        .into_any_element(),
    );

    div()
        .w_full()
        .rounded_lg()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap_3()
                .when_some(title, |panel, title| panel.child(section_title(title)))
                .when(has_header_content, |panel| panel.children(header_content))
                .child(
                    div()
                        .w_full()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(div().flex_1().min_w_0().child(Input::new(search_input)))
                        .when(has_search_actions, |row| row.children(search_actions)),
                )
                .child(div().flex().flex_col().gap_2().children(if has_results {
                    rows
                } else {
                    vec![empty_list_state(tokens, empty_text).into_any_element()]
                }))
                .child(
                    div().w_full().flex().items_center().justify_center().child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .flex_wrap()
                            .children(controls),
                    ),
                ),
        )
}

fn selectable_list_row(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    title: impl Into<SharedString>,
    subtitle: impl Into<SharedString>,
    active: bool,
    trailing: Option<SharedString>,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let title = title.into();
    let subtitle = subtitle.into();

    div()
        .id(id.into())
        .w_full()
        .min_w_0()
        .flex()
        .items_center()
        .justify_between()
        .gap_3()
        .px_3()
        .py_2()
        .rounded_lg()
        .bg(if active {
            tokens.nav_subitem_active_bg
        } else {
            tokens.nav_item_bg
        })
        .border_1()
        .border_color(if active {
            tokens.border_strong
        } else {
            tokens.border
        })
        .cursor_pointer()
        .hover(|style| style.bg(tokens.nav_item_hover_bg))
        .active(|style| style.opacity(0.92))
        .on_click(on_click)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap_3()
                .child(
                    div()
                        .w(px(5.0))
                        .h(px(if active { 20.0 } else { 10.0 }))
                        .flex_shrink_0()
                        .rounded_full()
                        .bg(if active {
                            tokens.border_strong
                        } else {
                            tokens.border
                        }),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .overflow_hidden()
                        .child(
                            div()
                                .w_full()
                                .text_sm()
                                .font_weight(if active {
                                    gpui::FontWeight::SEMIBOLD
                                } else {
                                    gpui::FontWeight::NORMAL
                                })
                                .text_color(tokens.app_fg)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(title),
                        )
                        .child(
                            div()
                                .w_full()
                                .text_xs()
                                .text_color(tokens.text_muted)
                                .line_height(px(18.0))
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(subtitle),
                        ),
                ),
        )
        .when_some(trailing, |row, trailing| {
            row.child(
                div()
                    .flex_shrink_0()
                    .rounded_full()
                    .bg(tokens.panel_alt_bg)
                    .border_1()
                    .border_color(tokens.border)
                    .px_2()
                    .py_1()
                    .child(div().text_xs().text_color(tokens.text_soft).child(trailing)),
            )
        })
}

fn row_action_svg_icon_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon_path: &'static str,
    tooltip: impl Into<SharedString>,
    tone: ToolbarButtonTone,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        ToolbarButtonTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
        ToolbarButtonTone::Primary => (
            tokens.toolbar_button_primary_bg,
            tokens.toolbar_button_primary_hover_bg,
            tokens.border_strong,
        ),
        ToolbarButtonTone::Danger => (
            tokens.toolbar_button_danger_bg,
            tokens.toolbar_button_danger_hover_bg,
            tokens.border_strong,
        ),
    };

    div()
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .child(
            div()
                .id(id.into())
                .w(px(28.0))
                .h(px(28.0))
                .flex()
                .items_center()
                .justify_center()
                .rounded_md()
                .bg(background)
                .border_1()
                .border_color(border_color)
                .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
                .cursor_pointer()
                .hover(move |style| style.bg(hover_background))
                .active(|style| style.opacity(0.92))
                .on_click(move |event, window, cx| {
                    cx.stop_propagation();
                    on_click(event, window, cx);
                })
                .child(
                    Icon::default()
                        .path(icon_path)
                        .small()
                        .text_color(tokens.app_fg),
                ),
        )
}

fn marker_group_list_row(
    id: impl Into<SharedString>,
    hover_group: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    title: impl Into<SharedString>,
    subtitle: impl Into<SharedString>,
    active: bool,
    point_count: usize,
    confirming_delete: bool,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
    on_edit: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
    on_request_delete: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
    on_confirm_delete: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
    on_cancel_delete: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let title = title.into();
    let subtitle = subtitle.into();
    let row_id = id.into();
    let hover_group = hover_group.into();
    let overlay = if confirming_delete {
        div()
            .absolute()
            .top_2()
            .right_2()
            .flex()
            .items_center()
            .gap_1()
            .px_2()
            .py_1()
            .rounded_lg()
            .bg(tokens.panel_bg)
            .border_1()
            .border_color(tokens.border_strong)
            .shadow_xs()
            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(tokens.app_fg)
                    .child("确认删除"),
            )
            .child(row_action_svg_icon_button(
                format!("{row_id}-confirm-delete"),
                tokens,
                "assets/icons/check.svg",
                "确认删除路线",
                ToolbarButtonTone::Danger,
                on_confirm_delete,
            ))
            .child(row_action_svg_icon_button(
                format!("{row_id}-cancel-delete"),
                tokens,
                "assets/icons/close.svg",
                "取消删除",
                ToolbarButtonTone::Neutral,
                on_cancel_delete,
            ))
            .into_any_element()
    } else {
        div()
            .absolute()
            .top_2()
            .right_2()
            .flex()
            .items_center()
            .gap_1()
            .p_1()
            .rounded_lg()
            .bg(tokens.panel_bg)
            .border_1()
            .border_color(tokens.border)
            .shadow_xs()
            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .invisible()
            .group_hover(hover_group.clone(), |style| style.visible())
            .child(row_action_svg_icon_button(
                format!("{row_id}-edit"),
                tokens,
                "assets/icons/edit.svg",
                "编辑标题和注释",
                ToolbarButtonTone::Neutral,
                on_edit,
            ))
            .child(row_action_svg_icon_button(
                format!("{row_id}-delete"),
                tokens,
                "assets/icons/trash.svg",
                "删除路线",
                ToolbarButtonTone::Danger,
                on_request_delete,
            ))
            .into_any_element()
    };

    div()
        .id(row_id.clone())
        .group(hover_group.clone())
        .relative()
        .w_full()
        .min_w_0()
        .flex()
        .items_center()
        .justify_between()
        .gap_3()
        .px_3()
        .py_2()
        .rounded_lg()
        .bg(if active {
            tokens.nav_subitem_active_bg
        } else {
            tokens.nav_item_bg
        })
        .border_1()
        .border_color(if active {
            tokens.border_strong
        } else {
            tokens.border
        })
        .cursor_pointer()
        .hover(|style| style.bg(tokens.nav_item_hover_bg))
        .active(|style| style.opacity(0.92))
        .on_click(on_click)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap_3()
                .child(
                    div()
                        .w(px(5.0))
                        .h(px(if active { 20.0 } else { 10.0 }))
                        .flex_shrink_0()
                        .rounded_full()
                        .bg(if active {
                            tokens.border_strong
                        } else {
                            tokens.border
                        }),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .overflow_hidden()
                        .child(
                            div()
                                .w_full()
                                .text_sm()
                                .font_weight(if active {
                                    gpui::FontWeight::SEMIBOLD
                                } else {
                                    gpui::FontWeight::NORMAL
                                })
                                .text_color(tokens.app_fg)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(title),
                        )
                        .child(
                            div()
                                .w_full()
                                .text_xs()
                                .text_color(tokens.text_muted)
                                .line_height(px(18.0))
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(subtitle),
                        ),
                ),
        )
        .child(
            div().flex_shrink_0().flex().items_center().gap_1().child(
                div()
                    .rounded_full()
                    .bg(tokens.panel_alt_bg)
                    .border_1()
                    .border_color(tokens.border)
                    .px_2()
                    .py_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(tokens.text_soft)
                            .child(format!("{} 点", point_count)),
                    ),
            ),
        )
        .child(overlay)
}

fn marker_group_edit_row(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    title_input: &gpui::Entity<gpui_component::input::InputState>,
    note_input: &gpui::Entity<gpui_component::input::InputState>,
    point_count: usize,
    on_save: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
    on_cancel: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let row_id = id.into();

    div()
        .id(row_id.clone())
        .w_full()
        .min_w_0()
        .flex()
        .items_start()
        .justify_between()
        .gap_3()
        .px_3()
        .py_3()
        .rounded_lg()
        .bg(tokens.nav_subitem_active_bg)
        .border_1()
        .border_color(tokens.border_strong)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_start()
                .gap_3()
                .child(
                    div()
                        .mt_2()
                        .w(px(5.0))
                        .h(px(20.0))
                        .flex_shrink_0()
                        .rounded_full()
                        .bg(tokens.border_strong),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(Input::new(title_input))
                        .child(Input::new(note_input)),
                ),
        )
        .child(
            div()
                .flex_shrink_0()
                .flex()
                .items_center()
                .gap_1()
                .child(
                    div()
                        .rounded_full()
                        .bg(tokens.panel_alt_bg)
                        .border_1()
                        .border_color(tokens.border)
                        .px_2()
                        .py_1()
                        .child(
                            div()
                                .text_xs()
                                .text_color(tokens.text_soft)
                                .child(format!("{} 点", point_count)),
                        ),
                )
                .child(row_action_svg_icon_button(
                    format!("{row_id}-save"),
                    tokens,
                    "assets/icons/check.svg",
                    "保存标题和注释",
                    ToolbarButtonTone::Primary,
                    on_save,
                ))
                .child(row_action_svg_icon_button(
                    format!("{row_id}-cancel"),
                    tokens,
                    "assets/icons/close.svg",
                    "取消编辑",
                    ToolbarButtonTone::Neutral,
                    on_cancel,
                )),
        )
}

fn pager_icon_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    disabled: bool,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    toolbar_icon_button(
        id,
        tokens,
        icon,
        match icon {
            "<<" => "跳到第一页",
            "<" => "上一页",
            ">" => "下一页",
            ">>" => "跳到最后一页",
            _ => "分页操作",
        },
        ToolbarButtonTone::Neutral,
        disabled,
        on_click,
    )
}

fn pager_jump_control(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    page_input: &gpui::Entity<gpui_component::input::InputState>,
    page_count: usize,
) -> impl IntoElement {
    div()
        .id(id.into())
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .w(px(44.0))
                .h(px(30.0))
                .rounded_md()
                .bg(tokens.nav_item_bg)
                .border_1()
                .border_color(tokens.border)
                .px_1()
                .child(Input::new(page_input).appearance(false).px(px(0.0)).gap_0()),
        )
        .child(
            div()
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.text_muted)
                .child(format!("/ {}", page_count)),
        )
}

fn empty_list_state(
    tokens: WorkbenchThemeTokens,
    text: impl Into<SharedString>,
) -> impl IntoElement {
    div()
        .rounded_lg()
        .bg(tokens.panel_alt_bg)
        .border_1()
        .border_color(tokens.border)
        .px_3()
        .py_6()
        .child(
            div()
                .text_sm()
                .text_color(tokens.text_muted)
                .line_height(px(22.0))
                .child(text.into()),
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolbarButtonTone {
    Neutral,
    Primary,
    Danger,
}

fn toolbar_cluster(children: Vec<AnyElement>) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .flex_wrap()
        .children(children)
}

fn toolbar_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    label: impl Into<SharedString>,
    tone: ToolbarButtonTone,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    toolbar_button_with_tooltip(id, tokens, icon, label, None, tone, false, on_click)
}

fn toolbar_button_with_tooltip(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    label: impl Into<SharedString>,
    tooltip: Option<SharedString>,
    tone: ToolbarButtonTone,
    disabled: bool,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let label = label.into();
    let (background, hover_background, border_color) = match tone {
        ToolbarButtonTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
        ToolbarButtonTone::Primary => (
            tokens.toolbar_button_primary_bg,
            tokens.toolbar_button_primary_hover_bg,
            tokens.border_strong,
        ),
        ToolbarButtonTone::Danger => (
            tokens.toolbar_button_danger_bg,
            tokens.toolbar_button_danger_hover_bg,
            tokens.border_strong,
        ),
    };

    div()
        .id(id.into())
        .h(px(34.0))
        .px_3()
        .flex()
        .items_center()
        .gap_2()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .when_some(tooltip, |button, tooltip| {
            button.tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
        })
        .when(!disabled, |button| {
            button
                .cursor_pointer()
                .hover(move |style| style.bg(hover_background))
                .active(|style| style.opacity(0.92))
                .on_click(on_click)
        })
        .when(disabled, |button| button.opacity(0.42))
        .child(
            div()
                .w(px(18.0))
                .h(px(18.0))
                .flex()
                .items_center()
                .justify_center()
                .rounded_full()
                .bg(tokens.nav_item_bg)
                .child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(tokens.app_fg)
                        .child(icon),
                ),
        )
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .child(label),
        )
}

fn toolbar_icon_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    tooltip: impl Into<SharedString>,
    tone: ToolbarButtonTone,
    disabled: bool,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        ToolbarButtonTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
        ToolbarButtonTone::Primary => (
            tokens.toolbar_button_primary_bg,
            tokens.toolbar_button_primary_hover_bg,
            tokens.border_strong,
        ),
        ToolbarButtonTone::Danger => (
            tokens.toolbar_button_danger_bg,
            tokens.toolbar_button_danger_hover_bg,
            tokens.border_strong,
        ),
    };

    div()
        .id(id.into())
        .w(px(34.0))
        .h(px(34.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
        .when(!disabled, |button| {
            button
                .cursor_pointer()
                .hover(move |style| style.bg(hover_background))
                .active(|style| style.opacity(0.92))
                .on_click(on_click)
        })
        .when(disabled, |button| button.opacity(0.42))
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .child(icon),
        )
}

fn toolbar_svg_icon_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon_path: &'static str,
    tooltip: impl Into<SharedString>,
    tone: ToolbarButtonTone,
    disabled: bool,
    on_click: impl Fn(&ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let tooltip = tooltip.into();
    let (background, hover_background, border_color) = match tone {
        ToolbarButtonTone::Neutral => (
            tokens.toolbar_button_bg,
            tokens.toolbar_button_hover_bg,
            tokens.border,
        ),
        ToolbarButtonTone::Primary => (
            tokens.toolbar_button_primary_bg,
            tokens.toolbar_button_primary_hover_bg,
            tokens.border_strong,
        ),
        ToolbarButtonTone::Danger => (
            tokens.toolbar_button_danger_bg,
            tokens.toolbar_button_danger_hover_bg,
            tokens.border_strong,
        ),
    };

    div()
        .id(id.into())
        .w(px(34.0))
        .h(px(34.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
        .when(!disabled, |button| {
            button
                .cursor_pointer()
                .hover(move |style| style.bg(hover_background))
                .active(|style| style.opacity(0.92))
                .on_click(on_click)
        })
        .when(disabled, |button| button.opacity(0.42))
        .child(
            Icon::default()
                .path(icon_path)
                .small()
                .text_color(tokens.app_fg),
        )
}

fn theme_preference_icon_path(preference: ThemePreference) -> &'static str {
    match preference {
        ThemePreference::FollowSystem => "assets/icons/theme-system.svg",
        ThemePreference::Light => "assets/icons/theme-light.svg",
        ThemePreference::Dark => "assets/icons/theme-dark.svg",
    }
}

fn page_header(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let theme_button = toolbar_svg_icon_button(
        "theme-cycle",
        tokens,
        theme_preference_icon_path(this.theme_preference),
        format!("切换主题 · 当前 {}", this.theme_preference),
        ToolbarButtonTone::Neutral,
        false,
        cx.listener(|this, _: &ClickEvent, window, cx| {
            let next = match this.theme_preference {
                ThemePreference::FollowSystem => ThemePreference::Light,
                ThemePreference::Light => ThemePreference::Dark,
                ThemePreference::Dark => ThemePreference::FollowSystem,
            };
            this.set_theme_preference(next, window, cx);
            cx.notify();
        }),
    )
    .into_any_element();
    let controls = match (this.active_page, this.map_page) {
        (WorkbenchPage::Map, MapPage::Tracker) => vec![
            toolbar_cluster(vec![
                toolbar_icon_button(
                    "preview-prev-top",
                    tokens,
                    "◀",
                    "上一个节点",
                    ToolbarButtonTone::Neutral,
                    false,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.step_preview(-1);
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_icon_button(
                    "preview-next-top",
                    tokens,
                    "▶",
                    "下一个节点",
                    ToolbarButtonTone::Neutral,
                    false,
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.step_preview(1);
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button(
                    "focus-toggle-top",
                    tokens,
                    "F",
                    if this.is_auto_focus_enabled() {
                        "关闭聚焦"
                    } else {
                        "开启聚焦"
                    },
                    if this.is_auto_focus_enabled() {
                        ToolbarButtonTone::Danger
                    } else {
                        ToolbarButtonTone::Primary
                    },
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.set_auto_focus_enabled(!this.is_auto_focus_enabled());
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button(
                    "tracker-popup-toggle-top",
                    tokens,
                    "P",
                    if this.is_tracker_point_popup_enabled() {
                        "隐藏浮窗"
                    } else {
                        "显示浮窗"
                    },
                    if this.is_tracker_point_popup_enabled() {
                        ToolbarButtonTone::Neutral
                    } else {
                        ToolbarButtonTone::Primary
                    },
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.set_tracker_point_popup_enabled(
                            !this.is_tracker_point_popup_enabled(),
                        );
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button_with_tooltip(
                    "tracker-toggle-top",
                    tokens,
                    if this.is_tracker_transition_pending() {
                        this.busy_spinner_icon()
                    } else {
                        "R"
                    },
                    this.tracker_toggle_label(),
                    Some(this.tracker_status_tooltip()),
                    if this.is_tracking_active()
                        || matches!(this.tracker_status_summary().as_ref(), "停止中")
                    {
                        ToolbarButtonTone::Danger
                    } else {
                        ToolbarButtonTone::Primary
                    },
                    this.is_tracker_transition_pending(),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        if this.is_tracking_active() {
                            this.stop_tracker(false);
                        } else {
                            this.start_tracker();
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button(
                    "tracker-pip-toggle",
                    tokens,
                    "P",
                    this.tracker_pip_toggle_label(),
                    if this.is_tracker_pip_open() || this.is_tracker_pip_pending_open() {
                        ToolbarButtonTone::Primary
                    } else {
                        ToolbarButtonTone::Neutral
                    },
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.toggle_tracker_pip_window(window, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button_with_tooltip(
                    "tracker-pip-topmost",
                    tokens,
                    if this.is_tracker_pip_always_on_top() {
                        "T"
                    } else {
                        "^"
                    },
                    this.tracker_pip_topmost_label(),
                    Some(this.tracker_pip_topmost_tooltip()),
                    if this.is_tracker_pip_always_on_top() {
                        ToolbarButtonTone::Primary
                    } else {
                        ToolbarButtonTone::Neutral
                    },
                    !this.is_tracker_pip_open(),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_tracker_pip_always_on_top(cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
                toolbar_button_with_tooltip(
                    "engine-cycle",
                    tokens,
                    "A",
                    this.selected_engine.to_string(),
                    Some(
                        if this.is_tracking_active() || this.is_tracker_transition_pending() {
                            "追踪运行或切换中时不能切换引擎。".into()
                        } else {
                            format!("切换追踪引擎，当前为 {}。", this.selected_engine).into()
                        },
                    ),
                    ToolbarButtonTone::Neutral,
                    this.is_tracking_active() || this.is_tracker_transition_pending(),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_engine();
                        cx.notify();
                    }),
                )
                .into_any_element(),
                theme_button,
            ])
            .into_any_element(),
        ],
        (WorkbenchPage::Map, MapPage::Bwiki) => {
            vec![toolbar_cluster(vec![theme_button]).into_any_element()]
        }
        _ => vec![toolbar_cluster(vec![theme_button]).into_any_element()],
    };

    div()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .px_4()
        .py_3()
        .child(
            div()
                .flex()
                .items_start()
                .justify_between()
                .gap_3()
                .flex_wrap()
                .child(div().flex_1())
                .children(controls),
        )
}

fn map_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    match this.map_page {
        MapPage::Tracker => div()
            .flex_1()
            .min_h(px(0.0))
            .flex()
            .gap_4()
            .overflow_hidden()
            .child(map_sidebar(this, cx, tokens))
            .child(map_panel(this, cx, tokens))
            .into_any_element(),
        MapPage::Bwiki => div()
            .flex_1()
            .min_h(px(0.0))
            .flex()
            .gap_4()
            .overflow_hidden()
            .child(bwiki_types_sidebar(this, cx, tokens))
            .child(bwiki_map_panel(this, cx, tokens))
            .into_any_element(),
    }
}

fn markers_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .flex_1()
        .min_h(px(0.0))
        .flex()
        .gap_4()
        .overflow_hidden()
        .child(group_manager(this, cx, tokens))
        .child(
            div()
                .flex_1()
                .min_w(px(0.0))
                .min_h(px(0.0))
                .flex()
                .flex_col()
                .gap_3()
                .overflow_hidden()
                .child(group_detail_panel(this, cx, tokens))
                .child(route_editor_map_panel(this, cx, tokens)),
        )
}

fn settings_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    match this.settings_page {
        SettingsPage::Runtime => settings_runtime_page(this, cx, tokens).into_any_element(),
        SettingsPage::Capture => settings_capture_page(this, cx, tokens).into_any_element(),
        SettingsPage::Convolution => settings_convolution_page(this, cx, tokens).into_any_element(),
        SettingsPage::Template => settings_template_page(this, cx, tokens).into_any_element(),
        SettingsPage::Debug => settings_debug_page(this, tokens).into_any_element(),
        SettingsPage::Resources => settings_resources_page(this, cx, tokens).into_any_element(),
    }
}

fn map_sidebar(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let group_query = normalized_query(&this.map_group_list.search, cx);
    let filtered_groups = this
        .route_groups
        .iter()
        .filter(|group| {
            matches_query(
                &group_query,
                [
                    group.display_name().to_owned(),
                    group.notes.clone(),
                    group.metadata.file_name.clone(),
                ],
            )
        })
        .collect::<Vec<_>>();
    let (visible_groups, pagination) = paginate_items(
        filtered_groups,
        this.map_group_list.page,
        this.map_group_list.page_size,
    );
    let group_rows = visible_groups
        .into_iter()
        .map(|group| {
            let group_id = group.id.clone();
            selectable_list_row(
                format!("map-group-{}", group.id),
                tokens,
                group.display_name().to_owned(),
                if group.notes.trim().is_empty() {
                    format!("{} 个节点", group.point_count())
                } else {
                    group.notes.clone()
                },
                this.selected_group_id.as_ref() == Some(&group.id),
                Some(format!("{} 点", group.point_count()).into()),
                cx.listener(move |this, _: &ClickEvent, window, cx| {
                    this.select_group(group_id.clone(), window, cx);
                    cx.notify();
                }),
            )
            .into_any_element()
        })
        .collect::<Vec<_>>();

    div()
        .w(px(320.0))
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .overflow_y_scrollbar()
        .child(paginated_list(
            "map-groups",
            cx,
            tokens,
            Some("路线".into()),
            Vec::new(),
            &this.map_group_list.search,
            Vec::new(),
            &this.map_group_list.page_input,
            pagination,
            "当前还没有可显示的路线。",
            group_rows,
            TrackerWorkbench::set_map_group_page,
        ))
}

fn group_manager(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let group_query = normalized_query(&this.marker_group_list.search, cx);
    let filtered_groups = this
        .route_groups
        .iter()
        .filter(|group| {
            matches_query(
                &group_query,
                [
                    group.display_name().to_owned(),
                    group.notes.clone(),
                    group.metadata.file_name.clone(),
                ],
            )
        })
        .collect::<Vec<_>>();
    let (visible_groups, pagination) = paginate_items(
        filtered_groups,
        this.marker_group_list.page,
        this.marker_group_list.page_size,
    );
    let group_rows = visible_groups
        .into_iter()
        .map(|group| {
            if this.is_group_being_edited(&group.id) {
                marker_group_edit_row(
                    format!("marker-group-edit-{}", group.id),
                    tokens,
                    &this.group_inline_edit.name,
                    &this.group_inline_edit.description,
                    group.point_count(),
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.commit_inline_group_edit(window, cx);
                        cx.notify();
                    }),
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.cancel_inline_group_edit(window, cx);
                        cx.notify();
                    }),
                )
                .into_any_element()
            } else {
                let group_id = group.id.clone();
                marker_group_list_row(
                    format!("marker-group-{}", group.id),
                    format!("marker-group-hover-{}", group.id),
                    tokens,
                    group.display_name().to_owned(),
                    if group.notes.trim().is_empty() {
                        group.metadata.file_name.clone()
                    } else {
                        group.notes.clone()
                    },
                    this.selected_group_id.as_ref() == Some(&group.id),
                    group.point_count(),
                    this.is_group_delete_confirmation_active(&group.id),
                    cx.listener(move |this, _: &ClickEvent, window, cx| {
                        this.select_group(group_id.clone(), window, cx);
                        cx.notify();
                    }),
                    {
                        let group_id = group.id.clone();
                        cx.listener(move |this, _: &ClickEvent, window, cx| {
                            this.start_group_inline_edit(group_id.clone(), window, cx);
                            cx.notify();
                        })
                    },
                    {
                        let group_id = group.id.clone();
                        cx.listener(move |this, _: &ClickEvent, window, cx| {
                            this.begin_group_delete_confirmation(group_id.clone(), window, cx);
                            cx.notify();
                        })
                    },
                    {
                        let group_id = group.id.clone();
                        cx.listener(move |this, _: &ClickEvent, window, cx| {
                            this.confirm_group_delete(group_id.clone(), window, cx);
                            cx.notify();
                        })
                    },
                    {
                        let group_id = group.id.clone();
                        cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.cancel_group_delete_confirmation(group_id.clone());
                            cx.notify();
                        })
                    },
                )
                .into_any_element()
            }
        })
        .collect::<Vec<_>>();

    div()
        .w(px(400.0))
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .gap_4()
        .overflow_y_scrollbar()
        .child(paginated_list(
            "marker-groups",
            cx,
            tokens,
            Some("路线管理".into()),
            vec![
                div()
                    .w_full()
                    .flex()
                    .justify_end()
                    .child(toolbar_cluster(vec![
                        toolbar_button_with_tooltip(
                            "group-import-files",
                            tokens,
                            if this.is_route_import_busy() {
                                this.busy_spinner_icon()
                            } else {
                                "I"
                            },
                            if this.is_route_import_busy() {
                                "处理中"
                            } else {
                                "导入文件"
                            },
                            Some(this.route_import_tooltip()),
                            ToolbarButtonTone::Neutral,
                            this.is_route_import_busy(),
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.import_route_files(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                        toolbar_button_with_tooltip(
                            "group-import-folder",
                            tokens,
                            if this.is_route_import_busy() {
                                this.busy_spinner_icon()
                            } else {
                                "D"
                            },
                            if this.is_route_import_busy() {
                                "处理中"
                            } else {
                                "导入文件夹"
                            },
                            Some(this.route_import_tooltip()),
                            ToolbarButtonTone::Neutral,
                            this.is_route_import_busy(),
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.import_route_folder(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                        toolbar_icon_button(
                            "group-inline-new",
                            tokens,
                            "+",
                            "新建路线",
                            ToolbarButtonTone::Neutral,
                            false,
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.create_group_inline_item(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                    ]))
                    .into_any_element(),
            ],
            &this.marker_group_list.search,
            Vec::new(),
            &this.marker_group_list.page_input,
            pagination,
            "当前还没有路线，请先导入文件或新建路线。",
            group_rows,
            TrackerWorkbench::set_marker_group_page,
        ))
}

fn group_detail_panel(
    this: &TrackerWorkbench,
    _cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let active_group = this.active_group().cloned();
    let group_icon_definition = this
        .bwiki_resources
        .resolve_icon_definition(this.group_icon.as_str());
    let bwiki_dataset_ready = this.bwiki_resources.dataset_snapshot().is_some();

    div()
        .flex_shrink_0()
        .w_full()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .when_some(active_group, |panel, _group| {
            panel
                .child(
                    div().flex().flex_wrap().gap_3().children([
                        div()
                            .w(px(280.0))
                            .child(labeled_input(tokens, "路线名称", &this.group_form.name))
                            .into_any_element(),
                        div()
                            .flex_1()
                            .min_w(px(360.0))
                            .child(labeled_input(
                                tokens,
                                "路线说明",
                                &this.group_form.description,
                            ))
                            .into_any_element(),
                    ]),
                )
                .child(
                    div().flex().flex_wrap().gap_3().mt_3().children([
                        div()
                            .w(px(180.0))
                            .child(labeled_input(
                                tokens,
                                "路线颜色",
                                &this.group_form.color_hex,
                            ))
                            .into_any_element(),
                        div()
                            .flex_1()
                            .min_w(px(360.0))
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(field_label(tokens, "默认节点图标"))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .when_some(group_icon_definition.clone(), |row, definition| {
                                        row.child(bwiki_type_icon_preview(
                                            definition.mark_type,
                                            definition.icon_url.clone(),
                                            this.bwiki_resources.clone(),
                                            tokens,
                                        ))
                                    })
                                    .child(
                                        Select::new(&this.group_icon_picker)
                                            .w_full()
                                            .menu_width(px(420.0))
                                            .placeholder("搜索并选择默认节点图标")
                                            .search_placeholder("按图标名、分类或编号搜索")
                                            .disabled(!bwiki_dataset_ready)
                                            .empty_message("BWiki 图标目录加载中，请稍后重试。"),
                                    ),
                            )
                            .into_any_element(),
                    ]),
                )
        })
        .when(this.active_group().is_none(), |panel| {
            panel.child(empty_list_state(tokens, "请先从左侧创建或选择一条路线。"))
        })
}

fn settings_editor_toolbar(
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    toolbar_cluster(vec![
        toolbar_button(
            "settings-config-reload",
            tokens,
            "R",
            "回填当前配置",
            ToolbarButtonTone::Neutral,
            cx.listener(|this, _: &ClickEvent, window, cx| {
                this.sync_config_form_from_workspace(window, cx);
                this.status_text = "已回填当前配置文件内容。".into();
                cx.notify();
            }),
        )
        .into_any_element(),
        toolbar_button(
            "settings-config-save",
            tokens,
            "S",
            "保存配置",
            ToolbarButtonTone::Primary,
            cx.listener(|this, _: &ClickEvent, window, cx| {
                this.save_app_config(window, cx);
                cx.notify();
            }),
        )
        .into_any_element(),
    ])
}

fn tracker_cache_rebuild_section(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
    kind: TrackerCacheKind,
    button_id: &'static str,
    tooltip: &'static str,
) -> AnyElement {
    editable_config_section(
        "缓存",
        vec![
            config_row(vec![
                toolbar_button_with_tooltip(
                    button_id,
                    tokens,
                    "R",
                    "重建缓存",
                    Some(tooltip.into()),
                    ToolbarButtonTone::Neutral,
                    this.is_cache_rebuild_running(kind),
                    cx.listener(move |this, _: &ClickEvent, window, cx| {
                        this.rebuild_tracker_cache(kind, window, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            ]),
            body_text(tokens, this.cache_rebuild_summary(kind)).into_any_element(),
        ],
        tokens,
    )
    .into_any_element()
}

fn settings_page_shell(
    title: &'static str,
    description: impl Into<SharedString>,
    header_actions: Option<AnyElement>,
    sections: Vec<AnyElement>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let description = description.into();

    div()
        .flex_1()
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .gap_4()
        .overflow_y_scrollbar()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .child(section_title(title))
                .when_some(header_actions, |header, actions| header.child(actions)),
        )
        .child(body_text(tokens, description))
        .children(sections)
}

fn settings_runtime_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let theme_mode_picker = ButtonGroup::new("theme-mode-picker")
        .children(
            ThemePreference::iter()
                .enumerate()
                .map(|(index, preference)| {
                    Button::new(("theme-mode", index))
                        .label(preference.to_string())
                        .selected(this.theme_preference == preference)
                })
                .collect::<Vec<_>>(),
        )
        .compact()
        .on_click(cx.listener(|this, indices: &Vec<usize>, window, cx| {
            if let Some(index) = indices.first().copied() {
                if let Some(preference) = ThemePreference::iter().nth(index) {
                    this.set_theme_preference(preference, window, cx);
                    cx.notify();
                }
            }
        }));

    settings_page_shell(
        "界面与运行",
        "这里放工作区级别的运行参数：界面外观、预览窗口、追踪容忍度和网络端口。它们不依赖某个具体引擎。",
        Some(settings_editor_toolbar(cx, tokens).into_any_element()),
        vec![
            editable_config_section(
                "界面",
                vec![
                    div()
                        .min_w(px(180.0))
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(field_label(tokens, "主题模式"))
                        .child(theme_mode_picker)
                        .into_any_element(),
                    config_row(vec![
                        labeled_input(tokens, "窗口几何", &this.config_form.window_geometry)
                            .into_any_element(),
                    ]),
                ],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "追踪运行",
                vec![config_row(vec![
                    labeled_input(tokens, "视图尺寸", &this.config_form.view_size)
                        .into_any_element(),
                    labeled_input(tokens, "最大丢帧", &this.config_form.max_lost_frames)
                        .into_any_element(),
                    labeled_input(
                        tokens,
                        "传送等效距离",
                        &this.config_form.teleport_link_distance,
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "网络接口",
                vec![config_row(vec![
                    labeled_input(tokens, "HTTP 端口", &this.config_form.network_http_port)
                        .into_any_element(),
                    labeled_input(
                        tokens,
                        "WebSocket 端口",
                        &this.config_form.network_websocket_port,
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
        ],
        tokens,
    )
}

fn settings_capture_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    settings_page_shell(
        "截图与局部搜索",
        "这组参数决定每帧从屏幕哪里取图，以及锁定后最多允许在周围多大范围内继续搜索。一般先校准截图，再调局部搜索阈值。",
        Some(settings_editor_toolbar(cx, tokens).into_any_element()),
        vec![
            editable_config_section(
                "截图区域",
                vec![
                    toolbar_cluster(vec![
                        toolbar_button_with_tooltip(
                            "settings-minimap-picker",
                            tokens,
                            if this.is_minimap_region_picker_active() {
                                this.busy_spinner_icon()
                            } else {
                                "P"
                            },
                            if this.is_minimap_region_picker_active() {
                                "取区中"
                            } else {
                                "手动取区"
                            },
                            Some(if this.is_minimap_region_picker_active() {
                                "小地图取区窗口已打开：先拖出圆，再拖圆心移动、拖圆边改半径，最后点确认。最终会保存为圆的外接正方形截图框。".into()
                            } else {
                                "打开屏幕取区窗口，拖出并微调小地图圆形范围。".into()
                            }),
                            if this.is_minimap_region_picker_active() {
                                ToolbarButtonTone::Primary
                            } else {
                                ToolbarButtonTone::Neutral
                            },
                            false,
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.toggle_minimap_region_picker(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                    ])
                    .into_any_element(),
                    config_row(vec![
                        labeled_input(tokens, "Top", &this.config_form.minimap_top)
                            .into_any_element(),
                        labeled_input(tokens, "Left", &this.config_form.minimap_left)
                            .into_any_element(),
                        labeled_input(tokens, "Width", &this.config_form.minimap_width)
                            .into_any_element(),
                        labeled_input(tokens, "Height", &this.config_form.minimap_height)
                            .into_any_element(),
                    ]),
                ],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "F1-P 标签探针",
                vec![
                    toolbar_cluster(vec![
                        toolbar_button_with_tooltip(
                            "settings-minimap-presence-probe-picker",
                            tokens,
                            if this.is_minimap_presence_probe_picker_active() {
                                this.busy_spinner_icon()
                            } else {
                                "F"
                            },
                            if this.is_minimap_presence_probe_picker_active() {
                                "取区中"
                            } else {
                                "标签取区"
                            },
                            Some(if this.is_minimap_presence_probe_picker_active() {
                                "F1-P 标签探针取区窗口已打开：请只框住标签带，不要包含上方图标；确认后会把当前区域抓成模板。".into()
                            } else {
                                "打开屏幕取区窗口，手动框选 F1 到 P 这排标签，并在确认时抓取模板。".into()
                            }),
                            if this.is_minimap_presence_probe_picker_active() {
                                ToolbarButtonTone::Primary
                            } else {
                                ToolbarButtonTone::Neutral
                            },
                            false,
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.toggle_minimap_presence_probe_picker(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                    ])
                    .into_any_element(),
                    config_row(vec![
                        labeled_input(
                            tokens,
                            "启用 true/false",
                            &this.config_form.minimap_presence_probe_enabled,
                        )
                        .into_any_element(),
                        labeled_input(
                            tokens,
                            "Top",
                            &this.config_form.minimap_presence_probe_top,
                        )
                        .into_any_element(),
                        labeled_input(
                            tokens,
                            "Left",
                            &this.config_form.minimap_presence_probe_left,
                        )
                        .into_any_element(),
                        labeled_input(
                            tokens,
                            "Width",
                            &this.config_form.minimap_presence_probe_width,
                        )
                        .into_any_element(),
                        labeled_input(
                            tokens,
                            "Height",
                            &this.config_form.minimap_presence_probe_height,
                        )
                        .into_any_element(),
                    ]),
                    config_row(vec![
                        labeled_input(
                            tokens,
                            "匹配阈值",
                            &this.config_form.minimap_presence_probe_match_threshold,
                        )
                        .into_any_element(),
                        labeled_select(
                            tokens,
                            "设备",
                            Select::new(&this.minimap_presence_probe_device_picker)
                                .icon(Icon::new(IconName::ChevronsUpDown))
                                .w_full()
                                .menu_width(px(420.0))
                                .placeholder("选择执行设备")
                                .search_placeholder("搜索 CPU / CUDA / Metal")
                                .empty_message("当前没有可用设备。"),
                        )
                        .into_any_element(),
                        labeled_select(
                            tokens,
                            "设备序号",
                            Select::new(&this.minimap_presence_probe_device_index_picker)
                                .icon(Icon::new(IconName::ChevronsUpDown))
                                .w_full()
                                .menu_width(px(420.0))
                                .placeholder("选择设备序号")
                                .search_placeholder("搜索设备序号")
                                .empty_message("当前后端没有可用设备。"),
                        )
                        .into_any_element(),
                    ]),
                ],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "局部搜索",
                vec![config_row(vec![
                    labeled_input(
                        tokens,
                        "启用 true/false",
                        &this.config_form.local_search_enabled,
                    )
                    .into_any_element(),
                    labeled_input(tokens, "搜索半径", &this.config_form.local_search_radius_px)
                        .into_any_element(),
                    labeled_input(
                        tokens,
                        "锁定失败阈值",
                        &this.config_form.local_search_lock_fail_threshold,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "最大跳变",
                        &this.config_form.local_search_max_accepted_jump_px,
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
        ],
        tokens,
    )
}

fn settings_convolution_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    settings_page_shell(
        "卷积特征匹配",
        "卷积特征匹配现在统一走 Burn 后端。设备下拉只显示当前后端真实可见的设备；CUDA 只列 NVIDIA CUDA 设备，Vulkan 可以覆盖 Intel / AMD / NVIDIA 的核显或独显。",
        Some(settings_editor_toolbar(cx, tokens).into_any_element()),
        vec![
            tracker_cache_rebuild_section(
                this,
                cx,
                tokens,
                TrackerCacheKind::Convolution,
                "settings-convolution-cache-rebuild",
                "删除并立即重建卷积特征匹配相关的预处理地图与全局搜索缓存。运行中的追踪需先停止。",
            ),
            editable_config_section(
                "执行设备与模型",
                vec![
                    config_row(vec![
                        labeled_select(
                            tokens,
                            "执行设备",
                            Select::new(&this.ai_device_picker)
                                .icon(Icon::new(IconName::ChevronsUpDown))
                                .w_full()
                                .menu_width(px(420.0))
                                .placeholder("选择执行设备")
                                .search_placeholder("搜索 CPU / CUDA / Vulkan / Metal")
                                .empty_message("当前没有可用设备。"),
                        )
                        .into_any_element(),
                        labeled_select(
                            tokens,
                            "设备序号",
                            Select::new(&this.ai_device_index_picker)
                                .icon(Icon::new(IconName::ChevronsUpDown))
                                .w_full()
                                .menu_width(px(420.0))
                                .placeholder("选择设备序号")
                                .search_placeholder("搜索设备序号")
                                .empty_message("当前后端没有可用设备。"),
                        )
                        .into_any_element(),
                    ]),
                    config_row(vec![
                        labeled_input(tokens, "权重路径", &this.config_form.ai_weights_path)
                            .into_any_element(),
                    ]),
                ],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "匹配参数",
                vec![config_row(vec![
                    labeled_input(tokens, "刷新间隔", &this.config_form.ai_refresh_rate_ms)
                        .into_any_element(),
                    labeled_input(
                        tokens,
                        "置信度阈值",
                        &this.config_form.ai_confidence_threshold,
                    )
                    .into_any_element(),
                    labeled_input(tokens, "最小匹配数", &this.config_form.ai_min_match_count)
                        .into_any_element(),
                    labeled_input(tokens, "RANSAC 阈值", &this.config_form.ai_ransac_threshold)
                        .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "搜索窗口",
                vec![config_row(vec![
                    labeled_input(tokens, "扫描尺寸", &this.config_form.ai_scan_size)
                        .into_any_element(),
                    labeled_input(tokens, "扫描步长", &this.config_form.ai_scan_step)
                        .into_any_element(),
                    labeled_input(tokens, "跟踪半径", &this.config_form.ai_track_radius)
                        .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
        ],
        tokens,
    )
}

fn settings_template_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    settings_page_shell(
        "多尺度模板匹配",
        "多尺度模板匹配现在同样走 Burn 设备抽象。设备下拉只显示当前后端真实可见的设备；CUDA 只列 NVIDIA CUDA 设备，Vulkan 可以覆盖 Intel / AMD / NVIDIA 的核显或独显。",
        Some(settings_editor_toolbar(cx, tokens).into_any_element()),
        vec![
            tracker_cache_rebuild_section(
                this,
                cx,
                tokens,
                TrackerCacheKind::Template,
                "settings-template-cache-rebuild",
                "删除并立即重建多尺度模板匹配相关的预处理地图与全局搜索缓存。运行中的追踪需先停止。",
            ),
            editable_config_section(
                "执行设备",
                vec![config_row(vec![
                    labeled_select(
                        tokens,
                        "执行设备",
                        Select::new(&this.template_device_picker)
                            .icon(Icon::new(IconName::ChevronsUpDown))
                            .w_full()
                            .menu_width(px(420.0))
                            .placeholder("选择执行设备")
                            .search_placeholder("搜索 CPU / CUDA / Vulkan / Metal")
                            .empty_message("当前没有可用设备。"),
                    )
                    .into_any_element(),
                    labeled_select(
                        tokens,
                        "设备序号",
                        Select::new(&this.template_device_index_picker)
                            .icon(Icon::new(IconName::ChevronsUpDown))
                            .w_full()
                            .menu_width(px(420.0))
                            .placeholder("选择设备序号")
                            .search_placeholder("搜索设备序号")
                            .empty_message("当前后端没有可用设备。"),
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "金字塔与细化",
                vec![config_row(vec![
                    labeled_input(
                        tokens,
                        "刷新间隔",
                        &this.config_form.template_refresh_rate_ms,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "局部缩放",
                        &this.config_form.template_local_downscale,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "全局缩放",
                        &this.config_form.template_global_downscale,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "全局细化半径",
                        &this.config_form.template_global_refine_radius_px,
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
            editable_config_section(
                "匹配阈值与遮罩",
                vec![config_row(vec![
                    labeled_input(
                        tokens,
                        "局部阈值",
                        &this.config_form.template_local_match_threshold,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "全局阈值",
                        &this.config_form.template_global_match_threshold,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "外圈半径",
                        &this.config_form.template_mask_outer_radius,
                    )
                    .into_any_element(),
                    labeled_input(
                        tokens,
                        "内圈半径",
                        &this.config_form.template_mask_inner_radius,
                    )
                    .into_any_element(),
                ])],
                tokens,
            )
            .into_any_element(),
        ],
        tokens,
    )
}

fn settings_debug_page(this: &TrackerWorkbench, tokens: WorkbenchThemeTokens) -> impl IntoElement {
    let snapshot = this.debug_snapshot.clone();
    let images = snapshot
        .as_ref()
        .map(|snapshot| snapshot.images.clone())
        .unwrap_or_default();
    let fields = snapshot
        .as_ref()
        .map(|snapshot| snapshot.fields.clone())
        .unwrap_or_default();

    settings_page_shell(
        "追踪调试",
        snapshot.as_ref().map_or_else(
            || "启动 tracker 后，这里会显示 minimap、heatmap、refine 预览和状态字段。".to_owned(),
            |snapshot| {
                format!(
                    "引擎 {}，阶段 {}，帧序号 {}。",
                    snapshot.engine, snapshot.stage_label, snapshot.frame_index
                )
            },
        ),
        None,
        vec![
            div()
                .flex()
                .gap_3()
                .flex_wrap()
                .children(
                    images
                        .into_iter()
                        .map(|image| debug_image_card(image, tokens))
                        .collect::<Vec<_>>(),
                )
                .into_any_element(),
            div()
                .flex()
                .gap_3()
                .flex_wrap()
                .children(
                    fields
                        .into_iter()
                        .map(|field| debug_field_card(field, tokens).into_any_element())
                        .collect::<Vec<_>>(),
                )
                .into_any_element(),
        ],
        tokens,
    )
}

fn settings_resources_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let models_dir = this
        .workspace
        .project_root
        .join("models")
        .display()
        .to_string();

    settings_page_shell(
        "本地数据路径",
        "路线文件、配置和 BWiki 运行时缓存都会真实落盘。BWiki 只缓存点位目录、图标和按需下载的瓦片，不再生成或保留整张拼接地图。",
        None,
        vec![
            resource_path(
                "resource-data-dir",
                cx,
                tokens,
                "数据目录",
                &this.project_root.to_string(),
            )
            .into_any_element(),
            resource_path(
                "resource-routes-dir",
                cx,
                tokens,
                "路线目录",
                &this.workspace.assets.routes_dir.display().to_string(),
            )
            .into_any_element(),
            resource_path(
                "resource-bwiki-cache-dir",
                cx,
                tokens,
                "BWiki 缓存目录",
                &this.workspace.assets.bwiki_cache_dir.display().to_string(),
            )
            .into_any_element(),
            resource_path("resource-models-dir", cx, tokens, "模型目录", &models_dir)
                .into_any_element(),
            resource_path(
                "resource-config-file",
                cx,
                tokens,
                "配置文件",
                &this.workspace.assets.config_path.display().to_string(),
            )
            .into_any_element(),
            resource_path(
                "resource-ui-preferences",
                cx,
                tokens,
                "界面偏好",
                &this.ui_preferences_path.display().to_string(),
            )
            .into_any_element(),
        ],
        tokens,
    )
}

fn config_section(
    title: &'static str,
    lines: Vec<String>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .rounded_xl()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .child(section_title(title))
                .children(
                    lines
                        .into_iter()
                        .map(|line| body_text(tokens, line).into_any_element())
                        .collect::<Vec<_>>(),
                ),
        )
}

fn editable_config_section(
    title: &'static str,
    rows: Vec<AnyElement>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .rounded_xl()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_3()
                .child(section_title(title))
                .children(rows),
        )
}

fn config_row(children: Vec<AnyElement>) -> AnyElement {
    div()
        .flex()
        .gap_2()
        .flex_wrap()
        .children(children)
        .into_any_element()
}

type MapOverlayPainter = fn(
    entity: &gpui::Entity<TrackerWorkbench>,
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    cx: &mut gpui::App,
    bounds_width: f32,
    bounds_height: f32,
    camera: crate::domain::geometry::MapCamera,
    tokens: WorkbenchThemeTokens,
);

fn map_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    map_canvas_panel(
        cx.entity(),
        tokens,
        "路线地图",
        None,
        MapCanvasKind::Tracker,
        this.workspace.report.map_dimensions,
        paint_tracker_map_overlay,
        selected_tracker_point_info_popup(this, tokens),
    )
}

fn route_editor_map_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    map_canvas_panel(
        cx.entity(),
        tokens,
        "路线预览与地图编辑",
        Some(
            toolbar_cluster(
                vec![
                    this.is_selected_point_move_armed().then(|| {
                        toolbar_button(
                            "group-map-pick-cancel",
                            tokens,
                            "x",
                            "取消取点",
                            ToolbarButtonTone::Neutral,
                            cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.toggle_selected_point_move_mode();
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    }),
                    Some(
                        toolbar_button(
                            "group-map-add-toggle",
                            tokens,
                            "+",
                            if this.is_map_point_insert_armed() {
                                "取消添加节点"
                            } else {
                                "添加节点"
                            },
                            if this.is_map_point_insert_armed() {
                                ToolbarButtonTone::Primary
                            } else {
                                ToolbarButtonTone::Neutral
                            },
                            cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.toggle_map_point_insert_mode();
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                    ),
                ]
                .into_iter()
                .flatten()
                .collect(),
            )
            .into_any_element(),
        ),
        MapCanvasKind::RouteEditor,
        this.workspace.report.map_dimensions,
        paint_tracker_map_overlay,
        selected_tracker_point_editor_popup(this, cx, tokens),
    )
}

fn bwiki_types_sidebar(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let dataset = this.bwiki_resources.dataset_snapshot();
    let bwiki_dataset_ready = dataset.is_some();
    let category_query = normalized_query(&this.bwiki_category_search, cx);
    let type_query = normalized_query(&this.bwiki_type_search, cx);
    let bwiki_resources = this.bwiki_resources.clone();
    let planner_selected_count = this.bwiki_planner_selected_count();
    let planner_total_cost = this.bwiki_planner_preview_total_cost();
    let mut grouped = BTreeMap::<String, Vec<(u32, String, String, usize)>>::new();
    if let Some(dataset) = dataset.as_ref() {
        for definition in &dataset.types {
            grouped
                .entry(definition.category.clone())
                .or_default()
                .push((
                    definition.mark_type,
                    definition.name.clone(),
                    definition.icon_url.clone(),
                    definition.point_count,
                ));
        }
    }

    let category_cards = grouped
        .into_iter()
        .enumerate()
        .filter_map(|(category_index, (category, definitions))| {
            if !matches_query(&category_query, [category.as_str()]) {
                return None;
            }

            let filtered_definitions = definitions
                .into_iter()
                .filter(|(mark_type, name, _, point_count)| {
                    matches_query(
                        &type_query,
                        [name.clone(), mark_type.to_string(), point_count.to_string()],
                    )
                })
                .collect::<Vec<_>>();
            if !type_query.is_empty() && filtered_definitions.is_empty() {
                return None;
            }

            let bwiki_resources = bwiki_resources.clone();
            let visible_type_count = filtered_definitions
                .iter()
                .filter(|(mark_type, _, _, _)| this.bwiki_visible_mark_types.contains(mark_type))
                .count();
            let total_definition_count = filtered_definitions.len();
            let total_point_count = filtered_definitions
                .iter()
                .map(|(_, _, _, count)| count)
                .sum::<usize>();
            let expanded = this.bwiki_expanded_categories.contains(&category);
            let category_for_expand = category.clone();
            let category_for_show = category.clone();
            let category_for_hide = category.clone();

            Some(
                div()
                    .rounded_lg()
                    .bg(tokens.panel_sunken_bg)
                    .border_1()
                    .border_color(tokens.border)
                    .p_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .id(("bwiki-category-toggle", category_index))
                                    .flex_1()
                                    .cursor_pointer()
                                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.toggle_bwiki_category_expanded(&category_for_expand);
                                        cx.notify();
                                    }))
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                                    .text_color(tokens.app_fg)
                                                    .child(format!(
                                                        "{} {}",
                                                        if expanded { "v" } else { ">" },
                                                        category
                                                    )),
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(tokens.text_muted)
                                                    .child(format!(
                                                        "{}/{} 项显示 · {} 点",
                                                        visible_type_count,
                                                        total_definition_count,
                                                        total_point_count
                                                    )),
                                            ),
                                    ),
                            )
                            .child(toolbar_cluster(vec![
                                toolbar_icon_button(
                                    format!("bwiki-category-show-{category_for_show}"),
                                    tokens,
                                    "+",
                                    format!("显示分类 {}", category_for_show),
                                    ToolbarButtonTone::Primary,
                                    false,
                                    cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.set_bwiki_category_visibility(
                                            &category_for_show,
                                            true,
                                        );
                                        cx.notify();
                                    }),
                                )
                                .into_any_element(),
                                toolbar_icon_button(
                                    format!("bwiki-category-hide-{category_for_hide}"),
                                    tokens,
                                    "-",
                                    format!("隐藏分类 {}", category_for_hide),
                                    ToolbarButtonTone::Danger,
                                    false,
                                    cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.set_bwiki_category_visibility(
                                            &category_for_hide,
                                            false,
                                        );
                                        cx.notify();
                                    }),
                                )
                                .into_any_element(),
                            ])),
                    )
                    .when(expanded, |card| {
                        let type_rows = filtered_definitions
                            .chunks(2)
                            .map(|definition_pair| {
                                let row = definition_pair.iter().fold(
                                    div().flex().gap_2(),
                                    |row, (mark_type, name, icon_url, point_count)| {
                                        row.child(
                                            bwiki_type_toggle_button(
                                                *mark_type,
                                                name.clone(),
                                                icon_url.clone(),
                                                *point_count,
                                                this.bwiki_visible_mark_types.contains(mark_type),
                                                bwiki_resources.clone(),
                                                tokens,
                                                cx,
                                            )
                                            .into_any_element(),
                                        )
                                    },
                                );
                                let row = if definition_pair.len() == 1 {
                                    row.child(
                                        div()
                                            .w(px(BWIKI_TYPE_BUTTON_WIDTH))
                                            .h(px(BWIKI_TYPE_BUTTON_HEIGHT)),
                                    )
                                } else {
                                    row
                                };
                                row.into_any_element()
                            })
                            .collect::<Vec<_>>();
                        card.child(div().mt_3().flex().flex_col().gap_2().children(type_rows))
                    })
                    .into_any_element(),
            )
        })
        .collect::<Vec<_>>();

    let last_error = this.bwiki_resources.last_error();
    div()
        .w(px(420.0))
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .gap_4()
        .overflow_y_scrollbar()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(section_title("节点图鉴"))
        .child(bwiki_route_planner_card(
            this,
            cx,
            tokens,
            bwiki_dataset_ready,
            planner_selected_count,
            planner_total_cost,
        ))
        .child(
            div()
                .rounded_lg()
                .bg(tokens.panel_sunken_bg)
                .border_1()
                .border_color(tokens.border)
                .p_3()
                .flex()
                .flex_col()
                .gap_3()
                .child(field_label(tokens, "节点过滤"))
                .child(div().flex().gap_2().children([
                    labeled_input(tokens, "分类", &this.bwiki_category_search).into_any_element(),
                    labeled_input(tokens, "节点", &this.bwiki_type_search).into_any_element(),
                ]))
                .child(div().h(px(1.0)).w_full().bg(tokens.border))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_2()
                        .child(
                            toolbar_button(
                                "bwiki-sidebar-show-all",
                                tokens,
                                "A",
                                "显示全部",
                                ToolbarButtonTone::Primary,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.show_all_bwiki_types();
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                        )
                        .child(
                            toolbar_button(
                                "bwiki-sidebar-hide-all",
                                tokens,
                                "H",
                                "全部隐藏",
                                ToolbarButtonTone::Danger,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.hide_all_bwiki_types();
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                        )
                        .child(
                            toolbar_button_with_tooltip(
                                "bwiki-sidebar-refresh",
                                tokens,
                                if this.is_bwiki_refreshing() {
                                    this.busy_spinner_icon()
                                } else {
                                    "R"
                                },
                                if this.is_bwiki_refreshing() {
                                    "刷新中"
                                } else {
                                    "刷新数据"
                                },
                                Some(this.bwiki_status_tooltip()),
                                ToolbarButtonTone::Neutral,
                                this.is_bwiki_refreshing(),
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.refresh_bwiki_dataset();
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                        ),
                ),
        )
        .when_some(last_error, |panel, error| {
            panel.child(config_section("最近一次缓存错误", vec![error], tokens))
        })
        .when(dataset.is_none(), |panel| {
            panel.child(config_section(
                "正在同步节点图鉴数据",
                vec![
                    "首次启动会请求点位目录与类型目录。".to_owned(),
                    "缓存准备好以后，这里会列出所有分类与节点类型。".to_owned(),
                ],
                tokens,
            ))
        })
        .when(dataset.is_some() && category_cards.is_empty(), |panel| {
            panel.child(empty_list_state(tokens, "没有匹配的分类或节点。"))
        })
        .when(!category_cards.is_empty(), |panel| {
            panel.children(category_cards)
        })
}

fn bwiki_route_planner_card(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
    bwiki_dataset_ready: bool,
    planner_selected_count: usize,
    planner_total_cost: Option<f32>,
) -> impl IntoElement {
    let preview_summary = if this.is_bwiki_planner_busy() {
        format!("已选 {} 个点 · 正在规划", planner_selected_count)
    } else {
        planner_total_cost
            .map(|total_cost| {
                format!(
                    "已选 {} 个点 · 预计长度 {:.0}",
                    planner_selected_count, total_cost
                )
            })
            .unwrap_or_else(|| format!("已选 {} 个点 · 尚未规划", planner_selected_count))
    };
    let plan_disabled =
        planner_selected_count == 0 || !bwiki_dataset_ready || this.is_bwiki_planner_busy();
    let create_disabled = this.is_bwiki_planner_busy() || !this.bwiki_planner_has_preview();

    div()
        .rounded_lg()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .flex()
        .flex_col()
        .gap_3()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .child(field_label(tokens, "路线规划"))
                .child(
                    toolbar_button_with_tooltip(
                        "bwiki-planner-toggle",
                        tokens,
                        "P",
                        if this.is_bwiki_planner_active() {
                            "退出规划"
                        } else {
                            "进入规划模式"
                        },
                        if bwiki_dataset_ready {
                            None
                        } else {
                            Some(this.bwiki_status_tooltip())
                        },
                        if this.is_bwiki_planner_active() {
                            ToolbarButtonTone::Primary
                        } else {
                            ToolbarButtonTone::Neutral
                        },
                        !bwiki_dataset_ready,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.toggle_bwiki_planner_mode(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ),
        )
        .when(this.is_bwiki_planner_active(), |card| {
            card.child(
                div()
                    .text_xs()
                    .text_color(tokens.text_muted)
                    .child("左键点选，Ctrl 左键手绘闭合曲线可反选。"),
            )
            .child(
                div()
                    .rounded_lg()
                    .bg(tokens.panel_alt_bg)
                    .border_1()
                    .border_color(tokens.border)
                    .px_3()
                    .py_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(tokens.app_fg)
                            .child(preview_summary),
                    ),
            )
            .child(
                div().flex().gap_2().children([
                    labeled_input(tokens, "名称", &this.bwiki_planner_form.name).into_any_element(),
                    labeled_input(tokens, "说明", &this.bwiki_planner_form.description)
                        .into_any_element(),
                ]),
            )
            .child(
                div().flex().gap_2().children([
                    labeled_input(tokens, "颜色", &this.bwiki_planner_form.color_hex)
                        .into_any_element(),
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(field_label(tokens, "默认图标"))
                        .child(
                            Select::new(&this.bwiki_planner_icon_picker)
                                .w_full()
                                .menu_width(px(360.0))
                                .placeholder("选择路线默认图标")
                                .search_placeholder("按名称、分类或编号搜索")
                                .empty_message("当前没有可用的 BWiki 图标。"),
                        )
                        .into_any_element(),
                ]),
            )
            .child(
                div().flex().justify_end().gap_2().children([
                    toolbar_button_with_tooltip(
                        "bwiki-planner-clear",
                        tokens,
                        "C",
                        "清空已选",
                        None,
                        ToolbarButtonTone::Neutral,
                        planner_selected_count == 0,
                        cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.clear_bwiki_planner_selection();
                            this.status_text = "已清空当前规划点。".into();
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button_with_tooltip(
                        "bwiki-planner-plan",
                        tokens,
                        if this.is_bwiki_planner_busy() {
                            this.busy_spinner_icon()
                        } else {
                            "P"
                        },
                        if this.is_bwiki_planner_busy() {
                            "规划中"
                        } else {
                            "规划路线"
                        },
                        Some(this.bwiki_planner_tooltip()),
                        ToolbarButtonTone::Neutral,
                        plan_disabled,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.plan_bwiki_route_preview(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button_with_tooltip(
                        "bwiki-planner-create",
                        tokens,
                        "+",
                        "创建路线",
                        None,
                        ToolbarButtonTone::Primary,
                        create_disabled,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.create_route_from_bwiki_planner(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ]),
            )
        })
}

fn bwiki_type_toggle_button(
    mark_type: u32,
    name: String,
    icon_url: String,
    point_count: usize,
    visible: bool,
    bwiki_resources: BwikiResourceManager,
    tokens: WorkbenchThemeTokens,
    cx: &mut Context<TrackerWorkbench>,
) -> impl IntoElement {
    let label = name.clone();
    let count_label = point_count.to_string();
    let background = if visible {
        tokens.toolbar_button_primary_bg
    } else {
        tokens.panel_alt_bg
    };
    let hover_background = if visible {
        tokens.toolbar_button_primary_hover_bg
    } else {
        tokens.toolbar_button_hover_bg
    };
    let border_color = if visible {
        tokens.border_strong
    } else {
        tokens.border
    };

    div()
        .id(("bwiki-type-toggle", mark_type))
        .w(px(BWIKI_TYPE_BUTTON_WIDTH))
        .h(px(BWIKI_TYPE_BUTTON_HEIGHT))
        .px_2()
        .py_2()
        .flex()
        .flex_row()
        .items_center()
        .gap_2()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .when(point_count > 0, |button| {
            button
                .cursor_pointer()
                .hover(move |style| style.bg(hover_background))
                .active(|style| style.opacity(0.92))
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                    this.toggle_bwiki_type_visibility(mark_type, &label);
                    cx.notify();
                }))
        })
        .when(point_count == 0, |button| button.opacity(0.45))
        .child(bwiki_type_icon_preview(
            mark_type,
            icon_url,
            bwiki_resources,
            tokens,
        ))
        .child(
            div()
                .w(px(BWIKI_TYPE_NAME_WIDTH))
                .min_w(px(BWIKI_TYPE_NAME_WIDTH))
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(tokens.app_fg)
                .whitespace_nowrap()
                .text_ellipsis()
                .child(name),
        )
        .child(
            div()
                .w(px(BWIKI_TYPE_COUNT_WIDTH))
                .min_w(px(BWIKI_TYPE_COUNT_WIDTH))
                .flex()
                .justify_center()
                .child(
                    div()
                        .w_full()
                        .text_xs()
                        .text_color(tokens.text_muted)
                        .whitespace_nowrap()
                        .text_ellipsis()
                        .child(count_label),
                ),
        )
}

fn bwiki_type_icon_preview(
    mark_type: u32,
    icon_url: String,
    bwiki_resources: BwikiResourceManager,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    canvas(
        move |_, _, _| bwiki_resources.ensure_icon_path(mark_type, &icon_url),
        move |bounds, icon_path, window, cx| {
            window.paint_quad(fill(bounds, tokens.panel_sunken_bg).corner_radii(px(10.0)));

            if let Some(path) = icon_path.as_ref() {
                let resource = gpui::Resource::from(path.clone());
                let image_result = window.use_asset::<ImgResourceLoader>(&resource, cx);
                if let Some(Ok(image)) = image_result.as_ref() {
                    let image_bounds = bwiki_type_icon_bounds(bounds);
                    let _ = window.paint_image(image_bounds, 0.0.into(), image.clone(), 0, false);
                    return;
                }
            }

            paint_bwiki_placeholder_marker(
                window,
                point(
                    bounds.origin.x + px(f32::from(bounds.size.width) * 0.5),
                    bounds.origin.y + px(f32::from(bounds.size.height) * 0.5),
                ),
                mark_type,
                tokens,
            );
        },
    )
    .w(px(BWIKI_TYPE_ICON_BOX_SIZE))
    .h(px(BWIKI_TYPE_ICON_BOX_SIZE))
}

fn bwiki_map_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    map_canvas_panel(
        cx.entity(),
        tokens,
        "节点图鉴",
        None,
        MapCanvasKind::Bwiki,
        this.workspace.report.map_dimensions,
        paint_bwiki_map_overlay,
        None,
    )
}

fn debug_image_card(
    image: crate::tracking::debug::DebugImage,
    tokens: WorkbenchThemeTokens,
) -> AnyElement {
    let title = format!("{} · {}", image.label, image.kind);
    div()
        .w(px(220.0))
        .rounded_lg()
        .bg(tokens.debug_card_bg)
        .border_1()
        .border_color(tokens.border_strong)
        .p_3()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .child(div().text_xs().text_color(tokens.text_muted).child(title))
                .child(debug_image_canvas(Some(image), tokens)),
        )
        .into_any_element()
}

fn debug_image_canvas(
    image: Option<crate::tracking::debug::DebugImage>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    canvas(
        move |_, _, _| image.clone(),
        move |bounds, image, window, _| {
            window.paint_quad(fill(bounds, tokens.debug_canvas_bg));
            let Some(image) = image.as_ref() else {
                return;
            };
            if image.width == 0 || image.height == 0 || image.pixels.is_empty() {
                return;
            }

            let scale_x = f32::from(bounds.size.width) / image.width as f32;
            let scale_y = f32::from(bounds.size.height) / image.height as f32;
            match image.format {
                crate::tracking::debug::DebugImageFormat::Gray8 => {
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
                crate::tracking::debug::DebugImageFormat::Rgba8 => {
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
    .h(px(180.0))
    .w_full()
}

fn debug_field_card(
    field: crate::tracking::debug::DebugField,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    div()
        .rounded_lg()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .px_3()
        .py_2()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_xs()
                        .text_color(tokens.text_muted)
                        .child(field.label),
                )
                .child(div().text_sm().text_color(tokens.app_fg).child(field.value)),
        )
}

fn map_canvas_panel(
    entity: gpui::Entity<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
    title: &'static str,
    header_actions: Option<AnyElement>,
    map_kind: MapCanvasKind,
    map_dimensions: crate::domain::geometry::MapDimensions,
    overlay_painter: MapOverlayPainter,
    overlay_ui: Option<AnyElement>,
) -> impl IntoElement {
    div()
        .flex_1()
        .min_h(px(0.0))
        .flex()
        .flex_col()
        .overflow_hidden()
        .rounded_xl()
        .bg(tokens.panel_deep_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .mb_3()
                .child(section_title(title))
                .when_some(header_actions, |row, actions| row.child(actions)),
        )
        .child(
            div()
                .flex_1()
                .overflow_hidden()
                .relative()
                .child(
                    canvas(
                        move |_, _, _| (),
                        move |bounds, _, window, cx| {
                            let bounds_width = f32::from(bounds.size.width);
                            let bounds_height = f32::from(bounds.size.height);
                            sync_map_canvas_viewport(
                                &entity,
                                cx,
                                map_kind,
                                bounds_width,
                                bounds_height,
                                map_dimensions,
                            );

                            let (camera, bwiki_resources, bwiki_tile_cache) = {
                                let this = entity.read(cx);
                                (
                                    this.map_camera(map_kind),
                                    this.bwiki_resources.clone(),
                                    this.bwiki_tile_cache.clone(),
                                )
                            };

                            paint_bwiki_tile_layers(
                                window,
                                bounds,
                                cx,
                                camera,
                                &bwiki_resources,
                                &bwiki_tile_cache,
                                tokens.map_canvas_backdrop,
                            );
                            overlay_painter(
                                &entity,
                                window,
                                bounds,
                                cx,
                                bounds_width,
                                bounds_height,
                                camera,
                                tokens,
                            );
                            install_map_canvas_navigation_handlers(
                                window,
                                entity.clone(),
                                bounds,
                                map_kind,
                            );
                        },
                    )
                    .size_full(),
                )
                .when_some(overlay_ui, |container, overlay| container.child(overlay)),
        )
}

fn selected_tracker_point_info_popup(
    this: &TrackerWorkbench,
    tokens: WorkbenchThemeTokens,
) -> Option<AnyElement> {
    let popup = this.selected_tracker_point_popup()?;
    let point = this.selected_point()?;

    Some(
        div()
            .absolute()
            .left(px(popup.left))
            .top(px(popup.top))
            .w(px(popup.width))
            .rounded_xl()
            .bg(tokens.panel_bg)
            .border_1()
            .border_color(tokens.border_strong)
            .shadow_xs()
            .px_3()
            .py_3()
            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .on_mouse_up(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .child(
                div().flex().items_start().gap_2().child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(tokens.app_fg)
                                .child(point.display_label().to_owned()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(tokens.text_muted)
                                .child(popup.route_name),
                        ),
                ),
            )
            .child(
                div()
                    .mt_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(tokens.text_muted)
                            .child(format!("坐标 {:.0}, {:.0}", point.x, point.y)),
                    )
                    .when(!point.note.trim().is_empty(), |column| {
                        column.child(
                            div()
                                .text_xs()
                                .line_height(px(18.0))
                                .text_color(tokens.text_soft)
                                .child(point.note.clone()),
                        )
                    }),
            )
            .into_any_element(),
    )
}

fn selected_tracker_point_editor_popup(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> Option<AnyElement> {
    let popup = this.selected_tracker_point_popup()?;
    let point = this.selected_point()?;
    let reorder_picker_disabled = this
        .active_group()
        .map(|group| group.points.len() < 2)
        .unwrap_or(true);
    let reorder_action_disabled = reorder_picker_disabled || this.point_reorder_target_id.is_none();
    let delete_confirming = this.is_selected_point_delete_confirming();

    Some(
        div()
            .absolute()
            .left(px(popup.left))
            .top(px(popup.top))
            .w(px(popup.width))
            .rounded_xl()
            .bg(tokens.panel_bg)
            .border_1()
            .border_color(tokens.border_strong)
            .shadow_xs()
            .px_3()
            .py_3()
            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .on_mouse_up(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
            .child(
                div().flex().items_start().justify_between().gap_3().child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(tokens.app_fg)
                                .child(point.display_label().to_owned()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(tokens.text_muted)
                                .child(popup.route_name),
                        ),
                ),
            )
            .child(
                div()
                    .mt_3()
                    .child(labeled_input(tokens, "节点名称", &this.marker_form.label)),
            )
            .child(
                div()
                    .mt_3()
                    .child(labeled_input(tokens, "备注", &this.marker_form.note)),
            )
            .child(
                div().mt_3().flex().items_end().gap_3().children([
                    labeled_input(tokens, "X", &this.marker_form.x).into_any_element(),
                    labeled_input(tokens, "Y", &this.marker_form.y).into_any_element(),
                    toolbar_button(
                        "popup-point-move-toggle",
                        tokens,
                        "M",
                        "取点",
                        ToolbarButtonTone::Neutral,
                        cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.toggle_selected_point_move_mode();
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ]),
            )
            .child(
                div()
                    .mt_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(field_label(tokens, "顺序调整"))
                    .child(div().flex().justify_center().child(toolbar_cluster(vec![
                            toolbar_icon_button(
                                "popup-point-move-start",
                                tokens,
                                "|<",
                                "移到起点",
                                ToolbarButtonTone::Neutral,
                                false,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_to_start(window, cx);
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                            toolbar_icon_button(
                                "popup-point-move-prev",
                                tokens,
                                "↑",
                                "上移一个节点",
                                ToolbarButtonTone::Neutral,
                                false,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_prev(window, cx);
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                            toolbar_icon_button(
                                "popup-point-move-next",
                                tokens,
                                "↓",
                                "下移一个节点",
                                ToolbarButtonTone::Neutral,
                                false,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_next(window, cx);
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                            toolbar_icon_button(
                                "popup-point-move-end",
                                tokens,
                                ">|",
                                "移到终点",
                                ToolbarButtonTone::Neutral,
                                false,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_to_end(window, cx);
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                        ]))),
            )
            .child(
                div()
                    .mt_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(field_label(tokens, "相对目标重排"))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div().flex_1().min_w_0().child(
                                    Select::new(&this.point_reorder_picker)
                                        .w_full()
                                        .menu_width(px(360.0))
                                        .placeholder("搜索并选择目标节点")
                                        .search_placeholder("按序号、名称、备注或坐标搜索")
                                        .disabled(reorder_picker_disabled)
                                        .empty_message("当前没有可作为目标的其他节点。"),
                                ),
                            )
                            .child(toolbar_icon_button(
                                "popup-point-move-before-target",
                                tokens,
                                "<",
                                "移到目标前",
                                ToolbarButtonTone::Neutral,
                                reorder_action_disabled,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_before_target(window, cx);
                                    cx.notify();
                                }),
                            ))
                            .child(toolbar_icon_button(
                                "popup-point-move-after-target",
                                tokens,
                                ">",
                                "移到目标后",
                                ToolbarButtonTone::Neutral,
                                reorder_action_disabled,
                                cx.listener(|this, _: &ClickEvent, window, cx| {
                                    this.move_selected_point_after_target(window, cx);
                                    cx.notify();
                                }),
                            )),
                    ),
            )
            .child(
                div()
                    .mt_4()
                    .flex()
                    .justify_end()
                    .child(toolbar_cluster(vec![
                        toolbar_button(
                            "popup-point-delete",
                            tokens,
                            if delete_confirming { "!" } else { "-" },
                            if delete_confirming {
                                "确认删除节点"
                            } else {
                                "删除节点"
                            },
                            ToolbarButtonTone::Danger,
                            cx.listener(|this, _: &ClickEvent, window, cx| {
                                this.confirm_or_delete_selected_point(window, cx);
                                cx.notify();
                            }),
                        )
                        .into_any_element(),
                    ])),
            )
            .into_any_element(),
    )
}

pub(super) fn paint_tracker_map_overlay(
    entity: &gpui::Entity<TrackerWorkbench>,
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    cx: &mut gpui::App,
    _: f32,
    _: f32,
    camera: crate::domain::geometry::MapCamera,
    tokens: WorkbenchThemeTokens,
) {
    let (snapshot, bwiki_resources) = {
        let this = entity.read(cx);
        (
            this.tracker_map_render_snapshot(),
            this.bwiki_resources.clone(),
        )
    };

    paint_tracker_map_overlay_snapshot(
        window,
        bounds,
        cx,
        camera,
        tokens,
        &snapshot,
        &bwiki_resources,
    );
}

pub(super) fn paint_tracker_map_overlay_snapshot(
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    cx: &mut gpui::App,
    camera: crate::domain::geometry::MapCamera,
    tokens: WorkbenchThemeTokens,
    snapshot: &TrackerMapRenderSnapshot,
    bwiki_resources: &BwikiResourceManager,
) {
    let route_color_hex = snapshot.route_color_hex.clone();
    let trail = snapshot.trail.clone();
    let preview_position = snapshot.preview_position.clone();
    let route_world = snapshot.route_world.clone();
    let point_visuals = snapshot.point_visuals.clone();
    let selected_group_id = snapshot.selected_group_id.clone();
    let selected_point_id = snapshot.selected_point_id.clone();

    window.paint_layer(bounds, |window| {
        window.with_content_mask(Some(ContentMask { bounds }), |window| {
            let route_icon_assets = point_visuals
                .iter()
                .filter_map(|marker| {
                    let icon_name = marker.style.icon.clone();
                    let definition = bwiki_resources.resolve_icon_definition(icon_name.as_str())?;
                    let image = bwiki_resources
                        .ensure_icon_path(definition.mark_type, &definition.icon_url)
                        .and_then(|path| {
                            let resource = gpui::Resource::from(path);
                            window
                                .use_asset::<ImgResourceLoader>(&resource, cx)
                                .and_then(|result| result.ok())
                        });
                    Some((icon_name, (definition.mark_type, image)))
                })
                .collect::<HashMap<_, _>>();

            if trail.len() > 1 {
                let trail_points = screen_points(camera, &trail);
                let mut builder = PathBuilder::stroke(px(2.0));
                let first = trail_points[0];
                builder.move_to(point(
                    bounds.origin.x + px(first.x),
                    bounds.origin.y + px(first.y),
                ));
                for screen_point in trail_points.iter().skip(1) {
                    builder.line_to(point(
                        bounds.origin.x + px(screen_point.x),
                        bounds.origin.y + px(screen_point.y),
                    ));
                }
                if let Ok(path) = builder.build() {
                    window.paint_path(path, tokens.trail_path);
                }
            }

            if let Some(route_color_hex) = route_color_hex.as_ref() {
                let route_color = gpui::rgb(parse_hex_color(route_color_hex, 0xff6b6b));
                if route_world.len() > 1 {
                    let route_screen = screen_points(camera, &route_world);
                    let route_canvas = route_screen
                        .iter()
                        .map(|screen_point| {
                            point(
                                bounds.origin.x + px(screen_point.x),
                                bounds.origin.y + px(screen_point.y),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut builder = PathBuilder::stroke(px(3.0));
                    builder.move_to(route_canvas[0]);
                    for canvas_point in route_canvas.iter().skip(1) {
                        builder.line_to(*canvas_point);
                    }
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, route_color);
                    }

                    for segment in route_canvas.windows(2) {
                        paint_route_arrow(window, segment[0], segment[1], route_color.into());
                    }
                    paint_route_segment_lengths(
                        window,
                        cx,
                        &route_world,
                        &route_canvas,
                        tokens,
                        route_color.into(),
                    );
                }
            }

            let mut badge_items = Vec::new();
            for marker in point_visuals {
                let screen = camera.world_to_screen(marker.world);
                let highlighted = selected_group_id.as_ref() == Some(&marker.group_id)
                    && selected_point_id.as_ref() == Some(&marker.point_id);
                let anchor = point(
                    bounds.origin.x + px(screen.x),
                    bounds.origin.y + px(screen.y),
                );
                let accent = parse_hex_color(&marker.style.color_hex, 0xff6b6b);
                let marker_bounds = if let Some((mark_type, image)) =
                    route_icon_assets.get(&marker.style.icon)
                {
                    let marker_bounds = bwiki_marker_image_bounds(anchor);
                    paint_bwiki_style_marker(
                        window,
                        anchor,
                        highlighted,
                        tokens,
                        |window, bounds| {
                            if let Some(image) = image.as_ref() {
                                let _ =
                                    window.paint_image(bounds, 0.0.into(), image.clone(), 0, false);
                            } else {
                                paint_bwiki_placeholder_marker(window, anchor, *mark_type, tokens);
                            }
                        },
                    );
                    marker_bounds
                } else {
                    let marker_bounds = route_marker_bounds(anchor, 24.0);
                    paint_route_marker(window, anchor, 24.0, accent, highlighted, false, tokens);
                    marker_bounds
                };
                if marker.is_start || marker.is_end {
                    badge_items.push(RoutePointBadgeRenderItem {
                        marker_bounds,
                        is_start: marker.is_start,
                        is_end: marker.is_end,
                    });
                }
            }

            if !badge_items.is_empty() {
                window.paint_layer(bounds, |window| {
                    window.with_content_mask(Some(ContentMask { bounds }), |window| {
                        for badge in &badge_items {
                            paint_route_point_badges(
                                window,
                                badge.marker_bounds,
                                badge.is_start,
                                badge.is_end,
                                tokens,
                            );
                            paint_route_point_endpoint_label(
                                window,
                                cx,
                                badge.marker_bounds,
                                badge.is_start,
                                badge.is_end,
                                tokens,
                            );
                        }
                    });
                });
            }

            if let Some(position) = preview_position
                .as_ref()
                .filter(|position| position.source != TrackingSource::ManualPreview)
            {
                let screen = camera.world_to_screen(position.world);
                let marker_color = if position.inertial {
                    tokens.preview_inertial
                } else {
                    tokens.preview_live
                };
                let outer = Bounds {
                    origin: point(
                        bounds.origin.x + px(screen.x - 9.0),
                        bounds.origin.y + px(screen.y - 9.0),
                    ),
                    size: size(px(18.0), px(18.0)),
                };
                let inner = Bounds {
                    origin: point(
                        bounds.origin.x + px(screen.x - 6.0),
                        bounds.origin.y + px(screen.y - 6.0),
                    ),
                    size: size(px(12.0), px(12.0)),
                };
                window.paint_quad(fill(outer, tokens.preview_ring).corner_radii(px(9.0)));
                window.paint_quad(fill(inner, marker_color).corner_radii(px(6.0)));
            }
        });
    });
}

fn paint_bwiki_map_overlay(
    entity: &gpui::Entity<TrackerWorkbench>,
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    cx: &mut gpui::App,
    bounds_width: f32,
    bounds_height: f32,
    camera: crate::domain::geometry::MapCamera,
    tokens: WorkbenchThemeTokens,
) {
    let (
        dataset,
        bwiki_resources,
        visible_mark_types,
        selected_keys,
        planner_preview_points,
        planner_preview_worlds,
        planner_color_hex,
        planner_lasso_selection,
    ) = {
        let this = entity.read(cx);
        (
            this.bwiki_resources.dataset_snapshot(),
            this.bwiki_resources.clone(),
            this.bwiki_visible_mark_types.clone(),
            this.bwiki_planner_selected_points.clone(),
            this.bwiki_planner_preview_points(),
            this.bwiki_planner_preview_worlds(),
            this.bwiki_planner_route_color_hex(cx),
            this.bwiki_planner_lasso_selection.clone(),
        )
    };

    window.paint_layer(bounds, |window| {
        window.with_content_mask(Some(ContentMask { bounds }), |window| {
            if let Some(dataset) = dataset.as_ref() {
                let preview_color = gpui::rgb(parse_hex_color(&planner_color_hex, 0xff6b6b));
                let preview_keys = planner_preview_points
                    .iter()
                    .map(|point| point.key.clone())
                    .collect::<HashSet<_>>();
                let first_preview_key = planner_preview_points
                    .first()
                    .map(|point| point.key.clone());
                let last_preview_key = planner_preview_points.last().map(|point| point.key.clone());
                let visible_definitions = dataset
                    .types
                    .iter()
                    .filter(|item| visible_mark_types.contains(&item.mark_type))
                    .collect::<Vec<_>>();
                let hidden_preview_points = planner_preview_points
                    .iter()
                    .filter(|point| !visible_mark_types.contains(&point.record.mark_type))
                    .collect::<Vec<_>>();
                let icon_images = visible_definitions
                    .iter()
                    .map(|definition| {
                        (
                            definition.mark_type,
                            bwiki_resources
                                .ensure_icon_path(definition.mark_type, &definition.icon_url)
                                .and_then(|path| {
                                    let resource = gpui::Resource::from(path);
                                    window
                                        .use_asset::<ImgResourceLoader>(&resource, cx)
                                        .and_then(|result| result.ok())
                                }),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                let hidden_preview_icon_images = hidden_preview_points
                    .iter()
                    .filter_map(|point| {
                        let definition = point.type_definition.as_ref()?;
                        Some((
                            definition.mark_type,
                            bwiki_resources
                                .ensure_icon_path(definition.mark_type, &definition.icon_url)
                                .and_then(|path| {
                                    let resource = gpui::Resource::from(path);
                                    window
                                        .use_asset::<ImgResourceLoader>(&resource, cx)
                                        .and_then(|result| result.ok())
                                }),
                        ))
                    })
                    .collect::<BTreeMap<_, _>>();
                let mut badge_items = Vec::new();

                if planner_preview_worlds.len() > 1 {
                    let preview_screen = screen_points(camera, &planner_preview_worlds);
                    let preview_canvas = preview_screen
                        .iter()
                        .map(|screen_point| {
                            point(
                                bounds.origin.x + px(screen_point.x),
                                bounds.origin.y + px(screen_point.y),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut builder = PathBuilder::stroke(px(3.0));
                    let first = preview_canvas[0];
                    builder.move_to(point(first.x, first.y));
                    for canvas_point in preview_canvas.iter().skip(1) {
                        builder.line_to(*canvas_point);
                    }
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, preview_color);
                    }

                    for segment in preview_canvas.windows(2) {
                        paint_route_arrow(window, segment[0], segment[1], preview_color.into());
                    }
                    paint_route_segment_lengths(
                        window,
                        cx,
                        &planner_preview_worlds,
                        &preview_canvas,
                        tokens,
                        preview_color.into(),
                    );
                }

                if let Some(selection) = planner_lasso_selection.as_ref()
                    && selection.path.len() > 1
                {
                    let mut builder = PathBuilder::stroke(px(2.0));
                    let first = selection.path[0];
                    builder.move_to(point(
                        bounds.origin.x + px(first.x),
                        bounds.origin.y + px(first.y),
                    ));
                    for screen_point in selection.path.iter().skip(1) {
                        builder.line_to(point(
                            bounds.origin.x + px(screen_point.x),
                            bounds.origin.y + px(screen_point.y),
                        ));
                    }
                    if selection.path.len() > 2 {
                        builder.line_to(point(
                            bounds.origin.x + px(first.x),
                            bounds.origin.y + px(first.y),
                        ));
                    }
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, preview_color);
                    }
                }

                for definition in visible_definitions {
                    let Some(points) = dataset.points_by_type.get(&definition.mark_type) else {
                        continue;
                    };

                    for point_record in points {
                        let screen = camera.world_to_screen(point_record.world);
                        if screen.x < -BWIKI_MARKER_CULL_MARGIN
                            || screen.y < -BWIKI_MARKER_CULL_MARGIN
                            || screen.x > bounds_width + BWIKI_MARKER_CULL_MARGIN
                            || screen.y > bounds_height + BWIKI_MARKER_CULL_MARGIN
                        {
                            continue;
                        }

                        let anchor = point(
                            bounds.origin.x + px(screen.x),
                            bounds.origin.y + px(screen.y),
                        );
                        let key = super::BwikiPointKey::from_record(point_record);
                        let highlighted =
                            selected_keys.contains(&key) || preview_keys.contains(&key);
                        let marker_bounds = bwiki_marker_image_bounds(anchor);
                        paint_bwiki_style_marker(
                            window,
                            anchor,
                            highlighted,
                            tokens,
                            |window, bounds| {
                                if let Some(Some(image)) = icon_images.get(&definition.mark_type) {
                                    let _ = window.paint_image(
                                        bounds,
                                        0.0.into(),
                                        image.clone(),
                                        0,
                                        false,
                                    );
                                } else {
                                    paint_bwiki_placeholder_marker(
                                        window,
                                        anchor,
                                        definition.mark_type,
                                        tokens,
                                    );
                                }
                            },
                        );
                        if first_preview_key
                            .as_ref()
                            .is_some_and(|preview_key| preview_key == &key)
                            || last_preview_key
                                .as_ref()
                                .is_some_and(|preview_key| preview_key == &key)
                        {
                            badge_items.push(RoutePointBadgeRenderItem {
                                marker_bounds,
                                is_start: first_preview_key
                                    .as_ref()
                                    .is_some_and(|preview_key| preview_key == &key),
                                is_end: last_preview_key
                                    .as_ref()
                                    .is_some_and(|preview_key| preview_key == &key),
                            });
                        }
                    }
                }

                for preview_point in hidden_preview_points {
                    let screen = camera.world_to_screen(preview_point.record.world);
                    if screen.x < -BWIKI_MARKER_CULL_MARGIN
                        || screen.y < -BWIKI_MARKER_CULL_MARGIN
                        || screen.x > bounds_width + BWIKI_MARKER_CULL_MARGIN
                        || screen.y > bounds_height + BWIKI_MARKER_CULL_MARGIN
                    {
                        continue;
                    }

                    let anchor = point(
                        bounds.origin.x + px(screen.x),
                        bounds.origin.y + px(screen.y),
                    );
                    let marker_bounds = bwiki_marker_image_bounds(anchor);
                    let mark_type = preview_point.record.mark_type;
                    let preview_definition = preview_point.type_definition.as_ref();
                    paint_bwiki_style_marker(window, anchor, true, tokens, |window, bounds| {
                        if let Some(definition) = preview_definition
                            && let Some(Some(image)) =
                                hidden_preview_icon_images.get(&definition.mark_type)
                        {
                            let _ = window.paint_image(bounds, 0.0.into(), image.clone(), 0, false);
                        } else {
                            paint_bwiki_placeholder_marker(window, anchor, mark_type, tokens);
                        }
                    });
                    let is_start = first_preview_key
                        .as_ref()
                        .is_some_and(|preview_key| preview_key == &preview_point.key);
                    let is_end = last_preview_key
                        .as_ref()
                        .is_some_and(|preview_key| preview_key == &preview_point.key);
                    if is_start || is_end {
                        badge_items.push(RoutePointBadgeRenderItem {
                            marker_bounds,
                            is_start,
                            is_end,
                        });
                    }
                }

                if !badge_items.is_empty() {
                    window.paint_layer(bounds, |window| {
                        window.with_content_mask(Some(ContentMask { bounds }), |window| {
                            for badge in &badge_items {
                                paint_route_point_badges(
                                    window,
                                    badge.marker_bounds,
                                    badge.is_start,
                                    badge.is_end,
                                    tokens,
                                );
                                paint_route_point_endpoint_label(
                                    window,
                                    cx,
                                    badge.marker_bounds,
                                    badge.is_start,
                                    badge.is_end,
                                    tokens,
                                );
                            }
                        });
                    });
                }
            }
        });
    });
}

pub(super) fn paint_bwiki_tile_layers(
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    cx: &mut gpui::App,
    camera: crate::domain::geometry::MapCamera,
    bwiki_resources: &BwikiResourceManager,
    bwiki_tile_cache: &gpui::Entity<crate::ui::tile_cache::TileImageCache>,
    backdrop: gpui::Hsla,
) {
    let viewport = crate::domain::geometry::ViewportSize {
        width: f32::from(bounds.size.width),
        height: f32::from(bounds.size.height),
    };
    let preferred_layer = visible_tile_layers(camera, viewport, 1).into_iter().last();

    window.paint_layer(bounds, |window| {
        window.with_content_mask(Some(ContentMask { bounds }), |window| {
            window.paint_quad(fill(bounds, backdrop));

            if let Some(layer) = preferred_layer.as_ref() {
                for tile in &layer.tiles {
                    let Some(path) = bwiki_resources.ensure_tile_path(tile.zoom, tile.x, tile.y)
                    else {
                        continue;
                    };
                    let Some(tile_origin) =
                        tile_coordinate_to_world_origin(tile.zoom, tile.x, tile.y)
                    else {
                        continue;
                    };

                    let resource = gpui::Resource::from(path);
                    let image_result =
                        bwiki_tile_cache.update(cx, |cache, cx| cache.load(&resource, window, cx));
                    if let Some(Ok(image)) = image_result {
                        let screen_origin = camera.world_to_screen(tile_origin);
                        let screen_bottom_right =
                            camera.world_to_screen(crate::domain::geometry::WorldPoint::new(
                                tile_origin.x + tile.world_size as f32,
                                tile_origin.y + tile.world_size as f32,
                            ));
                        let Some(tile_range) = zoom_world_bounds(tile.zoom) else {
                            continue;
                        };
                        let image_bounds = snapped_tile_image_bounds(
                            bounds,
                            screen_origin.x,
                            screen_origin.y,
                            screen_bottom_right.x,
                            screen_bottom_right.y,
                            window.scale_factor(),
                            tile.x > tile_range.min_x,
                            tile.y > tile_range.min_y,
                            tile.x < tile_range.max_x,
                            tile.y < tile_range.max_y,
                        );
                        let _ = window.paint_image(image_bounds, 0.0.into(), image, 0, false);
                    }
                }
            }
        });
    });
}

fn snapped_tile_image_bounds(
    canvas_bounds: Bounds<gpui::Pixels>,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    scale_factor: f32,
    extend_left: bool,
    extend_top: bool,
    extend_right: bool,
    extend_bottom: bool,
) -> Bounds<gpui::Pixels> {
    let canvas_left = f32::from(canvas_bounds.origin.x);
    let canvas_top = f32::from(canvas_bounds.origin.y);
    let scale_factor = scale_factor.max(0.0001);
    let overdraw = 1.0f32;

    // GPUI snaps image origins with floor() and sizes with ceil(). If adjacent tiles
    // are painted from independent float origins/sizes, tiny rounding differences turn
    // into 1px seams or overlaps. Snap shared boundaries first in device pixels so all
    // neighboring tiles land on the same final edge.
    let left_device = ((canvas_left + left) * scale_factor).round();
    let top_device = ((canvas_top + top) * scale_factor).round();
    let right_device = ((canvas_left + right) * scale_factor).round();
    let bottom_device = ((canvas_top + bottom) * scale_factor).round();

    let snapped_left = left_device - if extend_left { overdraw } else { 0.0 };
    let snapped_top = top_device - if extend_top { overdraw } else { 0.0 };
    let snapped_right = right_device + if extend_right { overdraw } else { 0.0 };
    let snapped_bottom = bottom_device + if extend_bottom { overdraw } else { 0.0 };

    let width_device = (snapped_right - snapped_left).max(1.0);
    let height_device = (snapped_bottom - snapped_top).max(1.0);

    Bounds {
        origin: point(
            px(snapped_left / scale_factor),
            px(snapped_top / scale_factor),
        ),
        size: size(
            px(width_device / scale_factor),
            px(height_device / scale_factor),
        ),
    }
}

fn sync_map_canvas_viewport(
    entity: &gpui::Entity<TrackerWorkbench>,
    cx: &mut gpui::App,
    map_kind: MapCanvasKind,
    bounds_width: f32,
    bounds_height: f32,
    map_dimensions: crate::domain::geometry::MapDimensions,
) {
    _ = entity.update(cx, |this, cx| {
        if this.sync_map_canvas_view(map_kind, bounds_width, bounds_height, map_dimensions) {
            cx.notify();
        }
    });
}

fn install_map_canvas_navigation_handlers(
    window: &mut gpui::Window,
    entity: gpui::Entity<TrackerWorkbench>,
    bounds: Bounds<gpui::Pixels>,
    map_kind: MapCanvasKind,
) {
    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseDownEvent, _, _, cx| {
            if event.button != MouseButton::Left || !bounds.contains(&event.position) {
                return;
            }
            let route_point_map = matches!(
                map_kind,
                MapCanvasKind::Tracker | MapCanvasKind::RouteEditor
            );
            let reorder_menu_open = if map_kind == MapCanvasKind::RouteEditor {
                entity.read(cx).point_reorder_picker.read(cx).is_open()
            } else {
                false
            };
            let bwiki_circle_mode = map_kind == MapCanvasKind::Bwiki
                && event.modifiers.control
                && entity.read(cx).is_bwiki_planner_active();

            _ = entity.update(cx, |this, _| {
                let local_x = f32::from(event.position.x) - f32::from(bounds.origin.x);
                let local_y = f32::from(event.position.y) - f32::from(bounds.origin.y);
                if (route_point_map && this.tracker_popup_hit_test(local_x, local_y))
                    || reorder_menu_open
                {
                    this.suppress_next_tracker_mouse_up();
                    return;
                }
                if map_kind == MapCanvasKind::RouteEditor && this.is_selected_point_move_armed() {
                    return;
                }
                if bwiki_circle_mode {
                    this.begin_bwiki_planner_lasso_selection(local_x, local_y);
                    return;
                }
                this.begin_map_drag(
                    map_kind,
                    f32::from(event.position.x),
                    f32::from(event.position.y),
                );
            });
        }
    });
    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseMoveEvent, _, _, cx| {
            let route_point_map = matches!(
                map_kind,
                MapCanvasKind::Tracker | MapCanvasKind::RouteEditor
            );
            let reorder_menu_open = if map_kind == MapCanvasKind::RouteEditor {
                entity.read(cx).point_reorder_picker.read(cx).is_open()
            } else {
                false
            };
            let bwiki_circle_active = map_kind == MapCanvasKind::Bwiki
                && event.pressed_button == Some(MouseButton::Left)
                && entity.read(cx).bwiki_planner_lasso_selection.is_some();
            _ = entity.update(cx, |this, cx| {
                let local_x = f32::from(event.position.x) - f32::from(bounds.origin.x);
                let local_y = f32::from(event.position.y) - f32::from(bounds.origin.y);
                if bwiki_circle_active
                    && this.update_bwiki_planner_lasso_selection(local_x, local_y)
                {
                    cx.notify();
                    return;
                }
                if route_point_map
                    && bounds.contains(&event.position)
                    && (this.tracker_popup_hit_test(local_x, local_y) || reorder_menu_open)
                {
                    return;
                }
                if map_kind == MapCanvasKind::RouteEditor
                    && bounds.contains(&event.position)
                    && this.preview_selected_point_move(local_x, local_y)
                {
                    cx.notify();
                    return;
                }

                if this.update_map_drag(
                    map_kind,
                    f32::from(event.position.x),
                    f32::from(event.position.y),
                    MAP_INTERACTION_FRAME_INTERVAL,
                ) {
                    cx.notify();
                }
            });
        }
    });
    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &MouseUpEvent, _, window, cx| {
            let local_x = f32::from(event.position.x) - f32::from(bounds.origin.x);
            let local_y = f32::from(event.position.y) - f32::from(bounds.origin.y);
            let released_inside = bounds.contains(&event.position);
            let route_point_map = matches!(
                map_kind,
                MapCanvasKind::Tracker | MapCanvasKind::RouteEditor
            );
            let reorder_menu_open = if map_kind == MapCanvasKind::RouteEditor {
                entity.read(cx).point_reorder_picker.read(cx).is_open()
            } else {
                false
            };
            let bwiki_circle_active = map_kind == MapCanvasKind::Bwiki
                && event.button == MouseButton::Left
                && entity.read(cx).bwiki_planner_lasso_selection.is_some();
            _ = entity.update(cx, |this, cx| {
                if bwiki_circle_active {
                    if this.finish_bwiki_planner_lasso_selection(local_x, local_y) {
                        cx.notify();
                    }
                    return;
                }

                let popup_hit = route_point_map
                    && released_inside
                    && (this.tracker_popup_hit_test(local_x, local_y) || reorder_menu_open);
                if popup_hit {
                    let outcome = this.end_map_drag(map_kind);
                    if outcome.redraw {
                        cx.notify();
                    }
                    this.consume_tracker_mouse_up_guard();
                    return;
                }

                if route_point_map && this.consume_tracker_mouse_up_guard() {
                    let outcome = this.end_map_drag(map_kind);
                    if outcome.redraw {
                        cx.notify();
                    }
                    return;
                }

                if map_kind == MapCanvasKind::RouteEditor && this.is_selected_point_move_armed() {
                    if released_inside {
                        this.handle_route_map_click(map_kind, local_x, local_y, window, cx);
                        cx.notify();
                    }
                    return;
                }

                let outcome = this.end_map_drag(map_kind);
                if outcome.redraw {
                    cx.notify();
                }
                if outcome.clicked && released_inside && route_point_map {
                    this.handle_route_map_click(map_kind, local_x, local_y, window, cx);
                    cx.notify();
                } else if outcome.clicked
                    && released_inside
                    && map_kind == MapCanvasKind::Bwiki
                    && this.handle_bwiki_planner_click(local_x, local_y)
                {
                    cx.notify();
                }
            });
        }
    });
    window.on_mouse_event({
        let entity = entity.clone();
        move |event: &ScrollWheelEvent, _, _, cx| {
            if !bounds.contains(&event.position) {
                return;
            }

            let anchor_x = f32::from(event.position.x) - f32::from(bounds.origin.x);
            let anchor_y = f32::from(event.position.y) - f32::from(bounds.origin.y);
            let route_point_map = matches!(
                map_kind,
                MapCanvasKind::Tracker | MapCanvasKind::RouteEditor
            );
            let reorder_menu_open = if map_kind == MapCanvasKind::RouteEditor {
                entity.read(cx).point_reorder_picker.read(cx).is_open()
            } else {
                false
            };
            _ = entity.update(cx, |this, cx| {
                if route_point_map
                    && (this.tracker_popup_hit_test(anchor_x, anchor_y) || reorder_menu_open)
                {
                    return;
                }
                let delta = match event.delta {
                    ScrollDelta::Pixels(delta) => (f32::from(delta.y) / 320.0).clamp(-0.35, 0.35),
                    ScrollDelta::Lines(delta) => (delta.y / 8.0).clamp(-0.35, 0.35),
                };
                this.zoom_map_canvas(map_kind, anchor_x, anchor_y, delta);
                cx.notify();
            });
        }
    });
}

fn paint_route_arrow(
    window: &mut gpui::Window,
    from: gpui::Point<gpui::Pixels>,
    to: gpui::Point<gpui::Pixels>,
    color: gpui::Hsla,
) {
    let from_x = f32::from(from.x);
    let from_y = f32::from(from.y);
    let to_x = f32::from(to.x);
    let to_y = f32::from(to.y);
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    let length = (dx * dx + dy * dy).sqrt();
    if length < 28.0 {
        return;
    }

    let ux = dx / length;
    let uy = dy / length;
    let nx = -uy;
    let ny = ux;
    let arrow_length = (length * 0.22).clamp(12.0, 18.0);
    let arrow_half_width = (arrow_length * 0.42).clamp(5.0, 8.0);
    let tip_x = from_x + dx * 0.62;
    let tip_y = from_y + dy * 0.62;
    let base_x = tip_x - ux * arrow_length;
    let base_y = tip_y - uy * arrow_length;
    let points = [
        point(px(tip_x), px(tip_y)),
        point(
            px(base_x + nx * arrow_half_width),
            px(base_y + ny * arrow_half_width),
        ),
        point(
            px(base_x - nx * arrow_half_width),
            px(base_y - ny * arrow_half_width),
        ),
    ];

    let mut builder = PathBuilder::fill();
    builder.add_polygon(&points, false);
    if let Ok(path) = builder.build() {
        window.paint_path(path, color);
    }
}

fn format_segment_length(length: f32) -> String {
    let rounded = length.round();
    if (length - rounded).abs() < 0.05 {
        format!("{rounded:.0}")
    } else {
        format!("{length:.1}")
    }
}

fn paint_canvas_text_pill(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    center: gpui::Point<gpui::Pixels>,
    text: &str,
    background: gpui::Hsla,
    foreground: gpui::Hsla,
    border: gpui::Hsla,
    font_size: f32,
    horizontal_padding: f32,
    vertical_padding: f32,
) {
    if text.is_empty() {
        return;
    }

    let mut text_style = window.text_style();
    text_style.color = foreground;
    text_style.font_size = px(font_size).into();
    text_style.line_height = px(font_size + 2.0).into();
    text_style.font_weight = gpui::FontWeight::SEMIBOLD;

    let rendered_font_size = text_style.font_size.to_pixels(window.rem_size());
    let rendered_line_height = text_style.line_height_in_pixels(window.rem_size());
    let shaped_text: SharedString = text.to_owned().into();
    let shaped = window.text_system().shape_line(
        shaped_text,
        rendered_font_size,
        &[text_style.to_run(text.len())],
        None,
    );

    let pill_width = f32::from(shaped.width) + horizontal_padding * 2.0;
    let pill_height = f32::from(rendered_line_height) + vertical_padding * 2.0;
    let bounds = Bounds {
        origin: point(
            center.x - px(pill_width * 0.5),
            center.y - px(pill_height * 0.5),
        ),
        size: size(px(pill_width), px(pill_height)),
    };
    let frame_bounds = inflate_bounds(bounds, 1.0);
    let radius = px((pill_height * 0.5).min(12.0));

    window.paint_quad(fill(frame_bounds, border).corner_radii(radius + px(1.0)));
    window.paint_quad(fill(bounds, background).corner_radii(radius));
    let _ = shaped.paint(
        point(
            bounds.origin.x + px(horizontal_padding),
            bounds.origin.y + px(vertical_padding),
        ),
        rendered_line_height,
        window,
        cx,
    );
}

fn paint_route_segment_lengths(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    world_points: &[crate::domain::geometry::WorldPoint],
    canvas_points: &[gpui::Point<gpui::Pixels>],
    tokens: WorkbenchThemeTokens,
    accent: gpui::Hsla,
) {
    for (world_segment, canvas_segment) in world_points.windows(2).zip(canvas_points.windows(2)) {
        let [from_world, to_world] = [world_segment[0], world_segment[1]];
        let [from_canvas, to_canvas] = [canvas_segment[0], canvas_segment[1]];

        let from_x = f32::from(from_canvas.x);
        let from_y = f32::from(from_canvas.y);
        let to_x = f32::from(to_canvas.x);
        let to_y = f32::from(to_canvas.y);
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let screen_length = (dx * dx + dy * dy).sqrt();
        if screen_length < 56.0 {
            continue;
        }

        let world_dx = to_world.x - from_world.x;
        let world_dy = to_world.y - from_world.y;
        let world_length = (world_dx * world_dx + world_dy * world_dy).sqrt();
        let mut normal_x = -dy / screen_length;
        let mut normal_y = dx / screen_length;
        if normal_y > 0.0 {
            normal_x = -normal_x;
            normal_y = -normal_y;
        }

        paint_canvas_text_pill(
            window,
            cx,
            point(
                px((from_x + to_x) * 0.5 + normal_x * 14.0),
                px((from_y + to_y) * 0.5 + normal_y * 14.0),
            ),
            &format_segment_length(world_length),
            tokens.panel_bg.opacity(0.96),
            tokens.app_fg,
            accent.opacity(0.82),
            10.0,
            7.0,
            2.0,
        );
    }
}

fn paint_route_point_endpoint_label(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    marker_bounds: Bounds<gpui::Pixels>,
    is_start: bool,
    is_end: bool,
    tokens: WorkbenchThemeTokens,
) {
    let (label, color) = match (is_start, is_end) {
        (true, true) => ("起 / 终", gpui::rgb(0xB7791F).into()),
        (true, false) => ("起点", gpui::rgb(0x1F9D55).into()),
        (false, true) => ("终点", gpui::rgb(0xC24141).into()),
        (false, false) => return,
    };

    let center_x = marker_bounds.origin.x + px(f32::from(marker_bounds.size.width) * 0.5);
    let marker_top = f32::from(marker_bounds.origin.y);
    let marker_bottom = marker_top + f32::from(marker_bounds.size.height);
    let center_y = if marker_top >= 26.0 {
        px(marker_top - 10.0)
    } else {
        px(marker_bottom + 10.0)
    };

    paint_canvas_text_pill(
        window,
        cx,
        point(center_x, center_y),
        label,
        color,
        gpui::rgb(0xFFFFFF).into(),
        tokens.preview_ring,
        10.0,
        8.0,
        2.0,
    );
}

const MAP_INTERACTION_FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
const BWIKI_TYPE_BUTTON_WIDTH: f32 = 168.0;
const BWIKI_TYPE_BUTTON_HEIGHT: f32 = 44.0;
const BWIKI_TYPE_ICON_BOX_SIZE: f32 = 28.0;
const BWIKI_TYPE_NAME_WIDTH: f32 = 78.0;
const BWIKI_TYPE_COUNT_WIDTH: f32 = 38.0;
const BWIKI_MARKER_CULL_MARGIN: f32 = 64.0;

fn bwiki_type_icon_bounds(bounds: Bounds<gpui::Pixels>) -> Bounds<gpui::Pixels> {
    let available_width = (f32::from(bounds.size.width) - 8.0).max(1.0);
    let available_height = (f32::from(bounds.size.height) - 8.0).max(1.0);
    let scale = (available_width / BWIKI_MARKER_ICON_WIDTH)
        .min(available_height / BWIKI_MARKER_ICON_HEIGHT);
    let width = BWIKI_MARKER_ICON_WIDTH * scale;
    let height = BWIKI_MARKER_ICON_HEIGHT * scale;

    Bounds {
        origin: point(
            bounds.origin.x + px((f32::from(bounds.size.width) - width) * 0.5),
            bounds.origin.y + px((f32::from(bounds.size.height) - height) * 0.5),
        ),
        size: size(px(width), px(height)),
    }
}

fn paint_bwiki_style_marker(
    window: &mut gpui::Window,
    anchor: gpui::Point<gpui::Pixels>,
    highlighted: bool,
    tokens: WorkbenchThemeTokens,
    paint_contents: impl FnOnce(&mut gpui::Window, Bounds<gpui::Pixels>),
) {
    let image_bounds = bwiki_marker_image_bounds(anchor);
    if highlighted {
        let highlight_bounds = inflate_bounds(image_bounds, 4.0);
        window.paint_quad(
            fill(highlight_bounds, tokens.selected_marker_border)
                .corner_radii(bounds_corner_radius(highlight_bounds, 16.0)),
        );
    }
    paint_contents(window, image_bounds);
}

fn paint_route_marker(
    window: &mut gpui::Window,
    anchor: gpui::Point<gpui::Pixels>,
    size_px: f32,
    accent: u32,
    highlighted: bool,
    has_icon: bool,
    tokens: WorkbenchThemeTokens,
) {
    let size_px = size_px.clamp(14.0, 64.0);
    let radius = size_px * 0.34;
    let outer_bounds = Bounds {
        origin: point(anchor.x - px(radius + 4.0), anchor.y - px(radius + 4.0)),
        size: size(px((radius + 4.0) * 2.0), px((radius + 4.0) * 2.0)),
    };
    let inner_bounds = Bounds {
        origin: point(anchor.x - px(radius), anchor.y - px(radius)),
        size: size(px(radius * 2.0), px(radius * 2.0)),
    };
    let core_bounds = Bounds {
        origin: point(anchor.x - px(radius * 0.38), anchor.y - px(radius * 0.38)),
        size: size(px(radius * 0.76), px(radius * 0.76)),
    };

    if highlighted {
        window.paint_quad(
            fill(outer_bounds, tokens.selected_marker_border).corner_radii(px(radius + 4.0)),
        );
    } else if has_icon {
        window.paint_quad(
            fill(outer_bounds, gpui::rgba((accent << 8) | 0x44)).corner_radii(px(radius + 4.0)),
        );
    } else {
        window.paint_quad(
            fill(outer_bounds, gpui::rgba((accent << 8) | 0x55)).corner_radii(px(radius + 4.0)),
        );
    }
    if has_icon {
        return;
    }
    window.paint_quad(fill(inner_bounds, gpui::rgb(accent)).corner_radii(px(radius)));
    window.paint_quad(fill(core_bounds, gpui::rgb(0xFFFFFF)).corner_radii(px(radius * 0.38)));
}

fn route_marker_bounds(anchor: gpui::Point<gpui::Pixels>, size_px: f32) -> Bounds<gpui::Pixels> {
    let size_px = size_px.clamp(14.0, 64.0);
    let radius = size_px * 0.34 + 4.0;
    Bounds {
        origin: point(anchor.x - px(radius), anchor.y - px(radius)),
        size: size(px(radius * 2.0), px(radius * 2.0)),
    }
}

#[derive(Debug, Clone, Copy)]
struct RoutePointBadgeRenderItem {
    marker_bounds: Bounds<gpui::Pixels>,
    is_start: bool,
    is_end: bool,
}

fn paint_route_point_badges(
    window: &mut gpui::Window,
    marker_bounds: Bounds<gpui::Pixels>,
    is_start: bool,
    is_end: bool,
    tokens: WorkbenchThemeTokens,
) {
    const BADGE_SIZE: f32 = 12.0;
    const BADGE_MARGIN: f32 = 1.5;

    if !is_start && !is_end {
        return;
    }

    let top = marker_bounds.origin.y + px(BADGE_MARGIN);
    if is_start {
        paint_route_point_badge(
            window,
            Bounds {
                origin: point(marker_bounds.origin.x + px(BADGE_MARGIN), top),
                size: size(px(BADGE_SIZE), px(BADGE_SIZE)),
            },
            RoutePointBadgeKind::Start,
            tokens,
        );
    }
    if is_end {
        let right = marker_bounds.origin.x + px(f32::from(marker_bounds.size.width));
        paint_route_point_badge(
            window,
            Bounds {
                origin: point(right - px(BADGE_SIZE + BADGE_MARGIN), top),
                size: size(px(BADGE_SIZE), px(BADGE_SIZE)),
            },
            RoutePointBadgeKind::End,
            tokens,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutePointBadgeKind {
    Start,
    End,
}

fn paint_route_point_badge(
    window: &mut gpui::Window,
    bounds: Bounds<gpui::Pixels>,
    kind: RoutePointBadgeKind,
    tokens: WorkbenchThemeTokens,
) {
    let frame_bounds = inflate_bounds(bounds, 1.0);
    let frame_radius = match kind {
        RoutePointBadgeKind::Start => px(6.0),
        RoutePointBadgeKind::End => px(4.0),
    };
    let badge_color = match kind {
        RoutePointBadgeKind::Start => gpui::rgb(0x1F9D55),
        RoutePointBadgeKind::End => gpui::rgb(0xC24141),
    };

    window.paint_quad(fill(frame_bounds, tokens.preview_ring).corner_radii(frame_radius));
    window.paint_quad(fill(bounds, badge_color).corner_radii(frame_radius));

    match kind {
        RoutePointBadgeKind::Start => {
            let left = bounds.origin.x + px(3.0);
            let top = bounds.origin.y + px(2.4);
            let center_y = bounds.origin.y + px(f32::from(bounds.size.height) * 0.5);
            let bottom = bounds.origin.y + px(f32::from(bounds.size.height) - 2.4);
            let right = bounds.origin.x + px(f32::from(bounds.size.width) - 2.3);
            let points = [
                point(left, top),
                point(right, center_y),
                point(left, bottom),
            ];
            let mut builder = PathBuilder::fill();
            builder.add_polygon(&points, false);
            if let Ok(path) = builder.build() {
                window.paint_path(path, gpui::rgb(0xFFFFFF));
            }
        }
        RoutePointBadgeKind::End => {
            let stop_bounds = Bounds {
                origin: point(bounds.origin.x + px(3.0), bounds.origin.y + px(3.0)),
                size: size(
                    px(f32::from(bounds.size.width) - 6.0),
                    px(f32::from(bounds.size.height) - 6.0),
                ),
            };
            window.paint_quad(fill(stop_bounds, gpui::rgb(0xFFFFFF)).corner_radii(px(1.4)));
        }
    }
}

fn paint_bwiki_placeholder_marker(
    window: &mut gpui::Window,
    anchor: gpui::Point<gpui::Pixels>,
    mark_type: u32,
    tokens: WorkbenchThemeTokens,
) {
    let radius = 6.0 + (mark_type % 3) as f32;
    let outer = Bounds {
        origin: point(anchor.x - px(radius + 2.0), anchor.y - px(radius + 2.0)),
        size: size(px((radius + 2.0) * 2.0), px((radius + 2.0) * 2.0)),
    };
    let inner = Bounds {
        origin: point(anchor.x - px(radius), anchor.y - px(radius)),
        size: size(px(radius * 2.0), px(radius * 2.0)),
    };
    window.paint_quad(fill(outer, tokens.preview_ring).corner_radii(px(radius + 2.0)));
    window.paint_quad(fill(inner, tokens.preview_live).corner_radii(px(radius)));
}

fn resource_path(
    id: &'static str,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    value: &str,
) -> impl IntoElement {
    let copied_value = value.to_owned();

    div()
        .rounded_lg()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .child(
            div()
                .flex()
                .items_start()
                .justify_between()
                .gap_3()
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(div().text_xs().text_color(tokens.text_muted).child(label))
                        .child(
                            div()
                                .text_sm()
                                .line_height(px(20.0))
                                .text_color(tokens.app_fg)
                                .child(value.to_owned()),
                        ),
                )
                .child(toolbar_icon_button(
                    id,
                    tokens,
                    "⎘",
                    format!("复制{}", label),
                    ToolbarButtonTone::Neutral,
                    false,
                    cx.listener(move |this, _: &ClickEvent, _, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(copied_value.clone()));
                        this.status_text = format!("已复制{}：{}", label, copied_value).into();
                        cx.notify();
                    }),
                )),
        )
}

fn labeled_input(
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    input: &gpui::Entity<gpui_component::input::InputState>,
) -> impl IntoElement {
    div()
        .min_w(px(180.0))
        .flex_1()
        .flex()
        .flex_col()
        .gap_2()
        .child(field_label(tokens, label))
        .child(Input::new(input))
}

fn labeled_select<I>(
    tokens: WorkbenchThemeTokens,
    label: &'static str,
    select: Select<I>,
) -> impl IntoElement
where
    I: gpui_component::select::SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    div()
        .min_w(px(180.0))
        .flex_1()
        .flex()
        .flex_col()
        .gap_2()
        .child(field_label(tokens, label))
        .child(select)
}

fn field_label(tokens: WorkbenchThemeTokens, title: impl Into<SharedString>) -> impl IntoElement {
    div()
        .text_xs()
        .text_color(tokens.text_muted)
        .child(title.into())
}

fn section_title(title: impl Into<SharedString>) -> impl IntoElement {
    div()
        .text_lg()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .child(title.into())
}

fn body_text(tokens: WorkbenchThemeTokens, text: impl Into<SharedString>) -> impl IntoElement {
    div()
        .text_sm()
        .line_height(px(22.0))
        .text_color(tokens.text_soft)
        .child(text.into())
}
