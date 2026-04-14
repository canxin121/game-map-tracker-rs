use std::collections::BTreeMap;

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
    select::Select,
    tooltip::Tooltip,
};
use strum::IntoEnumIterator;

use crate::{
    domain::{
        geometry::WorldPoint, marker::MarkerIconStyle, theme::ThemePreference,
        tracker::TrackingSource,
    },
    resources::{
        BWIKI_WORLD_ZOOM, raw_coordinate_to_world, tile_coordinate_to_world_origin,
        zoom_world_bounds,
    },
    ui::map_canvas::{bounds_corner_radius, parse_hex_color, route_points, screen_points},
};

use super::{
    TrackerWorkbench,
    forms::read_input_value,
    page::{MapPage, MarkersPage, SettingsPage, WorkbenchPage},
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
    let map_is_active = this.active_page == WorkbenchPage::Map;
    let map_is_open = this.map_nav_expanded;
    let markers_is_active = this.active_page == WorkbenchPage::Markers;
    let markers_is_open = this.markers_nav_expanded;
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
                    "sidebar-nav-map",
                    tokens,
                    "地图",
                    map_is_active,
                    tokens.nav_item_active_bg,
                    Some(if map_is_open { "v" } else { ">" }),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_map_navigation();
                        cx.notify();
                    }),
                ))
                .when(map_is_open, |column| {
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
                                "sidebar-nav-map-tracker",
                                tokens,
                                "路线追踪",
                                map_is_active && this.map_page == MapPage::Tracker,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_map_page(MapPage::Tracker);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-map-bwiki",
                                tokens,
                                "BWiki 全图",
                                map_is_active && this.map_page == MapPage::Bwiki,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_map_page(MapPage::Bwiki);
                                    cx.notify();
                                }),
                            )),
                    )
                })
                .child(sidebar_nav_item(
                    "sidebar-nav-markers",
                    tokens,
                    "标记",
                    markers_is_active,
                    tokens.nav_item_active_bg,
                    Some(if markers_is_open { "v" } else { ">" }),
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_marker_navigation();
                        cx.notify();
                    }),
                ))
                .when(markers_is_open, |column| {
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
                                "sidebar-nav-markers-groups",
                                tokens,
                                "标记组",
                                markers_is_active && this.markers_page == MarkersPage::Groups,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_markers_page(MarkersPage::Groups);
                                    cx.notify();
                                }),
                            ))
                            .child(sidebar_nav_item(
                                "sidebar-nav-markers-points",
                                tokens,
                                "标记点",
                                markers_is_active && this.markers_page == MarkersPage::Points,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_markers_page(MarkersPage::Points);
                                    cx.notify();
                                }),
                            )),
                    )
                })
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
                                "sidebar-nav-settings-config",
                                tokens,
                                "配置",
                                settings_is_active && this.settings_page == SettingsPage::Config,
                                tokens.nav_subitem_active_bg,
                                None,
                                cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.select_settings_page(SettingsPage::Config);
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
        .rounded_lg()
        .bg(tokens.panel_sunken_bg)
        .border_1()
        .border_color(tokens.border)
        .p_3()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
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
                "确认删除标记组",
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
                "删除标记组",
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

fn status_chip(
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    value: impl Into<SharedString>,
) -> impl IntoElement {
    div()
        .rounded_lg()
        .bg(tokens.toolbar_chip_bg)
        .border_1()
        .border_color(tokens.border)
        .px_3()
        .py_2()
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
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
                                .text_color(tokens.text_soft)
                                .child(icon),
                        ),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(tokens.app_fg)
                        .child(value.into()),
                ),
        )
}

fn toolbar_button(
    id: impl Into<SharedString>,
    tokens: WorkbenchThemeTokens,
    icon: &'static str,
    label: impl Into<SharedString>,
    tone: ToolbarButtonTone,
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
        .cursor_pointer()
        .hover(move |style| style.bg(hover_background))
        .active(|style| style.opacity(0.92))
        .on_click(on_click)
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
                status_chip(tokens, "G", format!("当前组 {}", this.active_group_name()))
                    .into_any_element(),
                status_chip(
                    tokens,
                    "N",
                    format!("当前节点 {}", this.current_point_label()),
                )
                .into_any_element(),
                status_chip(
                    tokens,
                    "X",
                    format!("坐标 {}", this.current_position_label()),
                )
                .into_any_element(),
            ])
            .into_any_element(),
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
                    "tracker-toggle-top",
                    tokens,
                    "R",
                    if this.is_tracking_active() {
                        "停止追踪"
                    } else {
                        "启动追踪"
                    },
                    if this.is_tracking_active() {
                        ToolbarButtonTone::Danger
                    } else {
                        ToolbarButtonTone::Primary
                    },
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
                    "engine-cycle",
                    tokens,
                    "A",
                    this.selected_engine.to_string(),
                    ToolbarButtonTone::Neutral,
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
            let (total_types, total_points) = this
                .bwiki_resources
                .dataset_snapshot()
                .map(|dataset| (dataset.types.len(), dataset.total_point_count()))
                .unwrap_or_default();
            vec![
                toolbar_cluster(vec![
                    status_chip(tokens, "T", format!("类型 {}", total_types)).into_any_element(),
                    status_chip(
                        tokens,
                        "V",
                        format!("显示 {} 类", this.bwiki_visible_type_count()),
                    )
                    .into_any_element(),
                    status_chip(
                        tokens,
                        "P",
                        format!("点位 {}/{}", this.bwiki_visible_point_count(), total_points),
                    )
                    .into_any_element(),
                ])
                .into_any_element(),
                toolbar_cluster(vec![
                    toolbar_button(
                        "bwiki-show-all",
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
                    toolbar_button(
                        "bwiki-hide-all",
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
                    toolbar_button(
                        "bwiki-refresh",
                        tokens,
                        "R",
                        "刷新数据",
                        ToolbarButtonTone::Neutral,
                        cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.refresh_bwiki_dataset();
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    theme_button,
                ])
                .into_any_element(),
            ]
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
    match this.markers_page {
        MarkersPage::Groups => markers_groups_page(this, cx, tokens).into_any_element(),
        MarkersPage::Points => markers_points_page(this, cx, tokens).into_any_element(),
    }
}

fn markers_groups_page(
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
        .child(group_detail_panel(this, cx, tokens))
}

fn markers_points_page(
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
        .child(point_sidebar_panel(this, cx, tokens))
        .child(point_detail_panel(this, cx, tokens))
}

fn settings_page(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    match this.settings_page {
        SettingsPage::Config => settings_config_page(this, cx, tokens).into_any_element(),
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
        .gap_4()
        .overflow_y_scrollbar()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(section_title("标记组"))
        .child(paginated_list(
            "map-groups",
            cx,
            tokens,
            &this.map_group_list.search,
            Vec::new(),
            &this.map_group_list.page_input,
            pagination,
            "当前还没有可显示的标记组。",
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
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(section_title("标记组"))
        .child(paginated_list(
            "marker-groups",
            cx,
            tokens,
            &this.marker_group_list.search,
            vec![
                toolbar_icon_button(
                    "group-inline-new",
                    tokens,
                    "+",
                    "新建标记组",
                    ToolbarButtonTone::Neutral,
                    false,
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.create_group_inline_item(window, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            ],
            &this.marker_group_list.page_input,
            pagination,
            "当前还没有标记组，请先导入文件或新建分组。",
            group_rows,
            TrackerWorkbench::set_marker_group_page,
        ))
        .child(toolbar_cluster(vec![
            toolbar_button(
                "group-import-files",
                tokens,
                "I",
                "导入文件",
                ToolbarButtonTone::Neutral,
                cx.listener(|this, _: &ClickEvent, window, cx| {
                    this.import_route_files(window, cx);
                    cx.notify();
                }),
            )
            .into_any_element(),
            toolbar_button(
                "group-import-folder",
                tokens,
                "D",
                "导入文件夹",
                ToolbarButtonTone::Neutral,
                cx.listener(|this, _: &ClickEvent, window, cx| {
                    this.import_route_folder(window, cx);
                    cx.notify();
                }),
            )
            .into_any_element(),
        ]))
}

fn group_detail_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let group_icon_picker = ButtonGroup::new("group-icon-picker")
        .children(
            MarkerIconStyle::iter()
                .enumerate()
                .map(|(index, icon)| {
                    Button::new(("group-icon", index))
                        .label(icon.to_string())
                        .selected(this.group_icon == icon)
                })
                .collect::<Vec<_>>(),
        )
        .compact()
        .on_click(cx.listener(|this, indices: &Vec<usize>, _, cx| {
            if let Some(index) = indices.first().copied() {
                if let Some(icon) = MarkerIconStyle::iter().nth(index) {
                    this.group_icon = icon;
                    cx.notify();
                }
            }
        }));
    let active_group = this.active_group().cloned();

    div()
        .rounded_xl()
        .bg(tokens.panel_bg)
        .border_1()
        .border_color(tokens.border)
        .p_4()
        .child(section_title("所选标记组"))
        .when_some(active_group, |panel, group| {
            panel
                .child(
                    div()
                        .text_sm()
                        .text_color(tokens.text_muted)
                        .line_height(px(20.0))
                        .child(format!("文件 {}", group.metadata.file_name)),
                )
                .child(toolbar_cluster(vec![
                    toolbar_button(
                        "group-visible-toggle",
                        tokens,
                        if group.visible { "V" } else { "H" },
                        if group.visible {
                            "隐藏本组"
                        } else {
                            "显示本组"
                        },
                        if group.visible {
                            ToolbarButtonTone::Neutral
                        } else {
                            ToolbarButtonTone::Primary
                        },
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.toggle_selected_group_visible(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button(
                        "group-loop-toggle",
                        tokens,
                        if group.looped { "L" } else { "-" },
                        if group.looped {
                            "关闭闭环"
                        } else {
                            "开启闭环"
                        },
                        if group.looped {
                            ToolbarButtonTone::Primary
                        } else {
                            ToolbarButtonTone::Neutral
                        },
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.toggle_selected_group_looped(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button(
                        "group-save",
                        tokens,
                        "S",
                        "保存其他配置",
                        ToolbarButtonTone::Primary,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.save_group(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button(
                        "group-delete",
                        tokens,
                        "X",
                        "删除当前组",
                        ToolbarButtonTone::Danger,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.delete_selected_group(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ]))
                .child(
                    div().flex().gap_3().children([
                        labeled_input(tokens, "默认强调色", &this.group_form.color_hex)
                            .into_any_element(),
                        labeled_input(tokens, "默认图标尺寸", &this.group_form.size_px)
                            .into_any_element(),
                    ]),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(field_label(tokens, "默认图标"))
                        .child(group_icon_picker),
                )
        })
        .when(this.active_group().is_none(), |panel| {
            panel.child(empty_list_state(tokens, "请先从左侧创建或选择一个标记组。"))
        })
}

fn point_sidebar_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let point_query = normalized_query(&this.point_list.search, cx);
    let active_group = this.active_group();
    let filtered_points = active_group
        .map(|group| {
            group
                .points
                .iter()
                .filter(|point| {
                    matches_query(
                        &point_query,
                        [
                            point.display_label().to_owned(),
                            point.note.clone(),
                            format!("{:.0}", point.x),
                            format!("{:.0}", point.y),
                        ],
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (visible_points, pagination) = paginate_items(
        filtered_points,
        this.point_list.page,
        this.point_list.page_size,
    );
    let point_rows = visible_points
        .into_iter()
        .map(|point| {
            let point_id = point.id.clone();
            selectable_list_row(
                format!("point-{}", point.id),
                tokens,
                point.display_label().to_owned(),
                if point.note.trim().is_empty() {
                    format!("坐标 {:.0}, {:.0}", point.x, point.y)
                } else {
                    format!("坐标 {:.0}, {:.0} · {}", point.x, point.y, point.note)
                },
                this.selected_point_id.as_ref() == Some(&point.id),
                None,
                cx.listener(move |this, _: &ClickEvent, window, cx| {
                    this.select_point(point_id.clone(), window, cx);
                    cx.notify();
                }),
            )
            .into_any_element()
        })
        .collect::<Vec<_>>();

    div()
        .w(px(400.0))
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
        .child(section_title("标记点"))
        .child(field_label(tokens, "标记组"))
        .child(
            Select::new(&this.marker_group_picker)
                .w_full()
                .menu_width(px(360.0))
                .icon(IconName::Search)
                .placeholder("搜索并选择标记组")
                .disabled(this.route_groups.is_empty())
                .empty(empty_list_state(tokens, "当前还没有标记组。")),
        )
        .child(paginated_list(
            "group-points",
            cx,
            tokens,
            &this.point_list.search,
            vec![
                toolbar_icon_button(
                    "point-new-inline",
                    tokens,
                    "+",
                    "新建节点",
                    ToolbarButtonTone::Neutral,
                    active_group.is_none(),
                    cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.new_point_draft(window, cx);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            ],
            &this.point_list.page_input,
            pagination,
            if active_group.is_some() {
                "当前分组没有匹配的节点。"
            } else {
                "请先选择一个标记组。"
            },
            point_rows,
            TrackerWorkbench::set_point_page,
        ))
}

fn point_detail_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let marker_icon_picker = ButtonGroup::new("marker-icon-picker")
        .children(
            MarkerIconStyle::iter()
                .enumerate()
                .map(|(index, icon)| {
                    Button::new(("marker-icon", index))
                        .label(icon.to_string())
                        .selected(this.marker_icon == icon)
                })
                .collect::<Vec<_>>(),
        )
        .compact()
        .on_click(cx.listener(|this, indices: &Vec<usize>, _, cx| {
            if let Some(index) = indices.first().copied() {
                if let Some(icon) = MarkerIconStyle::iter().nth(index) {
                    this.marker_icon = icon;
                    cx.notify();
                }
            }
        }));

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
        .when_some(this.active_group(), |panel, group| {
            panel.child(
                div()
                    .text_sm()
                    .text_color(tokens.text_muted)
                    .line_height(px(20.0))
                    .child(format!("当前标记组 {}", group.display_name())),
            )
        })
        .when(this.active_group().is_none(), |panel| {
            panel.child(empty_list_state(tokens, "请先从左侧选择一个标记组。"))
        })
        .when(this.active_group().is_some(), |panel| {
            panel
                .child(labeled_input(tokens, "节点名称", &this.marker_form.label))
                .child(labeled_input(tokens, "备注", &this.marker_form.note))
                .child(div().flex().gap_3().children([
                    labeled_input(tokens, "X 坐标", &this.marker_form.x).into_any_element(),
                    labeled_input(tokens, "Y 坐标", &this.marker_form.y).into_any_element(),
                ]))
                .child(div().flex().gap_3().children([
                    labeled_input(tokens, "强调色", &this.marker_form.color_hex).into_any_element(),
                    labeled_input(tokens, "图标尺寸", &this.marker_form.size_px).into_any_element(),
                ]))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(field_label(tokens, "图标样式"))
                        .child(marker_icon_picker),
                )
                .child(toolbar_cluster(vec![
                    toolbar_button(
                        "point-use-preview",
                        tokens,
                        "P",
                        "使用当前位置",
                        ToolbarButtonTone::Neutral,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.use_preview_position_for_point(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button(
                        "point-use-center",
                        tokens,
                        "C",
                        "使用画布中心",
                        ToolbarButtonTone::Neutral,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.use_map_center_for_point(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_icon_button(
                        "point-focus",
                        tokens,
                        "◎",
                        "聚焦节点",
                        ToolbarButtonTone::Neutral,
                        false,
                        cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.focus_selected_point();
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ]))
                .child(toolbar_cluster(vec![
                    toolbar_button(
                        "point-save",
                        tokens,
                        "S",
                        "保存节点",
                        ToolbarButtonTone::Primary,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.save_point(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                    toolbar_button(
                        "point-delete",
                        tokens,
                        "-",
                        "删除节点",
                        ToolbarButtonTone::Danger,
                        cx.listener(|this, _: &ClickEvent, window, cx| {
                            this.delete_selected_point(window, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                ]))
        })
}

fn settings_config_page(
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

    let config = &this.workspace.config;
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
        .child(section_title("界面与追踪配置"))
        .child(
            div()
                .rounded_lg()
                .bg(tokens.panel_sunken_bg)
                .border_1()
                .border_color(tokens.border)
                .p_3()
                .child(field_label(tokens, "主题模式"))
                .child(theme_mode_picker),
        )
        .child(config_section(
            "截图区域",
            vec![
                format!("top = {}", config.minimap.top),
                format!("left = {}", config.minimap.left),
                format!("width = {}", config.minimap.width),
                format!("height = {}", config.minimap.height),
            ],
            tokens,
        ))
        .child(config_section(
            "模板追踪",
            vec![
                format!("refresh_rate_ms = {}", config.template.refresh_rate_ms),
                format!("local_downscale = {}", config.template.local_downscale),
                format!("global_downscale = {}", config.template.global_downscale),
                format!(
                    "local_match_threshold = {:.2}",
                    config.template.local_match_threshold
                ),
                format!(
                    "global_match_threshold = {:.2}",
                    config.template.global_match_threshold
                ),
            ],
            tokens,
        ))
        .child(config_section(
            "AI 追踪",
            vec![
                format!("refresh_rate_ms = {}", config.ai.refresh_rate_ms),
                format!(
                    "confidence_threshold = {:.2}",
                    config.ai.confidence_threshold
                ),
                format!("scan_size = {}", config.ai.scan_size),
                format!("scan_step = {}", config.ai.scan_step),
                format!("track_radius = {}", config.ai.track_radius),
            ],
            tokens,
        ))
        .child(config_section(
            "网络",
            vec![
                format!("http_port = {}", config.network.http_port),
                format!("websocket_port = {}", config.network.websocket_port),
            ],
            tokens,
        ))
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
        .child(section_title("追踪调试"))
        .child(body_text(
            tokens,
            snapshot.as_ref().map_or_else(
                || {
                    "启动 tracker 后，这里会显示 minimap、heatmap、refine 预览和状态字段。"
                        .to_owned()
                },
                |snapshot| {
                    format!(
                        "引擎 {}，阶段 {}，帧序号 {}。",
                        snapshot.engine, snapshot.stage_label, snapshot.frame_index
                    )
                },
            ),
        ))
        .child(
            div().flex().gap_3().flex_wrap().children(
                images
                    .into_iter()
                    .map(|image| debug_image_card(image, tokens))
                    .collect::<Vec<_>>(),
            ),
        )
        .child(
            div().flex().gap_3().flex_wrap().children(
                fields
                    .into_iter()
                    .map(|field| debug_field_card(field, tokens).into_any_element())
                    .collect::<Vec<_>>(),
            ),
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
        .child(section_title("本地数据路径"))
        .child(body_text(
            tokens,
            "路线文件、配置和 BWiki 运行时缓存都会真实落盘。地图瓦片、拼接底图和点位图标只会在首次需要时下载到缓存目录。",
        ))
        .child(resource_path(
            "resource-data-dir",
            cx,
            tokens,
            "数据目录",
            &this.project_root.to_string(),
        ))
        .child(resource_path(
            "resource-routes-dir",
            cx,
            tokens,
            "标记组目录",
            &this.workspace.assets.routes_dir.display().to_string(),
        ))
        .child(resource_path(
            "resource-bwiki-cache-dir",
            cx,
            tokens,
            "BWiki 缓存目录",
            &this.workspace.assets.bwiki_cache_dir.display().to_string(),
        ))
        .child(resource_path(
            "resource-models-dir",
            cx,
            tokens,
            "模型目录",
            &models_dir,
        ))
        .child(resource_path(
            "resource-config-file",
            cx,
            tokens,
            "配置文件",
            &this.workspace.assets.config_path.display().to_string(),
        ))
        .child(resource_path(
            "resource-ui-preferences",
            cx,
            tokens,
            "界面偏好",
            &this.ui_preferences_path.display().to_string(),
        ))
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

fn map_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let entity = cx.entity();
    let map_dimensions = this.workspace.report.map_dimensions;

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
                .child(section_title("路线地图"))
                .child(
                    div()
                        .text_xs()
                        .text_color(tokens.text_muted)
                        .child("滚轮缩放，左键拖拽；首次使用会懒加载并缓存 BWiki 拼接底图"),
                ),
        )
        .child(
            div().flex_1().overflow_hidden().child(
                canvas(
                    move |_, _, _| (),
                    move |bounds, _, window, cx| {
                        let bounds_width = f32::from(bounds.size.width);
                        let bounds_height = f32::from(bounds.size.height);
                        _ = entity.update(cx, |this, cx| {
                            this.tracker_map_view
                                .update_viewport(bounds_width, bounds_height);
                            let needs_fit = this.tracker_map_view.needs_fit;
                            let active_group = this.active_group().cloned();
                            this.tracker_map_view.fit_to_route_or_map(
                                active_group.as_ref(),
                                map_dimensions,
                                24.0,
                            );
                            let centered = this.tracker_map_view.apply_pending_center();
                            if needs_fit || centered {
                                cx.notify();
                            }
                        });

                        let (
                            camera,
                            active_group,
                            trail,
                            preview_position,
                            point_visuals,
                            selected_group_id,
                            selected_point_id,
                            logic_map_path,
                        ) = {
                            let this = entity.read(cx);
                            (
                                this.tracker_map_view.camera,
                                this.active_group().cloned(),
                                this.trail.clone(),
                                this.preview_position.clone(),
                                this.active_group_points(),
                                this.selected_group_id.clone(),
                                this.selected_point_id.clone(),
                                this.bwiki_resources
                                    .ensure_stitched_map_path(BWIKI_WORLD_ZOOM),
                            )
                        };

                        window.paint_layer(bounds, |window| {
                            window.with_content_mask(Some(ContentMask { bounds }), |window| {
                                window.paint_quad(fill(bounds, tokens.map_canvas_backdrop));

                                if let Some(path) = logic_map_path {
                                    let resource = gpui::Resource::from(path);
                                    let image_result =
                                        window.use_asset::<ImgResourceLoader>(&resource, cx);
                                    if let Some(Ok(image)) = image_result.as_ref() {
                                        let image_bounds = Bounds {
                                            origin: point(
                                                bounds.origin.x + px(camera.offset_x),
                                                bounds.origin.y + px(camera.offset_y),
                                            ),
                                            size: size(
                                                px(map_dimensions.width as f32 * camera.zoom),
                                                px(map_dimensions.height as f32 * camera.zoom),
                                            ),
                                        };

                                        let _ = window.paint_image(
                                            image_bounds,
                                            0.0.into(),
                                            image.clone(),
                                            0,
                                            false,
                                        );
                                    }
                                }
                            });
                        });

                        window.paint_layer(bounds, |window| {
                            window.with_content_mask(Some(ContentMask { bounds }), |window| {
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

                                if let Some(group) = active_group.as_ref() {
                                    let route_world = route_points(group);
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
                                        if group.looped {
                                            builder.line_to(route_canvas[0]);
                                        }
                                        if let Ok(path) = builder.build() {
                                            window.paint_path(path, tokens.route_path);
                                        }

                                        for segment in route_canvas.windows(2) {
                                            paint_route_arrow(
                                                window,
                                                segment[0],
                                                segment[1],
                                                tokens.route_path,
                                            );
                                        }
                                        if group.looped {
                                            paint_route_arrow(
                                                window,
                                                *route_canvas.last().unwrap_or(&route_canvas[0]),
                                                route_canvas[0],
                                                tokens.route_path,
                                            );
                                        }
                                    }
                                }

                                for marker in point_visuals {
                                    let screen = camera.world_to_screen(marker.world);
                                    let highlighted = selected_group_id.as_ref()
                                        == Some(&marker.group_id)
                                        && selected_point_id.as_ref() == Some(&marker.point_id);
                                    let anchor = point(
                                        bounds.origin.x + px(screen.x),
                                        bounds.origin.y + px(screen.y),
                                    );
                                    let accent = parse_hex_color(&marker.style.color_hex, 0x4ecdc4);
                                    paint_route_marker(
                                        window,
                                        anchor,
                                        marker.style.size_px,
                                        accent,
                                        highlighted,
                                        tokens,
                                    );
                                }

                                if let Some(position) =
                                    preview_position.as_ref().filter(|position| {
                                        position.source != TrackingSource::ManualPreview
                                    })
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
                                    window.paint_quad(
                                        fill(outer, tokens.preview_ring).corner_radii(px(9.0)),
                                    );
                                    window.paint_quad(
                                        fill(inner, marker_color).corner_radii(px(6.0)),
                                    );
                                }
                            });
                        });

                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseDownEvent, _, _, cx| {
                                if event.button != MouseButton::Left
                                    || !bounds.contains(&event.position)
                                {
                                    return;
                                }

                                _ = entity.update(cx, |this, _| {
                                    this.tracker_map_view.dragging_from =
                                        Some(crate::domain::geometry::WorldPoint::new(
                                            f32::from(event.position.x),
                                            f32::from(event.position.y),
                                        ));
                                    this.tracker_map_view.reset_interaction_redraw();
                                });
                            }
                        });
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseMoveEvent, _, _, cx| {
                                _ = entity.update(cx, |this, cx| {
                                    let Some(from) = this.tracker_map_view.dragging_from.take()
                                    else {
                                        return;
                                    };
                                    let dx = f32::from(event.position.x) - from.x;
                                    let dy = f32::from(event.position.y) - from.y;
                                    this.tracker_map_view.camera.offset_x += dx;
                                    this.tracker_map_view.camera.offset_y += dy;
                                    this.tracker_map_view.dragging_from =
                                        Some(crate::domain::geometry::WorldPoint::new(
                                            f32::from(event.position.x),
                                            f32::from(event.position.y),
                                        ));
                                    if this
                                        .tracker_map_view
                                        .should_redraw_interaction(MAP_INTERACTION_FRAME_INTERVAL)
                                    {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |_: &MouseUpEvent, _, _, cx| {
                                _ = entity.update(cx, |this, cx| {
                                    let dragged =
                                        this.tracker_map_view.dragging_from.take().is_some();
                                    this.tracker_map_view.reset_interaction_redraw();
                                    if dragged {
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

                                let delta = match event.delta {
                                    ScrollDelta::Pixels(delta) => {
                                        (f32::from(delta.y) / 320.0).clamp(-0.35, 0.35)
                                    }
                                    ScrollDelta::Lines(delta) => (delta.y / 8.0).clamp(-0.35, 0.35),
                                };
                                let anchor_x =
                                    f32::from(event.position.x) - f32::from(bounds.origin.x);
                                let anchor_y =
                                    f32::from(event.position.y) - f32::from(bounds.origin.y);
                                _ = entity.update(cx, |this, cx| {
                                    this.tracker_map_view.reset_interaction_redraw();
                                    this.tracker_map_view
                                        .camera
                                        .zoom_at(anchor_x, anchor_y, delta);
                                    cx.notify();
                                });
                            }
                        });
                    },
                )
                .size_full(),
            ),
        )
}

fn bwiki_types_sidebar(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let dataset = this.bwiki_resources.dataset_snapshot();
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
        .map(|(category_index, (category, definitions))| {
            let visible_type_count = definitions
                .iter()
                .filter(|(mark_type, _, _, _)| this.bwiki_visible_mark_types.contains(mark_type))
                .count();
            let total_point_count = definitions
                .iter()
                .map(|(_, _, _, count)| count)
                .sum::<usize>();
            let expanded = this.bwiki_expanded_categories.contains(&category);
            let category_for_expand = category.clone();
            let category_for_show = category.clone();
            let category_for_hide = category.clone();

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
                                            div().text_xs().text_color(tokens.text_muted).child(
                                                format!(
                                                    "{} / {} 类型可见 · {} 点位",
                                                    visible_type_count,
                                                    definitions.len(),
                                                    total_point_count
                                                ),
                                            ),
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
                                    this.set_bwiki_category_visibility(&category_for_show, true);
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
                                    this.set_bwiki_category_visibility(&category_for_hide, false);
                                    cx.notify();
                                }),
                            )
                            .into_any_element(),
                        ])),
                )
                .when(expanded, |card| {
                    let type_rows = definitions
                        .into_iter()
                        .map(|(mark_type, name, _icon_url, point_count)| {
                            let title = if point_count == 0 {
                                format!("{name} (空)")
                            } else {
                                name.clone()
                            };
                            let subtitle = format!("markType {mark_type} · {point_count} 点");
                            let visible = this.bwiki_visible_mark_types.contains(&mark_type);
                            let label = name.clone();
                            selectable_list_row(
                                format!("bwiki-type-{mark_type}"),
                                tokens,
                                title,
                                subtitle,
                                visible,
                                Some(if visible { "显示中" } else { "已隐藏" }.into()),
                                cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.toggle_bwiki_type_visibility(mark_type, &label);
                                    cx.notify();
                                }),
                            )
                            .into_any_element()
                        })
                        .collect::<Vec<_>>();
                    card.child(div().mt_3().flex().flex_col().gap_2().children(type_rows))
                })
                .into_any_element()
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
        .child(section_title("BWiki 类型过滤"))
        .child(body_text(
            tokens,
            "这里按 Wiki 原始分类列出全部类型。点击分类可展开，点击类型可切换地图显示；图标会在地图上首次需要时懒下载并缓存。",
        ))
        .when_some(last_error, |panel, error| {
            panel.child(config_section("最近一次缓存错误", vec![error], tokens))
        })
        .when(dataset.is_none(), |panel| {
            panel.child(config_section(
                "正在同步 BWiki 数据",
                vec![
                    "首次启动会请求点位目录与类型目录。".to_owned(),
                    "缓存准备好以后，这里会自动列出所有分类与类型。".to_owned(),
                ],
                tokens,
            ))
        })
        .when(!category_cards.is_empty(), |panel| panel.children(category_cards))
}

fn bwiki_map_panel(
    this: &TrackerWorkbench,
    cx: &mut Context<TrackerWorkbench>,
    tokens: WorkbenchThemeTokens,
) -> impl IntoElement {
    let entity = cx.entity();
    let map_dimensions = this.workspace.report.map_dimensions;

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
                .child(section_title("BWiki 大地图"))
                .child(
                    div()
                        .text_xs()
                        .text_color(tokens.text_muted)
                        .child("按视野懒加载瓦片与图标；滚轮缩放，左键拖拽"),
                ),
        )
        .child(
            div().flex_1().overflow_hidden().child(
                canvas(
                    move |_, _, _| (),
                    move |bounds, _, window, cx| {
                        let bounds_width = f32::from(bounds.size.width);
                        let bounds_height = f32::from(bounds.size.height);
                        _ = entity.update(cx, |this, cx| {
                            this.bwiki_map_view
                                .update_viewport(bounds_width, bounds_height);
                            let needs_fit = this.bwiki_map_view.needs_fit;
                            this.bwiki_map_view
                                .fit_to_route_or_map(None, map_dimensions, 24.0);
                            let centered = this.bwiki_map_view.apply_pending_center();
                            if needs_fit || centered {
                                cx.notify();
                            }
                        });

                        let (camera, dataset, bwiki_resources, visible_mark_types, last_error) = {
                            let this = entity.read(cx);
                            (
                                this.bwiki_map_view.camera,
                                this.bwiki_resources.dataset_snapshot(),
                                this.bwiki_resources.clone(),
                                this.bwiki_visible_mark_types.clone(),
                                this.bwiki_resources.last_error(),
                            )
                        };
                        let stitched_map_path =
                            bwiki_resources.ensure_stitched_map_path(BWIKI_WORLD_ZOOM);

                        window.paint_layer(bounds, |window| {
                            window.with_content_mask(Some(ContentMask { bounds }), |window| {
                                window.paint_quad(fill(bounds, tokens.map_canvas_backdrop));

                                let mut drew_stitched_map = false;
                                if let Some(path) = stitched_map_path.as_ref() {
                                    let resource = gpui::Resource::from(path.clone());
                                    let image_result =
                                        window.use_asset::<ImgResourceLoader>(&resource, cx);
                                    if let Some(Ok(image)) = image_result.as_ref() {
                                        let image_bounds = Bounds {
                                            origin: point(
                                                bounds.origin.x + px(camera.offset_x),
                                                bounds.origin.y + px(camera.offset_y),
                                            ),
                                            size: size(
                                                px(map_dimensions.width as f32 * camera.zoom),
                                                px(map_dimensions.height as f32 * camera.zoom),
                                            ),
                                        };
                                        let _ = window.paint_image(
                                            image_bounds,
                                            0.0.into(),
                                            image.clone(),
                                            0,
                                            false,
                                        );
                                        drew_stitched_map = true;
                                    }
                                }

                                let tile_zoom = preferred_bwiki_tile_zoom(camera.zoom);
                                if !drew_stitched_map
                                    && let Some(range) = zoom_world_bounds(tile_zoom)
                                {
                                    // BWiki/Leaflet keeps every zoom level anchored to the same
                                    // z8 tile grid. Coarser tiles extend past the visible content
                                    // bounds, so both tile lookup and placement must use that
                                    // shared world origin instead of a per-zoom local origin.
                                    let world_origin = raw_coordinate_to_world(0, 0);
                                    let world_tile_size = range.world_tile_size() as f32;
                                    let top_left =
                                        camera.screen_to_world(WorldPoint::new(0.0, 0.0));
                                    let bottom_right = camera.screen_to_world(WorldPoint::new(
                                        bounds_width,
                                        bounds_height,
                                    ));
                                    let min_world_x = top_left.x.min(bottom_right.x).max(0.0);
                                    let min_world_y = top_left.y.min(bottom_right.y).max(0.0);
                                    let max_world_x = top_left
                                        .x
                                        .max(bottom_right.x)
                                        .min(map_dimensions.width as f32);
                                    let max_world_y = top_left
                                        .y
                                        .max(bottom_right.y)
                                        .min(map_dimensions.height as f32);

                                    let start_col =
                                        (((min_world_x - world_origin.x) / world_tile_size).floor()
                                            as i32
                                            - 1)
                                        .clamp(range.min_x, range.max_x);
                                    let end_col =
                                        (((max_world_x - world_origin.x) / world_tile_size).ceil()
                                            as i32
                                            + 1)
                                        .clamp(range.min_x, range.max_x);
                                    let start_row =
                                        (((min_world_y - world_origin.y) / world_tile_size).floor()
                                            as i32
                                            - 1)
                                        .clamp(range.min_y, range.max_y);
                                    let end_row =
                                        (((max_world_y - world_origin.y) / world_tile_size).ceil()
                                            as i32
                                            + 1)
                                        .clamp(range.min_y, range.max_y);

                                    for tile_y in start_row..=end_row {
                                        for tile_x in start_col..=end_col {
                                            let tile_world = tile_coordinate_to_world_origin(
                                                tile_zoom, tile_x, tile_y,
                                            )
                                            .expect("validated BWiki tile origin");
                                            let tile_screen = camera.world_to_screen(tile_world);
                                            let tile_bounds = Bounds {
                                                origin: point(
                                                    bounds.origin.x + px(tile_screen.x),
                                                    bounds.origin.y + px(tile_screen.y),
                                                ),
                                                size: size(
                                                    px(world_tile_size * camera.zoom),
                                                    px(world_tile_size * camera.zoom),
                                                ),
                                            };
                                            window.paint_quad(
                                                fill(tile_bounds, tokens.panel_alt_bg)
                                                    .corner_radii(bounds_corner_radius(
                                                        tile_bounds,
                                                        10.0,
                                                    )),
                                            );
                                            if let Some(path) = bwiki_resources
                                                .ensure_tile_path(tile_zoom, tile_x, tile_y)
                                            {
                                                let resource = gpui::Resource::from(path);
                                                let image_result = window
                                                    .use_asset::<ImgResourceLoader>(&resource, cx);
                                                if let Some(Ok(image)) = image_result.as_ref() {
                                                    let _ = window.paint_image(
                                                        tile_bounds,
                                                        0.0.into(),
                                                        image.clone(),
                                                        0,
                                                        false,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            });
                        });

                        window.paint_layer(bounds, |window| {
                            window.with_content_mask(Some(ContentMask { bounds }), |window| {
                                if let Some(dataset) = dataset.as_ref() {
                                    let render_full_icons =
                                        camera.zoom >= BWIKI_ICON_RENDER_MIN_ZOOM;
                                    let visible_definitions = dataset
                                        .types
                                        .iter()
                                        .filter(|item| visible_mark_types.contains(&item.mark_type))
                                        .collect::<Vec<_>>();
                                    let icon_images = render_full_icons.then(|| {
                                        visible_definitions
                                            .iter()
                                            .map(|definition| {
                                                (
                                                    definition.mark_type,
                                                    bwiki_resources
                                                        .ensure_icon_path(
                                                            definition.mark_type,
                                                            &definition.icon_url,
                                                        )
                                                        .and_then(|path| {
                                                            let resource =
                                                                gpui::Resource::from(path);
                                                            window
                                                                .use_asset::<ImgResourceLoader>(
                                                                    &resource, cx,
                                                                )
                                                                .and_then(|result| result.ok())
                                                        }),
                                                )
                                            })
                                            .collect::<BTreeMap<_, _>>()
                                    });

                                    for definition in visible_definitions {
                                        let Some(points) =
                                            dataset.points_by_type.get(&definition.mark_type)
                                        else {
                                            continue;
                                        };

                                        for point_record in points {
                                            let screen = camera.world_to_screen(point_record.world);
                                            if screen.x < -BWIKI_MARKER_CULL_MARGIN
                                                || screen.y < -BWIKI_MARKER_CULL_MARGIN
                                                || screen.x
                                                    > bounds_width + BWIKI_MARKER_CULL_MARGIN
                                                || screen.y
                                                    > bounds_height + BWIKI_MARKER_CULL_MARGIN
                                            {
                                                continue;
                                            }

                                            let anchor = point(
                                                bounds.origin.x + px(screen.x),
                                                bounds.origin.y + px(screen.y),
                                            );
                                            if let Some(Some(image)) =
                                                icon_images.as_ref().and_then(|images| {
                                                    images.get(&definition.mark_type)
                                                })
                                            {
                                                let image_bounds =
                                                    bwiki_marker_image_bounds(anchor);
                                                let _ = window.paint_image(
                                                    image_bounds,
                                                    0.0.into(),
                                                    image.clone(),
                                                    0,
                                                    false,
                                                );
                                                continue;
                                            }

                                            paint_bwiki_placeholder_marker(
                                                window,
                                                anchor,
                                                definition.mark_type,
                                                tokens,
                                            );
                                        }
                                    }
                                }
                            });
                        });

                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseDownEvent, _, _, cx| {
                                if event.button != MouseButton::Left
                                    || !bounds.contains(&event.position)
                                {
                                    return;
                                }

                                _ = entity.update(cx, |this, _| {
                                    this.bwiki_map_view.dragging_from =
                                        Some(crate::domain::geometry::WorldPoint::new(
                                            f32::from(event.position.x),
                                            f32::from(event.position.y),
                                        ));
                                    this.bwiki_map_view.reset_interaction_redraw();
                                });
                            }
                        });
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseMoveEvent, _, _, cx| {
                                _ = entity.update(cx, |this, cx| {
                                    let Some(from) = this.bwiki_map_view.dragging_from.take()
                                    else {
                                        return;
                                    };
                                    let dx = f32::from(event.position.x) - from.x;
                                    let dy = f32::from(event.position.y) - from.y;
                                    this.bwiki_map_view.camera.pan_by(dx, dy);
                                    this.bwiki_map_view.dragging_from =
                                        Some(crate::domain::geometry::WorldPoint::new(
                                            f32::from(event.position.x),
                                            f32::from(event.position.y),
                                        ));
                                    if this
                                        .bwiki_map_view
                                        .should_redraw_interaction(MAP_INTERACTION_FRAME_INTERVAL)
                                    {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |_: &MouseUpEvent, _, _, cx| {
                                _ = entity.update(cx, |this, cx| {
                                    let dragged =
                                        this.bwiki_map_view.dragging_from.take().is_some();
                                    this.bwiki_map_view.reset_interaction_redraw();
                                    if dragged {
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

                                let delta = match event.delta {
                                    ScrollDelta::Pixels(delta) => {
                                        (f32::from(delta.y) / 320.0).clamp(-0.35, 0.35)
                                    }
                                    ScrollDelta::Lines(delta) => (delta.y / 8.0).clamp(-0.35, 0.35),
                                };
                                let anchor_x =
                                    f32::from(event.position.x) - f32::from(bounds.origin.x);
                                let anchor_y =
                                    f32::from(event.position.y) - f32::from(bounds.origin.y);
                                _ = entity.update(cx, |this, cx| {
                                    this.bwiki_map_view.reset_interaction_redraw();
                                    this.bwiki_map_view
                                        .camera
                                        .zoom_at(anchor_x, anchor_y, delta);
                                    cx.notify();
                                });
                            }
                        });

                        if dataset.is_none() || last_error.is_some() {
                            _ = window;
                        }
                    },
                )
                .size_full(),
            ),
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

fn preferred_bwiki_tile_zoom(camera_zoom: f32) -> u8 {
    if camera_zoom <= 0.10 {
        4
    } else if camera_zoom <= 0.18 {
        5
    } else if camera_zoom <= 0.35 {
        6
    } else if camera_zoom <= 0.75 {
        7
    } else {
        BWIKI_WORLD_ZOOM
    }
}

const MAP_INTERACTION_FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
const BWIKI_ICON_RENDER_MIN_ZOOM: f32 = 0.18;
const BWIKI_MARKER_ICON_WIDTH: f32 = 40.0;
const BWIKI_MARKER_ICON_HEIGHT: f32 = 50.0;
const BWIKI_MARKER_ICON_ANCHOR_X: f32 = 15.0;
const BWIKI_MARKER_ICON_ANCHOR_Y: f32 = 42.0;
const BWIKI_MARKER_CULL_MARGIN: f32 = 64.0;

fn bwiki_marker_image_bounds(anchor: gpui::Point<gpui::Pixels>) -> Bounds<gpui::Pixels> {
    Bounds {
        origin: point(
            anchor.x - px(BWIKI_MARKER_ICON_ANCHOR_X),
            anchor.y - px(BWIKI_MARKER_ICON_ANCHOR_Y),
        ),
        size: size(px(BWIKI_MARKER_ICON_WIDTH), px(BWIKI_MARKER_ICON_HEIGHT)),
    }
}

fn paint_route_marker(
    window: &mut gpui::Window,
    anchor: gpui::Point<gpui::Pixels>,
    size_px: f32,
    accent: u32,
    highlighted: bool,
    tokens: WorkbenchThemeTokens,
) {
    let radius = size_px.clamp(14.0, 64.0) * 0.34;
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
    } else {
        window.paint_quad(
            fill(outer_bounds, gpui::rgba((accent << 8) | 0x55)).corner_radii(px(radius + 4.0)),
        );
    }
    window.paint_quad(fill(inner_bounds, gpui::rgb(accent)).corner_radii(px(radius)));
    window.paint_quad(fill(core_bounds, gpui::rgb(0xFFFFFF)).corner_radii(px(radius * 0.38)));
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
        .flex_1()
        .flex()
        .flex_col()
        .gap_2()
        .child(field_label(tokens, label))
        .child(Input::new(input))
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
