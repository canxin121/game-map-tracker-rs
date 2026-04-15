use gpui::{
    AnyElement, App, AppContext, Context, IntoElement, ParentElement, SharedString, Styled, Window,
    div, prelude::FluentBuilder as _,
};
use gpui_component::{input::InputState, select::SelectItem};

use crate::{
    config::{
        AiTrackingConfig, AppConfig, CaptureRegion, LocalSearchConfig, NetworkConfig,
        SiftTrackingConfig, TemplateTrackingConfig,
    },
    domain::{
        geometry::WorldPoint,
        marker::{MarkerIconStyle, MarkerStyle, normalize_hex_color},
        route::{RouteId, RoutePointId},
    },
};

use super::TrackerWorkbench;

#[derive(Clone)]
pub(super) struct GroupFormInputs {
    pub(super) name: gpui::Entity<InputState>,
    pub(super) description: gpui::Entity<InputState>,
    pub(super) color_hex: gpui::Entity<InputState>,
    pub(super) size_px: gpui::Entity<InputState>,
}

impl GroupFormInputs {
    pub(super) fn new(window: &mut Window, cx: &mut Context<TrackerWorkbench>) -> Self {
        Self {
            name: cx.new(|cx| InputState::new(window, cx).placeholder("路线名称")),
            description: cx.new(|cx| InputState::new(window, cx).placeholder("路线说明")),
            color_hex: cx.new(|cx| InputState::new(window, cx).default_value("#FF6B6B")),
            size_px: cx.new(|cx| InputState::new(window, cx).default_value("24")),
        }
    }
}

#[derive(Clone)]
pub(super) struct GroupInlineEditInputs {
    pub(super) name: gpui::Entity<InputState>,
    pub(super) description: gpui::Entity<InputState>,
}

impl GroupInlineEditInputs {
    pub(super) fn new(window: &mut Window, cx: &mut Context<TrackerWorkbench>) -> Self {
        Self {
            name: cx.new(|cx| InputState::new(window, cx).placeholder("输入标题")),
            description: cx.new(|cx| InputState::new(window, cx).placeholder("输入注释")),
        }
    }
}

#[derive(Clone)]
pub(super) struct MarkerFormInputs {
    pub(super) label: gpui::Entity<InputState>,
    pub(super) note: gpui::Entity<InputState>,
    pub(super) x: gpui::Entity<InputState>,
    pub(super) y: gpui::Entity<InputState>,
    pub(super) color_hex: gpui::Entity<InputState>,
    pub(super) size_px: gpui::Entity<InputState>,
}

impl MarkerFormInputs {
    pub(super) fn new(window: &mut Window, cx: &mut Context<TrackerWorkbench>) -> Self {
        Self {
            label: cx.new(|cx| InputState::new(window, cx).placeholder("路线节点名称")),
            note: cx.new(|cx| InputState::new(window, cx).placeholder("备注")),
            x: cx.new(|cx| InputState::new(window, cx).placeholder("X")),
            y: cx.new(|cx| InputState::new(window, cx).placeholder("Y")),
            color_hex: cx.new(|cx| InputState::new(window, cx).default_value("#4ECDC4")),
            size_px: cx.new(|cx| InputState::new(window, cx).default_value("24")),
        }
    }
}

#[derive(Clone)]
pub(super) struct RoutePlannerFormInputs {
    pub(super) name: gpui::Entity<InputState>,
    pub(super) description: gpui::Entity<InputState>,
    pub(super) color_hex: gpui::Entity<InputState>,
}

impl RoutePlannerFormInputs {
    pub(super) fn new(window: &mut Window, cx: &mut Context<TrackerWorkbench>) -> Self {
        Self {
            name: cx.new(|cx| InputState::new(window, cx).placeholder("新路线名称")),
            description: cx.new(|cx| InputState::new(window, cx).placeholder("路线说明")),
            color_hex: cx.new(|cx| InputState::new(window, cx).default_value("#FF6B6B")),
        }
    }
}

#[derive(Clone)]
pub(super) struct ConfigFormInputs {
    pub(super) minimap_top: gpui::Entity<InputState>,
    pub(super) minimap_left: gpui::Entity<InputState>,
    pub(super) minimap_width: gpui::Entity<InputState>,
    pub(super) minimap_height: gpui::Entity<InputState>,
    pub(super) window_geometry: gpui::Entity<InputState>,
    pub(super) view_size: gpui::Entity<InputState>,
    pub(super) max_lost_frames: gpui::Entity<InputState>,
    pub(super) teleport_link_distance: gpui::Entity<InputState>,
    pub(super) local_search_enabled: gpui::Entity<InputState>,
    pub(super) local_search_radius_px: gpui::Entity<InputState>,
    pub(super) local_search_lock_fail_threshold: gpui::Entity<InputState>,
    pub(super) local_search_max_accepted_jump_px: gpui::Entity<InputState>,
    pub(super) sift_refresh_rate_ms: gpui::Entity<InputState>,
    pub(super) sift_clahe_limit: gpui::Entity<InputState>,
    pub(super) sift_match_ratio: gpui::Entity<InputState>,
    pub(super) sift_min_match_count: gpui::Entity<InputState>,
    pub(super) sift_ransac_threshold: gpui::Entity<InputState>,
    pub(super) ai_refresh_rate_ms: gpui::Entity<InputState>,
    pub(super) ai_confidence_threshold: gpui::Entity<InputState>,
    pub(super) ai_min_match_count: gpui::Entity<InputState>,
    pub(super) ai_ransac_threshold: gpui::Entity<InputState>,
    pub(super) ai_scan_size: gpui::Entity<InputState>,
    pub(super) ai_scan_step: gpui::Entity<InputState>,
    pub(super) ai_track_radius: gpui::Entity<InputState>,
    pub(super) ai_device: gpui::Entity<InputState>,
    pub(super) ai_device_index: gpui::Entity<InputState>,
    pub(super) ai_weights_path: gpui::Entity<InputState>,
    pub(super) template_refresh_rate_ms: gpui::Entity<InputState>,
    pub(super) template_local_downscale: gpui::Entity<InputState>,
    pub(super) template_global_downscale: gpui::Entity<InputState>,
    pub(super) template_global_refine_radius_px: gpui::Entity<InputState>,
    pub(super) template_local_match_threshold: gpui::Entity<InputState>,
    pub(super) template_global_match_threshold: gpui::Entity<InputState>,
    pub(super) template_mask_outer_radius: gpui::Entity<InputState>,
    pub(super) template_mask_inner_radius: gpui::Entity<InputState>,
    pub(super) template_device: gpui::Entity<InputState>,
    pub(super) template_device_index: gpui::Entity<InputState>,
    pub(super) network_http_port: gpui::Entity<InputState>,
    pub(super) network_websocket_port: gpui::Entity<InputState>,
}

impl ConfigFormInputs {
    pub(super) fn new(window: &mut Window, cx: &mut Context<TrackerWorkbench>) -> Self {
        Self {
            minimap_top: config_input(window, cx, "top"),
            minimap_left: config_input(window, cx, "left"),
            minimap_width: config_input(window, cx, "width"),
            minimap_height: config_input(window, cx, "height"),
            window_geometry: config_input(window, cx, "窗口几何"),
            view_size: config_input(window, cx, "view_size"),
            max_lost_frames: config_input(window, cx, "max_lost_frames"),
            teleport_link_distance: config_input(window, cx, "传送等效距离"),
            local_search_enabled: config_input(window, cx, "true / false"),
            local_search_radius_px: config_input(window, cx, "radius_px"),
            local_search_lock_fail_threshold: config_input(window, cx, "lock_fail_threshold"),
            local_search_max_accepted_jump_px: config_input(window, cx, "max_accepted_jump_px"),
            sift_refresh_rate_ms: config_input(window, cx, "refresh_rate_ms"),
            sift_clahe_limit: config_input(window, cx, "clahe_limit"),
            sift_match_ratio: config_input(window, cx, "match_ratio"),
            sift_min_match_count: config_input(window, cx, "min_match_count"),
            sift_ransac_threshold: config_input(window, cx, "ransac_threshold"),
            ai_refresh_rate_ms: config_input(window, cx, "refresh_rate_ms"),
            ai_confidence_threshold: config_input(window, cx, "confidence_threshold"),
            ai_min_match_count: config_input(window, cx, "min_match_count"),
            ai_ransac_threshold: config_input(window, cx, "ransac_threshold"),
            ai_scan_size: config_input(window, cx, "scan_size"),
            ai_scan_step: config_input(window, cx, "scan_step"),
            ai_track_radius: config_input(window, cx, "track_radius"),
            ai_device: config_input(window, cx, "cpu / cuda / metal"),
            ai_device_index: config_input(window, cx, "通常填 0"),
            ai_weights_path: config_input(window, cx, "相对项目根目录或绝对路径，可留空"),
            template_refresh_rate_ms: config_input(window, cx, "refresh_rate_ms"),
            template_local_downscale: config_input(window, cx, "local_downscale"),
            template_global_downscale: config_input(window, cx, "global_downscale"),
            template_global_refine_radius_px: config_input(window, cx, "global_refine_radius_px"),
            template_local_match_threshold: config_input(window, cx, "local_match_threshold"),
            template_global_match_threshold: config_input(window, cx, "global_match_threshold"),
            template_mask_outer_radius: config_input(window, cx, "mask_outer_radius"),
            template_mask_inner_radius: config_input(window, cx, "mask_inner_radius"),
            template_device: config_input(window, cx, "cpu / cuda / metal"),
            template_device_index: config_input(window, cx, "通常填 0"),
            network_http_port: config_input(window, cx, "http_port"),
            network_websocket_port: config_input(window, cx, "websocket_port"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MarkerGroupPickerItem {
    pub(super) id: RouteId,
    pub(super) title: SharedString,
    pub(super) subtitle: SharedString,
    pub(super) searchable_text: SharedString,
}

impl MarkerGroupPickerItem {
    pub(super) fn new(
        id: RouteId,
        title: impl Into<SharedString>,
        subtitle: impl Into<SharedString>,
        searchable_text: impl Into<SharedString>,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            subtitle: subtitle.into(),
            searchable_text: searchable_text.into(),
        }
    }
}

impl SelectItem for MarkerGroupPickerItem {
    type Value = RouteId;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn display_title(&self) -> Option<AnyElement> {
        let label = if self.subtitle.is_empty() {
            self.title.to_string()
        } else {
            format!("{} · {}", self.title, self.subtitle)
        };
        Some(
            div()
                .w_full()
                .min_w_0()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(label)
                .into_any_element(),
        )
    }

    fn value(&self) -> &Self::Value {
        &self.id
    }

    fn render(&self, _: &mut Window, _: &mut App) -> impl IntoElement {
        picker_menu_row(&self.title, &self.subtitle)
    }

    fn matches(&self, query: &str) -> bool {
        self.searchable_text
            .to_lowercase()
            .contains(&query.to_lowercase())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BwikiIconPickerItem {
    pub(super) value: MarkerIconStyle,
    pub(super) title: SharedString,
    pub(super) subtitle: SharedString,
    pub(super) searchable_text: SharedString,
}

impl BwikiIconPickerItem {
    pub(super) fn new(
        value: MarkerIconStyle,
        title: impl Into<SharedString>,
        subtitle: impl Into<SharedString>,
        searchable_text: impl Into<SharedString>,
    ) -> Self {
        Self {
            value,
            title: title.into(),
            subtitle: subtitle.into(),
            searchable_text: searchable_text.into(),
        }
    }
}

impl SelectItem for BwikiIconPickerItem {
    type Value = MarkerIconStyle;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn display_title(&self) -> Option<AnyElement> {
        let label = if self.subtitle.is_empty() {
            self.title.to_string()
        } else {
            format!("{} · {}", self.title, self.subtitle)
        };
        Some(
            div()
                .w_full()
                .min_w_0()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(label)
                .into_any_element(),
        )
    }

    fn value(&self) -> &Self::Value {
        &self.value
    }

    fn render(&self, _: &mut Window, _: &mut App) -> impl IntoElement {
        picker_menu_row(&self.title, &self.subtitle)
    }

    fn matches(&self, query: &str) -> bool {
        self.searchable_text
            .to_lowercase()
            .contains(&query.to_lowercase())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PointReorderTargetItem {
    pub(super) id: RoutePointId,
    pub(super) title: SharedString,
    pub(super) subtitle: SharedString,
    pub(super) searchable_text: SharedString,
}

impl PointReorderTargetItem {
    pub(super) fn new(
        id: RoutePointId,
        title: impl Into<SharedString>,
        subtitle: impl Into<SharedString>,
        searchable_text: impl Into<SharedString>,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            subtitle: subtitle.into(),
            searchable_text: searchable_text.into(),
        }
    }
}

impl SelectItem for PointReorderTargetItem {
    type Value = RoutePointId;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn display_title(&self) -> Option<AnyElement> {
        let label = if self.subtitle.is_empty() {
            self.title.to_string()
        } else {
            format!("{} · {}", self.title, self.subtitle)
        };
        Some(
            div()
                .w_full()
                .min_w_0()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(label)
                .into_any_element(),
        )
    }

    fn value(&self) -> &Self::Value {
        &self.id
    }

    fn render(&self, _: &mut Window, _: &mut App) -> impl IntoElement {
        picker_menu_row(&self.title, &self.subtitle)
    }

    fn matches(&self, query: &str) -> bool {
        self.searchable_text
            .to_lowercase()
            .contains(&query.to_lowercase())
    }
}

#[derive(Clone)]
pub(super) struct PagedListState {
    pub(super) search: gpui::Entity<InputState>,
    pub(super) page_input: gpui::Entity<InputState>,
    pub(super) page: usize,
    pub(super) page_size: usize,
}

impl PagedListState {
    pub(super) fn new(
        window: &mut Window,
        cx: &mut Context<TrackerWorkbench>,
        placeholder: impl Into<SharedString>,
        page_size: usize,
    ) -> Self {
        Self {
            search: cx.new(|cx| InputState::new(window, cx).placeholder(placeholder)),
            page_input: cx.new(|cx| InputState::new(window, cx).default_value("1")),
            page: 0,
            page_size,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct GroupDraft {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) style: MarkerStyle,
}

impl GroupDraft {
    pub(super) fn read(
        workbench: &TrackerWorkbench,
        cx: &mut Context<TrackerWorkbench>,
    ) -> Result<Self, String> {
        let name = read_input_value(&workbench.group_form.name, cx);
        if name.trim().is_empty() {
            return Err("路线名称不能为空。".to_owned());
        }

        let description = read_input_value(&workbench.group_form.description, cx);
        let size_px = read_input_value(&workbench.group_form.size_px, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "路线默认图标尺寸必须是数字。".to_owned())?;

        Ok(Self {
            name,
            description,
            style: MarkerStyle {
                icon: workbench.group_icon.clone(),
                color_hex: normalize_hex_color(&read_input_value(
                    &workbench.group_form.color_hex,
                    cx,
                )),
                size_px,
            }
            .normalized(),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct MarkerDraft {
    pub(super) label: String,
    pub(super) note: String,
    pub(super) world: WorldPoint,
    pub(super) style: MarkerStyle,
}

impl MarkerDraft {
    pub(super) fn read(
        workbench: &TrackerWorkbench,
        cx: &mut Context<TrackerWorkbench>,
    ) -> Result<Self, String> {
        if workbench.selected_group_id.is_none() {
            return Err("请先选择一条路线，再保存路线节点。".to_owned());
        }

        let x = read_input_value(&workbench.marker_form.x, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "路线节点 X 坐标必须是数字。".to_owned())?;
        let y = read_input_value(&workbench.marker_form.y, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "路线节点 Y 坐标必须是数字。".to_owned())?;
        let size_px = read_input_value(&workbench.marker_form.size_px, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "路线节点图标尺寸必须是数字。".to_owned())?;

        Ok(Self {
            label: read_input_value(&workbench.marker_form.label, cx),
            note: read_input_value(&workbench.marker_form.note, cx),
            world: WorldPoint::new(x, y),
            style: MarkerStyle {
                icon: workbench.marker_icon.clone(),
                color_hex: normalize_hex_color(&read_input_value(
                    &workbench.marker_form.color_hex,
                    cx,
                )),
                size_px,
            }
            .normalized(),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct PlannerRouteDraft {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) style: MarkerStyle,
}

impl PlannerRouteDraft {
    pub(super) fn read(
        workbench: &TrackerWorkbench,
        cx: &mut Context<TrackerWorkbench>,
    ) -> Result<Self, String> {
        let name = read_input_value(&workbench.bwiki_planner_form.name, cx);
        if name.trim().is_empty() {
            return Err("路线名称不能为空。".to_owned());
        }

        Ok(Self {
            name,
            description: read_input_value(&workbench.bwiki_planner_form.description, cx),
            style: MarkerStyle {
                icon: workbench.bwiki_planner_icon.clone(),
                color_hex: normalize_hex_color(&read_input_value(
                    &workbench.bwiki_planner_form.color_hex,
                    cx,
                )),
                size_px: 24.0,
            }
            .normalized(),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct ConfigDraft {
    pub(super) config: AppConfig,
}

impl ConfigDraft {
    pub(super) fn read(
        workbench: &TrackerWorkbench,
        cx: &mut Context<TrackerWorkbench>,
    ) -> Result<Self, String> {
        let form = &workbench.config_form;
        let weights_path = read_input_value(&form.ai_weights_path, cx);
        let weights_path = weights_path.trim();

        Ok(Self {
            config: AppConfig {
                minimap: CaptureRegion {
                    top: parse_input_value(&form.minimap_top, "minimap.top", cx)?,
                    left: parse_input_value(&form.minimap_left, "minimap.left", cx)?,
                    width: parse_input_value(&form.minimap_width, "minimap.width", cx)?,
                    height: parse_input_value(&form.minimap_height, "minimap.height", cx)?,
                },
                window_geometry: read_input_value(&form.window_geometry, cx),
                view_size: parse_input_value(&form.view_size, "view_size", cx)?,
                max_lost_frames: parse_input_value(&form.max_lost_frames, "max_lost_frames", cx)?,
                teleport_link_distance: parse_input_value(
                    &form.teleport_link_distance,
                    "teleport_link_distance",
                    cx,
                )?,
                local_search: LocalSearchConfig {
                    enabled: parse_bool_input_value(
                        &form.local_search_enabled,
                        "local_search.enabled",
                        cx,
                    )?,
                    radius_px: parse_input_value(
                        &form.local_search_radius_px,
                        "local_search.radius_px",
                        cx,
                    )?,
                    lock_fail_threshold: parse_input_value(
                        &form.local_search_lock_fail_threshold,
                        "local_search.lock_fail_threshold",
                        cx,
                    )?,
                    max_accepted_jump_px: parse_input_value(
                        &form.local_search_max_accepted_jump_px,
                        "local_search.max_accepted_jump_px",
                        cx,
                    )?,
                },
                sift: SiftTrackingConfig {
                    refresh_rate_ms: parse_input_value(
                        &form.sift_refresh_rate_ms,
                        "sift.refresh_rate_ms",
                        cx,
                    )?,
                    clahe_limit: parse_input_value(&form.sift_clahe_limit, "sift.clahe_limit", cx)?,
                    match_ratio: parse_input_value(&form.sift_match_ratio, "sift.match_ratio", cx)?,
                    min_match_count: parse_input_value(
                        &form.sift_min_match_count,
                        "sift.min_match_count",
                        cx,
                    )?,
                    ransac_threshold: parse_input_value(
                        &form.sift_ransac_threshold,
                        "sift.ransac_threshold",
                        cx,
                    )?,
                },
                ai: AiTrackingConfig {
                    refresh_rate_ms: parse_input_value(
                        &form.ai_refresh_rate_ms,
                        "ai.refresh_rate_ms",
                        cx,
                    )?,
                    confidence_threshold: parse_input_value(
                        &form.ai_confidence_threshold,
                        "ai.confidence_threshold",
                        cx,
                    )?,
                    min_match_count: parse_input_value(
                        &form.ai_min_match_count,
                        "ai.min_match_count",
                        cx,
                    )?,
                    ransac_threshold: parse_input_value(
                        &form.ai_ransac_threshold,
                        "ai.ransac_threshold",
                        cx,
                    )?,
                    scan_size: parse_input_value(&form.ai_scan_size, "ai.scan_size", cx)?,
                    scan_step: parse_input_value(&form.ai_scan_step, "ai.scan_step", cx)?,
                    track_radius: parse_input_value(&form.ai_track_radius, "ai.track_radius", cx)?,
                    device: parse_enum_input_value(&form.ai_device, "ai.device", cx)?,
                    device_index: parse_input_value(&form.ai_device_index, "ai.device_index", cx)?,
                    weights_path: (!weights_path.is_empty()).then(|| weights_path.to_owned()),
                },
                template: TemplateTrackingConfig {
                    refresh_rate_ms: parse_input_value(
                        &form.template_refresh_rate_ms,
                        "template.refresh_rate_ms",
                        cx,
                    )?,
                    local_downscale: parse_input_value(
                        &form.template_local_downscale,
                        "template.local_downscale",
                        cx,
                    )?,
                    global_downscale: parse_input_value(
                        &form.template_global_downscale,
                        "template.global_downscale",
                        cx,
                    )?,
                    global_refine_radius_px: parse_input_value(
                        &form.template_global_refine_radius_px,
                        "template.global_refine_radius_px",
                        cx,
                    )?,
                    local_match_threshold: parse_input_value(
                        &form.template_local_match_threshold,
                        "template.local_match_threshold",
                        cx,
                    )?,
                    global_match_threshold: parse_input_value(
                        &form.template_global_match_threshold,
                        "template.global_match_threshold",
                        cx,
                    )?,
                    mask_outer_radius: parse_input_value(
                        &form.template_mask_outer_radius,
                        "template.mask_outer_radius",
                        cx,
                    )?,
                    mask_inner_radius: parse_input_value(
                        &form.template_mask_inner_radius,
                        "template.mask_inner_radius",
                        cx,
                    )?,
                    device: parse_enum_input_value(&form.template_device, "template.device", cx)?,
                    device_index: parse_input_value(
                        &form.template_device_index,
                        "template.device_index",
                        cx,
                    )?,
                },
                network: NetworkConfig {
                    http_port: parse_input_value(&form.network_http_port, "network.http_port", cx)?,
                    websocket_port: parse_input_value(
                        &form.network_websocket_port,
                        "network.websocket_port",
                        cx,
                    )?,
                },
            },
        })
    }
}

pub(super) fn read_input_value(
    input: &gpui::Entity<InputState>,
    cx: &mut Context<TrackerWorkbench>,
) -> String {
    input.read(cx).value().to_string()
}

pub(super) fn set_input_value(
    input: &gpui::Entity<InputState>,
    value: impl Into<SharedString>,
    window: &mut Window,
    cx: &mut Context<TrackerWorkbench>,
) {
    let value = value.into();
    input.update(cx, |input, cx| {
        input.set_value(value.clone(), window, cx);
    });
}

fn config_input(
    window: &mut Window,
    cx: &mut Context<TrackerWorkbench>,
    placeholder: &'static str,
) -> gpui::Entity<InputState> {
    cx.new(|cx| InputState::new(window, cx).placeholder(placeholder))
}

fn parse_input_value<T>(
    input: &gpui::Entity<InputState>,
    field_name: &'static str,
    cx: &mut Context<TrackerWorkbench>,
) -> Result<T, String>
where
    T: std::str::FromStr,
{
    read_input_value(input, cx)
        .trim()
        .parse::<T>()
        .map_err(|_| format!("{field_name} 必须是有效数字。"))
}

fn parse_bool_input_value(
    input: &gpui::Entity<InputState>,
    field_name: &'static str,
    cx: &mut Context<TrackerWorkbench>,
) -> Result<bool, String> {
    let value = read_input_value(input, cx).trim().to_ascii_lowercase();
    match value.as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        _ => Err(format!("{field_name} 必须是 true 或 false。")),
    }
}

fn parse_enum_input_value<T>(
    input: &gpui::Entity<InputState>,
    field_name: &'static str,
    cx: &mut Context<TrackerWorkbench>,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: ToString,
{
    read_input_value(input, cx)
        .trim()
        .parse::<T>()
        .map_err(|error| format!("{field_name} 配置无效：{}", error.to_string()))
}

fn picker_menu_row(title: &SharedString, subtitle: &SharedString) -> impl IntoElement {
    div()
        .w_full()
        .min_w_0()
        .flex()
        .flex_col()
        .items_start()
        .gap_1()
        .child(
            div()
                .w_full()
                .min_w_0()
                .whitespace_normal()
                .line_height(gpui::px(18.0))
                .child(title.clone()),
        )
        .when(!subtitle.is_empty(), |column| {
            column.child(
                div()
                    .w_full()
                    .min_w_0()
                    .whitespace_normal()
                    .line_height(gpui::px(16.0))
                    .text_xs()
                    .opacity(0.72)
                    .child(subtitle.clone()),
            )
        })
}
