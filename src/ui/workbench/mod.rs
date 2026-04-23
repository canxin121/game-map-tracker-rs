mod capture_utils;
mod debug_images;
mod forms;
mod minimap_picker;
mod page;
mod panels;
mod picker_geometry;
mod picker_shared;
mod probe_region_picker;
mod probe_region_review;
mod select;
mod theme;
mod tracker_pip;

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use gpui::{
    AnyWindowHandle, App, AppContext, Bounds, Context, PathPromptOptions, Pixels, Render,
    RenderImage, SharedString, Subscription, Window, WindowBackgroundAppearance, WindowBounds,
    WindowHandle, WindowKind, WindowOptions, point, px, size,
};
use gpui_component::{Root, input::InputEvent};
use tracing::{debug, error, info, warn};

use crate::{
    config::{
        AiDevicePreference, AppConfig, CONFIG_FILE_NAME, CaptureRegion, TemplateInputMode,
        save_config,
    },
    domain::{
        geometry::WorldPoint,
        marker::{MarkerIconStyle, MarkerStyle},
        route::{RouteDocument, RouteId, RouteMetadata, RoutePoint, RoutePointId},
        theme::ThemePreference,
        tracker::{PositionEstimate, TrackerEngineKind, TrackerLifecycle, TrackingSource},
    },
    error::{ContextExt as _, Result},
    logging,
    resources::{
        AssetManifest, BwikiPointRecord, BwikiResourceManager, BwikiTypeDefinition,
        RouteImportReport, RouteRepository, UiPreferences, UiPreferencesRepository,
        WorkspaceLoadReport, WorkspaceSnapshot, default_map_dimensions,
    },
    tracking::{
        TrackerSession, TrackingEvent,
        ai::rebuild_convolution_engine_cache,
        burn_support::{available_burn_backend_preferences, available_burn_device_descriptors},
        debug::TrackingDebugSnapshot,
        presence::{
            MinimapPresenceModel, build_minimap_presence_probe_model,
            delete_minimap_presence_model, load_minimap_presence_model,
            save_minimap_presence_model,
        },
        spawn_tracker_session,
        template::rebuild_template_engine_cache,
    },
    ui::tile_cache::TileImageCache,
};

use self::{
    capture_utils::save_capture_region_png,
    debug_images::render_image_from_debug_image,
    forms::{
        BwikiIconPickerItem, ConfigDraft, ConfigFormInputs, DeviceIndexPickerItem,
        DevicePreferencePickerItem, GroupDraft, GroupFormInputs, GroupInlineEditInputs,
        MarkerDraft, MarkerFormInputs, MarkerGroupPickerItem, PagedListState, PlannerRouteDraft,
        PointReorderTargetItem, RoutePlannerFormInputs, TemplateInputModePickerItem,
        parse_input_value, read_input_value, set_input_value,
    },
    minimap_picker::{MinimapRegionPickResult, MinimapRegionPicker},
    page::{MapPage, SettingsPage, WorkbenchPage},
    panels::render_workbench,
    probe_region_picker::MinimapPresenceProbePicker,
    probe_region_review::MinimapPresenceProbeReviewWindow,
    select::{SelectEvent, SelectState},
    theme::apply_theme_preference,
    tracker_pip::{
        TrackerPipCapturePanelWindow, TrackerPipWindow, apply_window_bounds, apply_window_topmost,
    },
};

#[derive(Debug, Clone)]
pub(super) struct MapPointRenderItem {
    group_id: RouteId,
    point_id: RoutePointId,
    world: WorldPoint,
    style: MarkerStyle,
    is_start: bool,
    is_end: bool,
}

#[derive(Debug, Clone)]
pub(super) struct RouteSegmentRenderItem {
    pub(super) from: WorldPoint,
    pub(super) to: WorldPoint,
}

#[derive(Debug, Clone, Default)]
pub(super) struct TrackerMapRenderSnapshot {
    pub(super) route_color_hex: Option<String>,
    pub(super) trail: Vec<WorldPoint>,
    pub(super) preview_position: Option<PositionEstimate>,
    pub(super) route_segments: Vec<RouteSegmentRenderItem>,
    pub(super) point_visuals: Vec<MapPointRenderItem>,
    pub(super) selected_group_id: Option<RouteId>,
    pub(super) selected_point_id: Option<RoutePointId>,
    pub(super) selected_point_ids: HashSet<RoutePointId>,
    pub(super) route_editor_lasso_path: Option<Vec<WorldPoint>>,
    pub(super) follow_point: Option<WorldPoint>,
    pub(super) pip_always_on_top: bool,
    pub(super) pip_tracker_toggle_state: TrackerPipToggleState,
    pub(super) pip_tracker_status_tooltip: SharedString,
    pub(super) pip_probe_summary: SharedString,
    pub(super) pip_locate_summary: SharedString,
    pub(super) pip_test_case_capture_enabled: bool,
    pub(super) pip_capture_panel_expanded: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum TrackerPipToggleState {
    #[default]
    Start,
    Starting,
    Stop,
    Stopping,
    Restart,
}

const TRACKER_PIP_PROBE_IDLE_SUMMARY: &str = "未启动";
const TRACKER_PIP_LOCATE_IDLE_SUMMARY: &str = "等待首帧";

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct BwikiPointKey {
    mark_type: u32,
    raw_lat: i32,
    raw_lng: i32,
    id: String,
    title: String,
    layer: String,
    time: Option<u64>,
    version: Option<u32>,
}

impl BwikiPointKey {
    fn from_record(record: &BwikiPointRecord) -> Self {
        Self {
            mark_type: record.mark_type,
            raw_lat: record.raw_lat,
            raw_lng: record.raw_lng,
            id: record.id.clone(),
            title: record.title.clone(),
            layer: record.layer.clone(),
            time: record.time,
            version: record.version,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct BwikiPlannerResolvedPoint {
    pub(super) key: BwikiPointKey,
    pub(super) record: BwikiPointRecord,
    pub(super) type_definition: Option<BwikiTypeDefinition>,
}

#[derive(Debug, Clone)]
struct BwikiPlannerLassoSelection {
    path: Vec<WorldPoint>,
}

impl BwikiPlannerLassoSelection {
    fn new(anchor: WorldPoint) -> Self {
        Self { path: vec![anchor] }
    }

    fn push_point(&mut self, point: WorldPoint) -> bool {
        let Some(last) = self.path.last_mut() else {
            self.path.push(point);
            return true;
        };

        let dx = point.x - last.x;
        let dy = point.y - last.y;
        if dx.hypot(dy) >= 3.0 {
            self.path.push(point);
            true
        } else {
            *last = point;
            false
        }
    }

    fn travel_distance(&self) -> f32 {
        self.path
            .windows(2)
            .map(|segment| {
                let dx = segment[1].x - segment[0].x;
                let dy = segment[1].y - segment[0].y;
                dx.hypot(dy)
            })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RouteGraphEdge {
    from: RoutePointId,
    to: RoutePointId,
}

impl RouteGraphEdge {
    fn new(from: RoutePointId, to: RoutePointId) -> Self {
        Self { from, to }
    }
}

#[derive(Debug, Clone)]
struct RouteGraphEditState {
    group_id: RouteId,
    point_ids: HashSet<RoutePointId>,
    edges: HashSet<RouteGraphEdge>,
}

impl RouteGraphEditState {
    fn from_group(group: &RouteDocument) -> Self {
        Self {
            group_id: group.id.clone(),
            point_ids: route_point_id_set(&group.points),
            edges: route_graph_edges_from_points(&group.points),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RouteGraphRenderState {
    segments: Vec<RouteSegmentRenderItem>,
    start_ids: HashSet<RoutePointId>,
    end_ids: HashSet<RoutePointId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteGraphInsertOutcome {
    Unchanged,
    Added { replaced_edges: usize },
}

#[derive(Debug, Clone, Default)]
struct BwikiRoutePlanPreview {
    route_keys: Vec<BwikiPointKey>,
    total_cost: f32,
}

#[derive(Debug, Clone)]
struct BwikiPlannerTaskResult {
    requested_count: usize,
    normalized_selection_keys: HashSet<BwikiPointKey>,
    preview: Option<BwikiRoutePlanPreview>,
    failure_message: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct SelectedPointPopup {
    pub(super) left: f32,
    pub(super) top: f32,
    pub(super) width: f32,
    pub(super) height: f32,
    pub(super) route_name: String,
}

impl SelectedPointPopup {
    fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left
            && x <= self.left + self.width
            && y >= self.top
            && y <= self.top + self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PagedListKind {
    MapGroups,
    MarkerGroups,
    Points,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MapCanvasKind {
    Tracker,
    RouteEditor,
    Bwiki,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MapInteractionEnd {
    redraw: bool,
    clicked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointMoveTarget {
    Start,
    Prev,
    Next,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackerPendingAction {
    Starting,
    Stopping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrackerCacheKind {
    Convolution,
    Template,
}

impl TrackerCacheKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Convolution => "卷积特征匹配",
            Self::Template => "多尺度模板匹配",
        }
    }

    const fn idle_summary(self) -> &'static str {
        match self {
            Self::Convolution => "尚未手动重建卷积特征匹配缓存。",
            Self::Template => "尚未手动重建多尺度模板匹配缓存。",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsyncTaskPhase {
    Idle,
    Working,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestCaseLabel {
    HasMap,
    NoMap,
}

impl TestCaseLabel {
    pub(super) const fn display_name(self) -> &'static str {
        match self {
            Self::HasMap => "有图",
            Self::NoMap => "无图",
        }
    }

    const fn file_prefix(self) -> &'static str {
        match self {
            Self::HasMap => "has_map",
            Self::NoMap => "no_map",
        }
    }
}

#[derive(Debug, Clone)]
struct AsyncTaskStatus {
    phase: AsyncTaskPhase,
    summary: SharedString,
}

impl AsyncTaskStatus {
    fn idle(summary: impl Into<SharedString>) -> Self {
        Self {
            phase: AsyncTaskPhase::Idle,
            summary: summary.into(),
        }
    }

    fn working(summary: impl Into<SharedString>) -> Self {
        Self {
            phase: AsyncTaskPhase::Working,
            summary: summary.into(),
        }
    }

    fn succeeded(summary: impl Into<SharedString>) -> Self {
        Self {
            phase: AsyncTaskPhase::Succeeded,
            summary: summary.into(),
        }
    }

    fn failed(summary: impl Into<SharedString>) -> Self {
        Self {
            phase: AsyncTaskPhase::Failed,
            summary: summary.into(),
        }
    }
}

const BWIKI_TILE_CACHE_MAX_ITEMS: usize = 192;
const BWIKI_TILE_CACHE_MAX_BYTES: usize = 96 * 1024 * 1024;
const MAP_CLICK_DRAG_THRESHOLD: f32 = 4.0;
const BWIKI_PLANNER_EXACT_LIMIT: usize = 15;
const BWIKI_PLANNER_HIERARCHICAL_LIMIT: usize = 768;
const BWIKI_PLANNER_CLUSTER_TARGET_SIZE: usize = 96;
const BWIKI_PLANNER_MULTI_START_LIMIT: usize = 10;
const BWIKI_PLANNER_2OPT_PASS_LIMIT: usize = 4;
const BWIKI_PLANNER_2OPT_NEIGHBOR_LIMIT: usize = 24;
const BWIKI_PLANNER_FULL_2OPT_LIMIT: usize = 256;
const BWIKI_PLANNER_FULL_2OPT_PASS_LIMIT: usize = 2;
const BWIKI_PLANNER_SPATIAL_NEIGHBOR_WINDOW: usize = 6;
const BWIKI_PLANNER_HIERARCHICAL_2OPT_PASS_LIMIT: usize = 2;
const BWIKI_TELEPORT_MARK_TYPE: u32 = 202;

fn normalized_list_query(
    input: &gpui::Entity<gpui_component::input::InputState>,
    cx: &mut Context<TrackerWorkbench>,
) -> String {
    read_input_value(input, cx).trim().to_lowercase()
}

fn query_matches<I, S>(query: &str, values: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    query.is_empty()
        || values
            .into_iter()
            .any(|value| value.as_ref().to_lowercase().contains(query))
}

fn screen_bounds_contains(bounds: Bounds<Pixels>, x: f32, y: f32) -> bool {
    let left = f32::from(bounds.origin.x);
    let top = f32::from(bounds.origin.y);
    let right = left + f32::from(bounds.size.width);
    let bottom = top + f32::from(bounds.size.height);
    x >= left && x < right && y >= top && y < bottom
}

pub struct TrackerWorkbench {
    main_window_handle: AnyWindowHandle,
    pub(super) project_root: SharedString,
    pub(super) workspace: Arc<WorkspaceSnapshot>,
    pub(super) tracker_session: Option<TrackerSession>,
    pub(super) tracker_lifecycle: TrackerLifecycle,
    pub(super) selected_engine: TrackerEngineKind,
    pub(super) status_text: SharedString,
    pub(super) preview_position: Option<PositionEstimate>,
    pub(super) preview_cursor: Option<usize>,
    pub(super) trail: Vec<WorldPoint>,
    pub(super) tracker_map_view: crate::ui::map_canvas::MapViewportState,
    pub(super) route_editor_map_view: crate::ui::map_canvas::MapViewportState,
    pub(super) bwiki_map_view: crate::ui::map_canvas::MapViewportState,
    pub(super) frame_index: u64,
    pub(super) last_source: Option<TrackingSource>,
    pub(super) last_match_score: Option<f32>,
    pub(super) debug_snapshot: Option<TrackingDebugSnapshot>,
    pub(super) debug_snapshot_render_images: Vec<Option<Arc<RenderImage>>>,
    pub(super) tracker_pip_probe_summary: SharedString,
    pub(super) tracker_pip_locate_summary: SharedString,
    pub(super) route_groups: Vec<RouteDocument>,
    pub(super) selected_group_id: Option<RouteId>,
    pub(super) selected_point_id: Option<RoutePointId>,
    pub(super) group_icon: MarkerIconStyle,
    pub(super) marker_icon: MarkerIconStyle,
    pub(super) theme_preference: ThemePreference,
    pub(super) auto_focus_enabled: bool,
    pub(super) tracker_point_popup_enabled: bool,
    pub(super) debug_mode_enabled: bool,
    pub(super) test_case_capture_enabled: bool,
    tracker_pending_action: Option<TrackerPendingAction>,
    tracker_status_text: SharedString,
    route_import_status: AsyncTaskStatus,
    convolution_cache_status: AsyncTaskStatus,
    template_cache_status: AsyncTaskStatus,
    spinner_frame: usize,
    pub(super) map_point_insert_armed: bool,
    moving_point_id: Option<RoutePointId>,
    moving_point_preview: Option<WorldPoint>,
    route_editor_selected_point_ids: HashSet<RoutePointId>,
    route_editor_lasso_selection: Option<BwikiPlannerLassoSelection>,
    route_editor_graph_state: Option<RouteGraphEditState>,
    route_editor_draw_mode: bool,
    route_editor_draw_sequence: Vec<RoutePointId>,
    ignore_next_tracker_mouse_up: bool,
    suspend_group_autosave: bool,
    suspend_point_autosave: bool,
    pub(super) editing_group_id: Option<RouteId>,
    pub(super) pending_new_group_id: Option<RouteId>,
    pub(super) confirming_delete_group_id: Option<RouteId>,
    pub(super) confirming_delete_point_id: Option<RoutePointId>,
    active_page: WorkbenchPage,
    map_page: MapPage,
    settings_page: SettingsPage,
    settings_nav_expanded: bool,
    map_group_list: PagedListState,
    marker_group_list: PagedListState,
    point_list: PagedListState,
    bwiki_category_search: gpui::Entity<gpui_component::input::InputState>,
    bwiki_type_search: gpui::Entity<gpui_component::input::InputState>,
    bwiki_planner_active: bool,
    bwiki_planner_selected_points: HashSet<BwikiPointKey>,
    bwiki_planner_lasso_selection: Option<BwikiPlannerLassoSelection>,
    bwiki_planner_preview: Option<BwikiRoutePlanPreview>,
    bwiki_planner_status: AsyncTaskStatus,
    bwiki_planner_task_id: u64,
    bwiki_planner_form: RoutePlannerFormInputs,
    bwiki_planner_icon: MarkerIconStyle,
    bwiki_planner_icon_picker: gpui::Entity<SelectState<BwikiIconPickerItem>>,
    ai_device_preference: AiDevicePreference,
    ai_device_index: usize,
    ai_device_picker: gpui::Entity<SelectState<DevicePreferencePickerItem>>,
    ai_device_index_picker: gpui::Entity<SelectState<DeviceIndexPickerItem>>,
    minimap_presence_probe_device_preference: AiDevicePreference,
    minimap_presence_probe_device_index: usize,
    minimap_presence_probe_device_picker: gpui::Entity<SelectState<DevicePreferencePickerItem>>,
    minimap_presence_probe_device_index_picker: gpui::Entity<SelectState<DeviceIndexPickerItem>>,
    template_input_mode: TemplateInputMode,
    template_input_mode_picker: gpui::Entity<SelectState<TemplateInputModePickerItem>>,
    template_device_preference: AiDevicePreference,
    template_device_index: usize,
    template_device_picker: gpui::Entity<SelectState<DevicePreferencePickerItem>>,
    template_device_index_picker: gpui::Entity<SelectState<DeviceIndexPickerItem>>,
    marker_group_picker: gpui::Entity<SelectState<MarkerGroupPickerItem>>,
    group_icon_picker: gpui::Entity<SelectState<BwikiIconPickerItem>>,
    marker_icon_picker: gpui::Entity<SelectState<BwikiIconPickerItem>>,
    point_reorder_target_id: Option<RoutePointId>,
    point_reorder_picker: gpui::Entity<SelectState<PointReorderTargetItem>>,
    minimap_region_picker_window: Option<AnyWindowHandle>,
    minimap_presence_probe_picker_window: Option<AnyWindowHandle>,
    minimap_presence_probe_review_window: Option<AnyWindowHandle>,
    tracker_pip_window: Option<WindowHandle<Root>>,
    tracker_pip_capture_panel_window: Option<WindowHandle<Root>>,
    tracker_pip_window_bounds: Option<WindowBounds>,
    tracker_pip_always_on_top: bool,
    tracker_pip_pending_open: bool,
    debug_log_revision: u64,
    pub(super) ui_preferences_path: PathBuf,
    pub(super) bwiki_resources: BwikiResourceManager,
    pub(super) bwiki_tile_cache: gpui::Entity<TileImageCache>,
    pub(super) bwiki_version: u64,
    pub(super) bwiki_visible_mark_types: HashSet<u32>,
    pub(super) bwiki_expanded_categories: HashSet<String>,
    bwiki_visibility_initialized: bool,
    bwiki_icon_picker_version: u64,
    config_form: ConfigFormInputs,
    group_form: GroupFormInputs,
    group_inline_edit: GroupInlineEditInputs,
    marker_form: MarkerFormInputs,
    subscriptions: Vec<Subscription>,
}

pub(crate) fn init(cx: &mut gpui::App) {
    select::init(cx);
}

impl TrackerWorkbench {
    pub fn new(project_root: PathBuf, window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.on_release(|this, cx| {
            this.release_debug_snapshot_render_images_in_app(cx);
        })
        .detach();
        let config_form = ConfigFormInputs::new(window, cx);
        let group_form = GroupFormInputs::new(window, cx);
        let group_inline_edit = GroupInlineEditInputs::new(window, cx);
        let marker_form = MarkerFormInputs::new(window, cx);
        let bwiki_planner_form = RoutePlannerFormInputs::new(window, cx);
        let map_group_list = PagedListState::new(window, cx, "搜索地图中的路线", 8);
        let marker_group_list = PagedListState::new(window, cx, "搜索路线", 8);
        let point_list = PagedListState::new(window, cx, "搜索当前路线节点", 10);
        let bwiki_category_search =
            cx.new(|cx| gpui_component::input::InputState::new(window, cx).placeholder("过滤分类"));
        let bwiki_type_search =
            cx.new(|cx| gpui_component::input::InputState::new(window, cx).placeholder("过滤节点"));
        let marker_group_picker = cx.new(|cx| SelectState::new(Vec::new(), None, 8, window, cx));
        let group_icon_picker = cx.new(|cx| SelectState::new(Vec::new(), None, 10, window, cx));
        let marker_icon_picker = cx.new(|cx| SelectState::new(Vec::new(), None, 10, window, cx));
        let bwiki_planner_icon_picker =
            cx.new(|cx| SelectState::new(Vec::new(), None, 10, window, cx));
        let ai_device_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_preference_picker_items(),
                Some(AiDevicePreference::Cpu),
                6,
                window,
                cx,
            )
        });
        let ai_device_index_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_index_picker_items(AiDevicePreference::Cpu),
                Some(0),
                6,
                window,
                cx,
            )
        });
        let minimap_presence_probe_device_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_preference_picker_items(),
                Some(AiDevicePreference::Cpu),
                6,
                window,
                cx,
            )
        });
        let minimap_presence_probe_device_index_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_index_picker_items(AiDevicePreference::Cpu),
                Some(0),
                6,
                window,
                cx,
            )
        });
        let template_device_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_preference_picker_items(),
                Some(AiDevicePreference::Cpu),
                6,
                window,
                cx,
            )
        });
        let template_input_mode_picker = cx.new(|cx| {
            SelectState::new(
                Self::template_input_mode_picker_items(),
                Some(TemplateInputMode::Color),
                6,
                window,
                cx,
            )
        });
        let template_device_index_picker = cx.new(|cx| {
            SelectState::new(
                Self::device_index_picker_items(AiDevicePreference::Cpu),
                Some(0),
                6,
                window,
                cx,
            )
        });
        let point_reorder_picker = cx.new(|cx| SelectState::new(Vec::new(), None, 8, window, cx));
        let bwiki_tile_cache =
            TileImageCache::new(BWIKI_TILE_CACHE_MAX_ITEMS, BWIKI_TILE_CACHE_MAX_BYTES, cx);
        let ui_preferences_path = UiPreferencesRepository::path_for(&project_root);
        let (
            theme_preference,
            auto_focus_enabled,
            tracker_point_popup_enabled,
            debug_mode_enabled,
            test_case_capture_enabled,
            preferences_error,
        ) = match UiPreferencesRepository::load(&project_root) {
            Ok(preferences) => (
                preferences.theme_mode,
                preferences.auto_focus_enabled,
                preferences.tracker_point_popup_enabled,
                preferences.debug_mode_enabled,
                preferences.test_case_capture_enabled,
                None,
            ),
            Err(error) => (
                ThemePreference::default(),
                true,
                true,
                false,
                false,
                Some(format!("载入界面偏好失败：{error:#}")),
            ),
        };
        info!(
            project_root = %project_root.display(),
            theme = %theme_preference,
            auto_focus_enabled,
            tracker_point_popup_enabled,
            debug_mode_enabled,
            test_case_capture_enabled,
            "initialized UI preferences for workbench"
        );

        let mut workbench = match WorkspaceSnapshot::load(project_root.clone()) {
            Ok(workspace) => {
                info!(
                    project_root = %project_root.display(),
                    group_count = workspace.report.group_count,
                    point_count = workspace.report.point_count,
                    "workbench loaded workspace snapshot"
                );
                let workspace = Arc::new(workspace);
                let (bwiki_resources, bwiki_manager_error) =
                    Self::new_bwiki_resource_manager(workspace.assets.bwiki_cache_dir.clone());
                let bwiki_version = bwiki_resources.version();
                let route_groups = workspace.groups.clone();
                let selected_group_id = route_groups.first().map(|group| group.id.clone());
                let preview_position = selected_group_id
                    .as_ref()
                    .and_then(|group_id| route_groups.iter().find(|group| &group.id == group_id))
                    .and_then(RouteDocument::first_point)
                    .map(PositionEstimate::manual);

                Self {
                    main_window_handle: window.window_handle(),
                    project_root: project_root.to_string_lossy().into_owned().into(),
                    workspace,
                    tracker_session: None,
                    tracker_lifecycle: TrackerLifecycle::Idle,
                    selected_engine: TrackerEngineKind::MultiScaleTemplateMatch,
                    status_text: "数据目录已经完成解析。路线追踪页负责查看地图和运行 tracker，路线管理页负责在地图上管理 routes 目录下的单线路线。".into(),
                    preview_position: preview_position.clone(),
                    preview_cursor: selected_group_id.as_ref().map(|_| 0),
                    trail: preview_position
                        .as_ref()
                        .map(|position| vec![position.world])
                        .unwrap_or_default(),
                    tracker_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    route_editor_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    bwiki_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    frame_index: 0,
                    last_source: None,
                    last_match_score: None,
                    debug_snapshot: None,
                    debug_snapshot_render_images: Vec::new(),
                    tracker_pip_probe_summary: TRACKER_PIP_PROBE_IDLE_SUMMARY.into(),
                    tracker_pip_locate_summary: TRACKER_PIP_LOCATE_IDLE_SUMMARY.into(),
                    route_groups,
                    selected_group_id,
                    selected_point_id: None,
                    group_icon: MarkerIconStyle::default(),
                    marker_icon: MarkerIconStyle::default(),
                    theme_preference,
                    auto_focus_enabled,
                    tracker_point_popup_enabled,
                    debug_mode_enabled,
                    test_case_capture_enabled,
                    tracker_pending_action: None,
                    tracker_status_text: "追踪未启动。".into(),
                    route_import_status: AsyncTaskStatus::idle("尚未导入路线文件。"),
                    convolution_cache_status: AsyncTaskStatus::idle(
                        TrackerCacheKind::Convolution.idle_summary(),
                    ),
                    template_cache_status: AsyncTaskStatus::idle(
                        TrackerCacheKind::Template.idle_summary(),
                    ),
                    spinner_frame: 0,
                    map_point_insert_armed: false,
                    moving_point_id: None,
                    moving_point_preview: None,
                    route_editor_selected_point_ids: HashSet::new(),
                    route_editor_lasso_selection: None,
                    route_editor_graph_state: None,
                    route_editor_draw_mode: false,
                    route_editor_draw_sequence: Vec::new(),
                    ignore_next_tracker_mouse_up: false,
                    suspend_group_autosave: false,
                    suspend_point_autosave: false,
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    confirming_delete_point_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Bwiki,
                    settings_page: SettingsPage::default(),
                    settings_nav_expanded: false,
                    map_group_list: map_group_list.clone(),
                    marker_group_list: marker_group_list.clone(),
                    point_list: point_list.clone(),
                    bwiki_category_search: bwiki_category_search.clone(),
                    bwiki_type_search: bwiki_type_search.clone(),
                    bwiki_planner_active: false,
                    bwiki_planner_selected_points: HashSet::new(),
                    bwiki_planner_lasso_selection: None,
                    bwiki_planner_preview: None,
                    bwiki_planner_status: AsyncTaskStatus::idle("选择节点后可在后台规划路线。"),
                    bwiki_planner_task_id: 0,
                    bwiki_planner_form: bwiki_planner_form.clone(),
                    bwiki_planner_icon: MarkerIconStyle::default(),
                    bwiki_planner_icon_picker: bwiki_planner_icon_picker.clone(),
                    ai_device_preference: AiDevicePreference::Cpu,
                    ai_device_index: 0,
                    ai_device_picker: ai_device_picker.clone(),
                    ai_device_index_picker: ai_device_index_picker.clone(),
                    minimap_presence_probe_device_preference: AiDevicePreference::Cpu,
                    minimap_presence_probe_device_index: 0,
                    minimap_presence_probe_device_picker: minimap_presence_probe_device_picker
                        .clone(),
                    minimap_presence_probe_device_index_picker:
                        minimap_presence_probe_device_index_picker.clone(),
                    template_input_mode: TemplateInputMode::Color,
                    template_input_mode_picker: template_input_mode_picker.clone(),
                    template_device_preference: AiDevicePreference::Cpu,
                    template_device_index: 0,
                    template_device_picker: template_device_picker.clone(),
                    template_device_index_picker: template_device_index_picker.clone(),
                    marker_group_picker: marker_group_picker.clone(),
                    group_icon_picker: group_icon_picker.clone(),
                    marker_icon_picker: marker_icon_picker.clone(),
                    point_reorder_target_id: None,
                    point_reorder_picker: point_reorder_picker.clone(),
                    minimap_region_picker_window: None,
                    minimap_presence_probe_picker_window: None,
                    minimap_presence_probe_review_window: None,
                    tracker_pip_window: None,
                    tracker_pip_capture_panel_window: None,
                    tracker_pip_window_bounds: None,
                    tracker_pip_always_on_top: false,
                    tracker_pip_pending_open: false,
                    debug_log_revision: logging::debug_log_revision(),
                    ui_preferences_path: ui_preferences_path.clone(),
                    bwiki_resources,
                    bwiki_tile_cache: bwiki_tile_cache.clone(),
                    bwiki_version,
                    bwiki_visible_mark_types: HashSet::new(),
                    bwiki_expanded_categories: HashSet::new(),
                    bwiki_visibility_initialized: false,
                    bwiki_icon_picker_version: 0,
                    config_form: config_form.clone(),
                    group_form,
                    group_inline_edit,
                    marker_form,
                    subscriptions: Vec::new(),
                }
                .with_optional_status_suffix(bwiki_manager_error)
            }
            Err(error) => {
                error!(
                    project_root = %project_root.display(),
                    error = %error,
                    "workbench failed to load workspace snapshot"
                );
                let workspace = Arc::new(Self::empty_workspace(project_root.clone()));
                let (bwiki_resources, bwiki_manager_error) =
                    Self::new_bwiki_resource_manager(workspace.assets.bwiki_cache_dir.clone());
                let bwiki_version = bwiki_resources.version();
                Self {
                    main_window_handle: window.window_handle(),
                    project_root: project_root.to_string_lossy().into_owned().into(),
                    workspace,
                    tracker_session: None,
                    tracker_lifecycle: TrackerLifecycle::Failed,
                    selected_engine: TrackerEngineKind::MultiScaleTemplateMatch,
                    status_text: format!("载入数据目录失败：{error:#}").into(),
                    preview_position: None,
                    preview_cursor: None,
                    trail: Vec::new(),
                    tracker_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    route_editor_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    bwiki_map_view: crate::ui::map_canvas::MapViewportState {
                        needs_fit: true,
                        ..Default::default()
                    },
                    frame_index: 0,
                    last_source: None,
                    last_match_score: None,
                    debug_snapshot: None,
                    debug_snapshot_render_images: Vec::new(),
                    tracker_pip_probe_summary: TRACKER_PIP_PROBE_IDLE_SUMMARY.into(),
                    tracker_pip_locate_summary: TRACKER_PIP_LOCATE_IDLE_SUMMARY.into(),
                    route_groups: Vec::new(),
                    selected_group_id: None,
                    selected_point_id: None,
                    group_icon: MarkerIconStyle::default(),
                    marker_icon: MarkerIconStyle::default(),
                    theme_preference,
                    auto_focus_enabled,
                    tracker_point_popup_enabled,
                    debug_mode_enabled,
                    test_case_capture_enabled,
                    tracker_pending_action: None,
                    tracker_status_text: "追踪未启动。".into(),
                    route_import_status: AsyncTaskStatus::idle("尚未导入路线文件。"),
                    convolution_cache_status: AsyncTaskStatus::idle(
                        TrackerCacheKind::Convolution.idle_summary(),
                    ),
                    template_cache_status: AsyncTaskStatus::idle(
                        TrackerCacheKind::Template.idle_summary(),
                    ),
                    spinner_frame: 0,
                    map_point_insert_armed: false,
                    moving_point_id: None,
                    moving_point_preview: None,
                    route_editor_selected_point_ids: HashSet::new(),
                    route_editor_lasso_selection: None,
                    route_editor_graph_state: None,
                    route_editor_draw_mode: false,
                    route_editor_draw_sequence: Vec::new(),
                    ignore_next_tracker_mouse_up: false,
                    suspend_group_autosave: false,
                    suspend_point_autosave: false,
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    confirming_delete_point_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Bwiki,
                    settings_page: SettingsPage::default(),
                    settings_nav_expanded: false,
                    map_group_list,
                    marker_group_list,
                    point_list,
                    bwiki_category_search,
                    bwiki_type_search,
                    bwiki_planner_active: false,
                    bwiki_planner_selected_points: HashSet::new(),
                    bwiki_planner_lasso_selection: None,
                    bwiki_planner_preview: None,
                    bwiki_planner_status: AsyncTaskStatus::idle("选择节点后可在后台规划路线。"),
                    bwiki_planner_task_id: 0,
                    bwiki_planner_form,
                    bwiki_planner_icon: MarkerIconStyle::default(),
                    bwiki_planner_icon_picker,
                    ai_device_preference: AiDevicePreference::Cpu,
                    ai_device_index: 0,
                    ai_device_picker,
                    ai_device_index_picker,
                    minimap_presence_probe_device_preference: AiDevicePreference::Cpu,
                    minimap_presence_probe_device_index: 0,
                    minimap_presence_probe_device_picker,
                    minimap_presence_probe_device_index_picker,
                    template_input_mode: TemplateInputMode::Color,
                    template_input_mode_picker,
                    template_device_preference: AiDevicePreference::Cpu,
                    template_device_index: 0,
                    template_device_picker,
                    template_device_index_picker,
                    marker_group_picker,
                    group_icon_picker,
                    marker_icon_picker,
                    point_reorder_target_id: None,
                    point_reorder_picker,
                    minimap_region_picker_window: None,
                    minimap_presence_probe_picker_window: None,
                    minimap_presence_probe_review_window: None,
                    tracker_pip_window: None,
                    tracker_pip_capture_panel_window: None,
                    tracker_pip_window_bounds: None,
                    tracker_pip_always_on_top: false,
                    tracker_pip_pending_open: false,
                    debug_log_revision: logging::debug_log_revision(),
                    ui_preferences_path: ui_preferences_path.clone(),
                    bwiki_resources,
                    bwiki_tile_cache,
                    bwiki_version,
                    bwiki_visible_mark_types: HashSet::new(),
                    bwiki_expanded_categories: HashSet::new(),
                    bwiki_visibility_initialized: false,
                    bwiki_icon_picker_version: 0,
                    config_form,
                    group_form,
                    group_inline_edit,
                    marker_form,
                    subscriptions: Vec::new(),
                }
                .with_optional_status_suffix(bwiki_manager_error)
            }
        };

        apply_theme_preference(workbench.theme_preference, window, cx);
        workbench
            .subscriptions
            .push(cx.observe_window_appearance(window, |this, window, cx| {
                if this.theme_preference == ThemePreference::FollowSystem {
                    apply_theme_preference(this.theme_preference, window, cx);
                    cx.notify();
                }
            }));
        let map_group_search = workbench.map_group_list.search.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &map_group_search,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.reset_paged_list_page(PagedListKind::MapGroups, window, cx);
                    cx.notify();
                }
            },
        ));
        let map_group_page_input = workbench.map_group_list.page_input.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &map_group_page_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.jump_map_group_page_from_input(window, cx);
                    cx.notify();
                }
            },
        ));
        let marker_group_search = workbench.marker_group_list.search.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_group_search,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.reset_paged_list_page(PagedListKind::MarkerGroups, window, cx);
                    cx.notify();
                }
            },
        ));
        let marker_group_page_input = workbench.marker_group_list.page_input.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_group_page_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.jump_marker_group_page_from_input(window, cx);
                    cx.notify();
                }
            },
        ));
        let point_search = workbench.point_list.search.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &point_search,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.reset_paged_list_page(PagedListKind::Points, window, cx);
                    cx.notify();
                }
            },
        ));
        let point_page_input = workbench.point_list.page_input.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &point_page_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.jump_point_page_from_input(window, cx);
                    cx.notify();
                }
            },
        ));
        let bwiki_category_search = workbench.bwiki_category_search.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_category_search,
            window,
            |_, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        ));
        let bwiki_type_search = workbench.bwiki_type_search.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_type_search,
            window,
            |_, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        ));
        let bwiki_planner_name = workbench.bwiki_planner_form.name.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_planner_name,
            window,
            |_, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        ));
        let bwiki_planner_description = workbench.bwiki_planner_form.description.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_planner_description,
            window,
            |_, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        ));
        let bwiki_planner_color = workbench.bwiki_planner_form.color_hex.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_planner_color,
            window,
            |_, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        ));
        let marker_group_picker = workbench.marker_group_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_group_picker,
            window,
            |this, _, event: &SelectEvent<MarkerGroupPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(group_id)) = event else {
                    return;
                };
                this.select_group(group_id.clone(), window, cx);
                cx.notify();
            },
        ));
        let group_name_input = workbench.group_form.name.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &group_name_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_group(window, cx);
                    cx.notify();
                }
            },
        ));
        let group_description_input = workbench.group_form.description.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &group_description_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_group(window, cx);
                    cx.notify();
                }
            },
        ));
        let group_color_input = workbench.group_form.color_hex.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &group_color_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_group(window, cx);
                    cx.notify();
                }
            },
        ));
        let group_icon_picker = workbench.group_icon_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &group_icon_picker,
            window,
            |this, _, event: &SelectEvent<BwikiIconPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(icon_name)) = event else {
                    return;
                };
                this.group_icon = icon_name.clone();
                this.autosave_selected_group(window, cx);
                cx.notify();
            },
        ));
        let bwiki_planner_icon_picker = workbench.bwiki_planner_icon_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &bwiki_planner_icon_picker,
            window,
            |this, _, event: &SelectEvent<BwikiIconPickerItem>, _, cx| {
                let SelectEvent::Confirm(Some(icon_name)) = event else {
                    return;
                };
                this.bwiki_planner_icon = icon_name.clone();
                cx.notify();
            },
        ));
        let ai_device_picker = workbench.ai_device_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &ai_device_picker,
            window,
            |this, _, event: &SelectEvent<DevicePreferencePickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                this.ai_device_preference = *device;
                this.sync_ai_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let ai_device_index_picker = workbench.ai_device_index_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &ai_device_index_picker,
            window,
            |this, _, event: &SelectEvent<DeviceIndexPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device_index)) = event else {
                    return;
                };
                this.ai_device_index = *device_index;
                this.sync_ai_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let minimap_presence_probe_device_picker =
            workbench.minimap_presence_probe_device_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &minimap_presence_probe_device_picker,
            window,
            |this, _, event: &SelectEvent<DevicePreferencePickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                this.minimap_presence_probe_device_preference = *device;
                this.sync_minimap_presence_probe_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let minimap_presence_probe_device_index_picker =
            workbench.minimap_presence_probe_device_index_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &minimap_presence_probe_device_index_picker,
            window,
            |this, _, event: &SelectEvent<DeviceIndexPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device_index)) = event else {
                    return;
                };
                this.minimap_presence_probe_device_index = *device_index;
                this.sync_minimap_presence_probe_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let template_input_mode_picker = workbench.template_input_mode_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &template_input_mode_picker,
            window,
            |this, _, event: &SelectEvent<TemplateInputModePickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(mode)) = event else {
                    return;
                };
                this.template_input_mode = *mode;
                this.sync_template_input_mode_picker_state(window, cx);
                cx.notify();
            },
        ));
        let template_device_picker = workbench.template_device_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &template_device_picker,
            window,
            |this, _, event: &SelectEvent<DevicePreferencePickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                this.template_device_preference = *device;
                this.sync_template_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let template_device_index_picker = workbench.template_device_index_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &template_device_index_picker,
            window,
            |this, _, event: &SelectEvent<DeviceIndexPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(device_index)) = event else {
                    return;
                };
                this.template_device_index = *device_index;
                this.sync_template_device_picker_state(window, cx);
                cx.notify();
            },
        ));
        let marker_icon_picker = workbench.marker_icon_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_icon_picker,
            window,
            |this, _, event: &SelectEvent<BwikiIconPickerItem>, window, cx| {
                let SelectEvent::Confirm(Some(icon_name)) = event else {
                    return;
                };
                this.marker_icon = icon_name.clone();
                this.autosave_selected_point(window, cx);
                cx.notify();
            },
        ));
        let marker_label_input = workbench.marker_form.label.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_label_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_point(window, cx);
                    cx.notify();
                }
            },
        ));
        let marker_note_input = workbench.marker_form.note.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_note_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_point(window, cx);
                    cx.notify();
                }
            },
        ));
        let marker_x_input = workbench.marker_form.x.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_x_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_point(window, cx);
                    cx.notify();
                }
            },
        ));
        let marker_y_input = workbench.marker_form.y.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_y_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.autosave_selected_point(window, cx);
                    cx.notify();
                }
            },
        ));
        let point_reorder_picker = workbench.point_reorder_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &point_reorder_picker,
            window,
            |this, _, event: &SelectEvent<PointReorderTargetItem>, _, cx| {
                let SelectEvent::Confirm(target_id) = event;
                this.point_reorder_target_id = target_id.clone();
                cx.notify();
            },
        ));
        let inline_group_name = workbench.group_inline_edit.name.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &inline_group_name,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.commit_inline_group_edit(window, cx);
                    cx.notify();
                }
            },
        ));
        let inline_group_description = workbench.group_inline_edit.description.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &inline_group_description,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.commit_inline_group_edit(window, cx);
                    cx.notify();
                }
            },
        ));
        workbench.sync_config_form_from_workspace(window, cx);
        workbench.ensure_tracker_page_selected_point();
        workbench.sync_editor_from_selection(window, cx);
        workbench.request_center_on_current_point();
        workbench.sync_bwiki_visibility_defaults();
        if let Some(message) = preferences_error {
            warn!(
                message,
                "continuing with default UI preferences after load failure"
            );
            workbench.status_text = format!("{} {message}", workbench.status_text).into();
        }

        cx.spawn(async move |this, cx| {
            loop {
                let updated = this.update(cx, |this, cx| {
                    if this.poll_tracking_events(cx)
                        || this.poll_bwiki_resources()
                        || this.poll_runtime_logs()
                        || this.tick_busy_indicator()
                    {
                        cx.notify();
                    }
                    this.refresh_tracker_pip_window(cx);
                });
                if updated.is_err() {
                    break;
                }

                cx.background_executor()
                    .timer(Duration::from_millis(33))
                    .await;
            }
        })
        .detach();

        workbench
    }

    fn empty_workspace(project_root: PathBuf) -> WorkspaceSnapshot {
        let map_dimensions = default_map_dimensions();
        WorkspaceSnapshot {
            project_root: project_root.clone(),
            config: AppConfig::default(),
            assets: AssetManifest {
                config_path: project_root.join(CONFIG_FILE_NAME),
                routes_dir: project_root.join("routes"),
                bwiki_cache_dir: project_root.join("cache").join("bwiki"),
                map_dimensions,
            },
            groups: Vec::new(),
            report: WorkspaceLoadReport {
                group_count: 0,
                point_count: 0,
                map_dimensions,
            },
        }
    }

    fn new_bwiki_resource_manager(cache_dir: PathBuf) -> (BwikiResourceManager, Option<String>) {
        match BwikiResourceManager::new(cache_dir.clone()) {
            Ok(manager) => {
                manager.ensure_dataset_loaded();
                info!(cache_dir = %cache_dir.display(), "BWiki resource manager initialized");
                (manager, None)
            }
            Err(error) => {
                warn!(
                    cache_dir = %cache_dir.display(),
                    error = %error,
                    "failed to initialize BWiki resource manager, trying fallback cache"
                );
                let fallback = env::temp_dir().join("rocom-compass").join("bwiki-cache");
                let manager = BwikiResourceManager::new(fallback.clone())
                    .unwrap_or_else(|fallback_error| {
                        error!(
                            cache_dir = %cache_dir.display(),
                            fallback = %fallback.display(),
                            error = %error,
                            fallback_error = %fallback_error,
                            "failed to initialize both primary and fallback BWiki caches"
                        );
                        panic!(
                            "failed to initialize BWiki cache at {} ({error:#}) or fallback {} ({fallback_error:#})",
                            cache_dir.display(),
                            fallback.display()
                        )
                    });
                manager.ensure_dataset_loaded();
                info!(
                    cache_dir = %cache_dir.display(),
                    fallback = %fallback.display(),
                    "BWiki resource manager initialized with fallback cache"
                );
                (
                    manager,
                    Some(format!(
                        "BWiki 缓存目录 {} 初始化失败，已临时改用 {}：{error:#}",
                        cache_dir.display(),
                        fallback.display()
                    )),
                )
            }
        }
    }

    fn with_optional_status_suffix(mut self, suffix: Option<String>) -> Self {
        if let Some(suffix) = suffix {
            self.status_text = format!("{} {}", self.status_text, suffix).into();
        }
        self
    }

    fn device_preference_picker_items() -> Vec<DevicePreferencePickerItem> {
        available_burn_backend_preferences()
            .into_iter()
            .filter(|preference| !available_burn_device_descriptors(*preference).is_empty())
            .map(|preference| match preference {
                AiDevicePreference::Cpu => DevicePreferencePickerItem::new(
                    preference,
                    "CPU",
                    "始终可用",
                    "cpu processor host",
                ),
                AiDevicePreference::Cuda => DevicePreferencePickerItem::new(
                    preference,
                    "CUDA",
                    format!(
                        "NVIDIA CUDA · {} 台设备",
                        available_burn_device_descriptors(preference).len()
                    ),
                    "cuda nvidia gpu rtx geforce",
                ),
                AiDevicePreference::Vulkan => DevicePreferencePickerItem::new(
                    preference,
                    "Vulkan",
                    format!(
                        "通用 GPU · {} 台设备",
                        available_burn_device_descriptors(preference).len()
                    ),
                    "vulkan intel amd nvidia gpu integrated discrete",
                ),
                AiDevicePreference::Metal => DevicePreferencePickerItem::new(
                    preference,
                    "Metal",
                    format!(
                        "Apple GPU · {} 台设备",
                        available_burn_device_descriptors(preference).len()
                    ),
                    "metal apple gpu",
                ),
            })
            .collect()
    }

    fn device_index_picker_items(preference: AiDevicePreference) -> Vec<DeviceIndexPickerItem> {
        let prefix = match preference {
            AiDevicePreference::Cpu => "CPU",
            AiDevicePreference::Cuda => "CUDA",
            AiDevicePreference::Vulkan => "Vulkan",
            AiDevicePreference::Metal => "Metal",
        };

        available_burn_device_descriptors(preference)
            .into_iter()
            .map(|device| {
                let title = format!("{prefix} #{}", device.ordinal);
                let subtitle = if preference == AiDevicePreference::Cpu {
                    device.name.clone()
                } else {
                    format!("{} · {}", device.name, prefix)
                };
                DeviceIndexPickerItem::new(
                    device.ordinal,
                    title,
                    subtitle,
                    format!("{prefix} {} {}", device.ordinal, device.name),
                )
            })
            .collect()
    }

    fn template_input_mode_picker_items() -> Vec<TemplateInputModePickerItem> {
        vec![
            TemplateInputModePickerItem::new(
                TemplateInputMode::Color,
                "彩色",
                "默认，保留颜色与亮度信息",
                "color rgb chroma 彩色 颜色",
            ),
            TemplateInputModePickerItem::new(
                TemplateInputMode::Grayscale,
                "灰度",
                "只保留亮度纹理，忽略色差",
                "grayscale gray mono 灰度 黑白 亮度",
            ),
        ]
    }

    fn normalized_device_selection(
        preference: AiDevicePreference,
        device_index: usize,
    ) -> (AiDevicePreference, usize) {
        let preference_items = Self::device_preference_picker_items();
        let fallback_preference = preference_items
            .first()
            .map(|item| item.value)
            .unwrap_or(AiDevicePreference::Cpu);
        let preference = if preference_items.iter().any(|item| item.value == preference) {
            preference
        } else {
            fallback_preference
        };

        let index_items = Self::device_index_picker_items(preference);
        let fallback_index = index_items.first().map(|item| item.value).unwrap_or(0);
        let device_index = if index_items.iter().any(|item| item.value == device_index) {
            device_index
        } else {
            fallback_index
        };

        (preference, device_index)
    }

    fn sync_ai_device_picker_state(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (preference, device_index) =
            Self::normalized_device_selection(self.ai_device_preference, self.ai_device_index);
        self.ai_device_preference = preference;
        self.ai_device_index = device_index;

        let preference_items = Self::device_preference_picker_items();
        self.ai_device_picker.update(cx, |picker, cx| {
            picker.set_items(preference_items.clone(), window, cx);
            picker.set_selected_value(&preference, window, cx);
        });

        let index_items = Self::device_index_picker_items(preference);
        self.ai_device_index_picker.update(cx, |picker, cx| {
            picker.set_items(index_items.clone(), window, cx);
            picker.set_selected_value(&device_index, window, cx);
        });
    }

    fn sync_minimap_presence_probe_device_picker_state(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (preference, device_index) = Self::normalized_device_selection(
            self.minimap_presence_probe_device_preference,
            self.minimap_presence_probe_device_index,
        );
        self.minimap_presence_probe_device_preference = preference;
        self.minimap_presence_probe_device_index = device_index;

        let preference_items = Self::device_preference_picker_items();
        self.minimap_presence_probe_device_picker
            .update(cx, |picker, cx| {
                picker.set_items(preference_items.clone(), window, cx);
                picker.set_selected_value(&preference, window, cx);
            });

        let index_items = Self::device_index_picker_items(preference);
        self.minimap_presence_probe_device_index_picker
            .update(cx, |picker, cx| {
                picker.set_items(index_items.clone(), window, cx);
                picker.set_selected_value(&device_index, window, cx);
            });
    }

    fn sync_template_input_mode_picker_state(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let items = Self::template_input_mode_picker_items();
        let selected = if items
            .iter()
            .any(|item| item.value == self.template_input_mode)
        {
            self.template_input_mode
        } else {
            TemplateInputMode::Color
        };
        self.template_input_mode = selected;
        self.template_input_mode_picker.update(cx, |picker, cx| {
            picker.set_items(items.clone(), window, cx);
            picker.set_selected_value(&selected, window, cx);
        });
    }

    fn sync_template_device_picker_state(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (preference, device_index) = Self::normalized_device_selection(
            self.template_device_preference,
            self.template_device_index,
        );
        self.template_device_preference = preference;
        self.template_device_index = device_index;

        let preference_items = Self::device_preference_picker_items();
        self.template_device_picker.update(cx, |picker, cx| {
            picker.set_items(preference_items.clone(), window, cx);
            picker.set_selected_value(&preference, window, cx);
        });

        let index_items = Self::device_index_picker_items(preference);
        self.template_device_index_picker.update(cx, |picker, cx| {
            picker.set_items(index_items.clone(), window, cx);
            picker.set_selected_value(&device_index, window, cx);
        });
    }

    pub(super) fn is_tracking_active(&self) -> bool {
        self.tracker_session.is_some()
    }

    fn tracking_debug_enabled(&self) -> bool {
        self.debug_mode_enabled
            && self.active_page == WorkbenchPage::Settings
            && self.settings_page == SettingsPage::Debug
    }

    fn sync_tracker_debug_enabled(&self) {
        if let Some(session) = self.tracker_session.as_ref() {
            session.set_debug_enabled(self.tracking_debug_enabled());
        }
    }

    fn reset_tracker_pip_debug_summaries(&mut self) {
        self.tracker_pip_probe_summary = TRACKER_PIP_PROBE_IDLE_SUMMARY.into();
        self.tracker_pip_locate_summary = TRACKER_PIP_LOCATE_IDLE_SUMMARY.into();
    }

    fn set_tracker_pip_debug_summaries(
        &mut self,
        probe_summary: impl Into<SharedString>,
        locate_summary: impl Into<SharedString>,
    ) {
        self.tracker_pip_probe_summary = probe_summary.into();
        self.tracker_pip_locate_summary = locate_summary.into();
    }

    pub(super) fn is_tracker_transition_pending(&self) -> bool {
        self.tracker_pending_action.is_some()
    }

    pub(super) fn busy_spinner_icon(&self) -> &'static str {
        match self.spinner_frame % 4 {
            0 => "|",
            1 => "/",
            2 => "-",
            _ => "\\",
        }
    }

    pub(super) fn tracker_status_summary(&self) -> SharedString {
        match self.tracker_pending_action {
            Some(TrackerPendingAction::Starting) => "启动中".into(),
            Some(TrackerPendingAction::Stopping) => "停止中".into(),
            None if self.is_tracking_active()
                && self.tracker_lifecycle == TrackerLifecycle::Running =>
            {
                "运行中".into()
            }
            None if self.tracker_lifecycle == TrackerLifecycle::Failed => "失败".into(),
            _ => "未启动".into(),
        }
    }

    pub(super) fn tracker_status_detail(&self) -> SharedString {
        self.tracker_status_text.clone()
    }

    pub(super) fn tracker_status_tooltip(&self) -> SharedString {
        format!(
            "追踪状态：{}。{}",
            self.tracker_status_summary(),
            self.tracker_status_detail()
        )
        .into()
    }

    pub(super) fn tracker_toggle_label(&self) -> SharedString {
        match self.tracker_pending_action {
            Some(TrackerPendingAction::Starting) => "启动中".into(),
            Some(TrackerPendingAction::Stopping) => "停止中".into(),
            None if self.is_tracking_active() => "停止追踪".into(),
            None if self.tracker_lifecycle == TrackerLifecycle::Failed => "重新启动".into(),
            _ => "启动追踪".into(),
        }
    }

    pub(super) fn is_tracker_pip_open(&self) -> bool {
        self.tracker_pip_window.is_some()
    }

    pub(super) const fn is_tracker_pip_pending_open(&self) -> bool {
        self.tracker_pip_pending_open
    }

    pub(super) fn tracker_pip_toggle_label(&self) -> SharedString {
        if self.tracker_pip_pending_open {
            "打开中".into()
        } else if self.is_tracker_pip_open() {
            "关闭画中画".into()
        } else {
            "打开画中画".into()
        }
    }

    pub(super) fn is_bwiki_refreshing(&self) -> bool {
        self.bwiki_resources.is_dataset_refresh_pending()
    }

    pub(super) fn is_bwiki_planner_busy(&self) -> bool {
        self.bwiki_planner_status.phase == AsyncTaskPhase::Working
    }

    pub(super) fn bwiki_planner_tooltip(&self) -> SharedString {
        self.bwiki_planner_status.summary.clone()
    }

    pub(super) fn bwiki_status_summary(&self) -> SharedString {
        if self.is_bwiki_refreshing() {
            "同步中".into()
        } else if self.bwiki_resources.last_error().is_some() {
            "失败".into()
        } else if self.bwiki_resources.dataset_snapshot().is_some() {
            "已就绪".into()
        } else {
            "等待数据".into()
        }
    }

    pub(super) fn bwiki_status_detail(&self) -> SharedString {
        if self.is_bwiki_refreshing() {
            return "正在同步节点图鉴数据与图标目录。".into();
        }
        if let Some(error) = self.bwiki_resources.last_error() {
            return format!("节点图鉴同步失败：{error}").into();
        }
        if let Some(dataset) = self.bwiki_resources.dataset_snapshot() {
            return format!(
                "节点图鉴数据已就绪，共 {} 个分类、{} 个类型。",
                dataset.sorted_category_names().len(),
                dataset.types.len()
            )
            .into();
        }
        "正在等待节点图鉴数据。".into()
    }

    pub(super) fn bwiki_status_tooltip(&self) -> SharedString {
        format!(
            "节点图鉴状态：{}。{}",
            self.bwiki_status_summary(),
            self.bwiki_status_detail()
        )
        .into()
    }

    pub(super) fn is_route_import_busy(&self) -> bool {
        self.route_import_status.phase == AsyncTaskPhase::Working
    }

    pub(super) fn route_import_tooltip(&self) -> SharedString {
        self.route_import_status.summary.clone()
    }

    pub(super) const fn is_auto_focus_enabled(&self) -> bool {
        self.auto_focus_enabled
    }

    pub(super) const fn is_tracker_point_popup_enabled(&self) -> bool {
        self.tracker_point_popup_enabled
    }

    pub(super) const fn is_debug_mode_enabled(&self) -> bool {
        self.debug_mode_enabled
    }

    pub(super) const fn is_test_case_capture_enabled(&self) -> bool {
        self.test_case_capture_enabled
    }

    pub(super) fn current_position_label(&self) -> String {
        self.preview_position.as_ref().map_or_else(
            || "--".to_owned(),
            |position| format!("{:.0}, {:.0}", position.world.x, position.world.y),
        )
    }

    pub(super) const fn is_map_point_insert_armed(&self) -> bool {
        self.map_point_insert_armed
    }

    pub(super) const fn is_route_editor_draw_mode(&self) -> bool {
        self.route_editor_draw_mode
    }

    fn effective_route_editor_selected_point_ids(&self) -> HashSet<RoutePointId> {
        if !self.route_editor_selected_point_ids.is_empty() {
            return self.route_editor_selected_point_ids.clone();
        }

        self.selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect()
    }

    pub(super) fn route_editor_selected_count(&self) -> usize {
        self.effective_route_editor_selected_point_ids().len()
    }

    fn route_editor_graph_state_matches_group(&self, group: &RouteDocument) -> bool {
        self.route_editor_graph_state.as_ref().is_some_and(|state| {
            state.group_id == group.id && state.point_ids == route_point_id_set(&group.points)
        })
    }

    pub(super) fn route_editor_has_graph_draft(&self) -> bool {
        if !self.route_editor_draw_mode {
            return false;
        }
        let Some(group) = self.active_group() else {
            return false;
        };
        let Some(state) = self.route_editor_graph_state.as_ref() else {
            return false;
        };
        if !self.route_editor_graph_state_matches_group(group) {
            return false;
        }

        state.edges != route_graph_edges_from_points(&group.points)
    }

    pub(super) fn route_editor_can_remove_selected_edges(&self) -> bool {
        if !self.route_editor_draw_mode {
            return false;
        }
        let Some(group) = self.active_group() else {
            return false;
        };
        let selected = self.effective_route_editor_selected_point_ids();
        if selected.len() < 2 {
            return false;
        }

        self.route_graph_edges_for_group(group, true)
            .into_iter()
            .any(|edge| selected.contains(&edge.from) && selected.contains(&edge.to))
    }

    pub(super) fn route_editor_can_save_graph_edit(&self) -> bool {
        self.route_editor_resolved_order().is_some()
    }

    pub(super) fn is_selected_point_move_armed(&self) -> bool {
        self.selected_point_id
            .as_ref()
            .is_some_and(|point_id| self.moving_point_id.as_ref() == Some(point_id))
    }

    pub(super) fn is_selected_point_delete_confirming(&self) -> bool {
        self.selected_point_id
            .as_ref()
            .is_some_and(|point_id| self.confirming_delete_point_id.as_ref() == Some(point_id))
    }

    pub(super) fn active_group(&self) -> Option<&RouteDocument> {
        let group_id = self.selected_group_id.as_ref()?;
        self.route_groups.iter().find(|group| &group.id == group_id)
    }

    pub(super) fn selected_point(&self) -> Option<&RoutePoint> {
        let point_id = self.selected_point_id.as_ref()?;
        self.active_group()?.find_point(point_id)
    }

    fn selected_point_index(&self) -> Option<usize> {
        let point_id = self.selected_point_id.as_ref()?;
        self.active_group()?
            .points
            .iter()
            .position(|point| &point.id == point_id)
    }

    fn route_graph_edges_for_group(
        &self,
        group: &RouteDocument,
        use_graph_edit_state: bool,
    ) -> HashSet<RouteGraphEdge> {
        if use_graph_edit_state && self.route_editor_graph_state_matches_group(group) {
            return self
                .route_editor_graph_state
                .as_ref()
                .map(|state| state.edges.clone())
                .unwrap_or_default();
        }

        route_graph_edges_from_points(&group.points)
    }

    fn active_route_graph_render_state(&self, use_graph_edit_state: bool) -> RouteGraphRenderState {
        let Some(group) = self.active_group() else {
            return RouteGraphRenderState::default();
        };
        let edges = self.route_graph_edges_for_group(group, use_graph_edit_state);

        let point_lookup = group
            .points
            .iter()
            .map(|point| {
                (
                    point.id.clone(),
                    self.moving_point_preview_world(&point.id)
                        .unwrap_or_else(|| point.world()),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut incoming = group
            .points
            .iter()
            .map(|point| (point.id.clone(), 0usize))
            .collect::<HashMap<_, _>>();
        let mut outgoing = incoming.clone();
        let mut segments = edges
            .iter()
            .filter_map(|edge| {
                let from = point_lookup.get(&edge.from).copied()?;
                let to = point_lookup.get(&edge.to).copied()?;
                *incoming.entry(edge.to.clone()).or_default() += 1;
                *outgoing.entry(edge.from.clone()).or_default() += 1;
                Some(RouteSegmentRenderItem { from, to })
            })
            .collect::<Vec<_>>();
        segments.sort_by(|left, right| {
            left.from
                .x
                .total_cmp(&right.from.x)
                .then_with(|| left.from.y.total_cmp(&right.from.y))
                .then_with(|| left.to.x.total_cmp(&right.to.x))
                .then_with(|| left.to.y.total_cmp(&right.to.y))
        });

        let (start_ids, end_ids) = if use_graph_edit_state {
            // While the user is editing edges, the draft graph can be incomplete or mid-gesture.
            // Hide start/end badges entirely until the edit is saved back as an ordered route.
            (HashSet::new(), HashSet::new())
        } else {
            let mut start_ids = HashSet::new();
            let mut end_ids = HashSet::new();
            for point in &group.points {
                let in_degree = incoming.get(&point.id).copied().unwrap_or(0);
                let out_degree = outgoing.get(&point.id).copied().unwrap_or(0);
                if in_degree == 0 {
                    start_ids.insert(point.id.clone());
                }
                if out_degree == 0 {
                    end_ids.insert(point.id.clone());
                }
            }
            (start_ids, end_ids)
        };

        RouteGraphRenderState {
            segments,
            start_ids,
            end_ids,
        }
    }

    fn active_group_points(&self, use_graph_edit_state: bool) -> Vec<MapPointRenderItem> {
        let render_state = self.active_route_graph_render_state(use_graph_edit_state);
        self.active_group()
            .map(|group| {
                group
                    .points
                    .iter()
                    .map(|point| MapPointRenderItem {
                        group_id: group.id.clone(),
                        point_id: point.id.clone(),
                        world: self
                            .moving_point_preview_world(&point.id)
                            .unwrap_or_else(|| point.world()),
                        style: group.effective_style(point),
                        is_start: render_state.start_ids.contains(&point.id),
                        is_end: render_state.end_ids.contains(&point.id),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn point_reorder_picker_items(&self) -> Vec<PointReorderTargetItem> {
        let selected_point_id = self.selected_point_id.as_ref();
        self.active_group()
            .map(|group| {
                group
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(_, point)| Some(&point.id) != selected_point_id)
                    .map(|(index, point)| {
                        let title = format!("{:02}. {}", index + 1, point.display_label());
                        PointReorderTargetItem::new(
                            point.id.clone(),
                            title.clone(),
                            "",
                            format!("{title} {} {:.0} {:.0}", point.note, point.x, point.y),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn sync_point_reorder_picker_state(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let items = self.point_reorder_picker_items();
        let selected_target_id = self
            .point_reorder_target_id
            .as_ref()
            .filter(|target_id| items.iter().any(|item| &item.id == *target_id))
            .cloned()
            .or_else(|| items.first().map(|item| item.id.clone()));

        self.point_reorder_target_id = selected_target_id.clone();
        self.point_reorder_picker.update(cx, |picker, cx| {
            picker.set_items(items, window, cx);
            if let Some(target_id) = selected_target_id.as_ref() {
                picker.set_selected_value(target_id, window, cx);
            } else {
                picker.set_selected_index(None, window, cx);
            }
        });
    }

    pub(super) fn tracker_map_render_snapshot(&self) -> TrackerMapRenderSnapshot {
        let use_graph_edit_state =
            self.active_page == WorkbenchPage::Markers && self.route_editor_draw_mode;
        let render_state = self.active_route_graph_render_state(use_graph_edit_state);
        let pip_tracker_toggle_state = match self.tracker_pending_action {
            Some(TrackerPendingAction::Starting) => TrackerPipToggleState::Starting,
            Some(TrackerPendingAction::Stopping) => TrackerPipToggleState::Stopping,
            None if self.is_tracking_active() => TrackerPipToggleState::Stop,
            None if self.tracker_lifecycle == TrackerLifecycle::Failed => {
                TrackerPipToggleState::Restart
            }
            _ => TrackerPipToggleState::Start,
        };

        TrackerMapRenderSnapshot {
            route_color_hex: self
                .active_group()
                .map(|group| group.default_style.color_hex.clone()),
            trail: self.trail.clone(),
            preview_position: self.preview_position.clone(),
            route_segments: render_state.segments,
            point_visuals: self.active_group_points(use_graph_edit_state),
            selected_group_id: self.selected_group_id.clone(),
            selected_point_id: self.selected_point_id.clone(),
            selected_point_ids: self.effective_route_editor_selected_point_ids(),
            route_editor_lasso_path: self
                .route_editor_lasso_selection
                .as_ref()
                .map(|selection| selection.path.clone()),
            follow_point: self
                .auto_focus_enabled
                .then(|| {
                    self.preview_position
                        .as_ref()
                        .map(|position| position.world)
                })
                .flatten(),
            pip_always_on_top: self.tracker_pip_always_on_top,
            pip_tracker_toggle_state,
            pip_tracker_status_tooltip: self.tracker_status_tooltip(),
            pip_probe_summary: self.tracker_pip_probe_summary.clone(),
            pip_locate_summary: self.tracker_pip_locate_summary.clone(),
            pip_test_case_capture_enabled: self.test_case_capture_enabled,
            pip_capture_panel_expanded: self.tracker_pip_capture_panel_window.is_some(),
        }
    }

    fn moving_point_preview_world(&self, point_id: &RoutePointId) -> Option<WorldPoint> {
        (self.moving_point_id.as_ref() == Some(point_id))
            .then_some(self.moving_point_preview)
            .flatten()
    }

    fn selected_point_world_for_render(&self) -> Option<WorldPoint> {
        let point = self.selected_point()?;
        Some(
            self.moving_point_preview_world(&point.id)
                .unwrap_or_else(|| point.world()),
        )
    }

    pub(super) fn selected_tracker_point_popup(&self) -> Option<SelectedPointPopup> {
        if self.is_selected_point_move_armed() {
            return None;
        }
        if !self.tracker_point_popup_enabled {
            return None;
        }
        if self.active_page == WorkbenchPage::Markers
            && (self.route_editor_selected_count() != 1 || self.route_editor_draw_mode)
        {
            return None;
        }
        let group = self.active_group()?;
        self.selected_point()?;
        let map_view = if self.active_page == WorkbenchPage::Markers {
            &self.route_editor_map_view
        } else {
            &self.tracker_map_view
        };
        let viewport = map_view.viewport;
        if !viewport.is_valid() {
            return None;
        }

        let screen = map_view
            .camera
            .world_to_screen(self.selected_point_world_for_render()?);
        let popup_width = if self.active_page == WorkbenchPage::Markers {
            360.0
        } else {
            248.0
        };
        let popup_height = if self.active_page == WorkbenchPage::Markers {
            520.0
        } else {
            96.0
        };
        let max_left = (viewport.width - popup_width - 8.0).max(8.0);
        let max_top = (viewport.height - popup_height - 8.0).max(8.0);

        Some(SelectedPointPopup {
            left: (screen.x + 18.0).clamp(8.0, max_left),
            top: (screen.y - popup_height - 16.0).clamp(8.0, max_top),
            width: popup_width,
            height: popup_height,
            route_name: group.display_name().to_owned(),
        })
    }

    pub(super) fn tracker_popup_hit_test(&self, local_x: f32, local_y: f32) -> bool {
        self.selected_tracker_point_popup()
            .is_some_and(|popup| popup.contains(local_x, local_y))
    }

    pub(super) fn default_marker_world(&self) -> WorldPoint {
        if let Some(point) = self.selected_point() {
            return point.world();
        }
        if let Some(position) = self.preview_position.as_ref() {
            return position.world;
        }

        WorldPoint::new(
            self.workspace.report.map_dimensions.width as f32 * 0.5,
            self.workspace.report.map_dimensions.height as f32 * 0.5,
        )
    }

    fn clamp_tracker_world(&self, world: WorldPoint) -> WorldPoint {
        let map = self.workspace.report.map_dimensions;
        WorldPoint::new(
            world.x.clamp(0.0, map.width as f32),
            world.y.clamp(0.0, map.height as f32),
        )
    }

    fn clear_selected_point_move_state(&mut self) {
        self.moving_point_id = None;
        self.moving_point_preview = None;
        self.ignore_next_tracker_mouse_up = false;
    }

    pub(super) fn consume_tracker_mouse_up_guard(&mut self) -> bool {
        std::mem::take(&mut self.ignore_next_tracker_mouse_up)
    }

    pub(super) fn suppress_next_tracker_mouse_up(&mut self) {
        self.ignore_next_tracker_mouse_up = true;
    }

    fn set_route_editor_selected_points(
        &mut self,
        selected: HashSet<RoutePointId>,
        preferred_primary: Option<RoutePointId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.route_editor_selected_point_ids = selected;
        self.selected_point_id = preferred_primary
            .filter(|point_id| self.route_editor_selected_point_ids.contains(point_id))
            .or_else(|| {
                self.selected_point_id
                    .clone()
                    .filter(|point_id| self.route_editor_selected_point_ids.contains(point_id))
            })
            .or_else(|| self.route_editor_selected_point_ids.iter().next().cloned());
        self.confirming_delete_point_id = None;
        if self.route_editor_selected_point_ids.len() != 1 {
            self.clear_selected_point_move_state();
        }
        self.sync_editor_from_selection(window, cx);
    }

    fn reset_route_editor_draw_state(&mut self) {
        self.route_editor_draw_mode = false;
        self.route_editor_draw_sequence.clear();
        self.route_editor_lasso_selection = None;
    }

    fn reset_route_editor_graph_state(&mut self) {
        self.route_editor_graph_state = None;
        self.reset_route_editor_draw_state();
    }

    fn ensure_route_editor_graph_state(&mut self) -> Option<&mut RouteGraphEditState> {
        let group = self.active_group()?.clone();
        if !self.route_editor_graph_state_matches_group(&group) {
            self.route_editor_graph_state = Some(RouteGraphEditState::from_group(&group));
        }
        self.route_editor_graph_state.as_mut()
    }

    fn route_editor_points_in_lasso(&self, lasso_path: &[WorldPoint]) -> Vec<RoutePointId> {
        if lasso_path.len() < 3 {
            return Vec::new();
        }

        let camera = self.route_editor_map_view.camera;
        self.active_group_points(true)
            .into_iter()
            .filter_map(|point| {
                let screen = camera.world_to_screen(point.world);
                point_in_polygon(screen, lasso_path).then_some(point.point_id)
            })
            .collect()
    }

    fn toggle_route_editor_point_selection(
        &mut self,
        point_id: RoutePointId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut selected = self.effective_route_editor_selected_point_ids();
        if !selected.remove(&point_id) {
            selected.insert(point_id.clone());
        }
        self.set_route_editor_selected_points(selected, Some(point_id.clone()), window, cx);
        self.status_text = if self.route_editor_selected_count() == 0 {
            "已清空路线编辑多选。".into()
        } else {
            format!(
                "已切换节点多选，当前共选中 {} 个节点。",
                self.route_editor_selected_count()
            )
            .into()
        };
    }

    pub(super) fn clear_route_editor_point_selection(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.route_editor_selected_point_ids.clear();
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;
        self.clear_selected_point_move_state();
        self.sync_editor_from_selection(window, cx);
        self.status_text = "已清空路线编辑选中节点。".into();
    }

    pub(super) fn begin_route_editor_lasso_selection(&mut self, screen_x: f32, screen_y: f32) {
        self.route_editor_lasso_selection = Some(BwikiPlannerLassoSelection::new(WorldPoint::new(
            screen_x, screen_y,
        )));
    }

    pub(super) fn update_route_editor_lasso_selection(
        &mut self,
        screen_x: f32,
        screen_y: f32,
    ) -> bool {
        let Some(selection) = self.route_editor_lasso_selection.as_mut() else {
            return false;
        };
        selection.push_point(WorldPoint::new(screen_x, screen_y))
    }

    pub(super) fn finish_route_editor_lasso_selection(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(mut selection) = self.route_editor_lasso_selection.take() else {
            return false;
        };
        selection.push_point(WorldPoint::new(screen_x, screen_y));
        if selection.travel_distance() < MAP_CLICK_DRAG_THRESHOLD {
            if let Some(point_id) =
                self.point_hit_test(MapCanvasKind::RouteEditor, screen_x, screen_y)
            {
                self.toggle_route_editor_point_selection(point_id, window, cx);
                return true;
            }
            self.status_text = "当前点击位置没有可切换的节点。".into();
            return false;
        }

        let affected = self.route_editor_points_in_lasso(&selection.path);
        if affected.is_empty() {
            self.status_text = "当前圈选范围内没有可切换的节点。".into();
            return false;
        }

        let mut selected = self.effective_route_editor_selected_point_ids();
        for point_id in &affected {
            if !selected.remove(point_id) {
                selected.insert(point_id.clone());
            }
        }
        let preferred_primary = affected.last().cloned();
        self.set_route_editor_selected_points(selected, preferred_primary, window, cx);
        self.status_text = format!(
            "已反选 {} 个节点，当前共选中 {} 个节点。",
            affected.len(),
            self.route_editor_selected_count()
        )
        .into();
        true
    }

    fn route_editor_resolved_order(&self) -> Option<Vec<RoutePointId>> {
        if !self.route_editor_draw_mode {
            return None;
        }
        let group = self.active_group()?;
        let state = self.route_editor_graph_state.as_ref()?;
        if !self.route_editor_graph_state_matches_group(group) {
            return None;
        }

        resolve_route_graph_order(&group.points, &state.edges)
    }

    pub(super) fn start_route_editor_graph_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.route_editor_draw_mode {
            self.status_text = "当前已经处于连线编辑模式。".into();
            return;
        }

        let Some(group) = self.active_group().cloned() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        if group.points.len() < 2 {
            self.status_text = "当前路线至少需要 2 个节点，才能编辑连线。".into();
            return;
        }

        self.map_point_insert_armed = false;
        self.confirming_delete_point_id = None;
        self.clear_selected_point_move_state();
        self.route_editor_draw_mode = true;
        self.route_editor_draw_sequence.clear();
        self.route_editor_lasso_selection = None;
        self.route_editor_graph_state = Some(RouteGraphEditState::from_group(&group));
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.sync_editor_from_selection(window, cx);
        self.status_text = "连线编辑已开启：点击节点按顺序重连，按 Ctrl+点击或框选多选节点后可删除连线，完成后点击“保存退出”。".into();
    }

    pub(super) fn cancel_route_editor_graph_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.route_editor_draw_mode {
            self.status_text = "当前不在连线编辑模式。".into();
            return;
        }

        self.reset_route_editor_graph_state();
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.sync_editor_from_selection(window, cx);
        self.status_text = "已放弃本次连线编辑，恢复为已保存路线。".into();
    }

    pub(super) fn reset_route_editor_graph_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.route_editor_draw_mode {
            self.status_text = "请先进入连线编辑模式。".into();
            return;
        }

        let Some(group) = self.active_group().cloned() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };

        self.route_editor_graph_state = Some(RouteGraphEditState::from_group(&group));
        self.route_editor_draw_sequence.clear();
        self.route_editor_lasso_selection = None;
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.sync_editor_from_selection(window, cx);
        self.status_text = "已恢复为已保存连线，当前仍处于连线编辑模式。".into();
    }

    fn route_editor_insert_edge(
        &mut self,
        from: RoutePointId,
        to: RoutePointId,
    ) -> std::result::Result<RouteGraphInsertOutcome, String> {
        let Some(state) = self.ensure_route_editor_graph_state() else {
            return Err("请先选择一条路线。".to_owned());
        };
        route_graph_insert_edge(&mut state.edges, from, to)
    }

    pub(super) fn save_route_editor_graph_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.route_editor_draw_mode {
            self.status_text = "请先进入连线编辑模式。".into();
            return;
        }
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        let Some(order) = self.route_editor_resolved_order() else {
            self.status_text = "当前连线还没有形成完整的单向单链，无法保存退出。".into();
            return;
        };

        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            let mut points_by_id = group
                .points
                .drain(..)
                .map(|point| (point.id.clone(), point))
                .collect::<HashMap<_, _>>();
            group.points = order
                .iter()
                .filter_map(|point_id| points_by_id.remove(point_id))
                .collect();
        }

        self.reset_route_editor_graph_state();
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.preview_cursor = self.selected_point_index();
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        if self.persist_group(&group_id, "连线编辑已保存") {
            self.sync_editor_from_selection(window, cx);
        }
    }

    fn route_editor_append_draw_point(
        &mut self,
        point_id: RoutePointId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let point_label = self
            .active_group()
            .and_then(|group| group.find_point(&point_id))
            .map(|point| point.display_label().to_owned())
            .unwrap_or_else(|| point_id.0.clone());
        self.set_route_editor_selected_points(
            [point_id.clone()].into_iter().collect(),
            Some(point_id.clone()),
            window,
            cx,
        );

        let Some(previous) = self.route_editor_draw_sequence.last().cloned() else {
            self.route_editor_draw_sequence.push(point_id.clone());
            self.status_text = format!("已选择连线起点「{}」。", point_label).into();
            return true;
        };
        if previous == point_id {
            self.status_text = "当前节点已经是上一条连线的终点。".into();
            return true;
        }

        match self.route_editor_insert_edge(previous.clone(), point_id.clone()) {
            Ok(RouteGraphInsertOutcome::Unchanged) => {
                self.route_editor_draw_sequence.push(point_id);
                self.status_text = "这条有向线已存在，已将当前节点设为新的连线起点。".into();
                true
            }
            Ok(RouteGraphInsertOutcome::Added { replaced_edges }) => {
                self.route_editor_draw_sequence.push(point_id.clone());
                let graph_ready = self.route_editor_can_save_graph_edit();
                self.status_text = if graph_ready {
                    if replaced_edges == 0 {
                        "已新增一条有向线，当前图已形成可保存的单向单链，请点击“保存退出”。".into()
                    } else {
                        format!(
                            "已重连 1 条线并替换 {} 条冲突连线，当前图已形成可保存的单向单链，请点击“保存退出”。",
                            replaced_edges
                        )
                        .into()
                    }
                } else if replaced_edges == 0 {
                    "已新增一条有向线，当前图尚未形成可保存的单向单链。".into()
                } else {
                    format!(
                        "已重连 1 条线并替换 {} 条冲突连线，当前图尚未形成可保存的单向单链。",
                        replaced_edges
                    )
                    .into()
                };
                true
            }
            Err(message) => {
                self.status_text = message.into();
                true
            }
        }
    }

    pub(super) fn remove_route_editor_edges_between_selected(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.route_editor_draw_mode {
            self.status_text = "请先进入连线编辑模式。".into();
            return;
        }
        let selected = self.effective_route_editor_selected_point_ids();
        if selected.len() < 2 {
            self.status_text = "请至少多选 2 个节点，再移除它们之间的线。".into();
            return;
        }

        let Some(state) = self.ensure_route_editor_graph_state() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        let before = state.edges.len();
        state
            .edges
            .retain(|edge| !(selected.contains(&edge.from) && selected.contains(&edge.to)));
        let removed = before.saturating_sub(state.edges.len());
        self.route_editor_draw_sequence.clear();
        if removed == 0 {
            self.status_text = "当前选中节点之间没有可移除的连线。".into();
            return;
        }

        self.clear_route_editor_point_selection(window, cx);
        self.status_text = format!(
            "已移除 {} 条连线，并自动清空节点选中。请继续点击节点按顺序重连，直到重新形成单向单链后再保存退出。",
            removed
        )
        .into();
    }

    fn point_hit_test(
        &self,
        map_kind: MapCanvasKind,
        screen_x: f32,
        screen_y: f32,
    ) -> Option<RoutePointId> {
        if !matches!(
            map_kind,
            MapCanvasKind::Tracker | MapCanvasKind::RouteEditor
        ) {
            return None;
        }

        let camera = self.map_view(map_kind).camera;
        self.active_group_points(map_kind == MapCanvasKind::RouteEditor)
            .into_iter()
            .filter_map(|marker| {
                let screen = camera.world_to_screen(marker.world);
                if !crate::ui::map_canvas::bwiki_marker_hit_test(screen, screen_x, screen_y) {
                    return None;
                }
                let dx = screen.x - screen_x;
                let dy = screen.y - screen_y;
                let distance_sq = dx * dx + dy * dy;
                Some((distance_sq, marker.point_id))
            })
            .min_by(|(left_distance, _), (right_distance, _)| {
                left_distance.total_cmp(right_distance)
            })
            .map(|(_, point_id)| point_id)
    }

    fn bwiki_point_hit_test(&self, screen_x: f32, screen_y: f32) -> Option<BwikiPointKey> {
        let dataset = self.bwiki_resources.dataset_snapshot()?;
        let camera = self.bwiki_map_view.camera;

        dataset
            .types
            .iter()
            .filter(|definition| {
                self.bwiki_visible_mark_types
                    .contains(&definition.mark_type)
            })
            .filter_map(|definition| dataset.points_by_type.get(&definition.mark_type))
            .flatten()
            .filter_map(|record| {
                let screen = camera.world_to_screen(record.world);
                if !crate::ui::map_canvas::bwiki_marker_hit_test(screen, screen_x, screen_y) {
                    return None;
                }
                let dx = screen.x - screen_x;
                let dy = screen.y - screen_y;
                Some((dx * dx + dy * dy, BwikiPointKey::from_record(record)))
            })
            .min_by(|(left_distance, _), (right_distance, _)| {
                left_distance.total_cmp(right_distance)
            })
            .map(|(_, key)| key)
    }

    fn bwiki_points_in_lasso(&self, lasso_path: &[WorldPoint]) -> Vec<BwikiPointKey> {
        if lasso_path.len() < 3 {
            return Vec::new();
        }

        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return Vec::new();
        };
        let camera = self.bwiki_map_view.camera;

        dataset
            .types
            .iter()
            .filter(|definition| {
                self.bwiki_visible_mark_types
                    .contains(&definition.mark_type)
            })
            .filter_map(|definition| dataset.points_by_type.get(&definition.mark_type))
            .flatten()
            .filter_map(|record| {
                let screen = camera.world_to_screen(record.world);
                if point_in_polygon(screen, lasso_path) {
                    Some(BwikiPointKey::from_record(record))
                } else {
                    None
                }
            })
            .collect()
    }

    fn resolve_bwiki_points_by_keys(
        &self,
        keys: &[BwikiPointKey],
    ) -> Vec<BwikiPlannerResolvedPoint> {
        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return Vec::new();
        };
        resolve_bwiki_points_by_keys_from_dataset(&dataset, keys)
    }

    fn resolve_bwiki_preview_points(&self) -> Vec<BwikiPlannerResolvedPoint> {
        let Some(preview) = self.bwiki_planner_preview.as_ref() else {
            return Vec::new();
        };
        self.resolve_bwiki_points_by_keys(&preview.route_keys)
    }

    fn invalidate_bwiki_route_plan_preview(&mut self) {
        let had_preview = self.bwiki_planner_preview.take().is_some();
        if self.is_bwiki_planner_busy() {
            self.bwiki_planner_task_id = self.bwiki_planner_task_id.wrapping_add(1);
            self.bwiki_planner_status = AsyncTaskStatus::idle("规划条件已变化，请重新规划。");
        } else if had_preview || self.bwiki_planner_status.phase == AsyncTaskPhase::Succeeded {
            self.bwiki_planner_status = AsyncTaskStatus::idle("当前规划结果已失效，请重新规划。");
        }
    }

    fn toggle_bwiki_planner_keys<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = BwikiPointKey>,
    {
        let mut changed = 0usize;
        for key in keys {
            if !self.bwiki_planner_selected_points.remove(&key) {
                self.bwiki_planner_selected_points.insert(key);
            }
            changed += 1;
        }
        if changed > 0 {
            self.invalidate_bwiki_route_plan_preview();
        }
        changed
    }

    pub(super) fn is_bwiki_planner_active(&self) -> bool {
        self.bwiki_planner_active
    }

    pub(super) fn bwiki_planner_selected_count(&self) -> usize {
        self.bwiki_planner_selected_points.len()
    }

    pub(super) fn bwiki_planner_preview_total_cost(&self) -> Option<f32> {
        self.bwiki_planner_preview
            .as_ref()
            .map(|preview| preview.total_cost)
    }

    pub(super) fn bwiki_planner_has_preview(&self) -> bool {
        self.bwiki_planner_preview.is_some()
    }

    pub(super) fn bwiki_planner_route_color_hex(&self, cx: &gpui::App) -> String {
        self.bwiki_planner_form
            .color_hex
            .read(cx)
            .value()
            .to_string()
    }

    pub(super) fn bwiki_planner_preview_worlds(&self) -> Vec<WorldPoint> {
        self.resolve_bwiki_preview_points()
            .into_iter()
            .map(|point| point.record.world)
            .collect()
    }

    pub(super) fn bwiki_planner_preview_points(&self) -> Vec<BwikiPlannerResolvedPoint> {
        self.resolve_bwiki_preview_points()
    }

    fn map_view(&self, map_kind: MapCanvasKind) -> &crate::ui::map_canvas::MapViewportState {
        match map_kind {
            MapCanvasKind::Tracker => &self.tracker_map_view,
            MapCanvasKind::RouteEditor => &self.route_editor_map_view,
            MapCanvasKind::Bwiki => &self.bwiki_map_view,
        }
    }

    fn map_view_mut(
        &mut self,
        map_kind: MapCanvasKind,
    ) -> &mut crate::ui::map_canvas::MapViewportState {
        match map_kind {
            MapCanvasKind::Tracker => &mut self.tracker_map_view,
            MapCanvasKind::RouteEditor => &mut self.route_editor_map_view,
            MapCanvasKind::Bwiki => &mut self.bwiki_map_view,
        }
    }

    pub(super) fn map_camera(&self, map_kind: MapCanvasKind) -> crate::domain::geometry::MapCamera {
        self.map_view(map_kind).camera
    }

    pub(super) fn sync_map_canvas_view(
        &mut self,
        map_kind: MapCanvasKind,
        width: f32,
        height: f32,
        map_dimensions: crate::domain::geometry::MapDimensions,
    ) -> bool {
        let active_group = match map_kind {
            MapCanvasKind::Tracker | MapCanvasKind::RouteEditor => self.active_group().cloned(),
            MapCanvasKind::Bwiki => None,
        };
        let map_view = self.map_view_mut(map_kind);
        map_view.update_viewport(width, height);
        let needs_fit = map_view.needs_fit;
        map_view.fit_to_route_or_map(active_group.as_ref(), map_dimensions, 24.0);
        let centered = map_view.apply_pending_center();
        needs_fit || centered
    }

    pub(super) fn begin_map_drag(&mut self, map_kind: MapCanvasKind, screen_x: f32, screen_y: f32) {
        let map_view = self.map_view_mut(map_kind);
        let screen = WorldPoint::new(screen_x, screen_y);
        map_view.dragging_from = Some(screen);
        map_view.drag_origin = Some(screen);
        map_view.drag_moved = false;
        map_view.reset_interaction_redraw();
    }

    pub(super) fn update_map_drag(
        &mut self,
        map_kind: MapCanvasKind,
        screen_x: f32,
        screen_y: f32,
        min_interval: Duration,
    ) -> bool {
        let map_view = self.map_view_mut(map_kind);
        let Some(from) = map_view.dragging_from.take() else {
            return false;
        };
        let current = WorldPoint::new(screen_x, screen_y);

        if !map_view.drag_moved {
            let origin = map_view.drag_origin.unwrap_or(from);
            let total_dx = current.x - origin.x;
            let total_dy = current.y - origin.y;
            if total_dx.hypot(total_dy) < MAP_CLICK_DRAG_THRESHOLD {
                map_view.dragging_from = Some(current);
                return false;
            }

            map_view.drag_moved = true;
            map_view.camera.pan_by(total_dx, total_dy);
            map_view.dragging_from = Some(current);
            return map_view.should_redraw_interaction(min_interval);
        }

        let dx = current.x - from.x;
        let dy = current.y - from.y;
        map_view.camera.pan_by(dx, dy);
        map_view.dragging_from = Some(current);
        map_view.should_redraw_interaction(min_interval)
    }

    pub(super) fn end_map_drag(&mut self, map_kind: MapCanvasKind) -> MapInteractionEnd {
        let map_view = self.map_view_mut(map_kind);
        let had_pointer = map_view.dragging_from.take().is_some();
        let outcome = MapInteractionEnd {
            redraw: had_pointer && map_view.drag_moved,
            clicked: had_pointer && !map_view.drag_moved,
        };
        map_view.drag_origin = None;
        map_view.drag_moved = false;
        map_view.reset_interaction_redraw();
        outcome
    }

    pub(super) fn zoom_map_canvas(
        &mut self,
        map_kind: MapCanvasKind,
        anchor_x: f32,
        anchor_y: f32,
        delta: f32,
    ) {
        let map_view = self.map_view_mut(map_kind);
        map_view.reset_interaction_redraw();
        map_view.camera.zoom_at(anchor_x, anchor_y, delta);
    }

    pub(super) fn preview_selected_point_move(&mut self, screen_x: f32, screen_y: f32) -> bool {
        let _ = (screen_x, screen_y);
        false
    }

    pub(super) fn handle_route_map_click(
        &mut self,
        map_kind: MapCanvasKind,
        screen_x: f32,
        screen_y: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_selected_point_move_armed() {
            return self.confirm_selected_point_move(map_kind, screen_x, screen_y, window, cx);
        }

        if let Some(point_id) = self.point_hit_test(map_kind, screen_x, screen_y) {
            if map_kind == MapCanvasKind::RouteEditor && self.route_editor_draw_mode {
                return self.route_editor_append_draw_point(point_id, window, cx);
            }
            self.select_point(point_id, window, cx);
            return true;
        }

        if map_kind == MapCanvasKind::RouteEditor && self.route_editor_draw_mode {
            self.route_editor_draw_sequence.clear();
            self.status_text = "已清除当前连线起点，请重新点击节点开始连线。".into();
            return true;
        }

        if !self.map_point_insert_armed && self.selected_point_id.is_some() {
            self.clear_route_editor_point_selection(window, cx);
            self.status_text = "已取消节点选中。".into();
            return true;
        }

        if self.active_page != WorkbenchPage::Markers {
            return false;
        }

        if !self.map_point_insert_armed {
            self.status_text = "已点击空白地图。若要新建节点，请先点击“添加节点”。".into();
            return false;
        }

        self.insert_point_from_map_click(
            self.map_view(map_kind)
                .camera
                .screen_to_world(WorldPoint::new(screen_x, screen_y)),
            window,
            cx,
        );
        true
    }

    pub(super) fn toggle_selected_point_move_mode(&mut self) {
        if self.route_editor_draw_mode {
            self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
            return;
        }
        if self.is_selected_point_move_armed() {
            self.clear_selected_point_move_state();
            self.status_text = "已取消取点。".into();
            return;
        }

        let Some(point_id) = self.selected_point_id.clone() else {
            self.status_text = "请先选择一个节点，再进入移动状态。".into();
            return;
        };
        if self.selected_point().is_none() {
            self.status_text = "当前节点不存在，无法移动。".into();
            return;
        }

        self.map_point_insert_armed = false;
        self.reset_route_editor_draw_state();
        self.confirming_delete_point_id = None;
        self.moving_point_id = Some(point_id);
        self.moving_point_preview = None;
        self.ignore_next_tracker_mouse_up = true;
        self.status_text = "取点已开启：点击地图确认新位置。".into();
    }

    fn confirm_selected_point_move(
        &mut self,
        map_kind: MapCanvasKind,
        screen_x: f32,
        screen_y: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.clear_selected_point_move_state();
            self.status_text = "请先选择一条路线。".into();
            return false;
        };
        let Some(point_id) = self.selected_point_id.clone() else {
            self.clear_selected_point_move_state();
            self.status_text = "请先选择一个节点。".into();
            return false;
        };

        let world = self.clamp_tracker_world(
            self.map_view(map_kind)
                .camera
                .screen_to_world(WorldPoint::new(screen_x, screen_y)),
        );

        let mut moved_label = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            if let Some(point) = group.find_point_mut(&point_id) {
                point.x = world.x;
                point.y = world.y;
                moved_label = Some(point.display_label().to_owned());
            }
        }

        let Some(label) = moved_label else {
            self.clear_selected_point_move_state();
            self.status_text = "选中的节点不存在，无法移动。".into();
            return false;
        };

        self.clear_selected_point_move_state();
        self.preview_cursor = self.selected_point_index();
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        self.suspend_point_autosave = true;
        set_input_value(&self.marker_form.x, format!("{:.0}", world.x), window, cx);
        set_input_value(&self.marker_form.y, format!("{:.0}", world.y), window, cx);
        self.suspend_point_autosave = false;

        if self.persist_group(&group_id, &format!("节点「{label}」位置已更新")) {
            self.sync_editor_from_selection(window, cx);
        }
        true
    }

    fn select_map_page(&mut self, page: MapPage) {
        if self.route_editor_draw_mode {
            self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
            return;
        }
        self.active_page = WorkbenchPage::Map;
        self.map_page = page;
        self.map_point_insert_armed = false;
        self.reset_route_editor_draw_state();
        self.route_editor_selected_point_ids.clear();
        self.clear_selected_point_move_state();
        if matches!(page, MapPage::Tracker) {
            self.ensure_tracker_page_selected_point();
            self.request_center_on_current_point();
        }
        self.sync_tracker_debug_enabled();
        self.status_text = format!("已切换到{}。", page).into();
    }

    pub(super) fn select_routes_page(&mut self) {
        if self.route_editor_draw_mode {
            self.status_text = "当前已经处于连线编辑模式。".into();
            return;
        }
        self.active_page = WorkbenchPage::Markers;
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.route_editor_map_view.request_fit();
        self.sync_tracker_debug_enabled();
        self.status_text = "已切换到路线管理。".into();
    }

    fn select_settings_page(&mut self, page: SettingsPage) {
        if self.route_editor_draw_mode {
            self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
            return;
        }
        self.active_page = WorkbenchPage::Settings;
        self.settings_page = page;
        self.settings_nav_expanded = true;
        self.map_point_insert_armed = false;
        self.reset_route_editor_draw_state();
        self.route_editor_selected_point_ids.clear();
        self.clear_selected_point_move_state();
        self.sync_tracker_debug_enabled();
        info!(page = %page, "selected settings page");
        self.status_text = format!("设置页面已切换到{}。", page).into();
    }

    pub(super) fn toggle_map_point_insert_mode(&mut self) {
        if self.route_editor_draw_mode {
            self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
            return;
        }
        if self.selected_group_id.is_none() {
            self.status_text = "请先选择一条路线，再开启添加节点。".into();
            return;
        }

        self.map_point_insert_armed = !self.map_point_insert_armed;
        if self.map_point_insert_armed {
            self.reset_route_editor_draw_state();
            self.clear_selected_point_move_state();
            self.route_editor_selected_point_ids.clear();
            self.selected_point_id = None;
            self.confirming_delete_point_id = None;
        }
        self.status_text = if self.map_point_insert_armed {
            "添加节点已开启：下一次点击空白地图会插入新节点。".into()
        } else {
            "添加节点已取消。".into()
        };
    }

    fn toggle_settings_navigation(&mut self) {
        if self.active_page != WorkbenchPage::Settings {
            if self.route_editor_draw_mode {
                self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
                return;
            }
            self.active_page = WorkbenchPage::Settings;
            self.settings_nav_expanded = true;
            self.sync_tracker_debug_enabled();
            info!("switched to settings page from navigation toggle");
            self.status_text = "已切换到设置页面。".into();
            return;
        }

        self.settings_nav_expanded = !self.settings_nav_expanded;
        self.sync_tracker_debug_enabled();
        debug!(
            expanded = self.settings_nav_expanded,
            "toggled settings navigation"
        );
        self.status_text = if self.settings_nav_expanded {
            "已展开设置导航。".into()
        } else {
            "已收起设置导航。".into()
        };
    }

    fn has_busy_operation(&self) -> bool {
        self.is_tracker_transition_pending()
            || self.is_bwiki_refreshing()
            || self.is_bwiki_planner_busy()
            || self.is_route_import_busy()
    }

    fn poll_runtime_logs(&mut self) -> bool {
        let revision = logging::debug_log_revision();
        let changed = revision != self.debug_log_revision;
        self.debug_log_revision = revision;
        changed
            && self.debug_mode_enabled
            && self.active_page == WorkbenchPage::Settings
            && self.settings_page == SettingsPage::Debug
    }

    fn tick_busy_indicator(&mut self) -> bool {
        if self.has_busy_operation() {
            self.spinner_frame = self.spinner_frame.wrapping_add(1);
            return true;
        }
        if self.spinner_frame != 0 {
            self.spinner_frame = 0;
            return true;
        }
        false
    }

    fn poll_bwiki_resources(&mut self) -> bool {
        let version = self.bwiki_resources.version();
        let mut changed = version != self.bwiki_version;
        if changed {
            self.bwiki_version = version;
        }
        if self.sync_bwiki_visibility_defaults() {
            changed = true;
        }
        if changed && self.bwiki_planner_active {
            self.invalidate_bwiki_route_plan_preview();
        }
        changed
    }

    pub(super) fn sync_bwiki_visibility_defaults(&mut self) -> bool {
        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return false;
        };

        let mut changed = false;
        let valid_categories = dataset
            .sorted_category_names()
            .into_iter()
            .collect::<HashSet<_>>();
        let expanded_before = self.bwiki_expanded_categories.len();
        self.bwiki_expanded_categories
            .retain(|category| valid_categories.contains(category));
        if self.bwiki_expanded_categories.len() != expanded_before {
            changed = true;
        }

        let valid_mark_types = dataset
            .types
            .iter()
            .filter(|item| item.point_count > 0)
            .map(|item| item.mark_type)
            .collect::<HashSet<_>>();
        let visible_before = self.bwiki_visible_mark_types.len();
        self.bwiki_visible_mark_types
            .retain(|mark_type| valid_mark_types.contains(mark_type));
        if self.bwiki_visible_mark_types.len() != visible_before {
            changed = true;
        }

        if !self.bwiki_visibility_initialized {
            self.bwiki_visibility_initialized = true;
            changed = true;
        }

        changed
    }

    pub(super) fn refresh_bwiki_dataset(&mut self) {
        info!("requested BWiki dataset refresh from UI");
        self.bwiki_resources.refresh_dataset();
        self.status_text = "已请求刷新 BWiki 点位与图标目录。".into();
    }

    pub(super) fn show_all_bwiki_types(&mut self) {
        if let Some(dataset) = self.bwiki_resources.dataset_snapshot() {
            self.bwiki_visibility_initialized = true;
            self.bwiki_visible_mark_types = dataset
                .types
                .iter()
                .filter(|item| item.point_count > 0)
                .map(|item| item.mark_type)
                .collect();
            self.status_text = "已显示所有有点位的 BWiki 类型。".into();
        }
    }

    pub(super) fn hide_all_bwiki_types(&mut self) {
        self.bwiki_visibility_initialized = true;
        self.bwiki_visible_mark_types.clear();
        self.status_text = "已隐藏所有 BWiki 类型。".into();
    }

    pub(super) fn toggle_bwiki_type_visibility(&mut self, mark_type: u32, label: &str) {
        self.bwiki_visibility_initialized = true;
        if !self.bwiki_visible_mark_types.remove(&mark_type) {
            self.bwiki_visible_mark_types.insert(mark_type);
            self.status_text = format!("已显示 BWiki 类型「{}」。", label).into();
        } else {
            self.status_text = format!("已隐藏 BWiki 类型「{}」。", label).into();
        }
    }

    pub(super) fn set_bwiki_category_visibility(&mut self, category: &str, visible: bool) {
        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return;
        };

        self.bwiki_visibility_initialized = true;
        let mut affected = 0usize;
        for definition in dataset
            .types
            .iter()
            .filter(|item| item.category == category)
        {
            if definition.point_count == 0 {
                continue;
            }
            affected += 1;
            if visible {
                self.bwiki_visible_mark_types.insert(definition.mark_type);
            } else {
                self.bwiki_visible_mark_types.remove(&definition.mark_type);
            }
        }

        self.status_text = if visible {
            format!("已显示分类「{}」下的 {} 个类型。", category, affected).into()
        } else {
            format!("已隐藏分类「{}」下的 {} 个类型。", category, affected).into()
        };
    }

    pub(super) fn toggle_bwiki_category_expanded(&mut self, category: &str) {
        if !self.bwiki_expanded_categories.remove(category) {
            self.bwiki_expanded_categories.insert(category.to_owned());
        }
    }

    fn ensure_bwiki_planner_defaults(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if read_input_value(&self.bwiki_planner_form.name, cx)
            .trim()
            .is_empty()
        {
            set_input_value(&self.bwiki_planner_form.name, "新规划路线", window, cx);
        }
        if read_input_value(&self.bwiki_planner_form.color_hex, cx)
            .trim()
            .is_empty()
        {
            set_input_value(&self.bwiki_planner_form.color_hex, "#FF6B6B", window, cx);
        }
        self.bwiki_planner_icon = MarkerIconStyle::default();
        self.bwiki_planner_icon_picker.update(cx, |picker, cx| {
            picker.set_selected_value(&self.bwiki_planner_icon, window, cx);
        });
    }

    pub(super) fn clear_bwiki_planner_selection(&mut self) {
        self.bwiki_planner_selected_points.clear();
        self.bwiki_planner_lasso_selection = None;
        self.invalidate_bwiki_route_plan_preview();
    }

    pub(super) fn toggle_bwiki_planner_mode(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.bwiki_planner_active {
            self.bwiki_planner_active = false;
            self.clear_bwiki_planner_selection();
            self.status_text = "已退出路线规划模式。".into();
            return;
        }

        self.bwiki_planner_active = true;
        self.clear_bwiki_planner_selection();
        self.ensure_bwiki_planner_defaults(window, cx);
        self.bwiki_planner_status = AsyncTaskStatus::idle("选择节点后可在后台规划路线。");
        self.status_text =
            "已进入路线规划模式。单击节点切换选中，按住 Ctrl 左键手绘闭合曲线可反选。".into();
    }

    fn begin_bwiki_route_plan_task(&mut self, selected_count: usize) -> u64 {
        self.bwiki_planner_task_id = self.bwiki_planner_task_id.wrapping_add(1);
        self.bwiki_planner_preview = None;
        let message = format!("正在规划 {} 个节点，请稍候。", selected_count);
        self.bwiki_planner_status = AsyncTaskStatus::working(message.clone());
        self.status_text = message.into();
        self.bwiki_planner_task_id
    }

    fn apply_bwiki_route_plan_task_result(&mut self, task_id: u64, result: BwikiPlannerTaskResult) {
        if task_id != self.bwiki_planner_task_id {
            return;
        }

        let resolved_count = result.normalized_selection_keys.len();
        self.bwiki_planner_selected_points = result.normalized_selection_keys;

        match result.preview {
            Some(preview) => {
                let total_cost = preview.total_cost;
                self.bwiki_planner_preview = Some(preview);
                let message = if resolved_count == 1 {
                    "当前只选中了 1 个节点，已生成单点路线。".to_owned()
                } else {
                    format!(
                        "已完成路线规划，共 {} 个节点，预计长度 {:.0}。",
                        resolved_count, total_cost
                    )
                };
                self.bwiki_planner_status = AsyncTaskStatus::succeeded(message.clone());
                self.status_text = message.into();
            }
            None => {
                self.bwiki_planner_preview = None;
                let message = result.failure_message.unwrap_or_else(|| {
                    if result.requested_count > 0 {
                        "当前选中的节点已失效，请重新选择后再规划。".to_owned()
                    } else {
                        "请先在地图中选择至少一个节点。".to_owned()
                    }
                });
                self.bwiki_planner_status = AsyncTaskStatus::idle(message.clone());
                self.status_text = message.into();
            }
        }
    }

    pub(super) fn plan_bwiki_route_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_bwiki_planner_busy() {
            return;
        }

        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            self.bwiki_planner_preview = None;
            let message = "节点图鉴数据尚未就绪，暂时无法规划。".to_owned();
            self.bwiki_planner_status = AsyncTaskStatus::failed(message.clone());
            self.status_text = message.into();
            return;
        };

        let mut selected_keys = self
            .bwiki_planner_selected_points
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        selected_keys.sort();
        if selected_keys.is_empty() {
            self.invalidate_bwiki_route_plan_preview();
            let message = "请先在地图中选择至少一个节点。".to_owned();
            self.bwiki_planner_status = AsyncTaskStatus::idle(message.clone());
            self.status_text = message.into();
            return;
        }

        let teleport_link_distance = self.teleport_link_distance();
        let task_id = self.begin_bwiki_route_plan_task(selected_keys.len());

        cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_executor().spawn(async move {
                build_bwiki_route_plan_task_result(&dataset, selected_keys, teleport_link_distance)
            });

            let result = result.await;

            this.update_in(cx, |this, _, cx| {
                this.apply_bwiki_route_plan_task_result(task_id, result);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub(super) fn handle_bwiki_planner_click(&mut self, screen_x: f32, screen_y: f32) -> bool {
        if !self.bwiki_planner_active {
            return false;
        }

        let Some(key) = self.bwiki_point_hit_test(screen_x, screen_y) else {
            return false;
        };
        let selecting = !self.bwiki_planner_selected_points.contains(&key);
        self.toggle_bwiki_planner_keys([key]);
        self.status_text = if selecting {
            format!(
                "已选中 {} 个规划点。若要生成路线，请点击“规划路线”。",
                self.bwiki_planner_selected_count()
            )
            .into()
        } else {
            format!(
                "已取消 1 个规划点，当前还剩 {} 个。若要更新路线，请重新点击“规划路线”。",
                self.bwiki_planner_selected_count()
            )
            .into()
        };
        true
    }

    pub(super) fn begin_bwiki_planner_lasso_selection(&mut self, screen_x: f32, screen_y: f32) {
        let anchor = WorldPoint::new(screen_x, screen_y);
        self.bwiki_planner_lasso_selection = Some(BwikiPlannerLassoSelection::new(anchor));
    }

    pub(super) fn update_bwiki_planner_lasso_selection(
        &mut self,
        screen_x: f32,
        screen_y: f32,
    ) -> bool {
        let Some(selection) = self.bwiki_planner_lasso_selection.as_mut() else {
            return false;
        };
        selection.push_point(WorldPoint::new(screen_x, screen_y))
    }

    pub(super) fn finish_bwiki_planner_lasso_selection(
        &mut self,
        screen_x: f32,
        screen_y: f32,
    ) -> bool {
        let Some(mut selection) = self.bwiki_planner_lasso_selection.take() else {
            return false;
        };
        selection.push_point(WorldPoint::new(screen_x, screen_y));
        if selection.travel_distance() < MAP_CLICK_DRAG_THRESHOLD {
            return self.handle_bwiki_planner_click(screen_x, screen_y);
        }

        let affected_keys = self.bwiki_points_in_lasso(&selection.path);
        let affected = self.toggle_bwiki_planner_keys(affected_keys);
        self.status_text = if affected == 0 {
            "当前圈选范围内没有可切换的节点。".into()
        } else {
            format!(
                "已反选 {} 个节点，当前已选 {} 个规划点。若要更新路线，请重新点击“规划路线”。",
                affected,
                self.bwiki_planner_selected_count()
            )
            .into()
        };
        true
    }

    pub(super) fn create_route_from_bwiki_planner(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.bwiki_planner_active {
            return;
        }

        let draft = match PlannerRouteDraft::read(self, cx) {
            Ok(draft) => draft,
            Err(message) => {
                self.status_text = message.into();
                return;
            }
        };

        let Some(_) = self.bwiki_planner_preview.as_ref() else {
            self.status_text = "请先点击“规划路线”，确认预览结果后再创建路线。".into();
            return;
        };
        let preview = self
            .bwiki_planner_preview
            .as_ref()
            .expect("preview checked above");
        if !bwiki_route_keys_form_simple_path(&preview.route_keys) {
            self.status_text = "当前规划结果包含重复节点，无法生成单行路线，请重新规划。".into();
            return;
        }
        if !preview
            .route_keys
            .first()
            .is_some_and(|key| key.mark_type == BWIKI_TELEPORT_MARK_TYPE)
        {
            self.status_text = "当前规划结果不是从传送点开始，请重新规划。".into();
            return;
        }

        let planned_points = self.resolve_bwiki_preview_points();
        if planned_points.is_empty() {
            self.status_text = "当前没有可创建的规划结果，请重新规划。".into();
            return;
        }

        let file_name = self.allocate_group_file_name();
        let mut route = RouteDocument::new(draft.name.clone());
        route.notes = draft.description.clone();
        route.visible = true;
        route.default_style = draft.style.clone();
        route.points = planned_points
            .iter()
            .cloned()
            .map(|point| planner_point_to_route_point(point, &draft.style))
            .collect();
        route.id = RouteId(file_name.clone());
        route.metadata = RouteMetadata {
            id: route.id.clone(),
            file_name,
            display_name: route.display_name().to_owned(),
        };

        let group_id = route.id.clone();
        let created_point_count = route.points.len();
        self.route_groups.push(route);
        self.sync_workspace_routes_snapshot();
        if !self.persist_group(&group_id, "规划路线已保存") {
            self.route_groups.retain(|group| group.id != group_id);
            self.sync_workspace_routes_snapshot();
            return;
        }

        self.bwiki_planner_active = false;
        self.clear_bwiki_planner_selection();
        self.select_routes_page();
        self.select_group(group_id, window, cx);
        self.status_text = format!(
            "已根据 {} 个节点创建路线「{}」。",
            created_point_count, draft.name
        )
        .into();
    }

    fn paged_list_state(&self, kind: PagedListKind) -> &PagedListState {
        match kind {
            PagedListKind::MapGroups => &self.map_group_list,
            PagedListKind::MarkerGroups => &self.marker_group_list,
            PagedListKind::Points => &self.point_list,
        }
    }

    fn paged_list_state_mut(&mut self, kind: PagedListKind) -> &mut PagedListState {
        match kind {
            PagedListKind::MapGroups => &mut self.map_group_list,
            PagedListKind::MarkerGroups => &mut self.marker_group_list,
            PagedListKind::Points => &mut self.point_list,
        }
    }

    fn paged_list_filtered_count(&self, kind: PagedListKind, cx: &mut Context<Self>) -> usize {
        match kind {
            PagedListKind::MapGroups => {
                let query = normalized_list_query(&self.map_group_list.search, cx);
                self.route_groups
                    .iter()
                    .filter(|group| {
                        query_matches(
                            &query,
                            [
                                group.display_name().to_owned(),
                                group.notes.clone(),
                                group.metadata.file_name.clone(),
                            ],
                        )
                    })
                    .count()
            }
            PagedListKind::MarkerGroups => {
                let query = normalized_list_query(&self.marker_group_list.search, cx);
                self.route_groups
                    .iter()
                    .filter(|group| {
                        query_matches(
                            &query,
                            [
                                group.display_name().to_owned(),
                                group.notes.clone(),
                                group.metadata.file_name.clone(),
                            ],
                        )
                    })
                    .count()
            }
            PagedListKind::Points => {
                let query = normalized_list_query(&self.point_list.search, cx);
                self.active_group()
                    .map(|group| {
                        group
                            .points
                            .iter()
                            .filter(|point| {
                                query_matches(
                                    &query,
                                    [
                                        point.display_label().to_owned(),
                                        point.note.clone(),
                                        format!("{:.0}", point.x),
                                        format!("{:.0}", point.y),
                                    ],
                                )
                            })
                            .count()
                    })
                    .unwrap_or(0)
            }
        }
    }

    fn paged_list_page_count(&self, kind: PagedListKind, cx: &mut Context<Self>) -> usize {
        let page_size = self.paged_list_state(kind).page_size.max(1);
        self.paged_list_filtered_count(kind, cx)
            .max(1)
            .div_ceil(page_size)
    }

    fn filtered_group_position(
        &self,
        kind: PagedListKind,
        target_group_id: &RouteId,
        cx: &mut Context<Self>,
    ) -> Option<usize> {
        let query = match kind {
            PagedListKind::MapGroups => normalized_list_query(&self.map_group_list.search, cx),
            PagedListKind::MarkerGroups => {
                normalized_list_query(&self.marker_group_list.search, cx)
            }
            PagedListKind::Points => return None,
        };

        self.route_groups
            .iter()
            .filter(|group| {
                query_matches(
                    &query,
                    [
                        group.display_name().to_owned(),
                        group.notes.clone(),
                        group.metadata.file_name.clone(),
                    ],
                )
            })
            .position(|group| &group.id == target_group_id)
    }

    fn filtered_point_position(
        &self,
        target_point_id: &RoutePointId,
        cx: &mut Context<Self>,
    ) -> Option<usize> {
        let query = normalized_list_query(&self.point_list.search, cx);
        self.active_group()?
            .points
            .iter()
            .filter(|point| {
                query_matches(
                    &query,
                    [
                        point.display_label().to_owned(),
                        point.note.clone(),
                        format!("{:.0}", point.x),
                        format!("{:.0}", point.y),
                    ],
                )
            })
            .position(|point| &point.id == target_point_id)
    }

    fn set_paged_list_page(
        &mut self,
        kind: PagedListKind,
        page: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let last_page = self.paged_list_page_count(kind, cx).saturating_sub(1);
        let page = page.min(last_page);
        let page_input = {
            let state = self.paged_list_state_mut(kind);
            state.page = page;
            state.page_input.clone()
        };
        set_input_value(&page_input, (page + 1).to_string(), window, cx);
    }

    fn reset_paged_list_page(
        &mut self,
        kind: PagedListKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_paged_list_page(kind, 0, window, cx);
    }

    fn jump_paged_list_from_input(
        &mut self,
        kind: PagedListKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let input = read_input_value(&self.paged_list_state(kind).page_input, cx);
        let Some(page) = input.trim().parse::<usize>().ok().filter(|page| *page > 0) else {
            let current_page = self.paged_list_state(kind).page + 1;
            let page_input = self.paged_list_state(kind).page_input.clone();
            set_input_value(&page_input, current_page.to_string(), window, cx);
            self.status_text = "页码必须是大于等于 1 的整数。".into();
            return;
        };

        self.set_paged_list_page(kind, page - 1, window, cx);
    }

    fn set_map_group_page(&mut self, page: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.set_paged_list_page(PagedListKind::MapGroups, page, window, cx);
    }

    fn set_marker_group_page(&mut self, page: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.set_paged_list_page(PagedListKind::MarkerGroups, page, window, cx);
    }

    fn jump_map_group_page_from_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.jump_paged_list_from_input(PagedListKind::MapGroups, window, cx);
    }

    fn jump_marker_group_page_from_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.jump_paged_list_from_input(PagedListKind::MarkerGroups, window, cx);
    }

    fn jump_point_page_from_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.jump_paged_list_from_input(PagedListKind::Points, window, cx);
    }

    fn defer_marker_group_page_to_group(
        &mut self,
        target_group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.defer_in(window, move |this, window, cx| {
            let target_page = this
                .filtered_group_position(PagedListKind::MarkerGroups, &target_group_id, cx)
                .map(|index| index / this.marker_group_list.page_size.max(1))
                .unwrap_or_else(|| {
                    this.paged_list_page_count(PagedListKind::MarkerGroups, cx)
                        .saturating_sub(1)
                });
            this.set_marker_group_page(target_page, window, cx);
            cx.notify();
        });
    }

    pub(super) fn is_group_being_edited(&self, group_id: &RouteId) -> bool {
        self.editing_group_id.as_ref() == Some(group_id)
    }

    pub(super) fn is_group_delete_confirmation_active(&self, group_id: &RouteId) -> bool {
        self.confirming_delete_group_id.as_ref() == Some(group_id)
    }

    pub(super) fn start_group_inline_edit(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group) = self.route_groups.iter().find(|group| group.id == group_id) else {
            self.status_text = "待编辑的路线不存在。".into();
            return;
        };

        self.selected_group_id = Some(group_id.clone());
        self.editing_group_id = Some(group_id);
        self.confirming_delete_group_id = None;
        set_input_value(&self.group_inline_edit.name, group.name.clone(), window, cx);
        set_input_value(
            &self.group_inline_edit.description,
            group.notes.clone(),
            window,
            cx,
        );
        self.sync_editor_from_selection(window, cx);
        self.status_text = "已进入路线行内编辑。".into();
    }

    pub(super) fn create_group_inline_item(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        set_input_value(&self.marker_group_list.search, "", window, cx);
        self.marker_group_list.page = 0;

        let mut group = RouteDocument::new("");
        group.notes.clear();
        group.default_style = MarkerStyle::default();
        let file_name = self.allocate_group_file_name();
        group.id = RouteId(file_name.clone());
        group.metadata = RouteMetadata {
            id: group.id.clone(),
            file_name,
            display_name: String::new(),
        };
        let group_id = group.id.clone();
        self.route_groups.push(group);
        self.sync_workspace_routes_snapshot();

        self.selected_group_id = Some(group_id.clone());
        self.selected_point_id = None;
        self.preview_cursor = None;
        self.pending_new_group_id = Some(group_id.clone());
        self.confirming_delete_group_id = None;
        self.confirming_delete_point_id = None;
        self.sync_editor_from_selection(window, cx);
        self.start_group_inline_edit(group_id, window, cx);
        self.defer_marker_group_page_to_group(
            self.selected_group_id
                .clone()
                .expect("new group should be selected"),
            window,
            cx,
        );
        self.status_text = "已创建新的空白路线，请先填写标题和说明。".into();
    }

    pub(super) fn commit_inline_group_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.editing_group_id.clone() else {
            return;
        };

        let name = read_input_value(&self.group_inline_edit.name, cx);
        if name.trim().is_empty() {
            self.status_text = "路线标题不能为空。".into();
            return;
        }
        let description = read_input_value(&self.group_inline_edit.description, cx);

        let needs_file_name = {
            let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            else {
                self.editing_group_id = None;
                self.pending_new_group_id = None;
                self.status_text = "待保存的路线不存在。".into();
                return;
            };

            group.name = name;
            group.notes = description;
            group.metadata.display_name = group.display_name().to_owned();
            group.metadata.file_name.trim().is_empty()
        };

        if needs_file_name {
            let file_name = self.allocate_group_file_name();
            if let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            {
                group.id = RouteId(file_name.clone());
                group.metadata.id = group.id.clone();
                group.metadata.file_name = file_name;
            }
        }

        if self.persist_group(&group_id, "路线已保存") {
            self.editing_group_id = None;
            if self.pending_new_group_id.as_ref() == Some(&group_id) {
                self.pending_new_group_id = None;
            }
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn cancel_inline_group_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.editing_group_id.clone() else {
            return;
        };

        if self.pending_new_group_id.as_ref() == Some(&group_id) {
            self.editing_group_id = None;
            self.pending_new_group_id = None;
            self.delete_group_by_id(group_id, false, window, cx);
            self.status_text = "已取消新建路线。".into();
            return;
        }

        self.editing_group_id = None;
        if let Some(group) = self.route_groups.iter().find(|group| group.id == group_id) {
            set_input_value(&self.group_inline_edit.name, group.name.clone(), window, cx);
            set_input_value(
                &self.group_inline_edit.description,
                group.notes.clone(),
                window,
                cx,
            );
        }
        self.status_text = "已取消路线行内编辑。".into();
    }

    pub(super) fn begin_group_delete_confirmation(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_name) = self
            .route_groups
            .iter()
            .find(|group| group.id == group_id)
            .map(|group| group.display_name().to_owned())
        else {
            self.confirming_delete_group_id = None;
            self.status_text = "待删除的路线不存在。".into();
            return;
        };

        if self
            .filtered_group_position(PagedListKind::MarkerGroups, &group_id, cx)
            .is_none()
        {
            set_input_value(&self.marker_group_list.search, "", window, cx);
            self.marker_group_list.page = 0;
        }

        self.selected_group_id = Some(group_id.clone());
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;
        self.preview_cursor = None;
        if self.editing_group_id.as_ref() == Some(&group_id) {
            self.editing_group_id = None;
        }
        self.confirming_delete_group_id = Some(group_id);
        self.sync_editor_from_selection(window, cx);
        self.defer_marker_group_page_to_group(
            self.selected_group_id
                .clone()
                .expect("confirmed group should stay selected"),
            window,
            cx,
        );
        self.status_text = format!("请确认是否删除路线「{}」。", group_name).into();
    }

    pub(super) fn cancel_group_delete_confirmation(&mut self, group_id: RouteId) {
        if self.confirming_delete_group_id.as_ref() == Some(&group_id) {
            self.confirming_delete_group_id = None;
            self.status_text = "已取消删除路线。".into();
        }
    }

    pub(super) fn confirm_group_delete(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.confirming_delete_group_id = None;
        self.delete_group_by_id(group_id, true, window, cx);
    }

    pub(super) fn delete_group_by_id(
        &mut self,
        group_id: RouteId,
        delete_persisted_file: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(index) = self
            .route_groups
            .iter()
            .position(|group| group.id == group_id)
        else {
            self.status_text = "选中的路线不存在。".into();
            return;
        };

        let removed = self.route_groups[index].clone();
        let removed_path = self.route_file_path(&removed.metadata.file_name);
        let removed_file_exists = removed_path.exists();
        if delete_persisted_file {
            if removed_file_exists && let Err(error) = RouteRepository::delete(&removed_path) {
                self.status_text = format!("删除路线文件失败：{error:#}").into();
                return;
            }
        }

        self.route_groups.remove(index);
        self.sync_workspace_routes_snapshot();
        if self.editing_group_id.as_ref() == Some(&group_id) {
            self.editing_group_id = None;
        }
        if self.pending_new_group_id.as_ref() == Some(&group_id) {
            self.pending_new_group_id = None;
        }
        if self.confirming_delete_group_id.as_ref() == Some(&group_id) {
            self.confirming_delete_group_id = None;
        }

        self.selected_group_id = self.route_groups.first().map(|group| group.id.clone());
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;

        self.sync_editor_from_selection(window, cx);
        self.status_text = if delete_persisted_file && removed_file_exists {
            format!("路线「{}」已删除。", removed.display_name()).into()
        } else {
            format!("已移除未保存的路线占位「{}」。", removed.display_name()).into()
        };
    }

    fn marker_group_picker_items(&self) -> Vec<MarkerGroupPickerItem> {
        self.route_groups
            .iter()
            .map(|group| {
                let title = group.display_name().to_owned();
                let subtitle = if group.notes.trim().is_empty() {
                    group.metadata.file_name.clone()
                } else {
                    format!("{} · {}", group.metadata.file_name, group.notes)
                };

                MarkerGroupPickerItem::new(
                    group.id.clone(),
                    title.clone(),
                    subtitle.clone(),
                    format!("{title} {} {}", group.metadata.file_name, group.notes),
                )
            })
            .collect::<Vec<_>>()
    }

    fn bwiki_icon_picker_items(&self) -> Vec<BwikiIconPickerItem> {
        let items = self
            .bwiki_resources
            .dataset_snapshot()
            .map(|dataset| {
                dataset
                    .types
                    .iter()
                    .filter(|definition| !definition.icon_url.trim().is_empty())
                    .map(|definition| {
                        let title = definition.name.clone();
                        let subtitle = format!(
                            "{} · {} · {} 个点位",
                            definition.category, definition.mark_type, definition.point_count
                        );
                        BwikiIconPickerItem::new(
                            MarkerIconStyle::new(title.clone()),
                            title.clone(),
                            subtitle.clone(),
                            format!(
                                "{} {} {} {}",
                                title,
                                definition.category,
                                definition.mark_type,
                                definition.point_count
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        items
    }

    fn sync_marker_group_picker_state(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let items = self.marker_group_picker_items();
        let selected_group_id = self.selected_group_id.clone();
        self.marker_group_picker.update(cx, |picker, cx| {
            picker.set_items(items, window, cx);
            if let Some(group_id) = selected_group_id.as_ref() {
                picker.set_selected_value(group_id, window, cx);
            } else {
                picker.set_selected_index(None, window, cx);
            }
        });
    }

    pub(super) fn sync_bwiki_icon_picker_state(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let group_items = self.bwiki_icon_picker_items();
        let planner_items = self.bwiki_icon_picker_items();
        let marker_items = self.bwiki_icon_picker_items();
        let group_icon = self.group_icon.clone();
        let planner_icon = self.bwiki_planner_icon.clone();
        let marker_icon = self.marker_icon.clone();

        self.group_icon_picker.update(cx, |picker, cx| {
            picker.set_items(group_items, window, cx);
            picker.set_selected_value(&group_icon, window, cx);
        });
        self.bwiki_planner_icon_picker.update(cx, |picker, cx| {
            picker.set_items(planner_items, window, cx);
            picker.set_selected_value(&planner_icon, window, cx);
        });
        self.marker_icon_picker.update(cx, |picker, cx| {
            picker.set_items(marker_items, window, cx);
            picker.set_selected_value(&marker_icon, window, cx);
        });
        self.bwiki_icon_picker_version = self.bwiki_version;
    }

    pub(super) fn set_theme_preference(
        &mut self,
        preference: ThemePreference,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.theme_preference = preference;
        apply_theme_preference(self.theme_preference, window, cx);
        info!(theme = %self.theme_preference, "updated theme preference");
        self.persist_ui_preferences(&format!("界面主题已切换为 {}", self.theme_preference));
    }

    pub(super) fn set_auto_focus_enabled(&mut self, enabled: bool) {
        self.auto_focus_enabled = enabled;
        info!(enabled, "updated auto-focus preference");
        if enabled {
            self.request_center_on_current_point();
            self.persist_ui_preferences("自动聚焦已开启");
        } else {
            self.tracker_map_view.pending_center = None;
            self.persist_ui_preferences("自动聚焦已关闭");
        }
    }

    pub(super) fn set_tracker_point_popup_enabled(&mut self, enabled: bool) {
        self.tracker_point_popup_enabled = enabled;
        info!(enabled, "updated tracker point popup preference");
        self.persist_ui_preferences(if enabled {
            "节点浮窗已开启"
        } else {
            "节点浮窗已关闭"
        });
    }

    fn release_debug_snapshot_render_images_in_app(&mut self, cx: &mut App) {
        for image in self.debug_snapshot_render_images.drain(..).flatten() {
            cx.drop_image(image, None);
        }
    }

    fn clear_debug_snapshot_render_images(&mut self, cx: &mut Context<Self>) {
        self.release_debug_snapshot_render_images_in_app(cx);
    }

    fn clear_debug_snapshot(&mut self, cx: &mut Context<Self>) {
        self.debug_snapshot = None;
        self.clear_debug_snapshot_render_images(cx);
    }

    fn set_debug_snapshot(&mut self, snapshot: TrackingDebugSnapshot, cx: &mut Context<Self>) {
        self.clear_debug_snapshot_render_images(cx);
        self.debug_snapshot_render_images = snapshot
            .images
            .iter()
            .map(render_image_from_debug_image)
            .collect();
        self.debug_snapshot = Some(snapshot);
    }

    pub(super) fn set_debug_mode_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug_mode_enabled = enabled;
        if !enabled {
            self.clear_debug_snapshot(cx);
        }
        self.sync_tracker_debug_enabled();
        info!(enabled, "updated debug mode preference");
        self.persist_ui_preferences(if enabled {
            "调试模式已开启"
        } else {
            "调试模式已关闭"
        });
    }

    pub(super) fn set_test_case_capture_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.test_case_capture_enabled = enabled;
        if !enabled {
            self.close_tracker_pip_capture_panel_window(cx);
        }
        self.refresh_tracker_pip_window(cx);
        info!(enabled, "updated test case capture preference");
        self.persist_ui_preferences(if enabled {
            "测试样本捕获已开启"
        } else {
            "测试样本捕获已关闭"
        });
    }

    pub(super) fn select_group(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.route_editor_draw_mode {
            self.status_text = "连线编辑进行中，请先保存退出或放弃编辑。".into();
            return;
        }
        let selected_point_id = self
            .route_groups
            .iter()
            .find(|group| group.id == group_id)
            .and_then(|group| group.points.first())
            .map(|point| point.id.clone());
        if self.confirming_delete_group_id.as_ref() != Some(&group_id) {
            self.confirming_delete_group_id = None;
        }
        self.clear_selected_point_move_state();
        self.reset_route_editor_graph_state();
        self.selected_group_id = Some(group_id.clone());
        self.selected_point_id = selected_point_id;
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.confirming_delete_point_id = None;
        self.preview_cursor = self.selected_point_index();
        self.route_editor_map_view.request_fit();
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        self.request_center_on_current_point();
        self.sync_editor_from_selection(window, cx);
        if let Some(group) = self.active_group() {
            self.status_text = format!(
                "已选中路线「{}」，共 {} 个节点。",
                group.display_name(),
                group.point_count()
            )
            .into();
        }
    }

    pub(super) fn select_point(
        &mut self,
        point_id: RoutePointId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.moving_point_id.as_ref() != Some(&point_id) {
            self.clear_selected_point_move_state();
        }
        self.confirming_delete_point_id = None;
        self.selected_point_id = Some(point_id.clone());
        self.route_editor_selected_point_ids = [point_id.clone()].into_iter().collect();
        self.preview_cursor = self.selected_point_index();
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        self.request_center_on_current_point();
        self.sync_editor_from_selection(window, cx);
        if let Some(point) = self.selected_point() {
            self.status_text = format!(
                "已选中节点「{}」，坐标 {:.0}, {:.0}。",
                point.display_label(),
                point.x,
                point.y
            )
            .into();
        }
    }

    fn insert_point_from_map_click(
        &mut self,
        world: WorldPoint,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先在左侧选择一条路线，再在地图上插入节点。".into();
            return;
        };

        let insert_index = self
            .active_group()
            .map(RouteDocument::point_count)
            .unwrap_or(0);
        let label = format!(
            "节点 {}",
            self.active_group()
                .map(RouteDocument::point_count)
                .unwrap_or(0)
                .saturating_add(1)
        );
        let world = self.clamp_tracker_world(world);

        let mut created_point_id = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            let mut point = RoutePoint::new(label.clone(), world);
            point.style = group.default_style.clone().normalized();
            let point_id = point.id.clone();
            let bounded_index = insert_index.min(group.points.len());
            group.points.insert(bounded_index, point);
            created_point_id = Some((point_id, bounded_index));
        }

        let Some((point_id, bounded_index)) = created_point_id else {
            self.status_text = "当前路线不存在，无法插入节点。".into();
            return;
        };

        self.reset_route_editor_graph_state();
        self.selected_point_id = Some(point_id);
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.confirming_delete_point_id = None;
        self.preview_cursor = Some(bounded_index);
        self.map_point_insert_armed = false;
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        self.request_center_on_current_point();

        if self.persist_group(
            &group_id,
            &format!("已在第 {} 位插入节点「{}」", bounded_index + 1, label),
        ) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn move_selected_point_to_start(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.move_selected_point(PointMoveTarget::Start, window, cx);
    }

    pub(super) fn move_selected_point_prev(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.move_selected_point(PointMoveTarget::Prev, window, cx);
    }

    pub(super) fn move_selected_point_next(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.move_selected_point(PointMoveTarget::Next, window, cx);
    }

    pub(super) fn move_selected_point_to_end(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.move_selected_point(PointMoveTarget::End, window, cx);
    }

    pub(super) fn move_selected_point_before_target(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.move_selected_point_relative_to_target(false, window, cx);
    }

    pub(super) fn move_selected_point_after_target(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.move_selected_point_relative_to_target(true, window, cx);
    }

    fn move_selected_point(
        &mut self,
        target: PointMoveTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        let Some(point_id) = self.selected_point_id.clone() else {
            self.status_text = "请先选择一个节点。".into();
            return;
        };

        let mut move_result = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            let Some(current_index) = group.points.iter().position(|point| point.id == point_id)
            else {
                self.status_text = "选中的节点不存在。".into();
                return;
            };

            match target {
                PointMoveTarget::Start => {
                    if current_index == 0 {
                        self.status_text = "当前节点已经在路线起点。".into();
                        return;
                    }
                    let point = group.points.remove(current_index);
                    let label = point.display_label().to_owned();
                    group.points.insert(0, point);
                    move_result = Some((0usize, label, "已移动到路线起点".to_owned()));
                }
                PointMoveTarget::Prev => {
                    if current_index == 0 {
                        self.status_text = "当前节点已经是第一个节点。".into();
                        return;
                    }
                    group.points.swap(current_index, current_index - 1);
                    let label = group.points[current_index - 1].display_label().to_owned();
                    move_result = Some((
                        current_index - 1,
                        label,
                        format!("已上移到第 {} 位", current_index),
                    ));
                }
                PointMoveTarget::Next => {
                    if current_index + 1 >= group.points.len() {
                        self.status_text = "当前节点已经是最后一个节点。".into();
                        return;
                    }
                    group.points.swap(current_index, current_index + 1);
                    let label = group.points[current_index + 1].display_label().to_owned();
                    move_result = Some((
                        current_index + 1,
                        label,
                        format!("已下移到第 {} 位", current_index + 2),
                    ));
                }
                PointMoveTarget::End => {
                    if current_index + 1 >= group.points.len() {
                        self.status_text = "当前节点已经在路线终点。".into();
                        return;
                    }
                    let point = group.points.remove(current_index);
                    let label = point.display_label().to_owned();
                    let next_index = group.points.len();
                    group.points.push(point);
                    move_result = Some((next_index, label, "已移动到路线终点".to_owned()));
                }
            }
        }

        let Some((next_index, label, action)) = move_result else {
            self.status_text = "当前路线不存在，无法调整节点顺序。".into();
            return;
        };

        self.reset_route_editor_graph_state();
        self.selected_point_id = Some(point_id);
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.confirming_delete_point_id = None;
        self.preview_cursor = Some(next_index);
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }

        if self.persist_group(&group_id, &format!("节点「{label}」{action}")) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    fn move_selected_point_relative_to_target(
        &mut self,
        place_after: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        let Some(point_id) = self.selected_point_id.clone() else {
            self.status_text = "请先选择一个节点。".into();
            return;
        };
        let Some(target_id) = self.point_reorder_target_id.clone() else {
            self.status_text = "请先选择一个目标节点。".into();
            return;
        };
        if target_id == point_id {
            self.status_text = "不能把节点移动到它自己前后。".into();
            return;
        }

        let mut move_result = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            let Some(current_index) = group.points.iter().position(|point| point.id == point_id)
            else {
                self.status_text = "选中的节点不存在。".into();
                return;
            };
            let Some(target_index) = group.points.iter().position(|point| point.id == target_id)
            else {
                self.status_text = "目标节点不存在。".into();
                return;
            };

            let target_label = group.points[target_index].display_label().to_owned();
            let point = group.points.remove(current_index);
            let adjusted_target_index = if current_index < target_index {
                target_index - 1
            } else {
                target_index
            };
            let insert_index = if place_after {
                adjusted_target_index + 1
            } else {
                adjusted_target_index
            };
            if insert_index == current_index {
                self.status_text = if place_after {
                    format!("当前节点已经在「{target_label}」后面。").into()
                } else {
                    format!("当前节点已经在「{target_label}」前面。").into()
                };
                group.points.insert(current_index, point);
                return;
            }

            let label = point.display_label().to_owned();
            group.points.insert(insert_index, point);
            move_result = Some((
                insert_index,
                label,
                if place_after {
                    format!("已移动到节点「{target_label}」后面")
                } else {
                    format!("已移动到节点「{target_label}」前面")
                },
            ));
        }

        let Some((next_index, label, action)) = move_result else {
            self.status_text = "当前路线不存在，无法调整节点顺序。".into();
            return;
        };

        self.reset_route_editor_graph_state();
        self.selected_point_id = Some(point_id);
        self.route_editor_selected_point_ids = self
            .selected_point_id
            .as_ref()
            .cloned()
            .into_iter()
            .collect();
        self.confirming_delete_point_id = None;
        self.preview_cursor = Some(next_index);
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }

        if self.persist_group(&group_id, &format!("节点「{label}」{action}")) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn rebuild_preview(&mut self) {
        let Some(group) = self.active_group().cloned() else {
            self.preview_position = None;
            self.preview_cursor = None;
            self.trail.clear();
            return;
        };
        if group.points.is_empty() {
            self.preview_position = None;
            self.preview_cursor = None;
            self.trail.clear();
            return;
        }

        let cursor = self
            .preview_cursor
            .unwrap_or_else(|| self.selected_point_index().unwrap_or(0))
            .min(group.points.len().saturating_sub(1));

        self.preview_cursor = Some(cursor);
        self.preview_position = group
            .points
            .get(cursor)
            .map(|point| PositionEstimate::manual(point.world()));
        self.trail = group
            .points
            .iter()
            .take(cursor.saturating_add(1))
            .map(RoutePoint::world)
            .collect();
        self.last_source = self
            .preview_position
            .as_ref()
            .map(|position| position.source);
        self.last_match_score = self
            .preview_position
            .as_ref()
            .and_then(|position| position.match_score);
    }

    pub(super) fn step_preview(&mut self, delta: isize) {
        if self.is_tracking_active() {
            self.status_text = "实时追踪运行中。请先停止追踪，再手动步进当前路线节点。".into();
            return;
        }

        let Some(group) = self.active_group().cloned() else {
            self.status_text = "请先选择一条路线。".into();
            return;
        };
        if group.points.is_empty() {
            self.status_text = "当前路线没有节点。".into();
            return;
        }

        let current =
            self.preview_cursor
                .unwrap_or_else(|| self.selected_point_index().unwrap_or(0)) as isize;
        let last = group.points.len().saturating_sub(1) as isize;
        let next = (current + delta).clamp(0, last) as usize;
        self.preview_cursor = Some(next);
        if let Some(point) = group.points.get(next) {
            self.confirming_delete_point_id = None;
            self.selected_point_id = Some(point.id.clone());
            self.route_editor_selected_point_ids = [point.id.clone()].into_iter().collect();
        }
        self.rebuild_preview();
        self.request_center_on_current_point();
        self.status_text = format!(
            "预演节点切换到第 {} 个点，坐标 {}。",
            next + 1,
            self.current_position_label()
        )
        .into();
    }

    pub(super) fn start_tracker(&mut self, cx: &mut Context<Self>) {
        if self.is_tracking_active() || self.is_tracker_transition_pending() {
            return;
        }

        if !self.workspace.config.minimap.is_configured() {
            self.status_text = "小地图区域尚未配置。请先完成“小地图取区”后再启动追踪。".into();
            cx.notify();
            return;
        }

        let probe = &self.workspace.config.minimap_presence_probe;
        if !probe.enabled || !probe.is_configured() {
            self.status_text =
                "F1-P 标签探针尚未完成配置。请先通过“标签取区”完成建模并确认保存。".into();
            cx.notify();
            return;
        }

        if let Err(error) = load_minimap_presence_model(&self.workspace.project_root) {
            self.status_text = format!(
                "F1-P 标签模型尚未就绪：{error:#}。请重新执行“标签取区”，完成建模预览后再保存。"
            )
            .into();
            cx.notify();
            return;
        }

        info!(
            engine = %self.selected_engine,
            debug_enabled = self.tracking_debug_enabled(),
            "starting tracker session from workbench"
        );
        match spawn_tracker_session(
            self.workspace.clone(),
            self.selected_engine,
            self.tracking_debug_enabled(),
        ) {
            Ok(session) => {
                self.tracker_session = Some(session);
                self.sync_tracker_debug_enabled();
                self.tracker_pending_action = Some(TrackerPendingAction::Starting);
                self.tracker_lifecycle = TrackerLifecycle::Running;
                self.preview_position = None;
                self.trail.clear();
                self.frame_index = 0;
                self.last_source = None;
                self.last_match_score = None;
                self.clear_debug_snapshot(cx);
                self.reset_tracker_pip_debug_summaries();
                self.tracker_status_text =
                    format!("正在启动 {} 追踪线程。", self.selected_engine).into();
                self.status_text = self.tracker_status_text.clone();
            }
            Err(error) => {
                error!(
                    engine = %self.selected_engine,
                    error = %error,
                    "failed to start tracker session"
                );
                self.tracker_pending_action = None;
                self.tracker_lifecycle = TrackerLifecycle::Failed;
                self.reset_tracker_pip_debug_summaries();
                self.tracker_status_text = format!("启动追踪失败：{error:#}").into();
                self.status_text = self.tracker_status_text.clone();
            }
        }
    }

    pub(super) fn stop_tracker(&mut self, preserve_preview: bool, cx: &mut Context<Self>) {
        info!(preserve_preview, "stopping tracker session from workbench");
        self.tracker_pending_action = Some(TrackerPendingAction::Stopping);
        self.tracker_status_text = "正在停止追踪线程。".into();
        self.release_tracker_session();
        self.clear_debug_snapshot(cx);
        self.tracker_pending_action = None;
        self.tracker_lifecycle = TrackerLifecycle::Idle;
        self.reset_tracker_pip_debug_summaries();
        self.tracker_status_text = "追踪线程已停止。".into();
        self.status_text = self.tracker_status_text.clone();
        if !preserve_preview {
            self.rebuild_preview();
        }
    }

    fn release_tracker_session(&mut self) {
        if let Some(mut session) = self.tracker_session.take() {
            debug!("releasing tracker session");
            session.stop();
        }
    }

    fn poll_tracking_events(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(session) = self.tracker_session.as_ref() else {
            return false;
        };

        let events = session.event_rx().try_iter().collect::<Vec<_>>();
        if events.is_empty() {
            return false;
        }

        let mut should_release_session = false;
        for event in events {
            match event {
                TrackingEvent::LifecycleChanged(lifecycle) => {
                    info!(lifecycle = ?lifecycle, "received tracker lifecycle event");
                    self.tracker_lifecycle = lifecycle;
                    match lifecycle {
                        TrackerLifecycle::Idle => {
                            self.tracker_pending_action = None;
                            self.reset_tracker_pip_debug_summaries();
                            self.tracker_status_text = "追踪线程已停止。".into();
                            self.status_text = self.tracker_status_text.clone();
                            should_release_session = true;
                        }
                        TrackerLifecycle::Running => {
                            self.tracker_pending_action = None;
                        }
                        TrackerLifecycle::Failed => {
                            self.tracker_pending_action = None;
                            should_release_session = true;
                        }
                    }
                }
                TrackingEvent::Status(status) => self.apply_tracking_status(status),
                TrackingEvent::Position(position) => self.apply_tracking_position(position),
                TrackingEvent::Debug(snapshot) => {
                    debug!(
                        frame_index = snapshot.frame_index,
                        image_count = snapshot.images.len(),
                        field_count = snapshot.fields.len(),
                        "received tracker debug snapshot"
                    );
                    self.set_debug_snapshot(snapshot, cx);
                }
                TrackingEvent::Error(message) => {
                    error!(message, "received tracker error event");
                    self.tracker_pending_action = None;
                    self.tracker_lifecycle = TrackerLifecycle::Failed;
                    self.reset_tracker_pip_debug_summaries();
                    self.tracker_status_text = format!("追踪线程异常：{message}").into();
                    self.status_text = self.tracker_status_text.clone();
                    should_release_session = true;
                }
            }
        }

        if should_release_session {
            self.release_tracker_session();
        }

        true
    }

    fn apply_tracking_status(&mut self, status: crate::tracking::TrackingStatus) {
        if status.frame_index <= 3 || status.frame_index % 30 == 0 {
            debug!(
                engine = %status.engine,
                frame_index = status.frame_index,
                lifecycle = ?status.lifecycle,
                source = ?status.source,
                match_score = ?status.match_score,
                "applied tracking status"
            );
        }
        self.frame_index = status.frame_index;
        self.last_source = status.source;
        self.last_match_score = status.match_score;
        self.set_tracker_pip_debug_summaries(
            status.probe_summary.clone(),
            status.locate_summary.clone(),
        );
        self.tracker_status_text = status.message.clone().into();
        self.status_text = status.message.into();
        self.tracker_lifecycle = status.lifecycle;
        if status.lifecycle == TrackerLifecycle::Running {
            self.tracker_pending_action = None;
        }
    }

    fn apply_tracking_position(&mut self, position: PositionEstimate) {
        debug!(
            world_x = position.world.x,
            world_y = position.world.y,
            source = ?position.source,
            match_score = ?position.match_score,
            "applied tracking position"
        );
        let position = resolve_tracking_position_heading(self.preview_position.as_ref(), position);
        self.last_source = Some(position.source);
        self.last_match_score = position.match_score;
        self.preview_position = Some(position);
    }

    pub(super) fn toggle_engine(&mut self) {
        if self.is_tracking_active() {
            self.status_text = "请先停止当前追踪，再切换引擎。".into();
            return;
        }

        self.selected_engine = match self.selected_engine {
            TrackerEngineKind::MultiScaleTemplateMatch => {
                TrackerEngineKind::ConvolutionFeatureMatch
            }
            TrackerEngineKind::ConvolutionFeatureMatch => {
                TrackerEngineKind::MultiScaleTemplateMatch
            }
        };
        self.status_text = format!(
            "当前追踪方式已切换为 {}。多尺度模板匹配使用灰度模板相关与局部/全局搜索，卷积特征匹配使用固定卷积特征与张量相似度搜索。",
            self.selected_engine
        )
        .into();
    }

    pub(super) fn sync_editor_from_selection(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .editing_group_id
            .as_ref()
            .is_some_and(|group_id| !self.route_groups.iter().any(|group| &group.id == group_id))
        {
            self.editing_group_id = None;
        }
        if self
            .pending_new_group_id
            .as_ref()
            .is_some_and(|group_id| !self.route_groups.iter().any(|group| &group.id == group_id))
        {
            self.pending_new_group_id = None;
        }
        if self
            .confirming_delete_group_id
            .as_ref()
            .is_some_and(|group_id| !self.route_groups.iter().any(|group| &group.id == group_id))
        {
            self.confirming_delete_group_id = None;
        }

        if self
            .selected_group_id
            .as_ref()
            .is_some_and(|group_id| !self.route_groups.iter().any(|group| &group.id == group_id))
        {
            self.selected_group_id = self.route_groups.first().map(|group| group.id.clone());
        }
        if self.selected_group_id.is_none() {
            self.map_point_insert_armed = false;
            self.clear_selected_point_move_state();
        }

        let active_point_ids = self
            .active_group()
            .map(|group| route_point_id_set(&group.points))
            .unwrap_or_default();
        self.route_editor_selected_point_ids
            .retain(|point_id| active_point_ids.contains(point_id));
        if let Some(group) = self.active_group() {
            if !self.route_editor_graph_state_matches_group(group) {
                self.reset_route_editor_graph_state();
            }
        } else {
            self.route_editor_selected_point_ids.clear();
            self.reset_route_editor_graph_state();
        }

        if let Some(group_id) = self.selected_group_id.clone() {
            let point_exists = self.selected_point_id.as_ref().is_some_and(|point_id| {
                self.route_groups
                    .iter()
                    .find(|group| group.id == group_id)
                    .and_then(|group| group.find_point(point_id))
                    .is_some()
            });
            if !point_exists {
                self.selected_point_id = None;
            }
        } else {
            self.selected_point_id = None;
            self.clear_selected_point_move_state();
        }
        if let Some(point_id) = self.selected_point_id.clone() {
            self.route_editor_selected_point_ids.insert(point_id);
        } else if let Some(point_id) = self.route_editor_selected_point_ids.iter().next().cloned() {
            self.selected_point_id = Some(point_id);
        }
        if self.route_editor_selected_point_ids.len() != 1 {
            self.clear_selected_point_move_state();
        }
        if self.confirming_delete_point_id.as_ref() != self.selected_point_id.as_ref() {
            self.confirming_delete_point_id = None;
        }

        self.sync_marker_group_picker_state(window, cx);

        let selected_group = self.active_group().cloned();
        self.suspend_group_autosave = true;
        self.group_icon = selected_group
            .as_ref()
            .map(|group| group.default_style.icon.clone())
            .unwrap_or_default();
        set_input_value(
            &self.group_form.name,
            selected_group
                .as_ref()
                .map(|group| group.name.clone())
                .unwrap_or_default(),
            window,
            cx,
        );
        set_input_value(
            &self.group_form.description,
            selected_group
                .as_ref()
                .map(|group| group.notes.clone())
                .unwrap_or_default(),
            window,
            cx,
        );
        set_input_value(
            &self.group_form.color_hex,
            selected_group
                .as_ref()
                .map(|group| group.default_style.color_hex.clone())
                .unwrap_or_else(|| "#FF6B6B".to_owned()),
            window,
            cx,
        );
        set_input_value(
            &self.group_form.size_px,
            format!(
                "{:.0}",
                selected_group
                    .as_ref()
                    .map_or(24.0, |group| group.default_style.size_px)
            ),
            window,
            cx,
        );
        self.suspend_group_autosave = false;

        let selected_point = self.selected_point().cloned();
        self.suspend_point_autosave = true;
        self.marker_icon = selected_point
            .as_ref()
            .map(|point| point.style.icon.clone())
            .or_else(|| {
                selected_group
                    .as_ref()
                    .map(|group| group.default_style.icon.clone())
            })
            .unwrap_or_default();
        let default_world = selected_point
            .as_ref()
            .map(RoutePoint::world)
            .unwrap_or_else(|| self.default_marker_world());
        set_input_value(
            &self.marker_form.label,
            selected_point
                .as_ref()
                .and_then(|point| point.label.clone())
                .unwrap_or_default(),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.note,
            selected_point
                .as_ref()
                .map(|point| point.note.clone())
                .unwrap_or_default(),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.x,
            format!("{:.0}", default_world.x),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.y,
            format!("{:.0}", default_world.y),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.color_hex,
            selected_point
                .as_ref()
                .map(|point| point.style.color_hex.clone())
                .unwrap_or_else(|| "#4ECDC4".to_owned()),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.size_px,
            format!(
                "{:.0}",
                selected_point
                    .as_ref()
                    .map_or(24.0, |point| point.style.size_px)
            ),
            window,
            cx,
        );

        self.sync_bwiki_icon_picker_state(window, cx);
        self.suspend_point_autosave = false;
        self.sync_point_reorder_picker_state(window, cx);
        self.sync_visible_list_pages(window, cx);
    }

    fn autosave_selected_group(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.suspend_group_autosave || self.selected_group_id.is_none() {
            return;
        }

        let Ok(draft) = GroupDraft::read(self, cx) else {
            return;
        };

        let _ = self.apply_group_draft(draft, false, window, cx);
    }

    fn autosave_selected_point(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.suspend_point_autosave
            || self.selected_group_id.is_none()
            || self.selected_point_id.is_none()
        {
            return;
        }

        let Ok(draft) = MarkerDraft::read(self, cx) else {
            return;
        };

        let _ = self.apply_selected_point_draft(draft, false, window, cx);
    }

    fn apply_group_draft(
        &mut self,
        draft: GroupDraft,
        announce: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let target_group_id = if let Some(group_id) = self.selected_group_id.clone() {
            if let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            {
                group.name = draft.name;
                group.notes = draft.description;
                group.visible = true;
                group.default_style = draft.style;
                group.metadata.display_name = group.name.clone();
                group_id
            } else {
                self.selected_group_id = None;
                return self.apply_group_draft(draft, announce, window, cx);
            }
        } else {
            let mut group = RouteDocument::new(draft.name);
            group.notes = draft.description;
            group.visible = true;
            group.default_style = draft.style;
            let file_name = self.allocate_group_file_name();
            group.id = RouteId(file_name.clone());
            group.metadata = RouteMetadata {
                id: group.id.clone(),
                file_name,
                display_name: group.display_name().to_owned(),
            };
            let group_id = group.id.clone();
            self.route_groups.push(group);
            self.selected_group_id = Some(group_id.clone());
            self.selected_point_id = None;
            self.confirming_delete_point_id = None;
            group_id
        };

        let persisted = if announce {
            self.persist_group(&target_group_id, "路线已保存")
        } else {
            self.persist_group_silently(&target_group_id)
        };
        if persisted && announce {
            self.sync_editor_from_selection(window, cx);
        }
        persisted
    }

    fn apply_selected_point_draft(
        &mut self,
        draft: MarkerDraft,
        announce: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(group_id) = self.selected_group_id.clone() else {
            return false;
        };
        let Some(point_id) = self.selected_point_id.clone() else {
            return false;
        };

        let saved_point_id = if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            if let Some(point) = group.find_point_mut(&point_id) {
                point.label = Some(draft.label);
                point.note = draft.note;
                point.x = draft.world.x;
                point.y = draft.world.y;
                point.style = draft.style;
                point.id.clone()
            } else {
                self.selected_point_id = None;
                self.confirming_delete_point_id = None;
                return false;
            }
        } else {
            return false;
        };

        self.selected_point_id = Some(saved_point_id);
        self.confirming_delete_point_id = None;

        self.preview_cursor = self.selected_point_index();
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }

        let persisted = if announce {
            self.persist_group(&group_id, "节点已保存")
        } else {
            self.persist_group_silently(&group_id)
        };
        if persisted && announce {
            self.sync_editor_from_selection(window, cx);
        }
        persisted
    }

    pub(super) fn import_route_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_route_import_busy() {
            return;
        }
        info!("prompting for route files to import");
        let paths_receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("选择要导入的路线文件".into()),
        });

        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(paths))) = paths_receiver.await else {
                return;
            };

            this.update_in(cx, |this, window, cx| {
                this.import_route_paths(paths, window, cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub(super) fn import_route_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_route_import_busy() {
            return;
        }
        info!("prompting for route folder to import");
        let folder_receiver = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("选择包含路线文件的文件夹".into()),
        });

        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(mut folders))) = folder_receiver.await else {
                return;
            };
            let Some(folder) = folders.pop() else {
                return;
            };
            let scan_folder = folder.clone();

            this.update_in(cx, |this, _, cx| {
                this.route_import_status =
                    AsyncTaskStatus::working(format!("正在扫描目录 {}。", folder.display()));
                cx.notify();
            })
            .ok();

            let paths_result = cx
                .background_executor()
                .spawn(async move { RouteRepository::collect_import_files(&scan_folder) })
                .await;

            this.update_in(cx, |this, window, cx| match paths_result {
                Ok(paths) if paths.is_empty() => {
                    let message =
                        format!("目录 {} 中没有可导入的 JSON 路线文件。", folder.display());
                    this.route_import_status = AsyncTaskStatus::succeeded(message.clone());
                    this.status_text = message.into();
                    cx.notify();
                }
                Ok(paths) => {
                    this.import_route_paths(paths, window, cx);
                    cx.notify();
                }
                Err(error) => {
                    let message = format!("扫描导入目录失败：{error:#}");
                    this.route_import_status = AsyncTaskStatus::failed(message.clone());
                    this.status_text = message.into();
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn import_route_paths(
        &mut self,
        paths: Vec<PathBuf>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if paths.is_empty() {
            self.status_text = "没有选择任何可导入的路线文件。".into();
            return;
        }

        let target_dir = self.workspace.assets.routes_dir.clone();
        let import_count = paths.len();
        info!(import_count, target_dir = %target_dir.display(), "starting route import");
        self.route_import_status =
            AsyncTaskStatus::working(format!("正在导入 {} 个路线文件。", import_count));

        cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_executor()
                .spawn(async move { RouteRepository::import_paths(paths, &target_dir) })
                .await;

            this.update_in(cx, |this, window, cx| {
                match result {
                    Ok(report) => this.apply_import_report(report, window, cx),
                    Err(error) => {
                        error!(error = %error, "route import failed");
                        let message = format!("导入路线失败：{error:#}");
                        this.route_import_status = AsyncTaskStatus::failed(message.clone());
                        this.status_text = message.into();
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn apply_import_report(
        &mut self,
        report: RouteImportReport,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        info!(
            imported_count = report.imported_count,
            imported_point_count = report.imported_point_count,
            failed_count = report.failed_sources.len(),
            "applying route import report"
        );
        if report.imported_count == 0 {
            if let Some(first_error) = report.failed_sources.first() {
                let message = format!(
                    "没有成功导入任何路线，共 {} 个文件失败。首个错误：{}",
                    report.failed_sources.len(),
                    first_error
                );
                self.route_import_status = AsyncTaskStatus::failed(message.clone());
                self.status_text = message.into();
            } else {
                let message = "没有发现可导入的路线文件。".to_owned();
                self.route_import_status = AsyncTaskStatus::succeeded(message.clone());
                self.status_text = message.into();
            }
            return;
        }

        if let Err(error) =
            self.reload_route_groups(report.first_imported_group_id.as_ref(), window, cx)
        {
            let message = format!("导入完成，但刷新路线失败：{error:#}");
            self.route_import_status = AsyncTaskStatus::failed(message.clone());
            self.status_text = message.into();
            return;
        }

        let mut message = format!(
            "已导入 {} 条路线，共 {} 个节点。",
            report.imported_count, report.imported_point_count
        );
        if !report.failed_sources.is_empty() {
            message.push_str(&format!(
                " 另有 {} 个文件失败。",
                report.failed_sources.len()
            ));
            if let Some(first_error) = report.failed_sources.first() {
                message.push_str(&format!(" 首个错误：{first_error}"));
            }
        }
        self.route_import_status = AsyncTaskStatus::succeeded(message.clone());
        self.status_text = message.into();
    }

    pub(super) fn confirm_or_delete_selected_point(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(point) = self.selected_point() else {
            self.confirming_delete_point_id = None;
            self.status_text = "当前没有选中的节点。".into();
            return;
        };
        let point_id = point.id.clone();
        let point_label = point.display_label().to_owned();

        if self.confirming_delete_point_id.as_ref() == Some(&point_id) {
            self.delete_selected_point(window, cx);
            return;
        }

        self.confirming_delete_point_id = Some(point_id);
        self.status_text = format!("再次点击删除节点「{}」。", point_label).into();
    }

    pub(super) fn delete_selected_point(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "当前没有选中的路线。".into();
            return;
        };
        let Some(point_id) = self.selected_point_id.clone() else {
            self.status_text = "当前没有选中的节点。".into();
            return;
        };

        let mut removed_label = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            removed_label = group
                .remove_point(&point_id)
                .map(|point| point.display_label().to_owned());
        }

        let Some(label) = removed_label else {
            self.status_text = "选中的节点不存在。".into();
            return;
        };

        self.reset_route_editor_graph_state();
        self.route_editor_selected_point_ids.clear();
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;
        self.clear_selected_point_move_state();

        if self.persist_group(&group_id, &format!("节点「{label}」已删除")) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    fn reload_route_groups(
        &mut self,
        preferred_group_id: Option<&RouteId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.route_groups = RouteRepository::load_all(&self.workspace.assets.routes_dir)?;
        self.sync_workspace_routes_snapshot();

        self.selected_group_id = preferred_group_id
            .filter(|group_id| self.route_groups.iter().any(|group| &group.id == *group_id))
            .cloned()
            .or_else(|| {
                self.selected_group_id
                    .clone()
                    .filter(|group_id| self.route_groups.iter().any(|group| &group.id == group_id))
            })
            .or_else(|| self.route_groups.first().map(|group| group.id.clone()));
        self.reset_route_editor_graph_state();
        self.route_editor_selected_point_ids.clear();
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;
        self.preview_cursor = None;

        self.sync_editor_from_selection(window, cx);
        if !self.is_tracking_active() {
            self.rebuild_preview();
            self.request_center_on_current_point();
        }

        Ok(())
    }

    fn sync_workspace_routes_snapshot(&mut self) {
        let mut workspace = (*self.workspace).clone();
        workspace.groups = self.route_groups.clone();
        workspace.report.group_count = self.route_groups.len();
        workspace.report.point_count = self
            .route_groups
            .iter()
            .map(RouteDocument::point_count)
            .sum();
        self.workspace = Arc::new(workspace);
    }

    fn sync_config_form_from_workspace(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let config = self.workspace.config.clone();
        set_input_value(
            &self.config_form.minimap_top,
            config.minimap.top.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_left,
            config.minimap.left.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_width,
            config.minimap.width.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_height,
            config.minimap.height.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_enabled,
            config.minimap_presence_probe.enabled.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_top,
            config.minimap_presence_probe.top.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_left,
            config.minimap_presence_probe.left.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_width,
            config.minimap_presence_probe.width.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_height,
            config.minimap_presence_probe.height.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_presence_probe_match_threshold,
            config.minimap_presence_probe.match_threshold.to_string(),
            window,
            cx,
        );
        self.minimap_presence_probe_device_preference = config.minimap_presence_probe.device;
        self.minimap_presence_probe_device_index = config.minimap_presence_probe.device_index;
        self.sync_minimap_presence_probe_device_picker_state(window, cx);
        set_input_value(
            &self.config_form.window_geometry,
            config.window_geometry,
            window,
            cx,
        );
        set_input_value(
            &self.config_form.view_size,
            config.view_size.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.max_lost_frames,
            config.max_lost_frames.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.teleport_link_distance,
            format!("{:.0}", config.teleport_link_distance),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.local_search_enabled,
            config.local_search.enabled.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.local_search_radius_px,
            config.local_search.radius_px.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.local_search_lock_fail_threshold,
            config.local_search.lock_fail_threshold.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.local_search_max_accepted_jump_px,
            config.local_search.max_accepted_jump_px.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.local_search_reacquire_jump_threshold_px,
            config.local_search.reacquire_jump_threshold_px.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_refresh_rate_ms,
            config.ai.refresh_rate_ms.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_confidence_threshold,
            config.ai.confidence_threshold.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_min_match_count,
            config.ai.min_match_count.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_ransac_threshold,
            config.ai.ransac_threshold.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_scan_size,
            config.ai.scan_size.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_scan_step,
            config.ai.scan_step.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.ai_track_radius,
            config.ai.track_radius.to_string(),
            window,
            cx,
        );
        self.ai_device_preference = config.ai.device;
        self.ai_device_index = config.ai.device_index;
        self.sync_ai_device_picker_state(window, cx);
        set_input_value(
            &self.config_form.ai_weights_path,
            config.ai.weights_path.unwrap_or_default(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_refresh_rate_ms,
            config.template.refresh_rate_ms.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_local_downscale,
            config.template.local_downscale.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_global_downscale,
            config.template.global_downscale.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_global_refine_radius_px,
            config.template.global_refine_radius_px.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_local_match_threshold,
            config.template.local_match_threshold.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_global_match_threshold,
            config.template.global_match_threshold.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_mask_outer_radius,
            config.template.mask_outer_radius.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_mask_inner_radius,
            config.template.mask_inner_radius.to_string(),
            window,
            cx,
        );
        self.template_input_mode = config.template.input_mode;
        self.sync_template_input_mode_picker_state(window, cx);
        self.template_device_preference = config.template.device;
        self.template_device_index = config.template.device_index;
        self.sync_template_device_picker_state(window, cx);
        set_input_value(
            &self.config_form.network_http_port,
            config.network.http_port.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.network_websocket_port,
            config.network.websocket_port.to_string(),
            window,
            cx,
        );
    }

    fn update_workspace_config(&mut self, config: AppConfig) {
        let mut workspace = (*self.workspace).clone();
        workspace.config = config;
        self.workspace = Arc::new(workspace);
    }

    pub(super) fn cache_rebuild_summary(&self, kind: TrackerCacheKind) -> SharedString {
        self.cache_rebuild_status(kind).summary.clone()
    }

    pub(super) fn is_cache_rebuild_running(&self, kind: TrackerCacheKind) -> bool {
        self.cache_rebuild_status(kind).phase == AsyncTaskPhase::Working
    }

    fn cache_rebuild_config(
        &self,
        kind: TrackerCacheKind,
        cx: &mut Context<Self>,
    ) -> Result<AppConfig, String> {
        let mut config = self.workspace.config.clone();
        config.view_size = parse_input_value(&self.config_form.view_size, "view_size", cx)?;
        config.template.local_downscale = parse_input_value(
            &self.config_form.template_local_downscale,
            "template.local_downscale",
            cx,
        )?;
        config.template.global_downscale = parse_input_value(
            &self.config_form.template_global_downscale,
            "template.global_downscale",
            cx,
        )?;
        config.template.mask_outer_radius = parse_input_value(
            &self.config_form.template_mask_outer_radius,
            "template.mask_outer_radius",
            cx,
        )?;
        config.template.mask_inner_radius = parse_input_value(
            &self.config_form.template_mask_inner_radius,
            "template.mask_inner_radius",
            cx,
        )?;

        match kind {
            TrackerCacheKind::Convolution => {
                let weights_path = read_input_value(&self.config_form.ai_weights_path, cx);
                let weights_path = weights_path.trim();
                config.ai.device = self.ai_device_preference;
                config.ai.device_index = self.ai_device_index;
                config.ai.weights_path =
                    (!weights_path.is_empty()).then(|| weights_path.to_owned());
            }
            TrackerCacheKind::Template => {
                config.template.input_mode = self.template_input_mode;
                config.template.device = self.template_device_preference;
                config.template.device_index = self.template_device_index;
            }
        }

        Ok(config)
    }

    pub(super) fn rebuild_tracker_cache(
        &mut self,
        kind: TrackerCacheKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.is_cache_rebuild_running(kind) {
            return;
        }

        if self.is_tracking_active() || self.is_tracker_transition_pending() {
            let message = format!("请先停止当前追踪，再重建{}缓存。", kind.label());
            *self.cache_rebuild_status_mut(kind) = AsyncTaskStatus::failed(message.clone());
            self.status_text = message.into();
            return;
        }

        let config = match self.cache_rebuild_config(kind, cx) {
            Ok(config) => config,
            Err(message) => {
                let detailed = format!("无法重建{}缓存：{message}", kind.label());
                *self.cache_rebuild_status_mut(kind) = AsyncTaskStatus::failed(detailed.clone());
                self.status_text = detailed.into();
                return;
            }
        };

        let mut workspace = (*self.workspace).clone();
        workspace.config = config;

        let started = format!("正在重建{}缓存。", kind.label());
        info!(kind = ?kind, "starting tracker cache rebuild");
        *self.cache_rebuild_status_mut(kind) = AsyncTaskStatus::working(started.clone());
        self.status_text = started.into();

        cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_executor().spawn(async move {
                match kind {
                    TrackerCacheKind::Convolution => rebuild_convolution_engine_cache(&workspace),
                    TrackerCacheKind::Template => rebuild_template_engine_cache(&workspace),
                }
            });

            let result = result.await;
            this.update_in(cx, |this, _, cx| {
                match result {
                    Ok(()) => {
                        info!(kind = ?kind, "tracker cache rebuild completed");
                        let message = format!("{}缓存已按当前表单参数重建完成。", kind.label());
                        *this.cache_rebuild_status_mut(kind) =
                            AsyncTaskStatus::succeeded(message.clone());
                        this.status_text = message.into();
                    }
                    Err(error) => {
                        error!(kind = ?kind, error = %error, "tracker cache rebuild failed");
                        let message = format!("重建{}缓存失败：{error:#}", kind.label());
                        *this.cache_rebuild_status_mut(kind) =
                            AsyncTaskStatus::failed(message.clone());
                        this.status_text = message.into();
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn cache_rebuild_status(&self, kind: TrackerCacheKind) -> &AsyncTaskStatus {
        match kind {
            TrackerCacheKind::Convolution => &self.convolution_cache_status,
            TrackerCacheKind::Template => &self.template_cache_status,
        }
    }

    fn cache_rebuild_status_mut(&mut self, kind: TrackerCacheKind) -> &mut AsyncTaskStatus {
        match kind {
            TrackerCacheKind::Convolution => &mut self.convolution_cache_status,
            TrackerCacheKind::Template => &mut self.template_cache_status,
        }
    }

    fn teleport_link_distance(&self) -> f32 {
        self.workspace.config.teleport_link_distance.max(0.0)
    }

    pub(super) fn save_app_config(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let draft = match ConfigDraft::read(self, cx) {
            Ok(draft) => draft,
            Err(message) => {
                warn!(message, "failed to parse config draft before save");
                self.status_text = message.into();
                return;
            }
        };

        if draft.config.window_geometry.trim().is_empty() {
            self.status_text = "window_geometry 不能为空。".into();
            return;
        }

        let probe_region_changed = self
            .workspace
            .config
            .minimap_presence_probe
            .capture_region()
            != draft.config.minimap_presence_probe.capture_region();

        match save_config(&self.workspace.project_root, &draft.config) {
            Ok(path) => {
                info!(path = %path.display(), "saved application config");
                let mut model_clear_error = None;
                let mut model_cleared = false;
                if probe_region_changed {
                    match delete_minimap_presence_model(&self.workspace.project_root) {
                        Ok(cleared) => {
                            model_cleared = cleared;
                        }
                        Err(error) => {
                            warn!(error = %error, "failed to clear stale F1-P model after config save");
                            model_clear_error = Some(error);
                        }
                    }
                }
                self.update_workspace_config(draft.config);
                self.invalidate_bwiki_route_plan_preview();
                self.sync_config_form_from_workspace(window, cx);
                self.refresh_tracker_pip_window(cx);
                self.refresh_tracker_pip_capture_panel_window(cx);
                self.status_text = if let Some(error) = model_clear_error {
                    format!(
                        "配置已保存到 {}，但清理旧 F1-P 模型失败：{error:#}。请先重新执行“标签取区”，确认模型与当前区域一致后再启动追踪。",
                        path.display()
                    )
                    .into()
                } else if probe_region_changed {
                    format!(
                        "配置已保存到 {}。F1-P 区域已改动{}，请重新执行“标签取区”完成建模确认后再启动追踪。",
                        path.display(),
                        if model_cleared {
                            "，旧模型已清除"
                        } else {
                            ""
                        }
                    )
                    .into()
                } else if self.is_tracking_active() {
                    format!(
                        "配置已保存到 {}。当前追踪需重启后才会完全应用新参数。",
                        path.display()
                    )
                    .into()
                } else {
                    format!("配置已保存到 {}。", path.display()).into()
                };
            }
            Err(error) => {
                error!(error = %error, "failed to save application config");
                self.status_text = format!("保存配置失败：{error:#}").into();
            }
        }
    }

    pub(super) fn toggle_tracker_pip_window(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.is_tracker_pip_open() {
            info!("closing tracker picture-in-picture window from toggle");
            self.close_tracker_pip_window(cx);
            return;
        }
        if self.tracker_pip_pending_open {
            return;
        }

        info!("opening tracker picture-in-picture window from toggle");
        self.tracker_pip_pending_open = true;
        self.status_text = "正在打开追踪画中画。".into();
        cx.defer_in(window, |this, window, cx| {
            this.open_tracker_pip_window(window, cx);
            cx.notify();
        });
    }

    pub(super) fn set_tracker_pip_always_on_top_from_pip(
        &mut self,
        always_on_top: bool,
        cx: &mut Context<Self>,
    ) {
        self.tracker_pip_always_on_top = always_on_top;
        self.apply_tracker_pip_capture_panel_topmost(always_on_top, cx);
        self.status_text = if always_on_top {
            "追踪画中画已置顶。".into()
        } else {
            "追踪画中画已取消置顶。".into()
        };
    }

    fn open_tracker_pip_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.tracker_pip_pending_open = false;
        if self.tracker_pip_window.is_some() {
            return;
        }

        let workbench = cx.entity().downgrade();
        let workbench_for_close = workbench.clone();
        let initial_camera = self.tracker_map_view.camera;
        let initial_focus = self
            .preview_position
            .as_ref()
            .map(|position| position.world);
        let initial_snapshot = self.tracker_map_render_snapshot();
        let minimap_region = self.workspace.config.minimap.clone();
        let bwiki_resources = self.bwiki_resources.clone();
        let bwiki_tile_cache = self.bwiki_tile_cache.clone();
        let initial_bounds = self
            .tracker_pip_window_bounds
            .unwrap_or_else(|| self.default_tracker_pip_window_bounds(window));
        let open_result = cx.open_window(
            WindowOptions {
                titlebar: None,
                window_bounds: Some(initial_bounds),
                kind: WindowKind::Normal,
                is_movable: true,
                is_resizable: true,
                is_minimizable: false,
                window_decorations: Some(gpui::WindowDecorations::Client),
                window_min_size: Some(gpui::size(gpui::px(280.0), gpui::px(240.0))),
                ..Default::default()
            },
            move |pip_window, cx| {
                pip_window.on_window_should_close(cx, move |_, cx| {
                    if let Some(workbench) = workbench_for_close.upgrade() {
                        let _ = workbench.update(cx, |this, cx| {
                            this.handle_tracker_pip_window_closed(cx);
                        });
                    }
                    true
                });

                let view = cx.new(|cx| {
                    TrackerPipWindow::new(
                        workbench.clone(),
                        initial_camera,
                        initial_focus,
                        initial_snapshot.clone(),
                        minimap_region.clone(),
                        bwiki_resources.clone(),
                        bwiki_tile_cache.clone(),
                        pip_window,
                        cx,
                    )
                });
                cx.new(|cx| Root::new(view, pip_window, cx))
            },
        );

        match open_result {
            Ok(handle) => {
                info!(
                    always_on_top = self.tracker_pip_always_on_top,
                    "tracker picture-in-picture window opened"
                );
                self.tracker_pip_window = Some(handle);
                self.tracker_pip_window_bounds = Some(initial_bounds);
                self.status_text = "追踪画中画已打开。".into();

                if self.tracker_pip_always_on_top {
                    if let Some(handle) = self.tracker_pip_window {
                        match handle.update(cx, |_, pip_window, _| {
                            apply_window_topmost(pip_window, true)
                        }) {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => {
                                self.status_text =
                                    format!("追踪画中画已打开，但置顶失败：{error:#}").into();
                            }
                            Err(_) => {
                                self.tracker_pip_window = None;
                                self.status_text = "追踪画中画窗口已经关闭。".into();
                            }
                        }
                    }
                }
            }
            Err(error) => {
                error!(error = %error, "failed to open tracker picture-in-picture window");
                self.tracker_pip_window = None;
                self.status_text = format!("打开追踪画中画失败：{error:#}").into();
            }
        }
    }

    fn close_tracker_pip_window(&mut self, cx: &mut Context<Self>) {
        self.close_tracker_pip_capture_panel_window(cx);
        if let Some(handle) = self.tracker_pip_window.take() {
            info!("closing tracker picture-in-picture window");
            let _ = handle.update(cx, |_, pip_window, cx| {
                pip_window.defer(cx, |pip_window, _| {
                    pip_window.remove_window();
                });
            });
            self.status_text = "追踪画中画已关闭。".into();
        }
        self.tracker_pip_pending_open = false;
    }

    fn default_tracker_pip_window_bounds(&self, window: &Window) -> WindowBounds {
        let main_bounds = window.window_bounds().get_bounds();
        let width = gpui::px(420.0);
        let height = gpui::px(320.0);
        let margin = 24.0;
        let left = (f32::from(main_bounds.right()) - 420.0 - margin)
            .max(f32::from(main_bounds.origin.x) + margin);
        let top = f32::from(main_bounds.origin.y) + margin + 32.0;

        WindowBounds::Windowed(Bounds {
            origin: gpui::point(gpui::px(left), gpui::px(top)),
            size: gpui::size(width, height),
        })
    }

    pub(super) fn handle_tracker_pip_window_closed(&mut self, cx: &mut Context<Self>) {
        self.close_tracker_pip_capture_panel_window(cx);
        if self.tracker_pip_window.take().is_some() {
            self.status_text = "追踪画中画已关闭。".into();
        }
        self.tracker_pip_pending_open = false;
    }

    fn refresh_tracker_pip_window(&mut self, cx: &mut Context<Self>) {
        let Some(handle) = self.tracker_pip_window else {
            return;
        };
        let snapshot = self.tracker_map_render_snapshot();
        let minimap_region = self.workspace.config.minimap.clone();

        match handle.update(cx, |root, pip_window, cx| {
            let Ok(pip) = root.view().clone().downcast::<TrackerPipWindow>() else {
                return None;
            };
            pip.update(cx, |pip, cx| {
                pip.update_snapshot(snapshot.clone(), minimap_region.clone(), cx);
            });
            let bounds = pip_window.window_bounds();
            pip_window.defer(cx, |pip_window, _| {
                pip_window.refresh();
            });
            Some(bounds)
        }) {
            Ok(Some(bounds)) => {
                self.tracker_pip_window_bounds = Some(bounds);
                self.refresh_tracker_pip_capture_panel_window(cx);
            }
            _ => {
                self.tracker_pip_window = None;
                self.tracker_pip_capture_panel_window = None;
                self.tracker_pip_pending_open = false;
            }
        }
    }

    pub(super) fn is_minimap_region_picker_active(&self) -> bool {
        self.minimap_region_picker_window.is_some()
    }

    pub(super) fn toggle_minimap_region_picker(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(handle) = self.minimap_region_picker_window.take() {
            info!("canceling minimap region picker");
            self.status_text = "已取消小地图取区。".into();
            let _ = handle.update(cx, |_, picker_window, cx| {
                picker_window.defer(cx, |picker_window, _| {
                    picker_window.remove_window();
                });
            });
            return;
        }

        info!("opening minimap region picker");
        self.open_minimap_region_picker(window, cx);
    }

    pub(super) fn toggle_minimap_region_picker_from_pip(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.minimap_region_picker_window.take() {
            info!("canceling minimap region picker from picture-in-picture");
            self.status_text = "已取消小地图取区。".into();
            let _ = handle.update(cx, |_, picker_window, cx| {
                picker_window.defer(cx, |picker_window, _| {
                    picker_window.remove_window();
                });
            });
            return;
        }

        info!("opening minimap region picker from picture-in-picture");
        let workbench = cx.entity().downgrade();
        let workbench_for_error = workbench.clone();
        let main_window_handle = self.main_window_handle;
        cx.defer(move |cx| {
            match main_window_handle.update(cx, move |_, main_window, cx| {
                if let Some(workbench) = workbench.upgrade() {
                    let _ = workbench.update(cx, |this, cx| {
                        this.open_minimap_region_picker(main_window, cx);
                    });
                }
            }) {
                Ok(()) => {}
                Err(_) => {
                    if let Some(workbench) = workbench_for_error.upgrade() {
                        let _ = workbench.update(cx, |this, _| {
                            this.status_text = "主工作区窗口已经关闭，无法打开小地图取区。".into();
                        });
                    }
                }
            }
        });
    }

    pub(super) fn toggle_minimap_presence_probe_picker_from_pip(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.minimap_presence_probe_review_window.take() {
            info!("canceling minimap presence probe review from picture-in-picture");
            self.status_text = "已取消 F1-P 标签建模确认。".into();
            let _ = handle.update(cx, |_, review_window, cx| {
                review_window.defer(cx, |review_window, _| {
                    review_window.remove_window();
                });
            });
            return;
        }

        if let Some(handle) = self.minimap_presence_probe_picker_window.take() {
            info!("canceling minimap presence probe picker from picture-in-picture");
            self.status_text = "已取消 F1-P 标签探针取区。".into();
            let _ = handle.update(cx, |_, picker_window, cx| {
                picker_window.defer(cx, |picker_window, _| {
                    picker_window.remove_window();
                });
            });
            return;
        }

        info!("opening minimap presence probe picker from picture-in-picture");
        let workbench = cx.entity().downgrade();
        let workbench_for_error = workbench.clone();
        let main_window_handle = self.main_window_handle;
        cx.defer(move |cx| {
            match main_window_handle.update(cx, move |_, main_window, cx| {
                if let Some(workbench) = workbench.upgrade() {
                    let _ = workbench.update(cx, |this, cx| {
                        this.open_minimap_presence_probe_picker(main_window, cx);
                    });
                }
            }) {
                Ok(()) => {}
                Err(_) => {
                    if let Some(workbench) = workbench_for_error.upgrade() {
                        let _ = workbench.update(cx, |this, _| {
                            this.status_text =
                                "主工作区窗口已经关闭，无法打开 F1-P 标签探针取区。".into();
                        });
                    }
                }
            }
        });
    }

    pub(super) fn is_minimap_presence_probe_picker_active(&self) -> bool {
        self.minimap_presence_probe_picker_window.is_some()
            || self.minimap_presence_probe_review_window.is_some()
    }

    pub(super) fn toggle_minimap_presence_probe_picker(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(handle) = self.minimap_presence_probe_review_window.take() {
            info!("canceling minimap presence probe review");
            self.status_text = "已取消 F1-P 标签建模确认。".into();
            let _ = handle.update(cx, |_, review_window, cx| {
                review_window.defer(cx, |review_window, _| {
                    review_window.remove_window();
                });
            });
            return;
        }

        if let Some(handle) = self.minimap_presence_probe_picker_window.take() {
            info!("canceling minimap presence probe picker");
            self.status_text = "已取消 F1-P 标签探针取区。".into();
            let _ = handle.update(cx, |_, picker_window, cx| {
                picker_window.defer(cx, |picker_window, _| {
                    picker_window.remove_window();
                });
            });
            return;
        }

        info!("opening minimap presence probe picker");
        self.open_minimap_presence_probe_picker(window, cx);
    }

    fn open_minimap_presence_probe_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let probe_region = self
            .workspace
            .config
            .minimap_presence_probe
            .capture_region();
        let Some((display_id, display_bounds)) =
            self.resolve_probe_picker_display(probe_region.as_ref(), window, cx)
        else {
            self.status_text = "无法定位可用显示器，不能进入 F1-P 标签探针取区模式。".into();
            return;
        };

        let workbench = cx.entity().downgrade();
        let workbench_for_close = workbench.clone();
        let picker_result = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(display_bounds)),
                focus: true,
                show: true,
                kind: WindowKind::PopUp,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                display_id: Some(display_id),
                window_background: WindowBackgroundAppearance::Transparent,
                titlebar: None,
                ..Default::default()
            },
            move |picker_window, cx| {
                picker_window.on_window_should_close(cx, move |_, cx| {
                    if let Some(workbench) = workbench_for_close.upgrade() {
                        let _ = workbench.update(cx, |this, _| {
                            this.handle_minimap_presence_probe_picker_closed();
                        });
                    }
                    true
                });

                let picker_bounds = picker_window.bounds();

                cx.new(|_| {
                    MinimapPresenceProbePicker::new(
                        workbench.clone(),
                        picker_bounds,
                        probe_region.clone(),
                    )
                })
            },
        );

        match picker_result {
            Ok(handle) => {
                info!("minimap presence probe picker opened");
                self.minimap_presence_probe_picker_window = Some(handle.into());
                self.status_text =
                    "F1-P 标签探针取区已开启：请只框住标签带，确认后会先建模预览，再由你手动确认保存。".into();
            }
            Err(error) => {
                error!(error = %error, "failed to open minimap presence probe picker");
                self.status_text = format!("打开 F1-P 标签探针取区窗口失败：{error:#}").into();
            }
        }
    }

    pub(super) fn prepare_minimap_presence_probe_model_preview(&mut self) {
        self.minimap_presence_probe_picker_window = None;
        self.status_text = "正在基于当前 F1-P 选区建立建模预览。".into();
    }

    pub(super) fn begin_minimap_presence_probe_model_preview(
        &mut self,
        region: CaptureRegion,
        cx: &mut Context<Self>,
    ) {
        if let Some(handle) = self.minimap_presence_probe_review_window.take() {
            let _ = handle.update(cx, |_, review_window, cx| {
                review_window.defer(cx, |review_window, _| {
                    review_window.remove_window();
                });
            });
        }

        match build_minimap_presence_probe_model(&region) {
            Ok(build) => {
                self.open_minimap_presence_probe_review_window(region, build, cx);
            }
            Err(error) => {
                self.status_text = format!(
                    "F1-P 标签建模失败：{error:#}。请重新选区，只框住 F1 到 P 标签带本身。"
                )
                .into();
            }
        }
    }

    fn open_minimap_presence_probe_review_window(
        &mut self,
        region: CaptureRegion,
        build: crate::tracking::presence::MinimapPresenceModelBuild,
        cx: &mut Context<Self>,
    ) {
        let Some((display_id, display_bounds)) =
            self.resolve_display_for_capture_region(&region, cx)
        else {
            self.status_text = "无法定位可用显示器，不能打开 F1-P 建模预览窗口。".into();
            return;
        };

        let workbench = cx.entity().downgrade();
        let workbench_for_close = workbench.clone();
        let main_window_handle = self.main_window_handle;
        let window_bounds = Self::minimap_presence_probe_review_bounds(display_bounds);
        let review_result = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(window_bounds)),
                focus: true,
                show: true,
                kind: WindowKind::PopUp,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                display_id: Some(display_id),
                titlebar: None,
                window_decorations: Some(gpui::WindowDecorations::Client),
                window_min_size: Some(size(px(760.0), px(300.0))),
                ..Default::default()
            },
            move |review_window, cx| {
                review_window.on_window_should_close(cx, move |_, cx| {
                    if let Some(workbench) = workbench_for_close.upgrade() {
                        let _ = workbench.update(cx, |this, _| {
                            this.handle_minimap_presence_probe_review_closed();
                        });
                    }
                    true
                });

                cx.new(|cx| {
                    MinimapPresenceProbeReviewWindow::new(
                        workbench.clone(),
                        main_window_handle,
                        region.clone(),
                        build.clone(),
                        review_window,
                        cx,
                    )
                })
            },
        );

        match review_result {
            Ok(handle) => {
                self.minimap_presence_probe_review_window = Some(handle.into());
                self.status_text = "F1-P 建模预览已生成：请核对结果后，再手动确认保存。".into();
            }
            Err(error) => {
                error!(error = %error, "failed to open minimap presence probe review window");
                self.status_text = format!("打开 F1-P 建模预览窗口失败：{error:#}").into();
            }
        }
    }

    fn resolve_display_for_capture_region(
        &self,
        region: &CaptureRegion,
        cx: &mut Context<Self>,
    ) -> Option<(gpui::DisplayId, Bounds<Pixels>)> {
        let center_x = region.left as f32 + region.width as f32 * 0.5;
        let center_y = region.top as f32 + region.height as f32 * 0.5;
        if let Some(display) = cx
            .displays()
            .into_iter()
            .find(|display| screen_bounds_contains(display.bounds(), center_x, center_y))
        {
            return Some((display.id(), display.bounds()));
        }

        cx.primary_display()
            .map(|display| (display.id(), display.bounds()))
            .or_else(|| {
                cx.displays()
                    .into_iter()
                    .next()
                    .map(|display| (display.id(), display.bounds()))
            })
    }

    fn minimap_presence_probe_review_bounds(display_bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        let display_width = f32::from(display_bounds.size.width);
        let display_height = f32::from(display_bounds.size.height);
        let width = (display_width - 96.0).clamp(760.0, 1020.0);
        let height = (display_height - 120.0).clamp(300.0, 420.0);
        let left = f32::from(display_bounds.origin.x) + ((display_width - width) * 0.5).max(24.0);
        let top = f32::from(display_bounds.origin.y) + ((display_height - height) * 0.5).max(24.0);

        Bounds {
            origin: point(px(left), px(top)),
            size: size(px(width), px(height)),
        }
    }

    pub(super) fn toggle_tracker_pip_capture_panel_from_pip(
        &mut self,
        pip_bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        if !self.test_case_capture_enabled {
            return;
        }

        self.tracker_pip_window_bounds = Some(WindowBounds::Windowed(pip_bounds));
        if self.tracker_pip_capture_panel_window.is_some() {
            self.close_tracker_pip_capture_panel_window(cx);
        } else {
            self.open_tracker_pip_capture_panel_window(pip_bounds, cx);
        }
        self.refresh_tracker_pip_window(cx);
    }

    pub(super) fn sync_tracker_pip_capture_panel_with_bounds(
        &mut self,
        pip_bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        self.tracker_pip_window_bounds = Some(WindowBounds::Windowed(pip_bounds));
        if !self.test_case_capture_enabled || self.tracker_pip_capture_panel_window.is_none() {
            return;
        }
        self.apply_tracker_pip_capture_panel_bounds(pip_bounds, cx);
    }

    pub(super) fn capture_test_case_with_feedback(
        &mut self,
        label: TestCaseLabel,
        cx: Option<&mut Context<Self>>,
    ) -> SharedString {
        let status = match self.capture_test_case_inner(label) {
            Ok(path) => {
                format!(
                    "已按“{}”保存当前 F1-P 测试样本到 {}。",
                    label.display_name(),
                    path.display()
                )
            }
            Err(error) => {
                format!(
                    "保存“{}” F1-P 测试样本失败：{error:#}",
                    label.display_name()
                )
            }
        };
        self.status_text = status.clone().into();
        if let Some(cx) = cx {
            cx.notify();
        }
        status.into()
    }

    fn resolve_probe_picker_display(
        &self,
        probe_region: Option<&CaptureRegion>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<(gpui::DisplayId, Bounds<Pixels>)> {
        if let Some(region) = probe_region {
            let probe_center_x = region.left as f32 + region.width as f32 * 0.5;
            let probe_center_y = region.top as f32 + region.height as f32 * 0.5;
            if let Some(display) = cx.displays().into_iter().find(|display| {
                screen_bounds_contains(display.bounds(), probe_center_x, probe_center_y)
            }) {
                return Some((display.id(), display.bounds()));
            }
        }

        self.resolve_minimap_picker_display(window, cx)
    }

    fn current_probe_capture_region(&self) -> Result<CaptureRegion> {
        self.workspace
            .config
            .minimap_presence_probe
            .capture_region()
            .ok_or_else(|| {
                crate::app_error!("F1-P 标签探针区域尚未完整配置，请先设置 top/left/width/height")
            })
    }

    fn capture_test_case_inner(&self, label: TestCaseLabel) -> Result<PathBuf> {
        let output_dir = self.resolve_test_case_output_dir()?;
        let base_name = self.next_test_case_base_name(label);
        let output_path = output_dir.join(format!("{base_name}.png"));
        let region = self.current_probe_capture_region()?;
        save_capture_region_png(&region, &output_path)?;
        Ok(output_path)
    }

    fn resolve_test_case_output_dir(&self) -> Result<PathBuf> {
        let output_dir = self.find_assets_test_dir();
        fs::create_dir_all(&output_dir).with_context(|| {
            format!(
                "failed to create test case output directory {}",
                output_dir.display()
            )
        })?;
        Ok(output_dir)
    }

    fn find_assets_test_dir(&self) -> PathBuf {
        self.workspace.project_root.join("assets").join("test")
    }

    fn next_test_case_base_name(&self, label: TestCaseLabel) -> String {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("{}_{}", label.file_prefix(), millis)
    }

    fn apply_tracker_pip_capture_panel_topmost(
        &mut self,
        always_on_top: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(handle) = self.tracker_pip_capture_panel_window else {
            return;
        };

        match handle.update(cx, |_, panel_window, _| {
            apply_window_topmost(panel_window, always_on_top)
        }) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.status_text = format!("同步捕获调试面板置顶失败：{error:#}").into();
            }
            Err(_) => {
                self.tracker_pip_capture_panel_window = None;
            }
        }
    }

    fn open_tracker_pip_capture_panel_window(
        &mut self,
        pip_bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(display_id) = self.resolve_display_id_for_bounds(pip_bounds, cx) else {
            self.status_text = "无法定位画中画所在显示器，不能打开捕获调试面板。".into();
            return;
        };

        let workbench = cx.entity().downgrade();
        let workbench_for_close = workbench.clone();
        let capture_region = self
            .workspace
            .config
            .minimap_presence_probe
            .capture_region();
        let panel_bounds = self.tracker_pip_capture_panel_bounds(pip_bounds);
        let open_result = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(panel_bounds)),
                focus: true,
                show: true,
                kind: WindowKind::PopUp,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                display_id: Some(display_id),
                titlebar: None,
                window_decorations: Some(gpui::WindowDecorations::Client),
                window_min_size: Some(gpui::size(gpui::px(240.0), gpui::px(220.0))),
                ..Default::default()
            },
            move |panel_window, cx| {
                panel_window.on_window_should_close(cx, move |_, cx| {
                    if let Some(workbench) = workbench_for_close.upgrade() {
                        let _ = workbench.update(cx, |this, cx| {
                            this.handle_tracker_pip_capture_panel_window_closed(cx);
                        });
                    }
                    true
                });

                let view = cx.new(|cx| {
                    TrackerPipCapturePanelWindow::new(
                        workbench.clone(),
                        capture_region.clone(),
                        panel_window,
                        cx,
                    )
                });
                cx.new(|cx| Root::new(view, panel_window, cx))
            },
        );

        match open_result {
            Ok(handle) => {
                self.tracker_pip_capture_panel_window = Some(handle);
                self.apply_tracker_pip_capture_panel_topmost(self.tracker_pip_always_on_top, cx);
                self.status_text = "画中画 F1-P 捕获调试面板已打开。".into();
            }
            Err(error) => {
                error!(error = %error, "failed to open tracker picture-in-picture capture panel");
                self.status_text = format!("打开画中画 F1-P 捕获调试面板失败：{error:#}").into();
            }
        }
    }

    fn close_tracker_pip_capture_panel_window(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.tracker_pip_capture_panel_window.take() {
            let _ = handle.update(cx, |_, panel_window, cx| {
                panel_window.defer(cx, |panel_window, _| {
                    panel_window.remove_window();
                });
            });
        }
    }

    fn handle_tracker_pip_capture_panel_window_closed(&mut self, cx: &mut Context<Self>) {
        self.tracker_pip_capture_panel_window = None;
        self.refresh_tracker_pip_window(cx);
    }

    fn refresh_tracker_pip_capture_panel_window(&mut self, cx: &mut Context<Self>) {
        if !self.test_case_capture_enabled {
            self.close_tracker_pip_capture_panel_window(cx);
            return;
        }

        let Some(handle) = self.tracker_pip_capture_panel_window else {
            return;
        };

        let capture_region = self
            .workspace
            .config
            .minimap_presence_probe
            .capture_region();
        let pip_bounds = self
            .tracker_pip_window_bounds
            .map(|bounds| bounds.get_bounds())
            .unwrap_or_default();

        match handle.update(cx, |root, panel_window, cx| {
            let Ok(panel) = root
                .view()
                .clone()
                .downcast::<TrackerPipCapturePanelWindow>()
            else {
                return None;
            };
            let region_changed = panel.update(cx, |panel, cx| {
                panel.update_capture_region(capture_region.clone(), cx)
            });
            if pip_bounds.size.width > px(0.0) && pip_bounds.size.height > px(0.0) {
                let next_bounds =
                    self::TrackerWorkbench::tracker_pip_capture_panel_bounds_for(pip_bounds);
                let _ = apply_window_bounds(panel_window, next_bounds);
            }
            if region_changed {
                panel_window.defer(cx, |panel_window, _| {
                    panel_window.refresh();
                });
            }
            Some(panel_window.window_bounds())
        }) {
            Ok(Some(_)) => {}
            _ => {
                self.tracker_pip_capture_panel_window = None;
            }
        }
    }

    fn apply_tracker_pip_capture_panel_bounds(
        &mut self,
        pip_bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(handle) = self.tracker_pip_capture_panel_window else {
            return;
        };
        let panel_bounds = self.tracker_pip_capture_panel_bounds(pip_bounds);
        let _ = handle.update(cx, |_, panel_window, _| {
            let _ = apply_window_bounds(panel_window, panel_bounds);
        });
    }

    fn tracker_pip_capture_panel_bounds(&self, pip_bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        Self::tracker_pip_capture_panel_bounds_for(pip_bounds)
    }

    fn tracker_pip_capture_panel_bounds_for(pip_bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        let width = px(292.0);
        let height = px(252.0);
        let margin = 12.0;
        let mut left = f32::from(pip_bounds.origin.x) - 292.0 - margin;
        if left < 8.0 {
            left = f32::from(pip_bounds.origin.x) + f32::from(pip_bounds.size.width) + margin;
        }
        let top = f32::from(pip_bounds.origin.y) + 44.0;
        Bounds {
            origin: gpui::point(px(left), px(top)),
            size: gpui::size(width, height),
        }
    }

    fn resolve_display_id_for_bounds(
        &self,
        bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) -> Option<gpui::DisplayId> {
        let center = bounds.center();
        cx.displays()
            .into_iter()
            .find(|display| {
                screen_bounds_contains(display.bounds(), f32::from(center.x), f32::from(center.y))
            })
            .map(|display| display.id())
            .or_else(|| cx.primary_display().map(|display| display.id()))
    }

    fn open_minimap_region_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some((display_id, display_bounds)) = self.resolve_minimap_picker_display(window, cx)
        else {
            self.status_text = "无法定位可用显示器，不能进入小地图取区模式。".into();
            return;
        };

        let workbench = cx.entity().downgrade();
        let workbench_for_close = workbench.clone();
        let main_window_handle = window.window_handle();
        let minimap_region = self.workspace.config.minimap.clone();
        let mask_inner_radius = self.workspace.config.template.mask_inner_radius;
        let mask_outer_radius = self.workspace.config.template.mask_outer_radius;
        let picker_result = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(display_bounds)),
                focus: true,
                show: true,
                kind: WindowKind::PopUp,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                display_id: Some(display_id),
                window_background: WindowBackgroundAppearance::Transparent,
                titlebar: None,
                ..Default::default()
            },
            move |picker_window, cx| {
                picker_window.on_window_should_close(cx, move |_, cx| {
                    if let Some(workbench) = workbench_for_close.upgrade() {
                        let _ = workbench.update(cx, |this, _| {
                            this.handle_minimap_region_picker_closed();
                        });
                    }
                    true
                });

                let picker_bounds = picker_window.bounds();

                cx.new(|_| {
                    MinimapRegionPicker::new(
                        workbench.clone(),
                        main_window_handle,
                        picker_bounds,
                        minimap_region.clone(),
                        mask_inner_radius,
                        mask_outer_radius,
                    )
                })
            },
        );

        match picker_result {
            Ok(handle) => {
                info!("minimap region picker opened");
                self.minimap_region_picker_window = Some(handle.into());
                self.status_text =
                    "小地图环形取区已开启：拖外圈改截图范围，拖内圈改中心挖空，最后点确认。".into();
            }
            Err(error) => {
                error!(error = %error, "failed to open minimap region picker");
                self.status_text = format!("打开小地图取区窗口失败：{error:#}").into();
            }
        }
    }

    fn resolve_minimap_picker_display(
        &self,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<(gpui::DisplayId, Bounds<Pixels>)> {
        let displays = cx.displays();
        if displays.is_empty() {
            return None;
        }

        let minimap = &self.workspace.config.minimap;
        if minimap.is_configured() {
            let minimap_center_x = minimap.left as f32 + minimap.width as f32 / 2.0;
            let minimap_center_y = minimap.top as f32 + minimap.height as f32 / 2.0;
            if let Some(display) = displays.iter().find(|display| {
                screen_bounds_contains(display.bounds(), minimap_center_x, minimap_center_y)
            }) {
                return Some((display.id(), display.bounds()));
            }
        }

        let window_bounds = window.window_bounds().get_bounds();
        let window_center = window_bounds.center();
        if let Some(display) = displays.iter().find(|display| {
            screen_bounds_contains(
                display.bounds(),
                f32::from(window_center.x),
                f32::from(window_center.y),
            )
        }) {
            return Some((display.id(), display.bounds()));
        }

        let display = cx.primary_display().or_else(|| displays.first().cloned())?;
        Some((display.id(), display.bounds()))
    }

    fn sync_minimap_form_region(
        &mut self,
        region: &crate::config::CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        Self::sync_capture_region_form_values(
            &self.config_form.minimap_top,
            &self.config_form.minimap_left,
            &self.config_form.minimap_width,
            &self.config_form.minimap_height,
            region,
            window,
            cx,
        );
    }

    fn sync_template_mask_form_radii(
        &mut self,
        inner_radius: f32,
        outer_radius: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        set_input_value(
            &self.config_form.template_mask_outer_radius,
            outer_radius.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.template_mask_inner_radius,
            inner_radius.to_string(),
            window,
            cx,
        );
    }

    fn sync_minimap_presence_probe_form_region(
        &mut self,
        region: &crate::config::CaptureRegion,
        enabled: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        set_input_value(
            &self.config_form.minimap_presence_probe_enabled,
            enabled.to_string(),
            window,
            cx,
        );
        Self::sync_capture_region_form_values(
            &self.config_form.minimap_presence_probe_top,
            &self.config_form.minimap_presence_probe_left,
            &self.config_form.minimap_presence_probe_width,
            &self.config_form.minimap_presence_probe_height,
            region,
            window,
            cx,
        );
    }

    fn sync_capture_region_form_values(
        top_input: &gpui::Entity<gpui_component::input::InputState>,
        left_input: &gpui::Entity<gpui_component::input::InputState>,
        width_input: &gpui::Entity<gpui_component::input::InputState>,
        height_input: &gpui::Entity<gpui_component::input::InputState>,
        region: &CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        set_input_value(top_input, region.top.to_string(), window, cx);
        set_input_value(left_input, region.left.to_string(), window, cx);
        set_input_value(width_input, region.width.to_string(), window, cx);
        set_input_value(height_input, region.height.to_string(), window, cx);
    }

    pub(super) fn finish_minimap_presence_probe_pick(
        &mut self,
        region: crate::config::CaptureRegion,
        model: MinimapPresenceModel,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        info!(
            top = region.top,
            left = region.left,
            width = region.width,
            height = region.height,
            "finished minimap presence probe pick"
        );
        self.minimap_presence_probe_picker_window = None;
        self.minimap_presence_probe_review_window = None;

        let mut config = self.workspace.config.clone();
        config.minimap_presence_probe.enabled = true;
        config.minimap_presence_probe.top = region.top;
        config.minimap_presence_probe.left = region.left;
        config.minimap_presence_probe.width = region.width;
        config.minimap_presence_probe.height = region.height;

        match save_config(&self.workspace.project_root, &config) {
            Ok(path) => {
                self.update_workspace_config(config.clone());
                self.sync_minimap_presence_probe_form_region(&region, true, window, cx);
                self.refresh_tracker_pip_capture_panel_window(cx);
                if let Err(error) = delete_minimap_presence_model(&self.workspace.project_root) {
                    self.status_text = format!(
                        "F1-P 标签探针区域已保存，但清理旧模型失败：{error:#}。请确认模型文件状态后再重新执行“标签取区”。"
                    )
                    .into();
                    return;
                }
                match save_minimap_presence_model(&self.workspace.project_root, &model) {
                    Ok(model_path) => {
                        self.status_text = if self.is_tracking_active() {
                            format!(
                                "已更新 F1-P 标签探针区域为 top {} / left {} / {}x{}，并保存配置 {} 与模型 {}。当前追踪需重启后才会应用新区域。",
                                region.top,
                                region.left,
                                region.width,
                                region.height,
                                path.display(),
                                model_path.display()
                            )
                            .into()
                        } else {
                            format!(
                                "已更新 F1-P 标签探针区域为 top {} / left {} / {}x{}，并保存配置 {} 与模型 {}。",
                                region.top,
                                region.left,
                                region.width,
                                region.height,
                                path.display(),
                                model_path.display()
                            )
                            .into()
                        };
                    }
                    Err(error) => {
                        self.status_text = format!(
                            "F1-P 标签探针区域已保存为 top {} / left {} / {}x{}，但模型写入失败：{error:#}。请重新执行“标签取区”后再启动追踪。",
                            region.top, region.left, region.width, region.height
                        )
                        .into();
                    }
                }
            }
            Err(error) => {
                self.status_text =
                    format!("F1-P 标签探针建模已完成，但写入配置失败：{error:#}").into();
            }
        }
    }

    fn finish_minimap_region_pick(
        &mut self,
        result: MinimapRegionPickResult,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        info!(
            top = result.region.top,
            left = result.region.left,
            width = result.region.width,
            height = result.region.height,
            mask_inner_radius = result.mask_inner_radius,
            mask_outer_radius = result.mask_outer_radius,
            "finished minimap region pick"
        );
        self.minimap_region_picker_window = None;

        let mut config = self.workspace.config.clone();
        config.minimap = result.region.clone();
        config.template.mask_outer_radius = result.mask_outer_radius;
        config.template.mask_inner_radius = result.mask_inner_radius;
        self.update_workspace_config(config.clone());
        self.sync_minimap_form_region(&result.region, window, cx);
        self.sync_template_mask_form_radii(
            result.mask_inner_radius,
            result.mask_outer_radius,
            window,
            cx,
        );
        self.refresh_tracker_pip_window(cx);
        self.refresh_tracker_pip_capture_panel_window(cx);

        match save_config(&self.workspace.project_root, &config) {
            Ok(path) => {
                self.status_text = if self.is_tracking_active() {
                    format!(
                        "已更新小地图环形取区为 top {} / left {} / {}x{}，内圈 {:.3}，并保存到 {}。当前追踪需重启后才会应用新区域。",
                        result.region.top,
                        result.region.left,
                        result.region.width,
                        result.region.height,
                        result.mask_inner_radius,
                        path.display()
                    )
                    .into()
                } else {
                    format!(
                        "已更新小地图环形取区为 top {} / left {} / {}x{}，内圈 {:.3}，并保存到 {}。",
                        result.region.top,
                        result.region.left,
                        result.region.width,
                        result.region.height,
                        result.mask_inner_radius,
                        path.display()
                    )
                    .into()
                };
            }
            Err(error) => {
                self.status_text = format!(
                    "小地图环形取区已更新为 top {} / left {} / {}x{}，内圈 {:.3}，但写入配置失败：{error:#}",
                    result.region.top,
                    result.region.left,
                    result.region.width,
                    result.region.height,
                    result.mask_inner_radius
                )
                .into();
            }
        }
    }

    pub(super) fn handle_minimap_presence_probe_picker_cancelled(&mut self) {
        self.minimap_presence_probe_picker_window = None;
        self.status_text = "已取消 F1-P 标签探针取区。".into();
    }

    pub(super) fn handle_minimap_presence_probe_picker_closed(&mut self) {
        if self.minimap_presence_probe_picker_window.take().is_some() {
            self.status_text = "已取消 F1-P 标签探针取区。".into();
        }
    }

    pub(super) fn handle_minimap_presence_probe_review_cancelled(&mut self) {
        self.minimap_presence_probe_review_window = None;
        self.status_text = "已取消 F1-P 标签建模确认。".into();
    }

    pub(super) fn handle_minimap_presence_probe_review_closed(&mut self) {
        if self.minimap_presence_probe_review_window.take().is_some() {
            self.status_text = "已取消 F1-P 标签建模确认。".into();
        }
    }

    pub(super) fn handle_minimap_region_picker_cancelled(&mut self) {
        self.minimap_region_picker_window = None;
        self.status_text = "已取消小地图取区。".into();
    }

    pub(super) fn handle_minimap_region_picker_closed(&mut self) {
        if self.minimap_region_picker_window.take().is_some() {
            self.status_text = "已取消小地图取区。".into();
        }
    }

    fn request_center_on_current_point(&mut self) {
        if !self.auto_focus_enabled {
            return;
        }
        if let Some(point) = self.selected_point().map(RoutePoint::world) {
            self.tracker_map_view.center_on_or_queue(point);
        }
    }

    fn ensure_tracker_page_selected_point(&mut self) {
        let Some(group) = self.active_group().cloned() else {
            self.selected_point_id = None;
            self.route_editor_selected_point_ids.clear();
            self.preview_cursor = None;
            return;
        };

        let Some(first_point_id) = group.points.first().map(|point| point.id.clone()) else {
            self.selected_point_id = None;
            self.route_editor_selected_point_ids.clear();
            self.preview_cursor = None;
            return;
        };

        let has_valid_selected_point = self
            .selected_point_id
            .as_ref()
            .is_some_and(|point_id| group.find_point(point_id).is_some());

        if !has_valid_selected_point {
            self.selected_point_id = Some(first_point_id.clone());
            self.route_editor_selected_point_ids = [first_point_id].into_iter().collect();
            self.preview_cursor = Some(0);
            if !self.is_tracking_active() {
                self.rebuild_preview();
            }
            return;
        }

        if let Some(point_id) = self.selected_point_id.clone() {
            self.route_editor_selected_point_ids = [point_id].into_iter().collect();
        }
        self.preview_cursor = self.selected_point_index();
    }

    fn sync_visible_list_pages(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let map_group_page = self
            .selected_group_id
            .as_ref()
            .and_then(|group_id| {
                self.filtered_group_position(PagedListKind::MapGroups, group_id, cx)
                    .map(|index| index / self.map_group_list.page_size.max(1))
            })
            .unwrap_or(0);
        let marker_group_page = self
            .selected_group_id
            .as_ref()
            .and_then(|group_id| {
                self.filtered_group_position(PagedListKind::MarkerGroups, group_id, cx)
                    .map(|index| index / self.marker_group_list.page_size.max(1))
            })
            .unwrap_or(0);
        let point_page = self
            .selected_point_id
            .as_ref()
            .and_then(|point_id| {
                self.filtered_point_position(point_id, cx)
                    .map(|index| index / self.point_list.page_size.max(1))
            })
            .unwrap_or(0);

        self.set_paged_list_page(PagedListKind::MapGroups, map_group_page, window, cx);
        self.set_paged_list_page(PagedListKind::MarkerGroups, marker_group_page, window, cx);
        self.set_paged_list_page(PagedListKind::Points, point_page, window, cx);
    }

    fn persist_group(&mut self, group_id: &RouteId, action_label: &str) -> bool {
        self.persist_group_inner(group_id, Some(action_label))
    }

    fn persist_group_silently(&mut self, group_id: &RouteId) -> bool {
        self.persist_group_inner(group_id, None)
    }

    fn persist_group_inner(&mut self, group_id: &RouteId, action_label: Option<&str>) -> bool {
        let Some(index) = self
            .route_groups
            .iter()
            .position(|group| &group.id == group_id)
        else {
            self.status_text = "待保存的路线不存在。".into();
            return false;
        };

        let file_name = if self.route_groups[index]
            .metadata
            .file_name
            .trim()
            .is_empty()
        {
            self.allocate_group_file_name()
        } else {
            self.route_groups[index].metadata.file_name.clone()
        };

        let mut group = self.route_groups[index].clone().normalized();
        group.id = RouteId(file_name.clone());
        group.metadata.id = group.id.clone();
        group.metadata.file_name = file_name.clone();
        group.metadata.display_name = group.display_name().to_owned();

        let path = self.route_file_path(&file_name);
        match RouteRepository::save(&path, &group) {
            Ok(()) => {
                self.route_groups[index] = group;
                if self.selected_group_id.as_ref() == Some(group_id)
                    && self.selected_point_id.as_ref().is_some_and(|point_id| {
                        self.route_groups[index].find_point(point_id).is_none()
                    })
                {
                    self.selected_point_id = None;
                    self.confirming_delete_point_id = None;
                    self.clear_selected_point_move_state();
                }
                if self.selected_group_id.as_ref() == Some(group_id) && !self.is_tracking_active() {
                    self.rebuild_preview();
                }
                self.sync_workspace_routes_snapshot();
                if let Some(action_label) = action_label {
                    self.status_text =
                        format!("{action_label}，已保存到 {}。", path.display()).into();
                }
                true
            }
            Err(error) => {
                self.status_text = format!("保存路线失败：{error:#}").into();
                false
            }
        }
    }

    fn allocate_group_file_name(&self) -> String {
        loop {
            let candidate = RouteRepository::random_file_name();
            if !self
                .route_groups
                .iter()
                .any(|group| group.metadata.file_name.eq_ignore_ascii_case(&candidate))
            {
                return candidate;
            }
        }
    }

    fn route_file_path(&self, file_name: &str) -> PathBuf {
        self.workspace.assets.routes_dir.join(file_name)
    }

    fn persist_ui_preferences(&mut self, action_label: &str) {
        let preferences = UiPreferences {
            theme_mode: self.theme_preference,
            auto_focus_enabled: self.auto_focus_enabled,
            tracker_point_popup_enabled: self.tracker_point_popup_enabled,
            debug_mode_enabled: self.debug_mode_enabled,
            test_case_capture_enabled: self.test_case_capture_enabled,
        };

        match UiPreferencesRepository::save(&self.workspace.project_root, &preferences) {
            Ok(path) => {
                info!(path = %path.display(), action_label, "persisted UI preferences");
                self.status_text =
                    format!("{action_label}，偏好已保存到 {}。", path.display()).into();
            }
            Err(error) => {
                error!(error = %error, action_label, "failed to persist UI preferences");
                self.status_text = format!("保存界面偏好失败：{error:#}").into();
            }
        }
    }
}

impl Drop for TrackerWorkbench {
    fn drop(&mut self) {
        self.release_tracker_session();
    }
}

fn route_point_id_set(points: &[RoutePoint]) -> HashSet<RoutePointId> {
    points.iter().map(|point| point.id.clone()).collect()
}

fn route_graph_edges_from_points(points: &[RoutePoint]) -> HashSet<RouteGraphEdge> {
    points
        .windows(2)
        .map(|segment| RouteGraphEdge::new(segment[0].id.clone(), segment[1].id.clone()))
        .collect()
}

fn route_graph_insert_edge(
    edges: &mut HashSet<RouteGraphEdge>,
    from: RoutePointId,
    to: RoutePointId,
) -> std::result::Result<RouteGraphInsertOutcome, String> {
    if from == to {
        return Err("不能把节点连到自己。".to_owned());
    }

    let edge = RouteGraphEdge::new(from.clone(), to.clone());
    if edges.contains(&edge) {
        return Ok(RouteGraphInsertOutcome::Unchanged);
    }

    let removed_edges = edges
        .iter()
        .filter(|current| {
            (current.from == from && current.to != to) || (current.to == to && current.from != from)
        })
        .cloned()
        .collect::<Vec<_>>();
    for current in &removed_edges {
        edges.remove(current);
    }

    if route_graph_would_create_cycle(edges, &from, &to) {
        for current in removed_edges {
            edges.insert(current);
        }
        return Err("这条线会形成环路，请调整点击顺序。".to_owned());
    }

    edges.insert(edge);
    Ok(RouteGraphInsertOutcome::Added {
        replaced_edges: removed_edges.len(),
    })
}

fn resolve_route_graph_order(
    points: &[RoutePoint],
    edges: &HashSet<RouteGraphEdge>,
) -> Option<Vec<RoutePointId>> {
    if points.is_empty() {
        return Some(Vec::new());
    }
    if points.len() == 1 {
        return edges.is_empty().then(|| vec![points[0].id.clone()]);
    }
    if edges.len() != points.len().saturating_sub(1) {
        return None;
    }

    let point_ids = route_point_id_set(points);
    let mut incoming = points
        .iter()
        .map(|point| (point.id.clone(), 0usize))
        .collect::<HashMap<_, _>>();
    let mut outgoing = HashMap::<RoutePointId, RoutePointId>::new();
    for edge in edges {
        if edge.from == edge.to
            || !point_ids.contains(&edge.from)
            || !point_ids.contains(&edge.to)
            || outgoing
                .insert(edge.from.clone(), edge.to.clone())
                .is_some()
        {
            return None;
        }
        let entry = incoming.entry(edge.to.clone()).or_default();
        *entry += 1;
        if *entry > 1 {
            return None;
        }
    }

    let starts = points
        .iter()
        .filter_map(|point| {
            (incoming.get(&point.id).copied().unwrap_or(0) == 0).then_some(point.id.clone())
        })
        .collect::<Vec<_>>();
    let ends = points
        .iter()
        .filter_map(|point| (!outgoing.contains_key(&point.id)).then_some(point.id.clone()))
        .collect::<Vec<_>>();
    if starts.len() != 1 || ends.len() != 1 {
        return None;
    }

    let mut order = Vec::with_capacity(points.len());
    let mut seen = HashSet::with_capacity(points.len());
    let mut current = starts[0].clone();
    loop {
        if !seen.insert(current.clone()) {
            return None;
        }
        order.push(current.clone());
        let Some(next) = outgoing.get(&current).cloned() else {
            break;
        };
        current = next;
    }

    (order.len() == points.len()).then_some(order)
}

fn route_graph_would_create_cycle(
    edges: &HashSet<RouteGraphEdge>,
    from: &RoutePointId,
    to: &RoutePointId,
) -> bool {
    let outgoing = edges
        .iter()
        .map(|edge| (edge.from.clone(), edge.to.clone()))
        .collect::<HashMap<_, _>>();
    let mut current = to.clone();
    let mut seen = HashSet::new();
    while seen.insert(current.clone()) {
        if &current == from {
            return true;
        }
        let Some(next) = outgoing.get(&current).cloned() else {
            return false;
        };
        current = next;
    }
    true
}

fn point_in_polygon(point: WorldPoint, polygon: &[WorldPoint]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut previous = *polygon.last().expect("polygon length already checked");
    for current in polygon.iter().copied() {
        let denominator = previous.y - current.y;
        let safe_denominator = if denominator.abs() < 0.0001 {
            0.0001
        } else {
            denominator
        };
        let intersects = ((current.y > point.y) != (previous.y > point.y))
            && (point.x
                < (previous.x - current.x) * (point.y - current.y) / safe_denominator + current.x);
        if intersects {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn planner_world_distance(from: WorldPoint, to: WorldPoint) -> f32 {
    let dx = from.x - to.x;
    let dy = from.y - to.y;
    dx.hypot(dy)
}

fn nearest_teleport_distance(
    point: WorldPoint,
    teleports: &[BwikiPlannerResolvedPoint],
) -> Option<f32> {
    teleports
        .iter()
        .map(|teleport| planner_world_distance(point, teleport.record.world))
        .min_by(|left, right| left.total_cmp(right))
}

fn build_bwiki_point_teleport_arrival_costs(
    points: &[WorldPoint],
    teleports: &[BwikiPlannerResolvedPoint],
) -> Vec<f32> {
    points
        .iter()
        .map(|point| nearest_teleport_distance(*point, teleports).unwrap_or(f32::INFINITY))
        .collect()
}

fn build_bwiki_segment_total_cost(
    from: WorldPoint,
    to: WorldPoint,
    target_teleport_distance: f32,
    teleport_link_distance: f32,
) -> f32 {
    let direct_cost = planner_world_distance(from, to);
    let teleport_cost = teleport_link_distance + target_teleport_distance;

    if teleport_cost + 0.001 < direct_cost {
        teleport_cost
    } else {
        direct_cost
    }
}

fn build_bwiki_planner_cost_matrix(
    points: &[WorldPoint],
    teleport_arrival_costs: &[f32],
    teleport_link_distance: f32,
) -> Vec<Vec<f32>> {
    let point_count = points.len();
    let mut costs = vec![vec![0.0; point_count]; point_count];

    for from_index in 0..point_count {
        for to_index in 0..point_count {
            if from_index == to_index {
                continue;
            }

            let cost = build_bwiki_segment_total_cost(
                points[from_index],
                points[to_index],
                teleport_arrival_costs[to_index],
                teleport_link_distance,
            );
            costs[from_index][to_index] = cost;
        }
    }

    costs
}

fn build_bwiki_planner_order_for_points(
    points: &[WorldPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_arrival_costs: &[f32],
    teleport_link_distance: f32,
) -> Vec<usize> {
    let point_count = points.len();
    if point_count <= 1 {
        return (0..point_count).collect();
    }
    if point_count <= BWIKI_PLANNER_HIERARCHICAL_LIMIT {
        let costs =
            build_bwiki_planner_cost_matrix(points, teleport_arrival_costs, teleport_link_distance);
        return build_bwiki_planner_order_from_costs(&costs);
    }

    build_bwiki_planner_order_hierarchical(
        points,
        teleports,
        teleport_arrival_costs,
        teleport_link_distance,
    )
}

fn build_bwiki_planner_order_hierarchical(
    points: &[WorldPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_arrival_costs: &[f32],
    teleport_link_distance: f32,
) -> Vec<usize> {
    let clusters = build_bwiki_spatial_clusters(points, BWIKI_PLANNER_CLUSTER_TARGET_SIZE);
    if clusters.len() <= 1 {
        let costs =
            build_bwiki_planner_cost_matrix(points, teleport_arrival_costs, teleport_link_distance);
        return build_bwiki_planner_order_from_costs(&costs);
    }

    let local_routes = clusters
        .iter()
        .map(|cluster| {
            let cluster_points = cluster
                .iter()
                .map(|index| points[*index])
                .collect::<Vec<_>>();
            let cluster_arrival_costs = cluster
                .iter()
                .map(|index| teleport_arrival_costs[*index])
                .collect::<Vec<_>>();
            let local_order = build_bwiki_planner_order_for_points(
                &cluster_points,
                teleports,
                &cluster_arrival_costs,
                teleport_link_distance,
            );
            local_order
                .into_iter()
                .filter_map(|local_index| cluster.get(local_index).copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let cluster_centroids = clusters
        .iter()
        .map(|cluster| cluster_centroid(cluster, points))
        .collect::<Vec<_>>();
    let cluster_centroid_arrival_costs =
        build_bwiki_point_teleport_arrival_costs(&cluster_centroids, teleports);
    let cluster_order = build_bwiki_planner_order_for_points(
        &cluster_centroids,
        teleports,
        &cluster_centroid_arrival_costs,
        teleport_link_distance,
    );
    let orientations = choose_bwiki_cluster_orientations(
        &cluster_order,
        &local_routes,
        points,
        teleport_arrival_costs,
        teleport_link_distance,
    );
    let mut order = flatten_bwiki_cluster_routes(&cluster_order, &orientations, &local_routes);
    let spatial_neighbors = build_bwiki_spatial_candidate_neighbors(
        points,
        BWIKI_PLANNER_2OPT_NEIGHBOR_LIMIT,
        BWIKI_PLANNER_SPATIAL_NEIGHBOR_WINDOW,
    );
    improve_bwiki_planner_order_point_2opt(
        &mut order,
        points,
        teleport_arrival_costs,
        &spatial_neighbors,
        teleport_link_distance,
        BWIKI_PLANNER_HIERARCHICAL_2OPT_PASS_LIMIT,
    );
    order
}

fn build_bwiki_planner_order_from_costs(costs: &[Vec<f32>]) -> Vec<usize> {
    let point_count = costs.len();
    if point_count <= 1 {
        return (0..point_count).collect();
    }
    if point_count <= BWIKI_PLANNER_EXACT_LIMIT {
        return build_bwiki_planner_order_exact(costs);
    }

    let candidate_neighbors =
        build_bwiki_planner_candidate_neighbors(costs, BWIKI_PLANNER_2OPT_NEIGHBOR_LIMIT);
    let (starts, seeded_pairs) = build_bwiki_planner_start_candidates(costs);
    let mut best_order = Vec::new();
    let mut best_cost = f32::INFINITY;

    for start in starts {
        for mut candidate in [
            build_bwiki_planner_order_nearest(costs, start),
            build_bwiki_planner_order_bidirectional(costs, start),
        ] {
            improve_bwiki_planner_order_2opt(
                &mut candidate,
                costs,
                &candidate_neighbors,
                BWIKI_PLANNER_2OPT_PASS_LIMIT,
            );
            let candidate_cost = ordered_bwiki_route_cost(&candidate, costs);
            if candidate_cost < best_cost {
                best_order = candidate;
                best_cost = candidate_cost;
            }
        }
    }

    for (left, right) in seeded_pairs {
        let mut candidate = build_bwiki_planner_order_from_pair(costs, left, right);
        improve_bwiki_planner_order_2opt(
            &mut candidate,
            costs,
            &candidate_neighbors,
            BWIKI_PLANNER_2OPT_PASS_LIMIT,
        );
        let candidate_cost = ordered_bwiki_route_cost(&candidate, costs);
        if candidate_cost < best_cost {
            best_order = candidate;
            best_cost = candidate_cost;
        }
    }

    if best_order.is_empty() {
        return build_bwiki_planner_order_nearest(costs, 0);
    }

    if point_count <= BWIKI_PLANNER_FULL_2OPT_LIMIT {
        improve_bwiki_planner_order_full_2opt(
            &mut best_order,
            costs,
            BWIKI_PLANNER_FULL_2OPT_PASS_LIMIT,
        );
    }

    best_order
}

fn build_bwiki_planner_candidate_neighbors(costs: &[Vec<f32>], limit: usize) -> Vec<Vec<usize>> {
    costs
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let mut ranked = (0..row.len())
                .filter(|candidate| *candidate != row_index)
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| row[*left].total_cmp(&row[*right]));
            ranked.truncate(limit.min(ranked.len()));
            ranked
        })
        .collect()
}

fn build_bwiki_spatial_clusters(points: &[WorldPoint], target_size: usize) -> Vec<Vec<usize>> {
    if points.is_empty() {
        return Vec::new();
    }

    let target_size = target_size.max(1);
    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut indexed = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            (
                morton_key(
                    normalized_morton_component(point.x, min_x, max_x),
                    normalized_morton_component(point.y, min_y, max_y),
                ),
                index,
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(key, _)| *key);

    indexed
        .chunks(target_size)
        .map(|chunk| chunk.iter().map(|(_, index)| *index).collect::<Vec<_>>())
        .collect()
}

fn build_bwiki_spatial_candidate_neighbors(
    points: &[WorldPoint],
    neighbor_limit: usize,
    window: usize,
) -> Vec<Vec<usize>> {
    let point_count = points.len();
    if point_count <= 1 {
        return vec![Vec::new(); point_count];
    }

    let mut neighbor_sets = vec![HashSet::new(); point_count];
    let mut add_neighbors_from_order = |ordered: &[usize]| {
        for (position, point_index) in ordered.iter().copied().enumerate() {
            let start = position.saturating_sub(window);
            let end = (position + window + 1).min(ordered.len());
            for neighbor_position in start..end {
                let neighbor_index = ordered[neighbor_position];
                if neighbor_index != point_index {
                    neighbor_sets[point_index].insert(neighbor_index);
                }
            }
        }
    };

    let mut by_x = (0..point_count).collect::<Vec<_>>();
    by_x.sort_by(|left, right| points[*left].x.total_cmp(&points[*right].x));
    add_neighbors_from_order(&by_x);

    let mut by_y = (0..point_count).collect::<Vec<_>>();
    by_y.sort_by(|left, right| points[*left].y.total_cmp(&points[*right].y));
    add_neighbors_from_order(&by_y);

    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut by_morton = (0..point_count)
        .map(|index| {
            (
                morton_key(
                    normalized_morton_component(points[index].x, min_x, max_x),
                    normalized_morton_component(points[index].y, min_y, max_y),
                ),
                index,
            )
        })
        .collect::<Vec<_>>();
    by_morton.sort_by_key(|(key, _)| *key);
    let by_morton = by_morton
        .into_iter()
        .map(|(_, index)| index)
        .collect::<Vec<_>>();
    add_neighbors_from_order(&by_morton);

    neighbor_sets
        .into_iter()
        .enumerate()
        .map(|(index, neighbors)| {
            let mut ranked = neighbors.into_iter().collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                planner_world_distance(points[index], points[*left])
                    .total_cmp(&planner_world_distance(points[index], points[*right]))
            });
            ranked.truncate(neighbor_limit.min(ranked.len()));
            ranked
        })
        .collect()
}

fn normalized_morton_component(value: f32, min: f32, max: f32) -> u16 {
    let span = (max - min).abs();
    if span < 0.001 {
        return 0;
    }

    let normalized = ((value - min) / span).clamp(0.0, 1.0);
    (normalized * f32::from(u16::MAX)).round() as u16
}

fn morton_key(x: u16, y: u16) -> u64 {
    interleave_morton_bits(x) | (interleave_morton_bits(y) << 1)
}

fn interleave_morton_bits(value: u16) -> u64 {
    let mut bits = u64::from(value);
    bits = (bits | (bits << 16)) & 0x0000_FFFF_0000_FFFF;
    bits = (bits | (bits << 8)) & 0x00FF_00FF_00FF_00FF;
    bits = (bits | (bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    bits = (bits | (bits << 2)) & 0x3333_3333_3333_3333;
    bits = (bits | (bits << 1)) & 0x5555_5555_5555_5555;
    bits
}

fn cluster_centroid(cluster: &[usize], points: &[WorldPoint]) -> WorldPoint {
    let (sum_x, sum_y, count) =
        cluster
            .iter()
            .fold((0.0, 0.0, 0usize), |(sum_x, sum_y, count), index| {
                let point = points[*index];
                (sum_x + point.x, sum_y + point.y, count + 1)
            });
    if count == 0 {
        return WorldPoint::new(0.0, 0.0);
    }

    WorldPoint::new(sum_x / count as f32, sum_y / count as f32)
}

fn choose_bwiki_cluster_orientations(
    cluster_order: &[usize],
    local_routes: &[Vec<usize>],
    points: &[WorldPoint],
    teleport_arrival_costs: &[f32],
    teleport_link_distance: f32,
) -> Vec<bool> {
    if cluster_order.is_empty() {
        return Vec::new();
    }

    let cluster_count = cluster_order.len();
    let mut dp = vec![[f32::INFINITY; 2]; cluster_count];
    let mut previous = vec![[0usize; 2]; cluster_count];
    dp[0] = [0.0, 0.0];

    for sequence_index in 1..cluster_count {
        let cluster_index = cluster_order[sequence_index];
        let cluster_route = &local_routes[cluster_index];
        if cluster_route.is_empty() {
            dp[sequence_index] = dp[sequence_index - 1];
            previous[sequence_index] = [0, 0];
            continue;
        }

        for current_orientation in 0..2 {
            let current_start =
                cluster_route_endpoint(cluster_route, current_orientation == 1, true);
            for previous_orientation in 0..2 {
                let previous_cluster_index = cluster_order[sequence_index - 1];
                let previous_route = &local_routes[previous_cluster_index];
                let Some(previous_end) =
                    cluster_route_endpoint_option(previous_route, previous_orientation == 1, false)
                else {
                    continue;
                };
                let transition_cost = bwiki_route_transition_cost(
                    points,
                    teleport_arrival_costs,
                    previous_end,
                    current_start,
                    teleport_link_distance,
                );
                let candidate_cost = dp[sequence_index - 1][previous_orientation] + transition_cost;
                if candidate_cost < dp[sequence_index][current_orientation] {
                    dp[sequence_index][current_orientation] = candidate_cost;
                    previous[sequence_index][current_orientation] = previous_orientation;
                }
            }
        }
    }

    let mut orientations = vec![false; cluster_count];
    let mut orientation = if dp[cluster_count - 1][1] < dp[cluster_count - 1][0] {
        1
    } else {
        0
    };

    for sequence_index in (0..cluster_count).rev() {
        orientations[sequence_index] = orientation == 1;
        if sequence_index > 0 {
            orientation = previous[sequence_index][orientation];
        }
    }

    orientations
}

fn cluster_route_endpoint(route: &[usize], reversed: bool, start: bool) -> usize {
    cluster_route_endpoint_option(route, reversed, start).unwrap_or(0)
}

fn cluster_route_endpoint_option(route: &[usize], reversed: bool, start: bool) -> Option<usize> {
    if route.is_empty() {
        return None;
    }

    if reversed ^ !start {
        route.last().copied()
    } else {
        route.first().copied()
    }
}

fn flatten_bwiki_cluster_routes(
    cluster_order: &[usize],
    orientations: &[bool],
    local_routes: &[Vec<usize>],
) -> Vec<usize> {
    let mut order = Vec::new();

    for (sequence_index, cluster_index) in cluster_order.iter().copied().enumerate() {
        let Some(route) = local_routes.get(cluster_index) else {
            continue;
        };
        if orientations.get(sequence_index).copied().unwrap_or(false) {
            order.extend(route.iter().rev().copied());
        } else {
            order.extend(route.iter().copied());
        }
    }

    order
}

fn bwiki_route_transition_cost(
    points: &[WorldPoint],
    teleport_arrival_costs: &[f32],
    from_index: usize,
    to_index: usize,
    teleport_link_distance: f32,
) -> f32 {
    build_bwiki_segment_total_cost(
        points[from_index],
        points[to_index],
        teleport_arrival_costs[to_index],
        teleport_link_distance,
    )
}

fn build_bwiki_planner_start_candidates(costs: &[Vec<f32>]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let point_count = costs.len();
    if point_count == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut starts = Vec::new();
    push_unique_planner_start(&mut starts, 0);

    let row_sums = costs
        .iter()
        .map(|row| row.iter().copied().sum::<f32>())
        .collect::<Vec<_>>();
    let mut ranked = (0..point_count).collect::<Vec<_>>();
    ranked.sort_by(|left, right| row_sums[*left].total_cmp(&row_sums[*right]));

    for rank in [
        0usize,
        point_count / 4,
        point_count / 2,
        (point_count * 3) / 4,
        point_count - 1,
    ] {
        push_unique_planner_start(&mut starts, ranked[rank]);
    }

    let mut farthest_pair = None;
    let mut farthest_cost = f32::NEG_INFINITY;
    for left in 0..point_count {
        for right in 0..point_count {
            if left == right {
                continue;
            }
            let cost = costs[left][right];
            if cost > farthest_cost {
                farthest_cost = cost;
                farthest_pair = Some((left, right));
            }
        }
    }

    let mut seeded_pairs = Vec::new();
    if let Some((left, right)) = farthest_pair {
        push_unique_planner_start(&mut starts, left);
        push_unique_planner_start(&mut starts, right);
        seeded_pairs.push((left, right));
    }

    starts.truncate(BWIKI_PLANNER_MULTI_START_LIMIT);
    (starts, seeded_pairs)
}

fn push_unique_planner_start(starts: &mut Vec<usize>, candidate: usize) {
    if !starts.contains(&candidate) {
        starts.push(candidate);
    }
}

fn build_bwiki_planner_order_exact(costs: &[Vec<f32>]) -> Vec<usize> {
    let point_count = costs.len();
    let state_count = 1usize << point_count;
    let mut dp = vec![f32::INFINITY; state_count * point_count];
    let mut previous = vec![usize::MAX; state_count * point_count];

    for point_index in 0..point_count {
        dp[(1usize << point_index) * point_count + point_index] = 0.0;
    }

    for mask in 1usize..state_count {
        for end in 0..point_count {
            if mask & (1usize << end) == 0 {
                continue;
            }
            let previous_mask = mask ^ (1usize << end);
            if previous_mask == 0 {
                continue;
            }

            let state_index = mask * point_count + end;
            for candidate_prev in 0..point_count {
                if previous_mask & (1usize << candidate_prev) == 0 {
                    continue;
                }
                let candidate_cost =
                    dp[previous_mask * point_count + candidate_prev] + costs[candidate_prev][end];
                if candidate_cost < dp[state_index] {
                    dp[state_index] = candidate_cost;
                    previous[state_index] = candidate_prev;
                }
            }
        }
    }

    let final_mask = state_count - 1;
    let best_end = (0..point_count)
        .min_by(|left, right| {
            dp[final_mask * point_count + *left].total_cmp(&dp[final_mask * point_count + *right])
        })
        .unwrap_or(0);

    let mut order = Vec::with_capacity(point_count);
    let mut mask = final_mask;
    let mut current = best_end;
    loop {
        order.push(current);
        let previous_index = previous[mask * point_count + current];
        if previous_index == usize::MAX {
            break;
        }
        mask ^= 1usize << current;
        current = previous_index;
    }
    order.reverse();
    order
}

fn build_bwiki_planner_order_nearest(costs: &[Vec<f32>], start: usize) -> Vec<usize> {
    let point_count = costs.len();
    let mut visited = vec![false; point_count];
    let mut order = Vec::with_capacity(point_count);
    let mut current = start.min(point_count.saturating_sub(1));
    visited[current] = true;
    order.push(current);

    while order.len() < point_count {
        let next = (0..point_count)
            .filter(|index| !visited[*index])
            .min_by(|left, right| costs[current][*left].total_cmp(&costs[current][*right]))
            .expect("unvisited point should exist");
        visited[next] = true;
        order.push(next);
        current = next;
    }

    order
}

fn nearest_unvisited_successor(costs: &[Vec<f32>], from: usize, visited: &[bool]) -> Option<usize> {
    (0..costs.len())
        .filter(|index| !visited[*index])
        .min_by(|left, right| costs[from][*left].total_cmp(&costs[from][*right]))
}

fn nearest_unvisited_predecessor(costs: &[Vec<f32>], to: usize, visited: &[bool]) -> Option<usize> {
    (0..costs.len())
        .filter(|index| !visited[*index])
        .min_by(|left, right| costs[*left][to].total_cmp(&costs[*right][to]))
}

fn build_bwiki_planner_order_bidirectional(costs: &[Vec<f32>], start: usize) -> Vec<usize> {
    let point_count = costs.len();
    let start = start.min(point_count.saturating_sub(1));
    let mut visited = vec![false; point_count];
    let mut order = Vec::with_capacity(point_count);
    order.push(start);
    visited[start] = true;

    while order.len() < point_count {
        let left = *order.first().expect("order should not be empty");
        let right = *order.last().expect("order should not be empty");
        let left_candidate = nearest_unvisited_predecessor(costs, left, &visited);
        let right_candidate = nearest_unvisited_successor(costs, right, &visited);

        match (left_candidate, right_candidate) {
            (Some(left_index), Some(right_index)) => {
                if costs[left_index][left] <= costs[right][right_index] {
                    visited[left_index] = true;
                    order.insert(0, left_index);
                } else {
                    visited[right_index] = true;
                    order.push(right_index);
                }
            }
            (Some(left_index), None) => {
                visited[left_index] = true;
                order.insert(0, left_index);
            }
            (None, Some(right_index)) => {
                visited[right_index] = true;
                order.push(right_index);
            }
            (None, None) => break,
        }
    }

    order
}

fn build_bwiki_planner_order_from_pair(
    costs: &[Vec<f32>],
    left_start: usize,
    right_start: usize,
) -> Vec<usize> {
    if left_start == right_start {
        return build_bwiki_planner_order_bidirectional(costs, left_start);
    }

    let point_count = costs.len();
    let mut visited = vec![false; point_count];
    let mut order = Vec::with_capacity(point_count);
    order.push(left_start);
    order.push(right_start);
    visited[left_start] = true;
    visited[right_start] = true;

    while order.len() < point_count {
        let left = order[0];
        let right = *order.last().expect("order should not be empty");
        let mut best = None;

        for candidate in 0..point_count {
            if visited[candidate] {
                continue;
            }

            let left_cost = costs[candidate][left];
            let right_cost = costs[right][candidate];
            let candidate_plan = if left_cost <= right_cost {
                (left_cost, true, candidate)
            } else {
                (right_cost, false, candidate)
            };

            if best.is_none_or(|current: (f32, bool, usize)| candidate_plan.0 < current.0) {
                best = Some(candidate_plan);
            }
        }

        let Some((_, attach_left, candidate)) = best else {
            break;
        };

        visited[candidate] = true;
        if attach_left {
            order.insert(0, candidate);
        } else {
            order.push(candidate);
        }
    }

    order
}

fn route_reversal_transition_costs<F>(
    order: &[usize],
    start: usize,
    end: usize,
    transition_cost: &F,
) -> (f32, f32)
where
    F: Fn(usize, usize) -> f32,
{
    let mut before = 0.0;
    let mut after = 0.0;

    if start > 0 {
        before += transition_cost(order[start - 1], order[start]);
        after += transition_cost(order[start - 1], order[end]);
    }

    for index in start..end {
        before += transition_cost(order[index], order[index + 1]);
    }
    for index in (start + 1..=end).rev() {
        after += transition_cost(order[index], order[index - 1]);
    }

    if let Some(&next) = order.get(end + 1) {
        before += transition_cost(order[end], next);
        after += transition_cost(order[start], next);
    }

    (before, after)
}

fn improve_bwiki_planner_order_2opt(
    order: &mut [usize],
    costs: &[Vec<f32>],
    candidate_neighbors: &[Vec<usize>],
    pass_limit: usize,
) {
    if order.len() < 4 {
        return;
    }

    let mut positions = vec![usize::MAX; costs.len()];
    for (position, point) in order.iter().copied().enumerate() {
        positions[point] = position;
    }

    for _ in 0..pass_limit {
        let mut improved = false;
        for start in 1..order.len() - 1 {
            let left = order[start - 1];
            let first = order[start];
            let mut candidate_ends = Vec::with_capacity(
                candidate_neighbors[left].len() + candidate_neighbors[first].len(),
            );

            for &neighbor in &candidate_neighbors[left] {
                let end = positions[neighbor];
                if end > start && !candidate_ends.contains(&end) {
                    candidate_ends.push(end);
                }
            }
            for &neighbor in &candidate_neighbors[first] {
                let Some(end) = positions[neighbor].checked_sub(1) else {
                    continue;
                };
                if end > start && !candidate_ends.contains(&end) {
                    candidate_ends.push(end);
                }
            }

            candidate_ends.sort_unstable();
            for end in candidate_ends {
                let transition_cost = |from: usize, to: usize| costs[from][to];
                let (before, after) =
                    route_reversal_transition_costs(order, start, end, &transition_cost);
                if after + 0.001 >= before {
                    continue;
                }

                order[start..=end].reverse();
                for position in start..=end {
                    positions[order[position]] = position;
                }
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }
}

fn improve_bwiki_planner_order_full_2opt(
    order: &mut [usize],
    costs: &[Vec<f32>],
    pass_limit: usize,
) {
    if order.len() < 4 {
        return;
    }

    for _ in 0..pass_limit {
        let mut improved = false;
        for start in 1..order.len() - 1 {
            for end in start + 1..order.len() {
                let transition_cost = |from: usize, to: usize| costs[from][to];
                let (before, after) =
                    route_reversal_transition_costs(order, start, end, &transition_cost);
                if after + 0.001 < before {
                    order[start..=end].reverse();
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }
}

fn improve_bwiki_planner_order_point_2opt(
    order: &mut [usize],
    points: &[WorldPoint],
    teleport_arrival_costs: &[f32],
    candidate_neighbors: &[Vec<usize>],
    teleport_link_distance: f32,
    pass_limit: usize,
) {
    if order.len() < 4 {
        return;
    }

    let mut positions = vec![usize::MAX; points.len()];
    for (position, point) in order.iter().copied().enumerate() {
        positions[point] = position;
    }

    for _ in 0..pass_limit {
        let mut improved = false;
        for start in 1..order.len() - 1 {
            let left = order[start - 1];
            let first = order[start];
            let mut candidate_ends = Vec::with_capacity(
                candidate_neighbors[left].len() + candidate_neighbors[first].len(),
            );

            for &neighbor in &candidate_neighbors[left] {
                let end = positions[neighbor];
                if end > start && !candidate_ends.contains(&end) {
                    candidate_ends.push(end);
                }
            }
            for &neighbor in &candidate_neighbors[first] {
                let Some(end) = positions[neighbor].checked_sub(1) else {
                    continue;
                };
                if end > start && !candidate_ends.contains(&end) {
                    candidate_ends.push(end);
                }
            }

            candidate_ends.sort_unstable();
            for end in candidate_ends {
                let transition_cost = |from: usize, to: usize| {
                    bwiki_route_transition_cost(
                        points,
                        teleport_arrival_costs,
                        from,
                        to,
                        teleport_link_distance,
                    )
                };
                let (before, after) =
                    route_reversal_transition_costs(order, start, end, &transition_cost);
                if after + 0.001 >= before {
                    continue;
                }

                order[start..=end].reverse();
                for position in start..=end {
                    positions[order[position]] = position;
                }
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }
}

fn ordered_bwiki_route_cost(order: &[usize], costs: &[Vec<f32>]) -> f32 {
    order
        .windows(2)
        .map(|segment| costs[segment[0]][segment[1]])
        .sum()
}

fn resolve_bwiki_points_by_keys_from_dataset(
    dataset: &crate::resources::BwikiDataset,
    keys: &[BwikiPointKey],
) -> Vec<BwikiPlannerResolvedPoint> {
    if keys.is_empty() {
        return Vec::new();
    }

    let remaining = keys.iter().cloned().collect::<HashSet<_>>();
    let mut lookup = HashMap::with_capacity(keys.len());
    for definition in &dataset.types {
        let Some(records) = dataset.points_by_type.get(&definition.mark_type) else {
            continue;
        };
        for record in records {
            let key = BwikiPointKey::from_record(record);
            if !remaining.contains(&key) {
                continue;
            }
            lookup.insert(
                key.clone(),
                BwikiPlannerResolvedPoint {
                    key,
                    record: record.clone(),
                    type_definition: Some(definition.clone()),
                },
            );

            if lookup.len() == keys.len() {
                break;
            }
        }

        if lookup.len() == keys.len() {
            break;
        }
    }

    keys.iter()
        .filter_map(|key| lookup.get(key).cloned())
        .collect()
}

fn resolve_bwiki_teleports_from_dataset(
    dataset: &crate::resources::BwikiDataset,
) -> Vec<BwikiPlannerResolvedPoint> {
    let type_definition = dataset.type_by_mark_type(BWIKI_TELEPORT_MARK_TYPE).cloned();
    dataset
        .points_by_type
        .get(&BWIKI_TELEPORT_MARK_TYPE)
        .map(|records| {
            records
                .iter()
                .map(|record| BwikiPlannerResolvedPoint {
                    key: BwikiPointKey::from_record(record),
                    record: record.clone(),
                    type_definition: type_definition.clone(),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn build_bwiki_route_plan_task_result(
    dataset: &crate::resources::BwikiDataset,
    selected_keys: Vec<BwikiPointKey>,
    teleport_link_distance: f32,
) -> BwikiPlannerTaskResult {
    let requested_count = selected_keys.len();
    let resolved = resolve_bwiki_points_by_keys_from_dataset(dataset, &selected_keys);
    let normalized_selection_keys = resolved
        .iter()
        .map(|point| point.key.clone())
        .collect::<HashSet<_>>();

    let (preview, failure_message) = if resolved.is_empty() {
        (None, None)
    } else {
        let teleports = resolve_bwiki_teleports_from_dataset(dataset);
        let preview = build_bwiki_route_plan_preview(&resolved, &teleports, teleport_link_distance);
        let failure_message = preview.is_none().then(|| {
            "当前无法生成从传送点开始且不重复经过节点的单行路线，请调整选点后重试。".to_owned()
        });
        (preview, failure_message)
    };

    BwikiPlannerTaskResult {
        requested_count,
        normalized_selection_keys,
        preview,
        failure_message,
    }
}

fn append_preview_key(route_keys: &mut Vec<BwikiPointKey>, key: &BwikiPointKey) {
    if route_keys.last() != Some(key) {
        route_keys.push(key.clone());
    }
}

fn bwiki_route_keys_form_simple_path(route_keys: &[BwikiPointKey]) -> bool {
    let mut seen = HashSet::with_capacity(route_keys.len());
    route_keys.iter().all(|key| seen.insert(key.clone()))
}

#[derive(Debug, Clone)]
struct BwikiTeleportClusterPlan {
    teleport_index: usize,
    walk_order: Vec<usize>,
    teleport_order: Vec<usize>,
    centroid: WorldPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BwikiClusterEntryMode {
    Teleport,
    Walk,
}

#[derive(Debug, Clone)]
struct BwikiClusterRouteVariant {
    route_keys: Vec<BwikiPointKey>,
    entry_world: WorldPoint,
    exit_world: WorldPoint,
    internal_cost: f32,
    entry_mode: BwikiClusterEntryMode,
}

fn bwiki_cluster_centroid(
    point_indices: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
    fallback: WorldPoint,
) -> WorldPoint {
    if point_indices.is_empty() {
        return fallback;
    }

    let (sum_x, sum_y, count) =
        point_indices
            .iter()
            .fold((0.0, 0.0, 0usize), |(sum_x, sum_y, count), index| {
                let point = &resolved_points[*index];
                (
                    sum_x + point.record.world.x,
                    sum_y + point.record.world.y,
                    count + 1,
                )
            });
    if count == 0 {
        return fallback;
    }

    WorldPoint::new(sum_x / count as f32, sum_y / count as f32)
}

fn build_bwiki_walk_order_for_point_indices(
    point_indices: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> Vec<usize> {
    if point_indices.len() <= 1 {
        return point_indices.to_vec();
    }

    let worlds = point_indices
        .iter()
        .map(|index| resolved_points[*index].record.world)
        .collect::<Vec<_>>();
    let direct_arrival_costs = vec![f32::INFINITY; worlds.len()];
    let local_order = build_bwiki_planner_order_for_points(
        &worlds,
        &[],
        &direct_arrival_costs,
        teleport_link_distance,
    );

    local_order
        .into_iter()
        .filter_map(|local_index| point_indices.get(local_index).copied())
        .collect()
}

fn orient_bwiki_order_toward_reference(
    mut order: Vec<usize>,
    resolved_points: &[BwikiPlannerResolvedPoint],
    reference_world: WorldPoint,
) -> Vec<usize> {
    if order.len() <= 1 {
        return order;
    }

    let forward_entry = order
        .first()
        .map(|index| planner_world_distance(reference_world, resolved_points[*index].record.world))
        .unwrap_or(0.0);
    let reverse_entry = order
        .last()
        .map(|index| planner_world_distance(reference_world, resolved_points[*index].record.world))
        .unwrap_or(0.0);
    if reverse_entry + 0.001 < forward_entry {
        order.reverse();
    }

    order
}

fn build_bwiki_cluster_teleport_order(
    point_indices: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleport: &BwikiPlannerResolvedPoint,
    teleport_link_distance: f32,
) -> Vec<usize> {
    let walk_points = point_indices
        .iter()
        .copied()
        .filter(|index| resolved_points[*index].key != teleport.key)
        .collect::<Vec<_>>();

    orient_bwiki_order_toward_reference(
        build_bwiki_walk_order_for_point_indices(
            &walk_points,
            resolved_points,
            teleport_link_distance,
        ),
        resolved_points,
        teleport.record.world,
    )
}

fn ordered_bwiki_walk_cost(
    point_indices: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
) -> f32 {
    point_indices
        .windows(2)
        .map(|segment| {
            let from = resolved_points[segment[0]].record.world;
            let to = resolved_points[segment[1]].record.world;
            planner_world_distance(from, to)
        })
        .sum()
}

fn nearest_bwiki_cluster_point_distance(
    point: WorldPoint,
    cluster_point_indices: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
) -> Option<f32> {
    cluster_point_indices
        .iter()
        .filter_map(|index| resolved_points.get(*index))
        .map(|resolved_point| planner_world_distance(point, resolved_point.record.world))
        .min_by(|left, right| left.total_cmp(right))
}

fn build_bwiki_teleport_cluster_plans(
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> Option<Vec<BwikiTeleportClusterPlan>> {
    if resolved_points.is_empty() {
        return Some(Vec::new());
    }
    if teleports.is_empty() {
        return None;
    }

    let teleport_index_by_key = teleports
        .iter()
        .enumerate()
        .map(|(index, teleport)| (teleport.key.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut cluster_points = vec![Vec::new(); teleports.len()];
    let non_teleport_points = resolved_points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            if point.key.mark_type == BWIKI_TELEPORT_MARK_TYPE {
                let &teleport_index = teleport_index_by_key.get(&point.key)?;
                cluster_points[teleport_index].push(index);
                None
            } else {
                Some((index, point.record.world))
            }
        })
        .collect::<Vec<_>>();
    let mut unassigned_points = non_teleport_points;

    while !unassigned_points.is_empty() {
        let best_assignment = unassigned_points
            .iter()
            .enumerate()
            .filter_map(|(unassigned_index, (point_index, point_world))| {
                teleports
                    .iter()
                    .enumerate()
                    .filter_map(|(teleport_index, teleport)| {
                        let cluster_is_open = !cluster_points[teleport_index].is_empty();
                        let (candidate_cost, tie_distance) = if cluster_is_open {
                            let distance = nearest_bwiki_cluster_point_distance(
                                *point_world,
                                &cluster_points[teleport_index],
                                resolved_points,
                            )?;
                            (distance, distance)
                        } else {
                            let distance =
                                planner_world_distance(*point_world, teleport.record.world);
                            (distance + teleport_link_distance, distance)
                        };

                        Some((
                            unassigned_index,
                            *point_index,
                            teleport_index,
                            candidate_cost,
                            cluster_is_open,
                            tie_distance,
                        ))
                    })
                    .min_by(|left, right| {
                        left.3
                            .total_cmp(&right.3)
                            .then_with(|| right.4.cmp(&left.4))
                            .then_with(|| left.5.total_cmp(&right.5))
                    })
            })
            .min_by(|left, right| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| right.4.cmp(&left.4))
                    .then_with(|| left.5.total_cmp(&right.5))
            })?;
        cluster_points[best_assignment.2].push(best_assignment.1);
        unassigned_points.swap_remove(best_assignment.0);
    }

    Some(
        cluster_points
            .into_iter()
            .enumerate()
            .filter_map(|(teleport_index, point_indices)| {
                (!point_indices.is_empty()).then(|| {
                    let teleport = &teleports[teleport_index];
                    let walk_order = build_bwiki_walk_order_for_point_indices(
                        &point_indices,
                        resolved_points,
                        teleport_link_distance,
                    );
                    let teleport_order = build_bwiki_cluster_teleport_order(
                        &point_indices,
                        resolved_points,
                        teleport,
                        teleport_link_distance,
                    );
                    let centroid = bwiki_cluster_centroid(
                        &point_indices,
                        resolved_points,
                        teleport.record.world,
                    );

                    BwikiTeleportClusterPlan {
                        teleport_index,
                        walk_order,
                        teleport_order,
                        centroid,
                    }
                })
            })
            .collect(),
    )
}

fn build_bwiki_teleport_cluster_order(
    cluster_plans: &[BwikiTeleportClusterPlan],
    teleport_link_distance: f32,
) -> Vec<usize> {
    if cluster_plans.len() <= 1 {
        return (0..cluster_plans.len()).collect();
    }

    let centroids = cluster_plans
        .iter()
        .map(|plan| plan.centroid)
        .collect::<Vec<_>>();
    let direct_arrival_costs = vec![f32::INFINITY; centroids.len()];
    let forward_order = build_bwiki_planner_order_for_points(
        &centroids,
        &[],
        &direct_arrival_costs,
        teleport_link_distance,
    );

    forward_order
}

fn build_bwiki_cluster_walk_variant(
    point_order: &[usize],
    resolved_points: &[BwikiPlannerResolvedPoint],
) -> Option<BwikiClusterRouteVariant> {
    let &first_index = point_order.first()?;
    let &last_index = point_order.last()?;

    Some(BwikiClusterRouteVariant {
        route_keys: point_order
            .iter()
            .filter_map(|index| resolved_points.get(*index).map(|point| point.key.clone()))
            .collect(),
        entry_world: resolved_points.get(first_index)?.record.world,
        exit_world: resolved_points.get(last_index)?.record.world,
        internal_cost: ordered_bwiki_walk_cost(point_order, resolved_points),
        entry_mode: BwikiClusterEntryMode::Walk,
    })
}

fn build_bwiki_cluster_teleport_variant(
    cluster_plan: &BwikiTeleportClusterPlan,
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleports: &[BwikiPlannerResolvedPoint],
) -> Option<BwikiClusterRouteVariant> {
    let teleport = teleports.get(cluster_plan.teleport_index)?;
    let mut route_keys = vec![teleport.key.clone()];
    let mut previous_world = teleport.record.world;
    let mut internal_cost = 0.0;

    for point_index in &cluster_plan.teleport_order {
        let point = resolved_points.get(*point_index)?;
        internal_cost += planner_world_distance(previous_world, point.record.world);
        append_preview_key(&mut route_keys, &point.key);
        previous_world = point.record.world;
    }

    Some(BwikiClusterRouteVariant {
        route_keys,
        entry_world: teleport.record.world,
        exit_world: previous_world,
        internal_cost,
        entry_mode: BwikiClusterEntryMode::Teleport,
    })
}

fn build_bwiki_cluster_route_variants(
    cluster_plan: &BwikiTeleportClusterPlan,
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleports: &[BwikiPlannerResolvedPoint],
) -> Option<Vec<BwikiClusterRouteVariant>> {
    let mut variants = Vec::new();
    variants.push(build_bwiki_cluster_teleport_variant(
        cluster_plan,
        resolved_points,
        teleports,
    )?);

    if let Some(forward_walk_variant) =
        build_bwiki_cluster_walk_variant(&cluster_plan.walk_order, resolved_points)
    {
        variants.push(forward_walk_variant);
    }

    if cluster_plan.walk_order.len() > 1 {
        let reverse_walk_order = cluster_plan
            .walk_order
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        if let Some(reverse_walk_variant) =
            build_bwiki_cluster_walk_variant(&reverse_walk_order, resolved_points)
        {
            variants.push(reverse_walk_variant);
        }
    }

    Some(variants)
}

fn bwiki_cluster_variant_transition_cost(
    previous_variant: &BwikiClusterRouteVariant,
    current_variant: &BwikiClusterRouteVariant,
    teleport_link_distance: f32,
) -> f32 {
    match current_variant.entry_mode {
        BwikiClusterEntryMode::Teleport => teleport_link_distance,
        BwikiClusterEntryMode::Walk => {
            planner_world_distance(previous_variant.exit_world, current_variant.entry_world)
        }
    }
}

fn build_bwiki_route_plan_preview_for_cluster_order(
    resolved_points: &[BwikiPlannerResolvedPoint],
    cluster_plans: &[BwikiTeleportClusterPlan],
    cluster_order: &[usize],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> Option<BwikiRoutePlanPreview> {
    if cluster_order.is_empty() {
        return Some(BwikiRoutePlanPreview::default());
    }

    let cluster_variants = cluster_order
        .iter()
        .map(|cluster_index| {
            build_bwiki_cluster_route_variants(
                cluster_plans.get(*cluster_index)?,
                resolved_points,
                teleports,
            )
        })
        .collect::<Option<Vec<_>>>()?;
    let mut best_costs = cluster_variants
        .iter()
        .map(|variants| vec![f32::INFINITY; variants.len()])
        .collect::<Vec<_>>();
    let mut predecessors = cluster_variants
        .iter()
        .map(|variants| vec![None; variants.len()])
        .collect::<Vec<_>>();

    for (variant_index, variant) in cluster_variants[0].iter().enumerate() {
        if variant
            .route_keys
            .first()
            .is_some_and(|key| key.mark_type == BWIKI_TELEPORT_MARK_TYPE)
        {
            best_costs[0][variant_index] = variant.internal_cost;
        }
    }

    for cluster_index in 1..cluster_variants.len() {
        for (variant_index, variant) in cluster_variants[cluster_index].iter().enumerate() {
            for (previous_variant_index, previous_variant) in
                cluster_variants[cluster_index - 1].iter().enumerate()
            {
                let previous_cost = best_costs[cluster_index - 1][previous_variant_index];
                if !previous_cost.is_finite() {
                    continue;
                }

                let candidate_cost = previous_cost
                    + bwiki_cluster_variant_transition_cost(
                        previous_variant,
                        variant,
                        teleport_link_distance,
                    )
                    + variant.internal_cost;
                if candidate_cost + 0.001 < best_costs[cluster_index][variant_index] {
                    best_costs[cluster_index][variant_index] = candidate_cost;
                    predecessors[cluster_index][variant_index] = Some(previous_variant_index);
                }
            }
        }
    }

    let last_cluster_index = cluster_variants.len().saturating_sub(1);
    let (mut best_variant_index, total_cost) = best_costs[last_cluster_index]
        .iter()
        .enumerate()
        .filter(|(_, cost)| cost.is_finite())
        .min_by(|left, right| left.1.total_cmp(right.1))?;
    let mut chosen_variant_indices = vec![0usize; cluster_variants.len()];
    chosen_variant_indices[last_cluster_index] = best_variant_index;

    for cluster_index in (1..cluster_variants.len()).rev() {
        best_variant_index = predecessors[cluster_index][best_variant_index]?;
        chosen_variant_indices[cluster_index - 1] = best_variant_index;
    }

    let mut route_keys = Vec::new();
    for (cluster_index, variant_index) in chosen_variant_indices.into_iter().enumerate() {
        for key in &cluster_variants[cluster_index][variant_index].route_keys {
            append_preview_key(&mut route_keys, key);
        }
    }

    let selected_keys = resolved_points
        .iter()
        .map(|point| point.key.clone())
        .collect::<HashSet<_>>();
    if !selected_keys.iter().all(|key| route_keys.contains(key)) {
        return None;
    }
    if !bwiki_route_keys_form_simple_path(&route_keys)
        || !route_keys
            .first()
            .is_some_and(|key| key.mark_type == BWIKI_TELEPORT_MARK_TYPE)
    {
        return None;
    }

    Some(BwikiRoutePlanPreview {
        route_keys,
        total_cost: *total_cost,
    })
}

fn build_bwiki_route_plan_preview(
    resolved_points: &[BwikiPlannerResolvedPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> Option<BwikiRoutePlanPreview> {
    if resolved_points.is_empty() {
        return Some(BwikiRoutePlanPreview::default());
    }

    let cluster_plans =
        build_bwiki_teleport_cluster_plans(resolved_points, teleports, teleport_link_distance)?;
    if cluster_plans.is_empty() {
        return None;
    }
    let cluster_order = build_bwiki_teleport_cluster_order(&cluster_plans, teleport_link_distance);
    let reverse_cluster_order = cluster_order.iter().rev().copied().collect::<Vec<_>>();
    let forward_preview = build_bwiki_route_plan_preview_for_cluster_order(
        resolved_points,
        &cluster_plans,
        &cluster_order,
        teleports,
        teleport_link_distance,
    );
    let reverse_preview = (reverse_cluster_order != cluster_order).then(|| {
        build_bwiki_route_plan_preview_for_cluster_order(
            resolved_points,
            &cluster_plans,
            &reverse_cluster_order,
            teleports,
            teleport_link_distance,
        )
    });

    match (forward_preview, reverse_preview.flatten()) {
        (Some(forward), Some(reverse)) if reverse.total_cost + 0.001 < forward.total_cost => {
            Some(reverse)
        }
        (Some(forward), _) => Some(forward),
        (None, Some(reverse)) => Some(reverse),
        (None, None) => None,
    }
}

fn planner_point_to_route_point(
    point: BwikiPlannerResolvedPoint,
    default_style: &MarkerStyle,
) -> RoutePoint {
    let mut route_point = RoutePoint::new(point.record.title.clone(), point.record.world);
    let mark_type_icon_name = point.record.mark_type.to_string();
    let mark_type_icon = MarkerIconStyle::new(mark_type_icon_name.clone());
    let icon = point
        .type_definition
        .as_ref()
        .and_then(|definition| {
            (!definition.name.trim().is_empty())
                .then(|| MarkerIconStyle::new(definition.name.clone()))
        })
        .or_else(|| {
            (mark_type_icon.as_str() != mark_type_icon_name).then_some(mark_type_icon.clone())
        })
        .unwrap_or_else(|| default_style.icon.clone());
    let type_label = point
        .type_definition
        .as_ref()
        .map(|definition| definition.name.clone())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| point.record.mark_type.to_string());
    let category = point
        .type_definition
        .as_ref()
        .map(|definition| definition.category.clone())
        .filter(|category| !category.trim().is_empty())
        .unwrap_or_else(|| "节点图鉴".to_owned());
    let source_id = if point.record.id.trim().is_empty() {
        point.record.uid.clone()
    } else {
        point.record.id.clone()
    };

    route_point.note = format!("{category} · {type_label} · BWiki {source_id}");
    route_point.style = MarkerStyle {
        icon,
        color_hex: default_style.color_hex.clone(),
        size_px: default_style.size_px,
    }
    .normalized();
    route_point
}

impl Render for TrackerWorkbench {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        if self.bwiki_icon_picker_version != self.bwiki_version {
            self.sync_bwiki_icon_picker_state(window, cx);
        }
        render_workbench(self, cx)
    }
}

fn resolve_tracking_position_heading(
    previous: Option<&PositionEstimate>,
    mut position: PositionEstimate,
) -> PositionEstimate {
    if let Some(heading) = position.heading_degrees {
        position.heading_degrees = Some(normalize_tracking_heading_degrees(heading));
        return position;
    }

    position.heading_degrees = previous.and_then(|previous| {
        tracking_motion_heading_degrees(previous.world, position.world).or(previous.heading_degrees)
    });
    position
}

fn tracking_motion_heading_degrees(previous: WorldPoint, current: WorldPoint) -> Option<f32> {
    let dx = current.x - previous.x;
    let dy = current.y - previous.y;
    if dx.hypot(dy) < 0.01 {
        return None;
    }

    Some(normalize_tracking_heading_degrees(
        dy.atan2(dx).to_degrees() + 90.0,
    ))
}

fn normalize_tracking_heading_degrees(degrees: f32) -> f32 {
    degrees.rem_euclid(360.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route_test_points(labels: &[&str]) -> Vec<RoutePoint> {
        labels
            .iter()
            .enumerate()
            .map(|(index, label)| {
                RoutePoint::new(*label, WorldPoint::new(index as f32 * 10.0, 0.0))
            })
            .collect()
    }

    fn planner_point(mark_type: u32, id: &str, x: f32, y: f32) -> BwikiPlannerResolvedPoint {
        let record = BwikiPointRecord {
            mark_type,
            title: id.to_owned(),
            id: id.to_owned(),
            raw_lat: 0,
            raw_lng: 0,
            world: WorldPoint::new(x, y),
            uid: format!("uid-{id}"),
            layer: "test".to_owned(),
            time: None,
            version: None,
        };

        BwikiPlannerResolvedPoint {
            key: BwikiPointKey::from_record(&record),
            record,
            type_definition: None,
        }
    }

    fn assert_cost_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.001,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn planner_cost_matrix_uses_target_teleport_arrival_cost() {
        let points = [WorldPoint::new(0.0, 0.0), WorldPoint::new(100.0, 0.0)];
        let teleports = vec![planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp", 80.0, 0.0)];
        let teleport_arrival_costs = build_bwiki_point_teleport_arrival_costs(&points, &teleports);
        let costs = build_bwiki_planner_cost_matrix(&points, &teleport_arrival_costs, 10.0);

        assert_cost_eq(costs[0][1], 30.0);
        assert_cost_eq(costs[1][0], 90.0);
    }

    #[test]
    fn route_preview_groups_points_by_used_teleport() {
        let teleports = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-b", 1000.0, 0.0),
        ];
        let resolved_points = vec![
            planner_point(1, "a-1", 20.0, 0.0),
            planner_point(1, "a-2", 45.0, 10.0),
            planner_point(1, "b-1", 1015.0, 0.0),
        ];

        let preview = build_bwiki_route_plan_preview(&resolved_points, &teleports, 320.0)
            .expect("preview should exist");

        let route_ids = preview
            .route_keys
            .iter()
            .map(|key| key.id.as_str())
            .collect::<Vec<_>>();
        assert!(
            route_ids == vec!["tp-a", "a-1", "a-2", "tp-b", "b-1"]
                || route_ids == vec!["tp-b", "b-1", "tp-a", "a-1", "a-2"],
            "unexpected route order: {:?}",
            route_ids
        );
        assert_cost_eq(preview.total_cost, 381.9258);
    }

    #[test]
    fn route_preview_avoids_opening_new_teleport_when_walking_is_cheaper() {
        let teleports = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-b", 400.0, 0.0),
        ];
        let resolved_points = vec![
            planner_point(1, "a-1", 20.0, 0.0),
            planner_point(1, "mid", 230.0, 0.0),
        ];

        let preview = build_bwiki_route_plan_preview(&resolved_points, &teleports, 320.0)
            .expect("preview should exist");

        assert_eq!(
            preview
                .route_keys
                .iter()
                .map(|key| key.id.as_str())
                .collect::<Vec<_>>(),
            vec!["tp-a", "a-1", "mid"]
        );
        assert_cost_eq(preview.total_cost, 230.0);
    }

    #[test]
    fn route_preview_skips_cluster_teleport_when_cross_cluster_walk_is_shorter() {
        let teleports = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-b", 1000.0, 0.0),
        ];
        let resolved_points = vec![
            planner_point(1, "a-1", 20.0, 0.0),
            planner_point(1, "a-2", 700.0, 0.0),
            planner_point(1, "b-1", 760.0, 0.0),
        ];

        let preview = build_bwiki_route_plan_preview(&resolved_points, &teleports, 450.0)
            .expect("preview should exist");

        assert_eq!(
            preview
                .route_keys
                .iter()
                .map(|key| key.id.as_str())
                .collect::<Vec<_>>(),
            vec!["tp-a", "a-1", "a-2", "b-1"]
        );
        assert_cost_eq(preview.total_cost, 760.0);
    }

    #[test]
    fn teleport_clusters_expand_from_existing_frontier_before_opening_new_one() {
        let teleports = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-b", 1000.0, 0.0),
        ];
        let resolved_points = vec![
            planner_point(1, "a-1", 20.0, 0.0),
            planner_point(1, "bridge", 700.0, 0.0),
            planner_point(1, "b-edge", 760.0, 0.0),
        ];

        let cluster_plans = build_bwiki_teleport_cluster_plans(&resolved_points, &teleports, 450.0)
            .expect("cluster plans should exist");

        assert_eq!(cluster_plans.len(), 1);
        assert_eq!(cluster_plans[0].teleport_index, 0);
        assert_eq!(cluster_plans[0].walk_order.len(), 3);
    }

    #[test]
    fn route_preview_keeps_selected_teleport_once_at_cluster_start() {
        let teleports = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-b", 1000.0, 0.0),
        ];
        let resolved_points = vec![
            planner_point(BWIKI_TELEPORT_MARK_TYPE, "tp-a", 0.0, 0.0),
            planner_point(1, "a-1", 20.0, 0.0),
        ];

        let preview = build_bwiki_route_plan_preview(&resolved_points, &teleports, 320.0)
            .expect("preview should exist");

        assert_eq!(
            preview
                .route_keys
                .iter()
                .map(|key| key.id.as_str())
                .collect::<Vec<_>>(),
            vec!["tp-a", "a-1"]
        );
        assert_cost_eq(preview.total_cost, 20.0);
    }

    #[test]
    fn planner_points_prefer_bwiki_type_icon_over_route_default_icon() {
        let record = BwikiPointRecord {
            mark_type: 702,
            title: "sample".to_owned(),
            id: "sample".to_owned(),
            raw_lat: 0,
            raw_lng: 0,
            world: WorldPoint::new(10.0, 20.0),
            uid: "uid-sample".to_owned(),
            layer: "test".to_owned(),
            time: None,
            version: None,
        };
        let point = BwikiPlannerResolvedPoint {
            key: BwikiPointKey::from_record(&record),
            record,
            type_definition: Some(BwikiTypeDefinition {
                category: "test".to_owned(),
                mark_type: 702,
                name: "黄石榴石".to_owned(),
                icon_url: String::new(),
                point_count: 1,
                type_known: true,
            }),
        };
        let default_style = MarkerStyle {
            icon: MarkerIconStyle::new("黑晶琉璃"),
            color_hex: "#123456".to_owned(),
            size_px: 28.0,
        };

        let route_point = planner_point_to_route_point(point, &default_style);

        assert_eq!(route_point.style.icon, MarkerIconStyle::new("黄石榴石"));
        assert_eq!(route_point.style.color_hex, default_style.color_hex);
        assert_eq!(route_point.style.size_px, default_style.size_px);
    }

    #[test]
    fn planner_points_use_legacy_mark_type_icon_before_route_default_icon() {
        let point = planner_point(702, "sample", 10.0, 20.0);
        let default_style = MarkerStyle {
            icon: MarkerIconStyle::new("黑晶琉璃"),
            color_hex: "#123456".to_owned(),
            size_px: 28.0,
        };

        let route_point = planner_point_to_route_point(point, &default_style);

        assert_eq!(route_point.style.icon, MarkerIconStyle::new("黄石榴石"));
    }

    #[test]
    fn planner_points_fall_back_to_route_default_icon_when_bwiki_icon_unknown() {
        let point = planner_point(999999, "sample", 10.0, 20.0);
        let default_style = MarkerStyle {
            icon: MarkerIconStyle::new("黑晶琉璃"),
            color_hex: "#123456".to_owned(),
            size_px: 28.0,
        };

        let route_point = planner_point_to_route_point(point, &default_style);

        assert_eq!(route_point.style.icon, default_style.icon);
    }

    #[test]
    fn full_2opt_respects_directed_internal_edges() {
        let mut order = vec![0usize, 1, 2, 3];
        let costs = vec![
            vec![0.0, 10.0, 1.0, 50.0],
            vec![50.0, 0.0, 1.0, 1.0],
            vec![50.0, 100.0, 0.0, 10.0],
            vec![50.0, 50.0, 50.0, 0.0],
        ];

        improve_bwiki_planner_order_full_2opt(&mut order, &costs, 1);

        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn route_graph_order_accepts_simple_directed_chain() {
        let points = route_test_points(&["a", "b", "c", "d"]);
        let edges = HashSet::from([
            RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
            RouteGraphEdge::new(points[1].id.clone(), points[2].id.clone()),
            RouteGraphEdge::new(points[2].id.clone(), points[3].id.clone()),
        ]);

        let order = resolve_route_graph_order(&points, &edges).expect("graph should be valid");

        assert_eq!(
            order,
            points
                .iter()
                .map(|point| point.id.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn route_graph_order_rejects_branching_and_cycle() {
        let points = route_test_points(&["a", "b", "c"]);
        let branching_edges = HashSet::from([
            RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
            RouteGraphEdge::new(points[0].id.clone(), points[2].id.clone()),
        ]);
        let cycle_edges = HashSet::from([
            RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
            RouteGraphEdge::new(points[1].id.clone(), points[2].id.clone()),
            RouteGraphEdge::new(points[2].id.clone(), points[0].id.clone()),
        ]);

        assert!(resolve_route_graph_order(&points, &branching_edges).is_none());
        assert!(resolve_route_graph_order(&points, &cycle_edges).is_none());
    }

    #[test]
    fn route_graph_cycle_check_detects_back_edge() {
        let points = route_test_points(&["a", "b", "c"]);
        let edges = HashSet::from([
            RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
            RouteGraphEdge::new(points[1].id.clone(), points[2].id.clone()),
        ]);

        assert!(route_graph_would_create_cycle(
            &edges,
            &points[2].id,
            &points[0].id
        ));
        assert!(!route_graph_would_create_cycle(
            &edges,
            &points[0].id,
            &points[2].id
        ));
    }

    #[test]
    fn route_graph_insert_edge_treats_existing_edge_as_noop() {
        let points = route_test_points(&["a", "b", "c"]);
        let mut edges = HashSet::from([
            RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
            RouteGraphEdge::new(points[1].id.clone(), points[2].id.clone()),
        ]);

        let outcome =
            route_graph_insert_edge(&mut edges, points[0].id.clone(), points[1].id.clone())
                .expect("existing edge should not error");

        assert_eq!(outcome, RouteGraphInsertOutcome::Unchanged);
        assert_eq!(
            edges,
            HashSet::from([
                RouteGraphEdge::new(points[0].id.clone(), points[1].id.clone()),
                RouteGraphEdge::new(points[1].id.clone(), points[2].id.clone()),
            ])
        );
    }

    #[test]
    fn tracking_motion_heading_uses_screen_up_as_zero_degrees() {
        assert_eq!(
            tracking_motion_heading_degrees(
                WorldPoint::new(100.0, 100.0),
                WorldPoint::new(100.0, 80.0),
            ),
            Some(0.0)
        );
        assert_eq!(
            tracking_motion_heading_degrees(
                WorldPoint::new(100.0, 100.0),
                WorldPoint::new(120.0, 100.0),
            ),
            Some(90.0)
        );
        assert_eq!(
            tracking_motion_heading_degrees(
                WorldPoint::new(100.0, 100.0),
                WorldPoint::new(100.0, 120.0),
            ),
            Some(180.0)
        );
        assert_eq!(
            tracking_motion_heading_degrees(
                WorldPoint::new(100.0, 100.0),
                WorldPoint::new(80.0, 100.0),
            ),
            Some(270.0)
        );
    }

    #[test]
    fn stationary_tracking_position_keeps_previous_heading() {
        let previous = PositionEstimate {
            world: WorldPoint::new(512.0, 256.0),
            found: true,
            inertial: false,
            heading_degrees: Some(135.0),
            source: TrackingSource::LocalTrack,
            match_score: Some(0.91),
        };

        let resolved = resolve_tracking_position_heading(
            Some(&previous),
            PositionEstimate::tracked(
                WorldPoint::new(512.0, 256.0),
                TrackingSource::InertialHold,
                None,
                true,
            ),
        );

        assert_eq!(resolved.heading_degrees, Some(135.0));
    }

    #[test]
    fn explicit_tracking_heading_is_normalized_and_preserved() {
        let previous = PositionEstimate {
            world: WorldPoint::new(10.0, 10.0),
            found: true,
            inertial: false,
            heading_degrees: Some(45.0),
            source: TrackingSource::LocalTrack,
            match_score: Some(0.8),
        };
        let mut next = PositionEstimate::tracked(
            WorldPoint::new(20.0, 20.0),
            TrackingSource::LocalTrack,
            Some(0.85),
            false,
        );
        next.heading_degrees = Some(-90.0);

        let resolved = resolve_tracking_position_heading(Some(&previous), next);

        assert_eq!(resolved.heading_degrees, Some(270.0));
    }
}
