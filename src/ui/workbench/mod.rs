mod forms;
mod minimap_picker;
mod page;
mod panels;
mod select;
mod theme;
mod tracker_pip;

use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use gpui::{
    AnyWindowHandle, AppContext, Bounds, Context, PathPromptOptions, Pixels, Render, SharedString,
    Subscription, Window, WindowBackgroundAppearance, WindowBounds, WindowHandle, WindowKind,
    WindowOptions,
};
use gpui_component::input::InputEvent;

use crate::{
    config::{AppConfig, CONFIG_FILE_NAME, save_config},
    domain::{
        geometry::WorldPoint,
        marker::{MarkerIconStyle, MarkerStyle},
        route::{RouteDocument, RouteId, RouteMetadata, RoutePoint, RoutePointId},
        theme::ThemePreference,
        tracker::{PositionEstimate, TrackerEngineKind, TrackerLifecycle, TrackingSource},
    },
    resources::{
        AssetManifest, BwikiPointRecord, BwikiResourceManager, BwikiTypeDefinition,
        RouteImportReport, RouteRepository, UiPreferences, UiPreferencesRepository,
        WorkspaceLoadReport, WorkspaceSnapshot, default_map_dimensions,
    },
    tracking::{
        TrackerSession, TrackingEvent, debug::TrackingDebugSnapshot, spawn_tracker_session,
    },
    ui::tile_cache::TileImageCache,
};

use self::{
    forms::{
        BwikiIconPickerItem, ConfigDraft, ConfigFormInputs, GroupDraft, GroupFormInputs,
        GroupInlineEditInputs, MarkerDraft, MarkerFormInputs, MarkerGroupPickerItem,
        PagedListState, PlannerRouteDraft, PointReorderTargetItem, RoutePlannerFormInputs,
        read_input_value, set_input_value,
    },
    minimap_picker::MinimapRegionPicker,
    page::{MapPage, SettingsPage, WorkbenchPage},
    panels::render_workbench,
    select::{SelectEvent, SelectState},
    theme::apply_theme_preference,
    tracker_pip::{TrackerPipWindow, apply_window_topmost},
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

#[derive(Debug, Clone, Default)]
pub(super) struct TrackerMapRenderSnapshot {
    pub(super) route_color_hex: Option<String>,
    pub(super) trail: Vec<WorldPoint>,
    pub(super) preview_position: Option<PositionEstimate>,
    pub(super) route_world: Vec<WorldPoint>,
    pub(super) point_visuals: Vec<MapPointRenderItem>,
    pub(super) selected_group_id: Option<RouteId>,
    pub(super) selected_point_id: Option<RoutePointId>,
    pub(super) follow_point: Option<WorldPoint>,
    pub(super) pip_always_on_top: bool,
}

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
enum AsyncTaskPhase {
    Idle,
    Working,
    Succeeded,
    Failed,
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
    pub(super) route_groups: Vec<RouteDocument>,
    pub(super) selected_group_id: Option<RouteId>,
    pub(super) selected_point_id: Option<RoutePointId>,
    pub(super) group_icon: MarkerIconStyle,
    pub(super) marker_icon: MarkerIconStyle,
    pub(super) theme_preference: ThemePreference,
    pub(super) auto_focus_enabled: bool,
    tracker_pending_action: Option<TrackerPendingAction>,
    tracker_status_text: SharedString,
    route_import_status: AsyncTaskStatus,
    spinner_frame: usize,
    pub(super) map_point_insert_armed: bool,
    moving_point_id: Option<RoutePointId>,
    moving_point_preview: Option<WorldPoint>,
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
    marker_group_picker: gpui::Entity<SelectState<MarkerGroupPickerItem>>,
    group_icon_picker: gpui::Entity<SelectState<BwikiIconPickerItem>>,
    marker_icon_picker: gpui::Entity<SelectState<BwikiIconPickerItem>>,
    point_reorder_target_id: Option<RoutePointId>,
    point_reorder_picker: gpui::Entity<SelectState<PointReorderTargetItem>>,
    minimap_region_picker_window: Option<AnyWindowHandle>,
    tracker_pip_window: Option<WindowHandle<TrackerPipWindow>>,
    tracker_pip_window_bounds: Option<WindowBounds>,
    tracker_pip_always_on_top: bool,
    tracker_pip_pending_open: bool,
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
        let point_reorder_picker = cx.new(|cx| SelectState::new(Vec::new(), None, 8, window, cx));
        let bwiki_tile_cache =
            TileImageCache::new(BWIKI_TILE_CACHE_MAX_ITEMS, BWIKI_TILE_CACHE_MAX_BYTES, cx);
        let ui_preferences_path = UiPreferencesRepository::path_for(&project_root);
        let (theme_preference, auto_focus_enabled, preferences_error) =
            match UiPreferencesRepository::load(&project_root) {
                Ok(preferences) => (preferences.theme_mode, preferences.auto_focus_enabled, None),
                Err(error) => (
                    ThemePreference::default(),
                    true,
                    Some(format!("载入界面偏好失败：{error:#}")),
                ),
            };

        let mut workbench = match WorkspaceSnapshot::load(project_root.clone()) {
            Ok(workspace) => {
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
                    project_root: project_root.to_string_lossy().into_owned().into(),
                    workspace,
                    tracker_session: None,
                    tracker_lifecycle: TrackerLifecycle::Idle,
                    selected_engine: TrackerEngineKind::RustTemplate,
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
                    route_groups,
                    selected_group_id,
                    selected_point_id: None,
                    group_icon: MarkerIconStyle::default(),
                    marker_icon: MarkerIconStyle::default(),
                    theme_preference,
                    auto_focus_enabled,
                    tracker_pending_action: None,
                    tracker_status_text: "追踪未启动。".into(),
                    route_import_status: AsyncTaskStatus::idle("尚未导入路线文件。"),
                    spinner_frame: 0,
                    map_point_insert_armed: false,
                    moving_point_id: None,
                    moving_point_preview: None,
                    ignore_next_tracker_mouse_up: false,
                    suspend_group_autosave: false,
                    suspend_point_autosave: false,
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    confirming_delete_point_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Tracker,
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
                    marker_group_picker: marker_group_picker.clone(),
                    group_icon_picker: group_icon_picker.clone(),
                    marker_icon_picker: marker_icon_picker.clone(),
                    point_reorder_target_id: None,
                    point_reorder_picker: point_reorder_picker.clone(),
                    minimap_region_picker_window: None,
                    tracker_pip_window: None,
                    tracker_pip_window_bounds: None,
                    tracker_pip_always_on_top: false,
                    tracker_pip_pending_open: false,
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
                let workspace = Arc::new(Self::empty_workspace(project_root.clone()));
                let (bwiki_resources, bwiki_manager_error) =
                    Self::new_bwiki_resource_manager(workspace.assets.bwiki_cache_dir.clone());
                let bwiki_version = bwiki_resources.version();
                Self {
                    project_root: project_root.to_string_lossy().into_owned().into(),
                    workspace,
                    tracker_session: None,
                    tracker_lifecycle: TrackerLifecycle::Failed,
                    selected_engine: TrackerEngineKind::RustTemplate,
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
                    route_groups: Vec::new(),
                    selected_group_id: None,
                    selected_point_id: None,
                    group_icon: MarkerIconStyle::default(),
                    marker_icon: MarkerIconStyle::default(),
                    theme_preference,
                    auto_focus_enabled,
                    tracker_pending_action: None,
                    tracker_status_text: "追踪未启动。".into(),
                    route_import_status: AsyncTaskStatus::idle("尚未导入路线文件。"),
                    spinner_frame: 0,
                    map_point_insert_armed: false,
                    moving_point_id: None,
                    moving_point_preview: None,
                    ignore_next_tracker_mouse_up: false,
                    suspend_group_autosave: false,
                    suspend_point_autosave: false,
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    confirming_delete_point_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Tracker,
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
                    marker_group_picker,
                    group_icon_picker,
                    marker_icon_picker,
                    point_reorder_target_id: None,
                    point_reorder_picker,
                    minimap_region_picker_window: None,
                    tracker_pip_window: None,
                    tracker_pip_window_bounds: None,
                    tracker_pip_always_on_top: false,
                    tracker_pip_pending_open: false,
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
        workbench.sync_editor_from_selection(window, cx);
        workbench.request_center_on_current_point();
        workbench.sync_bwiki_visibility_defaults();
        if let Some(message) = preferences_error {
            workbench.status_text = format!("{} {message}", workbench.status_text).into();
        }

        cx.spawn(async move |this, cx| {
            loop {
                let updated = this.update(cx, |this, cx| {
                    if this.poll_tracking_events()
                        || this.poll_bwiki_resources()
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
                (manager, None)
            }
            Err(error) => {
                let fallback = env::temp_dir()
                    .join("game-map-tracker-rs")
                    .join("bwiki-cache");
                let manager = BwikiResourceManager::new(fallback.clone())
                    .unwrap_or_else(|fallback_error| {
                        panic!(
                            "failed to initialize BWiki cache at {} ({error:#}) or fallback {} ({fallback_error:#})",
                            cache_dir.display(),
                            fallback.display()
                        )
                    });
                manager.ensure_dataset_loaded();
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

    pub(super) fn is_tracking_active(&self) -> bool {
        self.tracker_session.is_some()
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

    pub(super) const fn is_tracker_pip_always_on_top(&self) -> bool {
        self.tracker_pip_always_on_top
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

    pub(super) fn tracker_pip_topmost_label(&self) -> SharedString {
        if self.tracker_pip_always_on_top {
            "取消置顶".into()
        } else {
            "置顶".into()
        }
    }

    pub(super) fn tracker_pip_topmost_tooltip(&self) -> SharedString {
        if self.is_tracker_pip_open() {
            if self.tracker_pip_always_on_top {
                "追踪画中画当前会浮于其他窗口上方，点击可取消。".into()
            } else {
                "让追踪画中画浮于其他窗口上方。".into()
            }
        } else {
            "请先打开追踪画中画窗口。".into()
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

    pub(super) fn active_group_name(&self) -> SharedString {
        self.active_group().map_or_else(
            || "未选择路线".into(),
            |group| group.display_name().to_owned().into(),
        )
    }

    pub(super) fn current_position_label(&self) -> String {
        self.preview_position.as_ref().map_or_else(
            || "--".to_owned(),
            |position| format!("{:.0}, {:.0}", position.world.x, position.world.y),
        )
    }

    pub(super) fn current_point_label(&self) -> SharedString {
        self.selected_point().map_or_else(
            || "未选择节点".into(),
            |point| point.display_label().to_owned().into(),
        )
    }

    pub(super) const fn is_map_point_insert_armed(&self) -> bool {
        self.map_point_insert_armed
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

    fn active_group_points(&self) -> Vec<MapPointRenderItem> {
        self.active_group()
            .map(|group| {
                let last_index = group.points.len().saturating_sub(1);
                group
                    .points
                    .iter()
                    .enumerate()
                    .map(|(index, point)| MapPointRenderItem {
                        group_id: group.id.clone(),
                        point_id: point.id.clone(),
                        world: self
                            .moving_point_preview_world(&point.id)
                            .unwrap_or_else(|| point.world()),
                        style: group.effective_style(point),
                        is_start: index == 0,
                        is_end: index == last_index,
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

    fn active_group_route_worlds(&self) -> Vec<WorldPoint> {
        self.active_group()
            .map(|group| {
                group
                    .points
                    .iter()
                    .map(|point| {
                        self.moving_point_preview_world(&point.id)
                            .unwrap_or_else(|| point.world())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) fn tracker_map_render_snapshot(&self) -> TrackerMapRenderSnapshot {
        TrackerMapRenderSnapshot {
            route_color_hex: self
                .active_group()
                .map(|group| group.default_style.color_hex.clone()),
            trail: self.trail.clone(),
            preview_position: self.preview_position.clone(),
            route_world: self.active_group_route_worlds(),
            point_visuals: self.active_group_points(),
            selected_group_id: self.selected_group_id.clone(),
            selected_point_id: self.selected_point_id.clone(),
            follow_point: self
                .auto_focus_enabled
                .then(|| {
                    self.preview_position
                        .as_ref()
                        .map(|position| position.world)
                })
                .flatten(),
            pip_always_on_top: self.tracker_pip_always_on_top,
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
        self.active_group_points()
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
            self.select_point(point_id, window, cx);
            return true;
        }

        if !self.map_point_insert_armed && self.selected_point_id.is_some() {
            self.selected_point_id = None;
            self.confirming_delete_point_id = None;
            self.clear_selected_point_move_state();
            self.sync_editor_from_selection(window, cx);
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
        self.active_page = WorkbenchPage::Map;
        self.map_page = page;
        self.map_point_insert_armed = false;
        self.clear_selected_point_move_state();
        if matches!(page, MapPage::Tracker) {
            self.request_center_on_current_point();
        }
        self.status_text = format!("已切换到{}。", page).into();
    }

    pub(super) fn select_routes_page(&mut self) {
        self.active_page = WorkbenchPage::Markers;
        self.route_editor_map_view.request_fit();
        self.status_text = "已切换到路线管理。".into();
    }

    fn select_settings_page(&mut self, page: SettingsPage) {
        self.active_page = WorkbenchPage::Settings;
        self.settings_page = page;
        self.settings_nav_expanded = true;
        self.map_point_insert_armed = false;
        self.clear_selected_point_move_state();
        self.status_text = format!("设置页面已切换到{}。", page).into();
    }

    pub(super) fn toggle_map_point_insert_mode(&mut self) {
        if self.selected_group_id.is_none() {
            self.status_text = "请先选择一条路线，再开启添加节点。".into();
            return;
        }

        self.map_point_insert_armed = !self.map_point_insert_armed;
        if self.map_point_insert_armed {
            self.clear_selected_point_move_state();
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
            self.active_page = WorkbenchPage::Settings;
            self.settings_nav_expanded = true;
            self.status_text = "已切换到设置页面。".into();
            return;
        }

        self.settings_nav_expanded = !self.settings_nav_expanded;
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
                let message = if result.requested_count > 0 {
                    "当前选中的节点已失效，请重新选择后再规划。".to_owned()
                } else {
                    "请先在地图中选择至少一个节点。".to_owned()
                };
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
        self.persist_ui_preferences(&format!("界面主题已切换为 {}", self.theme_preference));
    }

    pub(super) fn set_auto_focus_enabled(&mut self, enabled: bool) {
        self.auto_focus_enabled = enabled;
        if enabled {
            self.request_center_on_current_point();
            self.persist_ui_preferences("自动聚焦已开启");
        } else {
            self.tracker_map_view.pending_center = None;
            self.persist_ui_preferences("自动聚焦已关闭");
        }
    }

    pub(super) fn select_group(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.confirming_delete_group_id.as_ref() != Some(&group_id) {
            self.confirming_delete_group_id = None;
        }
        self.clear_selected_point_move_state();
        self.selected_group_id = Some(group_id.clone());
        self.selected_point_id = None;
        self.confirming_delete_point_id = None;
        self.preview_cursor = None;
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

        self.selected_point_id = Some(point_id);
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

        self.selected_point_id = Some(point_id);
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

        self.selected_point_id = Some(point_id);
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

    pub(super) fn start_tracker(&mut self) {
        if self.is_tracking_active() || self.is_tracker_transition_pending() {
            return;
        }

        match spawn_tracker_session(self.workspace.clone(), self.selected_engine) {
            Ok(session) => {
                self.tracker_session = Some(session);
                self.tracker_pending_action = Some(TrackerPendingAction::Starting);
                self.tracker_lifecycle = TrackerLifecycle::Running;
                self.preview_position = None;
                self.trail.clear();
                self.frame_index = 0;
                self.last_source = None;
                self.last_match_score = None;
                self.debug_snapshot = None;
                self.tracker_status_text =
                    format!("正在启动 {} 追踪线程。", self.selected_engine).into();
                self.status_text = self.tracker_status_text.clone();
            }
            Err(error) => {
                self.tracker_pending_action = None;
                self.tracker_lifecycle = TrackerLifecycle::Failed;
                self.tracker_status_text = format!("启动追踪失败：{error:#}").into();
                self.status_text = self.tracker_status_text.clone();
            }
        }
    }

    pub(super) fn stop_tracker(&mut self, preserve_preview: bool) {
        self.tracker_pending_action = Some(TrackerPendingAction::Stopping);
        self.tracker_status_text = "正在停止追踪线程。".into();
        self.release_tracker_session();
        self.tracker_pending_action = None;
        self.tracker_lifecycle = TrackerLifecycle::Idle;
        self.tracker_status_text = "追踪线程已停止。".into();
        self.status_text = self.tracker_status_text.clone();
        if !preserve_preview {
            self.rebuild_preview();
        }
    }

    fn release_tracker_session(&mut self) {
        if let Some(mut session) = self.tracker_session.take() {
            session.stop();
        }
    }

    fn poll_tracking_events(&mut self) -> bool {
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
                    self.tracker_lifecycle = lifecycle;
                    match lifecycle {
                        TrackerLifecycle::Idle => {
                            self.tracker_pending_action = None;
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
                TrackingEvent::Debug(snapshot) => self.debug_snapshot = Some(snapshot),
                TrackingEvent::Error(message) => {
                    self.tracker_pending_action = None;
                    self.tracker_lifecycle = TrackerLifecycle::Failed;
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
        self.frame_index = status.frame_index;
        self.last_source = status.source;
        self.last_match_score = status.match_score;
        self.tracker_status_text = status.message.clone().into();
        self.status_text = status.message.into();
        self.tracker_lifecycle = status.lifecycle;
        if status.lifecycle == TrackerLifecycle::Running {
            self.tracker_pending_action = None;
        }
    }

    fn apply_tracking_position(&mut self, position: PositionEstimate) {
        self.last_source = Some(position.source);
        self.last_match_score = position.match_score;
        if self
            .trail
            .last()
            .copied()
            .is_none_or(|last| last != position.world)
        {
            self.trail.push(position.world);
        }
        if self.trail.len() > 2_048 {
            self.trail.drain(0..self.trail.len().saturating_sub(2_048));
        }

        self.preview_position = Some(position);
    }

    pub(super) fn toggle_engine(&mut self) {
        if self.is_tracking_active() {
            self.status_text = "请先停止当前追踪，再切换引擎。".into();
            return;
        }

        self.selected_engine = match self.selected_engine {
            TrackerEngineKind::RustTemplate => TrackerEngineKind::CandleAi,
            TrackerEngineKind::CandleAi => TrackerEngineKind::RustTemplate,
        };
        self.status_text = format!(
            "当前追踪方式已切换为 {}。传统图像匹配使用经典画面对比，AI 图像识别使用神经网络特征匹配。",
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
    ) -> anyhow::Result<()> {
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
            &self.config_form.sift_refresh_rate_ms,
            config.sift.refresh_rate_ms.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.sift_clahe_limit,
            config.sift.clahe_limit.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.sift_match_ratio,
            config.sift.match_ratio.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.sift_min_match_count,
            config.sift.min_match_count.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.sift_ransac_threshold,
            config.sift.ransac_threshold.to_string(),
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

    fn teleport_link_distance(&self) -> f32 {
        self.workspace.config.teleport_link_distance.max(0.0)
    }

    pub(super) fn save_app_config(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let draft = match ConfigDraft::read(self, cx) {
            Ok(draft) => draft,
            Err(message) => {
                self.status_text = message.into();
                return;
            }
        };

        if draft.config.window_geometry.trim().is_empty() {
            self.status_text = "window_geometry 不能为空。".into();
            return;
        }

        match save_config(&self.workspace.project_root, &draft.config) {
            Ok(path) => {
                self.update_workspace_config(draft.config);
                self.invalidate_bwiki_route_plan_preview();
                self.sync_config_form_from_workspace(window, cx);
                self.status_text = if self.is_tracking_active() {
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
            self.close_tracker_pip_window(cx);
            return;
        }
        if self.tracker_pip_pending_open {
            return;
        }

        self.tracker_pip_pending_open = true;
        self.status_text = "正在打开追踪画中画。".into();
        cx.defer_in(window, |this, window, cx| {
            this.open_tracker_pip_window(window, cx);
            cx.notify();
        });
    }

    pub(super) fn toggle_tracker_pip_always_on_top(&mut self, cx: &mut Context<Self>) {
        let always_on_top = !self.tracker_pip_always_on_top;

        if let Some(handle) = self.tracker_pip_window {
            match handle.update(cx, |_, pip_window, _| {
                apply_window_topmost(pip_window, always_on_top)
            }) {
                Ok(Ok(())) => {
                    self.tracker_pip_always_on_top = always_on_top;
                    self.status_text = if always_on_top {
                        "追踪画中画已置顶。".into()
                    } else {
                        "追踪画中画已取消置顶。".into()
                    };
                }
                Ok(Err(error)) => {
                    self.status_text = format!("切换追踪画中画置顶失败：{error:#}").into();
                }
                Err(_) => {
                    self.tracker_pip_window = None;
                    self.status_text = "追踪画中画窗口已经关闭。".into();
                }
            }
        } else {
            self.tracker_pip_always_on_top = always_on_top;
            self.status_text = if always_on_top {
                "已记住追踪画中画置顶设置，下次打开时生效。".into()
            } else {
                "已关闭追踪画中画置顶设置。".into()
            };
        }
    }

    pub(super) fn set_tracker_pip_always_on_top_from_pip(&mut self, always_on_top: bool) {
        self.tracker_pip_always_on_top = always_on_top;
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
                        let _ = workbench.update(cx, |this, _| {
                            this.handle_tracker_pip_window_closed();
                        });
                    }
                    true
                });

                cx.new(|cx| {
                    TrackerPipWindow::new(
                        workbench.clone(),
                        initial_camera,
                        initial_focus,
                        initial_snapshot.clone(),
                        bwiki_resources.clone(),
                        bwiki_tile_cache.clone(),
                        pip_window,
                        cx,
                    )
                })
            },
        );

        match open_result {
            Ok(handle) => {
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
                self.tracker_pip_window = None;
                self.status_text = format!("打开追踪画中画失败：{error:#}").into();
            }
        }
    }

    fn close_tracker_pip_window(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.tracker_pip_window.take() {
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

    pub(super) fn handle_tracker_pip_window_closed(&mut self) {
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

        match handle.update(cx, |pip, pip_window, cx| {
            pip.update_snapshot(snapshot.clone());
            let bounds = pip_window.window_bounds();
            pip_window.defer(cx, |pip_window, _| {
                pip_window.refresh();
            });
            bounds
        }) {
            Ok(bounds) => {
                self.tracker_pip_window_bounds = Some(bounds);
            }
            Err(_) => {
                self.tracker_pip_window = None;
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
            self.status_text = "已取消小地图取区。".into();
            let _ = handle.update(cx, |_, picker_window, _| {
                picker_window.remove_window();
            });
            return;
        }

        self.open_minimap_region_picker(window, cx);
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
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("小地图取区".into()),
                    appears_transparent: true,
                    ..Default::default()
                }),
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

                cx.new(|_| {
                    MinimapRegionPicker::new(workbench.clone(), main_window_handle, display_bounds)
                })
            },
        );

        match picker_result {
            Ok(handle) => {
                self.minimap_region_picker_window = Some(handle.into());
                self.status_text = "小地图取区已开启：拖动选择屏幕区域，右键取消。".into();
            }
            Err(error) => {
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
        let minimap_center_x = minimap.left as f32 + minimap.width as f32 / 2.0;
        let minimap_center_y = minimap.top as f32 + minimap.height as f32 / 2.0;
        if let Some(display) = displays.iter().find(|display| {
            screen_bounds_contains(display.bounds(), minimap_center_x, minimap_center_y)
        }) {
            return Some((display.id(), display.bounds()));
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
        set_input_value(
            &self.config_form.minimap_top,
            region.top.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_left,
            region.left.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_width,
            region.width.to_string(),
            window,
            cx,
        );
        set_input_value(
            &self.config_form.minimap_height,
            region.height.to_string(),
            window,
            cx,
        );
    }

    pub(super) fn finish_minimap_region_pick(
        &mut self,
        region: crate::config::CaptureRegion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.minimap_region_picker_window = None;

        let mut config = self.workspace.config.clone();
        config.minimap = region.clone();
        self.update_workspace_config(config.clone());
        self.sync_minimap_form_region(&region, window, cx);

        match save_config(&self.workspace.project_root, &config) {
            Ok(path) => {
                self.status_text = if self.is_tracking_active() {
                    format!(
                        "已更新小地图截图区域为 top {} / left {} / {}x{}，并保存到 {}。当前追踪需重启后才会应用新区域。",
                        region.top,
                        region.left,
                        region.width,
                        region.height,
                        path.display()
                    )
                    .into()
                } else {
                    format!(
                        "已更新小地图截图区域为 top {} / left {} / {}x{}，并保存到 {}。",
                        region.top,
                        region.left,
                        region.width,
                        region.height,
                        path.display()
                    )
                    .into()
                };
            }
            Err(error) => {
                self.status_text = format!(
                    "小地图截图区域已更新为 top {} / left {} / {}x{}，但写入配置失败：{error:#}",
                    region.top, region.left, region.width, region.height
                )
                .into();
            }
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
        };

        match UiPreferencesRepository::save(&self.workspace.project_root, &preferences) {
            Ok(path) => {
                self.status_text =
                    format!("{action_label}，偏好已保存到 {}。", path.display()).into();
            }
            Err(error) => {
                self.status_text = format!("保存界面偏好失败：{error:#}").into();
            }
        }
    }
}

impl Drop for TrackerWorkbench {
    fn drop(&mut self) {
        self.stop_tracker(true);
    }
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

#[derive(Debug, Clone, Copy)]
struct NearestTeleport {
    index: usize,
    distance: f32,
}

#[derive(Debug, Clone, Copy)]
struct BwikiSegmentPlan {
    total_cost: f32,
    source_teleport_index: Option<usize>,
    target_teleport_index: Option<usize>,
}

fn nearest_two_teleports(
    point: WorldPoint,
    teleports: &[BwikiPlannerResolvedPoint],
) -> [Option<NearestTeleport>; 2] {
    let mut nearest: [Option<NearestTeleport>; 2] = [None, None];

    for (index, teleport) in teleports.iter().enumerate() {
        let distance = planner_world_distance(point, teleport.record.world);
        let candidate = NearestTeleport { index, distance };
        if nearest[0].is_none_or(|current| distance < current.distance) {
            nearest[1] = nearest[0];
            nearest[0] = Some(candidate);
        } else if nearest[0].is_some_and(|current| current.index != index)
            && nearest[1].is_none_or(|current| distance < current.distance)
        {
            nearest[1] = Some(candidate);
        }
    }

    nearest
}

fn build_bwiki_segment_plan(
    from: WorldPoint,
    to: WorldPoint,
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> BwikiSegmentPlan {
    let direct_cost = planner_world_distance(from, to);
    if teleports.len() < 2 {
        return BwikiSegmentPlan {
            total_cost: direct_cost,
            source_teleport_index: None,
            target_teleport_index: None,
        };
    }

    let from_candidates = nearest_two_teleports(from, teleports);
    let to_candidates = nearest_two_teleports(to, teleports);
    let mut best_teleport_cost = f32::INFINITY;
    let mut best_pair = None;

    for from_candidate in from_candidates.into_iter().flatten() {
        for to_candidate in to_candidates.into_iter().flatten() {
            if from_candidate.index == to_candidate.index {
                continue;
            }
            let candidate_cost =
                from_candidate.distance + teleport_link_distance + to_candidate.distance;
            if candidate_cost < best_teleport_cost {
                best_teleport_cost = candidate_cost;
                best_pair = Some((from_candidate.index, to_candidate.index));
            }
        }
    }

    if let Some((source_teleport_index, target_teleport_index)) = best_pair
        && best_teleport_cost + 0.001 < direct_cost
    {
        return BwikiSegmentPlan {
            total_cost: best_teleport_cost,
            source_teleport_index: Some(source_teleport_index),
            target_teleport_index: Some(target_teleport_index),
        };
    }

    BwikiSegmentPlan {
        total_cost: direct_cost,
        source_teleport_index: None,
        target_teleport_index: None,
    }
}

fn build_bwiki_planner_cost_matrix(
    points: &[WorldPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> Vec<Vec<f32>> {
    (0..points.len())
        .map(|from_index| {
            (0..points.len())
                .map(|to_index| {
                    if from_index == to_index {
                        return 0.0;
                    }

                    build_bwiki_segment_plan(
                        points[from_index],
                        points[to_index],
                        teleports,
                        teleport_link_distance,
                    )
                    .total_cost
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn build_bwiki_planner_order(costs: &[Vec<f32>]) -> Vec<usize> {
    let point_count = costs.len();
    if point_count <= 1 {
        return (0..point_count).collect();
    }
    if point_count <= BWIKI_PLANNER_EXACT_LIMIT {
        return build_bwiki_planner_order_exact(costs);
    }

    let mut best_order = build_bwiki_planner_order_nearest(costs, 0);
    improve_bwiki_planner_order_2opt(&mut best_order, costs);
    let mut best_cost = ordered_bwiki_route_cost(&best_order, costs);

    for start in 1..point_count {
        let mut candidate = build_bwiki_planner_order_nearest(costs, start);
        improve_bwiki_planner_order_2opt(&mut candidate, costs);
        let candidate_cost = ordered_bwiki_route_cost(&candidate, costs);
        if candidate_cost < best_cost {
            best_order = candidate;
            best_cost = candidate_cost;
        }
    }

    best_order
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

fn improve_bwiki_planner_order_2opt(order: &mut [usize], costs: &[Vec<f32>]) {
    if order.len() < 4 {
        return;
    }

    let mut improved = true;
    while improved {
        improved = false;
        for start in 1..order.len() - 1 {
            for end in start + 1..order.len() {
                let left = order[start - 1];
                let first = order[start];
                let last = order[end];
                let next = order.get(end + 1).copied();

                let before = costs[left][first] + next.map_or(0.0, |next| costs[last][next]);
                let after = costs[left][last] + next.map_or(0.0, |next| costs[first][next]);
                if after + 0.001 < before {
                    order[start..=end].reverse();
                    improved = true;
                }
            }
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

    let mut lookup = HashMap::new();
    for definition in &dataset.types {
        let Some(records) = dataset.points_by_type.get(&definition.mark_type) else {
            continue;
        };
        for record in records {
            let key = BwikiPointKey::from_record(record);
            lookup.insert(
                key.clone(),
                BwikiPlannerResolvedPoint {
                    key,
                    record: record.clone(),
                    type_definition: Some(definition.clone()),
                },
            );
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

    let preview = if resolved.is_empty() {
        None
    } else if resolved.len() == 1 {
        Some(BwikiRoutePlanPreview {
            route_keys: vec![resolved[0].key.clone()],
            total_cost: 0.0,
        })
    } else {
        let teleports = resolve_bwiki_teleports_from_dataset(dataset);
        let worlds = resolved
            .iter()
            .map(|point| point.record.world)
            .collect::<Vec<_>>();
        let pair_costs =
            build_bwiki_planner_cost_matrix(&worlds, &teleports, teleport_link_distance);
        let order = build_bwiki_planner_order(&pair_costs);
        let ordered_points = order
            .iter()
            .filter_map(|index| resolved.get(*index))
            .cloned()
            .collect::<Vec<_>>();
        Some(build_bwiki_route_plan_preview(
            &ordered_points,
            &teleports,
            teleport_link_distance,
        ))
    };

    BwikiPlannerTaskResult {
        requested_count,
        normalized_selection_keys,
        preview,
    }
}

fn append_preview_key(route_keys: &mut Vec<BwikiPointKey>, key: &BwikiPointKey) {
    if route_keys.last() != Some(key) {
        route_keys.push(key.clone());
    }
}

fn build_bwiki_route_plan_preview(
    ordered_points: &[BwikiPlannerResolvedPoint],
    teleports: &[BwikiPlannerResolvedPoint],
    teleport_link_distance: f32,
) -> BwikiRoutePlanPreview {
    let Some(first_point) = ordered_points.first() else {
        return BwikiRoutePlanPreview::default();
    };

    let mut route_keys = vec![first_point.key.clone()];
    let mut total_cost = 0.0;

    for segment in ordered_points.windows(2) {
        let from = &segment[0];
        let to = &segment[1];
        let segment_plan = build_bwiki_segment_plan(
            from.record.world,
            to.record.world,
            teleports,
            teleport_link_distance,
        );
        total_cost += segment_plan.total_cost;

        if let Some(source_teleport_index) = segment_plan.source_teleport_index {
            append_preview_key(&mut route_keys, &teleports[source_teleport_index].key);
        }
        if let Some(target_teleport_index) = segment_plan.target_teleport_index {
            append_preview_key(&mut route_keys, &teleports[target_teleport_index].key);
        }
        append_preview_key(&mut route_keys, &to.key);
    }

    BwikiRoutePlanPreview {
        route_keys,
        total_cost,
    }
}

fn planner_point_to_route_point(
    point: BwikiPlannerResolvedPoint,
    default_style: &MarkerStyle,
) -> RoutePoint {
    let mut route_point = RoutePoint::new(point.record.title.clone(), point.record.world);
    let icon = point
        .type_definition
        .as_ref()
        .and_then(|definition| {
            (!definition.name.trim().is_empty())
                .then(|| MarkerIconStyle::new(definition.name.clone()))
        })
        .unwrap_or_else(|| MarkerIconStyle::new(point.record.mark_type.to_string()));
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
