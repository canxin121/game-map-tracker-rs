mod forms;
mod page;
mod panels;
mod theme;

use std::{collections::HashSet, env, path::PathBuf, sync::Arc, time::Duration};

use gpui::{AppContext, Context, PathPromptOptions, Render, SharedString, Subscription, Window};
use gpui_component::{
    input::InputEvent,
    select::{SelectEvent, SelectState},
};

use crate::{
    config::{AppConfig, CONFIG_FILE_NAME},
    domain::{
        geometry::WorldPoint,
        marker::{MarkerIconStyle, MarkerStyle},
        route::{RouteDocument, RouteId, RouteMetadata, RoutePoint, RoutePointId},
        theme::ThemePreference,
        tracker::{PositionEstimate, TrackerEngineKind, TrackerLifecycle, TrackingSource},
    },
    resources::{
        AssetManifest, BWIKI_WORLD_ZOOM, BwikiResourceManager, RouteImportReport, RouteRepository,
        UiPreferences, UiPreferencesRepository, WorkspaceLoadReport, WorkspaceSnapshot,
        default_map_dimensions,
    },
    tracking::{
        TrackerSession, TrackingEvent, debug::TrackingDebugSnapshot, spawn_tracker_session,
    },
};

use self::{
    forms::{
        GroupDraft, GroupFormInputs, GroupInlineEditInputs, MarkerDraft, MarkerFormInputs,
        MarkerGroupPickerDelegate, MarkerGroupPickerItem, PagedListState, read_input_value,
        set_input_value,
    },
    page::{MapPage, MarkersPage, SettingsPage, WorkbenchPage},
    panels::render_workbench,
    theme::apply_theme_preference,
};

#[derive(Debug, Clone)]
struct MapPointRenderItem {
    group_id: RouteId,
    point_id: RoutePointId,
    world: WorldPoint,
    style: MarkerStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PagedListKind {
    MapGroups,
    MarkerGroups,
    Points,
}

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
    pub(super) editing_group_id: Option<RouteId>,
    pub(super) pending_new_group_id: Option<RouteId>,
    pub(super) confirming_delete_group_id: Option<RouteId>,
    active_page: WorkbenchPage,
    map_page: MapPage,
    pub(super) map_nav_expanded: bool,
    markers_page: MarkersPage,
    pub(super) markers_nav_expanded: bool,
    settings_page: SettingsPage,
    settings_nav_expanded: bool,
    map_group_list: PagedListState,
    marker_group_list: PagedListState,
    point_list: PagedListState,
    marker_group_picker: gpui::Entity<SelectState<MarkerGroupPickerDelegate>>,
    pub(super) ui_preferences_path: PathBuf,
    pub(super) bwiki_resources: BwikiResourceManager,
    pub(super) bwiki_version: u64,
    pub(super) bwiki_visible_mark_types: HashSet<u32>,
    pub(super) bwiki_expanded_categories: HashSet<String>,
    bwiki_visibility_initialized: bool,
    group_form: GroupFormInputs,
    group_inline_edit: GroupInlineEditInputs,
    marker_form: MarkerFormInputs,
    subscriptions: Vec<Subscription>,
}

impl TrackerWorkbench {
    pub fn new(project_root: PathBuf, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let group_form = GroupFormInputs::new(window, cx);
        let group_inline_edit = GroupInlineEditInputs::new(window, cx);
        let marker_form = MarkerFormInputs::new(window, cx);
        let map_group_list = PagedListState::new(window, cx, "搜索地图中的标记组", 8);
        let marker_group_list = PagedListState::new(window, cx, "搜索标记组", 8);
        let point_list = PagedListState::new(window, cx, "搜索当前组节点", 10);
        let marker_group_picker = cx.new(|cx| {
            SelectState::new(MarkerGroupPickerDelegate::new(Vec::new()), None, window, cx)
        });
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
                let selected_point_id = selected_group_id
                    .as_ref()
                    .and_then(|group_id| route_groups.iter().find(|group| &group.id == group_id))
                    .and_then(|group| group.points.first())
                    .map(|point| point.id.clone());
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
                    status_text: "数据目录已经完成解析。地图页只负责查看地图和运行 tracker，标记页负责管理 routes 目录下的标记组与节点。".into(),
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
                    selected_point_id,
                    group_icon: MarkerIconStyle::default(),
                    marker_icon: MarkerIconStyle::default(),
                    theme_preference,
                    auto_focus_enabled,
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Tracker,
                    map_nav_expanded: true,
                    markers_page: MarkersPage::default(),
                    markers_nav_expanded: false,
                    settings_page: SettingsPage::default(),
                    settings_nav_expanded: false,
                    map_group_list: map_group_list.clone(),
                    marker_group_list: marker_group_list.clone(),
                    point_list: point_list.clone(),
                    marker_group_picker: marker_group_picker.clone(),
                    ui_preferences_path: ui_preferences_path.clone(),
                    bwiki_resources,
                    bwiki_version,
                    bwiki_visible_mark_types: HashSet::new(),
                    bwiki_expanded_categories: HashSet::new(),
                    bwiki_visibility_initialized: false,
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
                    editing_group_id: None,
                    pending_new_group_id: None,
                    confirming_delete_group_id: None,
                    active_page: WorkbenchPage::Map,
                    map_page: MapPage::Tracker,
                    map_nav_expanded: true,
                    markers_page: MarkersPage::default(),
                    markers_nav_expanded: false,
                    settings_page: SettingsPage::default(),
                    settings_nav_expanded: false,
                    map_group_list,
                    marker_group_list,
                    point_list,
                    marker_group_picker,
                    ui_preferences_path: ui_preferences_path.clone(),
                    bwiki_resources,
                    bwiki_version,
                    bwiki_visible_mark_types: HashSet::new(),
                    bwiki_expanded_categories: HashSet::new(),
                    bwiki_visibility_initialized: false,
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
        let marker_group_picker = workbench.marker_group_picker.clone();
        workbench.subscriptions.push(cx.subscribe_in(
            &marker_group_picker,
            window,
            |this, _, event: &SelectEvent<MarkerGroupPickerDelegate>, window, cx| {
                let SelectEvent::Confirm(Some(group_id)) = event else {
                    return;
                };
                this.select_group(group_id.clone(), window, cx);
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
        workbench.sync_editor_from_selection(window, cx);
        workbench.request_center_on_current_point();
        workbench.sync_bwiki_visibility_defaults();
        if let Some(message) = preferences_error {
            workbench.status_text = format!("{} {message}", workbench.status_text).into();
        }

        cx.spawn(async move |this, cx| {
            loop {
                let updated = this.update(cx, |this, cx| {
                    if this.poll_tracking_events() || this.poll_bwiki_resources() {
                        cx.notify();
                    }
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
                let startup_error = manager
                    .ensure_stitched_map_ready_blocking(BWIKI_WORLD_ZOOM)
                    .err()
                    .map(|error| format!("启动时预构建 BWiki 整图失败：{error:#}"));
                (manager, startup_error)
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
                let startup_error = manager
                    .ensure_stitched_map_ready_blocking(BWIKI_WORLD_ZOOM)
                    .err()
                    .map(|map_error| format!("启动时预构建 BWiki 整图失败：{map_error:#}"));
                (
                    manager,
                    Some(match startup_error {
                        Some(startup_error) => format!(
                            "BWiki 缓存目录 {} 初始化失败，已临时改用 {}：{error:#} {startup_error}",
                            cache_dir.display(),
                            fallback.display()
                        ),
                        None => format!(
                            "BWiki 缓存目录 {} 初始化失败，已临时改用 {}：{error:#}",
                            cache_dir.display(),
                            fallback.display()
                        ),
                    }),
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

    pub(super) const fn is_auto_focus_enabled(&self) -> bool {
        self.auto_focus_enabled
    }

    pub(super) fn active_group_name(&self) -> SharedString {
        self.active_group().map_or_else(
            || "未选择标记组".into(),
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
                group
                    .points
                    .iter()
                    .map(|point| MapPointRenderItem {
                        group_id: group.id.clone(),
                        point_id: point.id.clone(),
                        world: point.world(),
                        style: group.effective_style(point),
                    })
                    .collect()
            })
            .unwrap_or_default()
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

    fn select_map_page(&mut self, page: MapPage) {
        self.active_page = WorkbenchPage::Map;
        self.map_page = page;
        self.map_nav_expanded = true;
        if matches!(page, MapPage::Tracker) {
            self.request_center_on_current_point();
        }
        self.status_text = format!("地图页面已切换到{}。", page).into();
    }

    fn select_markers_page(&mut self, page: MarkersPage) {
        self.active_page = WorkbenchPage::Markers;
        self.markers_page = page;
        self.markers_nav_expanded = true;
        self.status_text = format!("标记页面已切换到{}。", page).into();
    }

    fn select_settings_page(&mut self, page: SettingsPage) {
        self.active_page = WorkbenchPage::Settings;
        self.settings_page = page;
        self.settings_nav_expanded = true;
        self.status_text = format!("设置页面已切换到{}。", page).into();
    }

    fn toggle_map_navigation(&mut self) {
        if self.active_page != WorkbenchPage::Map {
            self.active_page = WorkbenchPage::Map;
            self.map_nav_expanded = true;
            self.status_text = "已切换到地图页面。".into();
            return;
        }

        self.map_nav_expanded = !self.map_nav_expanded;
        self.status_text = if self.map_nav_expanded {
            "已展开地图导航。".into()
        } else {
            "已收起地图导航。".into()
        };
    }

    fn toggle_marker_navigation(&mut self) {
        if self.active_page != WorkbenchPage::Markers {
            self.active_page = WorkbenchPage::Markers;
            self.markers_nav_expanded = true;
            self.status_text = "已切换到标记页面。".into();
            return;
        }

        self.markers_nav_expanded = !self.markers_nav_expanded;
        self.status_text = if self.markers_nav_expanded {
            "已展开标记导航。".into()
        } else {
            "已收起标记导航。".into()
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

    fn poll_bwiki_resources(&mut self) -> bool {
        let version = self.bwiki_resources.version();
        let mut changed = version != self.bwiki_version;
        if changed {
            self.bwiki_version = version;
        }
        if self.sync_bwiki_visibility_defaults() {
            changed = true;
        }
        changed
    }

    pub(super) fn sync_bwiki_visibility_defaults(&mut self) -> bool {
        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return false;
        };

        let mut changed = false;
        if self.bwiki_expanded_categories.is_empty() {
            self.bwiki_expanded_categories = dataset.sorted_category_names().into_iter().collect();
            changed = true;
        }
        if !self.bwiki_visibility_initialized {
            self.bwiki_visible_mark_types = dataset
                .types
                .iter()
                .filter(|item| item.point_count > 0)
                .map(|item| item.mark_type)
                .collect();
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

    pub(super) fn bwiki_visible_point_count(&self) -> usize {
        let Some(dataset) = self.bwiki_resources.dataset_snapshot() else {
            return 0;
        };
        dataset
            .points_by_type
            .iter()
            .filter(|(mark_type, _)| self.bwiki_visible_mark_types.contains(mark_type))
            .map(|(_, points)| points.len())
            .sum()
    }

    pub(super) fn bwiki_visible_type_count(&self) -> usize {
        self.bwiki_visible_mark_types.len()
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

    fn set_point_page(&mut self, page: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.set_paged_list_page(PagedListKind::Points, page, window, cx);
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
            self.status_text = "待编辑的标记组不存在。".into();
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
        self.status_text = "已进入标记组行内编辑。".into();
    }

    pub(super) fn create_group_inline_item(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        set_input_value(&self.marker_group_list.search, "", window, cx);
        self.marker_group_list.page = 0;

        let mut group = RouteDocument::new("");
        group.notes.clear();
        group.default_style = MarkerStyle::default();
        group.metadata = RouteMetadata {
            id: group.id.clone(),
            file_name: self.allocate_group_file_name("", None),
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
        self.sync_editor_from_selection(window, cx);
        self.start_group_inline_edit(group_id, window, cx);
        self.defer_marker_group_page_to_group(
            self.selected_group_id
                .clone()
                .expect("new group should be selected"),
            window,
            cx,
        );
        self.status_text = "已创建新的空白标记组，请先填写标题和注释。".into();
    }

    pub(super) fn commit_inline_group_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.editing_group_id.clone() else {
            return;
        };

        let name = read_input_value(&self.group_inline_edit.name, cx);
        if name.trim().is_empty() {
            self.status_text = "标记组标题不能为空。".into();
            return;
        }
        let description = read_input_value(&self.group_inline_edit.description, cx);

        let (needs_file_name, desired_file_name_source) = {
            let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            else {
                self.editing_group_id = None;
                self.pending_new_group_id = None;
                self.status_text = "待保存的标记组不存在。".into();
                return;
            };

            group.name = name;
            group.notes = description;
            let display_name = group.display_name().to_owned();
            let needs_file_name = group.metadata.file_name.trim().is_empty();
            group.metadata.display_name = display_name.clone();
            (needs_file_name, display_name)
        };

        if needs_file_name {
            let file_name =
                self.allocate_group_file_name(&desired_file_name_source, Some(&group_id));
            if let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            {
                group.metadata.file_name = file_name;
            }
        }

        if self.persist_group(&group_id, "标记组已保存") {
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
            self.status_text = "已取消新建标记组。".into();
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
        self.status_text = "已取消标记组行内编辑。".into();
    }

    pub(super) fn begin_group_delete_confirmation(
        &mut self,
        group_id: RouteId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((first_point_id, group_name)) = self
            .route_groups
            .iter()
            .find(|group| group.id == group_id)
            .map(|group| {
                (
                    group.points.first().map(|point| point.id.clone()),
                    group.display_name().to_owned(),
                )
            })
        else {
            self.confirming_delete_group_id = None;
            self.status_text = "待删除的标记组不存在。".into();
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
        self.selected_point_id = first_point_id;
        self.preview_cursor = self.selected_point_index();
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
        self.status_text = format!("请确认是否删除标记组「{}」。", group_name).into();
    }

    pub(super) fn cancel_group_delete_confirmation(&mut self, group_id: RouteId) {
        if self.confirming_delete_group_id.as_ref() == Some(&group_id) {
            self.confirming_delete_group_id = None;
            self.status_text = "已取消删除标记组。".into();
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
            self.status_text = "选中的标记组不存在。".into();
            return;
        };

        let removed = self.route_groups[index].clone();
        let removed_path = self.route_file_path(&removed.metadata.file_name);
        let removed_file_exists = removed_path.exists();
        if delete_persisted_file {
            if removed_file_exists && let Err(error) = RouteRepository::delete(&removed_path) {
                self.status_text = format!("删除标记组文件失败：{error:#}").into();
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
        self.selected_point_id = self
            .selected_group_id
            .as_ref()
            .and_then(|current_id| {
                self.route_groups
                    .iter()
                    .find(|group| &group.id == current_id)
            })
            .and_then(|group| group.points.first())
            .map(|point| point.id.clone());

        self.sync_editor_from_selection(window, cx);
        self.status_text = if delete_persisted_file && removed_file_exists {
            format!("标记组「{}」已删除。", removed.display_name()).into()
        } else {
            format!("已移除未保存的标记组占位「{}」。", removed.display_name()).into()
        };
    }

    pub(super) fn toggle_selected_group_visible(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一个标记组。".into();
            return;
        };
        let visible = {
            let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            else {
                self.status_text = "选中的标记组不存在。".into();
                return;
            };
            group.visible = !group.visible;
            group.visible
        };
        if self.persist_group(
            &group_id,
            if visible {
                "标记组已设为显示"
            } else {
                "标记组已设为隐藏"
            },
        ) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn toggle_selected_group_looped(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一个标记组。".into();
            return;
        };
        let looped = {
            let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            else {
                self.status_text = "选中的标记组不存在。".into();
                return;
            };
            group.looped = !group.looped;
            group.looped
        };
        if self.persist_group(
            &group_id,
            if looped {
                "标记组已切换为闭环路径"
            } else {
                "标记组已切换为非闭环路径"
            },
        ) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    fn marker_group_picker_items(&self) -> MarkerGroupPickerDelegate {
        MarkerGroupPickerDelegate::new(
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
                .collect::<Vec<_>>(),
        )
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
        self.selected_group_id = Some(group_id.clone());
        self.selected_point_id = self
            .route_groups
            .iter()
            .find(|group| group.id == group_id)
            .and_then(|group| group.points.first())
            .map(|point| point.id.clone());
        self.preview_cursor = Some(self.selected_point_index().unwrap_or(0));
        if !self.is_tracking_active() {
            self.rebuild_preview();
        }
        self.request_center_on_current_point();
        self.sync_editor_from_selection(window, cx);
        if let Some(group) = self.active_group() {
            self.status_text = format!(
                "已选中标记组「{}」，共 {} 个节点。",
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
            self.status_text = "实时追踪运行中。请先停止追踪，再手动步进当前标记组节点。".into();
            return;
        }

        let Some(group) = self.active_group().cloned() else {
            self.status_text = "请先选择一个标记组。".into();
            return;
        };
        if group.points.is_empty() {
            self.status_text = "当前标记组没有节点。".into();
            return;
        }

        let current =
            self.preview_cursor
                .unwrap_or_else(|| self.selected_point_index().unwrap_or(0)) as isize;
        let last = group.points.len().saturating_sub(1) as isize;
        let next = (current + delta).clamp(0, last) as usize;
        self.preview_cursor = Some(next);
        if let Some(point) = group.points.get(next) {
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
        if self.is_tracking_active() {
            return;
        }

        match spawn_tracker_session(self.workspace.clone(), self.selected_engine) {
            Ok(session) => {
                self.tracker_session = Some(session);
                self.tracker_lifecycle = TrackerLifecycle::Running;
                self.preview_position = None;
                self.trail.clear();
                self.frame_index = 0;
                self.last_source = None;
                self.last_match_score = None;
                self.debug_snapshot = None;
                self.status_text = format!("正在启动 {} 追踪线程。", self.selected_engine).into();
            }
            Err(error) => {
                self.tracker_lifecycle = TrackerLifecycle::Failed;
                self.status_text = format!("启动追踪失败：{error:#}").into();
            }
        }
    }

    pub(super) fn stop_tracker(&mut self, preserve_preview: bool) {
        self.release_tracker_session();
        self.tracker_lifecycle = TrackerLifecycle::Idle;
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
                            self.status_text = "追踪线程已停止。".into();
                            should_release_session = true;
                        }
                        TrackerLifecycle::Running => {}
                        TrackerLifecycle::Failed => should_release_session = true,
                    }
                }
                TrackingEvent::Status(status) => self.apply_tracking_status(status),
                TrackingEvent::Position(position) => self.apply_tracking_position(position),
                TrackingEvent::Debug(snapshot) => self.debug_snapshot = Some(snapshot),
                TrackingEvent::Error(message) => {
                    self.tracker_lifecycle = TrackerLifecycle::Failed;
                    self.status_text = format!("追踪线程异常：{message}").into();
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
        self.status_text = status.message.into();
        self.tracker_lifecycle = status.lifecycle;
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

        if let Some(group_id) = self.selected_group_id.clone() {
            let point_exists = self.selected_point_id.as_ref().is_some_and(|point_id| {
                self.route_groups
                    .iter()
                    .find(|group| group.id == group_id)
                    .and_then(|group| group.find_point(point_id))
                    .is_some()
            });
            if !point_exists {
                self.selected_point_id = self
                    .route_groups
                    .iter()
                    .find(|group| group.id == group_id)
                    .and_then(|group| group.points.first())
                    .map(|point| point.id.clone());
            }
        } else {
            self.selected_point_id = None;
        }

        self.sync_marker_group_picker_state(window, cx);

        let selected_group = self.active_group().cloned();
        self.group_icon = selected_group
            .as_ref()
            .map(|group| group.default_style.icon)
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

        let selected_point = self.selected_point().cloned();
        self.marker_icon = selected_point
            .as_ref()
            .map(|point| point.style.icon)
            .or_else(|| {
                selected_group
                    .as_ref()
                    .map(|group| group.default_style.icon)
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

        self.sync_visible_list_pages(window, cx);
    }
    pub(super) fn new_point_draft(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.selected_point_id = None;
        self.marker_icon = self
            .active_group()
            .map(|group| group.default_style.icon)
            .unwrap_or_default();
        set_input_value(&self.marker_form.label, "", window, cx);
        set_input_value(&self.marker_form.note, "", window, cx);
        let world = self.default_marker_world();
        set_input_value(&self.marker_form.x, format!("{:.0}", world.x), window, cx);
        set_input_value(&self.marker_form.y, format!("{:.0}", world.y), window, cx);
        set_input_value(&self.marker_form.color_hex, "#4ECDC4", window, cx);
        set_input_value(&self.marker_form.size_px, "24", window, cx);
        self.status_text = "已切换到新建节点草稿。".into();
    }

    pub(super) fn save_group(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let draft = match GroupDraft::read(self, cx) {
            Ok(draft) => draft,
            Err(message) => {
                self.status_text = message.into();
                return;
            }
        };

        let target_group_id = if let Some(group_id) = self.selected_group_id.clone() {
            if let Some(group) = self
                .route_groups
                .iter_mut()
                .find(|group| group.id == group_id)
            {
                group.name = draft.name;
                group.notes = draft.description;
                group.default_style = draft.style;
                group.metadata.display_name = group.name.clone();
                group_id
            } else {
                self.selected_group_id = None;
                self.save_group(window, cx);
                return;
            }
        } else {
            let mut group = RouteDocument::new(draft.name);
            group.notes = draft.description;
            group.default_style = draft.style;
            group.metadata = RouteMetadata {
                id: group.id.clone(),
                file_name: self.allocate_group_file_name(group.display_name(), None),
                display_name: group.display_name().to_owned(),
            };
            let group_id = group.id.clone();
            self.route_groups.push(group);
            self.selected_group_id = Some(group_id.clone());
            self.selected_point_id = None;
            group_id
        };

        if self.persist_group(&target_group_id, "标记组已保存") {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn delete_selected_group(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "当前没有选中的标记组。".into();
            return;
        };
        self.begin_group_delete_confirmation(group_id, window, cx);
    }

    pub(super) fn import_route_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let paths_receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("选择要导入的标记组文件".into()),
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
        let folder_receiver = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("选择包含标记组文件的文件夹".into()),
        });

        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(mut folders))) = folder_receiver.await else {
                return;
            };
            let Some(folder) = folders.pop() else {
                return;
            };
            let scan_folder = folder.clone();

            let paths_result = cx
                .background_executor()
                .spawn(async move { RouteRepository::collect_import_files(&scan_folder) })
                .await;

            this.update_in(cx, |this, window, cx| match paths_result {
                Ok(paths) if paths.is_empty() => {
                    this.status_text =
                        format!("目录 {} 中没有可导入的 JSON 标记组文件。", folder.display())
                            .into();
                    cx.notify();
                }
                Ok(paths) => {
                    this.import_route_paths(paths, window, cx);
                    cx.notify();
                }
                Err(error) => {
                    this.status_text = format!("扫描导入目录失败：{error:#}").into();
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
            self.status_text = "没有选择任何可导入的标记组文件。".into();
            return;
        }

        match RouteRepository::import_paths(paths, &self.workspace.assets.routes_dir) {
            Ok(report) => self.apply_import_report(report, window, cx),
            Err(error) => {
                self.status_text = format!("导入标记组失败：{error:#}").into();
            }
        }
    }

    fn apply_import_report(
        &mut self,
        report: RouteImportReport,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if report.imported_count == 0 {
            if let Some(first_error) = report.failed_sources.first() {
                self.status_text = format!(
                    "没有成功导入任何标记组，共 {} 个文件失败。首个错误：{}",
                    report.failed_sources.len(),
                    first_error
                )
                .into();
            } else {
                self.status_text = "没有发现可导入的标记组文件。".into();
            }
            return;
        }

        if let Err(error) =
            self.reload_route_groups(report.first_imported_group_id.as_ref(), window, cx)
        {
            self.status_text = format!("导入完成，但刷新标记组失败：{error:#}").into();
            return;
        }

        let mut message = format!(
            "已导入 {} 个标记组，共 {} 个节点。",
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
        self.status_text = message.into();
    }

    pub(super) fn save_point(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "请先选择一个标记组，再保存节点。".into();
            return;
        };
        let draft = match MarkerDraft::read(self, cx) {
            Ok(draft) => draft,
            Err(message) => {
                self.status_text = message.into();
                return;
            }
        };

        let selected_point_id = self.selected_point_id.clone();
        let mut saved_point_id = None;
        if let Some(group) = self
            .route_groups
            .iter_mut()
            .find(|group| group.id == group_id)
        {
            if let Some(point_id) = selected_point_id {
                if let Some(point) = group.find_point_mut(&point_id) {
                    point.label = Some(draft.label);
                    point.note = draft.note;
                    point.x = draft.world.x;
                    point.y = draft.world.y;
                    point.style = draft.style;
                    saved_point_id = Some(point.id.clone());
                } else {
                    self.selected_point_id = None;
                    self.save_point(window, cx);
                    return;
                }
            } else {
                let mut point = RoutePoint::new(draft.label, draft.world);
                point.note = draft.note;
                point.style = draft.style;
                saved_point_id = Some(point.id.clone());
                group.points.push(point);
            }
        }

        if let Some(point_id) = saved_point_id {
            self.selected_point_id = Some(point_id);
        }

        if self.persist_group(&group_id, "节点已保存") {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn delete_selected_point(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group_id) = self.selected_group_id.clone() else {
            self.status_text = "当前没有选中的标记组。".into();
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
            self.selected_point_id = group.points.first().map(|point| point.id.clone());
        }

        let Some(label) = removed_label else {
            self.status_text = "选中的节点不存在。".into();
            return;
        };

        if self.persist_group(&group_id, &format!("节点「{label}」已删除")) {
            self.sync_editor_from_selection(window, cx);
        }
    }

    pub(super) fn use_preview_position_for_point(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(position) = self.preview_position.as_ref() else {
            self.status_text = "当前没有可用的预览 / 追踪坐标。".into();
            return;
        };

        set_input_value(
            &self.marker_form.x,
            format!("{:.0}", position.world.x),
            window,
            cx,
        );
        set_input_value(
            &self.marker_form.y,
            format!("{:.0}", position.world.y),
            window,
            cx,
        );
        self.status_text = "已用当前预览 / 追踪坐标填充节点位置。".into();
    }

    pub(super) fn use_map_center_for_point(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let world = self
            .tracker_map_view
            .camera
            .screen_to_world(WorldPoint::new(
                self.tracker_map_view.viewport.width * 0.5,
                self.tracker_map_view.viewport.height * 0.5,
            ));
        set_input_value(&self.marker_form.x, format!("{:.0}", world.x), window, cx);
        set_input_value(&self.marker_form.y, format!("{:.0}", world.y), window, cx);
        self.status_text = "已用当前画布中心填充节点位置。".into();
    }

    pub(super) fn focus_selected_point(&mut self) {
        if let Some((world, label)) = self
            .selected_point()
            .map(|point| (point.world(), point.display_label().to_owned()))
        {
            self.tracker_map_view.center_on_or_queue(world);
            self.status_text = format!("地图已居中到节点「{}」。", label).into();
        } else {
            self.status_text = "当前没有选中的节点。".into();
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
        let Some(index) = self
            .route_groups
            .iter()
            .position(|group| &group.id == group_id)
        else {
            self.status_text = "待保存的标记组不存在。".into();
            return false;
        };

        let file_name = if self.route_groups[index]
            .metadata
            .file_name
            .trim()
            .is_empty()
        {
            self.allocate_group_file_name(self.route_groups[index].display_name(), Some(group_id))
        } else {
            self.route_groups[index].metadata.file_name.clone()
        };

        let mut group = self.route_groups[index].clone().normalized();
        group.metadata.id = group.id.clone();
        group.metadata.file_name = file_name.clone();
        group.metadata.display_name = group.display_name().to_owned();

        let path = self.route_file_path(&file_name);
        match RouteRepository::save(&path, &group) {
            Ok(()) => {
                self.route_groups[index] = group;
                self.sync_workspace_routes_snapshot();
                self.status_text = format!("{action_label}，已保存到 {}。", path.display()).into();
                true
            }
            Err(error) => {
                self.status_text = format!("保存标记组失败：{error:#}").into();
                false
            }
        }
    }

    fn allocate_group_file_name(
        &self,
        desired_name: &str,
        current_group_id: Option<&RouteId>,
    ) -> String {
        let preferred = RouteRepository::suggested_file_name(desired_name);
        let (stem, ext) = preferred
            .rsplit_once('.')
            .map_or((preferred.as_str(), "json"), |(stem, ext)| (stem, ext));

        let mut candidate = preferred.clone();
        let mut counter = 2usize;
        while self.route_groups.iter().any(|group| {
            current_group_id.is_none_or(|current| &group.id != current)
                && group.metadata.file_name.eq_ignore_ascii_case(&candidate)
        }) {
            candidate = format!("{stem}_{counter}.{ext}");
            counter += 1;
        }
        candidate
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

impl Render for TrackerWorkbench {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        render_workbench(self, cx)
    }
}
