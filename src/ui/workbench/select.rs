use gpui::{
    Action, AnyElement, App, AppContext, Bounds, ClickEvent, Context, ElementId, Entity,
    EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement, KeyBinding, Length,
    ParentElement as _, Pixels, Render, RenderOnce, SharedString, StatefulInteractiveElement as _,
    StyleRefinement, Styled, Subscription, Window, actions, anchored, canvas, deferred, div,
    prelude::FluentBuilder as _, px, rems,
};
use gpui_component::{
    ActiveTheme as _, Icon, IconName, Sizable, Size, StyleSized as _, StyledExt as _,
    input::{Input, InputEvent, InputState},
    scroll::ScrollableElement as _,
    select::SelectItem,
};
use serde::Deserialize;

const CONTEXT: &str = "WorkbenchPagedSelect";

actions!(
    workbench_paged_select,
    [
        WorkbenchPagedSelectCancel,
        WorkbenchPagedSelectUp,
        WorkbenchPagedSelectDown
    ]
);

#[derive(Clone, Action, PartialEq, Eq, Deserialize)]
#[action(namespace = workbench_paged_select, no_json)]
struct WorkbenchPagedSelectConfirm {
    secondary: bool,
}

pub fn init(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("up", WorkbenchPagedSelectUp, Some(CONTEXT)),
        KeyBinding::new("down", WorkbenchPagedSelectDown, Some(CONTEXT)),
        KeyBinding::new(
            "enter",
            WorkbenchPagedSelectConfirm { secondary: false },
            Some(CONTEXT),
        ),
        KeyBinding::new(
            "secondary-enter",
            WorkbenchPagedSelectConfirm { secondary: true },
            Some(CONTEXT),
        ),
        KeyBinding::new("escape", WorkbenchPagedSelectCancel, Some(CONTEXT)),
    ]);
}

struct SelectOptions {
    style: StyleRefinement,
    size: Size,
    icon: Option<Icon>,
    placeholder: Option<SharedString>,
    search_placeholder: Option<SharedString>,
    empty_message: Option<SharedString>,
    menu_width: Length,
    disabled: bool,
    appearance: bool,
}

impl Default for SelectOptions {
    fn default() -> Self {
        Self {
            style: StyleRefinement::default(),
            size: Size::default(),
            icon: None,
            placeholder: None,
            search_placeholder: None,
            empty_message: None,
            menu_width: Length::Auto,
            disabled: false,
            appearance: true,
        }
    }
}

struct ActionMenuOptions {
    style: StyleRefinement,
    size: Size,
    icon: Option<Icon>,
    label: Option<SharedString>,
    center_label: bool,
    search_placeholder: Option<SharedString>,
    empty_message: Option<SharedString>,
    menu_width: Length,
    disabled: bool,
}

impl Default for ActionMenuOptions {
    fn default() -> Self {
        Self {
            style: StyleRefinement::default(),
            size: Size::default(),
            icon: None,
            label: None,
            center_label: false,
            search_placeholder: None,
            empty_message: None,
            menu_width: Length::Auto,
            disabled: false,
        }
    }
}

pub enum SelectEvent<I: SelectItem>
where
    I::Value: Clone + PartialEq + 'static,
{
    Confirm(Option<I::Value>),
}

pub enum ActionMenuEvent<I: SelectItem>
where
    I::Value: Clone + PartialEq + 'static,
{
    Confirm(I::Value),
}

pub struct SelectState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    focus_handle: FocusHandle,
    options: SelectOptions,
    items: Vec<I>,
    search_input: Entity<InputState>,
    page_input: Entity<InputState>,
    page: usize,
    page_size: usize,
    bounds: Bounds<Pixels>,
    open: bool,
    selected_value: Option<I::Value>,
    active_value: Option<I::Value>,
    _subscriptions: Vec<Subscription>,
}

pub struct ActionMenuState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    focus_handle: FocusHandle,
    options: ActionMenuOptions,
    items: Vec<I>,
    search_input: Entity<InputState>,
    page_input: Entity<InputState>,
    page: usize,
    page_size: usize,
    bounds: Bounds<Pixels>,
    open: bool,
    active_value: Option<I::Value>,
    _subscriptions: Vec<Subscription>,
}

#[derive(IntoElement)]
pub struct Select<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    id: ElementId,
    state: Entity<SelectState<I>>,
    options: SelectOptions,
}

#[derive(IntoElement)]
pub struct ActionMenu<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    id: ElementId,
    state: Entity<ActionMenuState<I>>,
    options: ActionMenuOptions,
}

impl<I> SelectState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    pub fn new(
        items: impl Into<Vec<I>>,
        selected_value: Option<I::Value>,
        page_size: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let search_input = cx
            .new(|cx| InputState::new(window, cx).placeholder(Self::default_search_placeholder()));
        let page_input = cx.new(|cx| InputState::new(window, cx).default_value("1"));
        let focus_handle = cx.focus_handle();

        let mut this = Self {
            focus_handle,
            options: SelectOptions {
                menu_width: Length::Auto,
                appearance: true,
                ..Default::default()
            },
            items: items.into(),
            search_input: search_input.clone(),
            page_input: page_input.clone(),
            page: 0,
            page_size: page_size.max(1),
            bounds: Bounds::default(),
            open: false,
            selected_value: selected_value.clone(),
            active_value: selected_value,
            _subscriptions: Vec::new(),
        };

        let search_input = this.search_input.clone();
        this._subscriptions.push(cx.subscribe_in(
            &search_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.reset_page(window, cx);
                }
            },
        ));
        let page_input = this.page_input.clone();
        this._subscriptions.push(cx.subscribe_in(
            &page_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.jump_page_from_input(window, cx);
                }
            },
        ));

        this.reconcile_selection(window, cx);
        this.sync_page_input(window, cx);
        this
    }

    pub fn set_items(
        &mut self,
        items: impl Into<Vec<I>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.items = items.into();
        self.reconcile_selection(window, cx);
        self.clamp_page(window, cx);
        cx.notify();
    }

    pub fn set_selected_value(
        &mut self,
        selected_value: &I::Value,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.selected_value = self
            .items
            .iter()
            .find(|item| item.value() == selected_value)
            .map(|item| item.value().clone());
        self.active_value = self.selected_value.clone();
        self.reveal_selected_page(window, cx);
        cx.notify();
    }

    pub fn set_selected_index(
        &mut self,
        selected_index: Option<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.selected_value = selected_index
            .and_then(|ix| self.items.get(ix))
            .map(|item| item.value().clone());
        self.active_value = self.selected_value.clone();
        self.reveal_selected_page(window, cx);
        cx.notify();
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    fn default_search_placeholder() -> SharedString {
        "搜索关键字".into()
    }

    fn effective_search_placeholder(options: &SelectOptions) -> SharedString {
        options
            .search_placeholder
            .clone()
            .unwrap_or_else(Self::default_search_placeholder)
    }

    fn effective_empty_message(&self) -> SharedString {
        self.options
            .empty_message
            .clone()
            .unwrap_or_else(|| "没有匹配项。".into())
    }

    fn apply_options(
        &mut self,
        options: SelectOptions,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let previous_search_placeholder = Self::effective_search_placeholder(&self.options);
        let next_search_placeholder = Self::effective_search_placeholder(&options);
        let should_close = !self.options.disabled && options.disabled && self.open;
        self.options = options;

        if previous_search_placeholder != next_search_placeholder {
            self.search_input.update(cx, |input, cx| {
                input.set_placeholder(next_search_placeholder.clone(), window, cx);
            });
        }

        if should_close {
            self.close_menu(window, cx);
        }
    }

    fn selected_item(&self) -> Option<&I> {
        self.selected_value
            .as_ref()
            .and_then(|selected| self.items.iter().find(|item| item.value() == selected))
    }

    fn selected_filtered_position_in(&self, filtered_items: &[I]) -> Option<usize> {
        let selected = self.selected_value.as_ref()?;
        filtered_items
            .iter()
            .position(|item| item.value() == selected)
    }

    fn active_filtered_position_in(&self, filtered_items: &[I]) -> Option<usize> {
        let active = self.active_value.as_ref()?;
        filtered_items
            .iter()
            .position(|item| item.value() == active)
    }

    fn first_visible_value(&self, filtered_items: &[I], page: usize) -> Option<I::Value> {
        filtered_items
            .iter()
            .skip(page.saturating_mul(self.page_size))
            .take(self.page_size)
            .next()
            .map(|item| item.value().clone())
    }

    fn sync_active_for_current_view(
        &mut self,
        filtered_items: &[I],
        page: usize,
        preserve_active: bool,
    ) {
        if filtered_items.is_empty() {
            self.active_value = None;
            return;
        }

        let start = page.saturating_mul(self.page_size);
        let end = (start + self.page_size).min(filtered_items.len());
        let visible_items = &filtered_items[start..end];
        if visible_items.is_empty() {
            self.active_value = None;
            return;
        }

        if preserve_active
            && self
                .active_value
                .as_ref()
                .is_some_and(|active| visible_items.iter().any(|item| item.value() == active))
        {
            return;
        }

        if self
            .selected_value
            .as_ref()
            .is_some_and(|selected| visible_items.iter().any(|item| item.value() == selected))
        {
            self.active_value = self.selected_value.clone();
            return;
        }

        self.active_value = visible_items.first().map(|item| item.value().clone());
    }

    fn trigger_title(&self) -> AnyElement {
        self.selected_item()
            .and_then(|item| item.display_title())
            .or_else(|| {
                self.selected_item()
                    .map(|item| item.title().into_any_element())
            })
            .unwrap_or_else(|| {
                self.options
                    .placeholder
                    .clone()
                    .unwrap_or_else(|| "请选择".into())
                    .into_any_element()
            })
    }

    fn normalized_query(&self, cx: &App) -> String {
        self.search_input.read(cx).value().trim().to_lowercase()
    }

    fn filtered_items(&self, query: &str) -> Vec<I> {
        self.items
            .iter()
            .filter(|item| item.matches(query))
            .cloned()
            .collect()
    }

    fn page_count(filtered_count: usize, page_size: usize) -> usize {
        filtered_count.max(1).div_ceil(page_size.max(1))
    }

    fn current_focus_handle(&self, cx: &App) -> FocusHandle {
        if self.open {
            self.search_input.read(cx).focus_handle(cx)
        } else {
            self.focus_handle.clone()
        }
    }

    fn set_page(&mut self, page: usize, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        self.page = page.min(page_count.saturating_sub(1));
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn reset_page(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        self.page = 0;
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn clamp_page(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        self.page = self.page.min(page_count.saturating_sub(1));
        self.sync_active_for_current_view(&filtered_items, self.page, true);
        self.sync_page_input(window, cx);
    }

    fn reveal_selected_page(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let selected_position = self
            .selected_filtered_position_in(&filtered_items)
            .unwrap_or(0);
        self.page = selected_position / self.page_size.max(1);
        self.sync_active_for_current_view(&filtered_items, self.page, true);
        self.sync_page_input(window, cx);
    }

    fn jump_page_from_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let input = self.page_input.read(cx).value().trim().to_string();
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        let Some(page) = input
            .parse::<usize>()
            .ok()
            .and_then(|page| page.checked_sub(1))
        else {
            self.sync_page_input(window, cx);
            return;
        };

        self.page = page.min(page_count.saturating_sub(1));
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn sync_page_input(&self, window: &mut Window, cx: &mut Context<Self>) {
        let value: SharedString = (self.page + 1).to_string().into();
        self.page_input.update(cx, |input, cx| {
            if input.value().as_ref() != value.as_ref() {
                input.set_value(value.clone(), window, cx);
            }
        });
    }

    fn clear_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.search_input.read(cx).value().is_empty() {
            return;
        }

        self.search_input.update(cx, |input, cx| {
            input.set_value("", window, cx);
        });
    }

    fn reconcile_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .selected_value
            .as_ref()
            .is_some_and(|selected| self.items.iter().any(|item| item.value() == selected))
        {
            self.reveal_selected_page(window, cx);
        } else {
            self.selected_value = None;
            self.active_value = None;
            self.page = 0;
            self.sync_page_input(window, cx);
        }
    }

    fn open_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        self.open = true;
        if let Some(selected_position) = self.selected_filtered_position_in(&filtered_items) {
            self.page = selected_position / self.page_size.max(1);
            self.active_value = self.selected_value.clone();
        } else {
            self.page = 0;
            self.active_value = self.first_visible_value(&filtered_items, self.page);
        }
        self.sync_active_for_current_view(&filtered_items, self.page, true);
        self.sync_page_input(window, cx);
        self.search_input.read(cx).focus_handle(cx).focus(window);
        cx.notify();
    }

    fn close_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open = false;
        self.active_value = self.selected_value.clone();
        self.clear_search(window, cx);
        self.focus_handle.focus(window);
        cx.notify();
    }

    fn toggle_menu(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        cx.stop_propagation();

        if self.options.disabled {
            return;
        }

        if self.open {
            self.close_menu(window, cx);
        } else {
            self.open_menu(window, cx);
        }
    }

    fn choose_value(&mut self, value: I::Value, window: &mut Window, cx: &mut Context<Self>) {
        self.selected_value = Some(value.clone());
        self.active_value = Some(value.clone());
        cx.emit(SelectEvent::<I>::Confirm(Some(value)));
        self.close_menu(window, cx);
    }

    fn move_active_by(&mut self, step: isize, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        if filtered_items.is_empty() {
            self.active_value = None;
            cx.notify();
            return;
        }

        let current_position = self
            .active_filtered_position_in(&filtered_items)
            .or_else(|| self.selected_filtered_position_in(&filtered_items))
            .unwrap_or(0);
        let next_position = if step.is_negative() {
            current_position.saturating_sub(step.unsigned_abs())
        } else {
            (current_position + step as usize).min(filtered_items.len().saturating_sub(1))
        };
        self.page = next_position / self.page_size.max(1);
        self.active_value = filtered_items
            .get(next_position)
            .map(|item| item.value().clone());
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn on_action_up(
        &mut self,
        _: &WorkbenchPagedSelectUp,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }
        self.move_active_by(-1, window, cx);
    }

    fn on_action_down(
        &mut self,
        _: &WorkbenchPagedSelectDown,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }
        self.move_active_by(1, window, cx);
    }

    fn on_action_confirm(
        &mut self,
        _: &WorkbenchPagedSelectConfirm,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }

        if let Some(active_value) = self.active_value.clone() {
            self.choose_value(active_value, window, cx);
        }
    }

    fn on_action_cancel(
        &mut self,
        _: &WorkbenchPagedSelectCancel,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.open {
            cx.stop_propagation();
            self.close_menu(window, cx);
        } else {
            cx.propagate();
        }
    }
}

impl<I> ActionMenuState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    pub fn new(
        items: impl Into<Vec<I>>,
        page_size: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let search_input = cx
            .new(|cx| InputState::new(window, cx).placeholder(Self::default_search_placeholder()));
        let page_input = cx.new(|cx| InputState::new(window, cx).default_value("1"));
        let focus_handle = cx.focus_handle();

        let mut this = Self {
            focus_handle,
            options: ActionMenuOptions {
                menu_width: Length::Auto,
                ..Default::default()
            },
            items: items.into(),
            search_input: search_input.clone(),
            page_input: page_input.clone(),
            page: 0,
            page_size: page_size.max(1),
            bounds: Bounds::default(),
            open: false,
            active_value: None,
            _subscriptions: Vec::new(),
        };

        let search_input = this.search_input.clone();
        this._subscriptions.push(cx.subscribe_in(
            &search_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    this.reset_page(window, cx);
                }
            },
        ));
        let page_input = this.page_input.clone();
        this._subscriptions.push(cx.subscribe_in(
            &page_input,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.jump_page_from_input(window, cx);
                }
            },
        ));

        this.sync_page_input(window, cx);
        this
    }

    fn default_search_placeholder() -> SharedString {
        "搜索关键字".into()
    }

    fn effective_label(&self) -> SharedString {
        self.options
            .label
            .clone()
            .unwrap_or_else(|| "打开菜单".into())
    }

    fn effective_search_placeholder(options: &ActionMenuOptions) -> SharedString {
        options
            .search_placeholder
            .clone()
            .unwrap_or_else(Self::default_search_placeholder)
    }

    fn effective_empty_message(&self) -> SharedString {
        self.options
            .empty_message
            .clone()
            .unwrap_or_else(|| "没有匹配项。".into())
    }

    fn apply_options(
        &mut self,
        options: ActionMenuOptions,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let previous_search_placeholder = Self::effective_search_placeholder(&self.options);
        let next_search_placeholder = Self::effective_search_placeholder(&options);
        let should_close = !self.options.disabled && options.disabled && self.open;
        self.options = options;

        if previous_search_placeholder != next_search_placeholder {
            self.search_input.update(cx, |input, cx| {
                input.set_placeholder(next_search_placeholder.clone(), window, cx);
            });
        }

        if should_close {
            self.close_menu(window, cx);
        }
    }

    fn active_filtered_position_in(&self, filtered_items: &[I]) -> Option<usize> {
        let active = self.active_value.as_ref()?;
        filtered_items
            .iter()
            .position(|item| item.value() == active)
    }

    fn first_visible_value(&self, filtered_items: &[I], page: usize) -> Option<I::Value> {
        filtered_items
            .iter()
            .skip(page.saturating_mul(self.page_size))
            .take(self.page_size)
            .next()
            .map(|item| item.value().clone())
    }

    fn sync_active_for_current_view(
        &mut self,
        filtered_items: &[I],
        page: usize,
        preserve_active: bool,
    ) {
        if filtered_items.is_empty() {
            self.active_value = None;
            return;
        }

        let start = page.saturating_mul(self.page_size);
        let end = (start + self.page_size).min(filtered_items.len());
        let visible_items = &filtered_items[start..end];
        if visible_items.is_empty() {
            self.active_value = None;
            return;
        }

        if preserve_active
            && self
                .active_value
                .as_ref()
                .is_some_and(|active| visible_items.iter().any(|item| item.value() == active))
        {
            return;
        }

        self.active_value = visible_items.first().map(|item| item.value().clone());
    }

    fn normalized_query(&self, cx: &App) -> String {
        self.search_input.read(cx).value().trim().to_lowercase()
    }

    fn filtered_items(&self, query: &str) -> Vec<I> {
        self.items
            .iter()
            .filter(|item| item.matches(query))
            .cloned()
            .collect()
    }

    fn page_count(filtered_count: usize, page_size: usize) -> usize {
        filtered_count.max(1).div_ceil(page_size.max(1))
    }

    fn current_focus_handle(&self, cx: &App) -> FocusHandle {
        if self.open {
            self.search_input.read(cx).focus_handle(cx)
        } else {
            self.focus_handle.clone()
        }
    }

    fn set_page(&mut self, page: usize, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        self.page = page.min(page_count.saturating_sub(1));
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn reset_page(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        self.page = 0;
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn jump_page_from_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let input = self.page_input.read(cx).value().trim().to_string();
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        let Some(page) = input
            .parse::<usize>()
            .ok()
            .and_then(|page| page.checked_sub(1))
        else {
            self.sync_page_input(window, cx);
            return;
        };

        self.page = page.min(page_count.saturating_sub(1));
        self.sync_active_for_current_view(&filtered_items, self.page, false);
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn sync_page_input(&self, window: &mut Window, cx: &mut Context<Self>) {
        let value: SharedString = (self.page + 1).to_string().into();
        self.page_input.update(cx, |input, cx| {
            if input.value().as_ref() != value.as_ref() {
                input.set_value(value.clone(), window, cx);
            }
        });
    }

    fn clear_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.search_input.read(cx).value().is_empty() {
            return;
        }

        self.search_input.update(cx, |input, cx| {
            input.set_value("", window, cx);
        });
    }

    fn open_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        self.open = true;
        self.page = 0;
        self.active_value = self.first_visible_value(&filtered_items, self.page);
        self.sync_active_for_current_view(&filtered_items, self.page, true);
        self.sync_page_input(window, cx);
        self.search_input.read(cx).focus_handle(cx).focus(window);
        cx.notify();
    }

    fn close_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open = false;
        self.active_value = None;
        self.clear_search(window, cx);
        self.focus_handle.focus(window);
        cx.notify();
    }

    fn toggle_menu(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        cx.stop_propagation();

        if self.options.disabled {
            return;
        }

        if self.open {
            self.close_menu(window, cx);
        } else {
            self.open_menu(window, cx);
        }
    }

    fn choose_value(&mut self, value: I::Value, window: &mut Window, cx: &mut Context<Self>) {
        self.active_value = Some(value.clone());
        cx.emit(ActionMenuEvent::<I>::Confirm(value));
        self.close_menu(window, cx);
    }

    fn move_active_by(&mut self, step: isize, window: &mut Window, cx: &mut Context<Self>) {
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        if filtered_items.is_empty() {
            self.active_value = None;
            cx.notify();
            return;
        }

        let current_position = self
            .active_filtered_position_in(&filtered_items)
            .unwrap_or(0);
        let next_position = if step.is_negative() {
            current_position.saturating_sub(step.unsigned_abs())
        } else {
            (current_position + step as usize).min(filtered_items.len().saturating_sub(1))
        };
        self.page = next_position / self.page_size.max(1);
        self.active_value = filtered_items
            .get(next_position)
            .map(|item| item.value().clone());
        self.sync_page_input(window, cx);
        cx.notify();
    }

    fn on_action_up(
        &mut self,
        _: &WorkbenchPagedSelectUp,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }
        self.move_active_by(-1, window, cx);
    }

    fn on_action_down(
        &mut self,
        _: &WorkbenchPagedSelectDown,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }
        self.move_active_by(1, window, cx);
    }

    fn on_action_confirm(
        &mut self,
        _: &WorkbenchPagedSelectConfirm,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.page_input.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        cx.stop_propagation();
        if !self.open {
            self.open_menu(window, cx);
            return;
        }

        if let Some(active_value) = self.active_value.clone() {
            self.choose_value(active_value, window, cx);
        }
    }

    fn on_action_cancel(
        &mut self,
        _: &WorkbenchPagedSelectCancel,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.open {
            cx.stop_propagation();
            self.close_menu(window, cx);
        } else {
            cx.propagate();
        }
    }
}

impl<I> Render for SelectState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_focused = self.focus_handle.is_focused(window);
        let outline_visible = self.open || (is_focused && !self.options.disabled);
        let popup_radius = cx.theme().radius.min(px(8.0));
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        let page = self.page.min(page_count.saturating_sub(1));
        let last_page = page_count.saturating_sub(1);
        let start = page.saturating_mul(self.page_size);
        let selected_exists = self.selected_value.is_some();
        let title_color = if self.options.disabled {
            cx.theme().muted_foreground
        } else if selected_exists {
            cx.theme().foreground
        } else {
            cx.theme().accent_foreground
        };
        let tracked_focus = self.current_focus_handle(cx);
        let visible_items = filtered_items
            .into_iter()
            .skip(start)
            .take(self.page_size)
            .enumerate()
            .map(|(index, item)| {
                let item_value = item.value().clone();
                let active = self
                    .active_value
                    .as_ref()
                    .is_some_and(|active_value| item.value() == active_value);
                let selected = self
                    .selected_value
                    .as_ref()
                    .is_some_and(|selected_value| item.value() == selected_value);
                div()
                    .id(("paged-select-item", start + index))
                    .w_full()
                    .cursor_pointer()
                    .rounded(cx.theme().radius)
                    .border_1()
                    .border_color(cx.theme().transparent)
                    .px_2()
                    .py_1p5()
                    .text_color(if active {
                        cx.theme().accent_foreground
                    } else {
                        cx.theme().foreground
                    })
                    .when(active, |this| this.bg(cx.theme().accent))
                    .when(selected && !active, |this| {
                        this.border_color(cx.theme().accent.alpha(0.45))
                            .bg(cx.theme().secondary.opacity(0.32))
                    })
                    .when(!active, |this| {
                        this.hover(|this| this.bg(cx.theme().accent.alpha(0.12)))
                    })
                    .child(div().w_full().min_w_0().child(item.render(window, cx)))
                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                        this.choose_value(item_value.clone(), window, cx);
                    }))
                    .into_any_element()
            })
            .collect::<Vec<_>>();

        div()
            .size_full()
            .relative()
            .key_context(CONTEXT)
            .when(!self.options.disabled, |this| {
                this.track_focus(&tracked_focus.tab_stop(true))
            })
            .on_action(cx.listener(Self::on_action_up))
            .on_action(cx.listener(Self::on_action_down))
            .on_action(cx.listener(Self::on_action_confirm))
            .on_action(cx.listener(Self::on_action_cancel))
            .child(
                div()
                    .id("input")
                    .relative()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_1()
                    .border_color(cx.theme().transparent)
                    .when(self.options.appearance, |this| {
                        this.bg(cx.theme().background)
                            .border_color(cx.theme().input)
                            .rounded(cx.theme().radius)
                            .when(cx.theme().shadow, |this| this.shadow_xs())
                    })
                    .overflow_hidden()
                    .input_size(self.options.size)
                    .input_text_size(self.options.size)
                    .refine_style(&self.options.style)
                    .when(outline_visible, |this| this.focused_border(cx))
                    .when(!self.options.disabled, |this| {
                        this.on_click(cx.listener(Self::toggle_menu))
                    })
                    .child(
                        div()
                            .w_full()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_1()
                            .px_3()
                            .py_2()
                            .child(
                                div()
                                    .w_full()
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .truncate()
                                    .text_color(title_color)
                                    .child(self.trigger_title()),
                            )
                            .child(
                                self.options
                                    .icon
                                    .clone()
                                    .unwrap_or_else(|| Icon::new(IconName::ChevronDown))
                                    .xsmall()
                                    .text_color(if self.options.disabled {
                                        cx.theme().muted_foreground.opacity(0.5)
                                    } else {
                                        cx.theme().muted_foreground
                                    }),
                            ),
                    )
                    .child(
                        canvas(
                            {
                                let state = cx.entity();
                                move |bounds, _, cx| {
                                    state.update(cx, |this, _| this.bounds = bounds)
                                }
                            },
                            |_, _, _, _| {},
                        )
                        .absolute()
                        .size_full(),
                    ),
            )
            .when(self.open, |this| {
                this.child(
                    deferred(
                        anchored().snap_to_window_with_margin(px(8.0)).child(
                            div()
                                .occlude()
                                .map(|this| match self.options.menu_width {
                                    Length::Auto => this.w(self.bounds.size.width + px(2.0)),
                                    Length::Definite(width) => this.w(width),
                                })
                                .child(
                                    div()
                                        .occlude()
                                        .mt_1p5()
                                        .bg(cx.theme().background)
                                        .border_1()
                                        .border_color(cx.theme().border)
                                        .rounded(popup_radius)
                                        .shadow_md()
                                        .p_2()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .child(Input::new(&self.search_input))
                                        .child(
                                            div()
                                                .w_full()
                                                .min_w_0()
                                                .flex()
                                                .flex_col()
                                                .items_start()
                                                .gap_1p5()
                                                .child(
                                                    div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .text_xs()
                                                        .text_color(cx.theme().muted_foreground)
                                                        .child(format!(
                                                            "第 {} / {} 页 · {} 项",
                                                            page + 1,
                                                            page_count,
                                                            filtered_count
                                                        )),
                                                )
                                                .child(
                                                    div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .flex()
                                                        .items_center()
                                                        .justify_end()
                                                        .flex_wrap()
                                                        .gap_1()
                                                        .child(pager_button(
                                                            0,
                                                            "<<",
                                                            page == 0,
                                                            cx.listener(
                                                                |this,
                                                                 _: &ClickEvent,
                                                                 window,
                                                                 cx| {
                                                                    this.set_page(0, window, cx);
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(pager_button(
                                                            1,
                                                            "<",
                                                            page == 0,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        page.saturating_sub(1),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(
                                                            div().w(px(58.0)).min_w(px(58.0)).child(
                                                                Input::new(&self.page_input)
                                                                    .appearance(false)
                                                                    .px(px(0.0))
                                                                    .gap_0(),
                                                            ),
                                                        )
                                                        .child(pager_button(
                                                            2,
                                                            ">",
                                                            page >= last_page,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        (page + 1).min(last_page),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(pager_button(
                                                            3,
                                                            ">>",
                                                            page >= last_page,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        last_page,
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        )),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .max_h(rems(18.0))
                                                .overflow_y_scrollbar()
                                                .child(
                                                    div().flex().flex_col().gap_1().children(
                                                        if visible_items.is_empty() {
                                                            vec![empty_state(
                                                                self.effective_empty_message(),
                                                                cx,
                                                            )]
                                                        } else {
                                                            visible_items
                                                        },
                                                    ),
                                                ),
                                        ),
                                )
                                .on_mouse_down_out(cx.listener(|this, _, window, cx| {
                                    this.close_menu(window, cx);
                                })),
                        ),
                    )
                    .with_priority(1),
                )
            })
    }
}

impl<I> Render for ActionMenuState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_focused = self.focus_handle.is_focused(window);
        let outline_visible = self.open || (is_focused && !self.options.disabled);
        let popup_radius = cx.theme().radius.min(px(8.0));
        let label_color = if self.options.disabled {
            cx.theme().muted_foreground
        } else {
            cx.theme().foreground
        };
        let icon_color = if self.options.disabled {
            cx.theme().muted_foreground.opacity(0.5)
        } else {
            cx.theme().muted_foreground
        };
        let query = self.normalized_query(cx);
        let filtered_items = self.filtered_items(&query);
        let filtered_count = filtered_items.len();
        let page_count = Self::page_count(filtered_count, self.page_size);
        let page = self.page.min(page_count.saturating_sub(1));
        let last_page = page_count.saturating_sub(1);
        let start = page.saturating_mul(self.page_size);
        let tracked_focus = self.current_focus_handle(cx);
        let visible_items = filtered_items
            .into_iter()
            .skip(start)
            .take(self.page_size)
            .enumerate()
            .map(|(index, item)| {
                let item_value = item.value().clone();
                let active = self
                    .active_value
                    .as_ref()
                    .is_some_and(|active_value| item.value() == active_value);
                div()
                    .id(("action-menu-item", start + index))
                    .w_full()
                    .cursor_pointer()
                    .rounded(cx.theme().radius)
                    .border_1()
                    .border_color(cx.theme().transparent)
                    .px_2()
                    .py_1p5()
                    .text_color(if active {
                        cx.theme().accent_foreground
                    } else {
                        cx.theme().foreground
                    })
                    .when(active, |this| this.bg(cx.theme().accent))
                    .when(!active, |this| {
                        this.hover(|this| this.bg(cx.theme().accent.alpha(0.12)))
                    })
                    .child(div().w_full().min_w_0().child(item.render(window, cx)))
                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                        this.choose_value(item_value.clone(), window, cx);
                    }))
                    .into_any_element()
            })
            .collect::<Vec<_>>();

        div()
            .relative()
            .key_context(CONTEXT)
            .when(!self.options.disabled, |this| {
                this.track_focus(&tracked_focus.tab_stop(true))
            })
            .on_action(cx.listener(Self::on_action_up))
            .on_action(cx.listener(Self::on_action_down))
            .on_action(cx.listener(Self::on_action_confirm))
            .on_action(cx.listener(Self::on_action_cancel))
            .child(
                div()
                    .id("action-menu-trigger")
                    .relative()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .border_1()
                    .border_color(if self.open {
                        cx.theme().accent.alpha(0.48)
                    } else {
                        cx.theme().border
                    })
                    .bg(if self.options.disabled {
                        cx.theme().secondary.opacity(0.5)
                    } else if self.open {
                        cx.theme().secondary.opacity(0.92)
                    } else {
                        cx.theme().secondary
                    })
                    .rounded(cx.theme().radius)
                    .when(cx.theme().shadow, |this| this.shadow_xs())
                    .input_size(self.options.size)
                    .input_text_size(self.options.size)
                    .refine_style(&self.options.style)
                    .when(outline_visible, |this| this.focused_border(cx))
                    .when(!self.options.disabled, |this| {
                        this.cursor_pointer()
                            .hover(|this| this.bg(cx.theme().secondary.opacity(0.82)))
                            .on_click(cx.listener(Self::toggle_menu))
                    })
                    .child(if self.options.center_label {
                        div()
                            .w_full()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .child(div().w(px(12.0)).flex_shrink_0())
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .justify_center()
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .truncate()
                                    .text_color(label_color)
                                    .child(self.effective_label()),
                            )
                            .child(
                                div()
                                    .w(px(12.0))
                                    .flex_shrink_0()
                                    .flex()
                                    .justify_center()
                                    .child(
                                        self.options
                                            .icon
                                            .clone()
                                            .unwrap_or_else(|| Icon::new(IconName::ChevronDown))
                                            .xsmall()
                                            .text_color(icon_color),
                                    ),
                            )
                            .into_any_element()
                    } else {
                        div()
                            .w_full()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .w_full()
                                    .min_w_0()
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .truncate()
                                    .text_color(label_color)
                                    .child(self.effective_label()),
                            )
                            .child(
                                self.options
                                    .icon
                                    .clone()
                                    .unwrap_or_else(|| Icon::new(IconName::ChevronDown))
                                    .xsmall()
                                    .text_color(icon_color),
                            )
                            .into_any_element()
                    })
                    .child(
                        canvas(
                            {
                                let state = cx.entity();
                                move |bounds, _, cx| {
                                    state.update(cx, |this, _| this.bounds = bounds)
                                }
                            },
                            |_, _, _, _| {},
                        )
                        .absolute()
                        .size_full(),
                    ),
            )
            .when(self.open, |this| {
                this.child(
                    deferred(
                        anchored().snap_to_window_with_margin(px(8.0)).child(
                            div()
                                .occlude()
                                .map(|this| match self.options.menu_width {
                                    Length::Auto => this.w(self.bounds.size.width + px(2.0)),
                                    Length::Definite(width) => this.w(width),
                                })
                                .child(
                                    div()
                                        .occlude()
                                        .mt_1p5()
                                        .bg(cx.theme().background)
                                        .border_1()
                                        .border_color(cx.theme().border)
                                        .rounded(popup_radius)
                                        .shadow_md()
                                        .p_2()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .child(Input::new(&self.search_input))
                                        .child(
                                            div()
                                                .w_full()
                                                .min_w_0()
                                                .flex()
                                                .flex_col()
                                                .items_start()
                                                .gap_1p5()
                                                .child(
                                                    div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .text_xs()
                                                        .text_color(cx.theme().muted_foreground)
                                                        .child(format!(
                                                            "第 {} / {} 页 · {} 项",
                                                            page + 1,
                                                            page_count,
                                                            filtered_count
                                                        )),
                                                )
                                                .child(
                                                    div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .flex()
                                                        .items_center()
                                                        .justify_end()
                                                        .flex_wrap()
                                                        .gap_1()
                                                        .child(pager_button(
                                                            0,
                                                            "<<",
                                                            page == 0,
                                                            cx.listener(
                                                                |this,
                                                                 _: &ClickEvent,
                                                                 window,
                                                                 cx| {
                                                                    this.set_page(0, window, cx);
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(pager_button(
                                                            1,
                                                            "<",
                                                            page == 0,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        page.saturating_sub(1),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(
                                                            div().w(px(58.0)).min_w(px(58.0)).child(
                                                                Input::new(&self.page_input)
                                                                    .appearance(false)
                                                                    .px(px(0.0))
                                                                    .gap_0(),
                                                            ),
                                                        )
                                                        .child(pager_button(
                                                            2,
                                                            ">",
                                                            page >= last_page,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        (page + 1).min(last_page),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        ))
                                                        .child(pager_button(
                                                            3,
                                                            ">>",
                                                            page >= last_page,
                                                            cx.listener(
                                                                move |this,
                                                                      _: &ClickEvent,
                                                                      window,
                                                                      cx| {
                                                                    this.set_page(
                                                                        last_page,
                                                                        window,
                                                                        cx,
                                                                    );
                                                                },
                                                            ),
                                                            cx,
                                                        )),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .max_h(rems(18.0))
                                                .overflow_y_scrollbar()
                                                .child(
                                                    div().flex().flex_col().gap_1().children(
                                                        if visible_items.is_empty() {
                                                            vec![empty_state(
                                                                self.effective_empty_message(),
                                                                cx,
                                                            )]
                                                        } else {
                                                            visible_items
                                                        },
                                                    ),
                                                ),
                                        ),
                                )
                                .on_mouse_down_out(cx.listener(|this, _, window, cx| {
                                    this.close_menu(window, cx);
                                })),
                        ),
                    )
                    .with_priority(1),
                )
            })
    }
}

impl<I> Select<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    pub fn new(state: &Entity<SelectState<I>>) -> Self {
        Self {
            id: ("select", state.entity_id()).into(),
            state: state.clone(),
            options: SelectOptions {
                menu_width: Length::Auto,
                appearance: true,
                ..Default::default()
            },
        }
    }

    pub fn menu_width(mut self, width: impl Into<Length>) -> Self {
        self.options.menu_width = width.into();
        self
    }

    pub fn placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.options.placeholder = Some(placeholder.into());
        self
    }

    pub fn search_placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.options.search_placeholder = Some(placeholder.into());
        self
    }

    pub fn empty_message(mut self, message: impl Into<SharedString>) -> Self {
        self.options.empty_message = Some(message.into());
        self
    }

    pub fn icon(mut self, icon: impl Into<Icon>) -> Self {
        self.options.icon = Some(icon.into());
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.options.disabled = disabled;
        self
    }
}

impl<I> ActionMenu<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    pub fn new(state: &Entity<ActionMenuState<I>>) -> Self {
        Self {
            id: ("action-menu", state.entity_id()).into(),
            state: state.clone(),
            options: ActionMenuOptions {
                menu_width: Length::Auto,
                ..Default::default()
            },
        }
    }

    pub fn label(mut self, label: impl Into<SharedString>) -> Self {
        self.options.label = Some(label.into());
        self
    }

    pub fn center_label(mut self) -> Self {
        self.options.center_label = true;
        self
    }

    pub fn search_placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.options.search_placeholder = Some(placeholder.into());
        self
    }

    pub fn empty_message(mut self, message: impl Into<SharedString>) -> Self {
        self.options.empty_message = Some(message.into());
        self
    }

    pub fn menu_width(mut self, width: impl Into<Length>) -> Self {
        self.options.menu_width = width.into();
        self
    }

    pub fn icon(mut self, icon: impl Into<Icon>) -> Self {
        self.options.icon = Some(icon.into());
        self
    }
}

impl<I> Sizable for Select<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn with_size(mut self, size: impl Into<Size>) -> Self {
        self.options.size = size.into();
        self
    }
}

impl<I> Sizable for ActionMenu<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn with_size(mut self, size: impl Into<Size>) -> Self {
        self.options.size = size.into();
        self
    }
}

impl<I> EventEmitter<SelectEvent<I>> for SelectState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
}

impl<I> EventEmitter<ActionMenuEvent<I>> for ActionMenuState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
}

impl<I> Focusable for SelectState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl<I> Focusable for ActionMenuState<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl<I> Styled for Select<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.options.style
    }
}

impl<I> Styled for ActionMenu<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.options.style
    }
}

impl<I> RenderOnce for Select<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        self.state.update(cx, |this, cx| {
            this.apply_options(self.options, window, cx);
        });

        div().id(self.id.clone()).size_full().child(self.state)
    }
}

impl<I> RenderOnce for ActionMenu<I>
where
    I: SelectItem + 'static,
    I::Value: Clone + PartialEq + 'static,
{
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        self.state.update(cx, |this, cx| {
            this.apply_options(self.options, window, cx);
        });

        div().id(self.id.clone()).child(self.state)
    }
}

fn pager_button<S>(
    id: usize,
    label: &'static str,
    disabled: bool,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    cx: &mut Context<S>,
) -> impl IntoElement
where
    S: 'static,
{
    div()
        .id(("paged-select-pager", id))
        .min_w(px(28.0))
        .h(px(28.0))
        .px_2()
        .rounded(cx.theme().radius)
        .border_1()
        .border_color(cx.theme().border)
        .bg(if disabled {
            cx.theme().secondary.opacity(0.5)
        } else {
            cx.theme().secondary
        })
        .text_xs()
        .text_color(if disabled {
            cx.theme().muted_foreground
        } else {
            cx.theme().foreground
        })
        .flex()
        .items_center()
        .justify_center()
        .cursor_pointer()
        .when(!disabled, |this| {
            this.hover(|this| this.bg(cx.theme().secondary.opacity(0.82)))
                .on_click(on_click)
        })
        .child(label)
}

fn empty_state<S>(message: SharedString, cx: &mut Context<S>) -> AnyElement
where
    S: 'static,
{
    div()
        .w_full()
        .rounded(cx.theme().radius)
        .bg(cx.theme().secondary.opacity(0.35))
        .px_3()
        .py_4()
        .text_sm()
        .text_color(cx.theme().muted_foreground)
        .child(message)
        .into_any_element()
}
