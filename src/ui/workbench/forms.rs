use gpui::{
    AnyElement, AppContext, Context, IntoElement, ParentElement, SharedString, Styled, Window, div,
};
use gpui_component::{
    input::InputState,
    select::{SearchableVec, SelectItem},
};

use crate::domain::{
    geometry::WorldPoint,
    marker::{MarkerStyle, normalize_hex_color},
    route::RouteId,
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
            name: cx.new(|cx| InputState::new(window, cx).placeholder("分组名称")),
            description: cx.new(|cx| InputState::new(window, cx).placeholder("分组描述")),
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
            label: cx.new(|cx| InputState::new(window, cx).placeholder("标记点名称")),
            note: cx.new(|cx| InputState::new(window, cx).placeholder("备注")),
            x: cx.new(|cx| InputState::new(window, cx).placeholder("X")),
            y: cx.new(|cx| InputState::new(window, cx).placeholder("Y")),
            color_hex: cx.new(|cx| InputState::new(window, cx).default_value("#4ECDC4")),
            size_px: cx.new(|cx| InputState::new(window, cx).default_value("24")),
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
        self.searchable_text.clone()
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
}

pub(super) type MarkerGroupPickerDelegate = SearchableVec<MarkerGroupPickerItem>;

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
            return Err("分组名称不能为空。".to_owned());
        }

        let description = read_input_value(&workbench.group_form.description, cx);
        let size_px = read_input_value(&workbench.group_form.size_px, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "分组默认尺寸必须是数字。".to_owned())?;

        Ok(Self {
            name,
            description,
            style: MarkerStyle {
                icon: workbench.group_icon,
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
            return Err("请先选择一个分组，再保存标记点。".to_owned());
        }

        let x = read_input_value(&workbench.marker_form.x, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "标记点 X 坐标必须是数字。".to_owned())?;
        let y = read_input_value(&workbench.marker_form.y, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "标记点 Y 坐标必须是数字。".to_owned())?;
        let size_px = read_input_value(&workbench.marker_form.size_px, cx)
            .trim()
            .parse::<f32>()
            .map_err(|_| "标记点尺寸必须是数字。".to_owned())?;

        Ok(Self {
            label: read_input_value(&workbench.marker_form.label, cx),
            note: read_input_value(&workbench.marker_form.note, cx),
            world: WorldPoint::new(x, y),
            style: MarkerStyle {
                icon: workbench.marker_icon,
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
