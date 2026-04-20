use gpui::{
    AnyWindowHandle, App, ClickEvent, Context, CursorStyle, InteractiveElement as _, IntoElement,
    MouseButton, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _,
    Window, div, hsla, px,
};

use super::TrackerWorkbench;

pub(super) fn close_picker<T>(
    workbench: &gpui::WeakEntity<TrackerWorkbench>,
    window: &mut Window,
    cx: &mut Context<T>,
    on_close: impl FnOnce(&mut TrackerWorkbench, &mut Context<TrackerWorkbench>),
) {
    if let Some(workbench) = workbench.upgrade() {
        let _ = workbench.update(cx, on_close);
    }
    window.remove_window();
}

pub(super) fn commit_picker_result<T, F>(
    workbench: &gpui::WeakEntity<TrackerWorkbench>,
    main_window_handle: AnyWindowHandle,
    window: &mut Window,
    cx: &mut Context<T>,
    on_commit: F,
) where
    F: FnOnce(&mut TrackerWorkbench, &mut Window, &mut Context<TrackerWorkbench>) + 'static,
{
    let workbench = workbench.clone();
    let _ = main_window_handle.update(cx, move |_, main_window, cx| {
        if let Some(workbench) = workbench.upgrade() {
            let _ = workbench.update(cx, |this, cx| on_commit(this, main_window, cx));
        }
    });
    window.remove_window();
}

pub(super) fn picker_control_button(
    id: impl Into<SharedString>,
    label: &'static str,
    background: gpui::Hsla,
    hover_background: gpui::Hsla,
    border_color: gpui::Hsla,
    enabled: bool,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    let button = div()
        .id(id.into())
        .h(px(34.0))
        .px_3()
        .min_w(px(72.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_lg()
        .bg(background)
        .border_1()
        .border_color(border_color)
        .on_mouse_down(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .on_mouse_up(MouseButton::Left, |_, _, cx| {
            cx.stop_propagation();
        })
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(hsla(0.0, 0.0, 1.0, 0.96))
                .child(label),
        );

    if enabled {
        button
            .cursor(CursorStyle::PointingHand)
            .hover(move |style| style.bg(hover_background))
            .active(|style| style.opacity(0.92))
            .on_click(
                move |event: &ClickEvent, window: &mut Window, cx: &mut App| {
                    cx.stop_propagation();
                    on_click(event, window, cx);
                },
            )
    } else {
        button.opacity(0.42)
    }
}
