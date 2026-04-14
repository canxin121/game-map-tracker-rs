use gpui::{App, Hsla, Window};
use gpui_component::{Colorize as _, Theme, ThemeMode};

use crate::domain::theme::ThemePreference;

#[derive(Debug, Clone, Copy)]
pub(super) struct WorkbenchThemeTokens {
    pub(super) app_bg: Hsla,
    pub(super) app_fg: Hsla,
    pub(super) panel_bg: Hsla,
    pub(super) panel_alt_bg: Hsla,
    pub(super) panel_deep_bg: Hsla,
    pub(super) panel_sunken_bg: Hsla,
    pub(super) border: Hsla,
    pub(super) border_strong: Hsla,
    pub(super) nav_item_bg: Hsla,
    pub(super) nav_item_hover_bg: Hsla,
    pub(super) nav_item_active_bg: Hsla,
    pub(super) nav_subitem_active_bg: Hsla,
    pub(super) nav_branch_bg: Hsla,
    pub(super) toolbar_chip_bg: Hsla,
    pub(super) toolbar_button_bg: Hsla,
    pub(super) toolbar_button_hover_bg: Hsla,
    pub(super) toolbar_button_primary_bg: Hsla,
    pub(super) toolbar_button_primary_hover_bg: Hsla,
    pub(super) toolbar_button_danger_bg: Hsla,
    pub(super) toolbar_button_danger_hover_bg: Hsla,
    pub(super) text_muted: Hsla,
    pub(super) text_soft: Hsla,
    pub(super) map_canvas_backdrop: Hsla,
    pub(super) debug_canvas_bg: Hsla,
    pub(super) debug_card_bg: Hsla,
    pub(super) trail_path: Hsla,
    pub(super) preview_live: Hsla,
    pub(super) preview_inertial: Hsla,
    pub(super) preview_ring: Hsla,
    pub(super) selected_marker_border: Hsla,
}

impl WorkbenchThemeTokens {
    pub(super) fn from_theme(theme: &Theme) -> Self {
        let is_dark = theme.mode.is_dark();

        Self {
            app_bg: theme.background,
            app_fg: theme.foreground,
            panel_bg: theme
                .background
                .mix(theme.secondary, if is_dark { 0.68 } else { 0.92 }),
            panel_alt_bg: theme
                .background
                .mix(theme.popover, if is_dark { 0.58 } else { 0.88 }),
            panel_deep_bg: theme
                .background
                .mix(theme.sidebar, if is_dark { 0.52 } else { 0.82 }),
            panel_sunken_bg: theme
                .background
                .mix(theme.muted, if is_dark { 0.74 } else { 0.9 }),
            border: theme.border,
            border_strong: theme
                .border
                .mix(theme.foreground, if is_dark { 0.8 } else { 0.65 }),
            nav_item_bg: theme
                .background
                .mix(theme.sidebar, if is_dark { 0.58 } else { 0.88 }),
            nav_item_hover_bg: theme
                .background
                .mix(theme.secondary, if is_dark { 0.7 } else { 0.9 }),
            nav_item_active_bg: theme
                .background
                .mix(theme.accent, if is_dark { 0.42 } else { 0.78 }),
            nav_subitem_active_bg: theme
                .background
                .mix(theme.accent, if is_dark { 0.26 } else { 0.88 }),
            nav_branch_bg: theme
                .background
                .mix(theme.sidebar, if is_dark { 0.42 } else { 0.8 }),
            toolbar_chip_bg: theme
                .background
                .mix(theme.popover, if is_dark { 0.52 } else { 0.84 }),
            toolbar_button_bg: theme
                .background
                .mix(theme.secondary, if is_dark { 0.62 } else { 0.9 }),
            toolbar_button_hover_bg: theme
                .background
                .mix(theme.secondary, if is_dark { 0.76 } else { 0.82 }),
            toolbar_button_primary_bg: theme
                .background
                .mix(theme.accent, if is_dark { 0.44 } else { 0.74 }),
            toolbar_button_primary_hover_bg: theme
                .background
                .mix(theme.accent, if is_dark { 0.58 } else { 0.64 }),
            toolbar_button_danger_bg: theme
                .background
                .mix(theme.danger, if is_dark { 0.38 } else { 0.76 }),
            toolbar_button_danger_hover_bg: theme
                .background
                .mix(theme.danger, if is_dark { 0.52 } else { 0.66 }),
            text_muted: theme.muted_foreground,
            text_soft: theme
                .muted_foreground
                .mix(theme.foreground, if is_dark { 0.92 } else { 0.84 }),
            map_canvas_backdrop: theme
                .background
                .mix(theme.muted, if is_dark { 0.8 } else { 0.94 }),
            debug_canvas_bg: theme
                .background
                .mix(theme.secondary, if is_dark { 0.86 } else { 0.96 }),
            debug_card_bg: theme
                .background
                .mix(theme.popover, if is_dark { 0.5 } else { 0.82 }),
            trail_path: theme.warning.opacity(0.55),
            preview_live: theme.danger,
            preview_inertial: theme.warning,
            preview_ring: theme
                .background
                .mix(theme.foreground, if is_dark { 0.12 } else { 0.45 }),
            selected_marker_border: theme.danger,
        }
    }
}

pub(super) fn apply_theme_preference(
    preference: ThemePreference,
    window: &mut Window,
    cx: &mut App,
) {
    match preference {
        ThemePreference::FollowSystem => Theme::sync_system_appearance(Some(window), cx),
        ThemePreference::Light => Theme::change(ThemeMode::Light, Some(window), cx),
        ThemePreference::Dark => Theme::change(ThemeMode::Dark, Some(window), cx),
    }
}
