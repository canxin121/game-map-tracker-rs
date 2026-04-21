use std::borrow::Cow;

use gpui::{
    AppContext, Application, AssetSource, Bounds, SharedString, WindowBounds, WindowOptions, px,
    size,
};
use gpui_component::Root;
use tracing::info;
use tracing_subscriber::{
    EnvFilter,
    layer::{Layer as _, SubscriberExt},
    util::SubscriberInitExt,
};

use crate::{
    embedded_assets,
    error::Result,
    logging::{DebugLogLayer, install_debug_log_store},
    resources::WorkspaceBootstrap,
    ui::{self, TrackerWorkbench},
};

pub fn launch() -> Result<()> {
    init_tracing();
    info!("starting application launch");
    configure_windows_capture_compatibility();

    let workspace = WorkspaceBootstrap::prepare()?;
    info!(workspace_root = %workspace.workspace_root.display(), "workspace prepared");

    Application::new()
        .with_assets(EmbeddedAssetSource)
        .run(move |cx| {
            gpui_component::init(cx);
            ui::init(cx);

            cx.open_window(
                WindowOptions {
                    titlebar: Some(gpui::TitlebarOptions {
                        title: Some("Game Map Tracker RS".into()),
                        appears_transparent: false,
                        ..Default::default()
                    }),
                    window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                        None,
                        size(px(1480.0), px(920.0)),
                        cx,
                    ))),
                    ..Default::default()
                },
                {
                    let workspace_root = workspace.workspace_root.clone();
                    move |window, cx| {
                        let view =
                            cx.new(|cx| TrackerWorkbench::new(workspace_root.clone(), window, cx));
                        cx.new(|cx| Root::new(view, window, cx))
                    }
                },
            )
            .expect("failed to open application window");

            cx.activate(true);
        });

    Ok(())
}

fn configure_windows_capture_compatibility() {
    #[cfg(target_os = "windows")]
    {
        const GAME_MAP_TRACKER_CAPTURE_COMPAT: &str = "GAME_MAP_TRACKER_CAPTURE_COMPAT";
        const GPUI_DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";

        if std::env::var_os(GPUI_DISABLE_DIRECT_COMPOSITION).is_none()
            && std::env::var_os(GAME_MAP_TRACKER_CAPTURE_COMPAT).is_some()
        {
            // Transparent picker overlays require DirectComposition on Windows.
            // Keep the older capture-compatibility path as an explicit opt-in.
            unsafe {
                std::env::set_var(GPUI_DISABLE_DIRECT_COMPOSITION, "1");
            }
        }
    }
}

pub fn init_tracing() {
    let debug_log_store = install_debug_log_store(2_000);
    let fmt_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,gpui_component=warn"));
    let debug_filter = EnvFilter::new("game_map_tracker_rs=debug,info,gpui_component=warn");

    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_thread_names(true)
                .with_target(true)
                .with_filter(fmt_filter),
        )
        .with(DebugLogLayer::new(debug_log_store).with_filter(debug_filter))
        .try_init();
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddedAssetSource;

impl AssetSource for EmbeddedAssetSource {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        Ok(embedded_assets::asset_bytes(path).map(Cow::Borrowed))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        let mut entries = embedded_assets::runtime_asset_paths(path)
            .into_iter()
            .map(SharedString::from)
            .collect::<Vec<_>>();
        entries.sort_unstable();
        Ok(entries)
    }
}
