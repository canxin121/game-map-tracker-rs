use std::borrow::Cow;

use anyhow::Result;
use gpui::{
    AppContext, Application, AssetSource, Bounds, SharedString, WindowBounds, WindowOptions, px,
    size,
};
use gpui_component::Root;

use crate::{
    embedded_assets,
    resources::WorkspaceBootstrap,
    ui::{self, TrackerWorkbench},
};

pub fn launch() -> Result<()> {
    init_tracing();
    configure_windows_capture_compatibility();

    let workspace = WorkspaceBootstrap::prepare()?;

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
        const GPUI_DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";

        if std::env::var_os(GPUI_DISABLE_DIRECT_COMPOSITION).is_none() {
            // GPUI's DirectComposition path uses WS_EX_NOREDIRECTIONBITMAP, which some
            // screen recorders fail to capture reliably and can cause the content layer
            // to blink or disappear. Default to the more compatible swap-chain path.
            unsafe {
                std::env::set_var(GPUI_DISABLE_DIRECT_COMPOSITION, "1");
            }
        }
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,gpui_component=warn")
        .try_init();
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddedAssetSource;

impl AssetSource for EmbeddedAssetSource {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok(embedded_assets::asset_bytes(path).map(Cow::Borrowed))
    }

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>> {
        let mut entries = embedded_assets::runtime_asset_paths(path)
            .into_iter()
            .map(SharedString::from)
            .collect::<Vec<_>>();
        entries.sort_unstable();
        Ok(entries)
    }
}
