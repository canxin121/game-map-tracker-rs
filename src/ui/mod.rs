use gpui::App;

pub mod map_canvas;
mod tile_cache;
mod workbench;

pub use workbench::TrackerWorkbench;

pub(crate) fn init(cx: &mut App) {
    workbench::init(cx);
}
