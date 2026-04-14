use derive_more::Display;
use strum::EnumIter;

#[derive(Debug, Clone, Copy, Default, Display, EnumIter, PartialEq, Eq, Hash)]
pub(super) enum WorkbenchPage {
    #[default]
    #[display("地图")]
    Map,
    #[display("路线管理")]
    Markers,
    #[display("设置")]
    Settings,
}

#[derive(Debug, Clone, Copy, Default, Display, EnumIter, PartialEq, Eq, Hash)]
pub(super) enum MapPage {
    #[default]
    #[display("路线追踪")]
    Tracker,
    #[display("节点图鉴")]
    Bwiki,
}

#[derive(Debug, Clone, Copy, Default, Display, EnumIter, PartialEq, Eq, Hash)]
pub(super) enum SettingsPage {
    #[default]
    #[display("配置")]
    Config,
    #[display("调试")]
    Debug,
    #[display("资源")]
    Resources,
}
