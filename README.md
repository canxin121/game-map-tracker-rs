# Game Map Tracker RS

完全独立的纯 Rust / GPUI 桌面版 Game Map Tracker。

这个 repo 不再依赖上级 `Game-Map-Tracker` 仓库，也不再依赖本机相邻目录里的 `gpui-component` 源码。地图和点位图标会直接内置进二进制；默认配置不再来自模板文件，而是由 Rust `Default` 直接生成并写入用户数据目录，标记组 routes 完全由用户自行导入或创建。

## Standalone 设计

- `gpui` 使用 crates.io 依赖。
- `gpui-component` 使用 crates.io 依赖，不再读取本机 `gpui-component` 子目录。
- `assets/map` / `assets/points` 存放编译期内置的地图和点位图标资源。
- `build.rs` 只会把静态地图和点位图标生成到编译期静态资源表。
- `src/resources/bootstrap.rs` 在本地缺失 `config.toml` 时，用 `AppConfig::default()` 直接生成默认配置。
- `src/resources/bootstrap.rs` 不再写出地图、图标和默认 routes。
- `src/embedded_assets.rs` 负责给 GPUI 和追踪运行时提供二进制内置资源。
- 运行时默认数据目录在系统用户数据目录下，不会读写父级旧仓库。
- 唯一支持的运行时环境变量是 `GAME_MAP_TRACKER_RS_DATA_DIR`，仅用于覆盖数据目录。

## 快速启动

```powershell
cargo run
```

默认启用 `ai-candle` 特性：

```powershell
cargo run --no-default-features
cargo run --features ai-candle
```

## 当前结构

```text
build.rs                编译期扫描 assets 并生成 include_bytes! 表
assets/map/             内置逻辑地图与显示地图资源
assets/points/          内置点位图标资源
src/app/                GPUI 启动、窗口初始化、资产加载源
src/config/             config.toml 的强类型映射与默认值
src/domain/             地图几何、路线、标记点、主题、追踪状态等领域模型
src/resources/          数据目录引导、资源扫描、标记点与 UI 偏好持久化
src/tracking/           截图、模板匹配、Candle 后端、运行时线程、调试快照
src/ui/                 GPUI 工作台、分页导航、地图画布、标记点编辑界面
```

## 已实现能力

- 独立数据目录引导
  - 首次启动自动生成默认 `config.toml`
  - 地图和标记图标直接由二进制内置提供，不再写入用户目录
  - 不再内置任何默认 routes，标记组完全由用户导入或创建

- GPUI 工作台
  - 多页面导航：地图、标记、设置
  - 设置页二级导航：配置、调试信息、资源
  - 地图页独立占满主内容区，避免被其它面板挤压
  - 缩放 / 拖拽相机
  - 路线节点与折线绘制
  - 实时追踪点与轨迹回显
  - 分组标记点 CRUD
  - 支持多选文件导入和整目录导入 routes
  - 分组多选显示开关
  - 标记点图标、颜色、尺寸样式
  - Follow System / Light / Dark 主题切换和持久化

- 纯 Rust 追踪运行时
  - 使用 `screenshots` 抓取桌面小地图
  - 使用 `image` / `imageproc` 进行灰度化、直方图均衡、模板匹配
  - 使用 `candle-core` / `candle-nn` 进行固定卷积特征编码与张量匹配
  - 统一状态机：`LocalTrack`、`GlobalRelocate`、`InertialHold`
  - 后台线程通过 channel 把状态、坐标和调试图送回 GPUI

## 追踪方案

当前有两个可运行引擎：

- `RustTemplate`
- `CandleAi`

`RustTemplate` 使用多尺度模板匹配金字塔：

- 本地锁定时，在上次坐标附近做局部匹配。
- 丢失时，切回全局低分辨率重定位，再局部精修。
- 连续失败时，进入惯性保位，超过阈值后重新全局搜。

`CandleAi` 使用 Candle 张量后端：

- 固定 3x3 卷积核提取边缘、Laplacian、对角纹理和平滑特征。
- 把 minimap 与地图搜索区域编码成多通道特征图。
- 用张量运算计算 masked cosine-style score map。
- 复用局部锁定、全局重定位、惯性保位状态机。

## 运行时数据目录

默认数据目录位置由 `directories` crate 按平台解析，例如 Windows 下会落到本机用户数据目录。程序会在 `config.toml` 缺失时用 Rust 默认值生成一份新的配置，不会覆盖已有的用户编辑结果；地图和点位图标始终直接使用二进制内置资源，`routes/`、`config.toml`、`.game-map-tracker-rs.toml` 都固定从该数据目录读取。

覆盖数据目录路径：

```powershell
$env:GAME_MAP_TRACKER_RS_DATA_DIR = "D:\path\to\data"
cargo run
```

数据目录结构：

```text
data/
  config.toml
  .game-map-tracker-rs.toml
  routes/
    *.json
```

## 配置字段

Rust 模板引擎支持以下扩展字段。`config.toml` 不写也能跑，会使用默认值：

- `TEMPLATE_REFRESH_RATE`
- `TEMPLATE_LOCAL_DOWNSCALE`
- `TEMPLATE_GLOBAL_DOWNSCALE`
- `TEMPLATE_GLOBAL_REFINE_RADIUS`
- `TEMPLATE_LOCAL_MATCH_THRESHOLD`
- `TEMPLATE_GLOBAL_MATCH_THRESHOLD`
- `TEMPLATE_MASK_OUTER_RADIUS`
- `TEMPLATE_MASK_INNER_RADIUS`

AI / Candle 后端支持：

- `AI_REFRESH_RATE`
- `AI_CONFIDENCE_THRESHOLD`
- `AI_MIN_MATCH_COUNT`
- `AI_RANSAC_THRESHOLD`
- `AI_SCAN_SIZE`
- `AI_SCAN_STEP`
- `AI_TRACK_RADIUS`
- `AI_WEIGHTS_PATH`

## 验证

```powershell
cargo fmt
cargo check
```
