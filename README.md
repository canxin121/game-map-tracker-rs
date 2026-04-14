# Game Map Tracker RS

完全独立的纯 Rust / GPUI 桌面版 Game Map Tracker。

这个 repo 不再依赖上级 `Game-Map-Tracker` 仓库，也不再依赖本机相邻目录里的 `gpui-component` 源码。运行时地图瓦片、Wiki 图标和点位目录全部由 Rust 直接从 Bilibili Wiki 拉取并缓存；默认配置不再来自模板文件，而是由 Rust `Default` 直接生成并写入用户数据目录，标记组 routes 完全由用户自行导入或创建。

## Standalone 设计

- `gpui` 使用 crates.io 依赖。
- `gpui-component` 使用 crates.io 依赖，不再读取本机 `gpui-component` 子目录。
- BWiki 是运行时真实数据源：
  - 点位类型目录：`Data:Mapnew/type/json`
  - 点位总表：`Data:Mapnew/point.json`
  - 瓦片底图：`map-3.0/{z}/tile-{x}_{y}.png`
- `build.rs` 现在只内置 UI SVG 图标，不再把地图瓦片和点位 PNG 编译进二进制。
- `src/resources/bootstrap.rs` 在本地缺失 `config.toml` 时，用 `AppConfig::default()` 直接生成默认配置。
- `src/resources/bwiki.rs` 负责 Rust 侧数据抓取、瓦片/图标缓存和点位坐标转换。
- `src/resources/bootstrap.rs` 不再写出地图、图标和默认 routes，只创建数据目录与缓存目录。
- JS 脚本不是产品运行链路。应用本身只走 Rust 代码路径。
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
build.rs                编译期扫描 UI 静态资源并生成 include_bytes! 表
src/app/                GPUI 启动、窗口初始化、资产加载源
src/config/             config.toml 的强类型映射与默认值
src/domain/             地图几何、路线、标记点、主题、追踪状态等领域模型
src/resources/          数据目录引导、BWiki 抓取缓存、标记点与 UI 偏好持久化
src/tracking/           截图、模板匹配、Candle 后端、运行时线程、调试快照
src/ui/                 GPUI 工作台、分页导航、双地图子页、标记点编辑界面
```

## 已实现能力

- 独立数据目录引导
  - 首次启动自动生成默认 `config.toml`
- 自动创建 `cache/bwiki/` 目录用于缓存点位、图标和瓦片
  - 不再内置任何默认 routes，标记组完全由用户导入或创建

- GPUI 工作台
  - 多页面导航：地图、标记、设置
  - 地图二级导航：路线追踪 / BWiki 全图
  - 设置页二级导航：配置、调试信息、资源
- 地图页会按视口和缩放级别按需加载 BWiki 瓦片
  - BWiki 全图页支持按分类展示全部 Wiki 类型，并支持多选开关显示
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

默认数据目录位置由 `directories` crate 按平台解析，例如 Windows 下会落到本机用户数据目录。程序会在 `config.toml` 缺失时用 Rust 默认值生成一份新的配置，不会覆盖已有的用户编辑结果；`routes/`、`config.toml`、`.game-map-tracker-rs.toml` 和 `cache/bwiki/` 都固定从该数据目录读取。

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
  cache/
    bwiki/
      data/
      icons/
      tiles/
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
