# Rocom BWiki 大地图结构说明

这份说明基于 2026-04-14 对 `https://wiki.biligame.com/rocom/?curid=981` 的页面结构检查，重点记录“图标和点位数据如何自动抓取”。

## 已确认的数据源

主页面本身只是容器，真正的数据来源分成两层：

- `Data:Mapnew/type/json`
  - 点位类型目录
  - 包含分类、`markType`、显示名称、图标、说明等
- `Data:Mapnew/point.json`
  - 聚合后的点位总表
  - 展开后是 `{ 201:[...], 202:[...], ... }` 这种对象

页面脚本里也明确写了：

- `mapData.dataPrefix = "Data:Mapnew"`
- `mapData.dataList = ["point.json"]`

也就是说，前端实际就是从 `Data:Mapnew` 下面拉这些数据页。

## 关键结论

- 图标 URL 可以直接从展开后的 `type/json` 里拿到，不需要手工扒页面元素。
- 点位列表可以直接从展开后的 `point.json` 总表拿到，不需要逐个点击页面上的点位。
- 点位详情页不是稳定主链路。抽样的真实点位 id 没有普遍对应到可访问的 `Data:Mapnew/point/<id>` 页面，所以不要把详情页当成主数据源。
- 当前最适合自动化的抓取对象是：
  - 类型目录
  - 图标 URL
  - 各 `markType` 对应的点位数组

## 字段现状

类型目录里常见字段：

- `type`
- `markType`
- `markTypeName`
- `icon`
- `desc`
- `class`
- `collectible`
- `geojson`

点位数组里常见字段：

- `markType`
- `title`
- `id`
- `point.lat`
- `point.lng`
- `uid`
- `layer`
- `time`
- `version`

当前页面上看到的主流点位数据 `version` 基本是 `4`，这类坐标已经是前端直接使用的地图坐标。

## 坐标注意事项

这个仓库自己的底图是 `assets/map/display_map.png` 和 `assets/map/logic_map.png`，尺寸都是 `6144x5888`。

BWiki 点位里的 `point.lat/lng` 则是 Leaflet 地图使用的原始坐标，不等同于这个仓库里 `0..width / 0..height` 的图片像素坐标。

这意味着：

- 当前同步脚本导出的是 BWiki 原始坐标
- 可以稳定抓取和落盘
- 但不能在没有额外校准的前提下，直接保证和仓库底图一一对齐

如果后面要直接导入到这个 Rust 工具里，需要再补一层坐标映射标定。

## 自动化方案

推荐的自动化主链路是：

1. 请求 `api.php?action=parse&page=Data:Mapnew/type/json&prop=text&format=json&formatversion=2`
2. 去掉 HTML 标签，得到展开后的类型 JSON
3. 请求 `api.php?action=parse&page=Data:Mapnew/point.json&prop=text&format=json&formatversion=2`
4. 去掉 HTML 标签，并按前端原逻辑把 `:Data:.../json` 替换成 `:[]`
5. 把结果解释成 JS 对象，得到 `pointsByType`
6. 把类型表、点位表、平铺点位表、图标文件统一落盘

这样做的优点：

- 请求数很少
- 不依赖浏览器点击
- 直接复用页面当前已经展开好的结果
- 图标 URL 已经是最终 CDN 地址

## 仓库里的同步脚本

已提供：

- `scripts/sync-bwiki-rocom.mjs`
- `scripts/sync-bwiki-rocom-tiles.mjs`

默认输出目录：

- `.tmp-bwiki-rocom/`

运行方式：

```powershell
node scripts/sync-bwiki-rocom.mjs
node scripts/sync-bwiki-rocom.mjs --skip-icons
node scripts/sync-bwiki-rocom.mjs --out-dir D:\tmp\rocom-bwiki
```

脚本会生成：

- `types.json`
- `points-by-type.json`
- `flat-points.json`
- `summary.json`
- `manifest.json`
- `icons/`

瓦片脚本用途：

- 自动探测 `z=4..8` 的有效瓦片范围
- 下载各个 zoom 的原始瓦片 PNG
- 可选把每个 zoom 拼成一张完整底图

运行方式：

```powershell
node scripts/sync-bwiki-rocom-tiles.mjs
node scripts/sync-bwiki-rocom-tiles.mjs --probe-only
node scripts/sync-bwiki-rocom-tiles.mjs --min-zoom 8 --max-zoom 8 --skip-stitch
```

目前已确认的有效范围是：

- `z=4` -> `x=-2..1`, `y=-2..1`
- `z=5` -> `x=-3..2`, `y=-3..2`
- `z=6` -> `x=-6..5`, `y=-5..4`
- `z=7` -> `x=-12..11`, `y=-9..8`
- `z=8` -> `x=-24..23`, `y=-18..17`

## 当前检查结果

按 2026-04-14 这次抓取结果，页面当前大致是：

- 类型定义约 `291` 个
- 聚合点位约 `2525` 个
- 点位总表里出现的类型约 `154` 个
- 其中非空类型约 `139` 个
- 其中有 `804/805/806` 三个 `markType` 在点位总表里存在，但当前 `type/json` 目录里没有对应定义

这些数字会随 Wiki 更新变化，脚本每次运行都会重新拉最新数据。
