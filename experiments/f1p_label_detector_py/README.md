# F1-P Label Detector Python Experiment

这个目录是独立实验区，只处理已经裁切好的 `587x36` 标签带图片。

目标：

- 精确判断是否存在完整的 `F1 F2 F3 F4 J P` 六连标签
- 使用成熟库：`OpenCV + NumPy + Pillow`
- 同时通过真实 `assets/test` 样本和随机生成负样本
- 检测阶段只保留单一路径，避免任何快慢路径和兜底分支

## 目录

- `samples/positive`: 从主项目复制来的真实正样本
- `samples/negative`: 从主项目复制来的真实负样本，加上脚本生成的随机负样本
- `debug`: 调试输出图
- `artifacts/f1p_tag_model.npz`: 训练出的锚框模板模型
- `artifacts/evaluation_report.json`: 评估报告
- `detector.py`: 建模、检测、同步样本、生成负样本、批量评估脚本

## 方法

这个检测器不是 OCR，也不是通用目标检测，而是针对固定样式标签带做的单一路径判定：

1. 从正样本里利用标签亮色胶囊的固定外观，自动学习 6 个标签的锚框位置。
2. 每张待测图统一缩放到训练尺寸，并只做一次整图灰度转换。
3. 在 6 个固定锚框里分别截取局部灰度图，缩放到统一模板尺寸并做直方图均衡。
4. 每个槽位直接与对应正样本模板库做归一化相关性匹配。
5. 用 `6` 个槽位的 `mean raw score` 和 `min raw score` 一次性完成最终判定。

检测阶段没有第二条路径，没有精细分割兜底，也没有额外慢分支。

## 用法

从主项目 `assets/test` 同步真实正负样本：

```bash
python detector.py sync-assets
```

重建模型：

```bash
python detector.py build-model
```

生成随机负样本：

```bash
python detector.py generate-negatives --count 80
```

评估正负样本：

```bash
python detector.py evaluate --generate-negatives-if-missing
```

单张图片检测：

```bash
python detector.py detect --image samples/positive/has_map_1776668145295.png
```

输出调试图：

```bash
python detector.py detect --image samples/positive/has_map_1776668145295.png --debug-output debug/single_debug.png
```

## 说明

- 当前模型针对的是“已经裁切成标签带”的输入。
- 如果后续要接入整屏截图，建议先做区域定位，再把标签带裁切结果送进这里。
- 构模阶段会重新从 `samples/positive` 学锚框和模板，样本量不大，重建成本很低。
