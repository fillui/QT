# 主要模块说明

## 1. Qt 桌面端

相关文件：

- `main.cpp`
- `mainwindow.cpp`
- `mainwindow.h`
- `mainwindow.ui`
- `CMakeLists.txt`

职责：

- 启动 Qt 应用并加载主窗口。
- 管理标注、训练、量化、推理等功能页。
- 通过 `QProcess` 调用 Python 脚本。
- 使用 Qt SQL 记录训练和推理相关结果。

## 2. 图像标注模块

相关文件：

- `imagecanvas.cpp`
- `imagecanvas.h`
- `annotation.h`
- `annotationio.cpp`
- `annotationio.h`

职责：

- 加载图片并展示在画布中。
- 绘制、选择、移动、调整矩形标注框。
- 支持缩放、平移、旋转、撤销和重做。
- 保存和读取 YOLO 风格标注文本。

## 3. 数据集模块

相关文件/目录：

- `data/data.yaml`
- `dataset/data.yaml`
- `data/images/`
- `data/labels/`
- `dataset/`

职责：

- 管理训练集和验证集图片。
- 管理 YOLO 标注文件。
- 生成或维护 `data.yaml`，供 YOLOv8 训练读取。

当前类别：

- `focused`
- `unfocused`

## 4. 训练模块

相关文件：

- `python/train.py`

职责：

- 调用 `ultralytics.YOLO` 执行训练。
- 支持设置 epoch、batch size、学习率、图像尺寸、设备等参数。
- 向 Qt 输出专用训练进度标记，便于界面实时更新。
- 训练完成后复制 `best.pt` 到固定模型目录。

## 5. 量化模块

相关文件：

- `python/quantize_model.py`

职责：

- 支持从 `.pt` 导出 `.onnx`。
- 支持 ONNX Runtime 动态量化和静态量化。
- 静态量化时使用校准图片生成 INT8 ONNX 模型。
- 对 YOLOv8 中容易出错的 Detect Head/DFL 等节点做排除处理。

## 6. 推理模块

相关文件：

- `python/infer_pt.py`
- `python/infer_onnx.py`

职责：

- 加载 PT 或 ONNX 模型执行目标检测。
- 保存带检测框的结果图片。
- 以 JSON 格式输出检测框、类别、置信度和结果图片路径，供 Qt 前端解析展示。

## 7. 训练输出与模型资产

相关目录：

- `models/`
- `runs/`

说明：

- `models/` 可放置当前应用要加载的模型。
- `runs/` 是 Ultralytics 训练输出目录，通常包含中间权重、ONNX 导出模型、日志和指标。
- 建议不要直接把 `runs/` 提交到 GitHub；如需发布模型，请使用 Git LFS 或 GitHub Releases。

