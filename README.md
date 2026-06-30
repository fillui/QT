# YOLOv8 Qt 标注训练与推理工具

这是一个基于 Qt Widgets 和 YOLOv8 的桌面端项目，用于完成图像标注、数据集整理、模型训练、模型量化以及 PT/ONNX 推理验证。

## 主要模块

| 模块 | 入口文件/目录 | 说明 |
| --- | --- | --- |
| Qt 主界面 | `main.cpp`, `mainwindow.cpp`, `mainwindow.h`, `mainwindow.ui` | 负责界面交互、数据集操作、训练/量化/推理流程调度。 |
| 图像标注画布 | `imagecanvas.cpp`, `imagecanvas.h` | 支持图片加载、矩形框绘制、选中、拖拽、缩放、旋转、撤销/重做。 |
| 标注数据读写 | `annotation.h`, `annotationio.cpp`, `annotationio.h` | 定义标注结构，并保存/读取 YOLO 格式标注文件。 |
| Python 训练脚本 | `python/train.py` | 调用 Ultralytics YOLOv8 训练模型，并向 Qt 输出训练进度。 |
| 模型量化脚本 | `python/quantize_model.py` | 将 PT/ONNX 模型导出或量化为 ONNX INT8 模型。 |
| 推理脚本 | `python/infer_pt.py`, `python/infer_onnx.py` | 对 PT 或 ONNX 模型执行推理，并输出 JSON 结果供 Qt 解析。 |
| 数据集配置 | `data/data.yaml`, `dataset/data.yaml` | YOLO 数据集配置，当前类别为 `focused` 和 `unfocused`。 |
| 项目文档 | `docs/` | 存放模块说明和 GitHub 上传说明。 |

## 环境要求

- Qt 5 或 Qt 6，需包含 `Widgets` 和 `Sql` 模块
- CMake 3.16+
- C++17 编译器
- Python 3.9+，并安装 `requirements.txt` 中的依赖

Python 依赖安装：

```bash
pip install -r requirements.txt
```

## 构建运行

使用 Qt Creator 打开项目根目录的 `CMakeLists.txt`，选择合适的 Kit 后构建运行。

也可以使用命令行构建：

```bash
cmake -S . -B build
cmake --build build
```

## 数据与模型说明

- `runs/` 是训练输出目录，不建议提交到 Git。
- `*.pt` 和 `*.onnx` 模型文件默认被 `.gitignore` 忽略。
- 如果需要发布模型权重，建议使用 Git LFS 或上传到 GitHub Releases，并在文档中写明下载位置。
- 如果数据集包含隐私图片，上传到公开仓库前请先脱敏或改为私有仓库。

## 文档索引

- [主要模块说明](docs/MODULES.md)
- [上传到 GitHub](docs/UPLOAD_TO_GITHUB.md)

