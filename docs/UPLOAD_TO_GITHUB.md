# 上传到 GitHub

## 1. 上传前检查

确认仓库中至少包含：

- 源码：`*.cpp`, `*.h`, `*.ui`, `CMakeLists.txt`
- Python 脚本：`python/*.py`
- 依赖说明：`requirements.txt`
- 项目说明：`README.md`
- 模块文档：`docs/MODULES.md`
- 上传说明：`docs/UPLOAD_TO_GITHUB.md`

不要直接提交：

- `build/`
- `.qtcreator/`
- `runs/`
- `*.cache`
- `*.db`
- 大量或重复的 `*.pt` / `*.onnx` 模型文件

这些内容已经在 `.gitignore` 中配置。

## 2. 初始化 Git 仓库

在项目根目录运行：

```bash
git init
git add README.md docs/ .gitignore requirements.txt CMakeLists.txt *.cpp *.h *.ui python/*.py data/data.yaml dataset/data.yaml
git status
git commit -m "Initial project upload"
```

如果你确定要提交数据集图片和标注文件，可以再执行：

```bash
git add data/images data/labels dataset
git commit -m "Add sample dataset"
```

公开仓库上传数据集前，请确认图片没有隐私信息。

## 3. 在 GitHub 新建仓库

1. 打开 GitHub，点击右上角 `+`。
2. 选择 `New repository`。
3. 填写仓库名，例如 `yolov8-qt-focus-detection`。
4. 如果包含私有图片或模型，建议选择 `Private`。
5. 不要勾选自动生成 README，因为本项目已经有 `README.md`。
6. 创建仓库后复制远程地址。

## 4. 关联远程仓库并推送

把下面的地址替换成你自己的 GitHub 仓库地址：

```bash
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

如果远程仓库已经存在 `origin`，使用：

```bash
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 5. 如果要上传模型文件

推荐方式一：GitHub Releases

1. 在 GitHub 仓库页面进入 `Releases`。
2. 创建一个新版本，例如 `v1.0-models`。
3. 上传 `best.pt`、`best_int8.onnx` 等模型文件。
4. 在 `README.md` 中补充模型下载链接。

推荐方式二：Git LFS

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.onnx"
git add .gitattributes
git add models/best.pt
git commit -m "Add model weights with Git LFS"
git push
```

如果使用 Git LFS，需要先从 `.gitignore` 中移除对应的 `*.pt` / `*.onnx` 忽略规则，或使用 `git add -f models/best.pt` 强制添加指定文件。

