# -*- coding: utf-8 -*-
"""
train.py

Qt 前端调用的 YOLOv8 训练脚本。

作用：
1. 训练 YOLO 模型；
2. 每个 epoch 结束时输出专用进度标记：__TRAIN_PROGRESS__ current/total；
3. 训练完成后自动保留 best.pt，并复制到固定目录 models/best.pt。

示例：
python train.py --data dataset/data.yaml --epochs 50 --batch-size 8 --lr 0.001 --model yolov8s.pt
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 避免 Qt/Anaconda/OpenMP 在 Windows 下冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 training script for Qt GUI.")
    parser.add_argument("--data", type=str, required=True, help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="base model, such as yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--project", type=str, default="../runs", help="Ultralytics project directory")
    parser.add_argument("--name", type=str, default="focus_detect", help="Ultralytics run name")
    parser.add_argument("--save-dir", type=str, default="../models", help="fixed directory to copy best.pt")
    parser.add_argument("--device", type=str, default="cpu", help="training device, such as cpu or 0")
    return parser.parse_args()


def resolve_save_dir(train_results, model):
    """兼容不同 Ultralytics 版本，尽量拿到本次训练输出目录。"""
    if train_results is not None and hasattr(train_results, "save_dir"):
        value = getattr(train_results, "save_dir")
        if value:
            return Path(value)

    trainer = getattr(model, "trainer", None)
    if trainer is not None and hasattr(trainer, "save_dir"):
        value = getattr(trainer, "save_dir")
        if value:
            return Path(value)

    return None


def copy_best_pt(save_dir, fixed_save_dir):
    if save_dir is None:
        print("[WARNING] 未能获取训练输出目录，无法自动复制 best.pt", flush=True)
        return

    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"train result dir: {save_dir}", flush=True)
    print(f"best.pt path: {best_pt}", flush=True)
    print(f"last.pt path: {last_pt}", flush=True)

    fixed_dir = Path(fixed_save_dir)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_best = fixed_dir / "best.pt"

    if best_pt.exists():
        shutil.copyfile(best_pt, fixed_best)
        print(f"best.pt copied to: {fixed_best.resolve()}", flush=True)
    else:
        print(f"[WARNING] best.pt not found: {best_pt}", flush=True)


def main():
    args = parse_args()

    print(f"data: {args.data}", flush=True)
    print(f"epochs: {args.epochs}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"lr0: {args.lr}", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"imgsz: {args.imgsz}", flush=True)
    print(f"project: {args.project}", flush=True)
    print(f"name: {args.name}", flush=True)
    print(f"device: {args.device}", flush=True)
    print("start training...", flush=True)

    if not os.path.exists(args.data):
        print(f"[ERROR] data.yaml not found: {args.data}", flush=True)
        sys.exit(1)

    model = YOLO(args.model)

    def on_train_epoch_end(trainer):
        current_epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(args.epochs)
        print(f"__TRAIN_PROGRESS__ {current_epoch}/{total_epochs}", flush=True)

    # 关键：让 Qt 只解析这个专用标记，不再从普通日志里猜 1/50。
    try:
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
    except Exception as exc:
        print(f"[WARNING] 添加 epoch 进度回调失败: {exc}", flush=True)

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr,
            project=args.project,
            name=args.name,
            verbose=False,
            device=args.device,
            workers=0,
            plots=False,
            amp=False
        )
    except Exception as exc:
        print(f"[ERROR] training failed: {exc}", flush=True)
        sys.exit(1)

    save_dir = resolve_save_dir(results, model)
    copy_best_pt(save_dir, args.save_dir)

    print("training finished", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
