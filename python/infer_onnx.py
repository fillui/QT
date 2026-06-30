# -*- coding: utf-8 -*-
"""
infer_onnx.py

用于 Qt 程序调用的 YOLOv8 ONNX 推理脚本。
输出格式使用 JSON 标记，方便 C++ 端解析。
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def emit_result(payload):
    """输出带标记的 JSON，避免被 Ultralytics 日志干扰。"""
    print("__RESULT_JSON_START__", flush=True)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    print("__RESULT_JSON_END__", flush=True)


def save_image_unicode(path, image):
    """兼容 Windows 中文路径保存图片。"""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ONNX 模型路径，例如 best_int8.onnx")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--out-dir", default="infer_results", help="输出结果目录")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)

    if not model_path.exists():
        emit_result({"success": False, "error": f"模型文件不存在: {model_path}"})
        return 1

    if not image_path.exists():
        emit_result({"success": False, "error": f"测试图片不存在: {image_path}"})
        return 1

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=False,
            verbose=False,
            agnostic_nms=True
        )

        if not results:
            emit_result({"success": False, "error": "模型没有返回推理结果"})
            return 2

        result = results[0]
        names = getattr(model, "names", None) or getattr(result, "names", {}) or {}

        detections = []
        boxes = getattr(result, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for idx, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss), start=1):
                x1, y1, x2, y2 = [float(v) for v in box]
                if isinstance(names, dict):
                    class_name = str(names.get(int(cls_id), int(cls_id)))
                else:
                    class_name = str(names[int(cls_id)]) if int(cls_id) < len(names) else str(int(cls_id))

                detections.append({
                    "id": idx,
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "x": x1,
                    "y": y1,
                    "w": max(0.0, x2 - x1),
                    "h": max(0.0, y2 - y1),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

        annotated = result.plot()
        result_image = out_dir / f"{image_path.stem}_infer.jpg"
        save_image_unicode(result_image, annotated)

        emit_result({
            "success": True,
            "model": str(model_path.resolve()),
            "image": str(image_path.resolve()),
            "result_image": str(result_image.resolve()),
            "count": len(detections),
            "detections": detections,
        })
        return 0

    except Exception as exc:
        emit_result({
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 3


if __name__ == "__main__":
    sys.exit(main())
