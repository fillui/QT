# -*- coding: utf-8 -*-
"""
infer_pt.py

用于 Qt 程序调用的 YOLO .pt 推理脚本。
输出格式使用 JSON 标记，方便 C++ 端解析。

命令示例：
python infer_pt.py --model best.pt --image test.jpg --out-dir infer_results --conf 0.2 --iou 0.3 --imgsz 640
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
    parser = argparse.ArgumentParser(description="Run YOLO .pt inference and return JSON for Qt.")
    parser.add_argument("--model", required=True, help="Path to .pt model, such as best.pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out-dir", required=True, help="Directory to save result image")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        emit_result({
            "success": False,
            "error": f"模型文件不存在：{model_path}",
            "result_image": "",
            "count": 0,
            "detections": []
        })
        return 1

    if not image_path.exists():
        emit_result({
            "success": False,
            "error": f"图片文件不存在：{image_path}",
            "result_image": "",
            "count": 0,
            "detections": []
        })
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
            raise RuntimeError("YOLO 未返回推理结果")

        result = results[0]
        names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
        detections = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for idx, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, clss), start=1):
                x1, y1, x2, y2 = [float(v) for v in box]
                class_id = int(cls_id)
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                else:
                    class_name = str(names[class_id]) if class_id < len(names) else str(class_id)

                detections.append({
                    "id": idx,
                    "class_id": class_id,
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

        plotted = result.plot()
        result_image = out_dir / (image_path.stem + "_pt_result.jpg")
        save_image_unicode(result_image, plotted)

        emit_result({
            "success": True,
            "error": "",
            "model": str(model_path.resolve()),
            "image": str(image_path.resolve()),
            "result_image": str(result_image.resolve()).replace("\\", "/"),
            "count": len(detections),
            "detections": detections,
        })
        return 0

    except Exception as exc:
        emit_result({
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "result_image": "",
            "count": 0,
            "detections": []
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
