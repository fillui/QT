import argparse
import json
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
#导入 ONNX Runtime 的量化工具
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    quantize_dynamic,
    quantize_static
)

try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
except Exception:
    quant_pre_process = None


def send_progress(value, msg):
    print(json.dumps({
        "type": "progress",
        "value": int(value),
        "msg": str(msg)
    }, ensure_ascii=False), flush=True)


def send_error(msg):
    print(json.dumps({
        "type": "error",
        "msg": str(msg)
    }, ensure_ascii=False), flush=True)


def collect_images(image_dir, max_images):
    image_dir = Path(image_dir)

    suffixes = [".jpg", ".jpeg", ".png", ".bmp"]

    image_paths = []

    for p in image_dir.rglob("*"):
        if p.suffix.lower() in suffixes:
            image_paths.append(str(p))

    image_paths = sorted(image_paths)

    if max_images > 0:
        image_paths = image_paths[:max_images]

    return image_paths


def letterbox_image(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]

    scale = min(new_size / h, new_size / w)

    nh = int(round(h * scale))
    nw = int(round(w * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top + nh, left:left + nw] = resized

    return canvas


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, img_size=640, max_images=100):
        self.input_name = input_name
        self.img_size = img_size
        self.image_paths = collect_images(image_dir, max_images)
        self.index = 0

        if len(self.image_paths) == 0:
            raise RuntimeError(f"校准图片为空: {image_dir}")

        send_progress(20, f"找到校准图片 {len(self.image_paths)} 张")

    def preprocess(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

        img = letterbox_image(img, self.img_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # CHW -> NCHW
        img = np.expand_dims(img, axis=0)

        return img

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        progress = 25 + int(50 * self.index / max(1, len(self.image_paths)))

        send_progress(
            progress,
            f"校准 {self.index}/{len(self.image_paths)}: {os.path.basename(image_path)}"
        )

        return {
            self.input_name: self.preprocess(image_path)
        }

# 用 ONNX Runtime 读取模型输入节点
def get_input_name(onnx_path):
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    return session.get_inputs()[0].name


def export_pt_to_onnx(pt_path, img_size, output_dir):
    from ultralytics import YOLO

    send_progress(8, "检测到 .pt 模型，开始导出 ONNX")

    model = YOLO(pt_path)

    exported = model.export(
        format="onnx",
        imgsz=img_size,
        opset=12,
        simplify=False,
        dynamic=False
    )

    exported_path = str(exported)

    if not os.path.exists(exported_path):
        raise RuntimeError(f"ONNX 导出失败: {exported_path}")

    output_fp32 = os.path.join(output_dir, "best_fp32.onnx")

    if os.path.abspath(exported_path) != os.path.abspath(output_fp32):
        shutil.copyfile(exported_path, output_fp32)

    send_progress(15, f"ONNX 导出完成: {output_fp32}")

    return output_fp32


def preprocess_onnx_for_quant(onnx_model, output_dir):
    if quant_pre_process is None:
        send_progress(19, "当前 onnxruntime 不支持 quant_pre_process，跳过预处理")
        return onnx_model

    preprocessed_model = os.path.join(output_dir, "best_fp32_preprocessed.onnx")

    send_progress(19, "执行 ONNX 量化预处理")

    try:
        quant_pre_process(
            input_model_path=onnx_model,
            output_model_path=preprocessed_model,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False
        )

        if os.path.exists(preprocessed_model):
            send_progress(21, f"ONNX 预处理完成: {preprocessed_model}")
            return preprocessed_model

    except Exception as e:
        send_progress(21, f"ONNX 预处理失败，继续使用原模型: {e}")

    return onnx_model


def collect_safe_conv_nodes(onnx_model):
    """
    只收集适合量化的 Conv 节点。
    排除 YOLOv8 Detect Head / DFL / Softmax 等容易出错的部分。
    """
    model = onnx.load(onnx_model)

    nodes_to_quantize = []
    nodes_to_exclude = []

    exclude_keywords = [
        "model.22",
        "/dfl/",
        "dfl",
        "Softmax",
        "softmax",
        "Concat",
        "Reshape",
        "Transpose",
        "ReduceMax",
        "ReduceMin",
        "Split",
        "Sigmoid",
        "Mul",
        "Add"
    ]

    for node in model.graph.node:
        node_name = node.name or ""
        io_text = " ".join(list(node.input) + list(node.output))
        full_text = node_name + " " + io_text

        need_exclude = any(k in full_text for k in exclude_keywords)

        if need_exclude:
            if node_name:
                nodes_to_exclude.append(node_name)
            continue

        if node.op_type == "Conv":
            if node_name:
                nodes_to_quantize.append(node_name)

    send_progress(
        78,
        f"可量化 Conv 节点数量: {len(nodes_to_quantize)}，排除节点数量: {len(nodes_to_exclude)}"
    )

    if len(nodes_to_quantize) == 0:
        raise RuntimeError("没有找到可安全量化的 Conv 节点，请检查 ONNX 模型结构")

    return nodes_to_quantize, nodes_to_exclude


def run_dynamic_quant(onnx_model, output_model, per_channel=False):
    send_progress(30, "开始动态量化")

    quantize_dynamic(
        model_input=onnx_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
        per_channel=per_channel
    )

    send_progress(100, f"动态量化完成: {output_model}")


def run_static_quant(onnx_model, output_model, calib_dir, img_size, max_calib, output_dir):
    if not calib_dir or not os.path.exists(calib_dir):
        raise FileNotFoundError("静态量化需要有效的校准图片文件夹")

    send_progress(22, "读取 ONNX 输入节点")
    input_name = get_input_name(onnx_model)
    send_progress(24, f"输入节点: {input_name}")

    data_reader = YoloCalibrationDataReader(
        image_dir=calib_dir,
        input_name=input_name,
        img_size=img_size,
        max_images=max_calib
    )

    onnx_model_for_quant = preprocess_onnx_for_quant(
        onnx_model=onnx_model,
        output_dir=output_dir if output_dir else "."
    )

    nodes_to_quantize, nodes_to_exclude = collect_safe_conv_nodes(onnx_model_for_quant)

    send_progress(80, "开始写入 INT8 量化模型：仅量化安全 Conv 节点")

    quantize_static(
        model_input=onnx_model_for_quant,
        model_output=output_model,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False
    )

    send_progress(100, f"静态 INT8 量化完成: {output_model}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="输入模型，支持 .pt 或 .onnx")
    parser.add_argument("--output", required=True, help="输出 INT8 ONNX 模型路径")
    parser.add_argument("--mode", default="static", choices=["static", "dynamic"])
    parser.add_argument("--calib_dir", default="", help="静态量化校准图片文件夹")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--max_calib", type=int, default=100)
    parser.add_argument("--per_channel", action="store_true")

    args = parser.parse_args()

    try:
        input_model = args.model
        output_model = args.output

        if not os.path.exists(input_model):
            raise FileNotFoundError(f"输入模型不存在: {input_model}")

        output_dir = os.path.dirname(output_model)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        send_progress(3, "开始模型量化")
        send_progress(5, f"输入模型: {input_model}")

        suffix = Path(input_model).suffix.lower()

        if suffix == ".pt":
            onnx_model = export_pt_to_onnx(
                pt_path=input_model,
                img_size=args.img_size,
                output_dir=output_dir if output_dir else "."
            )
        elif suffix == ".onnx":
            onnx_model = input_model
        else:
            raise RuntimeError("输入模型只支持 .pt 或 .onnx")

        send_progress(18, f"待量化 ONNX: {onnx_model}")

        if args.mode == "dynamic":
            run_dynamic_quant(
                onnx_model=onnx_model,
                output_model=output_model,
                per_channel=args.per_channel
            )
        else:
            run_static_quant(
                onnx_model=onnx_model,
                output_model=output_model,
                calib_dir=args.calib_dir,
                img_size=args.img_size,
                max_calib=args.max_calib,
                output_dir=output_dir if output_dir else "."
            )

    except Exception as e:
        send_error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()