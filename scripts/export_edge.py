#!/usr/bin/env python3
"""Export a trained checkpoint for edge deployment.

Common targets:
    onnx     - portable, works with onnxruntime on most edge boards
    engine   - TensorRT, for NVIDIA Jetson (fastest, device-specific build)
    openvino - Intel edge hardware (NCS2, NUC with iGPU, etc.)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--format", default="onnx", choices=["onnx", "engine", "openvino", "tflite"]
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 precision (recommended on Jetson)")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (needs calibration data)")
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    out_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        simplify=args.simplify,
    )
    print(f"Exported to: {out_path}")


if __name__ == "__main__":
    main()
