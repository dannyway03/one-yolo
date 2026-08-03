#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) Intel Corporation
#
# Export a YOLO26 person detector for crowd detection to OpenVINO IR.
# Usage: ./export_and_quantize.sh [MODEL_VARIANT] [PRECISION]
# Example: ./export_and_quantize.sh yolo26n FP16
#
# Supported precisions:
#   FP32  -- Full-precision floating-point weights
#   FP16  -- Half-precision floating-point weights (default)
#   INT8  -- Quantized 8-bit integer weights (requires NNCF)
#
# Precision / device compatibility:
#   | Precision | CPU | GPU | NPU |
#   |-----------|-----|-----|-----|
#   | FP32      | Yes | Yes | No  |
#   | FP16      | Yes | Yes | Yes |
#   | INT8      | Yes | Yes | Yes |

set -euo pipefail

MODEL_NAME="${1:-yolo26n}"
PRECISION="${2:-FP16}"
PRECISION="$(echo "${PRECISION}" | tr '[:lower:]' '[:upper:]')"

if [[ "${PRECISION}" != "FP32" && "${PRECISION}" != "FP16" && "${PRECISION}" != "INT8" ]]; then
    echo "ERROR: unsupported precision '${PRECISION}'. Choose FP32, FP16, or INT8." >&2
    exit 1
fi

echo "--- Installing dependencies ---"
if [[ "${PRECISION}" == "INT8" ]]; then
    pip install -qU openvino nncf ultralytics
else
    pip install -qU openvino ultralytics
fi

# Ask for approval before downloading models and sample files
echo ""
echo "This script will download:"
echo "  - YOLO26 model weights (if not cached locally)"
echo "  - Sample test image and video files"
echo ""
read -p "Continue with downloads? (yes/no): " APPROVAL
if [[ "${APPROVAL}" != "yes" ]]; then
    echo "Download cancelled by user."
    exit 0
fi
echo ""

echo "--- Downloading sample test image ---"
if [[ ! -f test.jpg ]]; then
    wget -q -O test.jpg https://ultralytics.com/images/bus.jpg
    echo "Downloaded: test.jpg"
else
    echo "Already present: test.jpg"
fi

echo "--- Downloading sample test video ---"
if [[ ! -f test_video.mp4 ]]; then
    wget -q -O test_video.mp4 \
        https://github.com/open-edge-platform/edge-ai-resources/raw/main/videos/VIRAT_S_000101.mp4
    echo "Downloaded: test_video.mp4"
else
    echo "Already present: test_video.mp4"
fi

if [[ "${PRECISION}" == "FP32" ]]; then
    HALF_FLAG="False"
    EXPORT_LABEL="FP32"
else
    HALF_FLAG="True"
    EXPORT_LABEL="FP16"
fi

echo "--- Exporting ${MODEL_NAME} to OpenVINO IR (${EXPORT_LABEL}) ---"
python3 -c "
from ultralytics import YOLO

model = YOLO('${MODEL_NAME}.pt')
model.export(format='openvino', half=${HALF_FLAG}, dynamic=False, imgsz=(384,640))
print('Export complete: ${MODEL_NAME}_openvino_model/')
"

if [[ "${PRECISION}" == "INT8" ]]; then
    echo "--- Quantizing to INT8 with NNCF ---"
    python3 -c "
import nncf
import openvino as ov
import numpy as np
import cv2

core = ov.Core()
model = core.read_model('${MODEL_NAME}_openvino_model/${MODEL_NAME}.xml')

# Use the downloaded test image for calibration instead of random noise.
img = cv2.imread('test.jpg')
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

def transform_fn(data_item):
    return img

calibration_dataset = nncf.Dataset(list(range(300)), transform_fn)

quantized = nncf.quantize(
    model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    subset_size=300,
)

ov.save_model(quantized, '${MODEL_NAME}_crowd_int8.xml')
print('Quantization complete: ${MODEL_NAME}_crowd_int8.xml')
"
fi
echo "--- Done ---"
