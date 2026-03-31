#!/bin/bash
# SmartGate Model Conversion Script for TRT7 (JetPack 4.5-4.6)
# Run this on the Jetson Nano after cloning the repo

set -e

HOME_DIR="/home/$(whoami)"
SMARTGATE_DIR="$HOME_DIR/SmartGate"
YOLOV5_DIR="$HOME_DIR/yolov5"
MARSUPIAL_DIR="$HOME_DIR/marsupial"

echo "[1/5] Cloning yolov5 v6.2..."
git clone --branch v6.2 https://github.com/ultralytics/yolov5.git $YOLOV5_DIR

echo "[2/5] Cloning marsupial weights..."
git clone https://github.com/carlosclaiton/marsupial.git $MARSUPIAL_DIR

echo "[3/5] Installing export dependencies..."
pip3 install seaborn==0.11.2
pip3 install onnx==1.9.0

echo "[4/5] Exporting marsupial16s to ONNX..."
cd $YOLOV5_DIR
python3 export.py \
  --weights $MARSUPIAL_DIR/weights/marsupial_16s.pt \
  --include onnx \
  --opset 11

echo "[5/5] Converting to TensorRT FP16 engine..."
/usr/src/tensorrt/bin/trtexec \
  --onnx=$MARSUPIAL_DIR/weights/marsupial_16s.onnx \
  --saveEngine=$SMARTGATE_DIR/models/marsupial16s_trt7_fp16.engine \
  --explicitBatch \
  --fp16

echo "Conversion complete! Engine saved to models/marsupial16s_trt7_fp16.engine"