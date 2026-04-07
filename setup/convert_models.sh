#!/bin/bash
### Convert marsupial .pt model to TensorRT engine ###
# Reference: https://www.youtube.com/watch?v=ErWC3nBuV6k
# This script converts the marsupial16s.pt model to a TensorRT .engine file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Check CUDA is in PATH
if ! command -v nvcc &> /dev/null; then
    echo "[-] nvcc not found. Setting up CUDA paths..."
    export PATH=/usr/local/cuda/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
fi

# Check if .pt file exists
if [ ! -f "$MODELS_DIR/marsupial16s.pt" ]; then
    echo "[-] marsupial16s.pt not found in $MODELS_DIR"
    echo "[-] Download it from https://github.com/carlosclaiton/marsupial and place it in models/"
    exit 1
fi

# Clone YOLOv5 v6.2 if not present
if [ ! -d "$PROJECT_DIR/yolov5" ]; then
    echo "[+] Cloning YOLOv5 v6.2..."
    cd "$PROJECT_DIR"
    git clone --branch v6.2 https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip3 install onnx==1.9.0
else
    cd "$PROJECT_DIR/yolov5"
fi

# Step 1: Export .pt to ONNX
echo "[+] Exporting marsupial16s.pt to ONNX..."
python3 export.py --weights "$MODELS_DIR/marsupial16s.pt" --include onnx --img 640

# Check ONNX was created
if [ ! -f "$MODELS_DIR/marsupial16s.onnx" ]; then
    # YOLOv5 export puts onnx next to the .pt file
    if [ -f "$PROJECT_DIR/yolov5/marsupial16s.onnx" ]; then
        mv "$PROJECT_DIR/yolov5/marsupial16s.onnx" "$MODELS_DIR/"
    else
        echo "[-] ONNX export failed"
        exit 1
    fi
fi

# Step 2: Convert ONNX to TensorRT engine
echo "[+] Converting ONNX to TensorRT engine (this will take a while)..."
/usr/src/tensorrt/bin/trtexec \
    --onnx="$MODELS_DIR/marsupial16s.onnx" \
    --saveEngine="$MODELS_DIR/marsupial16s.engine" \
    --fp16

if [ -f "$MODELS_DIR/marsupial16s.engine" ]; then
    echo "[+] Conversion complete! Engine saved to $MODELS_DIR/marsupial16s.engine"
else
    echo "[-] TensorRT conversion failed"
    exit 1
fi