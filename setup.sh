#!/usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! command -v python3 &> /dev/null
then
    echo "Installing Python"
    sudo apt install python3 python3-pip python3-venv
fi

echo "Creating Python Virtual Environment"
python3 -m venv $SCRIPT_DIR/venv

echo "Installing requirements"
$SCRIPT_DIR/venv/bin/pip install -r $SCRIPT_DIR/requirements.txt
echo "Installing ONNXRuntime for CUDA 12"
$SCRIPT_DIR/venv/bin/pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo "Installation finished"
