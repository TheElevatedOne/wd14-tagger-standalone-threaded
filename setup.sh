#!/usr/bin/bash
# Creates ./venv with the dependencies and the onnxruntime build for your GPU.
# Uses uv if it's installed, otherwise python -m venv and pip.
# Usage: ./setup.sh [auto|amd|nvidia|cpu]    (default: auto)
set -euo pipefail

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
VENV="$SCRIPT_DIR/venv"
PYTHON=${PYTHON:-python3}
BACKEND=${1:-auto}

if command -v uv &> /dev/null; then
    USE_UV=true
else
    USE_UV=false
fi

UV_PYTHON=()
if command -v "$PYTHON" &> /dev/null; then
    # The exact path, because uv can otherwise pick a Python it manages itself,
    # and that one doesn't see the system packages.
    UV_PYTHON=(--python "$("$PYTHON" -c 'import sys; print(sys.executable)')")
elif [[ $USE_UV == false ]]; then
    echo "Python not found. Install Python 3.11 or newer with your package manager." >&2
    exit 1
fi

gpu_vendor_present() {
    grep -qs "$1" /sys/class/drm/card*/device/vendor
}

if [[ $BACKEND == auto ]]; then
    if gpu_vendor_present 0x10de; then
        BACKEND=nvidia
    elif gpu_vendor_present 0x1002; then
        BACKEND=amd
    else
        BACKEND=cpu
    fi
    echo "Detected backend: $BACKEND"
fi

# Distro packages such as Arch's python-onnxruntime-rocm are built for the installed ROCm.
system_has_amd_runtime() {
    "$PYTHON" -c "
import sys, onnxruntime
amd = {'MIGraphXExecutionProvider', 'ROCMExecutionProvider'}
sys.exit(0 if amd & set(onnxruntime.get_available_providers()) else 1)" 2> /dev/null
}

migraphx_installed() {
    ldconfig -p 2> /dev/null | grep -q libmigraphx_c || compgen -G '/opt/rocm/lib/libmigraphx_c.so*' > /dev/null
}

VENV_ARGS=()
RUNTIME=
REQUIREMENTS=$SCRIPT_DIR/requirements.txt
case $BACKEND in
    nvidia)
        REQUIREMENTS=$SCRIPT_DIR/requirements-nvidia.txt
        ;;
    cpu)
        RUNTIME=onnxruntime
        ;;
    amd)
        if system_has_amd_runtime; then
            echo "Using the onnxruntime with AMD support from your system Python"
            VENV_ARGS=(--system-site-packages)
        elif migraphx_installed; then
            REQUIREMENTS=$SCRIPT_DIR/requirements-amd.txt
        else
            cat >&2 << 'MESSAGE'
AMD GPU support needs ROCm and MIGraphX, and neither onnxruntime nor libmigraphx_c was found.
  Arch:   sudo pacman -S python-onnxruntime-rocm
  Others: https://rocm.docs.amd.com/projects/install-on-linux/
Then run this script again, or run "./setup.sh cpu" to use the CPU.
MESSAGE
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [auto|amd|nvidia|cpu]" >&2
        exit 2
        ;;
esac

# The onnxruntime packages share one module and can't be installed side by side.
OLD_RUNTIMES=(onnxruntime onnxruntime-gpu onnxruntime-migraphx onnxruntime-rocm)

echo "Creating Python Virtual Environment"
if [[ $USE_UV == true ]]; then
    echo "Using uv"
    uv venv --allow-existing "${UV_PYTHON[@]}" "${VENV_ARGS[@]}" "$VENV"
    pip_uninstall() { uv pip uninstall --python "$VENV/bin/python" "$@"; }
    pip_install() { uv pip install --python "$VENV/bin/python" "$@"; }
else
    echo "uv not found, using python -m venv and pip"
    "$PYTHON" -m venv "${VENV_ARGS[@]}" "$VENV"
    pip_uninstall() { "$VENV/bin/pip" uninstall -y "$@"; }
    pip_install() { "$VENV/bin/pip" install "$@"; }
fi

pip_uninstall "${OLD_RUNTIMES[@]}" &> /dev/null || true

echo "Installing requirements"
pip_install -r "$REQUIREMENTS"
if [[ -n $RUNTIME ]]; then
    echo "Installing $RUNTIME"
    pip_install "$RUNTIME"
fi

if ! "$VENV/bin/python" -c 'import onnxruntime' 2> /dev/null; then
    echo "onnxruntime can't be imported from $VENV." >&2
    exit 1
fi

echo "Installation finished"

# Only ask when someone is there to answer.
if [[ -t 0 && -t 1 ]]; then
    echo
    "$VENV/bin/python" "$SCRIPT_DIR/download.py" || echo "Some models were not downloaded. Run $VENV/bin/python $SCRIPT_DIR/download.py to try again."
else
    echo "To download models, run: $VENV/bin/python $SCRIPT_DIR/download.py"
fi
