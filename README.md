# Multithreaded WD14 Tagging Script with GPU support

![Static Badge](https://img.shields.io/badge/CLAUDE-VIBECODED?style=for-the-badge&logo=claude&label=VIBECODED&labelColor=%23333333&color=%23de7356)

Tags folders of anime-style images with Danbooru-style tags and writes a caption file next to each image, using ONNX models on an AMD or NVIDIA GPU or the CPU.

Forked from [corkborg/wd14-tagger-standalone](https://github.com/corkborg/wd14-tagger-standalone).

## Navigation

- [Install](#install)
- [Usage](#usage)
- [Options](#options)
- [GPU support](#gpu-support)
- [Supported Models](#supported-models)
- [For fast use](#for-fast-use)
- [Todo](#todo)
- [Copyright](#copyright)

## Install

Requires Python 3.11 or newer on Linux.

```sh
./setup.sh
```

`setup.sh` creates `venv/` with [uv][uv], or with `python -m venv` and pip if uv isn't installed. It finds your GPU in `/sys/class/drm` and installs the matching onnxruntime. To pick one yourself, run `./setup.sh amd`, `./setup.sh nvidia` or `./setup.sh cpu`. See [GPU support](#gpu-support) for what each one needs.

To install by hand, use the requirements file for your GPU:

| GPU | File |
| --- | --- |
| NVIDIA | `requirements-nvidia.txt` |
| AMD | `requirements-amd.txt` |

```sh
uv venv venv
uv pip install --python venv/bin/python -r requirements-nvidia.txt
```

`requirements.txt` has everything except onnxruntime, so it's the one to use with a distro package such as `python-onnxruntime-rocm`.

### Downloading models

At the end of `setup.sh`, you get a list of the models with their size, tag count, year and whether you already have them. Type the ones you want, such as `1 3 5-7`, a model name, `all` or `missing`, or press <kbd>Enter</kbd> to skip. It shows the total size and your free space before it starts.

To open the list again, or download without being asked:

```sh
venv/bin/python download.py
venv/bin/python download.py wd-convnext-v3 camie-tagger-v2
venv/bin/python download.py --list
```

Models go to the Hugging Face cache, `~/.cache/huggingface/hub` by default. You don't have to download them first, since `wd14-tagger.sh` fetches a missing model when it needs it.

## Usage

Tag a directory, writing `image.txt` next to `image.png`. Images that already have a caption are skipped:

```sh
./wd14-tagger.sh dir/dir
```

Print the tags of one image:

```sh
./wd14-tagger.sh image.jpg
```

Tag a dataset with subdirectories, add a trigger tag, drop some tags and replace the old captions:

```sh
./wd14-tagger.sh dir/dir --recursive --prepend "my_trigger" --exclude "watermark, signature" --overwrite
```

Run two workers:

```sh
./wd14-tagger.sh dir/dir --threads 2
```

Each worker takes the next image when it's free, and each one loads its own copy of the model into VRAM. On a GPU, more workers add little: 400 images took 32 s with one worker and 27 s with two on an RX 6750 XT.

The exit code is 1 if any image failed. Failed images are listed and don't stop the run.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `path` | | Directory of images, or a single image. Reads `.png`, `.jpg`, `.jpeg` and `.webp` |
| `--threshold` | `0.35` | Prediction threshold from 0.0 to 1.0. `pixai-tagger-v0.9` defaults to 0.3 for general and 0.85 for character tags |
| `--general-threshold` | | Threshold for non-character tags, overrides `--threshold` |
| `--character-threshold` | | Threshold for character tags, overrides `--threshold` |
| `-l`, `--limit` | `100` | Maximum tags per image from the model, `0` for no limit |
| `--ext` | `.txt` | Caption extension, `.txt` or `.caption` |
| `--json` | | Output tags and ratings with confidences as JSON, in `.json` files for a directory |
| `-w`, `--overwrite` | | Replace existing captions. The old ones go to the trash |
| `-r`, `--recursive` | | Also tag images in subdirectories |
| `-tp`, `--prepend` | | Tags to add before the model tags, as `"tag1, tag2"` |
| `-ta`, `--append` | | Tags to add after the model tags, as `"tag1, tag2"` |
| `-e`, `--exclude` | | Tags to remove from the model tags, as `"tag1, tag2"` |
| `--rawtag` | | Keep the model's spelling, with underscores and unescaped brackets |
| `--model` | `wd-convnext-v3` | See [Supported Models](#supported-models) |
| `--provider` | `auto` | `auto`, `cuda`, `migraphx`, `rocm`, `directml`, `coreml`, `openvino` or `cpu` |
| `--cpu` | | Same as `--provider cpu` |
| `--device-id` | `0` | Index of the GPU to use |
| `-t`, `--threads` | `1` | Number of worker processes |

## GPU support

`--provider auto` uses the first provider that starts, in the order of the table below, and prints which one it picked as `[device]`. A provider you name yourself fails with an error instead of falling back to the CPU.

| GPU | Provider | Package |
| --- | --- | --- |
| NVIDIA | `cuda` | `onnxruntime-gpu` from `requirements-nvidia.txt`, needs [CUDA and cuDNN][cuda] |
| AMD on Linux | `migraphx` | `python-onnxruntime-rocm` on Arch, `onnxruntime-migraphx` from `requirements-amd.txt` elsewhere, needs [ROCm][rocm] |
| AMD on Linux, onnxruntime before 1.23 | `rocm` | `onnxruntime-rocm`. onnxruntime removed this provider in 1.23 |
| Any on Windows | `directml` | `onnxruntime-directml`. `setup.sh` doesn't cover Windows, install it by hand |
| Apple | `coreml` | `onnxruntime` |
| Intel | `openvino` | `onnxruntime-openvino` |

### AMD

On Arch, install the distro package before running `setup.sh`. The venv then uses it through `--system-site-packages`, so it always matches your ROCm version:

```sh
sudo pacman -S python-onnxruntime-rocm
./setup.sh amd
```

Your user has to be in the `render` and `video` groups.

- **Cards without official ROCm support:** if the provider fails to start on an RX 6000 card that `rocminfo` lists as something other than `gfx1030`, try `HSA_OVERRIDE_GFX_VERSION=10.3.0 ./wd14-tagger.sh ...`. An RX 6750 XT (`gfx1031`) with ROCm 7.2 runs without it.
- **Slow first run:** MIGraphX compiles a model the first time you use it, which takes about 2 minutes for `wd-convnext-v3` on an RX 6750 XT. The result is kept in `~/.cache/wd14-tagger/migraphx` (about 440 MB per model), so later runs start in seconds. Set `ORT_MIGRAPHX_MODEL_CACHE_PATH` to use another directory.
- **ML-Danbooru models:** `mld-*` keeps the aspect ratio of each image, and MIGraphX recompiles when the input size changes. Prefer the other models on AMD.

## Supported Models

| Family | `--model` |
| --- | --- |
| PixAI (2025, general and character thresholds, adds series tags) | `pixai-tagger-v0.9` |
| Camie (2025) | `camie-tagger`, `camie-tagger-v2` |
| SmilingWolf large v3 (2024) | `wd-vit-large-tagger-v3`, `wd-eva02-large-tagger-v3` |
| SmilingWolf v3 (2024) | `wd-v1-4-vit-tagger.v3`, `wd-v1-4-convnext-tagger.v3` (same as `wd-convnext-v3`), `wd-v1-4-swinv2-tagger.v3` |
| SmilingWolf v2 (2023) | `wd-v1-4-moat-tagger.v2`, `wd14-vit.v2`, `wd14-convnext.v2` |
| SmilingWolf v1 (2022) | `wd14-vit.v1`, `wd14-convnext.v1`, `wd14-convnextv2.v1`, `wd14-swinv2-v1` |
| Z3D e621 | `z3d-e621-convnext-toynya`, `z3d-e621-convnext-silveroxides` |
| ML-Danbooru | `mld-caformer.dec-5-97527`, `mld-tresnetd.6-30000` |

## For fast use

- Option 1: use an alias for the executable, `echo 'alias wd14-tagger=<path>/wd14-tagger-standalone-threaded/wd14-tagger.sh' >> ~/.bashrc`
- Option 2: add the directory to your path, `echo 'export PATH=<path>/wd14-tagger-standalone-threaded:$PATH' >> ~/.bashrc`

The old version without multithreading is kept in `.old-files/main.py`.

## TODO

- [x] Fork
- [x] Add multithreading
- [x] Add option to add custom tags
- [x] Add option to remove specified tags
- [x] Add AMD GPU support
- [ ] Add "support" for windows and Install info
- [ ] Add option to prune the least used tags

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)

[uv]: https://docs.astral.sh/uv/
[cuda]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
[rocm]: https://rocm.docs.amd.com/projects/install-on-linux/
