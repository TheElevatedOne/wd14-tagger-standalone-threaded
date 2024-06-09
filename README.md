# Multithreaded WD14 Tagging Script with GPU support
---
Forked from [https://github.com/corkborg/wd14-tagger-standalone](https://github.com/corkborg/wd14-tagger-standalone)
---
## Navigation
- [Install](#install)
- [Usage](#usage)
- [Single file](#single-file)
- [Batch execution](#batch-execution)
- [Multithreading](#multithreading)
- [For fast use](#for-fast-use)
- [Supported Models](#supported-models)
- [Todo](#todo)
- [Copyright](#copyright)
---
## Install
- Preparation:
  - Install Nvidia CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
  - Install Nvidia cuDNN from https://developer.nvidia.com/cudnn-downloads
  - Set environment variable CUDA_PATH to your CUDA installation ex.
  - ```echo 'export CUDA_PATH=/usr/local/cuda-12.4' >> ~/.bashrc```

- Easier option:
```
chmod +x setup.sh
./setup.sh
```
- Manual option:
```
# Install dependencies
sudo apt install -y python3 python3-pip python3-venv

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For CUDA 12
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# For CUDA 11 - pip install onnxruntime-gpu
```
## Usage

```
chmod +x wd14-tagger.sh
> ./wd14-tagger.sh --help
usage: threaded-main.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD] [--ext EXT] [--overwrite] [--cpu] [--rawtag]
                        [--model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,wd-v1-4-vit-tagger.v3,wd-v1-4-convnext-tagger.v3,wd-v1-4-swinv2-tagger.v3,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}]
                        [--threads THREADS]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only
  --rawtag              Use the raw output of the model
  --model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,wd-v1-4-vit-tagger.v3,wd-v1-4-convnext-tagger.v3,wd-v1-4-swinv2-tagger.v3,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}
                        Modelname to use for prediction (default is wd14-convnextv2.v1)
  --threads THREADS     Ppecify the number of threads you want to run it with (multithreading)
```
The `main.py` file is without multithreading, keeping it just cause.

### Single file

```
./wd14-tagger.sh --file image.jpg
```

### Batch execution

```
./wd14-tagger.sh --dir dir/dir
```

### Multithreading
```
./wd14-tagger.sh --threads 2 --dir dir/dir
```
Multithreading works only with batch tagging
With threads set at 2, the program will split the list of files into 2 lists (as the number of threads)
```
>>> import os
>>> files = os.listdir()
>>> files
['file1','file2',...]
>>> chunks(files, 2) # function uses the original list and the number of threads
[['file1', 'file2',...], ['file99','file100',...]]
```
Then the tagging runs on the two lists separately
The sideeffect (which don't really want to "fix") is that the tagger loads the model to the VRAM twice
The model normally uses around 1.5GB of VRAM, so for two threads it will use around 3GB

## For fast use
- Option 1 - Use alias for the executable ex. `echo 'alias wd14-tagger=<path>/wd14-tagger-standalone-threaded/wd14-tagger.sh' >> ~/.bashrc`
- Option 2 - Export the whole directory to path ex. `echo 'export PATH=<path>/wd14-tagger-standalone-threaded:$PATH' >> ~/.bashrc`

## Supported Models

```
./wd14-tagger.sh --file image.jpg --model wd14-vit.v1
./wd14-tagger.sh --file image.jpg --model wd14-vit.v2
./wd14-tagger.sh --file image.jpg --model wd14-convnext.v1
./wd14-tagger.sh --file image.jpg --model wd14-convnext.v2
./wd14-tagger.sh --file image.jpg --model wd14-convnextv2.v1
./wd14-tagger.sh --file image.jpg --model wd14-swinv2-v1
./wd14-tagger.sh --file image.jpg --model wd-v1-4-moat-tagger.v2
./wd14-tagger.sh --file image.jpg --model wd-v1-4-vit-tagger.v3
./wd14-tagger.sh --file image.jpg --model wd-v1-4-convnext-tagger.v3
./wd14-tagger.sh --file image.jpg --model wd-v1-4-swinv2-tagger.v3
./wd14-tagger.sh --file image.jpg --model mld-caformer.dec-5-97527
./wd14-tagger.sh --file image.jpg --model mld-tresnetd.6-30000
```

## TODO
- [x] Fork
- [x] Add multithreading
- [ ] Add "support" for windows and Install info
- [ ] Add option to add custom tags
- [ ] Add option to remove specified tags
- [ ] Add option to prune the least used tags 

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)
