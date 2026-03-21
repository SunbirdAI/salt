#!/bin/bash
set -e

# # install ffmpeg
# apt-get install -y software-properties-common
# add-apt-repository ppa:ubuntuhandbook1/ffmpeg6 -y
# apt-get update
# apt-get install -y --allow-downgrades ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswresample-dev

# install required packages
pip install -r requirements.txt
pip install peft
pip install gcsfs
pip install nvidia-ml-py

# # intall torchcodec
# pip uninstall torchcodec -y
# pip install torchcodec==0.2 --index-url=https://download.pytorch.org/whl/cu118
# python -c "import torchcodec; print('torchcodec loaded successfully')"

