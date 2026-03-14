#!/bin/bash
set -e

# install required packages
pip install -r requirements.txt
pip install peft
pip install gcsfs
pip install nvidia-ml-py
