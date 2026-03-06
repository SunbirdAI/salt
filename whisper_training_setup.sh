#!/bin/bash
set -e

# install required packages
pip install -r requirements.txt
pip install peft
pip install gcsfs
pip install nvidia-ml-py

# install and configure google cloud cli
sudo apt-get update
sudo apt-get install -y ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update
sudo apt-get install -y google-cloud-cli

gcloud init
gcloud auth application-default login