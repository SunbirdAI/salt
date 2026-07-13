# SALT (Sunbird African Language Technology)

SALT is an open-source library and documentation hub developed by [Sunbird AI](https://sunbird.ai). This repository contains the `salt` Python package for loading, preprocessing, training, and evaluating low-resource speech and translation models, as well as the source files for our public documentation.

## 📊 Datasets & Models

We build and release open-source resources for Ugandan and regional East African languages:
- **Datasets**: Multi-way parallel translation and speech corpus (SALT), diagnostic benchmarks (SALT-31), and environmental audio datasets (Urban Noise Uganda 61k).
- **Models**: State-of-the-art multilingual translation models (Sunflower-14B & 32B), speech recognition (Whisper Large v3 SALT), and text-to-speech models (Orpheus TTS).

---

## 📦 Python Library

The `salt` Python library contains modules to facilitate NLP and speech model training:
- **`salt.dataset`**: Abstraction for loading multi-way parallel datasets from Hugging Face or custom configurations.
- **`salt.preprocessing`**: Data normalizers, casing augmentations, and audio noise/downsampling utilities.
- **`salt.utils`**: Wrappers for sequence-to-sequence model architectures and custom Beginning-Of-Sequence (BOS) logits processors.
- **`salt.metrics`**: Automated evaluation functions to compute translation quality scores (SacreBLEU, etc.).

### Installation

Install the library locally in editable mode:
```bash
git clone https://github.com/SunbirdAI/salt.git
cd salt
pip install -e .
```

### Quick Example: Data Loading

```python
import yaml
import salt.dataset

config = yaml.safe_load("""
huggingface_load:
  path: Sunbird/salt
  name: text-all
  split: train
source:
  type: text
  language: eng
  preprocessing: [prefix_target_language]
target:
  type: text
  language: [lug, ach]
""")

dataset = salt.dataset.create(config)
for example in dataset.take(2):
    print(example)
```

---

## 📖 Documentation Site

This repository builds the official documentation site for SALT.

### Serving Locally
To run the documentation server locally for development:
```bash
uv run mkdocs serve
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

### Deploying Docs
To build and deploy the documentation site, execute:
```bash
./build_and_deploy_docs.sh
```

 
