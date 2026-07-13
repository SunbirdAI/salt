# Data Loading with the salt.dataset Module

The `salt.dataset` module provides an abstraction to load and validate parallel text and speech datasets from Hugging Face or custom configurations.

## The `create` Function

The main entry point for loading datasets is `salt.dataset.create(config)`. It takes a dictionary configuration (usually parsed from YAML) and returns a PyTorch/Hugging Face-compatible dataset object.

### Import Usage

```python
import yaml
import salt.dataset

yaml_config = """
huggingface_load:
  path: Sunbird/salt
  split: train
  name: text-all
source:
  type: text
  language: eng
  preprocessing:
      - prefix_target_language
target:
  type: text
  language: [lug, ach]
"""

# Load configuration and create the dataset loader
config = yaml.safe_load(yaml_config)
dataset = salt.dataset.create(config)

# Fetch first few items from dataset
for example in list(dataset.take(3)):
    print(example)
```

---

## Configuration Reference

The configuration dictionary supports the following parameters:

### 1. `huggingface_load` (Required)
A list or a single dictionary defining the dataset paths to load from Hugging Face:
- `path`: The Hugging Face dataset identifier (e.g. `"Sunbird/salt"`).
- `name`: Subfolder or subset configuration (e.g. `"text-all"`, `"multispeaker-lug"`).
- `split`: Train, validation, or test split (e.g. `"train"`).

### 2. `source` & `target` (Required)
Defines how the input (source) and output (target) data should be extracted and preprocessed:
- `type`: Either `"text"` or `"audio"`.
- `language`: ISO 639-3 language code (e.g. `"eng"`, `"lug"`, `"ach"`). Target language can be a list for multi-task setups (e.g. `[lug, ach]`).
- `preprocessing`: A list of preprocessing string keys (e.g., `["clean_text"]` or `["prefix_target_language"]`) that are automatically applied to the loaded records.
