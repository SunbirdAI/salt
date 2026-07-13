# SALT Documentation

Welcome to the official documentation for the **SALT** project, part of the [Sunbird AI Language Projects](https://sunbird.ai/portfolio/african-languages/). 

This documentation covers our datasets, open-source models, and the SALT Python library designed to support speech and language technology for African languages.

## Datasets
We build and curate high-quality datasets to advance research and application of machine learning for low-resource African languages:
- **[SALT](datasets/salt.md)**: A multi-way parallel text and speech corpus covering English and six widely spoken languages in Uganda and East Africa.
- **[SALT-31](datasets/salt-31.md)**: A context-aware Machine Translation evaluation benchmark covering 31 Ugandan and regional languages.
- **[Urban Noise Uganda 61k](datasets/urban-noise-uganda-61k.md)**: A dataset for urban environmental acoustic monitoring in Uganda.

## Models
We release highly-optimized models trained for translation, speech recognition, and synthesis:
- **[Sunflower-14B](models/sunflower-14b.md) & [Sunflower-32B](models/sunflower-32b.md)**: Our flagship multilingual language models for Ugandan languages and English, including various quantized formats (FP8, W8A8, FP4A16, GGUF).
- **[Whisper Large v3 SALT](models/asr-whisper-large-v3-salt.md)**: An Automatic Speech Recognition (ASR) model fine-tuned on Ugandan languages.
- **[Orpheus 3B TTS Multilingual](models/orpheus-3b-tts-multilingual.md)**: A Text-to-Speech (TTS) model supporting voice generation across Ugandan languages.
- **[SunbirdTutor Gemma 4 E2B](models/sunbirdtutor-gemma-4-e2b.md)**: A specialized educational model.

## 📦 SALT Python Package
The `salt` Python package provides helper utilities and pipelines for convenient experimentation, training, and deployment:
- **Getting Started**: Read the [Overview & Installation](tutorials/overview.md) guide.
- **Developer Guides**: Master [Data Loading](tutorials/data-loading.md), [Data Preprocessing](tutorials/preprocessing.md), [Model Training](tutorials/training.md), and [Evaluation Metrics](tutorials/evaluation.md).
- **Core Pipelines**: See guides for [Translation Models](tutorials/08-translation-models.md), [ASR Models](tutorials/09-asr-models.md), [TTS Models](tutorials/10-tts-spark-models.md), and [Speaker Diarization](tutorials/13-diarization.md).
