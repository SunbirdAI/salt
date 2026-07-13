# asr-whisper-large-v3-salt

## Overview

`asr-whisper-large-v3-salt` is an automatic speech recognition model adapted from Whisper Large v3 for Ugandan and regional languages. It is designed to transcribe speech in Luganda, Acholi, Lugbara, Ateso, Runyankole, Rutooro, Lumasaba, Swahili, Lusoga, Kinyarwanda, and Ugandan English.

The model extends Whisper for languages that are not fully supported by the base Whisper model, using custom SALT language tokens and additional training data from Ugandan and East African speech datasets.

## Task Type

Automatic Speech Recognition (ASR), Audio-to-Text.

## Base Architecture

Whisper Large v3.

Base model listed on Hugging Face: `jq/whisper-large-v3-salt-plus-xog-myx-kin-swa`.

## Model Size

Whisper Large v3 model size. Verify the exact parameter count from the Hugging Face repository before publishing if required.

## Quantization Format

None. This is not listed as a quantized model.

## Languages Supported

The model card describes support for the following languages:

| Language                | Code  |
| ----------------------- | ----- |
| English, Ugandan accent | `eng` |
| Swahili                 | `swa` |
| Acholi                  | `ach` |
| Lugbara                 | `lgg` |
| Luganda                 | `lug` |
| Runyankole              | `nyn` |
| Ateso                   | `teo` |
| Lusoga                  | `xog` |
| Rutooro                 | `ttj` |
| Kinyarwanda             | `kin` |
| Lumasaba                | `myx` |

The Hugging Face metadata lists `lg`, `en`, `nyn`, `ach`, `teo`, and `lgg`; the usage example uses the broader SALT language-token mapping above.

## Intended Use

This model is intended for transcribing speech in Ugandan and regional languages, especially for applications where the base Whisper model does not provide strong support. Suitable use cases include:

- Speech-to-text transcription for Ugandan languages.
- Transcription of Ugandan-accented English.
- ASR research for low-resource African languages.
- Data processing pipelines for speech datasets.
- Voice interfaces and speech-enabled applications.
- Transcription of phone-quality or noisy speech, subject to quality review.

## Known Limitations

- Language detection is not always accurate. Results may improve when the target language is specified explicitly.
- Accuracy varies by language, with stronger reported performance for English, Swahili, Kinyarwanda, and Luganda than for some lower-resource languages.
- Noisy audio, overlapping speakers, strong accents, poor microphones, or very low-quality recordings may reduce transcription quality.
- The model may make spelling, segmentation, or word substitution errors, especially for underrepresented languages.
- Human review is recommended for official, legal, medical, financial, or high-stakes transcriptions.

## How to Load / Run

Install the required dependencies:

```bash
pip install transformers datasets torch
```

Example usage:

```python
import torch
import transformers
import datasets

MODEL_PATH = "Sunbird/asr-whisper-large-v3-salt"

processor = transformers.WhisperProcessor.from_pretrained(MODEL_PATH)
model = transformers.WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

SALT_LANGUAGE_TOKENS_WHISPER = {
    "eng": 50259,  # English (Ugandan)
    "swa": 50318,  # Swahili
    "ach": 50357,  # Acholi
    "lgg": 50356,  # Lugbara
    "lug": 50355,  # Luganda
    "nyn": 50354,  # Runyankole
    "teo": 50353,  # Ateso
    "xog": 50352,  # Lusoga
    "ttj": 50351,  # Rutooro
    "kin": 50350,  # Kinyarwanda
    "myx": 50349,  # Lumasaba
}

# Load test audio from the SALT dataset.
ds = datasets.load_dataset("Sunbird/salt", "multispeaker-lug", split="test")
audio = ds[0]["audio"]
sample_rate = ds[0]["sample_rate"]

# Specify the target language.
# You can set language=None in generate() to let the model attempt language detection.
lang = "lug"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_features = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
).input_features.to(device)

predicted_ids = model.generate(
    input_features,
    language=processor.tokenizer.decode(SALT_LANGUAGE_TOKENS_WHISPER[lang]),
    forced_decoder_ids=None,
)

transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True,
)

print(transcription)
```

Example output:

```text
Ekikoola kya kasooli kya kyenvu wabula langi yaakyo etera okuba eya kitaka wansi.
```

## Training Data

The model was trained using speech data from:

- SALT dataset.
- Common Voice for Luganda, Swahili, and Kinyarwanda.
- Google FLEURS.
- Makerere Yogera dataset.

To improve generalization in practical settings, training included random noise augmentation and random downsampling to 8kHz to simulate phone speech. Street noise sampled from urban locations in Uganda was also added to improve robustness.

## Evaluation

The model was evaluated on SALT text and held-out splits from Common Voice for Swahili and Kinyarwanda, and Yogera for Rutooro and Lusoga.

### Word Error Rate

| Language    |   WER |
| ----------- | ----: |
| English     | 0.018 |
| Luganda     | 0.142 |
| Acholi      | 0.195 |
| Lugbara     | 0.189 |
| Ateso       | 0.202 |
| Runyankole  | 0.234 |
| Lumasaba    | 0.461 |
| Lusoga      | 0.453 |
| Swahili     | 0.069 |
| Kinyarwanda | 0.111 |
| Mean        | 0.207 |

### Character Error Rate

| Language    |   CER |
| ----------- | ----: |
| English     | 0.009 |
| Luganda     | 0.029 |
| Acholi      | 0.045 |
| Lugbara     | 0.045 |
| Ateso       | 0.051 |
| Runyankole  | 0.043 |
| Lumasaba    | 0.092 |
| Lusoga      | 0.081 |
| Swahili     | 0.015 |
| Kinyarwanda | 0.031 |
| Mean        | 0.044 |

## Licence

MIT.

## Hugging Face

[https://huggingface.co/Sunbird/asr-whisper-large-v3-salt](https://huggingface.co/Sunbird/asr-whisper-large-v3-salt)

## Last Updated

October 8, 2025

## Citation

```bibtex
@misc{sunbird_asr_whisper_large_v3_salt,
  title={{asr-whisper-large-v3-salt}},
  author={{Sunbird AI}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\url{https://huggingface.co/Sunbird/asr-whisper-large-v3-salt}}}
}
```

## References

- Base model: `jq/whisper-large-v3-salt-plus-xog-myx-kin-swa`
- Dataset: `Sunbird/salt`
- Architecture family: Whisper Large v3
