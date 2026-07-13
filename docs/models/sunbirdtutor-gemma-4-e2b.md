# Sunbird Tutor (Gemma 4 E2B)

## Overview

Sunbird Tutor (Gemma 4 E2B) is a fine-tune of Google's Gemma 4 E2B for spoken question-answering in English and Ugandan languages. Audio goes in through Gemma 4's native audio tower and a text answer comes out directly — there is no separate ASR transcription step, so the whole thing runs as a single forward pass. It powers the Sunflower educational assistant Android app, which runs fully offline so a child in a Ugandan classroom can ask a science question in Luganda and get an answer back in Luganda with no internet connection.

## Task Type

Audio-Text-to-Text (spoken question-answering; also handles transcription, translation, and spoken explanation depending on prompt)

## Base Architecture

Google Gemma 4 E2B (`google/gemma-4-e2b`), fine-tuned by Sunbird AI. Vision was not exercised during fine-tuning — this checkpoint is text plus audio input only, not image input.

## Model Size

5B params (BF16)

## Quantization Format

None (full precision, BF16). For on-device/mobile use, a separate quantized model exists: `Sunbird/sunflower-qa-cactus-int4` (Cactus INT4, ~3.8GB on disk). In that variant the audio tower is kept at FP16 so Luganda speech recognition survives quantization, while the decoder is pushed to INT4 — that quantized variant should get its own documentation page rather than being folded into this one.

## Languages Supported

| ISO 639-3 | Language   | Status                                                           |
| --------- | ---------- | ---------------------------------------------------------------- |
| eng       | English    | Strong                                                           |
| lug       | Luganda    | Strongest non-English. chrF ~0.51 on the project eval.           |
| ach       | Acholi     | Second tier. chrF ~0.40, classroom-usable for shorter responses. |
| nyn       | Runyankole | Transcription and short translation reliable; QA degrades.       |
| xog       | Lusoga     | Transcription and short translation reliable; QA degrades.       |
| nyo       | Lunyoro    | Transcription and short translation reliable; QA degrades.       |
| teo       | Ateso      | Transcription and short translation reliable; QA degrades.       |

Quality scales with training data volume, so Luganda is meaningfully ahead of the other five Ugandan languages. The broader Sunbird Tutor project targets 12 Ugandan languages across multiple checkpoints.

## Intended Use

Primary school science Q&A in Ugandan classrooms. The demo curriculum covers six Primary 5–7 topics: photosynthesis, the water cycle, the life cycle of an insect, malaria prevention, digestion, and the solar system. Beyond Q&A, the model handles speech transcription, translation between supported languages from spoken input, and short spoken explanations. Also useful as a research artifact for adapting multimodal foundation models to low-resource languages.

**Out of scope:** high-stakes domains (medical, legal, financial advice); languages outside the seven listed; image input; long-form generation past a few hundred tokens (drifts from the single-turn QA distribution the fine-tune was optimized for).

## Known Limitations

- Only Luganda reaches the strongest QA quality tier; Acholi is classroom-usable for shorter responses only
- The other four Ugandan languages (Runyankole, Lusoga, Lunyoro, Ateso) are present but full QA degrades outside Luganda/Acholi — transcription and short translation remain reliable for them
- Background-noise robustness has not been formally benchmarked in classroom environments
- Audio inference assumes 16 kHz mono PCM input
- The exact Transformers auto-class for Gemma 4 audio depends on your `transformers` version — if `AutoModelForImageTextToText` doesn't pick up the audio tower, use the loader scripts in the training repo (`SunbirdAI/sunbird-tutor-modelling`) for the correct class names

## Prompt Format

The fine-tune saw a fixed system prompt on every training example, and drift here measurably degrades quality. Use it verbatim:

```
You are an educational assistant that can give explanations, transcriptions and translations in Ugandan languages.
```

User content varies by mode:

| Mode             | User content                                         |
| ---------------- | ---------------------------------------------------- |
| Answer (default) | Empty string — the audio itself carries the question |
| Transcribe       | `"Transcribe this audio."`                           |
| Translate        | `"Translate this audio into {target language}."`     |
| Explain          | `"Explain what was said in this audio."`             |

The canonical runtime strings live in `lib/model_settings_sheet.dart` inside the Sunflower app.

## How to Load / Run

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Sunbird/sunbirdtutor-gemma-4-e2b")
model = AutoModelForImageTextToText.from_pretrained(
    "Sunbird/sunbirdtutor-gemma-4-e2b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            "You are an educational assistant that can give explanations, "
            "transcriptions and translations in Ugandan languages."
        ),
    },
    {
        "role": "user",
        "content": [{"type": "audio", "audio": "path/to/16khz_mono.wav"}],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
answer = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0]
print(answer)
```

For on-device/offline inference (mobile), use the Cactus INT4 quantization instead: `Sunbird/sunflower-qa-cactus-int4` (~3.8GB, fully offline, inference via the Cactus engine).

## Training Procedure

Fine-tuned from `google/gemma-4-e2b` in three stages:

1. **Continued pretraining** on ~600M characters of Ugandan-language text (web articles, books, translations, synthetic instruction-following examples)
2. **Multilingual SFT on transcription** using speech data from SALT, Google's Waxal, FLEURS, and the Makerere speech benchmark
3. **Short final fine-tune on speech QA**, built from the Ugandan primary school curriculum, machine-translated into Ugandan languages and converted to speech using Sunbird's own Orpheus-3B-based TTS model

The audio tower was preserved throughout — the decoder adapts to Ugandan-language QA while keeping Gemma 4's native speech understanding intact. Full configs, training scripts, and per-language eval numbers are in the training repo: `SunbirdAI/sunbird-tutor-modelling`.

## Licence

Gemma license (Google's custom Gemma terms)

## Hugging Face

[huggingface.co/Sunbird/sunbirdtutor-gemma-4-e2b](https://huggingface.co/Sunbird/sunbirdtutor-gemma-4-e2b)

## Last Updated

May 18, 2026

## Citation

```bibtex
@misc{sunbird-tutor-gemma-4-e2b-2026,
  author = {Sunbird AI},
  title  = {Sunbird Tutor: Gemma 4 E2B for spoken question-answering in Ugandan languages},
  year   = {2026},
  url    = {https://huggingface.co/Sunbird/sunbirdtutor-gemma-4-e2b}
}
```
