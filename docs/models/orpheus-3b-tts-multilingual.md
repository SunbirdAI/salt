# Orpheus-3B Sunbird Multilingual TTS

## Overview

Orpheus-3B Sunbird Multilingual TTS is a multilingual, multi-speaker text-to-speech model fine-tuned from `unsloth/orpheus-3b-0.1-pretrained` on the full `Sunbird/tts` corpus, covering 20 language configurations and every speaker present in the dataset. The model accepts arbitrary text and emits 24 kHz mono speech via the SNAC audio codec. Voice selection happens at the prompt level: prepend a chosen `speaker_id` followed by `": "` to the input text, and the model produces audio in that speaker's voice.

## Task Type

Text-to-Speech (TTS)

## Base Architecture

Llama-3 architecture (via `unsloth/orpheus-3b-0.1-pretrained`), paired with the `hubertsiuzdak/snac_24khz` audio codec (24 kHz, 7 codes per ~12 ms frame) for waveform decoding.

## Model Size

3B parameters

## Quantization Format

None (full precision). Weights are bfloat16, with a 16-bit merged LoRA adapter (see Training Details below).

## Languages Supported

20 languages/configs, spanning multiple African language families. Speaker IDs encode both source corpus (`salt_*`, `waxal_*`, `slr32_*`, `slr129_*`, `bateesa_*`) and language:

| Config | Language | ISO 639-1 | Region |
|---|---|---|---|
| ach | Acholi | — | Uganda, South Sudan |
| afr | Afrikaans | af | South Africa, Namibia |
| eng | English | en | (control language) |
| ewe | Ewe | ee | Ghana, Togo |
| ful | Fulah | ff | West Africa (Sahel) |
| hau | Hausa | ha | Nigeria, Niger, Chad |
| ibo | Igbo | ig | Nigeria |
| kik | Kikuyu | ki | Kenya |
| kin | Kinyarwanda | rw | Rwanda |
| lgg | Lugbara | — | Uganda, DRC |
| lin | Lingala | ln | DRC, Republic of Congo |
| lug | Luganda | lg | Uganda |
| luo | Luo (Dholuo) | — | Kenya, Tanzania |
| nyn | Runyankole | — | Uganda |
| sot | Sesotho | st | Lesotho, South Africa |
| swa | Swahili | sw | East Africa |
| teo | Ateso | — | Uganda, Kenya |
| tsn | Setswana | tn | Botswana, South Africa |
| xho | Xhosa | xh | South Africa |
| yor | Yoruba | yo | Nigeria, Benin |

Some configs (`lgg`, `sot`, `tsn`) are present in the training mix but don't currently expose individual voice IDs in this checkpoint. Speaker IDs can be enumerated directly from the `Sunbird/tts` dataset (see the model card's discovery snippet).

## Intended Use

**Intended for:**
- Multilingual voice synthesis for accessibility, language learning, human–computer interaction, audio content creation, and downstream speech research across the 20 covered languages
- Reference checkpoint for the `Sunbird/tts` → Orpheus-3B multilingual fine-tuning pipeline (reproducible recipe in `Orpheus_3B_Sunbird_Multilingual.ipynb`)

**Out of scope:**
- Voice impersonation or deception of identifiable real persons
- High-stakes / safety-critical use (medical, legal, emergency) without human review
- Languages outside the 20 covered configs
- Code-switching within a single prompt
- Cross-language voice transfer (e.g. sending Acholi text to a Luganda `speaker_id`) — language identity travels via the speaker tag, so text language must match the chosen speaker's language

## Known Limitations

- **Quality varies by language** — the training corpus is unbalanced across the 20 configs; languages with more speaker hours produce more natural speech. Audition per-language samples before relying on a voice in production.
- **No language conditioning** — there is no separate language token; mismatching `speaker_id` and text language is undefined behaviour.
- **Vocabulary coverage** limited to each config's training subset; unfamiliar words, code-switching, and out-of-distribution proper nouns may cause artefacts.
- **Long utterances** may degrade or truncate beyond ~10 seconds of speech (trained on utterances up to ~16s, `max_seq_length=4096`).
- **Sampling variance** — with `do_sample=True`, identical prompts can produce different deliveries between runs; pass `seed=` for reproducibility.
- **No emotion/style control** — unlike the upstream `orpheus-3b-0.1-ft`, this fine-tune was not exposed to in-text emotion tags (e.g. `<laugh>`, `<sigh>`); such tags are tokenized as ordinary text with no special effect.
- Inherits any biases present in the `Sunbird/tts` corpus and Llama-3's pretraining; not systematically audited per language.
- No automated quality metrics (WER, MOS, language-confusion eval) have been run for this release — evaluation was qualitative, on a held-out test sample.

## Hardware Requirements

| Mode | Min VRAM | Recommended |
|---|---|---|
| transformers + Unsloth, fp16 | 8 GB (with `load_in_4bit=True`) | 16 GB |
| transformers + Unsloth, bf16 | 14 GB | 24 GB |
| vLLM, bf16, `max_model_len=4096` | 14 GB | 24 GB |

Audio decoding via SNAC runs on CPU and adds ~50–150 ms per utterance.

## How to Load / Run

Two inference paths are supported: **transformers + Unsloth** (best for development/notebooks) and **vLLM** (best for high-throughput serving, batched requests, deployment).

### Option A — transformers + Unsloth

```bash
pip install unsloth snac soundfile torchcodec "datasets>=3.4.1,<4.0.0"
```

```python
import os
import numpy as np
import torch
import soundfile as sf
from unsloth import FastLanguageModel
from snac import SNAC

MODEL_ID = "sunbird/orpheus-3b-tts-multilingual"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
    token=os.environ.get("HF_TOKEN"),
)
FastLanguageModel.for_inference(model)

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")

# See full model card for synthesize() implementation and token constants.
wav = synthesize("Mwattu, Mukama yeebazibwe.", speaker_id="salt_lug_0001", seed=42)
sf.write("luganda.wav", wav, 24000)
```

### Option B — vLLM (high throughput / batched)

```bash
pip install vllm snac soundfile torchcodec "datasets>=3.4.1,<4.0.0"
```

> **Note:** vLLM ships its own torch/transformers versions and conflicts with Unsloth's pinned versions — use a separate Python environment for vLLM serving.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="sunbird/orpheus-3b-tts-multilingual",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

# See full model card for _build_prompt_token_ids(), synthesize(), and
# synthesize_batch() implementations, which support mixing different
# speaker_ids in a single batched call.
```

### Generation parameters

| Param | Default | What it does |
|---|---|---|
| `temperature` | 0.6 | Lower = more deterministic, flatter prosody |
| `top_p` | 0.95 | Nucleus sampling; don't drop below 0.9 (robotic audio) |
| `repetition_penalty` | 1.1 | Discourages stuck-on-one-frame artefacts |
| `max_new_tokens` / `max_tokens` | 1200 | ≈ 9–10s of audio |
| `seed` | None | Pass an int for reproducible output |

## Training Details

| Setting | Value |
|---|---|
| Base model | `unsloth/orpheus-3b-0.1-pretrained` |
| Adapter | LoRA r=64, α=64, dropout=0, bias=none |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | adamw_8bit, weight decay 0.001 |
| LR schedule | linear, lr=2e-4, warmup steps=5 |
| Batch size | 1 (grad accumulation=4, effective batch=4) |
| Epochs | 3 |
| max_seq_length | 4096 |
| Precision | bfloat16 weights, 16-bit LoRA |
| Hardware | single NVIDIA RTX 4090 (24 GB) |
| Final save | LoRA merged into 16-bit weights |

Training data was drawn from all 20 configs of `Sunbird/tts` (train + test splits concatenated, no speaker filter), with audio encoded via `hubertsiuzdak/snac_24khz` and tagged per-row with `speaker_id` so the model learns the multi-speaker prompt format across every voice.

## Licence

Apache 2.0 (matching the upstream `unsloth/orpheus-3b-0.1-pretrained` license). Transitively inherits attribution obligations from the Orpheus-TTS project (CanopyAI), the Llama-3 base architecture (Meta Llama 3 Community License), the SNAC codec (MIT), and the `Sunbird/tts` dataset and its voice donors.

## Hugging Face

[huggingface.co/Sunbird/orpheus-3b-tts-multilingual](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual)

## Related Model

A single-speaker variant is also available for those who only need one voice: `sunbird/orpheus-3b-tts-salt-lug-0001` (same recipe, scoped to a single Luganda speaker).

## Last Updated

2026

## Citation

```bibtex
@misc{sunbird_orpheus3b_multilingual_2026,
  title        = {Orpheus-3B Sunbird Multilingual TTS},
  author       = {Sunbird AI},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sunbird/orpheus-3b-tts-multilingual}},
}

@misc{sunbird_tts_dataset,
  title        = {Sunbird Speech Dataset},
  author       = {Sunbird AI},
  howpublished = {\url{https://huggingface.co/datasets/Sunbird/tts}},
}

@misc{orpheus_tts_2025,
  title        = {Orpheus-TTS},
  author       = {Canopy Labs},
  year         = {2025},
  howpublished = {\url{https://github.com/canopyai/Orpheus-TTS}},
}
```
