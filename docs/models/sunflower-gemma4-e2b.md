---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-E2B-it
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- gemma4
- multimodal
- multilingual
- african-languages
- translation
- automatic-speech-recognition
- audio
- sunbird
datasets:
- Sunbird/sunflower-pretrain-data
- Sunbird/speech
- Sunbird/sunflower-posttrain-data
metrics:
- bleu
- chrf
- wer
- cer
language:
- eng
- ach
- adh
- afr
- aka
- alz
- amh
- bam
- bem
- ber
- bfa
- cgg
- dag
- dga
- din
- ewe
- fra
- ful
- gwr
- hau
- ibo
- ikx
- kab
- kau
- kdi
- kdj
- keo
- kik
- kin
- kln
- koo
- kpo
- kpz
- laj
- led
- lgg
- lin
- lsm
- lth
- luc
- lug
- luo
- luy
- mhi
- mlg
- myx
- nbl
- nuj
- nya
- nyn
- nyo
- orm
- pcm
- pok
- rub
- ruc
- run
- rwm
- sna
- som
- sot
- swa
- teo
- tlj
- tsn
- ttj
- wol
- xho
- xog
- yor
- zul
---

# Sunflower-Gemma4-E2B

## Overview

> [!WARNING]
> **This is a preview release.** The model weights will change between now and the stable
> release. Benchmark numbers, prompt behaviour, and output quality may all shift. Please don't
> pin production systems to this checkpoint — treat it as a preview for evaluation and
> feedback.

**Sunflower-Gemma4-E2B** is a multimodal adaptation of [Google's Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it) optimised to understand text and speech across **69 African languages**. 
The model is tuned for text translation, speech transcription, instruction-following, and multi-turn chat. It supports text, image, and audio inputs (up to 30s of 16kHz audio), generating text output.

## Task Type

Image-Text-to-Text, Multimodal Machine Translation, Speech Transcription, Multilingual Chat, Instruction Following.

## Base Architecture

Google Gemma 4 E2B (`google/gemma-4-E2B-it`).

## Model Size

5B parameters.

## Quantization Format

None. This is the full precision model.

## Languages Supported

Sunflower-Gemma4-E2B supports English and the following 69 African or regional languages:

Acholi, Afrikaans, Akan, Alur, Amharic, Bambara, Bari, Bemba, Berber, Chichewa, Dagaare, Dagbani, Dinka, Ewe, French, Fulani, Hausa, Igbo, Ik, Ikposo, Jopadhola, Kabyle, Kakwa, Kalenjin, Kanuri, Karamojong, Kikuyu, Kinyarwanda, Kirundi, Kumam, Kupsabiny, Kwamba, Lango, Lendu, Lingala, Lubwisi, Luganda, Lugbara, Lugungu, Lugwere, Luhya, Lumasaba, Lunyole, Luo, Lusoga, Ma'di, Malagasy, Ndebele, Nigerian Pidgin, Oromo, Pokot, Rukiga, Rukonjo, Runyankole, Runyoro, Ruruuli, Rutooro, Samia, Shona, Somali, Sotho, Swahili, Tswana, Wolof, Xhosa, Yoruba, and Zulu.

## Intended Use

Sunflower-Gemma4-E2B is intended for developers building text and speech applications for African languages. Key use cases include:

- Text-to-text translation between English and supported African languages.
- Speech transcription (ASR) for the supported languages.
- Multilingual chat and instruction following.
- Research on low-resource multimodal speech and language models.

## Known Limitations

- **Preview weights:** Output will change before the stable release.
- **Uneven multilingual quality:** Lower-resource languages have substantially lower automatic scores. The low-resource end of the table is genuinely weak, and output there should be verified by a speaker before use.
- **Translation variants:** BLEU and chrF compare against a single reference and can penalize acceptable wording, dialect, or orthographic variants.
- **Speech variation:** Accent, dialect, code-switching, background noise, and domain vocabulary can materially affect transcription.
- **Language prompting:** The model was trained with language-specific transcription prompts and may perform poorly when the spoken language is omitted or named incorrectly.
- **Hallucinations:** Like any LLM, it can produce fluent, confident, but incorrect translations or transcripts. This matters more in low-resource languages where errors are harder for non-speakers to spot.

## How to Load / Run

You can try the model directly using this [Colab Notebook](https://colab.research.google.com/drive/1P39eYR_KW16oH8tBzycjHG86RfxiRKMJ?usp=sharing)

Sunflower-Gemma4-E2B follows standard Transformers patterns, but expects specific prompt formats for translation and speech transcription. Like Qwen, the base model supports a reasoning phase, but we recommend setting `enable_thinking=False` for standard translation tasks.

### Text Translation

Translation expects a **fixed instruction template**:
`Translate from English to {language name}: {text}`

```python
messages = [
    {"role": "system", "content": "You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages."},
    {"role": "user", "content": "Translate from English to Luganda: Where is the nearest hospital?"}
]
```

*Tip: Use `do_sample=False` (greedy decoding) and `temperature=0` for translation tasks.*

### Audio Transcription

The model was trained with prompts that explicitly name the spoken language. Convert audio to a 16 kHz mono waveform and keep it to 30 seconds or less.

```python
messages = [
    {"role": "system", "content": "You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages."},
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_array}, # 16kHz mono numpy array
            {"type": "text", "text": "Please transcribe this Luganda audio."},
        ],
    },
]
```

### vLLM (Text-Only Inference)

vLLM is highly recommended for fast deployments. Because this is a multimodal model, text-only vLLM setup requires disabling vision/audio inputs to save memory.

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL_ID = "Sunbird/Sunflower-Gemma4-E2B"

llm = LLM(
    MODEL_ID,
    dtype="bfloat16",
    max_model_len=1024,
    limit_mm_per_prompt={"image": 0, "audio": 0}, # Text-only mode
    enforce_eager=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def make_prompt(text, target_lang):
    messages = [
        {"role": "system", "content": "You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages."},
        {"role": "user", "content": f"Translate from English to {target_lang}: {text}"}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling = SamplingParams(temperature=0.0, max_tokens=256)

examples = [
    ("Good morning, how are you?", "Luganda"),
    ("The children are going to school today.", "Acholi"),
    ("I would like to buy three kilograms of rice.", "Swahili"),
]
outputs = llm.generate([make_prompt(t, l) for t, l in examples], sampling)
for (text, lang), out in zip(examples, outputs):
    print(f"[{lang}] {text}\n     -> {out.outputs[0].text.strip()}\n")
```

### Transformers (Multimodal / Audio)

```python
import torch
import librosa
import numpy as np
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "Sunbird/Sunflower-Gemma4-E2B"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, dtype=dtype, device_map="auto", attn_implementation="sdpa"
).eval()

# Load Audio
audio, _ = librosa.load("audio.wav", sr=16_000, mono=True)
audio = np.asarray(audio, dtype=np.float32)
if len(audio) > 480_000:
    raise ValueError("Please provide no more than 30 seconds of audio.")

messages = [
    {"role": "system", "content": "You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages."},
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Please transcribe this Luganda audio."}
    ]}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[prompt], audio=[audio], return_tensors="pt",
    text_kwargs={"padding": False, "truncation": False, "add_special_tokens": False},
    audio_kwargs={"max_length": 480_000}
).to(model.device)
inputs["input_features"] = inputs["input_features"].to(dtype=dtype)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

new_tokens = output[0][inputs["input_ids"].shape[-1]:]
raw_response = processor.decode(new_tokens, skip_special_tokens=False)
print(processor.parse_response(raw_response)["content"])
```

## Training

The model was adapted via the following stages:
1. Continued pretraining on multilingual African-language text (documents, parallel text, transcripts, web, and synthetic text).
2. Speech SFT on audio-transcription examples.
3. Mixed text-and-audio SFT on translation, instruction, and speech-transcription tasks.
4. GRPO post-training to improve translation and general response quality using task-specific reward signals.

## Evaluation

### Text Translation
Translation quality measured as **chrF** on the `Sunbird/salt69` test split (100 examples per language).

| Languages | Directions | Macro BLEU | Macro chrF | Mean Length Ratio |
|---|---|---|---|---|
| 69 | 138 | 12.20 | 0.3413 | 1.06 |

<details>
<summary>Click to view full per-language translation results</summary>

| Code | Language | EN→X BLEU | EN→X chrF | X→EN BLEU | X→EN chrF | Mean chrF |
|---|---|---:|---:|---:|---:|---:|
| afr | Afrikaans | 57.68 | 0.7747 | 57.86 | 0.7649 | 0.7698 |
| pcm | Nigerian Pidgin | 32.76 | 0.6438 | 48.27 | 0.7306 | 0.6872 |
| fra | French | 34.08 | 0.6008 | 40.55 | 0.6532 | 0.6270 |
| swa | Swahili | 30.77 | 0.6126 | 38.06 | 0.6270 | 0.6198 |
| zul | Zulu | 19.95 | 0.5949 | 34.19 | 0.5887 | 0.5918 |
| nya | Chichewa | 25.38 | 0.6288 | 31.98 | 0.5537 | 0.5913 |
| xho | Xhosa | 15.57 | 0.5595 | 29.89 | 0.5357 | 0.5476 |
| som | Somali | 15.22 | 0.5180 | 31.48 | 0.5647 | 0.5413 |
| mlg | Malagasy | 19.62 | 0.5458 | 23.64 | 0.4907 | 0.5182 |
| nbl | Ndebele | 12.35 | 0.5109 | 29.58 | 0.5239 | 0.5174 |
| ibo | Igbo | 29.19 | 0.5273 | 24.54 | 0.4913 | 0.5093 |
| lin | Lingala | 23.92 | 0.5414 | 23.37 | 0.4670 | 0.5042 |
| kin | Kinyarwanda | 17.54 | 0.5148 | 25.29 | 0.4906 | 0.5027 |
| sot | Sotho | 21.60 | 0.5077 | 26.18 | 0.4950 | 0.5013 |
| hau | Hausa | 20.52 | 0.5102 | 24.03 | 0.4885 | 0.4993 |
| amh | Amharic | 14.52 | 0.3752 | 32.33 | 0.5735 | 0.4743 |
| lug | Luganda | 14.99 | 0.5089 | 22.69 | 0.4384 | 0.4737 |
| tsn | Tswana | 21.32 | 0.5051 | 20.01 | 0.4370 | 0.4710 |
| sna | Shona | 9.52 | 0.4642 | 23.02 | 0.4762 | 0.4702 |
| run | Kirundi | 10.80 | 0.4388 | 23.95 | 0.4554 | 0.4471 |
| nyo | Runyoro | 12.07 | 0.4644 | 18.33 | 0.4224 | 0.4434 |
| nyn | Runyankole | 9.33 | 0.4415 | 15.84 | 0.4114 | 0.4264 |
| yor | Yoruba | 19.98 | 0.3985 | 20.58 | 0.4467 | 0.4226 |
| cgg | Rukiga | 8.30 | 0.4172 | 16.60 | 0.4139 | 0.4155 |
| xog | Lusoga | 7.48 | 0.4086 | 15.87 | 0.3784 | 0.3935 |
| ttj | Rutooro | 8.18 | 0.3906 | 14.71 | 0.3868 | 0.3887 |
| ewe | Ewe | 13.83 | 0.4096 | 10.80 | 0.3265 | 0.3680 |
| ruc | Ruruuli | 5.14 | 0.3643 | 15.93 | 0.3628 | 0.3636 |
| kik | Kikuyu | 6.41 | 0.3459 | 14.91 | 0.3739 | 0.3599 |
| luo | Luo | 9.89 | 0.3885 | 12.86 | 0.3255 | 0.3570 |
| ach | Acholi | 10.98 | 0.4167 | 10.99 | 0.2872 | 0.3519 |
| orm | Oromo | 6.10 | 0.3643 | 8.88 | 0.2961 | 0.3302 |
| aka | Akan | 9.37 | 0.3244 | 10.23 | 0.3186 | 0.3215 |
| gwr | Lugwere | 3.98 | 0.3349 | 9.97 | 0.3074 | 0.3211 |
| bem | Bemba | 5.01 | 0.3305 | 10.24 | 0.3084 | 0.3195 |
| laj | Lango | 6.32 | 0.3329 | 9.13 | 0.2715 | 0.3022 |
| kdi | Kumam | 6.10 | 0.3176 | 7.24 | 0.2739 | 0.2957 |
| rub | Lugungu | 3.35 | 0.2826 | 9.40 | 0.3040 | 0.2933 |
| nuj | Lunyole | 4.38 | 0.2994 | 7.73 | 0.2608 | 0.2801 |
| koo | Rukonjo | 2.72 | 0.3139 | 7.14 | 0.2415 | 0.2777 |
| tlj | Lubwisi | 2.75 | 0.2743 | 8.86 | 0.2780 | 0.2762 |
| myx | Lumasaba | 3.18 | 0.2784 | 8.34 | 0.2674 | 0.2729 |
| lsm | Samia | 3.42 | 0.2859 | 9.46 | 0.2595 | 0.2727 |
| alz | Alur | 3.86 | 0.2605 | 5.24 | 0.2423 | 0.2514 |
| adh | Jopadhola | 4.52 | 0.2624 | 6.54 | 0.2371 | 0.2498 |
| lgg | Lugbara | 4.68 | 0.2869 | 4.00 | 0.2114 | 0.2491 |
| kab | Kabyle | 5.10 | 0.2313 | 6.91 | 0.2438 | 0.2375 |
| teo | Ateso | 3.38 | 0.2498 | 5.13 | 0.2187 | 0.2343 |
| wol | Wolof | 3.99 | 0.2148 | 5.60 | 0.2309 | 0.2228 |
| bfa | Bari | 2.88 | 0.1759 | 6.28 | 0.2441 | 0.2100 |
| rwm | Kwamba | 1.87 | 0.2103 | 5.19 | 0.2068 | 0.2086 |
| ful | Fulani | 3.24 | 0.1802 | 6.16 | 0.2354 | 0.2078 |
| dag | Dagbani | 3.71 | 0.2026 | 5.77 | 0.2106 | 0.2066 |
| bam | Bambara | 5.34 | 0.1920 | 4.87 | 0.2170 | 0.2045 |
| keo | Kakwa | 2.42 | 0.1812 | 4.37 | 0.2086 | 0.1949 |
| ber | Berber | 3.26 | 0.1749 | 4.08 | 0.1973 | 0.1861 |
| mhi | Ma'di | 2.29 | 0.1575 | 4.82 | 0.2127 | 0.1851 |
| luy | Luhya | 0.98 | 0.1500 | 5.49 | 0.2198 | 0.1849 |
| led | Lendu | 1.83 | 0.1356 | 3.69 | 0.2203 | 0.1779 |
| kdj | Karamojong | 1.50 | 0.1586 | 3.69 | 0.1842 | 0.1714 |
| pok | Pokot | 1.19 | 0.1231 | 3.60 | 0.1817 | 0.1524 |
| ikx | Ik | 0.90 | 0.1060 | 4.69 | 0.1970 | 0.1515 |
| kpz | Kupsabiny | 0.72 | 0.0973 | 3.68 | 0.1915 | 0.1444 |
| dga | Dagaare | 1.02 | 0.0952 | 3.79 | 0.1912 | 0.1432 |
| luc | Aringa | 1.07 | 0.0909 | 4.49 | 0.1908 | 0.1409 |
| kau | Kanuri | 0.68 | 0.0848 | 4.09 | 0.1971 | 0.1409 |
| kln | Kalenjin | 0.78 | 0.1094 | 4.14 | 0.1715 | 0.1405 |
| din | Dinka | 1.24 | 0.0859 | 2.97 | 0.1869 | 0.1364 |
| kpo | Ikposo | 0.14 | 0.0401 | 1.71 | 0.1526 | 0.0964 |

</details>

### Speech Transcription
Reported ASR results span 110 dataset configurations from `Sunbird/speech`. Performance varies substantially by language and recording source.

| Languages | Scored Examples | Macro Language WER | Macro Language CER | Macro Mean Utterance WER | Macro Mean Utterance CER |
|---|---|---|---|---|---|
| 51 | 5201 | 0.4720 | 0.1890 | 0.5110 | 0.2016 |

<details>
<summary>Click to view full per-language transcription results</summary>

*(Rates shown on a 0–1 scale; lower is better. WER can exceed 1.0 when insertions outnumber reference words.)*

| Code | Language | Scored | Corpus WER | Corpus CER |
|---|---|---:|---:|---:|
| ach | Acholi | 92 | 0.4086 | 0.1758 |
| afr | Afrikaans | 170 | 0.1413 | 0.0646 |
| aka | Akan | 50 | 0.4182 | 0.1779 |
| amh | Amharic | 250 | 0.3256 | 0.1474 |
| bam | Bambara | 149 | 0.6919 | 0.3454 |
| bem | Bemba | 100 | 0.5718 | 0.1392 |
| ber | Berber | 50 | 0.5000 | 0.1924 |
| cgg | Rukiga | 49 | 0.4254 | 0.1295 |
| dag | Dagbani | 100 | 0.4627 | 0.1871 |
| dga | Dagaare | 50 | 0.3920 | 0.1738 |
| eng | English | 100 | 0.1515 | 0.0819 |
| ewe | Ewe | 78 | 0.5833 | 0.2704 |
| fra | French | 50 | 0.1029 | 0.0525 |
| ful | Fulani | 149 | 0.5396 | 0.2009 |
| hau | Hausa | 287 | 0.2367 | 0.0844 |
| ibo | Igbo | 247 | 0.5664 | 0.2802 |
| kab | Kabyle | 50 | 0.7273 | 0.3696 |
| kau | Kanuri | 50 | 0.6795 | 0.2133 |
| kik | Kikuyu | 50 | 0.3798 | 0.1893 |
| kin | Kinyarwanda | 100 | 0.4653 | 0.1787 |
| kln | Kalenjin | 100 | 0.6164 | 0.1815 |
| koo | Rukonjo | 47 | 0.7333 | 0.2119 |
| kpo | Ikposo | 50 | 0.7418 | 0.3416 |
| led | Lendu | 50 | 0.5656 | 0.2646 |
| lgg | Lugbara | 50 | 0.3575 | 0.0990 |
| lin | Lingala | 193 | 0.3886 | 0.1800 |
| lth | Thur | 41 | 0.4299 | 0.2430 |
| lug | Luganda | 266 | 0.4999 | 0.1669 |
| luo | Luo | 149 | 0.3495 | 0.1096 |
| luy | Luhya | 50 | 0.4335 | 0.1015 |
| mlg | Malagasy | 49 | 0.3359 | 0.1561 |
| myx | Lumasaba | 45 | 0.6237 | 0.1814 |
| nbl | Ndebele | 10 | 0.5043 | 0.2032 |
| nya | Chichewa | 98 | 0.5949 | 0.2212 |
| nyn | Runyankole | 94 | 0.4977 | 0.1613 |
| orm | Oromo | 119 | 0.4602 | 0.1488 |
| pcm | Nigerian Pidgin | 100 | 0.3856 | 0.3043 |
| ruc | Ruruuli | 38 | 0.6701 | 0.1932 |
| rwm | Kwamba | 49 | 1.2251 | 0.7426 |
| sna | Shona | 147 | 0.3025 | 0.0951 |
| som | Somali | 149 | 0.4731 | 0.1898 |
| sot | Sotho | 94 | 0.3946 | 0.1780 |
| swa | Swahili | 99 | 0.1619 | 0.0585 |
| teo | Ateso | 50 | 0.4567 | 0.1516 |
| tsn | Tswana | 98 | 0.2225 | 0.0826 |
| ttj | Rutooro | 49 | 0.5753 | 0.2050 |
| wol | Wolof | 135 | 0.5280 | 0.2602 |
| xho | Xhosa | 126 | 0.4323 | 0.1247 |
| xog | Lusoga | 43 | 0.4995 | 0.1378 |
| yor | Yoruba | 249 | 0.5311 | 0.2042 |
| zul | Zulu | 143 | 0.3090 | 0.0863 |

</details>

## Licence

Released under the **Apache 2.0** license, matching the Gemma 4 base model.

## Hugging Face

[https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B)

## Last Updated

2026

## Citation

```bibtex
@misc{sunbird2026sunflowergemma4e2b,
  title        = {Sunflower-Gemma4-E2B},
  author       = {{Sunbird AI}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B}},
  note         = {Preview release: Multilingual text and speech model for African languages}
}
```
