---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
pipeline_tag: text-generation
tags:
- translation
- african-languages
- ugandan-languages
- multilingual
- sunbird
- qwen3.5
language:
- eng
- afr
- zul
- pcm
- swa
- xho
- nya
- lin
- som
- ibo
- sot
- mlg
- kin
- tsn
- lug
- hau
- sna
- nbl
- yor
- amh
- nyo
- run
- ewe
- xog
- luo
- nyn
- kik
- ach
- cgg
- orm
- ttj
- ruc
- bem
- laj
- aka
- gwr
- kdi
- rub
- lgg
- myx
- teo
- nuj
- kab
- lsm
- tlj
- koo
- bam
- adh
- wol
- alz
- bfa
- keo
- dag
- kdj
- ber
- rwm
- ful
- mhi
- luc
- luy
- kln
- led
- dga
- kpz
- pok
- ikx
- kau
- din
- kpo
- fra
---

# Sunflower-Qwen3.5-9B

## Overview

> [!WARNING]
> **This is a preview release.** The model weights will change between now and the final
> release. Benchmark numbers, prompt behaviour and output quality may all shift. Please don't
> pin production systems to this checkpoint — treat it as a preview for evaluation and
> feedback.

**Sunflower-Qwen3.5-9B** is an adaptation of Qwen3.5-9B to understand text in **67 African languages**. 
The model was optimised for translation, instruction-following and multi-turn chat in these languages.
It was trained by carrying out continued pretraining on a text corpus of the target languages, followed by instruction tuning.

## Task Type

Text Generation, Machine Translation, Multilingual Chat, Instruction Following.

## Base Architecture

Qwen3.5-9B (`Qwen/Qwen3.5-9B`).

## Model Size

9B parameters.

## Quantization Format

None. This is the full precision model. A pre-quantised 4-bit build is separately available as `Sunbird/Sunflower-Qwen3.5-9B-bnb-4bit` (~7.9 GB download).

## Languages Supported

Coverage varies considerably by language, roughly tracking the amount of training data
available. The tiers below are a practical guide to what to expect.

- **Good understanding:** Afrikaans, Zulu, Nigerian Pidgin, Chichewa, Swahili, Xhosa, Somali, Lingala, Sotho, Igbo, Malagasy, Kinyarwanda, Hausa, Tswana, Yoruba, Luganda, Shona, Ndebele, Amharic, Kirundi, Runyoro, Luo, Ewe, Runyankole.
- **Moderate understanding:** Kikuyu, Lusoga, Acholi, Rukiga, Rutooro, Oromo, Bemba, Ruruuli, Akan, Lango, Lugwere, Lugbara, Kumam, Lugungu, Lumasaba, Ateso, Lunyole.
- **Basic understanding:** Kabyle, Rukonjo, Bambara, Lubwisi, Wolof, Samia, Jopadhola, Alur, Bari, Kakwa, Dagbani, Karamojong, Berber, Fulani, Kwamba, Ma'di, Aringa, Luhya, Kalenjin, Kupsabiny, Lendu, Dagaare, Ik, Pokot, Kanuri, Dinka.

## Intended Use

Sunflower-Qwen3.5-9B is intended for developers building text applications in African languages. Suitable use cases include:

- Machine translation between English and supported African languages.
- Text generation and summarization in supported languages.
- Multilingual chat and instruction following.
- Research on low-resource language modeling.

## Known Limitations

- **Preview weights:** Output will change before the final release.
- **Coverage is uneven:** See the tiers and the metrics table — the low-resource end of the table is genuinely weak, and output there should be verified by a speaker before use.
- **Translation is the optimised path:** General chat and reasoning work, but the model has been tuned most heavily for the translation template.
- Like any LLM, it can produce fluent, confident and wrong output. This matters more in low-resource languages, where errors are harder for non-speakers to spot.

## How to Load / Run

Sunflower-Qwen3.5-9B follows the same usage pattern as the base Qwen3.5-9B model, though it expects prompts for some tasks in a certain format as specified below. Like Qwen3.5, the model supports a `<think>` reasoning phase. Reasoning has not yet been optimised so we recommend setting `enable_thinking=False`, especially for translation tasks.

### Translation prompt template

Translation was a primary optimisation target, and it expects a **fixed instruction template**:

```
Translate to {language name}: {text}
```

`{language name}` is the **English name** of the target language — `Luganda`, `Acholi`,
`Swahili`, `Runyankole` — not the ISO code. Use the names from the metrics table below.

```python
messages = [{"role": "user", "content": "Translate to Luganda: Good morning! How did you sleep?"}]
```

Two practical notes:

- Use temperature 0 (greedy) for translation.
- The source language is inferred from the text, and only the target needs to be specified. So
  `Translate to English: Ndagira kununua chakula...` works without naming the source.
  However the source language can also be specified optionally, e.g. 'Translate from Luganda to English: Oli otya?` which may improve quality in some cases.

### Other prompt patterns

The model is a general instruction-follower, not only a translation engine. Two other patterns
work well:

**1. Instruction in English, specifying the reply language.** Useful when you want a complex
task performed but the output in an African language:

```python
prompt = (
    "Invent a new version of the Cinderella story where the ending is different, "
    "focusing on a theme of self-reliance rather than rescue, formatted as a short "
    "story with exactly three paragraphs. Reply in Yoruba."
)
```

**2. Instruction written directly in the target language.** The model responds in the language
you write in:

```python
prompt = (
    "Hlaziya imiqobo yokungavumi kwamayeza okugonya kwiindawo ezithile zamasiko, "
    "ugxile ngokukodwa kwindima yeenkokheli zonqulo."
)
```

For these open-ended generation tasks temperature around 0.7 gives more natural output.

Multi-turn conversation works as normal: append the assistant's reply to `messages` and
continue.

### vLLM

vLLM is the default choice for deployment, being much faster than transformers.

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine; reliable in notebooks

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL_ID = "Sunbird/Sunflower-Qwen3.5-9B"

llm = LLM(
    MODEL_ID,
    quantization="fp8",               # ~19 GB of weights -> ~9.5 GB
    dtype="bfloat16",
    max_model_len=1024,               # translation prompts + outputs are short
    gpu_memory_utilization=0.9,
    limit_mm_per_prompt={"image": 0, "video": 0},  # text-only; frees memory
    enforce_eager=True,               # skips CUDA-graph capture (saves memory)
)
tokenizer = AutoProcessor.from_pretrained(MODEL_ID).tokenizer


def make_prompt(text, target_language):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Translate to {target_language}: {text}"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


sampling = SamplingParams(temperature=0.0, max_tokens=256)  # temp 0 for translation

examples = [
    ("Good morning, how are you?", "Luganda"),
    ("The children are going to school today.", "Acholi"),
    ("I would like to buy three kilograms of rice.", "Swahili"),
]
outputs = llm.generate([make_prompt(t, l) for t, l in examples], sampling)
for (text, lang), out in zip(examples, outputs):
    print(f"[{lang}] {text}\n     -> {out.outputs[0].text.strip()}\n")
```

Pass the whole list to a single `llm.generate()` call to translate a large batch at once —
far faster than looping.

### Transformers

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Sunbird/Sunflower-Qwen3.5-9B"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
tokenizer = AutoProcessor.from_pretrained(MODEL_ID).tokenizer

messages = [{"role": "user", "content": "Translate to Luganda: The children are going to school today."}]
enc = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

out = model.generate(**enc, max_new_tokens=256, do_sample=False)  # greedy for translation
print(tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

A pre-quantised 4-bit build, **`Sunbird/Sunflower-Qwen3.5-9B-bnb-4bit`**, is also available
(~7.9 GB download, runs on any CUDA GPU including a free Colab T4). Load it the same way —
the quantisation config is baked into its `config.json`, so you don't pass
`quantization_config`.


## Training

Built on **Qwen3.5-9B**, adapted through continued pretraining and supervised fine-tuning on
Ugandan and other African language data, followed by reinforcement learning (GRPO) against
translation-quality, factuality, language-consistency and instruction-following rewards.

## Evaluation

Translation quality measured as **chrF** on the
[`Sunbird/salt69`](https://huggingface.co/datasets/Sunbird/salt69) test split, in both
directions (English→language and language→English). Higher is better; scores are 0–1.
We find that translation accuracy is a strong predictor of ability to carry out other tasks
per language.

Sorted by average score.

| Language | Code | eng→xx | xx→eng | Average |
|---|---|---|---|---|
| Afrikaans | `afr` | 0.8016 | 0.8039 | **0.8027** |
| Zulu | `zul` | 0.6703 | 0.7050 | **0.6877** |
| Nigerian Pidgin | `pcm` | 0.5975 | 0.7536 | **0.6756** |
| Swahili | `swa` | 0.6584 | 0.6847 | **0.6715** |
| Xhosa | `xho` | 0.6185 | 0.6956 | **0.6570** |
| French | `fra` | 0.6084 | 0.6805 | **0.6444** |
| Chichewa | `nya` | 0.6451 | 0.6407 | **0.6429** |
| Lingala | `lin` | 0.6128 | 0.6589 | **0.6359** |
| Somali | `som` | 0.5578 | 0.6994 | **0.6286** |
| Igbo | `ibo` | 0.5845 | 0.6398 | **0.6122** |
| Sotho | `sot` | 0.5788 | 0.6419 | **0.6103** |
| Malagasy | `mlg` | 0.5752 | 0.6112 | **0.5932** |
| Kinyarwanda | `kin` | 0.5788 | 0.5941 | **0.5864** |
| Tswana | `tsn` | 0.5519 | 0.6083 | **0.5801** |
| Luganda | `lug` | 0.5551 | 0.5746 | **0.5649** |
| Hausa | `hau` | 0.5316 | 0.5912 | **0.5614** |
| Shona | `sna` | 0.5200 | 0.6010 | **0.5605** |
| Ndebele | `nbl` | 0.4858 | 0.6323 | **0.5591** |
| Yoruba | `yor` | 0.4638 | 0.6335 | **0.5487** |
| Amharic | `amh` | 0.4263 | 0.6459 | **0.5361** |
| Runyoro | `nyo` | 0.4952 | 0.5628 | **0.5290** |
| Kirundi | `run` | 0.4833 | 0.5701 | **0.5267** |
| Ewe | `ewe` | 0.4802 | 0.5358 | **0.5080** |
| Lusoga | `xog` | 0.4548 | 0.5507 | **0.5028** |
| Luo | `luo` | 0.4969 | 0.5078 | **0.5024** |
| Runyankole | `nyn` | 0.4987 | 0.5031 | **0.5009** |
| Kikuyu | `kik` | 0.3956 | 0.5858 | **0.4907** |
| Acholi | `ach` | 0.4864 | 0.4857 | **0.4861** |
| Rukiga | `cgg` | 0.4635 | 0.4961 | **0.4798** |
| Oromo | `orm` | 0.4261 | 0.5007 | **0.4634** |
| Rutooro | `ttj` | 0.4270 | 0.4973 | **0.4621** |
| Ruruuli | `ruc` | 0.3994 | 0.5035 | **0.4515** |
| Bemba | `bem` | 0.4391 | 0.4382 | **0.4387** |
| Lango | `laj` | 0.3820 | 0.4544 | **0.4182** |
| Akan | `aka` | 0.3727 | 0.4613 | **0.4170** |
| Lugwere | `gwr` | 0.3718 | 0.4406 | **0.4062** |
| Kumam | `kdi` | 0.3646 | 0.4237 | **0.3942** |
| Lugungu | `rub` | 0.3533 | 0.4182 | **0.3857** |
| Lugbara | `lgg` | 0.3946 | 0.3759 | **0.3852** |
| Lumasaba | `myx` | 0.3730 | 0.3852 | **0.3791** |
| Ateso | `teo` | 0.3516 | 0.3992 | **0.3754** |
| Lunyole | `nuj` | 0.3551 | 0.3767 | **0.3659** |
| Kabyle | `kab` | 0.2927 | 0.4132 | **0.3529** |
| Samia | `lsm` | 0.3325 | 0.3643 | **0.3484** |
| Lubwisi | `tlj` | 0.3299 | 0.3619 | **0.3459** |
| Rukonjo | `koo` | 0.3483 | 0.3395 | **0.3439** |
| Bambara | `bam` | 0.3118 | 0.3719 | **0.3419** |
| Jopadhola | `adh` | 0.3362 | 0.3414 | **0.3388** |
| Wolof | `wol` | 0.2947 | 0.3824 | **0.3385** |
| Alur | `alz` | 0.2768 | 0.3741 | **0.3254** |
| Bari | `bfa` | 0.2960 | 0.3196 | **0.3078** |
| Kakwa | `keo` | 0.3062 | 0.2805 | **0.2933** |
| Dagbani | `dag` | 0.2809 | 0.2911 | **0.2860** |
| Karamojong | `kdj` | 0.2662 | 0.2691 | **0.2676** |
| Berber | `ber` | 0.2212 | 0.3061 | **0.2636** |
| Kwamba | `rwm` | 0.2633 | 0.2525 | **0.2579** |
| Fulani | `ful` | 0.2099 | 0.2887 | **0.2493** |
| Ma'di | `mhi` | 0.2291 | 0.2618 | **0.2455** |
| Aringa | `luc` | 0.1972 | 0.2628 | **0.2300** |
| Luhya | `luy` | 0.1798 | 0.2570 | **0.2184** |
| Kalenjin | `kln` | 0.1673 | 0.2470 | **0.2071** |
| Lendu | `led` | 0.1626 | 0.2338 | **0.1982** |
| Dagaare | `dga` | 0.1602 | 0.2301 | **0.1952** |
| Kupsabiny | `kpz` | 0.1506 | 0.2205 | **0.1856** |
| Pokot | `pok` | 0.1567 | 0.2098 | **0.1832** |
| Ik | `ikx` | 0.1400 | 0.2249 | **0.1824** |
| Kanuri | `kau` | 0.1085 | 0.2251 | **0.1668** |
| Dinka | `din` | 0.1161 | 0.2061 | **0.1611** |
| Ikposo | `kpo` | 0.0465 | 0.1484 | **0.0974** |


`fra` (French) is included as a high-resource reference point.

## Licence

Released under the **Apache 2.0** license.

## Hugging Face

[https://huggingface.co/Sunbird/Sunflower-Qwen3.5-9B](https://huggingface.co/Sunbird/Sunflower-Qwen3.5-9B)

## Last Updated

2026

## Citation

```bibtex
@misc{sunflower_qwen35_9b,
  title        = {Sunflower-Qwen3.5-9B},
  author       = {Sunbird AI},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Sunbird/Sunflower-Qwen3.5-9B}},
  note         = {Preview release}
}
```
