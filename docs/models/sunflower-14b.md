# Sunflower-14B

## Overview

Sunflower-14B is a multilingual causal language model developed by Sunbird AI for Ugandan languages and English. It is built on the Qwen3-14B architecture and is designed for translation, text generation, factual question answering, summarization, explanation, and multilingual instruction following.

The current release includes GRPO fine-tuning to improve translation fluency, factual correctness, instruction following, response consistency, multilingual reasoning, and hallucination reduction compared with earlier releases.

## Task Type

Text Generation, Machine Translation, Multilingual Question Answering, Summarization, Instruction Following.

## Base Architecture

Qwen3-14B.

## Model Size

15B parameters.

## Quantization Format

None. This is the full precision Sunflower-14B model. For lower-memory inference, use one of the separately released quantized variants such as GGUF, FP8, FP4A16, or W8A8.

## Languages Supported

Sunflower-14B supports English and 31 Ugandan or regional African languages. The Hugging Face model card lists the supported languages using the following language codes:

`ach`, `adh`, `alz`, `bfa`, `cgg`, `en`, `gwr`, `kdi`, `kdj`, `keo`, `kin`, `koo`, `kpz`, `laj`, `lgg`, `lsm`, `luc`, `lug`, `mhi`, `myx`, `nuj`, `nyn`, `nyo`, `pok`, `rub`, `ruc`, `rwm`, `swa`, `teo`, `tlj`, `ttj`, `xog`.

## Intended Use

Sunflower-14B is intended for users building multilingual applications for Ugandan languages and English. Primary use cases include:

- Translation between English and Ugandan languages.
- Translation between Ugandan languages.
- Text generation in Ugandan languages.
- Multilingual factual question answering.
- Educational and knowledge-access applications.
- Summarization and explanation tasks.
- Research on low-resource African language technologies.

For production use, Sunbird AI also provides API access through [https://api.sunbird.ai/](https://api.sunbird.ai/).

## Known Limitations

- Performance varies by language depending on the amount and quality of available training data.
- Some lower-resource languages may produce inconsistent or less fluent outputs.
- The model may hallucinate, especially in highly specialized domains or when asked for unsupported factual details.
- It should not be used as the sole decision maker in high-risk domains such as medical, legal, financial, or safety-critical use cases.
- Human review is recommended for sensitive translations and critical applications.
- Outputs may reflect biases present in multilingual web, cultural, and training datasets.

## How to Load / Run

The model can be loaded with the Hugging Face `transformers` library. A high-memory GPU setup is recommended for the full precision model.

```python
import torch
import transformers

MODEL_PATH = "Sunbird/Sunflower-14B"

SYSTEM_MESSAGE = (
    "You are Sunflower, a multilingual assistant made by Sunbird AI "
    "who understands Ugandan languages and English. "
    "You specialise in accurate translations, factual question answering, "
    "summaries, explanations, and multilingual reasoning."
)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

instruction = "Translate from Luganda to English: Wano webawaaba?"

messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": instruction},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.5,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(
    outputs[0][len(inputs["input_ids"][0]):],
    skip_special_tokens=True,
)

print(response)
```

## Training Data

Sunflower-14B was trained on approximately 750 million characters of multilingual text collected from:

- Digitized books and educational materials.
- Radio transcripts, including more than 500 hours of transcribed audio.
- Web data from MADLAD-400 and Common Crawl.
- Existing multilingual datasets, including SALT, FLORES-200, MT560, and TICO-19.
- Dictionaries, proverbs, and cultural documents.

## Training Procedure

Sunflower-14B was developed in multiple stages:

1. Continued pretraining from Qwen3-14B using next-token prediction on multilingual data.
2. Supervised fine-tuning on instruction-response datasets for translation, summarization, question answering, and instruction following.
3. Preference optimization to improve alignment and reduce hallucinations or unstable outputs.
4. GRPO fine-tuning using the `jq/sunflower-14b-grpo-combined` checkpoint to improve factual correctness, multilingual reasoning, answer quality, consistency, and conversational alignment.

## Evaluation

Sunflower-14B was evaluated on a multilingual translation benchmark covering practical domains such as healthcare, agriculture, education, banking, and government services.

Average reported scores across the supported languages:

| Direction                     | Metric | Score |
| ----------------------------- | -----: | ----: |
| English to supported language |   chrF | 0.366 |
| Supported language to English |   chrF | 0.419 |
| Supported language to English |   BLEU | 19.61 |

The GRPO release is reported to improve translation fluency, context preservation, answer consistency, factual reliability, and multilingual question-answering quality.

## Licence

Apache 2.0.

## Hugging Face

[https://huggingface.co/Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B)

## Last Updated

May 23, 2026

## Citation

```bibtex
@misc{sunflower2025,
  title={Sunflower: A Regional Approach to Large Language Models for Ugandan Languages},
  author={Akera, Benjamin and Nafula, Evelyn and Yiga, Gilbert and Natukunda, Phionah and Nsumba, Solomon and Muhanguzi, Joel and Namara, Janat and Sekalala, Imran and Walukagga, Patrick and Bainomugisha, Engineer and Mwebaze, Ernest and Quinn, John},
  year={2025},
  publisher={Sunbird AI}
}
```

## References

- Base model: `Qwen/Qwen3-14B`
- Datasets: `Sunbird/ug40-instructions`, `Sunbird/salt`, `jq/sunflower-14b-grpo-combined`
- Paper: `arXiv:2510.07203`
