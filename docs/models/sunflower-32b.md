
# Sunflower-32B

## Overview

Sunflower-32B is a multilingual causal language model developed by Sunbird AI, built on Qwen3-32B and fine-tuned for translation and text generation across 31 Ugandan languages plus English. It is the flagship, full-precision (BF16) model in the Sunflower series and achieves the strongest translation accuracy of any evaluated model — including Gemini 2.5 Pro and GPT-4o — on Sunbird's internal benchmark.

## Task Type

Text Generation (LLM) — Translation, question answering, summarization, conversational.

## Base Architecture

Qwen3-32B (base model: `Qwen/Qwen3-32B` → fine-tuned as `Sunbird/Sunflower-32B`).

## Model Size

33B parameters, BF16 tensor type.

## Quantization Format

None (full precision / base model). Seven quantized derivatives of this model exist on HuggingFace (llama.cpp, LM Studio, Jan, Ollama compatible) — these should be documented as separate pages, following the same pattern as Sunflower-14B-GGUF.

## Languages Supported

31 Ugandan languages + English.

## Intended Use

- Translation between English and Ugandan languages
- Translation between Ugandan languages
- Text generation in Ugandan languages
- Question answering in Ugandan languages

Also available via the Sunbird AI API at [api.sunbird.ai](https://api.sunbird.ai/).

## Known Limitations

- Performance varies across languages depending on training data availability
- Limited evaluation beyond translation and basic Q&A
- May reflect biases present in training data
- Not suitable for critical applications (medical, legal) without human oversight
- Works best on text similar to the training distribution
- Training data includes historical texts that may contain archaic or outdated language forms; data filtering aimed to remove harmful content but cannot guarantee the complete absence of bias. Outputs should be reviewed by target-language speakers for critical applications.

## Access Note — ⚠️ Gated Model

Sunflower-32B is a **gated repository**. Users must log in to HuggingFace and accept the sharing conditions before they can download or load the model. A plain `from_pretrained()` call will fail without prior authentication (`huggingface-cli login` or an access token).

## How to Load / Run

### Transformers (pipeline)
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="Sunbird/Sunflower-32B")
messages = [{"role": "user", "content": "Who are you?"}]
pipe(messages)
```

### Transformers (direct, with recommended generation settings)
```python
import transformers
import torch

MODEL_PATH = 'Sunbird/Sunflower-32B'
SYSTEM_MESSAGE = 'You are Sunflower, a multilingual assistant made by Sunbird AI who understands all Ugandan languages. You specialise in accurate translations, explanations, summaries and other cross-lingual tasks.'

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

instruction = "Translate from Luganda to English: Wano webawaaba?"

messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": instruction}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    num_beams=5,
    do_sample=True,
    temperature=0.5,
)

response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(response)
```

### vLLM (serving)
```bash
pip install vllm
vllm serve "Sunbird/Sunflower-32B"

# Call via OpenAI-compatible API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "Sunbird/Sunflower-32B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### SGLang (serving)
```bash
pip install sglang
python3 -m sglang.launch_server \
    --model-path "Sunbird/Sunflower-32B" \
    --host 0.0.0.0 \
    --port 30000
```

### Docker Model Runner
```bash
docker model run hf.co/Sunbird/Sunflower-32B
```

> **Note:** At 33B parameters in BF16, this model requires a substantial GPU or multi-GPU setup to run at full precision. For local or low-resource deployment, consider one of the quantized variants instead.

## Training Details

**Training data (~750M characters):**
- Digitized books and educational materials
- Radio transcripts (500+ hours transcribed)
- Web data from MADLAD-400 and Common Crawl
- Existing multilingual datasets (SALT, FLORES-200, MT560, TICO-19)
- Dictionaries, proverbs, and cultural documents

**Training procedure — 3 stages:**
1. **Continued pretraining** — base Qwen3-32B, ~6 hours on 4× H200 GPUs, next-token prediction, DeepSpeed ZeRO-3, batch size 32,768 tokens, LR 1e-4
2. **Supervised fine-tuning** — ~700 instruction-response pairs, LoRA (rank 16, alpha 16), loss computed only on response tokens
3. **Preference optimization** — Iterative Reasoning Preference Optimization (RPO), targeting reduced hallucination/glitching, alpha 1.0

## Evaluation

Evaluated on a custom set of 100 sentences across 20 practical scenarios (healthcare, banking, education, agriculture, etc.) spanning all 31 languages.

| Metric | Score |
|---|---|
| chrF (xx→eng) | 0.435 |
| chrF (eng→xx) | 0.357 |
| BLEU (xx→eng) | 20.625 |
| BLEU (eng→xx) | 7.598 |

**Comparison with other models:**

| Model | chrF (xx→eng) | chrF (eng→xx) |
|---|---|---|
| **Sunflower-32B** | **0.435** | **0.357** |
| Gemini 2.5 Pro | 0.408 | 0.301 |
| GPT-4o | 0.354 | 0.235 |

⚠️ **Note:** the model card's Model Description states Sunflower-32B has the highest accuracy in **24 of 31** language pairs, while the Evaluation section states **25 of 31**. Both figures appear verbatim on the same HF page.

## Licence

Apache 2.0

## Hugging Face

[huggingface.co/Sunbird/Sunflower-32B](https://huggingface.co/Sunbird/Sunflower-32B)

**Training dataset:** [huggingface.co/datasets/Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt)

## Last Updated

October 2025

## Citation


```bibtex
@misc{sunflower2025,
  title={Sunflower: A New Approach To Expanding Coverage of African Languages in Large Language Models},
  author={Akera, Benjamin and Nafula, Evelyn and Yiga, Gilbert and Natukunda, Phionah and Nsumba, Solomon and Muhanguzi, Joel and Namara, Janat and Sekalala, Imran and Walukagga, Patrick and Bainomugisha, Engineer and Mwebaze, Ernest and Quinn, John},
  year={2025},
  publisher={Sunbird AI}
}
```

**Contact:** info@sunbird.ai
