
# Sunflower-14B-GGUF

## Overview

Sunflower-14B-GGUF is a set of GGUF-quantized versions of [Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B), a multilingual causal language model built on Qwen3-14B and fine-tuned by Sunbird AI for translation and text generation across Ugandan languages. The GGUF format converts the model for CPU-friendly, low-resource inference via `llama.cpp` or `Ollama`, trading some precision for the ability to run without a dedicated GPU. Six recommended quantizations and three experimental (extreme-compression) quantizations are provided.

## Task Type

Text Generation (LLM) — Translation, conversational / instruction-following, question answering.

## Base Architecture

Qwen3-14B (base model: `Qwen/Qwen3-14B-Base` → fine-tuned as `Sunbird/Sunflower-14B` → quantized to GGUF).

## Model Size

15B parameters (per HF sidebar). Note the naming convention calls it "14B" — this mismatch exists in the model name itself, not a documentation error; keep the "14B" name but state 15B params explicitly so users aren't confused when they see model size = 15B in the file browser.

## Quantization Formats Available

| Filename | Quant Type | File Size | Notes |
|---|---|---|---|
| `sunflower-14B-f16.gguf` | F16 | 28 GB | Original precision |
| `sunflower-14B-q8_0.gguf` | Q8_0 | 15 GB | Highest quality quantized |
| `sunflower-14B-q6_k.gguf` | Q6_K | 12 GB | High quality |
| `sunflower-14B-q5_k_m.gguf` | Q5_K_M | 9.8 GB | Balanced quality/size |
| `sunflower-14B-q5_k_s.gguf` | Q5_K_S | 9.6 GB | Smaller Q5 variant |
| `sunflower-14B-q4_k_m.gguf` | Q4_K_M | 8.4 GB | **Recommended for most users** |

**Experimental — research/testing only, quality may be significantly degraded for translation tasks:**

| Filename | Quant Type | File Size | Compression | Warning |
|---|---|---|---|---|
| `sunflower-14B-iq2_xxs.gguf` | IQ2_XXS | 4.1 GB | ~85% smaller | May lose translation accuracy |
| `sunflower-14B-tq1_0.gguf` | TQ1_0 | 3.7 GB | ~87% smaller | Experimental ternary quantization |
| `sunflower-14B-iq1_s.gguf` | IQ1_S | 3.4 GB | ~88% smaller | Extreme compression, quality heavily impacted |

Also included: `sunflower-imatrix.dat` — the importance matrix used during quantization (calibration data, not a runnable model file).

Quantization was performed using `llama.cpp` with importance-matrix calibration to preserve quality where possible.

## Languages Supported

English, Luganda, and other Ugandan languages. (Per the base model card, Sunflower-14B was trained across **31 Ugandan languages + English**; the GGUF quantizations inherit the same language coverage as the base model — no retraining occurs during quantization.)

## Intended Use

*(Inherited from base model — quantization does not change intended use, only deployment footprint.)*

- Translation between English and Ugandan languages
- Translation between Ugandan languages
- Text generation in Ugandan languages
- Question answering in Ugandan languages
- Specifically: **local / offline / low-resource deployment** where a full GPU isn't available (this is the GGUF format's specific value-add over the base safetensors model)

## Known Limitations

- Performance varies by language depending on training data availability for that language
- Limited evaluation beyond translation and basic Q&A
- May reflect biases present in training data (including archaic language forms from historical source texts)
- Not suitable for critical applications (medical, legal) without human review
- **Quantization-specific:** the three experimental formats (IQ1_S, IQ2_XXS, TQ1_0) are explicitly flagged by Sunbird as unreliable for translation accuracy — they should not be recommended for production use, only for research/experimentation on constrained hardware. Users should test experimental quantizations against known-good translations before relying on them.

## How to Load / Run

### llama.cpp
```bash
# Download model
huggingface-cli download Sunbird/Sunflower-14B-GGUF sunflower-14B-q4_k_m.gguf --local-dir .

# Run inference
./llama-cli -m sunflower-14B-q4_k_m.gguf -p "Translate to Luganda: Hello, how are you today?"
```

### Ollama
```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

Create a `Modelfile` (recommended Q4_K_M example):
```
FROM ./gguf_outputs/model-q4_k_m.gguf

SYSTEM """You are a linguist and translator specializing in Ugandan languages, made by Sunbird AI."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 500
```

Then:
```bash
ollama create sunflower-14b:q4 -f Modelfile.q4
ollama run sunflower-14b:q4 "Translate to Luganda: How are you?"
```

Ollama also exposes a local REST API at `http://localhost:11434` once `ollama serve` is running.

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(model_path="sunflower-14B-q4_k_m.gguf")
result = llm("Translate to Luganda: How are you?")
print(result['choices'][0]['text'])
```

## Performance Notes

| Quant | Recommendation |
|---|---|
| Q4_K_M | Recommended default for most use cases |
| Q5_K_M | Better quality, moderate size increase |
| Q6_K | High quality, suitable for production |
| Q8_0 | Near-lossless quality |
| IQ1_S / IQ2_XXS / TQ1_0 | Research/experimental only — do not use in production |

## Licence

Apache 2.0

## Hugging Face

[huggingface.co/Sunbird/Sunflower-14B-GGUF](https://huggingface.co/Sunbird/Sunflower-14B-GGUF)

## Last Updated

October 8, 2025

## Citation

```bibtex
@misc{sunflower2025,
  title={Sunflower: A Regional Approach to Large Language Models for Ugandan Languages},
  author={Akera, Benjamin and Nafula, Evelyn and Yiga, Gilbert and Natukunda, Phionah and Nsumba, Solomon and Muhanguzi, Joel and Namara, Janat and Sekalala, Imran and Walukagga, Patrick and Bainomugisha, Engineer and Mwebaze, Ernest and Quinn, John},
  year={2025},
  publisher={Sunbird AI}
}
```


