# Sunflower-14B-FP4A16

## Overview

Sunflower-14B-FP4A16 is a 4-bit weight-quantized variant of [Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B), Sunbird AI's multilingual language model for Ugandan languages. This variant applies FP4 quantization to model weights while keeping activations at FP16, giving an aggressive reduction in memory footprint for GPU inference while retaining higher-precision activations than fully quantized (e.g. W8A8) variants.

## Task Type

Text Generation (Translation, Q&A, general instruction-following)

## Base Architecture

Qwen3-14B (base: `Qwen/Qwen3-14B-Base` → finetuned `Qwen/Qwen3-14B` → finetuned `Sunbird/Sunflower-14B` → quantized to this model).

## Model Size

9B params (quantized weight footprint, per the safetensors metadata). Quantized from the full 14.8B-parameter Sunflower-14B base model — the reported parameter count is lower in the quantized release because weights are packed at 4-bit precision.

## Quantization Format

**NVFP4A16** — 4-bit floating-point weights (NVFP4) with 16-bit (FP16) activations. Quantized using the `llmcompressor` library (vLLM's quantization toolkit) with an `NVFP4A16` `QuantizationModifier` recipe, applied via one-shot quantization to the base `Sunbird/Sunflower-14B` model. Tensor types in the released safetensors: BF16 · F8_E4M3 · U8.

## Languages Supported

English plus 31 Ugandan languages, including Luganda, Acholi, Ateso, Lugbara, Runyankole, and others. Sunflower-14B achieves the highest translation accuracy of evaluated models in 24 of these 31 language pairs.

## Intended Use

- Translation between English and Ugandan languages, and between Ugandan languages themselves
- General text generation, Q&A, and light creative/instructional tasks in supported languages
- Deployment on GPUs with limited VRAM where the full-precision (BF16/FP16) model won't fit, but where FP8/W8A8 still exceed available memory
- Target users: developers and researchers building translation or language tools for Ugandan-language contexts who need to run inference on smaller/cheaper GPU hardware

## Hardware Requirements

**Recommended:**
- GPU: NVIDIA, Compute Capability 7.0+ (V100, A100, H100, or RTX 30xx/40xx series)
- VRAM: 12–16GB (down from ~28GB needed for the FP16 base model)
- System RAM: 32GB+

**Minimum:**
- GPU: NVIDIA with 8GB VRAM
- System RAM: 16GB+

## Memory & Performance

- Original model size (FP16): ~28GB
- Quantized model size (NVFP4A16): ~7GB
- Memory reduction: ~75%
- Generally faster token generation and lower first-token latency than the FP16 model; exact gains depend on hardware, batch size, and sequence length

## Known Limitations

- Slight quality degradation on some tasks compared to the full-precision model — this is the most aggressive quantization in the family, so benchmark against your use case before production use
- Currently optimized specifically for vLLM inference
- Inherits the base model's general limitations: factual recall can be unreliable, hallucinations may occur, informal/slang language is not handled as well, and some responses may appear in the wrong language
- Best suited for production deployments, real-time/low-latency applications, and memory-constrained GPUs; for tasks requiring maximum accuracy or research/fine-tuning work, consider the full-precision Sunflower-14B instead

## How to Load / Run

Install dependencies:

```bash
pip install vllm llmcompressor
```

Inference with vLLM:

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Load the quantized model
model = LLM("Sunbird/Sunflower-14B-FP4A16", enforce_eager=True)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000
)

# Using chat template (recommended)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate to Luganda: Uganda is a landlocked country in East Africa."}
]

formatted_prompt = model.get_tokenizer().apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = model.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

Alternatively, with Transformers directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Sunbird/Sunflower-14B-FP4A16")
model = AutoModelForCausalLM.from_pretrained("Sunbird/Sunflower-14B-FP4A16")

messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

> **Note:** this model requires accepting the repository's usage conditions on Hugging Face before downloading.

## Licence

Apache 2.0

## Hugging Face

[huggingface.co/Sunbird/Sunflower-14B-FP4A16](https://huggingface.co/Sunbird/Sunflower-14B-FP4A16)

## Last Updated

October 9, 2025

## Citation

```bibtex
@misc{Sunbird_Sunflower_14B_FP4A16,
  title={{Sunflower-14B FP4A16 Quantized Model}},
  author={{Sunbird AI}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\url{https://huggingface.co/Sunbird/Sunflower-14B-FP4A16}}}
}
```

## References

- Base model: [Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B)