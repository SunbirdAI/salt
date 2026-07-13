# Sunflower-14B-FP8

## Overview

Sunflower-14B-FP8 is a quantized version of [Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B), a multilingual language model developed by Sunbird AI for Ugandan languages. Built on the Qwen 3-14B architecture, it supports translation and text generation across 31 Ugandan languages plus English, and achieves the highest translation accuracy among evaluated models in 24 of 31 language pairs. This variant uses **FP8_DYNAMIC** quantization, optimized for efficient inference while maintaining model quality.

## Task Type

Text Generation, Translation

## Base Architecture

Qwen 3-14B

## Model Size

15B parameters (~28GB in FP16, ~14GB quantized)

## Quantization Format

FP8_DYNAMIC — 8-bit floating point quantization with dynamic scaling, applied using the `llmcompressor` framework (vLLM). Gives roughly a 50% reduction in model size versus the full-precision (FP16) model, with minimal accuracy loss.

## Languages Supported

English, plus 31 Ugandan languages, including Luganda, Acholi, and Runyankole.

## Intended Use

Best suited for:
- Production deployments requiring efficient inference
- Real-time applications needing low latency
- Scenarios with limited GPU memory
- Batch processing workloads

For tasks requiring maximum accuracy, research experiments involving fine-tuning, or scenarios where memory isn't a constraint, the full-precision base model (`Sunbird/Sunflower-14B`) is recommended instead.

## Known Limitations

- Slight quality degradation is possible on some tasks compared to the full-precision model, as is typical with quantization.
- Optimal performance requires a compatible NVIDIA GPU.
- Currently optimized specifically for vLLM inference.
- As with the base model: factual recall can be unreliable, hallucinations may occur, performance is weaker on slang/informal language, and some responses may appear in the wrong language.

## Hardware Requirements

**Recommended:**
- GPU: NVIDIA GPU with Compute Capability 7.0+ (V100, A100, H100, or RTX 30xx/40xx series)
- VRAM: 12–16GB (down from 28GB needed for the FP16 model)
- System RAM: 32GB+

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM
- System RAM: 16GB+

## How to Load / Run

Install dependencies:

```bash
pip install vllm llmcompressor
```

Run inference:

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Load the quantized model
model = LLM("Sunbird/Sunflower-14B-FP8", enforce_eager=True)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000
)

# Method 1: Using chat template (recommended)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate to Luganda: Uganda is a landlocked country in East Africa."}
]

formatted_prompt = model.get_tokenizer().apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = model.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)

# Method 2: Direct text generation
prompt = "Explain the importance of biodiversity:"
outputs = model.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Licence

Apache 2.0

## Hugging Face

[huggingface.co/Sunbird/Sunflower-14B-FP8](https://huggingface.co/Sunbird/Sunflower-14B-FP8)

## Last Updated

October 9, 2025

## Citation

```bibtex
@misc{Sunbird_Sunflower_14B_FP8,
    title={{Sunflower-14B FP8 Quantized Model}},
    author={{Sunbird AI}},
    year={{2025}},
    publisher={{HuggingFace}},
    howpublished={{\url{https://huggingface.co/Sunbird/Sunflower-14B-FP8}}}
}
```

## References

- Base model: [Sunbird/Sunflower-14B](https://huggingface.co/Sunbird/Sunflower-14B)
