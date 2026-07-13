# Sunflower-14B-W8A8

## Overview

Sunflower-14B-W8A8 is a quantized version of `Sunbird/Sunflower-14B` optimized for efficient inference. It uses W8A8 quantization, which applies 8-bit integer quantization to both model weights and activations, reducing memory usage while preserving most of the capabilities of the full precision Sunflower-14B model.

The model is intended for faster and more memory-efficient multilingual text generation, translation, and question answering across 31 Ugandan languages plus English.


## Task Type

Text Generation, Machine Translation, Multilingual Question Answering, Summarization, Instruction Following.

## Base Architecture

Qwen3-14B through the base model `Sunbird/Sunflower-14B`.

## Model Size

15B parameters.

## Quantization Format

W8A8.

This model uses 8-bit integer quantization for both weights and activations. According to the model card, the quantization provides approximately a 50% reduction in model size compared with the original FP16 model:

| Model Version             | Approximate Size |
| ------------------------- | ---------------: |
| Full precision FP16 model |             28GB |
| W8A8 quantized model      |             14GB |

The model was quantized using `llmcompressor` from the vLLM ecosystem and is optimized for vLLM inference.

## Languages Supported

Sunflower-14B-W8A8 inherits language support from `Sunbird/Sunflower-14B`. It supports English and 31 Ugandan or regional African languages. The base model card lists the supported languages using the following language codes:

`ach`, `adh`, `alz`, `bfa`, `cgg`, `en`, `gwr`, `kdi`, `kdj`, `keo`, `kin`, `koo`, `kpz`, `laj`, `lgg`, `lsm`, `luc`, `lug`, `mhi`, `myx`, `nuj`, `nyn`, `nyo`, `pok`, `rub`, `ruc`, `rwm`, `swa`, `teo`, `tlj`, `ttj`, `xog`.

## Intended Use

Sunflower-14B-W8A8 is intended for users who need the capabilities of Sunflower-14B with lower GPU memory requirements and faster inference. It is best suited for:

- Production deployments requiring efficient multilingual inference.
- Translation between English and Ugandan languages.
- Translation between Ugandan languages.
- Multilingual text generation.
- Multilingual factual question answering.
- Real-time applications where latency matters.
- Batch processing workloads.
- Deployments where full precision Sunflower-14B is too large for the available GPU memory.

For maximum accuracy, fine-tuning, or research settings where memory is not a constraint, the full precision `Sunbird/Sunflower-14B` model may be more appropriate.

## Known Limitations

- Quantization may introduce slight quality degradation compared with the full precision model.
- Performance may vary depending on the language, task, prompt style, sequence length, and available training data for each language.
- Optimal performance requires compatible NVIDIA GPUs with INT8 acceleration.
- The model is currently optimized for vLLM inference.
- It may still hallucinate or produce incorrect outputs, especially in specialized domains.
- Human review is recommended for sensitive translations and high-risk use cases such as medical, legal, financial, or safety-critical applications.

## Hardware Requirements

Recommended hardware:

- NVIDIA GPU with compute capability 7.0 or higher, such as V100, A100, H100, or RTX 30xx/40xx series.
- 12GB to 16GB VRAM.
- 32GB or more system RAM.

Minimum hardware:

- NVIDIA GPU with 8GB VRAM.
- 16GB system RAM.

Actual performance depends on hardware, batch size, sequence length, and deployment configuration.

## How to Load / Run

Install the required dependencies:

```bash
pip install vllm llmcompressor
```

Run inference with vLLM:

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

MODEL_PATH = "Sunbird/Sunflower-14B-W8A8"

model = LLM(MODEL_PATH, enforce_eager=True)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000,
)

messages = [
    {"role": "system", "content": "You are a helpful multilingual assistant."},
    {
        "role": "user",
        "content": "Translate to Luganda: Uganda is a landlocked country in East Africa.",
    },
]

formatted_prompt = model.get_tokenizer().apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = model.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

Direct text generation is also supported:

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

MODEL_PATH = "Sunbird/Sunflower-14B-W8A8"

model = LLM(MODEL_PATH, enforce_eager=True)
sampling_params = SamplingParams(max_tokens=500, temperature=0.7, top_p=0.9)

prompt = "Explain the importance of biodiversity:"
outputs = model.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)
```

## Quantization Process

The model card states that this model was quantized from `Sunbird/Sunflower-14B` using `llmcompressor`.

Conceptually, the quantization uses a W8A8 scheme:

```yaml
quant_scheme: W8A8
quant_modifiers:
  QuantizationModifier:
    scheme: W8A8
```

Example quantization command:

```python
from llmcompressor.transformers import oneshot

recipe = """
quant_scheme: W8A8
quant_modifiers:
  QuantizationModifier:
    scheme: W8A8
"""

oneshot(
    model="Sunbird/Sunflower-14B",
    recipe=recipe,
    output_dir="Sunbird/Sunflower-14B-W8A8",
)
```

## Performance

Compared with the original FP16 model, the W8A8 version is designed to provide:

- Lower GPU memory usage.
- Faster token generation.
- Lower first-token latency.
- Higher throughput for batch inference.

The model card reports an approximate memory reduction from 28GB for the FP16 model to 14GB for the W8A8 quantized model.

## Licence

Apache 2.0.

## Hugging Face

[https://huggingface.co/Sunbird/Sunflower-14B-W8A8](https://huggingface.co/Sunbird/Sunflower-14B-W8A8)

## Last Updated

October 9, 2025

## Citation

```bibtex
@misc{Sunbird_Sunflower_14B_W8A8,
  title={{Sunflower-14B W8A8 Quantized Model}},
  author={{Sunbird AI}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\url{https://huggingface.co/Sunbird/Sunflower-14B-W8A8}}}
}
```

## References

- Base model: `Sunbird/Sunflower-14B`
- Quantization tool: `llmcompressor`
- Inference framework: `vLLM`
