
# Sunbird Sunflower Quantized Model Usage
## Installation
````markdown
## Installation

Install the required Python packages:

```bash
!pip install torch transformers accelerate bitsandbytes
````

* **torch**: PyTorch library for model computation.
* **transformers**: Hugging Face Transformers library for model handling.
* **accelerate**: Optimizes model loading across devices.
* **bitsandbytes**: Enables efficient 4-bit and 8-bit model quantization.

---

## Import the necessary libraries for loading and running the model.

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from getpass import getpass
```
## 3. Authentication

```python
# Set your Hugging Face token to access models
os.environ["HF_TOKEN"] = getpass("HF_TOKEN: ")
```

---

## 4. Load Tokenizer and Model

Replace `<MODEL_NAME>` with the specific model you want to use, e.g., `Sunbird/Sunflower-14B-4bit-nf4-bnb` or `Sunbird/Sunflower-14B-8bit-bnb`.

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("<MODEL_NAME>")

# Configure quantization (choose one)
quantization_config = BitsAndBytesConfig(
    # For 4-bit models
    load_in_4bit=True,                  
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    
    # For 8-bit models, comment out above and use:
    # load_in_8bit=True,
    # llm_int8_threshold=6.0
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "<MODEL_NAME>",
    quantization_config=quantization_config,
    device_map="auto",       # or "cuda:0" for single GPU
    trust_remote_code=True
)
```

> Note: Only change `<MODEL_NAME>` and adjust quantization parameters. Everything else remains the same for any model.

---

## 5. Prepare Chat Prompt

```python
# Define system message (assistant role)
SYSTEM_MESSAGE = """You are Sunflower, a multilingual assistant for Ugandan languages made by Sunbird AI. You specialise in accurate translations, explanations, summaries, and other cross-lingual tasks."""

# Example user prompt
prompt_text = "Sunbird AI is a non-profit research organization in Kampala, Uganda. We build and deploy practical applied machine learning systems for African contexts."
user_prompt = f"Translate to Luganda: {prompt_text}"

# Structure messages
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": user_prompt}
]

# Prepare model input
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
```

---

## 6. Configure Text Generation

```python
# Streamer outputs tokens in real-time
text_streamer = transformers.TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generation parameters
num_beams = 5  # Adjust for higher quality
generation_config = {
    "max_new_tokens": 512,       # Maximum tokens to generate
    "temperature": 0.3,          # Lower = more deterministic
    "do_sample": True,           # Enable sampling
    "no_repeat_ngram_size": 5,   # Avoid repeated phrases
    "num_beams": num_beams,      # Beam search width
}
```

* Beam search improves output quality but slows generation.
* `TextStreamer` works only when `num_beams == 1`.

---

## 7. Generate Output

```python
# Generate model output
outputs = model.generate(
    **inputs,
    **generation_config,
    streamer=text_streamer if num_beams == 1 else None,
)

# Decode if using multi-beam search
if num_beams > 1:
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(response)
```

* Single-beam: real-time streaming.
* Multi-beam: output decoded after generation.

---

## 8. Notes for Future Models

* The workflow is **identical for any Sunflower model**.
* Only model-specific changes:

  1. `<MODEL_NAME>` in `from_pretrained`.
  2. Quantization parameters (`load_in_4bit` / `load_in_8bit`).
* Everything else (chat template, streaming, generation) stays the same.

---


