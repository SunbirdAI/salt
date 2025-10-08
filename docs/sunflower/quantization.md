# Converting Sunflower LoRA Fine-tuned Models to GGUF Quantizations

In this guide, provide a tutorial for converting LoRA fine-tuned Sunflower models to GGUF format with multiple quantization levels, including experimental ultra-low bit quantizations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [LoRA Merging](#lora-merging)
- [GGUF Conversion](#gguf-conversion)
- [Quantization Process](#quantization-process)
- [Experimental Quantizations](#experimental-quantizations)
- [Quality Testing](#quality-testing)
- [Ollama Integration](#ollama-integration)
- [Distribution](#distribution)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **RAM**: Minimum 32GB (64GB recommended for 14B+ models)
- **Storage**: 200GB+ free space for intermediate files
- **GPU**: Optional but recommended for faster processing

### Software Requirements

- Linux/macOS (WSL2 for Windows)
- Python 3.9+
- Git and Git LFS
- CUDA toolkit (optional, for GPU acceleration)

## Environment Setup

### 1. Install Dependencies

```bash
# Install Python packages
pip install torch transformers accelerate
pip install sentencepiece protobuf
pip install peft huggingface_hub
pip install -U huggingface_hub

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y cmake build-essential git git-lfs
```

### 2. Clone and Build llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build with CUDA support (if available)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# OR build for CPU only
make clean && make

# Create working directories
mkdir -p models gguf_outputs
cd ..
```

## Model Preparation

### 1. Download Models

```bash
# Download base model
huggingface-cli download jq/sunflower-14b-bs64-lr1e-4 --local-dir models/base_model

# Download LoRA adapter
huggingface-cli download jq/qwen3-14b-sunflower-20250915 --local-dir models/lora_model

# Download merged model (LoRA already merged)
huggingface-cli download Sunbird/qwen3-14b-sunflower-merged --local-dir models/merged_model

```

**Example:**

```bash
huggingface-cli download jq/sunflower-14b-bs64-lr1e-4 --local-dir models/base_model
huggingface-cli download jq/qwen3-14b-sunflower-20250915 --local-dir models/lora_model

# Download merged model (LoRA already merged)
huggingface-cli download Sunbird/qwen3-14b-sunflower-merged --local-dir models/merged_model
```

## LoRA Merging

1. Create Merge Script (Skip if Using Pre-merged Model)
   Note: If you downloaded Sunbird/qwen3-14b-sunflower-merged, skip this section and go directly to GGUF Conversion.
   Create merge_lora.py only if using separate base model and LoRA adapter:

Create `merge_lora.py`:

```python
#!/usr/bin/env python3
"""
Merge LoRA adapter with base model
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import sys

def merge_lora_model():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "models/base_model",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better precision
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "models/lora_model")

    print("Merging LoRA with base model...")
    merged_model = model.merge_and_unload()

    print("Saving merged model...")
    merged_model.save_pretrained("models/merged_model", safe_serialization=True)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("models/base_model")
    tokenizer.save_pretrained("models/merged_model")

    print("Merge completed successfully!")
    print("Merged model saved to: models/merged_model")

if __name__ == "__main__":
    merge_lora_model()
```

### 2. Run Merge Process

```bash
python merge_lora.py
```

**Expected output**: Merged model saved to `models/merged_model`

## GGUF Conversion

### 1. Convert to F16 GGUF

```bash
# Convert merged model to F16 GGUF (preserves quality for quantization)
python3 llama.cpp/convert_hf_to_gguf.py models/merged_model \
  --outfile gguf_outputs/model-merged-f16.gguf \
  --outtype f16

# Verify file size (should be ~2GB per billion parameters for F16)
ls -lh gguf_outputs/model-merged-f16.gguf
```

**Expected size**: ~28GB for 14B model in F16

## Quantization Process

### 1. Generate Importance Matrix

The importance matrix (imatrix) significantly improves quantization quality by identifying which weights are most critical to model performance.

```bash
# Download calibration dataset
wget [https://raw.githubusercontent.com/ggerganov/llama.cpp/master/examples/perplexity/wiki.test.raw](https://huggingface.co/nisten/llama3-8b-instruct-32k-gguf/raw/main/wiki.test.raw)

# Generate importance matrix (this takes 30-60 minutes)
./llama.cpp/build/bin/llama-imatrix \
  -m gguf_outputs/model-merged-f16.gguf \
  -f wiki.test.raw \
  --chunk 512 \
  -o gguf_outputs/model-imatrix.dat \
  -ngl 32 \
  --verbose
```

**Note**: Adjust `-ngl` based on your GPU memory (0 for CPU-only)

> **Note: Understanding Importance Matrix (imatrix)**
>
> The importance matrix is a calibration technique that identifies which model weights contribute most significantly to output quality. During quantization, weights deemed "important" by the matrix receive higher precision allocation, while less critical weights can be more aggressively compressed. This selective approach significantly improves quantized model quality compared to uniform quantization.
>
> The imatrix is generated by running representative text through the model and measuring activation patterns. While general text datasets (like WikiText) work well for most models, using domain-specific calibration data (e.g., translation examples for the Sunflower model) can provide marginal quality improvements. The process adds 30-60 minutes to quantization time but is highly recommended for production models, especially when using aggressive quantizations like Q4_K_M and below.

### 2. Standard Quantizations

Create quantized models with different quality/size trade-offs:

```bash
# Q8_0: Near-lossless quality (~15GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-q8_0.gguf \
  Q8_0

# Q6_K: High quality (~12GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-q6_k.gguf \
  Q6_K

# Q5_K_M: Balanced quality/size (~10GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-q5_k_m.gguf \
  Q5_K_M

# Q4_K_M: Recommended for most users (~8GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-q4_k_m.gguf \
  Q4_K_M
```

### 3. Quantization Options Reference

| Quantization | Bits per Weight | Quality    | Use Case                     |
| ------------ | --------------- | ---------- | ---------------------------- |
| Q8_0         | ~8.0            | Highest    | Production, quality critical |
| Q6_K         | ~6.6            | High       | Production, balanced         |
| Q5_K_M       | ~5.5            | Good       | Most users                   |
| Q4_K_M       | ~4.3            | Acceptable | Resource constrained         |

## Experimental Quantizations

**Warning**: These quantizations achieve extreme compression but may significantly impact model quality.

### Ultra-Low Bit Quantizations

```bash
# IQ2_XXS: Extreme compression (~4GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-iq2_xxs.gguf \
  IQ2_XXS

# TQ1_0: Ternary quantization (~3.7GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-tq1_0.gguf \
  TQ1_0

# IQ1_S: Maximum compression (~3.4GB for 14B model)
./llama.cpp/build/bin/llama-quantize \
  --imatrix gguf_outputs/model-imatrix.dat \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-iq1_s.gguf \
  IQ1_S
```

### Experimental Quantization Reference

| Quantization | Bits per Weight | Compression | Warning Level         |
| ------------ | --------------- | ----------- | --------------------- |
| IQ2_XXS      | 2.06            | 85% smaller | Moderate quality loss |
| TQ1_0        | 1.69            | 87% smaller | High quality loss     |
| IQ1_S        | 1.56            | 88% smaller | Severe quality loss   |

## Quality Testing

### 1. Quick Functionality Test

```bash
# Test standard quantization
./llama.cpp/build/bin/llama-cli \
  -m gguf_outputs/model-q4_k_m.gguf \
  -p "Your test prompt here" \
  -n 100 \
  --verbose

# Test experimental quantization
./llama.cpp/build/bin/llama-cli \
  -m gguf_outputs/model-iq1_s.gguf \
  -p "Your test prompt here" \
  -n 100 \
  --verbose
```

### 2. Perplexity Evaluation

```bash
# Download evaluation dataset
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip

# Test perplexity (lower is better)
./llama.cpp/build/bin/llama-perplexity \
  -m gguf_outputs/model-q4_k_m.gguf \
  -f wikitext-2-raw/wiki.test.raw \
  -ngl 32
```

### 3. Size Verification

```bash
# Check all quantization sizes
ls -lh gguf_outputs/*.gguf
```

**Expected output** (14B model):

```
28G  model-merged-f16.gguf
15G  model-q8_0.gguf
12G  model-q6_k.gguf
10G  model-q5_k_m.gguf
8.4G model-q4_k_m.gguf
4.1G model-iq2_xxs.gguf
3.7G model-tq1_0.gguf
3.4G model-iq1_s.gguf
```

## Ollama Integration

Ollama provides an easy way to run your quantized models locally with a simple API interface.

### Installation and Setup

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai for Windows

# Start Ollama service (runs in background)
ollama serve
```

### Creating Modelfiles for Different Quantizations

**Q4_K_M (Recommended) - Modelfile:**

```bash
cat > Modelfile.q4 << 'EOF'
FROM ./gguf_outputs/model-q4_k_m.gguf

# System prompt for your specific use case
SYSTEM """You are a linguist and translator specializing in Ugandan languages, made by Sunbird AI."""

# Chat template (adjust for your base model architecture)
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 500
EOF
```

**Experimental IQ1_S - Modelfile:**

```bash
cat > Modelfile.iq1s << 'EOF'
FROM ./gguf_outputs/model-iq1_s.gguf

SYSTEM """You are a translator for Ugandan languages. Note: This is an experimental ultra-compressed model - quality may be limited."""

# Same template and parameters as above
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
PARAMETER num_ctx 2048  # Smaller context for experimental model
EOF
```

### Importing Models to Ollama

```bash
# Import Q4_K_M model (recommended)
ollama create sunflower-14b:q4 -f Modelfile.q4

# Import experimental IQ1_S model
ollama create sunflower-14b:iq1s -f Modelfile.iq1s

# Import other quantizations
ollama create sunflower-14b:q5 -f Modelfile.q5
ollama create sunflower-14b:q6 -f Modelfile.q6

# Verify models are imported
ollama list
```

**Expected output:**

```
NAME                    ID              SIZE    MODIFIED
sunflower-14b:q4        abc123def       8.4GB   2 minutes ago
sunflower-14b:iq1s      def456ghi       3.4GB   1 minute ago
```

### Using Ollama Models

**Interactive Chat:**

```bash
# Start interactive session with Q4 model
ollama run sunflower-14b:q4

# Example conversation:
# >>> Translate to Luganda: Hello, how are you today?
# >>> Give a dictionary definition of the Samia term "ovulwaye" in English
# >>> /bye (to exit)

# Start with experimental model
ollama run sunflower-14b:iq1s
```

**Single Prompt Inference:**

```bash
# Quick translation with Q4 model
ollama run sunflower-14b:q4 "Translate to Luganda: People in villages rarely accept new technologies."

# Test experimental model
ollama run sunflower-14b:iq1s "Translate to Luganda: Good morning"

# Dictionary definition
ollama run sunflower-14b:q4 'Give a dictionary definition of the Samia term "ovulwaye" in English'
```

### Ollama API Usage

**Start API Server:**

```bash
# Ollama automatically serves API on http://localhost:11434
# Test API endpoint
curl http://localhost:11434/api/version
```

**Python API Client:**

```python
import requests
import json

def translate_with_ollama(text, target_lang="Luganda", model="sunflower-14b:q4"):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": f"Translate to {target_lang}: {text}",
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json()["response"]

# Test translation
result = translate_with_ollama("Hello, how are you?")
print(result)

# Test experimental model
result_experimental = translate_with_ollama(
    "Good morning",
    model="sunflower-14b:iq1s"
)
print("Experimental model:", result_experimental)
```

**curl API Examples:**

```bash
# Basic translation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sunflower-14b:q4",
    "prompt": "Translate to Luganda: How are you today?",
    "stream": false
  }'

# Streaming response
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sunflower-14b:q4",
    "prompt": "Translate to Luganda: People in villages rarely accept new technologies.",
    "stream": true
  }'
```

### Model Management

```bash
# List all models
ollama list

# Show model details
ollama show sunflower-14b:q4

# Remove a model
ollama rm sunflower-14b:iq1s

# Copy model with new name
ollama cp sunflower-14b:q4 sunflower-translator

# Pull/push to Ollama registry (if you publish there)
# ollama push sunflower-14b:q4
```

### Performance Comparison Script

Create `test_models.py`:

```python
import time
import requests

models = ["sunflower-14b:q4", "sunflower-14b:iq1s"]
test_prompt = "Translate to Luganda: Hello, how are you today?"

def test_model(model_name, prompt):
    start_time = time.time()

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model_name,
        "prompt": prompt,
        "stream": False
    })

    end_time = time.time()
    result = response.json()

    return {
        "model": model_name,
        "response": result["response"],
        "time": end_time - start_time,
        "tokens": len(result["response"].split())
    }

# Test all models
for model in models:
    result = test_model(model, test_prompt)
    print(f"Model: {result['model']}")
    print(f"Response: {result['response']}")
    print(f"Time: {result['time']:.2f}s")
    print(f"Tokens: {result['tokens']}")
    print("-" * 50)
```

### Production Deployment

```bash
# Create production Modelfile with optimized settings
cat > Modelfile.production << 'EOF'
FROM ./gguf_outputs/model-q4_k_m.gguf

SYSTEM """You are a professional translator for Ugandan languages, made by Sunbird AI. Provide accurate, contextually appropriate translations."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Production-optimized parameters
PARAMETER temperature 0.1        # Lower for consistency
PARAMETER top_p 0.9             # Slightly more focused
PARAMETER repeat_penalty 1.05   # Minimal repetition penalty
PARAMETER num_ctx 4096          # Full context
PARAMETER num_predict 200       # Reasonable response length
EOF

# Create production model
ollama create sunflower-translator:production -f Modelfile.production
```

## Distribution

### 1. Hugging Face Upload

Create upload script `upload_models.py`:

```python
#!/usr/bin/env python3
from huggingface_hub import HfApi, login, create_repo

def upload_gguf_models():
    repo_id = "your-org/your-model-GGUF"  # Update this
    local_folder = "gguf_outputs"

    # Login and create repository
    login()
    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload all files
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        allow_patterns=["*.gguf", "*.dat"],
        commit_message="Add GGUF quantized models"
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_gguf_models()
```

```bash
# Login to Hugging Face
huggingface-cli login

# Run upload
python upload_models.py
```

### 2. Ollama Integration (Complete Guide)

#### Installation and Setup

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai for Windows

# Start Ollama service (runs in background)
ollama serve
```

#### Creating Modelfiles for Different Quantizations

**Q4_K_M (Recommended) - Modelfile:**

```bash
cat > Modelfile.q4 << 'EOF'
FROM ./gguf_outputs/model-q4_k_m.gguf

# System prompt for your specific use case
SYSTEM """You are a linguist and translator specializing in Ugandan languages, made by Sunbird AI."""

# Chat template (adjust for your base model architecture)
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 500
EOF
```

**Experimental IQ1_S - Modelfile:**

```bash
cat > Modelfile.iq1s << 'EOF'
FROM ./gguf_outputs/model-iq1_s.gguf

SYSTEM """You are a translator for Ugandan languages. Note: This is an experimental ultra-compressed model - quality may be limited."""

# Same template and parameters as above
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
PARAMETER num_ctx 2048  # Smaller context for experimental model
EOF
```

#### Importing Models to Ollama

```bash
# Import Q4_K_M model (recommended)
ollama create sunflower-14b:q4 -f Modelfile.q4

# Import experimental IQ1_S model
ollama create sunflower-14b:iq1s -f Modelfile.iq1s

# Import other quantizations
ollama create sunflower-14b:q5 -f Modelfile.q5
ollama create sunflower-14b:q6 -f Modelfile.q6

# Verify models are imported
ollama list
```

**Expected output:**

```
NAME                    ID              SIZE    MODIFIED
sunflower-14b:q4        abc123def       8.4GB   2 minutes ago
sunflower-14b:iq1s      def456ghi       3.4GB   1 minute ago
```

#### Using Ollama Models

**Interactive Chat:**

```bash
# Start interactive session with Q4 model
ollama run sunflower-14b:q4

# Example conversation:
# >>> Translate to Luganda: Hello, how are you today?
# >>> Give a dictionary definition of the Samia term "ovulwaye" in English
# >>> /bye (to exit)

# Start with experimental model
ollama run sunflower-14b:iq1s
```

**Single Prompt Inference:**

```bash
# Quick translation with Q4 model
ollama run sunflower-14b:q4 "Translate to Luganda: People in villages rarely accept new technologies."

# Test experimental model
ollama run sunflower-14b:iq1s "Translate to Luganda: Good morning"

# Dictionary definition
ollama run sunflower-14b:q4 'Give a dictionary definition of the Samia term "ovulwaye" in English'
```

#### Ollama API Usage

**Start API Server:**

```bash
# Ollama automatically serves API on http://localhost:11434
# Test API endpoint
curl http://localhost:11434/api/version
```

**Python API Client:**

```python
import requests
import json

def translate_with_ollama(text, target_lang="Luganda", model="sunflower-14b:q4"):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": f"Translate to {target_lang}: {text}",
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json()["response"]

# Test translation
result = translate_with_ollama("Hello, how are you?")
print(result)

# Test experimental model
result_experimental = translate_with_ollama(
    "Good morning",
    model="sunflower-14b:iq1s"
)
print("Experimental model:", result_experimental)
```

**curl API Examples:**

```bash
# Basic translation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sunflower-14b:q4",
    "prompt": "Translate to Luganda: How are you today?",
    "stream": false
  }'

# Streaming response
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sunflower-14b:q4",
    "prompt": "Translate to Luganda: People in villages rarely accept new technologies.",
    "stream": true
  }'
```

#### Model Management

```bash
# List all models
ollama list

# Show model details
ollama show sunflower-14b:q4

# Remove a model
ollama rm sunflower-14b:iq1s

# Copy model with new name
ollama cp sunflower-14b:q4 sunflower-translator

# Pull/push to Ollama registry (if you publish there)
# ollama push sunflower-14b:q4
```

#### Performance Comparison Script

Create `test_models.py`:

```python
import time
import requests

models = ["sunflower-14b:q4", "sunflower-14b:iq1s"]
test_prompt = "Translate to Luganda: Hello, how are you today?"

def test_model(model_name, prompt):
    start_time = time.time()

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model_name,
        "prompt": prompt,
        "stream": False
    })

    end_time = time.time()
    result = response.json()

    return {
        "model": model_name,
        "response": result["response"],
        "time": end_time - start_time,
        "tokens": len(result["response"].split())
    }

# Test all models
for model in models:
    result = test_model(model, test_prompt)
    print(f"Model: {result['model']}")
    print(f"Response: {result['response']}")
    print(f"Time: {result['time']:.2f}s")
    print(f"Tokens: {result['tokens']}")
    print("-" * 50)
```

#### Troubleshooting Ollama

**Common Issues:**

1. **Model fails to load:**

```bash
# Check model file exists
ls -la gguf_outputs/model-q4_k_m.gguf

# Verify Modelfile syntax
ollama create sunflower-14b:q4 -f Modelfile.q4 --verbose
```

2. **Out of memory:**

```bash
# Use smaller quantization
ollama run sunflower-14b:iq1s  # Uses only 3.4GB

# Or reduce context size in Modelfile
PARAMETER num_ctx 2048
```

3. **Poor quality with experimental models:**

```bash
# Compare outputs
ollama run sunflower-14b:q4 "Your test prompt"
ollama run sunflower-14b:iq1s "Your test prompt"

# Expected: IQ1_S may have degraded quality
```

4. **Ollama service not running:**

```bash
# Start service
ollama serve

# Or check if running
ps aux | grep ollama
```

#### Production Deployment

```bash
# Create production Modelfile with optimized settings
cat > Modelfile.production << 'EOF'
FROM ./gguf_outputs/model-q4_k_m.gguf

SYSTEM """You are a professional translator for Ugandan languages, made by Sunbird AI. Provide accurate, contextually appropriate translations."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Production-optimized parameters
PARAMETER temperature 0.1        # Lower for consistency
PARAMETER top_p 0.9             # Slightly more focused
PARAMETER repeat_penalty 1.05   # Minimal repetition penalty
PARAMETER num_ctx 4096          # Full context
PARAMETER num_predict 200       # Reasonable response length
EOF

# Create production model
ollama create sunflower-translator:production -f Modelfile.production
```

## Troubleshooting

### Common Issues

**1. Out of Memory During Merge**

```bash
# Use smaller precision or CPU offloading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**2. GGUF Conversion Fails**

```bash
# Check model architecture compatibility
python3 llama.cpp/convert_hf_to_gguf.py models/merged_model --outfile test.gguf --dry-run
```

**3. Quantization Too Slow**

```bash
# Use fewer threads or disable imatrix for testing
./llama.cpp/build/bin/llama-quantize \
  gguf_outputs/model-merged-f16.gguf \
  gguf_outputs/model-q4_k_m.gguf \
  Q4_K_M \
  4  # Use 4 threads
```

**4. Experimental Quantizations Unusable**

- This is expected for extreme quantizations like IQ1_S
- Test with your specific use case
- Consider using IQ2_XXS as minimum viable quantization

### File Size Expectations

For a 14B parameter model:

- **Merge process**: Requires 2x model size in RAM (~56GB peak)
- **F16 GGUF**: ~28GB final size
- **Quantized models**: 3GB-15GB depending on level
- **Total storage needed**: ~200GB for all quantizations

### Performance Notes

- **Importance matrix generation**: 30-60 minutes on modern hardware
- **Each quantization**: 5-10 minutes per model
- **Upload time**: Varies by connection, large files use Git LFS
- **Memory usage**: Peaks during merge, lower during quantization

### Ollama-Specific Issues

**1. Model fails to load:**

```bash
# Check model file exists
ls -la gguf_outputs/model-q4_k_m.gguf

# Verify Modelfile syntax
ollama create sunflower-14b:q4 -f Modelfile.q4 --verbose
```

**2. Out of memory with Ollama:**

```bash
# Use smaller quantization
ollama run sunflower-14b:iq1s  # Uses only 3.4GB

# Or reduce context size in Modelfile
PARAMETER num_ctx 2048
```

**3. Poor quality with experimental models:**

```bash
# Compare outputs
ollama run sunflower-14b:q4 "Your test prompt"
ollama run sunflower-14b:iq1s "Your test prompt"

# Expected: IQ1_S may have degraded quality
```

**4. Ollama service not running:**

```bash
# Start service
ollama serve

# Or check if running
ps aux | grep ollama
```

**5. API connection issues:**

```bash
# Test API availability
curl http://localhost:11434/api/version

# Check if port is blocked
netstat -tlnp | grep 11434
```

## Conclusion

This tutorial demonstrates the complete pipeline for converting LoRA fine-tuned models to GGUF format with multiple quantization levels. The process enables deployment of large models on resource-constrained hardware while maintaining various quality/size trade-offs.

The experimental ultra-low quantizations (IQ1_S, IQ2_XXS, TQ1_0) push the boundaries of model compression and should be used with appropriate quality expectations.

For production use, Q4_K_M provides the best balance of quality and size, while Q5_K_M and Q6_K offer better quality at larger sizes. Always evaluate quantized models on your specific tasks before deployment.

## References

- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
