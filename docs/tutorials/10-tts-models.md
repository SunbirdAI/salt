# Spark-TTS Inference Guide

This documentation provides a comprehensive guide to using the Spark-TTS model for text-to-speech (TTS) inference. Spark-TTS is a multilingual TTS system capable of generating speech in various East African languages and accents, such as Acholi, Ateso, Runyankore, Lugbara, Swahili, and Luganda, using predefined speaker IDs. The model is based on a customized version hosted on Hugging Face and relies on an audio tokenizer from the original Spark-TTS repository.

The guide is derived from the provided example script (`spark_tts_inference_example_new.py`) and Jupyter notebook (`spark_tts_inference_example_new.ipynb`), which demonstrate setup, model loading, and speech generation. This is intended for developers, researchers, or users interested in TTS applications, particularly for low-resource languages.

## Prerequisites

- **Python Environment**: Python 3.8+ (tested with 3.12.3 in the examples).
- **Hardware**: GPU recommended (CUDA-enabled for faster inference). The examples use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Hugging Face Account**: Required for downloading models. You'll need to log in via `huggingface_hub.notebook_login()` or set up an access token.
- **Dependencies**: Install the required libraries using pip. No internet access is needed during inference after downloads, but initial setup requires it.

## Installation

1. **Install Dependencies**:
   Run the following command to install the necessary packages:

   ```
   pip install -qU xformers transformers unsloth omegaconf einx einops soundfile librosa torch torchaudio
   ```

   These include:
   - `transformers` and `unsloth` for model handling.
   - `soundfile` and `librosa` for audio processing.
   - `torch` and `torchaudio` for tensor operations and audio transforms.
   - Others for configuration and utilities.

   Note: If you encounter dependency conflicts (e.g., with `pyarrow`), resolve them based on your environment (e.g., downgrade if needed).

2. **Clone the Spark-TTS Repository**:
   The repository provides the audio tokenizer and utility code. Clone it from GitHub:

   ```
   git clone https://github.com/SparkAudio/Spark-TTS
   ```

   Add it to your Python path:

   ```python
   import sys
   sys.path.append('Spark-TTS')
   ```

3. **Download the Models from Hugging Face**:
   The models are hosted on Hugging Face Hub. Download them as follows:

   - **Audio Tokenizer**: From the repository `unsloth/Spark-TTS-0.5B`. This is the BiCodecTokenizer for encoding/decoding audio tokens.
     - Download URL: [https://huggingface.co/unsloth/Spark-TTS-0.5B](https://huggingface.co/unsloth/Spark-TTS-0.5B)
     - Use `snapshot_download` to fetch only the tokenizer parts (ignore LLM files):

       ```python
       from huggingface_hub import snapshot_download
       snapshot_download(
           "unsloth/Spark-TTS-0.5B", 
           local_dir="Spark-TTS-0.5B",
           ignore_patterns=["*LLM*"]
       )
       ```

   - **Customized TTS Model**: From the repository `jq/spark-tts-salt`. This is the core language model for TTS generation.
     - Download URL: [https://huggingface.co/jq/spark-tts-salt](https://huggingface.co/jq/spark-tts-salt)
     - Load it directly with `transformers`:

       ```python
       import transformers
       model = transformers.AutoModelForCausalLM.from_pretrained(
           "jq/spark-tts-salt",
           device_map='auto',
           torch_dtype="auto"
       )
       tokenizer = transformers.AutoTokenizer.from_pretrained("jq/spark-tts-salt")
       ```

   Before downloading, log in to Hugging Face:

   ```python
   import huggingface_hub
   huggingface_hub.notebook_login()  # Or use CLI: huggingface-cli login
   ```

   These models are open-source under permissive licenses (check the repository pages for details, e.g., Apache 2.0 or similar).

## Usage

### Importing Required Modules

Import the necessary libraries and utilities:

```python
import re
import numpy as np
import torch
import time
from IPython.display import Audio, display  # For Jupyter; optional in scripts
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize  # Optional for normalization
```

Load the audio tokenizer:

```python
audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")  # Or "cpu" if no GPU
```

### Speaker IDs

The model uses speaker IDs prefixed to the text input to select voices/languages. Available IDs from the examples:

- 241: Acholi (female)
- 242: Ateso (female)
- 243: Runyankore (female)
- 245: Lugbara (female)
- 246: Swahili (male)
- 248: Luganda (female)

Prefix the ID to your text, e.g., "248: Hello" for Luganda female voice.

### Generating Speech

Use the provided function to generate speech from text. It handles prompt formatting, token generation, extraction of semantic/global tokens, and decoding to waveform.

```python
@torch.inference_mode()
def generate_speech_from_text(
    text: str,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_audio_tokens: int = 2048,  # Limits audio length; increase for longer text
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> np.ndarray:
    """
    Generates speech audio from text using default voice control parameters.

    Args:
        text (str): The text input to be converted to speech. Prefix with speaker ID, e.g., "248: Hello".
        temperature (float): Sampling temperature for generation (higher = more diverse).
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        max_new_audio_tokens (int): Max number of new tokens to generate (limits audio length).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Generated waveform as a NumPy array (sample rate: 16,000 Hz).
    """
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    # Extract semantic token IDs
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        raise ValueError("No semantic tokens found in the generated output.")
    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)

    # Extract global token IDs
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    if not global_matches:
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
    else:
        pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)

    # Decode to waveform
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device),
        pred_semantic_ids.to(device)
    )
    return wav_np
```

### Examples

1. **Short Text**:
   ```python
   short_input_text = "248: Hello"
   print("Generating short text...")
   start_time = time.time()
   wav_np = generate_speech_from_text(short_input_text)
   print(f"Generation took {time.time() - start_time:.2f} seconds")
   display(Audio(wav_np, rate=16000))  # In Jupyter; save to file otherwise
   ```

2. **Long Text (English)**:
   ```python
   long_input_text = "248: Uganda is named after the Buganda kingdom, which encompasses a large portion of the south, including Kampala, and whose language Luganda is widely spoken"
   wav_np = generate_speech_from_text(long_input_text)
   display(Audio(wav_np, rate=16000))
   ```

3. **Long Text (Luganda)**:
   ```python
   long_lug_text = "248: Awo eizaalibwa gwe olasimba kulamusizzaako ne nkwebaza gyonna gy'otuuseeko. ..."  # Full text from examples
   wav_np = generate_speech_from_text(long_lug_text)
   display(Audio(wav_np, rate=16000))
   ```

To save audio to a file:
```python
import soundfile as sf
sf.write("output.wav", wav_np, 16000)
```

## Performance Notes

- **Inference Time**: Short texts take ~1-2 seconds; longer texts (200+ words) may take 10-30 seconds on GPU.
- **Limitations**: 
  - Max audio length is controlled by `max_new_audio_tokens`. Increase for longer outputs, but it may increase memory usage.
  - If no tokens are extracted, check input formatting or model outputs.
  - Audio is at 16 kHz sample rate.
- **Customization**: Adjust `temperature`, `top_k`, `top_p` for varied outputs. For volume normalization, use `audio_volume_normalize(wav_np)`.

## Troubleshooting

- **CUDA Errors**: Ensure CUDA is installed and compatible with PyTorch.
- **Model Download Issues**: Verify Hugging Face login and repository access.
- **No Output Tokens**: Ensure text is prefixed with a valid speaker ID.
- **Dependencies**: If conflicts arise, use a virtual environment (e.g., via `venv`).

## Resources

- Audio Tokenizer: [https://huggingface.co/unsloth/Spark-TTS-0.5B](https://huggingface.co/unsloth/Spark-TTS-0.5B)
- Customized Model: [https://huggingface.co/jq/spark-tts-salt](https://huggingface.co/jq/spark-tts-salt)
- Example Colab: The original notebook is from [https://colab.research.google.com/drive/1cUrPBJeGj7nRadAtQ4Fx8-5uOa2YYCE7?usp=sharing#scrollTo=be345faf-4a72-42f5-8413-5e4a808d47f3](https://colab.research.google.com/drive/1cUrPBJeGj7nRadAtQ4Fx8-5uOa2YYCE7?usp=sharing#scrollTo=be345faf-4a72-42f5-8413-5e4a808d47f3)

For contributions or issues, refer to the [https://github.com/SunbirdAI/salt](GitHub repo). This model supports open-source TTS research for underrepresented languages.