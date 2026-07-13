# Data Preprocessing with salt.preprocessing

The `salt.preprocessing` module contains utilities for formatting, cleaning, and augmenting text and audio datasets to make speech and translation models more robust in real-world environments.

## Text Preprocessing

### 1. Text Cleaning
The `clean_text` function standardizes capitalization and removes unwanted artifacts from text strings.

```python
from salt.preprocessing import clean_text

example = {"source": "  Noisy Text with !! Punctuation. "}
# Cleans the text and converts to lowercase
cleaned = clean_text(example, key="source", lower=True)
print(cleaned)
# Output: {'source': 'noisy text with punctuation'}
```

### 2. Random Casing (Augmentation)
The `random_case` function randomly applies title case, uppercase, or lowercase to the target sentences. This helps translation models generalize better across varying casing conventions.

```python
from salt.preprocessing import random_case

example = {"target": "Standard sentence representation"}
augmented = random_case(example, key="target")
print(augmented)
```

---

## Audio Preprocessing & Augmentation

### 1. Audio Noise Augmentation
The `augment_audio_noise` function adds environmental noise to raw audio arrays. This simulates real-world conditions like street background noise to improve speech-to-text model robustness.

```python
import numpy as np
from salt.preprocessing import augment_audio_noise

# 1 second of silent audio at 16kHz
audio_data = {
    "source": np.zeros(16000), 
    "source.sample_rate": 16000
}

# Add noise to the audio source array
augmented = augment_audio_noise(audio_data, key="source")
```
