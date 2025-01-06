**Documentation for the LEB Repository**

---

# Introduction

The LEB repository provides tools and resources for working with Sunbird African Language Technology (SALT) datasets. This repository facilitates the creation of multilingual datasets, the training and evaluation of multilingual models, and data preprocessing. It includes robust utilities for model training using HuggingFace frameworks, making it a valuable resource for machine translation and natural language processing (NLP) in underrepresented languages.

## Key Features

- Multilingual dataset handling and preprocessing.
- Metrics for evaluating NLP models.
- Utilities for training HuggingFace models.
- Jupyter notebooks demonstrating use cases.
- Documentation support via MkDocs.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or above.
- Git for cloning the repository.
- pip for managing Python packages.

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/jqug/leb.git
   cd leb
   ```
   Later this will just be 'pip install leb'.

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation by running tests:

   ```bash
   pytest
   ```

---

## Getting Started

### Loading a Dataset

To load a dataset using the tools provided in `dataset.py`:

```python
from leb.dataset import create

config = {
    'huggingface_load': [
        {
            'path': 'mozilla-foundation/common_voice',
            'name': 'en'
        }
    ],
    'source': {
        'language': 'en',
        'type': 'text',
        'preprocessing': ['clean_text']
    },
    'target': {
        'language': 'fr',
        'type': 'text',
        'preprocessing': ['clean_text']
    },
    'shuffle': True
}

dataset = create(config)
print(f"Dataset created with {len(dataset)} examples.")
```

### Preprocessing Data

Use `preprocessing.py` for operations like cleaning, augmentation, and formatting:

```python
from leb.preprocessing import clean_text, random_case

raw_data = {"source": "Hello, WORLD!", "target": "Bonjour, MONDE!"}
cleaned_data = clean_text(raw_data, "source", lower=True)
augmented_data = random_case(cleaned_data, "target")
print(augmented_data)
```

### Training a Model

Leverage HuggingFace training utilities:

```python
from leb.utils import TrainableM2MForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

model = TrainableM2MForConditionalGeneration.from_pretrained(
    "facebook/nllb-200-distilled-1.3B")

training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    evaluation_strategy="steps",
    save_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

## Modules Overview

### `dataset.py`

#### Purpose

Handles dataset loading, validation, and conversion tasks.

#### Key Functions

- **`create(config)`**:
  - Generates a dataset based on the provided configuration.
  - Example usage:
    ```python
    dataset = create(config)
    ```

### `preprocessing.py`

#### Purpose

Provides tools for cleaning and formatting text and audio data.

#### Key Functions

- **`clean_text`**:
  - Cleans text by removing noise and standardizing formatting.
  - Example usage:
    ```python
    clean_text({"source": "Noisy DATA!!"}, "source")
    ```
- **`random_case`**:
  - Randomizes casing to simulate realistic variability in text data.

- **`augment_audio_noise`**:
  - Adds controlled noise to audio samples for robustness.

### `metrics.py`

#### Purpose

Defines evaluation metrics for NLP tasks.

#### Key Functions

- **`multilingual_eval`**:
  - Computes BLEU and other metrics for multilingual tasks.
  - Example usage:
    ```python
    results = multilingual_eval(preds, "en", "fr", [metric_bleu], tokenizer)
    ```

### `utils.py`

#### Purpose

Provides utilities for model training, evaluation, and debugging.

#### Key Classes and Functions

- **`TrainableM2MForConditionalGeneration`**:
  - Customizes training for multilingual translation models.
  - Example usage:
    ```python
    model = TrainableM2MForConditionalGeneration.from_pretrained(checkpoint)
    ```

- **`ForcedVariableBOSTokenLogitsProcessor`**:
  - Allows dynamic BOS token adjustments.

---

## Advanced Usage Examples

### Example 1: Custom Audio Augmentation

```python
from leb.preprocessing import augment_audio_noise

audio_data = {"source": np.zeros(16000), "source.sample_rate": 16000}
augmented_audio = augment_audio_noise(audio_data, "source")
```

### Example 2: Evaluation with Multiple Metrics

```python
from leb.metrics import multilingual_eval_fn

metrics = [evaluate.load("sacrebleu")]
eval_fn = multilingual_eval_fn(eval_dataset, metrics, tokenizer)
results = eval_fn(predictions)
```
