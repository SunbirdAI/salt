# sb-salt Usage

> All examples use `sb-salt >= 0.1.1`.

`sb-salt` is the pip-installable version of `SunbirdAI/salt`, a toolkit for African language NLP and speech workflows. It provides utilities for loading multilingual Hugging Face datasets, applying text and speech preprocessing, inspecting datasets, and wiring SALT-style datasets into model training pipelines.

## Installation

1. Install the base package:

```bash
pip install sb-salt
```

2. Install with PyTorch support for model training:

```bash
pip install sb-salt[torch]
```

3. Install an exact release:

```bash
pip install sb-salt==0.1.1
```

## Quick Start

1. Load a small Luganda-to-Acholi text sample from `sunbird/salt`:

```python
import yaml
from salt import dataset
config = yaml.safe_load('''huggingface_load: {path: sunbird/salt, name: text-all, split: "train[:10]"}\nsource: {type: text, language: lug}\ntarget: {type: text, language: ach}''')
ds = dataset.create(config)
print(next(iter(ds)))
```

## Dataset Loading

1. Create datasets with `salt.dataset.create()` from a YAML configuration.

2. Use `huggingface_load` to pass arguments directly to `datasets.load_dataset`, including `path`, `name`, and `split`.

3. Specify source and target fields with a `type` and one or more language codes.

4. Supported language codes used by the SALT language-ID examples are:

```text
eng, lug, ach, teo, lgg, nyn
```

5. The language-ID notebook used `random_case`, `augment_characters`, and `clean_text`. In `sb-salt >= 0.1.1`, use `random_capitalise_source_and_target` for the random-case behavior.

6. In `sunbird/salt` `text-all`, English text is stored as `eng_source_text` and `eng_target_text`. `salt.dataset.create()` in `sb-salt 0.1.1` matches columns named `<language>_text`, so the public examples below use a non-English target language such as `lug` or `ach`. Use `eng` with datasets that expose an `eng_text` column.

7. Use this complete YAML pattern for train and validation splits:

```yaml
model_checkpoint: xlm-roberta-base

datasets:
  train:
    huggingface_load:
      - path: sunbird/salt
        name: text-all
        split: train
      - path: Sunbird/external_mt_datasets
        name: ai4d.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: flores200.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: lafand-en-lug-combined.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: lafand-en-luo-combined.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mozilla_110.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_ach.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_lug.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_nyn.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_from-eng-google.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_from-lug-google.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_ach_en_14_3_23.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_en_many_30_3.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_lug_en_14_3_23.parquet
        split: train
    source:
      type: text
      language: [ach, lgg, lug, nyn, teo, eng]
      preprocessing:
        - random_capitalise_source_and_target:
            p: 0.005
        - augment_characters:
            action: swap
            aug_char_p: 0.05
    target:
      type: text
      language: [lug]
      preprocessing:
        - clean_text
    shuffle: true

  validation:
    huggingface_load:
      path: sunbird/salt
      name: text-all
      split: dev
    source:
      type: text
      language: [ach, lgg, lug, nyn, teo, eng]
    target:
      type: text
      language: [lug]
```

8. Load both splits:

```python
import yaml
from salt import dataset

config = yaml.safe_load(open("language_id.yaml", "r", encoding="utf-8"))

train_dataset = dataset.create(config["datasets"]["train"])
validation_dataset = dataset.create(config["datasets"]["validation"])
```

## Preprocessing

1. `random_case` in the older notebook corresponds to `random_capitalise_source_and_target` in `sb-salt >= 0.1.1`. It randomly uppercases both `source` and `target`; set `p=1.0` for deterministic testing.

```python
from salt import preprocessing

record = {"source": ["hello kampala"], "target": ["gyebale ko"]}
result = preprocessing.random_capitalise_source_and_target(record, "source", p=1.0)
print(result["source"])
```

Example output:

```text
['HELLO KAMPALA']
```

2. `augment_characters` applies character-level augmentation using `nlpaug`.

```python
from salt import preprocessing

record = {"source": ["source text"]}
result = preprocessing.augment_characters(
    record,
    "source",
    action="swap",
    aug_char_p=1.0,
    aug_word_p=1.0,
)
print(result["source"])
```

Example output:

```text
['socure txte']
```

3. `clean_text` normalizes text with the `clean-text` package while preserving case by default.

```python
from salt import preprocessing

record = {"source": ["\\u2018Hello\\u2019 &lt;"]}
result = preprocessing.clean_text(record, "source")
print(result["source"])
```

Example output:

```text
["'Hello' <"]
```

## Utilities

1. Use `salt.utils.show_dataset()` to inspect examples in a notebook or IPython environment.

```python
from salt import dataset
from salt.utils import show_dataset

config = {
    "huggingface_load": {"path": "sunbird/salt", "name": "text-all", "split": "dev[:5]"},
    "source": {"type": "text", "language": ["lug", "ach", "eng"]},
    "target": {"type": "text", "language": ["lug"]},
}

ds = dataset.create(config)
show_dataset(ds, N=5)
```

Example output:

```text
source                 target                 source.language  target.language
Gyebale ko             Gyebale ko             lug              lug
Apwoyo matek           Gyebale ko             ach              lug
Enkuyege zifuuse...    Enkuyege zifuuse...    lug              lug
```

## Metrics

1. `salt.metrics.multilingual_eval()` and `salt.metrics.multilingual_eval_fn()` are available for multilingual generation evaluation with BLEU and WER-style metric objects.

2. The language-identification notebook performs classification evaluation with a Hugging Face `Trainer` `compute_metrics` callback. Use this NumPy implementation to compute weighted precision, recall, F1, and accuracy without adding another runtime dependency:

```python
import numpy as np

def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = np.unique(np.concatenate([labels, predictions]))

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for class_id in classes:
        true_positive = np.sum((predictions == class_id) & (labels == class_id))
        false_positive = np.sum((predictions == class_id) & (labels != class_id))
        false_negative = np.sum((predictions != class_id) & (labels == class_id))
        support = np.sum(labels == class_id)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

    supports = np.asarray(supports, dtype=float)
    weights = supports / supports.sum() if supports.sum() else np.ones_like(supports) / len(supports)

    return {
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(np.sum(weights * np.asarray(precisions))),
        "recall": float(np.sum(weights * np.asarray(recalls))),
        "f1": float(np.sum(weights * np.asarray(f1_scores))),
    }
```

## Full Worked Example: Language Identification

1. Install the package with PyTorch support:

```bash
pip install sb-salt[torch]==0.1.1
```

2. Train an XLM-RoBERTa language-identification classifier:

```python
import random
import yaml
import numpy as np
from salt import dataset
from salt.utils import show_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

yaml_config = """
model_checkpoint: xlm-roberta-base

datasets:
  train:
    huggingface_load:
      - path: sunbird/salt
        name: text-all
        split: train
      - path: Sunbird/external_mt_datasets
        name: ai4d.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: flores200.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: lafand-en-lug-combined.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: lafand-en-luo-combined.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mozilla_110.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_ach.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_lug.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: mt560_nyn.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_from-eng-google.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_from-lug-google.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_ach_en_14_3_23.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_en_many_30_3.parquet
        split: train
      - path: Sunbird/external_mt_datasets
        name: bt_lug_en_14_3_23.parquet
        split: train
    source:
      type: text
      language: [ach, lgg, lug, nyn, teo, eng]
      preprocessing:
        - random_capitalise_source_and_target:
            p: 0.005
        - augment_characters:
            action: swap
            aug_char_p: 0.05
    target:
      type: text
      language: [lug]
      preprocessing:
        - clean_text
    shuffle: true

  validation:
    huggingface_load:
      path: sunbird/salt
      name: text-all
      split: dev
    source:
      type: text
      language: [ach, lgg, lug, nyn, teo, eng]
    target:
      type: text
      language: [lug]
"""

config = yaml.safe_load(yaml_config)

train_dataset = dataset.create(config["datasets"]["train"])
validation_dataset = dataset.create(config["datasets"]["validation"])

label2id = {"eng": 0, "lug": 1, "ach": 2, "teo": 3, "lgg": 4, "nyn": 5}
id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
model_config = AutoConfig.from_pretrained(
    config["model_checkpoint"],
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
model = AutoModelForSequenceClassification.from_pretrained(
    config["model_checkpoint"],
    config=model_config,
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def random_subset(text, min_length=10):
    min_length = min(min_length, len(text))
    subset_length = random.randint(min_length, len(text))
    start = random.randint(0, len(text) - subset_length)
    return text[start : start + subset_length]

def prepare_language_id_examples(examples):
    examples["text"] = [random_subset(text.lower()) for text in examples["source"]]
    examples["label"] = [label2id[language] for language in examples["source.language"]]
    return examples

def tokenize_examples(examples, max_length=64):
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    tokenized["labels"] = examples["label"]
    return tokenized

def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = np.unique(np.concatenate([labels, predictions]))

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for class_id in classes:
        true_positive = np.sum((predictions == class_id) & (labels == class_id))
        false_positive = np.sum((predictions == class_id) & (labels != class_id))
        false_negative = np.sum((predictions != class_id) & (labels == class_id))
        support = np.sum(labels == class_id)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

    supports = np.asarray(supports, dtype=float)
    weights = supports / supports.sum() if supports.sum() else np.ones_like(supports) / len(supports)

    return {
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(np.sum(weights * np.asarray(precisions))),
        "recall": float(np.sum(weights * np.asarray(recalls))),
        "f1": float(np.sum(weights * np.asarray(f1_scores))),
    }

train_dataset = train_dataset.map(
    prepare_language_id_examples,
    batched=True,
    remove_columns=["source", "source.language", "target", "target.language"],
)
validation_dataset = validation_dataset.map(
    prepare_language_id_examples,
    batched=True,
    remove_columns=["source", "source.language", "target", "target.language"],
)

show_dataset(validation_dataset, N=5)

train_dataset = train_dataset.map(
    tokenize_examples,
    batched=True,
    remove_columns=["text", "label"],
)
validation_dataset = validation_dataset.map(
    tokenize_examples,
    batched=True,
    remove_columns=["text", "label"],
)

training_args = TrainingArguments(
    output_dir="training_output/language_id_xlm_roberta",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    max_steps=1000,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    report_to="none",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_classification_metrics,
)

trainer.train()
results = trainer.evaluate()
print(results)
```

## Available Datasets

1. `sunbird/salt`

| Dataset | Config | Splits |
| --- | --- | --- |
| `sunbird/salt` | `text-all` | `train`, `dev`, `test` |

2. `Sunbird/external_mt_datasets`

Use these names as the `name` field in `huggingface_load`:

```text
ai4d.parquet
flores200.parquet
lafand-en-lug-combined.parquet
lafand-en-luo-combined.parquet
mozilla_110.parquet
mt560_ach.parquet
mt560_lug.parquet
mt560_nyn.parquet
bt_from-eng-google.parquet
bt_from-lug-google.parquet
bt_ach_en_14_3_23.parquet
bt_en_many_30_3.parquet
bt_lug_en_14_3_23.parquet
```

## Supported Languages

| Language | Code | Family | Region |
| --- | --- | --- | --- |
| English | `eng` | Indo-European, Germanic | Uganda and global |
| Luganda | `lug` | Niger-Congo, Bantu | Central Uganda |
| Acholi | `ach` | Nilo-Saharan, Western Nilotic | Northern Uganda |
| Ateso | `teo` | Nilo-Saharan, Eastern Nilotic | Eastern Uganda |
| Lugbara | `lgg` | Nilo-Saharan, Central Sudanic | Northwestern Uganda |
| Runyankole | `nyn` | Niger-Congo, Bantu | Western Uganda |
