# Model Training with salt.utils

The `salt.utils` module contains helper utilities and classes for fine-tuning sequence-to-sequence neural machine translation (NMT) architectures, such as NLLB-200.

## TrainableM2MForConditionalGeneration

The class `TrainableM2MForConditionalGeneration` is a specialized subclass of Hugging Face's `M2M100ForConditionalGeneration`. It optimizes and prepares models for conditional generation tasks using African languages.

### Example Training Pipeline

The following example demonstrates how to initialize the training configuration, model, tokenizer, and kick off training using `Seq2SeqTrainer`.

```python
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from salt.utils import TrainableM2MForConditionalGeneration

# Initialize model from Hugging Face path
model_checkpoint = "facebook/nllb-Distilled-1.3B"
model = TrainableM2MForConditionalGeneration.from_pretrained(model_checkpoint)

# Define Hugging Face training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Setup Seq2SeqTrainer with your datasets
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Loaded using salt.dataset.create
    eval_dataset=eval_dataset,
)

# Run the training process
trainer.train()
```

---

## Logging Processors & Target Tokens

`salt.utils` also includes processors for managing special tokens during inference or training:

- **`ForcedVariableBOSTokenLogitsProcessor`**: Adjusts the Beginning-Of-Sequence (BOS) token distribution dynamically depending on the target language, enabling easier multi-task language modeling across different dialects.
