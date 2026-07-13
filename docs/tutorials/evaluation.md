# Evaluation Metrics with salt.metrics

The `salt.metrics` module provides standard evaluation wrapper functions to measure model translation quality across multiple African languages.

## `multilingual_eval` Function

The `multilingual_eval` function evaluates model predictions against reference datasets. It computes SacreBLEU and other metrics, automatically grouping score outputs by source and target languages.

### Example Usage

```python
import evaluate
from transformers import AutoTokenizer
from salt.metrics import multilingual_eval_fn

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

# Load standard SacreBLEU evaluation metric from Hugging Face
metrics = [evaluate.load("sacrebleu")]

# Create a validation function mapping over our evaluation dataset
eval_fn = multilingual_eval_fn(
    eval_dataset=eval_dataset, # Loaded using salt.dataset.create
    metrics=metrics, 
    tokenizer=tokenizer
)

# Run evaluation on generated predictions
results = eval_fn(predictions)
print(results)
```

---

## Output Metrics Format

The output returned is a dictionary of translation quality scores grouped by language mapping:

```json
{
  "eval_ach_to_eng_sacrebleu": 28.371,
  "eval_lgg_to_eng_sacrebleu": 30.450,
  "eval_lug_to_eng_sacrebleu": 41.978,
  "eval_nyn_to_eng_sacrebleu": 32.296,
  "eval_teo_to_eng_sacrebleu": 30.422
}
```
