# NLLB-Based Translation Model Training Documentation

## Overview
This documentation describes the training process and configuration for a multilingual translation model based on Facebook's NLLB-200-1.3B architecture. The model supports bidirectional translation between English and several African languages: Acholi, Lugbara, Luganda, Runyankole, and Ateso.

## Model Architecture
- Base Model: facebook/nllb-200-1.3B
- Model Type: M2M100ForConditionalGeneration
- Tokenizer: NllbTokenizer
- Special Adaptations: Custom token mappings for African languages not in the original NLLB vocabulary

## Supported Languages
| ISO 693-3 | Language Name |
|-----------|---------------|
| eng       | English       |
| ach       | Acholi        |
| lgg       | Lugbara       |
| lug       | Luganda       |
| nyn       | Runyankole    |
| teo       | Ateso         |

## Training Data
The model was trained on a diverse collection of datasets:

### Primary Dataset
- SALT dataset (Sunbird/salt)

### Additional External Resources
1. AI4D dataset
2. FLORES-200
3. LAFAND-MT (English-Luganda and English-Luo combinations)
4. Mozilla Common Voice 110
5. MT560 (Acholi, Luganda, Runyankole variants)
6. Back-translated data:
   - Google Translate based back-translations
   - Language-specific back-translations (Acholi-English, Luganda-English)

## Training Configuration

### Hardware Requirements
- CUDA-capable GPU (recommended)
- Sufficient RAM for large batch processing

### Key Training Parameters
```yaml
Effective Batch Size: 3000
Training Batch Size: 25
Evaluation Batch Size: 25
Gradient Accumulation Steps: 120
Learning Rate: 3.0e-4
Optimizer: Adafactor
Weight Decay: 0.01
Maximum Steps: 1500
FP16 Training: Enabled
```

### Data Preprocessing
The training pipeline includes several preprocessing steps:
1. Text cleaning
2. Target sentence format matching
3. Random case augmentation
4. Character augmentation
5. Dataset-specific tagging (MT560: '<mt560>', Backtranslation: '<bt>')

## Model Training Process

### Setup
1. Install required dependencies:
```bash
pip install peft transformers datasets bitsandbytes accelerate sentencepiece sacremoses wandb
```

2. Initialize the tokenizer with custom language mappings:
```python
tokenizer = transformers.NllbTokenizer.from_pretrained(
    'facebook/nllb-200-distilled-1.3B',
    src_lang='eng_Latn',
    tgt_lang='eng_Latn')
```

3. Configure language code mappings:
```python
code_mapping = {
    'eng': 'eng_Latn',
    'lug': 'lug_Latn',
    'ach': 'luo_Latn',
    'nyn': 'ace_Latn',
    'teo': 'afr_Latn',
    'lgg': 'aka_Latn'
}
```

### Training Process
1. Data Collation
   - Uses DataCollatorForSeq2Seq
   - Handles language code insertion
   - Manages padding and truncation

2. Evaluation Strategy
   - Evaluation every 100 steps
   - Model checkpointing every 100 steps
   - Early stopping with patience of 4
   - BLEU score monitoring

## Evaluation Results

### BLEU Scores on Development Set
| Source | Target | BLEU Score |
|--------|--------|------------|
| ach    | eng    | 28.371     |
| lgg    | eng    | 30.450     |
| lug    | eng    | 41.978     |
| nyn    | eng    | 32.296     |
| teo    | eng    | 30.422     |
| eng    | ach    | 20.972     |
| eng    | lgg    | 22.362     |
| eng    | lug    | 30.359     |
| eng    | nyn    | 15.305     |
| eng    | teo    | 21.391     |

## Usage Example

```python
import transformers
import torch

def translate_text(text, source_language, target_language):
    tokenizer = transformers.NllbTokenizer.from_pretrained(
        'Sunbird/translate-nllb-1.3b-salt')
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(
        'Sunbird/translate-nllb-1.3b-salt')

    language_tokens = {
        'eng': 256047,
        'ach': 256111,
        'lgg': 256008,
        'lug': 256110,
        'nyn': 256002,
        'teo': 256006,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    inputs['input_ids'][0][0] = language_tokens[source_language]
    
    translated_tokens = model.to(device).generate(
        **inputs,
        forced_bos_token_id=language_tokens[target_language],
        max_length=100,
        num_beams=5,
    )

    return tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True)[0]
```

## Model Limitations and Considerations
1. Performance varies significantly between language pairs
2. Best results achieved when English is involved (either source or target)
3. Performance may degrade for:
   - Very long sentences
   - Domain-specific terminology
   - Informal or colloquial language

## Future Improvements
1. Expand training data for lower-performing language pairs
2. Implement more robust data augmentation techniques
3. Explore domain adaptation for specific use cases
4. Fine-tune model size and architecture for optimal performance

## References
- NLLB-200 Paper: [No Language Left Behind](https://arxiv.org/abs/2207.04672)
- SALT Dataset: [Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt)
- Model Weights: [Sunbird/translate-nllb-1.3b-salt](https://huggingface.co/Sunbird/translate-nllb-1.3b-salt)
