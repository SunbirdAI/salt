
## Whisper large for Ugandan languages

This model is an adaptation of whisper-large-v2 for the following languages widely spoken in Uganda:
Luganda, Acholi, Lugbara, Ateso, Runyankole and English (Ugandan accent).

### Training

The model was trained with the SALT dataset, Common Voice (Luganda) and FLEURS datasets.
To help with generalisation in practical settings, training used addition of random noise
and random downsampling to 8kHz to simulate phone speech.

### Usage

The model is used in a similar way to the base Whisper model.
The model will attempt to auto-detect the language and provide a transcription. 
However, note that language detection is not always accurate and results may be
improved by specifying it instead. The languages in this model are not supported
by the base Whisper model, so the format is slightly different:


```python
import transformers
import datasets
import torch

processor = transformers.WhisperProcessor.from_pretrained(
    "Sunbird/asr-whisper-large-v2-salt")
model = transformers.WhisperForConditionalGeneration.from_pretrained(
    "Sunbird/asr-whisper-large-v2-salt")

SALT_LANGUAGE_TOKENS_WHISPER = {
    'eng': 50259,  # English (Ugandan)
    'ach': 50357,  # Acholi
    'lgg': 50356,  # Lugbara
    'lug': 50355,  # Luganda
    'nyn': 50354,  # Runyankole
    'teo': 50353,  # Ateso
}

# Get some test audio
ds = datasets.load_dataset('Sunbird/salt', 'multispeaker-lug', split='test')
audio = ds[0]['audio']
sample_rate = ds[0]['sample_rate']

# Specify a language from one of the above.
lang = 'lug'

# Apply the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_features = processor(
    audio, sampling_rate=sample_rate, return_tensors="pt").input_features
input_features = input_features.to(device)
predicted_ids = model.to(device).generate(
    input_features,
    # Optionally set language=None here instead to auto-detect.
    language=processor.tokenizer.decode(SALT_LANGUAGE_TOKENS_WHISPER[lang]),
    forced_decoder_ids=None)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)
# Ekikoola kya kasooli kya kyenvu wabula langi yaakyo etera okuba eya kitaka wansi.
```

## MMS speech recognition for Ugandan languages

This is a fine-tuned version of [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)
for Ugandan languages, trained with the [SALT](https://huggingface.co/datasets/Sunbird/salt) dataset. The languages supported are:

| code | language |
| --- | --- |
| lug | Luganda |
| ach | Acholi |
| lgg | Lugbara |
| teo | Ateso |
| nyn | Runyankole |
| eng | English (Ugandan) |

For each  language there are two adapters: one optimised for cases where the speech is only in that language,
and another in which code-switching with English is expected.

### Usage

Usage is the same as the base model, though with different adapters available.

```python
import torch
import transformers
import datasets

# Available adapters:
# ['lug', 'lug+eng', 'ach', 'ach+eng', 'lgg', 'lgg+eng',
#  'nyn', 'nyn+eng', 'teo', 'teo+eng']
language = 'lug'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = transformers.Wav2Vec2ForCTC.from_pretrained(
    'Sunbird/asr-mms-salt').to(device)
model.load_adapter(language)

processor = transformers.Wav2Vec2Processor.from_pretrained(
    'Sunbird/asr-mms-salt')
processor.tokenizer.set_target_lang(language)

# Get some test audio
ds = datasets.load_dataset('Sunbird/salt', 'multispeaker-lug', split='test')
audio = ds[0]['audio']
sample_rate = ds[0]['sample_rate']

# Apply the model
inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs.to(device)).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

print(transcription)
# ekikola ky'akasooli kyakyenvu wabula langi yakyo etera okuba eyaakitaka wansi
```

The output of this model is unpunctuated and lower case. For applications requiring formatted text, an alternative model is [Sunbird/asr-whisper-large-v2-salt](https://huggingface.co/Sunbird/asr-whisper-large-v2-salt).