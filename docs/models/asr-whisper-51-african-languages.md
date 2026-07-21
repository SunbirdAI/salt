# ASR Whisper 51 African Languages

## Overview

**ASR Whisper 51 African Languages** is an adaptation of Whisper Large v3 that supports Automatic Speech Recognition (ASR) for 51 languages widely spoken in Africa: English and French with African accents, Swahili, Afrikaans, Tswana, Kinyarwanda, Nigerian Pidgin, Luganda, Acholi, Lugbara, Ateso, Runyankole, Rutooro, Lumasaba, Lusoga, Rukiga, Akan, Amharic, Bambara, Bemba, Berber, Chichewa, Dagaare, Dagbani, Ewe, Fulani, Hausa, Igbo, Ikposo, Kabyle, Kalenjin, Kanuri, Kikuyu, Kwamba, Lendu, Lingala, Luhya, Luo, Malagasy, Ndebele, Oromo, Rukonjo, Ruruuli, Shona, Somali, Sotho, Thur, Wolof, Xhosa, Yoruba, and Zulu.

The model achieves lower Word Error Rate (WER) and Character Error Rate (CER) on most African languages when compared against proprietary and open-source alternatives like Gemini 3.5 Flash, GPT-4o Transcribe, and Meta's Omnilingual ASR (omniASR).

## Task Type

Automatic Speech Recognition (ASR), Audio-to-Text.

## Base Architecture

Whisper Large v3 (fine-tuned from `openai/whisper-large-v3` and `huwenjie333/whisper-v3-ft-af51`).

## Model Size

1.54B parameters.

## Quantization Format

None. This is the full precision model.

## Languages Supported

The model supports ASR for English, French, and the following 49 African languages:

| Code | Language | Code | Language |
|---|---|---|---|
| `ach` | Acholi | `lin` | Lingala |
| `afr` | Afrikaans | `lth` | Thur |
| `aka` | Akan | `lug` | Luganda |
| `amh` | Amharic | `luo` | Luo |
| `bam` | Bambara | `luy` | Luhya |
| `bem` | Bemba | `mlg` | Malagasy |
| `ber` | Berber | `myx` | Lumasaba |
| `cgg` | Rukiga | `nbl` | Ndebele |
| `dag` | Dagbani | `nya` | Chichewa |
| `dga` | Dagaare | `nyn` | Runyankole |
| `eng` | English (African Accented) | `orm` | Oromo |
| `ewe` | Ewe | `pcm` | Nigerian Pidgin |
| `fra` | French (African Accented) | `ruc` | Ruruuli |
| `ful` | Fulani | `rwm` | Kwamba |
| `hau` | Hausa | `sna` | Shona |
| `ibo` | Igbo | `som` | Somali |
| `kab` | Kabyle | `sot` | Sotho |
| `kau` | Kanuri | `swa` | Swahili |
| `kik` | Kikuyu | `teo` | Ateso |
| `kin` | Kinyarwanda | `tsn` | Tswana |
| `kln` | Kalenjin | `ttj` | Rutooro |
| `koo` | Rukonjo | `wol` | Wolof |
| `kpo` | Ikposo | `xho` | Xhosa |
| `led` | Lendu | `xog` | Lusoga |
| `lgg` | Lugbara | `yor` | Yoruba |
| | | `zul` | Zulu |

## Intended Use

This model is intended for Automatic Speech Recognition (ASR) applications across the 51 supported languages. Suitable use cases include:

- Transcribing audio to text in any of the 51 supported languages.
- Voice interface systems for local communities.
- Processing low-resource speech datasets for research and application development.

## Known Limitations

- Speech recognition performance varies depending on the target language, speaker accent, and audio recording quality.
- Accuracy may degrade in noisy environments, with overlapping speakers, or when the spoken language deviates significantly from the training distribution.
- The model might exhibit hallucination or substitution errors, especially for highly low-resource languages.
- Verification and human-in-the-loop validation is recommended for high-stakes domains (medical, legal, financial).

## How to Load / Run

You can try the model directly using this [Colab Notebook](https://colab.research.google.com/drive/1Bs2P3ULkj9zftPItuS2JnjzkFX8cgyz1?usp=sharing), which supports transcribing audio samples from Hugging Face datasets, uploaded audio files, or your microphone.

For better accuracy, specify the language during generation. See the example below, which transcribes an audio sample from the `Sunbird/salt` Hugging Face dataset:

```python
import transformers
import datasets
import torch

SAMPLE_RATE = 16000

LANGUAGE_TOKENS_WHISPER = {
    # Existing languages codes from Whisper
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    # Overwrite unused language tokens
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309
}

LANGUAGE_NAMES = {
    'Acholi': 'ach', 'Afrikaans': 'afr', 'Akan': 'aka', 'Amharic': 'amh', 'Ateso': 'teo',
    'Bambara': 'bam', 'Bemba': 'bem', 'Berber': 'ber', 'Chichewa': 'nya', 'Dagaare': 'dga',
    'Dagbani': 'dag', 'English': 'eng', 'Ewe': 'ewe', 'French': 'fra', 'Fulani': 'ful',
    'Hausa': 'hau', 'Igbo': 'ibo', 'Ikposo': 'kpo', 'Kabyle': 'kab', 'Kalenjin': 'kln',
    'Kanuri': 'kau', 'Kikuyu': 'kik', 'Kinyarwanda': 'kin', 'Kwamba': 'rwm', 'Lendu': 'led',
    'Lingala': 'lin', 'Luganda': 'lug', 'Lugbara': 'lgg', 'Luhya': 'luy', 'Lumasaba': 'myx',
    'Luo': 'luo', 'Lusoga': 'xog', 'Malagasy': 'mlg', 'Ndebele': 'nbl', 'Nigerian Pidgin': 'pcm',
    'Oromo': 'orm', 'Rukiga': 'cgg', 'Rukonjo': 'koo', 'Runyankole': 'nyn', 'Ruruuli': 'ruc',
    'Rutooro': 'ttj', 'Shona': 'sna', 'Somali': 'som', 'Sotho': 'sot', 'Swahili': 'swa',
    'Thur': 'lth', 'Tswana': 'tsn', 'Wolof': 'wol', 'Xhosa': 'xho', 'Yoruba': 'yor', 'Zulu': 'zul'
}

model_id = "Sunbird/asr-whisper-51-african-languages"
model = transformers.WhisperForConditionalGeneration.from_pretrained(model_id)
processor = transformers.WhisperProcessor.from_pretrained(model_id)

def transcribe_by_whisper(audio_array, language):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_features = processor(
        audio_array, sampling_rate=SAMPLE_RATE, do_normalize=True, return_tensors="pt"
    ).input_features.to(device)

    lang_tok = LANGUAGE_TOKENS_WHISPER[LANGUAGE_NAMES[language]]
    transcribe_tok = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_tok = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    forced_decoder_ids = [
        (1, lang_tok),
        (2, transcribe_tok),
        (3, notimestamps_tok),
    ]

    predicted_ids = model.to(device).generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        num_beams=1,
        do_sample=False,
    )
    transcription = processor.decode(
        predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(transcription[0])

# Get some test audio
import huggingface_hub
huggingface_hub.login()

ds = datasets.load_dataset('Sunbird/salt', 'multispeaker-lug', split='test')
audio_column = 'audio'
ds = ds.cast_column(audio_column, datasets.Audio(sampling_rate=SAMPLE_RATE))
audio_array = ds[0][audio_column]['array']

# Specify a language
lang = 'Luganda'

transcribe_by_whisper(audio_array, lang)
# Ekikoola kya kasooli kya kyenvu wabula langi yakyo etera okuba eya kitaka wansi.
```

## Performance Metrics

Our model is designed to excel on low-resource African languages such as Acholi and Ateso from Uganda, while achieving comparable performance on widely spoken African languages like Swahili, and African-accented English/French.

Below is a comparison of Word Error Rate (WER) and Character Error Rate (CER) measured for each language on the evaluation datasets. For `ach`, `eng`, `lgg`, `lug`, `nyn`, and `teo`, the dev splits of the `multispeaker-*` subsets in the [Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt) dataset were used for evaluation. For the remaining languages, we used the first 50 examples from the dev split of each subset in the [Training Datasets](#training-datasets) after filtering out clips longer than 30 seconds.

| Code | Language | WER | CER | n_examples |
|---|---|---|---|---|
| `afr` | Afrikaans | 0.084 | 0.028 | 170 |
| `aka` | Akan | 0.323 | 0.129 | 50 |
| `amh` | Amharic | 0.401 | 0.265 | 250 |
| `bam` | Bambara | 0.361 | 0.170 | 149 |
| `bem` | Bemba | 0.408 | 0.091 | 100 |
| `ber` | Berber | 0.147 | 0.041 | 50 |
| `cgg` | Rukiga | 0.242 | 0.057 | 49 |
| `dag` | Dagbani | 0.321 | 0.109 | 100 |
| `dga` | Dagaare | 0.294 | 0.113 | 50 |
| `ewe` | Ewe | 0.291 | 0.097 | 78 |
| `fra` | French | 0.032 | 0.019 | 50 |
| `ful` | Fulani | 0.395 | 0.115 | 149 |
| `hau` | Hausa | 0.168 | 0.047 | 287 |
| `ibo` | Igbo | 0.340 | 0.117 | 247 |
| `kab` | Kabyle | 0.447 | 0.219 | 50 |
| `kau` | Kanuri | 0.389 | 0.101 | 50 |
| `kik` | Kikuyu | 0.209 | 0.099 | 50 |
| `kin` | Kinyarwanda | 0.307 | 0.084 | 100 |
| `kln` | Kalenjin | 0.471 | 0.112 | 100 |
| `koo` | Rukonjo | 0.527 | 0.093 | 47 |
| `kpo` | Ikposo | 0.617 | 0.231 | 50 |
| `led` | Lendu | 0.273 | 0.093 | 50 |
| `lin` | Lingala | 0.243 | 0.104 | 193 |
| `lth` | Thur | 0.148 | 0.059 | 41 |
| `luo` | Luo | 0.226 | 0.051 | 149 |
| `luy` | Luhya | 0.203 | 0.038 | 50 |
| `mlg` | Malagasy | 0.143 | 0.060 | 49 |
| `nbl` | Ndebele | 0.263 | 0.051 | 10 |
| `nya` | Chichewa | 0.273 | 0.058 | 98 |
| `orm` | Oromo | 0.241 | 0.064 | 119 |
| `pcm` | Nigerian Pidgin | 0.167 | 0.065 | 100 |
| `ruc` | Ruruuli | 0.480 | 0.105 | 38 |
| `rwm` | Kwamba | 0.584 | 0.234 | 49 |
| `sna` | Shona | 0.196 | 0.050 | 147 |
| `som` | Somali | 0.381 | 0.127 | 149 |
| `sot` | Sotho | 0.229 | 0.117 | 94 |
| `swa` | Swahili | 0.087 | 0.022 | 99 |
| `tsn` | Tswana | 0.067 | 0.018 | 98 |
| `ttj` | Rutooro | 0.222 | 0.063 | 49 |
| `wol` | Wolof | 0.319 | 0.122 | 135 |
| `xho` | Xhosa | 0.262 | 0.054 | 126 |
| `yor` | Yoruba | 0.384 | 0.136 | 249 |
| `zul` | Zulu | 0.207 | 0.042 | 143 |
| `ach` | Acholi | 0.197 | 0.044 | 101 |
| `eng` | English | 0.041 | 0.016 | 101 |
| `lgg` | Lugbara | 0.216 | 0.052 | 101 |
| `lug` | Luganda | 0.109 | 0.022 | 103 |
| `nyn` | Runyankole | 0.262 | 0.051 | 103 |
| `teo` | Ateso | 0.275 | 0.068 | 101 |
| `myx` | Lumasaba | 0.347 | 0.071 | 100 |
| `xog` | Lusoga | 0.301 | 0.056 | 100 |

### Visual Comparisons

![model_metrics_comparison](https://cdn-uploads.huggingface.co/production/uploads/68f646665bd4055eaf795dfa/niD_kgFKEAiFW3xB-02iS.png)

*Figure 1: Comparative WER and CER performance against proprietary and open-source models.*

![whisper-v3-ft-af51-0703_wer_vs_train_hours](https://cdn-uploads.huggingface.co/production/uploads/68f646665bd4055eaf795dfa/hlMvKA2oKJBB5ZbvS6QCB.png)

*Figure 2: Word Error Rate (WER) vs. training hours.*

![whisper-v3-ft-af51-0703_cer_vs_train_hours](https://cdn-uploads.huggingface.co/production/uploads/68f646665bd4055eaf795dfa/LqZoYg1L5xOG1qtAXy7Mx.png)

*Figure 3: Character Error Rate (CER) vs. training hours.*

## Training Datasets

The model was fine-tuned on a multilingual corpus covering **51 African languages**, assembled from a range of publicly available and community-collected speech datasets. The table below lists each source dataset and the languages it contributes to training (ISO 639-3 code in parentheses). A language may appear in more than one source.

| Source dataset | Languages |
| --- | --- |
| [Mozilla Common Voice](https://commonvoice.mozilla.org) | Afrikaans (afr), Amharic (amh), Rukiga (cgg), Dagbani (dag), Hausa (hau), Igbo (ibo), Kabyle (kab), Kinyarwanda (kin), Kalenjin (kln), Rukonjo (koo), Lendu (led), Thur (lth), Luganda (lug), Luo (luo), Nigerian Pidgin (pcm), Ruruuli (ruc), Kwamba (rwm), Swahili (swa), Tswana (tsn), Rutooro (ttj), Yoruba (yor) |
| [Google FLEURS](https://huggingface.co/datasets/google/fleurs) | Afrikaans (afr), Fulani (ful), Hausa (hau), Igbo (ibo), Lingala (lin), Luganda (lug), Luo (luo), Chichewa (nya), Oromo (orm), Shona (sna), Somali (som), Sotho (sot), Swahili (swa), Wolof (wol), Xhosa (xho), Yoruba (yor), Zulu (zul) |
| [Google Waxal](https://huggingface.co/datasets/google/WaxalNLP) | Acholi (ach), Akan (aka), Amharic (amh), Dagbani (dag), Dagaare (dga), Ewe (ewe), Fulani (ful), Ikposo (kpo), Lingala (lin), Luganda (lug), Malagasy (mlg), Lumasaba (myx), Runyankole (nyn), Oromo (orm), Shona (sna), Lusoga (xog) |
| [ASR Africa Data Efficiency Benchmark](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark) | Afrikaans (afr), Amharic (amh), Bambara (bam), Bemba (bem), Ewe (ewe), Fulani (ful), Hausa (hau), Igbo (ibo), Kinyarwanda (kin), Luganda (lug), Oromo (orm), Shona (sna), Wolof (wol), Xhosa (xho), Yoruba (yor), Zulu (zul) |
| [African Next Voices](https://huggingface.co/datasets/dsfsi-anv/za-african-next-voices) | Kikuyu (kik), Kalenjin (kln), Luo (luo), Somali (som), Ndebele (nbl), Sotho (sot), Tswana (tsn), Xhosa (xho), Zulu (zul) |
| [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt) | Acholi (ach), English (eng), Lugbara (lgg), Luganda (lug), Runyankole (nyn), Ateso (teo) |
| [African Voices](https://africanvoices.io/download) | Hausa (hau), Igbo (ibo), Nigerian Pidgin (pcm), Yoruba (yor) |
| [NaijaVoices](https://huggingface.co/datasets/naijavoices/naijavoices-dataset) | Hausa (hau), Igbo (ibo), Yoruba (yor) |
| [CLEAR Global](https://huggingface.co/CLEAR-Global) | Hausa (hau), Kanuri (kau) |
| [Shunya Labs](https://huggingface.co/shunyalabs) | Amharic (amh), Lingala (lin) |
| [RobotsMali Bambara ASR](https://huggingface.co/datasets/RobotsMali/bam-asr-early) | Bambara (bam) |
| [AfriSpeech-200 (Intron Health)](https://huggingface.co/datasets/intronhealth/afrispeech-200) | English (eng) |
| [African-Accented French](https://huggingface.co/datasets/gigant/african_accented_french) | French (fra) |
| [Afrikaans-30s](https://huggingface.co/datasets/andreoosthuizen/afrikaans-30s) | Afrikaans (afr) |
| [BembaSpeech](https://huggingface.co/csikasote) | Bemba (bem) |
| [TutlaytAI Amazigh ASR](https://huggingface.co/TutlaytAI) | Berber (ber) |
| [KasuleTrevor Lingala](https://huggingface.co/KasuleTrevor) | Lingala (lin) |
| [Makerere Radio Speech](https://huggingface.co/datasets/allandclive/MakerereRadioSpeech_20Hrs) | Luganda (lug) |
| [Digital Divide Data](https://datacollective.mozillafoundation.org/datasets/cmkh2stj6004mmj0751rn8vq6) | Luhya (luy) |
| [michsethowusu](https://huggingface.co/michsethowusu) | Chichewa (nya) |
| [Soomali ASR](https://huggingface.co/datasets/skydheere/soomali-asr-dataset) | Somali (som) |
| [Kallaama](https://huggingface.co/datasets/yigagilbert/kallaama-trs-dataset-19hr) | Wolof (wol) |
| [KYAGABA](https://huggingface.co/KYAGABA) | Amharic (amh) |

> [!NOTE]
> *African Next Voices* dataset is drawn from two collection hubs: Kenya (Kikuyu, Kalenjin, Luo, Somali) and Southern Africa (Ndebele, Sotho, Tswana, Xhosa, Zulu).

## Licence

Released under the **Apache 2.0** license (matching the base architecture and training set details).

## Hugging Face

[https://huggingface.co/Sunbird/asr-whisper-51-african-languages](https://huggingface.co/Sunbird/asr-whisper-51-african-languages)

## Last Updated

2026

## Citation

```bibtex
@misc{sunbird_asr_whisper_51_african_languages,
  title        = {ASR Whisper 51 African Languages},
  author       = {Sunbird AI},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Sunbird/asr-whisper-51-african-languages}},
}
```
