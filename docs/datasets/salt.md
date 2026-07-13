# SALT (Sunbird African Language Technology) Dataset

`Sunbird/salt`

## Description

SALT is a multi-way parallel text and speech corpus covering English and six languages widely spoken in Uganda and East Africa. The core of the dataset consists of 25,000 sentences spanning topics of local relevance such as agriculture, health, and society. Every sentence is translated into all supported languages, enabling machine translation research, and speech recordings have been produced for approximately 5,000 of these sentences. Recordings were made in two settings: by a variety of speakers in natural conditions (suitable for Automatic Speech Recognition) and by professional speakers in a studio setting (suitable for Text-to-Speech). Because sentence IDs are consistent across subsets, content can be mapped across languages and modalities — for example, linking the Acholi text of a sentence to its Swahili studio recording — which also supports downstream tasks like speech-to-text translation and speech-to-speech translation.

## Languages Covered

| ISO 639-3 | Language                 | Translated Text | Multispeaker Speech | Studio Speech |
| --------- | ------------------------ | --------------- | ------------------- | ------------- |
| eng       | English (Ugandan accent) | Yes             | Yes                 | Yes           |
| lug       | Luganda                  | Yes             | Yes                 | Yes           |
| ach       | Acholi                   | Yes             | Yes                 | Yes           |
| lgg       | Lugbara                  | Yes             | Yes                 | Yes           |
| teo       | Ateso                    | Yes             | Yes                 | Yes           |
| nyn       | Runyankole               | Yes             | Yes                 | Yes           |
| swa       | Swahili                  | Yes             | No                  | Yes           |
| ibo       | Igbo                     | Yes             | No                  | No            |
| ttj       | Rutooro                  | Yes             | No                  | No            |
| xog       | Lusoga                   | Yes             | No                  | No            |

## Size / Examples

- **Core text corpus:** 25,000 parallel sentences, translated into all supported languages
- **Speech recordings:** ~5,000 of the core sentences, recorded in both natural (multispeaker) and studio settings
- **Total dataset size:** 1.74 GB
- **Downloads last month (at time of writing):** 337

## Subsets

| Subset name           | Contents                                                                         |
| --------------------- | -------------------------------------------------------------------------------- |
| `text-all`            | Text translations of each sentence                                               |
| `text-hard`           | Text translations of each sentence (harder subset)                               |
| `multispeaker-{lang}` | Speech recordings of each sentence, by a variety of speakers in natural settings |
| `studio-{lang}`       | Speech recordings in a studio setting, suitable for text-to-speech               |

## Data Collection Methodology

The dataset was collected through a practical collaboration between **Sunbird AI**, the **Makerere University AI Lab** (Ugandan languages), and **KenCorpus, Maseno University** (Swahili). Speech data was gathered via two parallel recording efforts: informal recordings from a variety of speakers in natural settings for ASR use cases, and studio-quality recordings from professional speakers for TTS use cases.

## Intended Use Cases

- Machine translation across the seven core languages (English, Luganda, Acholi, Lugbara, Ateso, Runyankole, Swahili)
- Automatic Speech Recognition (ASR) training and evaluation
- Text-to-Speech (TTS) training and evaluation
- Speech-to-text translation (by combining speech subsets with parallel text subsets)
- Speech-to-speech translation (by combining multispeaker/studio subsets across languages)

## Licence

CC BY-SA 4.0

## Last Updated
2025-05-06

## Citation

Akera, B., Mukiibi, J., Naggayi, L.S., Babirye, C., Owomugisha, I., Nsumba, S., Nakatumba-Nabende, J., Bainomugisha, E., Mwebaze, E., & Quinn, J. (2022). _Machine Translation For African Languages: Community Creation Of Datasets And Models In Uganda._ 3rd Workshop on African Natural Language Processing.

## HuggingFace Link

[huggingface.co/datasets/Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt)
