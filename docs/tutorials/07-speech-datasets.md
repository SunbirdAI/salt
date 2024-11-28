# Sunbird African Language Technology (SALT) dataset

SALT is a multi-way parallel text and speech corpus of Engish and six languages widely spoken in Uganda and East Africa: `Luganda`, `Lugbara`, `Acholi`, `Runyankole`, `Ateso` and `Swahili`.
The core of the dataset is a set of `25,000` sentences covering a range of topics of local relevance, such as agriculture, health and society.
Each sentence is translated into all languages, to support machine translation, and speech recordings are made for approximately `5,000` of the sentences both by a variety of speakers in natural settings (suitable for ASR) and by professionals in a studio setting (suitable for text-to-speech).

## Subsets

| Subset name           | Contents                                                                          |
| --------------------- | --------------------------------------------------------------------------------- |
| text-all              | Text translations of each sentence.                                               |
| multispeaker-`{lang}` | Speech recordings of each sentence, by a variety of speakers in natural settings. |
| studio-`{lang}`       | Speech recordings in a studio setting, suitable for text-to-speech.               |

The sentence IDs map across subsets, so that for example the text of a sentence in Acholi can be mapped to the studio recording of that concept being expressed in Swahili.
The subsets can therefore be combined to support the training and evaluation of several further tasks, such as speech-to-text translation and speech-to-speech translation.

## Language support

| ISO 639-3 | Language                 | Translated text | Multispeaker speech | Studio speech |
| --------- | ------------------------ | --------------- | ------------------- | ------------- |
| eng       | English (Ugandan accent) | Yes             | Yes                 | Yes           |
| lug       | Luganda                  | Yes             | Yes                 | Yes           |
| ach       | Acholi                   | Yes             | Yes                 | Yes           |
| lgg       | Lugbara                  | Yes             | Yes                 | Yes           |
| teo       | Ateso                    | Yes             | Yes                 | Yes           |
| nyn       | Runyankole               | Yes             | Yes                 | Yes           |
| swa       | Swahili                  | Yes             | No                  | Yes           |
| ibo       | Igbo                     | Yes             | No                  | No            |

## Helper utilities

Code for convenient experimentation with multilingual models can be found at [https://github.com/SunbirdAI/salt](https://github.com/SunbirdAI/salt).
See example notebooks [here](https://github.com/SunbirdAI/salt/tree/main/notebooks).

## Collaborators

This dataset was collected in practical collaboration between Sunbird AI and the Makerere University AI Lab (Ugandan languages) and KenCorpus, Maseno University (Swahili).

## Reference

[Machine Translation For African Languages: Community Creation Of Datasets And Models In Uganda](https://openreview.net/pdf?id=BK-z5qzEU-9). Benjamin Akera, Jonathan Mukiibi, Lydia Sanyu Naggayi, Claire Babirye, Isaac Owomugisha, Solomon Nsumba, Joyce Nakatumba-Nabende, Engineer Bainomugisha, Ernest Mwebaze, John Quinn. 3rd Workshop on African Natural Language Processing, 2022.
