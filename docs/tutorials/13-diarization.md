# Speaker Diarization

Speaker Diarization is the process of partitioning an audio stream into homogeneous segments according to the identity of the speaker. It answers the question "who spoke when?" in a given audio or video recording. This is a crucial step in many speech processing applications, such as transcription, speaker recognition, and meeting analysis.

Speaker Diarization at Sunbird is performed using pyannote's speaker-diarization-3.0 as the main tool for identifying speakers and the text that corresponds to them along with the Sunbird mms that aids in transcription of the text in the audio.

## Framework

**Setup and Installation**

The necessary libraries to perform speaker diarization required for efficient execution of the pipeline and determine various metrics are installed and imported.

```python
!pip install pyctcdecode
!pip install kenlm
!pip install jiwer
!pip install huggingface-hub
!pip install transformers
!pip install pandas
!pip install pyannote.audio
!pip install onnxruntime


import torch
from huggingface_hub import hf_hub_download
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    AutomaticSpeechRecognitionPipeline,
    AutoProcessor
)
from pyctcdecode import build_ctcdecoder
from jiwer import wer

import os
import csv
import pandas as pd
```

**Loading Models and LM Heads**

The Sunbird mms is a huggingface repository with a variety of models and adapters optimized for transcription and translation of languages. Currently, the Diarization developed caters for three languages English, Luganda and Acholi.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lang_config = {
    "ach": "Sunbird/sunbird-mms",
    "lug": "Sunbird/sunbird-mms",
    "eng": "Sunbird/sunbird-mms",
}
model_id = "Sunbird/sunbird-mms"
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
```

#### Processor Setup

```python
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
```

#### Tokenizer setup

```python 
tokenizer.set_target_lang("eng")
model.load_adapter("eng_meta")
```

#### Feature extractor setup

```python
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
```

#### Language model file setup
Within the `Sunbird/sunbird-mms` huggingface repository is a subfolder named `language_model` containing various language models capable of efficient transcription.

```python
lm_file_name = "eng_3gram.bin"
lm_file_subfolder = "language_model"
lm_file = hf_hub_download(
    repo_id=lang_config["eng"],
    filename=lm_file_name,
    subfolder=lm_file_subfolder,
)
```

#### Decoder setup -> Use KenLM as decoder

```python
decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=lm_file,
)
```

#### Use the lm as the Processor

```python
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder,
)
feature_extractor._set_processor_class("Wav2Vec2ProcessorWithLM")
```

#### ASR Pipeline, with a chunk and stride

The ASR pipeline is initialized with the pretrained `Sunbird-mms` model, `processor_with_lm` attributes `tokenizer`, `feature_extractor` and `decoder`, respective device, `chunch_length_s`, `stride_length_s` and `return_timestamps`

```python
pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor_with_lm.tokenizer,    feature_extractor=processor_with_lm.feature_extractor,
    decoder=processor_with_lm.decoder,
    device=device,
    chunk_length_s=10,
    stride_length_s=(4, 2),
    return_timestamps="word"
)
```

**Performing a transcription**
 ```python
 transcription = pipe("/content/kitakas_eng.mp3")
 ```

 The resulting dictionary `transcription` will contain a `text` key containing all the transcribed text as well as a `chunks` containing individual texts along with their time stamps of the format below:

 ```python
 {
    'text' : 'Hello world',
    'chunks': [
        {'text': 'Hello','timestamp': (0.5, 1.0)},
        {'text': 'world','timestamp': (1.5, 2.0)}
        ]
}
```

#### Diarization

**Imports**

```python
from typing import Optional, Union
import numpy as np
from pyannote.audio import Pipeline
import librosa
```

**Loading an audio file**
```python
SAMPLE_RATE = 16000

def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    
    try:
        # librosa automatically resamples to the given sample rate (if necessary)
        # and converts the signal to mono (by averaging channels)
        audio, _ = librosa.load(file, sr=sr, mono=True, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with librosa: {e}") from e

    return audio
```
The `load_audio` functions takes an audio file and sampling rate as one of its parameters. The sampling rate used for this Speaker Diarization is 16000. This sampling rate should be the same sampling rate used to transcribe the audio from using the Sunbird mms to ensure consistency with the output.

**Diarization Pipeline**

The class `Diarization Pipeline` is a custom class created to facilitate the diarization task. It initializes with a pretrained model and can be called with an audio file or waveform to perform diarization.

It returns a pandas DataFrame with with columns for the segment, label, speaker, start time, and end time of each speaker segment.


```python
class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.0",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name,
        use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> pd.DataFrame:

        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df
```

**Segment**

A class to represent a single segment of an audio with start time, end time and speaker label.

This class to encapsulates the information about a segment of audio that has been identified during a speaker diarization process, including the time the segment starts, when it ends, and which speaker is speaking.

```python
class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
```

**Assigning Speakers**

This is the process that involves taking the transcribed chunks and assigning them to the speakers discovered by the Speaker Diarization Pipeline.

In this function, timestamps of the different chunks are compared against the start and end times of speakers in the DataFrame returned by the `SpeakerDiarization` pipeline segments of a transcript are assigned speaker labels based on the overlap between the speech segments and diarization data.

The function iterates through segments of a transcript and assigns the speaker labels based on the overlap between the speech segments and the diarization data.

In case of no overlap, a the fill_nearest parameter can be set to `True`, then the function will assign the speakers to segments by finding the closest speaker in time.

The function takes parameters:
    
`diarize_df`: a pandas DataFrame returned by the DiarizationPipeline containing the diarization information with columns like `start`, `end` and `speaker`

`transcript_result`: A dictionary with a key `chunks` that contains a list of trancript `Segments` obtained from the ASR pipeline.

`fill_nearest`: Default is `False`

`Returns:` An updated `transcript_result` with speakers assigned to each segment in the form:

```python
{
    'text':'Hello World',
    'chunks':[
        {'text': 'Hello', 'timestamp': (0.5, 1.0), 'speaker': 0},
        {'text': 'world', 'timestamp': (1.5, 2.0), 'speaker': 1}]}
```

```python

def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
 
    transcript_segments = transcript_result["chunks"]

    for seg in transcript_segments:
        # Calculate intersection and union between diarization segments and transcript segment
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg["timestamp"][1]) - np.maximum(diarize_df['start'], seg["timestamp"][0])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg["timestamp"][1]) - np.minimum(diarize_df['start'], seg["timestamp"][0])

        # Filter out diarization segments with no overlap if fill_nearest is False
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df

        # If there are overlapping segments, assign the speaker with the greatest overlap
        if len(dia_tmp) > 0:
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker

    return transcript_result

```

**Running the diarization model**
```python
diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
diarize_segments = diarize_model("/content/kitakas_eng.mp3", min_speakers=1, max_speakers=2)

diarize_segments
```