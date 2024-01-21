from dataclasses import dataclass
import functools
from typing import Union
import pandas as pd
from IPython import display
import transformers

def single_batch_entry(func):
    """Split a batch into individual items, process and then recombine."""
    @functools.wraps(func)
    def single_batch_entry(r, src_or_tgt, **kwargs):
        keys = r.keys()        
        result = {k: [] for k in keys}
        for i in range(len(r['source'])):
            single_entry = {k: r[k][i] for k in keys}
            single_result = func(single_entry, src_or_tgt, **kwargs)
            for k in keys:
                result[k].append(single_result[k])
        return result
    return single_batch_entry

def show_dataset(ds, N=10, rate=16_000, audio_features=[]):
    '''Show dataset inside a Jupyter notebook with embedded audio.'''
    def create_audio_player_from_array(audio_data):   
        if isinstance(audio_data, dict) and 'array' in audio_data:
            audio_player = display.Audio(data=audio_data['array'], rate=rate)
        else:
            audio_player = display.Audio(data=audio_data, rate=rate) 
        return audio_player._repr_html_().replace('\n','')

    df_audio = pd.DataFrame(ds.take(N))
    for k in audio_features:
        df_audio[k] = df_audio[k].apply(create_audio_player_from_array)

    display.display(display.HTML(df_audio.to_html(escape=False)))

    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or
        :class:`~transformers.tokenization_utils_base.PaddingStrategy`,
        `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the 
            model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in 
              the batch (or no padding if only a single sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the 
              argument :obj:`max_length` or to the maximum acceptable input 
              length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e.,
              can output a batch with sequences of
              different lengths).
    """
    # TODO: Check updated version at https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py

    processor: transformers.Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        return batch