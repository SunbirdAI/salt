from dataclasses import dataclass
import html
import functools
from typing import Union
import pandas as pd
from IPython import display
import transformers
from transformers import TrainerCallback
import torch
import mlflow

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

def show_dataset(ds, N=10, rate=16_000, audio_features=[], normalize_audio=True):
    '''Show dataset inside a Jupyter notebook with embedded audio.'''
    def create_audio_player_from_array(audio_data):   
        if isinstance(audio_data, dict) and 'array' in audio_data:
            audio_player = display.Audio(
                data=audio_data['array'], rate=rate, normalize=normalize_audio)
        else:
            audio_player = display.Audio(
                data=audio_data, rate=rate, normalize=normalize_audio)
        return audio_player._repr_html_().replace('\n', '')

    df_audio = pd.DataFrame(ds.take(N))
    for k in list(df_audio.columns):
        if k in audio_features:
            df_audio[k] = df_audio[k].apply(create_audio_player_from_array)
        else:
            df_audio[k] = df_audio[k].apply(
                lambda x: html.escape(x) if isinstance(x, str) else x)

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


class MlflowExtendedLoggingCallback(TrainerCallback):
    """
    A custom callback that logs training loss and evaluation metrics to MLflow. Useful to track performance at every logging step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called every logging_steps during training. Use this for training metrics.
        """
        if logs is not None:
            # Check if 'loss' key is in logs, log it as training loss
            if "loss" in logs:
                mlflow.log_metrics({"train_loss": logs["loss"]}, step=state.global_step)
                # print(f"Logged Training Loss: {logs['loss']} at step: {state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called at the end of the evaluation phase. Use this for evaluation metrics like WER and validation loss.
        """
        if metrics is not None:
            # Log all metrics with the prefix "eval" to MLflow.
            metrics_to_log = {key: value for key, value in metrics.items() if "eval" in key}
            mlflow.log_metrics(metrics_to_log, step=state.global_step)
            # print(f"Logged Evaluation Metrics: {metrics_to_log} at step: {state.global_step}")



class TrainableM2MForConditionalGeneration(
    transformers.M2M100ForConditionalGeneration):
    '''
    M2M100ForConditionalGeneration trainable with multiple target languages.
    
    The Facebook M2M models (NLLB, mBART, M2M100) cannot be trained with
    multiple target languages as is, because their `generate()` functions
    require `forced_bos_token_id`, but the model call does not. We add a dummy
    input here so that `forced_bos_token_id` can be added to input batches
    during training without raising an error.
    '''
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        forced_bos_token_id=None,  # Ignored
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ForcedVariableBOSTokenLogitsProcessor(transformers.LogitsProcessor):
    '''LogitsProcessor which allows the forced BOS token to be different for
    each example in a batch. By default there is only a single BOS token used
    in every training example.
    
    Usage to replace the default `ForcedBOSTokenLogitsProcessor`, so that it
    is used automatically in training and generation:
    
    ```
    transformers.generation.utils.ForcedBOSTokenLogitsProcessor = leb.utils.ForcedVariableBOSTokenLogitsProcessor
    ``` 
    '''
    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            batch_size = input_ids.shape[0]
            num_tokens = scores.shape[1]
            for j in range(batch_size):
                scores[j, :] = -float("inf")
                scores[j, self.bos_token_id[j]] = 0
        return scores
