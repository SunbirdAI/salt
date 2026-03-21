import sys
from pathlib import Path
root_dir = Path.cwd().parent.parent.parent
sys.path.append(str(root_dir))

import torch
import transformers
from dataclasses import dataclass
from typing import Union, List, Dict, Any
import os
import numpy as np
import yaml
import evaluate
import salt.dataset
import salt.metrics
import salt.constants
import huggingface_hub
import peft
import gcsfs
import mlflow

def setup_mlflow():
    from getpass import getpass

    if 'MLFLOW_TRACKING_USERNAME' not in os.environ:
        os.environ['MLFLOW_TRACKING_USERNAME'] = getpass('Enter the MLFLOW_TRACKING_USERNAME: ')
    
    if 'MLFLOW_TRACKING_PASSWORD' not in os.environ:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter the MLFLOW_TRACKING_PASSWORD: ')

    mlflow.set_tracking_uri('https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/')
    mlflow.system_metrics.enable_system_metrics_logging()

def list_gcs_buckets(gcs_token):
    gcs = gcsfs.GCSFileSystem(project='sb-gcp-project-01', token=gcs_token)
    bucket_path = "sunflower-data/speech"
    all_datasets = gcs.ls(path=bucket_path, detail=False)
    print("all google cloud buckets: \n", all_datasets)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]    
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def load_model_pack(config):
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config['pretrained_model'])
    processor = transformers.WhisperProcessor.from_pretrained(
        config['pretrained_model'], language=None, task="transcribe")
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config['pretrained_model'], torch_dtype=torch.float32)

    from transformers import GenerationConfig

    gen_config = GenerationConfig.from_pretrained(config['pretrained_model'])
    gen_config.suppress_tokens = []  # This clears the default suppressed tokens
    gen_config.max_length = config['training_args']['generation_max_length'] #  maximum number of tokens to generate during evaluation and prediction
    gen_config.forced_decoder_ids = None
    model.generation_config = gen_config

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    if config['use_peft']:
        model = peft.prepare_model_for_kbit_training(model)
        lora_config = peft.LoraConfig(**config['lora_config'])
        model.enable_input_require_grads()
        model = peft.get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    return model, feature_extractor, processor


def load_datasets(config, feature_extractor, processor):
    train_ds = salt.dataset.create(config['train'], verbose=True)
    valid_ds = salt.dataset.create(config['validation'])

    def inspect_example(dataset):
        for i, example in enumerate(dataset.take(1)):
            print(f"--- Example {i} ---")
            for key in ['source', 'target']:    
                data = example.get(key)
                print(f"{key}: {data}")
                    
            print(f"Languages: {example.get('source.language')} -> {example.get('target.language')}")

    inspect_example(valid_ds)

    def prepare_dataset(example, p_prompt = 0.5):    
        audio = example["source"]
        input_features = feature_extractor(
            audio, sampling_rate=16000, device='cuda',
            do_normalize=True).input_features[0]

        # Encode target text to label ids
        labels = processor.tokenizer(str(example["target"])).input_ids

        # Insert the language ID token into the second position of the sequence.
        labels.insert(1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[example["target.language"]])

        # # If a prompt is known for a particular sentence, add it to the
        # # training example with probability `p_prompt`.
        # if example["target.language"] in sentence_to_prompt:
        #     prompt = sentence_to_prompt[example["target.language"]].get(example["target"], None)
        #     if prompt:
        #         if np.random.random() < p_prompt:
        #             prompt_ids = list(processor.get_prompt_ids(prompt))
        #             labels = prompt_ids + labels  

        # Create a new dictionary with the processed data
        processed_example = {
            "input_features": input_features,
            "labels": np.array(labels),
            "source.language": example["source.language"],
            "target.language": example["target.language"]
        }

        return processed_example
    
    
    train_data = train_ds.map(prepare_dataset, remove_columns=["source", "target"])
    train_data = train_data.filter(lambda x: len(x["labels"]) <= 448)
    val_data = valid_ds.map(prepare_dataset, remove_columns=["source", "target"])
    val_data = val_data.filter(lambda x: len(x["labels"]) <= 448)

    train_data.approx_row_count = getattr(train_ds, "approx_row_count", None)
    val_data.approx_row_count = getattr(valid_ds, "approx_row_count", None)

    return train_data, val_data


def launch_training(model, processor, train_data, val_data, config):
    training_args_dict = config["training_args"]

    import math
    if "num_train_epochs" in training_args_dict and getattr(train_data, 'approx_row_count', None):
        num_epochs = training_args_dict["num_train_epochs"]
        global_batch_size = training_args_dict.get("per_device_train_batch_size", 16) * training_args_dict.get("gradient_accumulation_steps", 4)
        steps_per_epoch = math.ceil(train_data.approx_row_count / global_batch_size)
        total_steps = math.ceil(num_epochs * steps_per_epoch)
        print(f"Calculated approximate training steps based on approx_row_count ({train_data.approx_row_count}): {total_steps}")
        training_args_dict["max_steps"] = total_steps

    training_args = transformers.Seq2SeqTrainingArguments(
        **training_args_dict,
        report_to= ["mlflow"]
    )
    print("training arguments:\n", training_args_dict)
    print(f"Total training steps to run: {training_args.max_steps}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    compute_metrics = salt.metrics.multilingual_eval_fn(
      val_data, [evaluate.load('wer'), evaluate.load('cer')],
      processor.tokenizer, log_first_N_predictions=5,
      speech_processor=processor)

    # --- Debug callback to track model.generate() progress per batch ---
    import time as _time

    class EvalProgressCallback(transformers.TrainerCallback):
        def __init__(self):
            self._batch_count = 0
            self._eval_start = _time.time()

        def on_evaluate(self, args, state, control, **kwargs):
            self._batch_count = 0
            self._eval_start = _time.time()
            print(f"\n[EVAL] Starting evaluation at step {state.global_step}", flush=True)

        def on_prediction_step(self, args, state, control, **kwargs):
            self._batch_count += 1
            elapsed = _time.time() - self._eval_start
            examples_done = self._batch_count * args.per_device_eval_batch_size
            rate = examples_done / elapsed if elapsed > 0 else 0
            print(f"[EVAL] Batch {self._batch_count} done | "
                  f"~{examples_done} examples | "
                  f"{elapsed:.1f}s elapsed | "
                  f"{rate:.1f} examples/s", flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and any(k.startswith("eval_") for k in logs):
                total_time = _time.time() - self._eval_start
                print(f"\n[EVAL] ✅ Evaluation complete in {total_time:.1f}s "
                      f"({self._batch_count} batches)")
                print(f"{'='*60}\n", flush=True)

    trainer = transformers.Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.add_callback(EvalProgressCallback())

    trainer.train()   

def test_model(model, processor, val_data):
    example = next(iter(val_data))
    input_features = torch.tensor(example["input_features"]).unsqueeze(0)
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    print(transcription)


def main(
    config_path="configs/whisper_finetuning_gcs.yaml",
    gcs_token="gcs-data-viewer-key.json",
):
    setup_mlflow()
    huggingface_hub.login()
    list_gcs_buckets(gcs_token)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model, feature_extractor, processor = load_model_pack(config)
    train_data, val_data = load_datasets(config, feature_extractor, processor)

    launch_training(model, processor, train_data, val_data, config)

    # push trained model to hugging face
    processor.push_to_hub(config['training_args']['hub_model_id'])
    model.push_to_hub(config['training_args']['hub_model_id'])

    # test model after training
    test_model(model, processor, val_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune Whisper models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/whisper_finetuning_gcs.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--gcs-token",
        dest="gcs_token",
        type=str,
        default="gcs-data-viewer-key.json",
        help="Path to GCS token JSON file",
    )
    args = parser.parse_args()

    main(config_path=args.config, gcs_token=args.gcs_token)
