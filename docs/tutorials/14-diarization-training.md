# PyanNet Model Training Speaker Diarization

This process highlights the steps taken for Model Training on the [CallHome Dataset](https://huggingface.co/datasets/talkbank/callhome). For this particular dataset we used the English version of the CallHome Dataset. The Model Training Architecture, Loss Functions, Optimisation Techniques, Data Augmentation and Metrics Used.

# Segmentation Model Configuration Explained

## Overview

### Model Architecture

- **SegmentationModel**: This is a wrapper for the PyanNet segmentation model used for speaker diarization tasks. Inherits from Pretrained model to be compatible with the HF Trainer. Can be used to train segmentation models to be used for the "SpeakerDiarisation Task" in pyannote.

**Forward**

`forward`: Forward pass function of the Pretrained Model.

Parameters:

`waveforms(torch.tensor)` : A tensor containing audio data to be processed by the model and ensures the waveforms parameter is a PyTorch tensor.

`labels`: Ground truth labels for Training. Defaults to None.

`nb_speakers`: Number of speakers. Defaults to `None`

Returns: A dictionary with loss(if predicted) and predictions.

**Setup loss function**

`setup_loss_func`: Sets up the loss function especially when using the powerset classes. ie `self.specifications.powerset=True`

**Segmentation Loss Function**

`segmentation_loss`: Defines the permutation-invariant segmentation loss. Computes the loss using either `nll_loss`(negative log likelihood) for `powerset` or `binary_cross_entropy`

Parameters:

`permutated_prediction`: Prediction after permutation. Type: `torch.Tensor`

`target`: Ground truth labels. Type: `torch.Tensor`

`weight`:  Type: `Optional[torch.Tensor]`

Returns: Permutation-invariant segmentation loss. `torch.Tensor`

**To pyannote**

`to_pyannote_model`: Converts the current model to a pyannote segmentation model for use in pyannote pipelines

```python
class SegmentationModel(PreTrainedModel):
    config_class = SegmentationModelConfig

    def __init__(
        self,
        config=SegmentationModelConfig(),
    ):
        super().__init__(config)

        self.model = PyanNet_nn(sincnet={"stride": 10})

        self.weigh_by_cardinality = config.weigh_by_cardinality
        self.max_speakers_per_frame = config.max_speakers_per_frame
        self.chunk_duration = config.chunk_duration
        self.min_duration = config.min_duration
        self.warm_up = config.warm_up
        self.max_speakers_per_chunk = config.max_speakers_per_chunk

        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.chunk_duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )
        self.model.specifications = self.specifications
        self.model.build()
        self.setup_loss_func()

    def forward(self, waveforms, labels=None, nb_speakers=None):

        prediction = self.model(waveforms.unsqueeze(1))
        batch_size, num_frames, _ = prediction.shape

        if labels is not None:
            weight = torch.ones(batch_size, num_frames, 1, device=waveforms.device)
            warm_up_left = round(self.specifications.warm_up[0] / self.specifications.duration * num_frames)
            weight[:, :warm_up_left] = 0.0
            warm_up_right = round(self.specifications.warm_up[1] / self.specifications.duration * num_frames)
            weight[:, num_frames - warm_up_right :] = 0.0

            if self.specifications.powerset:
                multilabel = self.model.powerset.to_multilabel(prediction)
                permutated_target, _ = permutate(multilabel, labels)

                permutated_target_powerset = self.model.powerset.to_powerset(permutated_target.float())
                loss = self.segmentation_loss(prediction, permutated_target_powerset, weight=weight)

            else:
                permutated_prediction, _ = permutate(labels, prediction)
                loss = self.segmentation_loss(permutated_prediction, labels, weight=weight)

            return {"loss": loss, "logits": prediction}

        return {"logits": prediction}

    def setup_loss_func(self):
        if self.specifications.powerset:
            self.model.powerset = Powerset(
                len(self.specifications.classes),
                self.specifications.powerset_max_classes,
            )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
       

        if self.specifications.powerset:
            # `clamp_min` is needed to set non-speech weight to 1.
            class_weight = torch.clamp_min(self.model.powerset.cardinality, 1.0) if self.weigh_by_cardinality else None
            seg_loss = nll_loss(
                permutated_prediction,
                torch.argmax(target, dim=-1),
                class_weight=class_weight,
                weight=weight,
            )
        else:
            seg_loss = binary_cross_entropy(permutated_prediction, target.float(), weight=weight)

        return seg_loss

    @classmethod
    def from_pyannote_model(cls, pretrained):
       
        # Initialize model:
        specifications = copy.deepcopy(pretrained.specifications)

        # Copy pretrained model hyperparameters:
        chunk_duration = specifications.duration
        max_speakers_per_frame = specifications.powerset_max_classes
        weigh_by_cardinality = False
        min_duration = specifications.min_duration
        warm_up = specifications.warm_up
        max_speakers_per_chunk = len(specifications.classes)

        config = SegmentationModelConfig(
            chunk_duration=chunk_duration,
            max_speakers_per_frame=max_speakers_per_frame,
            weigh_by_cardinality=weigh_by_cardinality,
            min_duration=min_duration,
            warm_up=warm_up,
            max_speakers_per_chunk=max_speakers_per_chunk,
        )

        model = cls(config)

        # Copy pretrained model weights:
        model.model.hparams = copy.deepcopy(pretrained.hparams)
        model.model.sincnet = copy.deepcopy(pretrained.sincnet)
        model.model.sincnet.load_state_dict(pretrained.sincnet.state_dict())
        model.model.lstm = copy.deepcopy(pretrained.lstm)
        model.model.lstm.load_state_dict(pretrained.lstm.state_dict())
        model.model.linear = copy.deepcopy(pretrained.linear)
        model.model.linear.load_state_dict(pretrained.linear.state_dict())
        model.model.classifier = copy.deepcopy(pretrained.classifier)
        model.model.classifier.load_state_dict(pretrained.classifier.state_dict())
        model.model.activation = copy.deepcopy(pretrained.activation)
        model.model.activation.load_state_dict(pretrained.activation.state_dict())

        return model

    def to_pyannote_model(self):
       
        seg_model = PyanNet(sincnet={"stride": 10})
        seg_model.hparams.update(self.model.hparams)

        seg_model.sincnet = copy.deepcopy(self.model.sincnet)
        seg_model.sincnet.load_state_dict(self.model.sincnet.state_dict())

        seg_model.lstm = copy.deepcopy(self.model.lstm)
        seg_model.lstm.load_state_dict(self.model.lstm.state_dict())

        seg_model.linear = copy.deepcopy(self.model.linear)
        seg_model.linear.load_state_dict(self.model.linear.state_dict())

        seg_model.classifier = copy.deepcopy(self.model.classifier)
        seg_model.classifier.load_state_dict(self.model.classifier.state_dict())

        seg_model.activation = copy.deepcopy(self.model.activation)
        seg_model.activation.load_state_dict(self.model.activation.state_dict())

        seg_model.specifications = self.specifications

        return seg_model
```

**Segmentation Model Configuration**

- `SegmentationModelConfig`Configuration class for the segmentation model, specifying various parameters like chunk duration, maximum speakers per frame, etc.
- Configuration parameters like chunk duration, number of speakers per chunk/frame, minimum duration, warm-up period, etc.

```python
class SegmentationModelConfig(PretrainedConfig):
    
    model_type = "pyannet"

    def __init__(
        self,
        chunk_duration=10,
        max_speakers_per_frame=2,
        max_speakers_per_chunk=3,
        min_duration=None,
        warm_up=(0.0, 0.0),
        weigh_by_cardinality=False,
        **kwargs,
    ):
       
        super().__init__(**kwargs)
        self.chunk_duration = chunk_duration
        self.max_speakers_per_frame = max_speakers_per_frame
        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.min_duration = min_duration
        self.warm_up = warm_up
        self.weigh_by_cardinality = weigh_by_cardinality
        # For now, the model handles only 16000 Hz sampling rate
        self.sample_rate = 16000
```

### Loss Functions

#### Binary Cross-Entropy

- Used when the model does not use the powerset approach.
- Computes the binary cross-entropy loss between the predicted and actual speaker activity.

#### Negative Log-Likelihood (NLL) Loss

- Used when the model uses the powerset approach.
- Computes the NLL loss considering class weights if specified.

### Optimization Techniques

### Batch Size
 - This refers to the number of samples that you feed into your model  at each iteration of the training process. This can be adjusted accordingly to optimise the performance of your model

### Learning Rate
 - This is an optimization tunning parameter that determines the step-size at each iteration while moving towards a minimum loss function

### Training Epochs
- An epoch refers to a complete pass through the entire training dataset. A model is exposed to all the training examples and updates its parametrs basd on the patterns it learns. In our case, we try and iterate and test with 5, 10 and 20 epochs and find that the Diarisation Error Rate remains constant at "'der': 0.23994926057695026"

#### Warm-up

- The warm-up period allows the model to adjust at the beginning of each chunk, ensuring the central part of the chunk is more accurate.
- The warm-up is applied to both the left and right parts of each chunk.

#### Permutation-Invariant Training

- This technique permutes predictions and targets to find the optimal alignment, ensuring the loss computation is invariant to the order of speakers.

### Data Augmentation Methods

- For our case this is done using the the DataCollator class. This class is responsible for collecting data and ensuring that the target labels are dynamically padded.
-  Pads the target labels to ensure they have the same shape.
- Pads with zeros if the number of speakers in a chunk is less than the maximum number of speakers per chunk



#### Preprocessing Steps

- Preprocessing steps like random overlap and fixed overlap during chunking can be considered a form of augmentation as they provide varied inputs to the model.
- `Preprocess` class used to handle these preprocessing steps is not detailed here, but it's responsible for preparing the input data.

```python
class Preprocess:
    def __init__(
        self,
        config,
    ):
    
        self.chunk_duration = config.chunk_duration
        self.max_speakers_per_frame = config.max_speakers_per_frame
        self.max_speakers_per_chunk = config.max_speakers_per_chunk
        self.min_duration = config.min_duration
        self.warm_up = config.warm_up

        self.sample_rate = config.sample_rate
        self.model = SegmentationModel(config).to_pyannote_model()

        # Get the number of frames associated to a chunk:
        _, self.num_frames_per_chunk, _ = self.model(
            torch.rand((1, int(self.chunk_duration * self.sample_rate)))
        ).shape

    def get_labels_in_file(self, file):
     

        file_labels = []
        for i in range(len(file["speakers"][0])):
            if file["speakers"][0][i] not in file_labels:
                file_labels.append(file["speakers"][0][i])

        return file_labels

    def get_segments_in_file(self, file, labels):
        

        file_annotations = []

        for i in range(len(file["timestamps_start"][0])):
            start_segment = file["timestamps_start"][0][i]
            end_segment = file["timestamps_end"][0][i]
            label = labels.index(file["speakers"][0][i])
            file_annotations.append((start_segment, end_segment, label))

        dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

        annotations = np.array(file_annotations, dtype)

        return annotations

    def get_chunk(self, file, start_time):
        

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        end_time = start_time + self.chunk_duration
        start_frame = math.floor(start_time * sample_rate)
        num_frames_waveform = math.floor(self.chunk_duration * sample_rate)
        end_frame = start_frame + num_frames_waveform

        waveform = file["audio"][0]["array"][start_frame:end_frame]

        labels = self.get_labels_in_file(file)

        file_segments = self.get_segments_in_file(file, labels)

        chunk_segments = file_segments[(file_segments["start"] < end_time) & (file_segments["end"] > start_time)]

        # compute frame resolution:
        # resolution = self.chunk_duration / self.num_frames_per_chunk

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_segments["start"], start_time) - start_time - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        # start_idx = np.floor(start / resolution).astype(int)
        end = np.minimum(chunk_segments["end"], end_time) - start_time - half
        end_idx = np.round(end / step).astype(int)

        # end_idx = np.ceil(end / resolution).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_segments["labels"]))
        num_labels = len(labels)
        # initial frame-level targets
        y = np.zeros((self.num_frames_per_chunk, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):
            mapped_label = mapping[label]
            y[start : end + 1, mapped_label] = 1

        return waveform, y, labels

    def get_start_positions(self, file, overlap, random=False):

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        file_duration = len(file["audio"][0]["array"]) / sample_rate
        start_positions = np.arange(0, file_duration - self.chunk_duration, self.chunk_duration * (1 - overlap))

        if random:
            nb_samples = int(file_duration / self.chunk_duration)
            start_positions = np.random.uniform(0, file_duration, nb_samples)

        return start_positions

    def __call__(self, file, random=False, overlap=0.0):

        new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

        if random:
            start_positions = self.get_start_positions(file, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, overlap)

        for start_time in start_positions:
            waveform, target, label = self.get_chunk(file, start_time)

            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["nb_speakers"].append(label)

        return new_batch
```

### Metrics and Trainer

- Initializes the Metrics class for evaluation.
- Configures the Trainer with the model, training arguments, datasets, data collator, and metrics.
- For the metrics we have the Diarisation Error Rate(DER), FalseAlarm Rate, MissedDetectionRate and the SpeakerConfusionRate with the implementation in the metrics class below.

```python
import numpy as np
import torch
from pyannote.audio.torchmetrics import (DiarizationErrorRate, FalseAlarmRate,
                                         MissedDetectionRate,
                                         SpeakerConfusionRate)
from pyannote.audio.utils.powerset import Powerset


class Metrics:
    """Metric class used by the HF trainer to compute speaker diarization metrics."""

    def __init__(self, specifications) -> None:
        """init method

        Args:
            specifications (_type_): specifications attribute from a SegmentationModel.
        """
        self.powerset = specifications.powerset
        self.classes = specifications.classes
        self.powerset_max_classes = specifications.powerset_max_classes

        self.model_powerset = Powerset(
            len(self.classes),
            self.powerset_max_classes,
        )

        self.metrics = {
            "der": DiarizationErrorRate(0.5),
            "confusion": SpeakerConfusionRate(0.5),
            "missed_detection": MissedDetectionRate(0.5),
            "false_alarm": FalseAlarmRate(0.5),
        }

    def __call__(self, eval_pred):

        logits, labels = eval_pred

        if self.powerset:
            predictions = self.model_powerset.to_multilabel(torch.tensor(logits))
        else:
            predictions = torch.tensor(logits)

        labels = torch.tensor(labels)

        predictions = torch.transpose(predictions, 1, 2)
        labels = torch.transpose(labels, 1, 2)

        metrics = {"der": 0, "false_alarm": 0, "missed_detection": 0, "confusion": 0}

        metrics["der"] += self.metrics["der"](predictions, labels).cpu().numpy()
        metrics["false_alarm"] += self.metrics["false_alarm"](predictions, labels).cpu().numpy()
        metrics["missed_detection"] += self.metrics["missed_detection"](predictions, labels).cpu().numpy()
        metrics["confusion"] += self.metrics["confusion"](predictions, labels).cpu().numpy()

        return metrics


class DataCollator:
    """Data collator that will dynamically pad the target labels to have max_speakers_per_chunk"""

    def __init__(self, max_speakers_per_chunk) -> None:
        self.max_speakers_per_chunk = max_speakers_per_chunk

    def __call__(self, features):
        """_summary_

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """

        batch = {}

        speakers = [f["nb_speakers"] for f in features]
        labels = [f["labels"] for f in features]

        batch["labels"] = self.pad_targets(labels, speakers)

        batch["waveforms"] = torch.stack([f["waveforms"] for f in features])

        return batch

    def pad_targets(self, labels, speakers):
        """
        labels:
        speakers:

        Returns:
            _type_:
                Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
                If one chunk has more than max_speakers_per_chunk speakers, we keep
                the max_speakers_per_chunk most talkative ones. If it has less, we pad with
                zeros (artificial inactive speakers).
        """

        targets = []

        for i in range(len(labels)):
            label = speakers[i]
            target = labels[i].numpy()
            num_speakers = len(label)

            if num_speakers > self.max_speakers_per_chunk:
                indices = np.argsort(-np.sum(target, axis=0), axis=0)
                target = target[:, indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                target = np.pad(
                    target,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            targets.append(target)

        return torch.from_numpy(np.stack(targets))

```

### Training Script

- The script [train_segmentation.py](https://github.com/huggingface/diarizers/)
 can be used to pre-process a diarization dataset and subsequently fine-tune the pyannote segmentation model. In the following example, we fine-tuned the segmentation model on the English subset of the CallHome dataset, a conversational dataset between native speakers:

```bash
!python3 train_segmentation.py \
    --dataset_name=diarizers-community/callhome \
    --dataset_config_name=eng \
    --split_on_subset=data \
    --model_name_or_path=pyannote/segmentation-3.0 \
    --output_dir=./speaker-segmentation-fine-tuned-callhome-eng \
    --do_train \
    --do_eval \
    --learning_rate=1e-3 \
    --num_train_epochs=20 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --preprocessing_num_workers=2 \
    --dataloader_num_workers=2 \
    --logging_steps=100 \
    --load_best_model_at_end \
    --push_to_hub
```

### Evaluation Script

The script [test_segmentation.py](https://github.com/huggingface/diarizers/)can be used to evaluate a fine-tuned model on a diarization dataset. In the following example, we evaluate the fine-tuned model from the previous step on the test split of the CallHome English dataset:

```bash
!python3 test_segmentation.py \
    --dataset_name=diarizers-community/callhome \
    --dataset_config_name=eng \
    --split_on_subset=data \
    --test_split_name=test \
    --model_name_or_path=diarizers-community/speaker-segmentation-fine-tuned-callhome-eng \
    --preprocessing_num_workers=2 \
    --evaluate_with_pipeline
```

**Sample Output**

![alt text](EVAL.PNG)


### Inference with Pyannote
- The fine-tuned segmentation model can easily be loaded into the pyannote speaker diarization pipeline for inference. To do so, we load the pre-trained speaker diarization pipeline, and subsequently override the segmentation model with our fine-tuned checkpoint:


```python
from diarizers import SegmentationModel
from pyannote.audio import Pipeline
from datasets import load_dataset
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(device)

# replace the segmentation model with your fine-tuned one
model = SegmentationModel().from_pretrained("diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn")
model = model.to_pyannote_model()
pipeline._segmentation.model = model.to(device)

# load dataset example
dataset = load_dataset("diarizers-community/callhome", "jpn", split="data")
sample = dataset[0]["audio"]

# pre-process inputs
sample["waveform"] = torch.from_numpy(sample.pop("array")[None, :]).to(device, dtype=model.dtype)
sample["sample_rate"] = sample.pop("sampling_rate")

# perform inference
diarization = pipeline(sample)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```


