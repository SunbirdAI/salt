## Multilingual ASR Training for Luganda+English using the Leb Module

This tutorial provides a step-by-step guide on how to perform multilingual Automatic Speech Recognition (ASR) training for Luganda and English languages using the Leb module.

### Prerequisites
Before getting started, ensure that you have the following prerequisites:

* Python 3.x installed
* [optional] - Access to a Google Cloud account with storage access
* MLflow tracking credentials (username and password)
* [optional] Path to your service account JSON key file

## Installation
To begin, install the necessary dependencies by running the following commands:

```{bash}
!pip install -q jiwer evaluate
!pip install -qU accelerate
!pip install -q transformers[torch]
!git clone https://github.com/jqug/leb.git
!pip install -qr leb/requirements.txt
!pip install -q mlflow psutil pynvml
```


These commands will install the required libraries, including Jiwer, Evaluate, Accelerate, Transformers, MLflow, and the Leb module.

###  Configuration
Create a YAML configuration file named asr_config.yml with the necessary settings for your training. Here's an example configuration:


```{yaml}
train:
  source:
    language: [luganda, english]
    # Add other training dataset configurations

validation:
  source:
    language: [luganda, english]
    # Add other validation dataset configurations

pretrained_model: "facebook/wav2vec2-large-xlsr-53"
pretrained_adapter: null

Wav2Vec2ForCTC_args:
  adapter_model_name: "wav2vec2"

training_args:
  output_dir: "luganda_english_asr"
  # Add other training arguments
```

Load the configuration file in your Python script.

### Data Preparation
1. Load the training and validation datasets using the Leb module.
2. Create a tokenizer and processor based on the configuration.
3. Prepare the datasets for training and validation.

### Model Setup
1. Load the pre-trained model and initialize the adapter layers.
2. Define the compute metrics function.
3. Set up the training arguments and create the trainer.
### MLflow Integration
1. Set up MLflow tracking credentials and tracking URI.
2. Set the MLflow experiment.

###  Training and Evaluation
1. Start an MLflow run and train the model.
2. Save the trained model and log artifacts.

### Usage
To use the trained model for inference, follow these steps:

### Load the trained model and processor:

To use the trained model for inference, follow these steps:

1. Load the trained model and processor:


```{python}
model = Wav2Vec2ForCTC.from_pretrained("path/to/trained/model")
processor = Wav2Vec2Processor.from_pretrained("path/to/processor")
```
