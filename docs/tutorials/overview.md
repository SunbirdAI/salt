# SALT Python Package: Overview & Installation

The **SALT** (Sunbird African Language Technology) Python library is designed to make it easy to work with multilingual speech and text datasets, and to train and evaluate models for low-resource African languages.

## Key Features

- **Standardized Data Loading**: Load multi-way parallel speech and text datasets with simple YAML configuration.
- **Robust Preprocessing**: Built-in utilities for text cleaning, random case/character augmentation, and audio noise/downsampling.
- **Hugging Face Integration**: Customized wrappers and trainers to easily fine-tune sequence-to-sequence models (like NLLB-200) on African languages.
- **Evaluation Metrics**: Multilingual evaluation wrappers for standard metrics (e.g. SacreBLEU).

---

## Installation

### Prerequisites

- **Python**: Python 3.11 or above.
- **Git**: For cloning the repository.
- **pip / uv**: For installing dependencies.

### Installation Steps

Currently, the `salt` package can be installed locally from the repository.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SunbirdAI/salt.git
   cd salt
   ```

2. **Install Package and Dependencies**:
   Install the package in editable mode along with its dependencies:
   ```bash
   pip install -e .
   ```
   *(Alternatively, if you use `uv`, you can run `uv pip install -e .`)*

3. **Verify the Installation**:
   Ensure everything is installed correctly by running tests:
   ```bash
   pytest
   ```

---

## Library Modules

The `salt` package is organized into several key modules:

- **`salt.dataset`**: Handles dataset loading, validation, and schema conversion based on YAML configuration.
- **`salt.preprocessing`**: Provides tools for text normalization, text augmentation, and audio noise augmentation.
- **`salt.metrics`**: Implements standardized wrappers to compute evaluation metrics (e.g. BLEU) for multilingual models.
- **`salt.utils`**: Custom tokenizers, logits processors, and wrappers for Hugging Face architectures.
