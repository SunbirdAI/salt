
## Training Data

Sunflower was trained on a **diverse multilingual dataset** covering Ugandan and regional languages:

* Text from books, articles, and online sources
* Conversational datasets and community-contributed corpora
* Parallel texts for translation tasks

This dataset enables Sunflower to handle multiple domains, from informal conversational text to formal written content, ensuring robust multilingual understanding.

---

## Architecture

Sunflower inherits the **Qwen-3-14B Transformer architecture**, a **decoder-only large language model**. Key components:

* **Transformer Layers:** Deep stack of self-attention layers to capture token dependencies
* **Multi-Head Self-Attention:** Captures relationships between all tokens in a sequence
* **Feed-Forward Networks (FFN):** Non-linear transformations at each layer
* **Layer Normalization & Residual Connections:** Stabilize training and preserve information across layers
* **Embedding Layers:** Token embeddings, positional embeddings, and language embeddings for multilingual support

This architecture enables Sunflower to **generate coherent text, translate languages, answer questions, and explain content**.

---

## Pre-training

* Pretrained on **Qwen-3-14B**, a large-scale multilingual model
* Learned **grammar, context, and cross-lingual patterns** across languages
* Provides a strong foundation for translation, summarization, Q&A, and explanation tasks

---

## Fine-tuning

Sunflower was fine-tuned on **task-specific datasets** using **UnslothTrainer** and **SFTTrainer**:

* **Translation:** Improve inter-language translation accuracy
* **Summarization:** Condense long passages while retaining meaning
* **Question & Answer:** Provide context-aware responses
* **Explanation:** Generate detailed clarifications

Fine-tuning ensures the model is practical for **real-world multilingual applications**, even in low-resource languages.




