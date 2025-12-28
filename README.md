# Part-of-Speech (POS) Tagging: BiLSTM vs. Transformer Architectures
> **Natural Language Processing (NLP) Project | University of Liverpool**

## 📝 Project Overview
This project addresses the fundamental NLP task of Part-of-Speech tagging. I implemented and benchmarked two distinct neural architectures—a Bidirectional LSTM (BiLSTM) and a Transformer Encoder—to classify words into 33 unique grammatical categories using a dataset of ~3,900 sentences.

### **The Result:**
The **BiLSTM** emerged as the superior model for this specific dataset size, achieving a **91.63% test accuracy** and a **0.8576 Macro F1-score**, outperforming the Transformer by nearly 5% in validation accuracy.

---

### **📄 [View Full Research Report (PDF)](./POS_Tagging_Technical_Report.pdf)**

---

## 🛠️ Tech Stack
* **Deep Learning Framework:** `PyTorch`
* **Data Engineering:** Custom `Dataset` and `DataLoader` with `collate_fn` for dynamic padding
* **Optimisation:** `Adam` with `ReduceLROnPlateau` scheduling
* **Evaluation:** `Scikit-learn` (Confusion Matrix, F1-Score)

## 🧪 Methodology & Technical Rigor
* **Data Management:** Implemented a custom vocabulary pipeline handling `<UNK>` (unknown) and `<PAD>` (padding) tokens to manage variable-length sequences.
* **BiLSTM Architecture:** Utilized a 2-layer bidirectional LSTM with **Packed Sequences** (`pack_padded_sequence`). This optimization allows the model to skip computations on padding tokens, significantly increasing training efficiency.
* **Transformer Architecture:** Built a Transformer Encoder with **Positional Encodings** and Multi-Head Attention, implementing a `src_key_padding_mask` to prevent padding tokens from influencing self-attention weights.
* **Loss Function:** Configured `CrossEntropyLoss` with `ignore_index=0` to ensure the model was only penalised for incorrect predictions on actual words, not padding.

## 📊 Performance Benchmark
| Model | Val Accuracy | Analysis |
| :--- | :--- | :--- |
| **BiLSTM** | **91.83%** | Superior sequential modeling for small-to-mid datasets. |
| **Transformer** | 86.90% | Showed signs of underfitting due to higher data requirements for attention mechanisms. |

## ⚙️ Professional Reflection & Growth
* **Inductive Bias:** This project validated that while Transformers are the industry standard for large-scale NLP, LSTMs often provide better results on smaller datasets due to their inherent sequential inductive bias.
* **Error Analysis:** Analysing the confusion matrix revealed that errors were concentrated in underrepresented tags (e.g., PDT vs DT), highlighting the importance of class weighting in future iterations.
