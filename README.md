# Transformer: From Scratch vs. Pretrained Model (Sentiment Analysis)

An academic NLP course project comparing a **from-scratch Encoder-only Transformer** built using pure PyTorch primitives against a **pretrained Transformer (`distilbert-base-uncased`)** fine-tuned on the exact same 3-class Sentiment Analysis dataset (`Negative / Neutral / Positive`).

---

## 1. Dataset Citation & Source

- **Dataset Name:** `cardiffnlp/tweet_eval` (`sentiment` configuration — originally SemEval-2017 Task 4A)
- **Direct Source URL:** https://huggingface.co/datasets/cardiffnlp/tweet_eval
- **Benchmark Repository:** https://github.com/cardiffnlp/tweeteval
- **License:** MIT License (TweetEval benchmark wrapper) / SemEval-2017 Task 4A research data
- **Labels (3 classes):** `0 = Negative`, `1 = Neutral`, `2 = Positive`
- **Reference:**
  > Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. Findings of EMNLP 2020.

### Subset & Split Strategy (Fixed Seed = `42`)
To keep training fast on standard laptops or Google Colab while maintaining a fair, leakage-free comparison, we sample a **strictly balanced** subset from the official disjoint splits:

| Split | Samples | Per-Class Count (Neg / Neu / Pos) | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 6,000 | 2,000 / 2,000 / 2,000 | Model training & scratch vocabulary creation |
| **Validation** | 900 | 300 / 300 / 300 | Checkpoint selection (best Macro-F1) |
| **Test** | 1,500 | 500 / 500 / 500 | Final held-out evaluation (touched once) |
| **Total** | **8,400** | **2,800 / 2,800 / 2,800** | Identical splits for both models |

---

## 2. Project Structure

```text
.
├── sentiment_transformer_comparison.ipynb  # Complete training, evaluation & visualization pipeline
├── app.py                                  # Interactive Streamlit comparison & live inference app
├── requirements.txt                        # Minimal Python dependencies
├── README.md                               # Documentation, dataset citation & report summary
│
│   (Generated after running the notebook:)
├── scratch_model.pt                        # Saved Model A weights & config
├── vocab.json                              # Saved word vocabulary built on train split
├── pretrained_model/                       # Saved fine-tuned DistilBERT weights & tokenizer
└── results.json                            # Measured test metrics, latency, params & confusion matrices
```

---

## 3. Setup & How to Run

### Step 1: Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Run the Notebook (Train & Save Artifacts)
Open and run all cells in `sentiment_transformer_comparison.ipynb`:
```bash
jupyter notebook sentiment_transformer_comparison.ipynb
```
*(You can also run the notebook on Google Colab with a free T4 GPU, then download `scratch_model.pt`, `vocab.json`, `results.json`, and the `pretrained_model/` folder into this directory.)*

### Step 3: Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 4. Model Architectures

### Model A — Scratch Transformer (Encoder-Only, Pure PyTorch)
Implemented strictly from basic PyTorch primitives (`nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `torch.matmul`, `torch.softmax`) — **without** `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerEncoderLayer`, or `nn.MultiheadAttention`:
- **Tokenization:** Regex word tokenizer built strictly on the training split (`max_vocab_size=10,000`, `min_freq=2`, `max_len=64`).
- **Components:** Token Embedding + Sinusoidal Positional Encoding -> 2 Encoder Blocks (Multi-Head Self-Attention with Padding Mask + Position-Wise FFN + Residual Connections + LayerNorm) -> Masked Mean Pooling -> Linear Classification Head (3 classes).
- **Config:** `d_model=128`, `n_heads=4`, `d_ff=256`, `n_layers=2`, `dropout=0.2`.

### Model B — Pretrained Transformer (`distilbert-base-uncased`)
- **Checkpoint:** `distilbert-base-uncased` (~66.9M parameters, 6 Transformer layers, `d_model=768`, 12 heads).
- **Tokenizer:** WordPiece subword tokenizer (`max_len=64`).
- **Fine-Tuning:** 3 epochs, `AdamW(lr=2e-5, weight_decay=0.01)`, linear warmup (10%), batch size 32, best validation Macro-F1 checkpoint saved.

---

## 5. Short Project Report Summary

### Measured Results (Held-Out Test Set — 1,500 Balanced Samples, NVIDIA Quadro T2000)

| Metric | Model A: Scratch Transformer | Model B: Pretrained DistilBERT |
| :--- | :---: | :---: |
| **Accuracy** | `0.5080` (50.80%) | `0.6727` (67.27%) |
| **Macro-F1** | `0.5113` (51.13%) | `0.6657` (66.57%) |
| **Macro Precision** | `0.5200` (52.00%) | `0.6765` (67.65%) |
| **Macro Recall** | `0.5080` (50.80%) | `0.6727` (67.27%) |
| **Avg Latency / Sample (ms)** | `4.02 ms` | `8.86 ms` |
| **Total Parameters** | `1,065,603` (~1.07M) | `66,955,779` (~66.96M) |
| **Training Time (s)** | `23.8 s` (10 epochs) | `175.8 s` (3 epochs) |

### Key Takeaways & Discussion
1. **Why Pretraining Wins on Accuracy & Macro-F1:** `distilbert-base-uncased` was pretrained on billions of words (Wikipedia + BookCorpus) using Masked Language Modeling, giving it rich representations of syntax, negation (`"not bad"`), idioms, and subword morphology (WordPiece handles slang and typos gracefully). Fine-tuning only needs to adapt its representations to 3 sentiment classes.
2. **Limitations of Training a Transformer From Scratch on 6,000 Samples:** Transformers lack strong built-in inductive biases compared to CNNs/RNNs and typically require massive corpora to learn general language semantics. On 6,000 tweets, Model A relies heavily on frequent sentiment keywords, struggles with unseen out-of-vocabulary (`<UNK>`) words, and is more prone to overfitting and misclassifying sarcasm or subtle negation.
3. **Efficiency Trade-off:** Model A has ~40x–60x fewer parameters than DistilBERT, resulting in noticeably faster single-sample inference latency and a much smaller disk footprint (`scratch_model.pt` is a few megabytes vs. ~260 MB for DistilBERT). In latency- or memory-constrained edge applications, compact models remain valuable — though distilled pretrained models offer the best balance when high accuracy is required.
