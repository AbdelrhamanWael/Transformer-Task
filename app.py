"""
Streamlit App: Transformer From Scratch vs. Pretrained DistilBERT
Sentiment Analysis (Negative / Neutral / Positive)

Run locally after executing sentiment_transformer_comparison.ipynb:
    streamlit run app.py
"""

import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. Constants & Artifact Paths
# ---------------------------------------------------------------------------
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_COLORS = {
    "Negative": "#d9534f",
    "Neutral": "#5b7c99",
    "Positive": "#2e8b57",
}
LABEL_EMOJI = {
    "Negative": "🔴",
    "Neutral": "⚪",
    "Positive": "🟢",
}

BASE_DIR = Path(__file__).resolve().parent
SCRATCH_MODEL_PATH = BASE_DIR / "scratch_model.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"
PRETRAINED_DIR = BASE_DIR / "pretrained_model"
RESULTS_PATH = BASE_DIR / "results.json"

EXAMPLE_SENTENCES = {
    "Positive": "Just finished watching the new sci-fi movie and honestly it blew my mind! Best soundtrack of the year.",
    "Negative": "Waited two hours for my delivery and the food arrived completely cold. Never ordering from here again.",
    "Neutral": "The quarterly earnings report will be published next Tuesday at 9 AM Eastern Time.",
    "Tricky / Sarcastic": "Oh great, another software update that broke my Wi-Fi right before an important meeting. Just what I needed!",
}


# ---------------------------------------------------------------------------
# 2. Scratch Transformer Architecture (Pure PyTorch Primitives Only)
# ---------------------------------------------------------------------------
def tokenize(text: str):
    """Simple regex word tokenizer matching the notebook preprocessing."""
    return re.findall(r"[a-z0-9']+|[!?.,]", text.lower())


def encode_text(text: str, vocab: dict, max_len: int):
    """Convert raw text to padded token IDs (guaranteed at least 1 non-pad token)."""
    tokens = tokenize(text)
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)
    ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    if not ids:
        ids = [unk_id]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return ids


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.self_attn(x, mask=mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class ScratchTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        max_len: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_encoding(x))

        for layer in self.layers:
            x = layer(x, mask=pad_mask)

        valid_mask = (input_ids != self.pad_idx).unsqueeze(-1).float()
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# 3. Cached Loaders & Inference Helpers
# ---------------------------------------------------------------------------
def check_artifacts():
    missing = []
    if not SCRATCH_MODEL_PATH.exists():
        missing.append("scratch_model.pt")
    if not VOCAB_PATH.exists():
        missing.append("vocab.json")
    if not PRETRAINED_DIR.exists():
        missing.append("pretrained_model/")
    if not RESULTS_PATH.exists():
        missing.append("results.json")
    return missing


@st.cache_data
def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model A (Scratch Transformer)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load(SCRATCH_MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scratch_model = ScratchTransformerClassifier(**cfg).to(device)
    scratch_model.load_state_dict(checkpoint["model_state_dict"])
    scratch_model.eval()

    # Load Model B (Pretrained DistilBERT)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(str(PRETRAINED_DIR))
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(str(PRETRAINED_DIR)).to(
        device
    )
    pretrained_model.eval()

    return device, vocab, cfg, scratch_model, pretrained_tokenizer, pretrained_model


def predict_scratch(text: str, model, vocab: dict, max_len: int, device):
    start = time.perf_counter()
    ids = encode_text(text, vocab, max_len)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], probs, latency_ms


def predict_pretrained(text: str, model, tokenizer, max_len: int, device):
    start = time.perf_counter()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], probs, latency_ms


# ---------------------------------------------------------------------------
# 4. Plotting Helpers
# ---------------------------------------------------------------------------
def plot_metrics_bar(scratch_res: dict, pretrained_res: dict):
    metrics = ["Accuracy", "Macro-F1", "Precision", "Recall"]
    scratch_vals = [
        scratch_res["accuracy"],
        scratch_res["macro_f1"],
        scratch_res["precision"],
        scratch_res["recall"],
    ]
    pretrained_vals = [
        pretrained_res["accuracy"],
        pretrained_res["macro_f1"],
        pretrained_res["precision"],
        pretrained_res["recall"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars1 = ax.bar(x - width / 2, scratch_vals, width, label="Model A: Scratch Transformer", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, pretrained_vals, width, label="Model B: Pretrained DistilBERT", color="#F58518")

    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Set Classification Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, title: str, cmap: str = "Blues"):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def render_prediction_card(title: str, subtitle: str, pred_label: str, probs, latency_ms: float):
    color = LABEL_COLORS[pred_label]
    emoji = LABEL_EMOJI[pred_label]
    top_conf = float(np.max(probs)) * 100.0

    st.markdown(f"### {title}")
    st.caption(subtitle)
    st.markdown(
        f"""
        <div style="padding: 12px 16px; border-radius: 8px; border-left: 6px solid {color};
                    background-color: rgba(128,128,128,0.08); margin-bottom: 12px;">
            <div style="font-size: 1.25rem; font-weight: 700;">{emoji} {pred_label} ({top_conf:.1f}%)</div>
            <div style="font-size: 0.9rem; opacity: 0.85;">Inference Latency: <b>{latency_ms:.2f} ms</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Class Probabilities**")
    for label, prob in zip(LABELS, probs):
        col_lbl, col_bar, col_val = st.columns([2, 6, 2])
        col_lbl.write(label)
        col_bar.progress(float(prob))
        col_val.write(f"{prob * 100:.1f}%")


# ---------------------------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Transformer: Scratch vs. Pretrained",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Transformer: From Scratch vs. Pretrained Model")
    st.markdown(
        """
        **Task:** 3-Class Sentiment Analysis (`Negative` / `Neutral` / `Positive`) on the
        [`cardiffnlp/tweet_eval` (`sentiment`)](https://huggingface.co/datasets/cardiffnlp/tweet_eval) benchmark.  
        Compare a compact **Encoder-only Transformer built from scratch in pure PyTorch** against a
        **fine-tuned Pretrained DistilBERT** trained and evaluated on the exact same data splits.
        """
    )

    # Check required notebook artifacts
    missing = check_artifacts()
    if missing:
        st.error(
            "Required model artifacts are missing: "
            + ", ".join(f"`{m}`" for m in missing)
            + "\n\nPlease run all cells in **`sentiment_transformer_comparison.ipynb`** first to train both models and generate these files."
        )
        st.stop()

    with st.spinner("Loading saved models and evaluation results..."):
        results = load_results()
        device, vocab, scratch_cfg, scratch_model, pretrained_tokenizer, pretrained_model = load_models()

    scratch_res = results["models"]["scratch"]
    pretrained_res = results["models"]["pretrained"]
    max_len = scratch_cfg.get("max_len", 64)

    # -----------------------------------------------------------------------
    # Section A: Interactive Side-by-Side Inference
    # -----------------------------------------------------------------------
    st.divider()
    st.header("1. Live Side-by-Side Sentiment Inference")

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = EXAMPLE_SENTENCES["Positive"]

    st.write("**Try an example sentence:**")
    btn_cols = st.columns(len(EXAMPLE_SENTENCES))
    for col, (label, sample_text) in zip(btn_cols, EXAMPLE_SENTENCES.items()):
        if col.button(label, width="stretch"):
            st.session_state["input_text"] = sample_text

    user_text = st.text_area(
        "Enter a sentence or tweet to analyze:",
        key="input_text",
        height=95,
        placeholder="Type any English sentence here...",
    )

    analyze_clicked = st.button("Analyze Sentiment", type="primary")

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before clicking Analyze Sentiment.")
        else:
            with st.spinner("Running inference on both models..."):
                pred_a, probs_a, lat_a = predict_scratch(
                    user_text, scratch_model, vocab, max_len, device
                )
                pred_b, probs_b, lat_b = predict_pretrained(
                    user_text, pretrained_model, pretrained_tokenizer, max_len, device
                )

            col_a, col_b = st.columns(2)
            with col_a:
                render_prediction_card(
                    "Model A: Scratch Transformer",
                    scratch_res.get("architecture", "2-layer Pure PyTorch Encoder"),
                    pred_a,
                    probs_a,
                    lat_a,
                )
            with col_b:
                render_prediction_card(
                    "Model B: Pretrained DistilBERT",
                    f"Checkpoint: {pretrained_res.get('checkpoint', 'distilbert-base-uncased')}",
                    pred_b,
                    probs_b,
                    lat_b,
                )

            if pred_a == pred_b:
                st.success(f"Both models agree: **{pred_a}**.")
            else:
                st.info(
                    f"Models disagree — **Scratch Transformer** predicted **{pred_a}**, "
                    f"while **Pretrained DistilBERT** predicted **{pred_b}**."
                )

    # -----------------------------------------------------------------------
    # Section B: Evaluation & Comparison Dashboard
    # -----------------------------------------------------------------------
    st.divider()
    st.header("2. Test Set Evaluation & Comparison Dashboard")

    ds_info = results.get("dataset", {})
    splits = ds_info.get("splits", {})
    st.caption(
        f"Evaluated on identical held-out test split ({splits.get('test', 1500):,} balanced samples) | "
        f"Train: {splits.get('train', 6000):,} | Val: {splits.get('validation', 900):,} | "
        f"Seed: {ds_info.get('seed', 42)} | Eval Device: {results.get('device', str(device))}"
    )

    # Summary KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Accuracy (Pretrained vs Scratch)",
        f"{pretrained_res['accuracy'] * 100:.2f}%",
        f"{(pretrained_res['accuracy'] - scratch_res['accuracy']) * 100:+.2f}%",
    )
    kpi2.metric(
        "Macro-F1 (Pretrained vs Scratch)",
        f"{pretrained_res['macro_f1'] * 100:.2f}%",
        f"{(pretrained_res['macro_f1'] - scratch_res['macro_f1']) * 100:+.2f}%",
    )
    kpi3.metric(
        "Scratch Latency (avg)",
        f"{scratch_res['avg_latency_ms']:.2f} ms",
        f"{scratch_res['avg_latency_ms'] - pretrained_res['avg_latency_ms']:+.2f} ms vs Pretrained",
        delta_color="inverse",
    )
    kpi4.metric(
        "Parameter Ratio",
        f"{pretrained_res['total_params'] / max(scratch_res['total_params'], 1):.1f}x",
        "Pretrained / Scratch size",
    )

    # Side-by-Side Metrics Table
    st.subheader("Side-by-Side Test Metrics")
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro-F1",
                "Macro Precision",
                "Macro Recall",
                "Avg Inference Latency (ms/sample)",
                "Total Parameters",
                "Trainable Parameters",
                "Training Time (seconds)",
            ],
            "Model A: Scratch Transformer": [
                f"{scratch_res['accuracy']:.4f}",
                f"{scratch_res['macro_f1']:.4f}",
                f"{scratch_res['precision']:.4f}",
                f"{scratch_res['recall']:.4f}",
                f"{scratch_res['avg_latency_ms']:.2f}",
                f"{scratch_res['total_params']:,}",
                f"{scratch_res['trainable_params']:,}",
                f"{scratch_res['training_time_sec']:.1f}",
            ],
            "Model B: Pretrained DistilBERT": [
                f"{pretrained_res['accuracy']:.4f}",
                f"{pretrained_res['macro_f1']:.4f}",
                f"{pretrained_res['precision']:.4f}",
                f"{pretrained_res['recall']:.4f}",
                f"{pretrained_res['avg_latency_ms']:.2f}",
                f"{pretrained_res['total_params']:,}",
                f"{pretrained_res['trainable_params']:,}",
                f"{pretrained_res['training_time_sec']:.1f}",
            ],
        }
    )
    st.dataframe(comparison_df, hide_index=True, width="stretch")

    # Classification Performance Bar Chart
    st.subheader("Classification Metrics Comparison")
    fig_bar = plot_metrics_bar(scratch_res, pretrained_res)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Confusion Matrices
    st.subheader("Confusion Matrices (Held-Out Test Set)")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        fig_cm_a = plot_confusion_matrix(
            scratch_res["confusion_matrix"],
            "Model A: Scratch Transformer",
            cmap="Blues",
        )
        st.pyplot(fig_cm_a)
        plt.close(fig_cm_a)

    with cm_col2:
        fig_cm_b = plot_confusion_matrix(
            pretrained_res["confusion_matrix"],
            "Model B: Pretrained DistilBERT",
            cmap="Oranges",
        )
        st.pyplot(fig_cm_b)
        plt.close(fig_cm_b)

    # Footer
    st.divider()
    st.caption(
        f"Dataset: {ds_info.get('name', 'cardiffnlp/tweet_eval')} ({ds_info.get('config', 'sentiment')}) — "
        f"{ds_info.get('url', 'https://huggingface.co/datasets/cardiffnlp/tweet_eval')} | "
        f"Live inference running on `{device}`."
    )


if __name__ == "__main__":
    main()
