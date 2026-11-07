# 📜 Shakespeare Next Word Predictor (LSTM)

An end-to-end Deep Learning Natural Language Processing (NLP) system trained on the **Tiny Shakespeare** dataset to predict single and multiple consecutive words using a custom single-layer **LSTM** neural network.

---

## 🌟 Key Features

1. **Dataset Pipeline (Tiny Shakespeare)**:
   - Automated fetching of raw Shakespeare corpus (`~1.1 MB`, `~40,000 lines`).
   - Text cleaning, noise removal, contraction preservation, and sentence structuring.
2. **Tokenization & Pre-Padding**:
   - Word-level mapping with Keras `Tokenizer`.
   - N-Gram sequence generation with **Pre-Padding** (`pad_sequences(..., padding='pre')`) so the most recent words directly feed the recurrent hidden state.
3. **Sequential Model Architecture (v1 Baseline)**:
   - **Layer 1: Embedding Layer** (100 dimensions) - maps token IDs to dense semantic vectors.
   - **Layer 2: Single LSTM Layer** (150 units) - models recurrent sequential patterns and cadence.
   - **Regularization: Dropout** (`0.2`) - prevents memorization and overfitting.
   - **Layer 3: Dense Softmax Output Layer** - projects to the full vocabulary size to output probability distributions over candidate words.
4. **Multi-Word Prediction Engine**:
   - Generates continuous sequences of $K$ words iteratively.
   - **Temperature Sampling** ($0.1$ to $1.5$) allowing either greedy deterministic predictions or poetic, diverse Shakespearean style.
   - Step-by-step Top-K probability distribution visualization.
5. **Interactive Streamlit Web UI (`app.py`)**:
   - Elegant Shakespearean-themed visual dashboard.
   - Real-time multi-word generation with highlighted seed vs. predicted token badges.
   - Interactive temperature and sequence length sliders.
   - Confidence percentage breakdown for candidate tokens.

---

## 📊 Empirical Validation & Web UI Test Runs (Version 1 Baseline)

The baseline model (Version 1) was trained and evaluated on the Streamlit web interface across classic Shakespearean dramatic prompts, demonstrating consistent adherence to grammatical meter and play dialogue.

### 📌 Verified Model Status & Metrics
| Metric | Value |
| :--- | :--- |
| **Model Status** | `Loaded & Verified` ✅ |
| **Vocabulary Size** | `1,787 unique tokens` |
| **Maximum Sequence Length** | `12 tokens` |
| **Training Epochs** | `25 epochs` |
| **Loss Metric** | `3.8421` (Sparse Categorical Crossentropy) |

---

### 🧪 Live Inference Test Results (Web UI)

The following test runs were captured directly from the interactive web interface:

#### Case 1: Coriolanus Civic Speech (Menenius Agrippa)
* **Input Seed**: `what work's my countrymen`
* **Model Prediction**: `in hand where go you shall`
* **Full Generated Output**:
  > *"what work's my countrymen in hand where go you shall"*
* **Source Play Reference**: Coriolanus, Act I, Scene 1 (*"What work's, my countrymen, in hand? where go you With bats and clubs?"*).

#### Case 2: Citizen Revolt Dialogue
* **Input Seed**: `let us kill him`
* **Model Prediction**: `and we'll have corn at our`
* **Full Generated Output**:
  > *"let us kill him and we'll have corn at our"*
* **Source Play Reference**: Coriolanus, Act I, Scene 1 (*"Let us kill him, and we'll have corn at our own price."*).

#### Case 3: Solemn Citizen Decision
* **Input Seed**: `you are all resolved rather`
* **Model Prediction**: `to die than to famish and`
* **Full Generated Output**:
  > *"you are all resolved rather to die than to famish and"*
* **Source Play Reference**: Coriolanus, Act I, Scene 1 (*"You are all resolved rather to die than to famish?"*).

#### Case 4: Dramatic Oratory Opening
* **Input Seed**: `before we proceed any further`
* **Model Prediction**: `hear me speak i do not`
* **Full Generated Output**:
  > *"before we proceed any further hear me speak i do not"*
* **Source Play Reference**: Coriolanus, Act I, Scene 1 (*"Before we proceed any further, hear me speak."*).

---

## 📂 Project Directory Structure

```plaintext
next_word_predictor/
│
├── .gitignore                 # Version control exclusions (models, environments, cache)
├── next_word_predictor.ipynb  # Full-fledged Jupyter Notebook (Step-by-step walkthrough)
├── app.py                     # Interactive Streamlit Web Application
├── train.py                   # Standalone CLI training script
├── requirements.txt           # Project dependencies
├── build_notebook.py          # Script used to generate the notebook
├── README.md                  # Comprehensive documentation & architectural roadmap
│
└── model/                     # Created upon training
    ├── shakespeare_lstm.keras # Trained LSTM model weights & architecture
    ├── tokenizer.pkl          # Pickled Keras Tokenizer
    └── config.json            # Model parameters (max_seq_len, vocab_size, loss)
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

Open your terminal or command prompt in this directory:

```bash
pip install -r requirements.txt
```

*(If on Windows with Python module: `python -m pip install -r requirements.txt`)*

---

### 2. Run in Jupyter Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook next_word_predictor.ipynb
```

The notebook contains 12 interactive sections:
1. Environment Setup & Library Imports
2. Dataset Acquisition & Loading (Tiny Shakespeare)
3. Data Cleaning & Text Preprocessing
4. Tokenization & Vocabulary Building
5. N-Gram Sequence Generation & Pre-Padding
6. Feature ($X$) and Target Label ($y$) Separation
7. Model Architecture: Single-Layer LSTM
8. Model Training & Checkpoint Callbacks
9. Training Curves & Loss/Accuracy Visualization
10. Multi-Word Prediction Engine (Greedy & Temperature)
11. Multi-Word Prediction Testing
12. Model & Tokenizer Serialization for Streamlit UI

---

### 3. Or Train Directly from Command Line

To train directly via the terminal:

```bash
python train.py --lines 1500 --epochs 25 --batch_size 64
```

**Configurable Arguments:**
- `--lines`: Number of lines from Tiny Shakespeare to train on (default: `1500`).
- `--epochs`: Number of training epochs (default: `25`).
- `--batch_size`: Batch size (default: `64`).
- `--embedding_dim`: Vector dimension of word embeddings (default: `100`).
- `--lstm_units`: Units in the single LSTM layer (default: `150`).

Once completed, it automatically saves the model, tokenizer, and config to the `model/` directory.

---

### 4. Launch the Interactive Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` to test with preset prompts or custom seed text.

---

## 🧠 Why Pre-Padding?

In natural language modeling with recurrent neural networks (LSTMs), tokens are processed left-to-right in time:

$$h_t = \text{LSTM}(x_t, h_{t-1})$$

When sequences are of variable length, we must pad them to uniform length:
- **Post-Padding (`padding='post'`)**: Appends zeroes to the end (`[word1, word2, 0, 0, 0]`). The LSTM processes several zero steps right before predicting the target word, causing its hidden state to dissipate or forget the actual context.
- **Pre-Padding (`padding='pre'`)**: Prepends zeroes to the front (`[0, 0, 0, word1, word2]`). The LSTM absorbs the padding first, and the actual context words occur immediately prior to the prediction step. This yields significantly higher accuracy in sequential next-word prediction!

---

## 🔮 Future Roadmap: Advanced Architectures & Pre-Trained Models

While the single-layer LSTM baseline (v1) successfully reproduces Shakespearean cadence, future iterations can introduce state-of-the-art NLP paradigms:

```
┌────────────────────────────────┐     ┌────────────────────────────────┐     ┌────────────────────────────────┐
│      Version 1 (Current)       │     │      Version 2 (Planned)       │     │      Version 3 (Advanced)      │
│  Single LSTM + Word Embeddings ├────►│ Stacked BiLSTM + Bahdanau Attn ├────►│  Fine-Tuned GPT-2 / LLaMA LoRA │
│   • Vocab: 1.7k words          │     │   • Subword BPE Tokenization   │     │   • 50k+ Subword BPE Vocab     │
│   • Context: 12 tokens         │     │   • Pretrained GloVe Vectors   │     │   • Long-range multi-head attn │
└────────────────────────────────┘     └────────────────────────────────┘     └────────────────────────────────┘
```

### 1. Architectural Evolutions (Recurrent & Attention)
* **Stacked Bi-directional LSTM / GRU (2–3 Layers)**:
  Stacking 2 to 3 recurrent layers allows higher layers to capture abstract structural semantics (poetic meter, speaker turn-taking) while lower layers learn local grammatical agreement.
* **Bahdanau / Luong Attention Mechanism**:
  Adding an attention layer over recurrent states enables the decoder to compute dynamic importance weights across all previous tokens rather than compressing the entire context history into a single vector.
* **Pre-Trained Word Vectors (GloVe / FastText)**:
  Replacing random embedding initialization with pre-trained 100d/300d GloVe or FastText vectors provides richer semantic geometry before fine-tuning.

---

### 2. Subword Tokenization (BPE / SentencePiece)
* **Current Limitation**: Word-level tokenization assigns `<OOV>` to unseen words and struggles with Shakespearean compound words (e.g., `soft-conscienced`, `outrageous`).
* **Upgrade**: Transitioning to **Byte-Pair Encoding (BPE)** (via Hugging Face `tokenizers` or OpenAI `tiktoken`) breaks rare words into subword units, completely eliminating `<OOV>` failures while compressing the vocabulary footprint.

---

### 3. Pre-Trained Foundation Models & Transfer Learning

| Model | Architecture | Training Approach | Advantages for Shakespeare Generation |
| :--- | :--- | :--- | :--- |
| **DistilGPT-2 / GPT-2** | 6–12 Layer Causal Transformer | Causal Language Modeling Fine-Tuning | Fast, lightweight autoregressive generation; native multi-head self-attention captures whole paragraphs. |
| **GPT-Neo / GPT-2 Medium** | 24-Layer Transformer (355M params) | Full / Layer-wise fine-tuning | Rich internal linguistic representations; deep understanding of archaic grammar and syntax. |
| **LLaMA-3-8B / Mistral-7B / Gemma-2-2B** | Modern Open LLMs | **LoRA / QLoRA** (Parameter-Efficient Fine-Tuning) | World-class language comprehension; zero `<OOV>` issues; produces complete Shakespearean scenes and iambic pentameter sonnets. |

#### Implementation Blueprint with Hugging Face `transformers`:
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# 1. Load Pretrained GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Fine-tune on Tiny Shakespeare corpus using Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./shakespeare-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    warmup_steps=100,
    logging_steps=50,
)
```

---

### 4. Advanced Decoding Strategies
* **Nucleus (Top-$p$) Sampling**: Instead of fixed Top-$K$, dynamically sample from the smallest set of words whose cumulative probability exceeds $p$ (typically $p = 0.9$).
* **Repetition Penalty**: Penalize already-generated tokens in the softmax distribution to prevent loops in extended dialogue.
* **Beam Search with Length Normalization**: Track multiple candidate hypotheses simultaneously for highest-probability output sequences.
* **KV-Cache Optimization**: Store Key-Value attention states across autoregressive steps to eliminate redundant re-computation of prompt tokens.
