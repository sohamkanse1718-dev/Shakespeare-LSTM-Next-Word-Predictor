# 📜 Shakespeare Next Word Predictor (LSTM)

An end-to-end **Deep Learning NLP project** trained on the **Tiny Shakespeare** dataset to predict the next word and generate multiple consecutive words using a custom single-layer **Long Short-Term Memory (LSTM)** neural network.

The project explores the fundamentals of autoregressive language modeling, word-level tokenization, sequential learning, and probabilistic text generation through an interactive Streamlit application.

---

## 🌟 Key Features

### 1. Dataset Pipeline – Tiny Shakespeare
- Automated fetching of the Tiny Shakespeare corpus (~1.1 MB, approximately 40,000 lines).
- Text cleaning, noise removal, contraction preservation, and sentence structuring.
- Preparation of sequential text data for language modeling.

### 2. Tokenization & Sequence Generation
- Word-level tokenization using Keras `Tokenizer`.
- Vocabulary construction and word-to-index mapping.
- N-gram sequence generation for supervised next-word prediction.
- Pre-padding using `pad_sequences(..., padding='pre')` to align recent context tokens at the end of input sequences.

### 3. LSTM Model Architecture – V1 Baseline
- **Embedding Layer (100 dimensions):** Converts token IDs into dense vector representations.
- **Single LSTM Layer (150 units):** Learns sequential patterns and contextual relationships.
- **Dropout (0.2):** Helps reduce overfitting during training.
- **Dense Softmax Output Layer:** Produces a probability distribution over the vocabulary for next-word prediction.

### 4. Multi-Word Prediction Engine
- Iterative autoregressive text generation.
- Greedy decoding for deterministic predictions.
- Temperature-based sampling (0.1–1.5) to control generation randomness.
- Top-K probability distribution analysis for candidate next words.

### 5. Interactive Streamlit Web Application
- Shakespeare-inspired interactive dashboard.
- Real-time next-word and multi-word generation.
- Highlighted seed text and predicted tokens.
- Interactive temperature and generation-length sliders.
- Candidate token confidence percentages.

---

## 🧠 Model Architecture

The current implementation uses a sequential neural network architecture designed for word-level autoregressive language modeling.

**Architecture Overview:**

```text
         Input Sequence
                |
                ▼
       Word Token IDs
                |
                ▼
       Embedding Layer
          (100 Dim)
                |
                ▼
        Single LSTM Layer
           (150 Units)
                |
                ▼
          Dropout
            (0.2)
                |
                ▼
       Dense + Softmax
                |
                ▼
      Next Word Probability
         Distribution
                |
                ▼
       Predicted Next Word
```

**Model Configuration:**

| Component | Configuration |
|---|---|
| Embedding Dimension | 100 |
| LSTM Units | 150 |
| LSTM Layers | 1 |
| Dropout | 0.2 |
| Output Activation | Softmax |
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | As configured during training |

---

## 📊 Empirical Validation – V1 Baseline

The first version of the model was successfully trained and tested through the interactive Streamlit interface using classic Shakespearean dramatic prompts.

### Verified Model Status & Metrics

| Metric | Value |
|---|---|
| Model Status | Loaded & Verified ✅ |
| Dataset | Tiny Shakespeare |
| Vocabulary Size | 1,787 unique tokens |
| Maximum Sequence Length | 12 tokens |
| Training Epochs | 25 |
| Training Loss | 3.8421 |
| Loss Function | Sparse Categorical Crossentropy |

*Note: The reported loss is the training result. It should not be interpreted as validation or test performance.*

### 🧪 Live Inference Test Results

The following examples were generated and captured directly from the Streamlit application.

#### Case 1: Coriolanus Civic Speech

**Input Seed:**
```text
what work's my countrymen
```

**Model Prediction:**
```text
in hand where go you shall
```

**Generated Output:**
> what work's my countrymen in hand where go you shall

**Source Play Reference:** Coriolanus, Act I, Scene 1.

#### Case 2: Citizen Revolt Dialogue

**Input Seed:**
```text
let us kill him
```

**Model Prediction:**
```text
and we'll have corn at our
```

**Generated Output:**
> let us kill him and we'll have corn at our

**Source Play Reference:** Coriolanus, Act I, Scene 1.

#### Case 3: Solemn Citizen Decision

**Input Seed:**
```text
you are all resolved rather
```

**Model Prediction:**
```text
to die than to famish and
```

**Generated Output:**
> you are all resolved rather to die than to famish and

**Source Play Reference:** Coriolanus, Act I, Scene 1.

#### Case 4: Dramatic Oratory Opening

**Input Seed:**
```text
before we proceed any further
```

**Model Prediction:**
```text
hear me speak i do not
```

**Generated Output:**
> before we proceed any further hear me speak i do not

**Source Play Reference:** Coriolanus, Act I, Scene 1.

These examples demonstrate that the model can reproduce several local sequential patterns present in the training corpus. They are qualitative inference examples, not a substitute for quantitative evaluation on held-out data.

---

## 📂 Project Directory Structure

```text
next_word_predictor/
│
├── .gitignore
│
├── next_word_predictor.ipynb
│   └── Complete step-by-step implementation
│
├── app.py
│   └── Interactive Streamlit application
│
├── train.py
│   └── Standalone CLI training script
│
├── requirements.txt
│   └── Project dependencies
│
├── build_notebook.py
│   └── Notebook generation script
│
├── README.md
│
└── model/
    ├── shakespeare_lstm.keras
    ├── tokenizer.pkl
    └── config.json
```

*Note: The `model/` directory is generated after training. Trained model files, environments, and cache files are excluded from version control where configured in `.gitignore`.*

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd next_word_predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, on Windows:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook next_word_predictor.ipynb
```

The notebook contains 12 implementation sections:

1. Environment Setup & Library Imports
2. Dataset Acquisition & Loading
3. Data Cleaning & Text Preprocessing
4. Tokenization & Vocabulary Building
5. N-Gram Sequence Generation & Pre-Padding
6. Feature (X) and Target (y) Separation
7. Single-Layer LSTM Architecture
8. Model Training & Checkpoint Callbacks
9. Training Curves & Loss/Accuracy Visualization
10. Multi-Word Prediction Engine
11. Multi-Word Prediction Testing
12. Model & Tokenizer Serialization

### 4. Train Using the Command Line

You can also train the model directly without executing the notebook.

```bash
python train.py --lines 1500 --epochs 25 --batch_size 64
```

**Available Configuration Arguments:**

| Argument | Default | Description |
|---|---:|---|
| `--lines` | 1500 | Number of dataset lines used for training |
| `--epochs` | 25 | Number of training epochs |
| `--batch_size` | 64 | Training batch size |
| `--embedding_dim` | 100 | Embedding vector dimension |
| `--lstm_units` | 150 | Number of LSTM units |

After successful training, the model, tokenizer, and configuration are saved in the `model/` directory.

### 5. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

Enter a Shakespearean phrase or choose a preset prompt to generate the next words interactively.

---

## 🔍 Understanding the Core Concepts

### 1. Autoregressive Language Modeling

The model predicts the next word based on the sequence of previously observed words.

Mathematically:

\[
P(w_t \mid w_1,w_2,\ldots,w_{t-1})
\]

The model learns these conditional probabilities from the training corpus.

For example:

```text
Input:  to be or not
Output: to
```

For multi-word generation, each predicted word is appended to the input sequence and used to predict the following word.

### 2. Why Pre-Padding?

LSTM networks process sequential input from left to right:

\[
h_t = \operatorname{LSTM}(x_t,h_{t-1})
\]

For consistent input dimensions, variable-length sequences are padded to a fixed length.

**Post-Padding:**

```text
[word1, word2, 0, 0, 0]
```

**Pre-Padding:**

```text
[0, 0, 0, word1, word2]
```

Pre-padding places the most recent actual context tokens at the end of the sequence.

This is useful when the model uses the final LSTM output to make its prediction, because the final recurrent steps correspond to actual context words rather than trailing padding.

However, padding alone does not guarantee higher accuracy. Proper masking, sequence handling, and training configuration also influence performance.

### 3. Temperature-Based Sampling

Temperature controls the randomness of token selection.

\[
P_i = \frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
\]

Where:

- \(z_i\): Logit for a candidate token.
- \(T\): Temperature parameter.
- \(P_i\): Adjusted probability of selecting the token.

**Temperature behavior:**

| Temperature | Behavior |
|---|---|
| 0.1 | Highly deterministic predictions |
| 0.5 | More focused generation |
| 1.0 | Original model probability distribution |
| 1.5 | More diverse and unpredictable generation |

Lower temperatures favor high-probability tokens, while higher temperatures produce a flatter probability distribution.

---

## 🔮 Future Development Roadmap

The current implementation establishes a foundational autoregressive language model using word-level tokenization and a single-layer LSTM.

Future development will focus on improving contextual understanding, vocabulary representation, generation quality, and computational efficiency.

### 🗺️ Model Evolution Strategy

| Version | Architecture | Key Improvements | Status |
|---|---|---|---|
| **V1 – Baseline** | Embedding + Single-Layer LSTM | Word-level tokenization, next-word prediction, temperature sampling | ✅ Completed |
| **V2 – Advanced RNN** | Stacked LSTM / GRU + Attention | Subword tokenization, attention mechanisms, richer contextual representations | 🔜 Planned |
| **V3 – Transformer-Based** | Fine-Tuned GPT-2 / DistilGPT-2 | Self-attention, long-range context, improved text generation | 🔮 Future Goal |

### 1. V2 – Advanced Recurrent Neural Networks

**Objective:** Improve contextual learning and sequential language modeling while retaining the computational simplicity of recurrent architectures.

**Planned Enhancements:**

- **Stacked LSTM / GRU Architecture:** Introduce multiple recurrent layers to learn more complex linguistic patterns and sequential dependencies.
- **Attention Mechanism:** Implement Bahdanau or Luong-style attention to assign different importance weights to previous hidden states.
- **Pre-trained Word Embeddings:** Experiment with GloVe or FastText embeddings to initialize the embedding layer with existing semantic representations.
- **Subword Tokenization:** Replace word-level tokenization with Byte-Pair Encoding (BPE) to improve handling of rare words and unfamiliar vocabulary.
- **Improved Regularization:** Experiment with recurrent dropout, early stopping, and learning-rate scheduling.

**Expected Outcome:** Better contextual representations, reduced vocabulary limitations, and more coherent multi-word generation.

### 2. V3 – Transformer-Based Language Modeling

**Objective:** Transition from recurrent sequence modeling to modern Transformer-based autoregressive language modeling.

**Proposed Models:**

| Model | Architecture | Potential Application |
|---|---|---|
| DistilGPT-2 | Lightweight Transformer | Faster fine-tuning and inference |
| GPT-2 | Decoder-only Transformer | Context-aware Shakespearean generation |
| GPT-Neo | Autoregressive Transformer | Experimentation with larger language models |
| Gemma / Mistral | Modern Transformer LLMs | Advanced parameter-efficient fine-tuning |

**Planned Enhancements:**

- **Causal Self-Attention:** Learn relationships between tokens across longer contexts while preventing access to future tokens.
- **Pre-trained Language Models:** Fine-tune existing autoregressive models on the Tiny Shakespeare corpus.
- **Transfer Learning:** Leverage existing linguistic representations rather than learning every language pattern from scratch.
- **LoRA / QLoRA:** Explore parameter-efficient fine-tuning to reduce GPU memory requirements.
- **Context Window Expansion:** Experiment with longer input sequences to generate more coherent dialogue and paragraphs.

**Expected Outcome:** More contextually consistent text generation, improved grammatical structure, and stronger long-range dependency modeling.

### 3. Advanced Text Generation & Decoding Strategies

Beyond model architecture, generation quality can be improved through advanced decoding algorithms.

| Technique | Purpose |
|---|---|
| Top-K Sampling | Restrict token selection to the K most probable candidates |
| Top-P Sampling | Dynamically select candidates based on cumulative probability |
| Repetition Penalty | Reduce repetitive words and phrases |
| Beam Search | Explore multiple candidate sequences simultaneously |
| Temperature Scaling | Control randomness and diversity |
| KV Cache | Reuse previously computed attention states to accelerate Transformer inference |

These techniques will be evaluated to understand their impact on text diversity, fluency, repetition, and contextual consistency.

### 4. Model Evaluation & Performance Benchmarking

Future versions will introduce a systematic evaluation framework to compare different architectures and generation strategies.

**Proposed Evaluation Metrics:**

- **Perplexity:** Measure how effectively the model predicts the next token.
- **Cross-Entropy Loss:** Evaluate next-token prediction performance.
- **Top-K Accuracy:** Measure whether the actual next token appears among the top K predictions.
- **Qualitative Evaluation:** Assess grammatical coherence, contextual relevance, and Shakespearean writing style.
- **Inference Latency:** Measure the time required to generate tokens.
- **Generation Diversity:** Evaluate vocabulary diversity and repetitive output patterns.

Each model version will be benchmarked against the V1 LSTM baseline to quantify improvements in predictive performance and generation quality.

### 🎯 Long-Term Project Vision

The ultimate objective is to transform this baseline next-word prediction system into a progressively advanced Shakespearean language generation platform.

```text
Word-Level LSTM
       |
       ▼
Attention-Based RNN
       |
       ▼
Transformer Fine-Tuning
       |
       ▼
Advanced Text Generation
```

The project will maintain a strong focus on understanding underlying NLP concepts, experimenting with different neural architectures, and evaluating improvements through reproducible experiments.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP & Tokenization | Keras Tokenizer |
| Data Processing | NumPy |
| Data Visualization | Matplotlib |
| Interactive UI | Streamlit |
| Development Environment | Jupyter Notebook |
| Model Serialization | Keras, Pickle |
| Dataset | Tiny Shakespeare |

---

## 📌 Key Learning Outcomes

Through this project, I explored and implemented:

- Fundamentals of Natural Language Processing and language modeling.
- Text preprocessing and word-level tokenization.
- Vocabulary construction and N-gram sequence generation.
- Embedding layers and sequential representation learning.
- LSTM architecture and recurrent neural networks.
- Model training, loss optimization, and regularization.
- Autoregressive text generation.
- Temperature-based probabilistic sampling.
- Model serialization and deployment through Streamlit.
- Evolution from recurrent neural networks toward Transformer-based language models.

---

## 👨‍💻 Author

**Soham Kanse**

Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast

**GitHub:** [sohamkanse1718-dev](https://github.com/sohamkanse1718-dev)

---

⭐ If you find this project interesting, consider giving the repository a star!