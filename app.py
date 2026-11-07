"""
app.py - Interactive Streamlit UI for Shakespeare Next Word Predictor
Provides single and multi-word prediction with temperature control and top-k confidence.
"""

import os
import json
import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Shakespeare Next Word Predictor",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Shakespearean / Modern Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;600&display=swap');
    
    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #E2B714;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Crimson Pro', serif;
        font-size: 1.25rem;
        color: #D1D5DB;
        text-align: center;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background: linear-gradient(135deg, #1A1F2C 0%, #11141D 100%);
        border: 1px solid rgba(226, 183, 20, 0.25);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .seed-badge {
        background-color: #2D3748;
        color: #90CDF4;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        display: inline;
    }
    .pred-badge {
        background: linear-gradient(90deg, #D97706, #B45309);
        color: #FEF3C7;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        display: inline;
        animation: fadeIn 0.5s ease-in;
    }
    .generated-text-container {
        font-family: 'Crimson Pro', serif;
        font-size: 1.35rem;
        line-height: 1.8;
        color: #F3F4F6;
        padding: 1.2rem;
        background: #0D1117;
        border-left: 4px solid #E2B714;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stat-pill {
        display: inline-block;
        background: rgba(226, 183, 20, 0.12);
        color: #E2B714;
        border: 1px solid rgba(226, 183, 20, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "shakespeare_lstm.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


@st.cache_resource
def load_model_and_tokenizer():
    """Caches and loads the trained LSTM model and tokenizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None, None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)

        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict_step(model, tokenizer, current_text, max_sequence_len, temperature=1.0, top_k=5):
    """
    Predicts probability distribution for next single token.
    Returns:
    - chosen_word (str)
    - top_candidates: list of (word, probability)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokens = tokenizer.texts_to_sequences([current_text])[0]
    padded = pad_sequences([tokens], maxlen=max_sequence_len - 1, padding="pre")

    preds = model.predict(padded, verbose=0)[0]

    # Calculate Top Candidates for visualization
    sorted_indices = np.argsort(preds)[::-1]
    top_candidates = []
    for idx in sorted_indices:
        word = tokenizer.index_word.get(idx, "")
        if word and word != "<OOV>":
            top_candidates.append((word, float(preds[idx])))
        if len(top_candidates) >= top_k:
            break

    # Sample chosen word with temperature
    if temperature <= 0.05:
        chosen_id = int(np.argmax(preds))
    else:
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        chosen_id = int(np.argmax(probas))

    chosen_word = tokenizer.index_word.get(chosen_id, "")
    if not chosen_word or chosen_word == "<OOV>":
        chosen_word = top_candidates[0][0] if top_candidates else "thee"

    return chosen_word, top_candidates


def generate_multi_words(model, tokenizer, seed_text, max_sequence_len, num_words, temperature):
    """Generates K words iteratively, tracking step-by-step top probabilities."""
    full_text = seed_text.strip()
    generated_tokens = []
    first_step_distribution = []

    for step in range(num_words):
        next_word, candidates = predict_step(
            model, tokenizer, full_text, max_sequence_len, temperature=temperature
        )
        if step == 0:
            first_step_distribution = candidates
        generated_tokens.append(next_word)
        full_text += " " + next_word

    return full_text, generated_tokens, first_step_distribution


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Generation Controls")
    
    num_words = st.slider(
        "Number of Words to Predict",
        min_value=1,
        max_value=25,
        value=6,
        help="How many continuous words to predict in sequence."
    )

    temperature = st.slider(
        "Sampling Temperature",
        min_value=0.05,
        max_value=1.5,
        value=0.7,
        step=0.05,
        help="Lower values (0.1) are deterministic & greedy; higher values (0.8+) add poetic variety."
    )

    st.markdown("---")
    st.markdown("### 🎭 Quick Shakespeare Prompts")
    prompt_presets = [
        "To be or not to be",
        "Shall I compare thee",
        "Good citizens hear me speak",
        "The king shall",
        "O Romeo Romeo wherefore",
        "Friends Romans countrymen"
    ]
    for p in prompt_presets:
        if st.button(p, use_container_width=True):
            st.session_state["seed_input"] = p

    st.markdown("---")
    st.markdown("### 📊 Model Architecture Info")
    st.markdown("""
    - **Dataset**: Tiny Shakespeare
    - **1st Layer**: Word Embedding (100d)
    - **2nd Layer**: Single LSTM (150 units)
    - **Output Layer**: Dense Softmax (Vocab Size)
    - **Sequence Padding**: Pre-Padding (`padding='pre'`)
    """)

# --- MAIN PAGE HEADER ---
st.markdown("<div class='main-title'>📜 Shakespeare Next Word Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Deep Learning LSTM Architecture for Sequential Word-Level Generation</div>",
    unsafe_allow_html=True
)

# Load model
model, tokenizer, config = load_model_and_tokenizer()

# Status Banner
if model is None:
    st.warning("""
    ⚠️ **No trained model found on disk yet!**
    
    You can train the model in either of two easy ways:
    1. **Open & Run the Notebook**: Open `next_word_predictor.ipynb` in Jupyter and run all cells.
    2. **Or Run via Terminal**: Open PowerShell/Terminal and run:
       ```bash
       python train.py --lines 1500 --epochs 25
       ```
    *The weights and tokenizer will be saved automatically to the `model/` folder.*
    """)
else:
    final_loss = config.get("final_loss")
    if isinstance(final_loss, (int, float)):
        loss_text = f"{final_loss:.4f}"
    else:
        loss_text = "Trained"

    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <span class='stat-pill'>Status: Model Loaded ✅</span>
            <span class='stat-pill'>Vocab Size: {config.get('vocab_size', 'N/A')}</span>
            <span class='stat-pill'>Max Seq Len: {config.get('max_sequence_len', 'N/A')}</span>
            <span class='stat-pill'>Loss Status: {loss_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- USER INPUT AREA ---
default_seed = st.session_state.get("seed_input", "To be or not to be")
user_seed = st.text_input(
    "Enter Seed Prompt (at least 1-2 words):",
    value=default_seed,
    placeholder="e.g. Shall I compare thee or The king shall..."
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("🔮 Predict Next Words", type="primary", use_container_width=True)

if predict_btn or (user_seed and "last_result" in st.session_state):
    if not user_seed.strip():
        st.error("Please enter a non-empty seed prompt.")
    elif model is None:
        st.info("Demonstrating simulated output since model file has not been trained yet. Please run `python train.py` to enable live LSTM inference.")
        # Simulation demo
        simulated_words = ["that", "is", "the", "question", "whether", "'tis"][:num_words]
        st.markdown("### 📝 Generated Passage:")
        st.markdown(
            f"<div class='generated-text-container'><span class='seed-badge'>{user_seed}</span> " +
            " ".join([f"<span class='pred-badge'>{w}</span>" for w in simulated_words]) +
            "</div>",
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Invoking Shakespearean LSTM..."):
            max_seq_len = config.get("max_sequence_len", 15)
            full_text, gen_tokens, top_candidates = generate_multi_words(
                model=model,
                tokenizer=tokenizer,
                seed_text=user_seed,
                max_sequence_len=max_seq_len,
                num_words=num_words,
                temperature=temperature
            )

        st.markdown("---")
        st.markdown("### 📝 Generated Passage:")
        
        # Display with visual badges
        highlighted_tokens = " ".join([f"<span class='pred-badge'>{w}</span>" for w in gen_tokens])
        st.markdown(
            f"<div class='generated-text-container'><span class='seed-badge'>{user_seed}</span> {highlighted_tokens}</div>",
            unsafe_allow_html=True
        )

        # Plain text for easy copy
        st.text_area("Copyable Full Text:", value=full_text, height=75)

        # Top-5 Distribution for the immediate next word
        if top_candidates:
            st.markdown("---")
            st.markdown("### 🎯 Immediate Next Word - Top 5 Probabilities")
            c_cols = st.columns(len(top_candidates))
            for i, (word, prob) in enumerate(top_candidates):
                with c_cols[i]:
                    st.metric(label=f"Rank #{i+1}: '{word}'", value=f"{prob*100:.1f}%")
                    st.progress(min(prob, 1.0))

# --- BOTTOM EXPLANATION ---
with st.expander("📚 How This Model Works (Architecture & Pipeline Breakdown)"):
    st.markdown("""
    #### 1. Data Cleaning & Tokenization
    - The raw Shakespearean text is cleaned, lowercased, and converted into continuous n-gram sequences.
    - An integer ID is assigned to every unique token via Keras `Tokenizer`.

    #### 2. Pre-Padding (`padding='pre'`)
    - Sequences are padded with zeros on the **left side** so that the most recent context words always appear immediately before the target label word.

    #### 3. Single-Layer LSTM Model
    - **Embedding Layer**: Projects sparse token indices into a dense continuous semantic vector space (dim=100).
    - **Single LSTM Layer**: Captures sequential order, recurrent state, and poetic meter (150 hidden units).
    - **Dense Layer (Softmax)**: Computes the probability distribution across all vocabulary words to select the next token.

    #### 4. Multi-Word Iterative Generation
    - The predicted token is appended to the input context, the sequence is re-padded, and fed back into the LSTM to predict subsequent words.
    """)
