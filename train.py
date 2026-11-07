"""
train.py - Standalone Training Script for Shakespeare Next Word Predictor
Builds, trains, and exports an LSTM-based next-word prediction model.
"""

import os
import re
import json
import pickle
import argparse
import requests
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "shakespeare_lstm.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Fallback text excerpt in case offline
FALLBACK_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
will still be, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

MENENIUS:
Worthy Menenius Agrippa; one that hath always loved the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have had
inkling this fortnight what we intend to do, which now
we'll show 'em in deeds. They say poor suitors have strong
breaths: they shall know we have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help. Alack,
You are transported by calamity
Thither where more attends you, and you slander
The helms of the state, who care for you like fathers,
When you curse them as enemies.

To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
"""


def load_dataset(data_file=DATA_FILE, url=DATA_URL):
    """Loads dataset from file, downloads if missing, or uses fallback."""
    if os.path.exists(data_file):
        print(f"[*] Reading existing dataset file: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[*] Attempting to download Tiny Shakespeare from {url}...")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"[+] Downloaded and saved {len(response.text):,} characters to {data_file}")
            return response.text
    except Exception as e:
        print(f"[-] Could not download directly ({e}). Using embedded Shakespeare text.")

    # Fallback
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(FALLBACK_SHAKESPEARE)
    return FALLBACK_SHAKESPEARE


def clean_text(raw_text, max_lines=1500):
    """
    Cleans raw Shakespeare text:
    - Lowercases text
    - Removes stage direction headers or artifacts
    - Cleans non-standard punctuation while preserving standard tokens
    - Filters empty lines
    - Returns a list of clean sentences/lines
    """
    lines = raw_text.split("\n")
    if max_lines and max_lines < len(lines):
        print(f"[*] Subsetting first {max_lines:,} lines for training efficiency.")
        lines = lines[:max_lines]

    cleaned_lines = []
    for line in lines:
        line = line.strip().lower()
        if not line:
            continue
        # Remove speaker names followed by colon if standalone (e.g., "first citizen:")
        line = re.sub(r'^[a-z0-9\s]+:$', '', line)
        # Clean extra symbols, keep word chars and apostrophes
        line = re.sub(r"[^a-zA-Z0-9\s']", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line.split()) >= 2:  # Must have at least 2 words to form an n-gram
            cleaned_lines.append(line)

    print(f"[+] Processed {len(cleaned_lines):,} valid text lines for sequence generation.")
    return cleaned_lines


def generate_sequences(cleaned_lines):
    """
    Fits Keras Tokenizer and generates n-gram sequences with pre-padding.
    Returns:
    - padded_sequences (np.array)
    - max_sequence_len (int)
    - total_words (int)
    - tokenizer (Tokenizer)
    """
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_lines)
    total_words = len(tokenizer.word_index) + 1  # +1 for 0 padding

    print(f"[+] Vocabulary size: {total_words:,} unique tokens.")

    # Generate N-gram sequences
    input_sequences = []
    for line in cleaned_lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    print(f"[+] Generated {len(input_sequences):,} n-gram training sequences.")

    # Find maximum sequence length
    max_sequence_len = max(len(seq) for seq in input_sequences)
    print(f"[+] Maximum sequence length: {max_sequence_len}")

    # Pre-pad sequences (pad with 0s at the beginning)
    padded_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )
    print(f"[+] Sequences pre-padded to shape: {padded_sequences.shape}")

    return padded_sequences, max_sequence_len, total_words, tokenizer


def build_model(vocab_size, max_seq_len, embedding_dim=100, lstm_units=150):
    """
    Builds the model:
    - Layer 1: Embedding
    - Layer 2: Single LSTM layer
    - Layer 3: Dense output layer with softmax
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_seq_len - 1,
            name="word_embedding"
        ),
        LSTM(lstm_units, return_sequences=False, name="lstm_layer"),
        Dropout(0.2, name="dropout_layer"),
        Dense(vocab_size, activation="softmax", name="output_dense")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        metrics=["accuracy"]
    )
    return model


def predict_next_words(model, tokenizer, seed_text, max_sequence_len, num_words=5, temperature=1.0):
    """
    Generates multiple next words iteratively using the trained LSTM model.
    Supports temperature sampling for natural text diversity.
    """
    output_text = seed_text
    predicted_tokens = []

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        
        preds = model.predict(token_list, verbose=0)[0]
        
        if temperature <= 0.05:
            # Deterministic argmax
            predicted_id = int(np.argmax(preds))
        else:
            # Temperature sampling
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            predicted_id = int(np.argmax(probas))
        
        output_word = tokenizer.index_word.get(predicted_id, "")
        if not output_word or output_word == "<OOV>":
            # Fallback to top token if OOV selected
            top_ids = np.argsort(preds)[::-1]
            for tid in top_ids:
                w = tokenizer.index_word.get(tid, "")
                if w and w != "<OOV>":
                    output_word = w
                    break
        
        output_text += " " + output_word
        predicted_tokens.append(output_word)

    return output_text, predicted_tokens


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Next Word Predictor on Tiny Shakespeare")
    parser.add_argument("--lines", type=int, default=1500, help="Number of text lines to train on")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--lstm_units", type=int, default=150, help="Number of units in LSTM layer")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print(" SHAKESPEARE NEXT WORD PREDICTOR - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    raw_text = load_dataset()

    # 2. Clean & Preprocess
    cleaned_lines = clean_text(raw_text, max_lines=args.lines)

    # 3. Tokenize & Pre-pad N-grams
    padded_seqs, max_seq_len, vocab_size, tokenizer = generate_sequences(cleaned_lines)

    # 4. Split Features (X) and Target Label (y)
    X = padded_seqs[:, :-1]
    y = padded_seqs[:, -1]

    print(f"[+] Feature matrix X shape: {X.shape}")
    print(f"[+] Label vector y shape:   {y.shape}")

    # 5. Build Model
    model = build_model(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units
    )
    model.summary()

    # 6. Train Callbacks
    callbacks = [
        EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="loss", save_best_only=True, verbose=1)
    ]

    # 7. Train Model
    print("\n[*] Starting LSTM Model Training...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Save Artifacts
    print("\n[*] Saving model artifacts...")
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    config = {
        "max_sequence_len": int(max_seq_len),
        "vocab_size": int(vocab_size),
        "embedding_dim": int(args.embedding_dim),
        "lstm_units": int(args.lstm_units),
        "epochs_trained": int(args.epochs),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1])
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print(f"[+] Saved model to:     {MODEL_PATH}")
    print(f"[+] Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"[+] Saved config to:    {CONFIG_PATH}")

    # 9. Test Multi-word Prediction
    print("\n" + "=" * 60)
    print(" SAMPLE MULTI-WORD PREDICTIONS")
    print("=" * 60)
    test_seeds = [
        "To be or not",
        "The king shall",
        "Shall I compare thee",
        "Good citizen hear me"
    ]
    for seed in test_seeds:
        full_text, words = predict_next_words(
            model, tokenizer, seed, max_seq_len, num_words=6, temperature=0.7
        )
        print(f"Seed:       '{seed}'")
        print(f"Predicted:  '{' '.join(words)}'")
        print(f"Full text:  '{full_text}'\n")

    print("[SUCCESS] Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
