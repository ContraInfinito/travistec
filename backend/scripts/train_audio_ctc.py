import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/audio")
WAVS_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../models/audio_ctc_model.keras")

# Audio Parameters
SAMPLE_RATE = 22050
FFT_LENGTH = 384
FRAME_LENGTH = 256
FRAME_STEP = 160

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 50

def load_data():
    """Loads metadata and validates files exist."""
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return None
    
    df = pd.read_csv(METADATA_PATH)
    # Ensure files exist
    valid_data = []
    for idx, row in df.iterrows():
        wav_path = os.path.join(WAVS_PATH, str(row['filename']))
        if os.path.exists(wav_path):
            valid_data.append(row)
        else:
            print(f"Warning: File {wav_path} not found. Skipping.")
    
    if not valid_data:
        print("No valid audio files found. Please populate datasets/audio/wavs and metadata.csv")
        return None
        
    return pd.DataFrame(valid_data)

def process_audio(file_path):
    """Reads wav file and converts to spectrogram."""
    # 1. Read wav
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. Get Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=FFT_LENGTH, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
    
    # 3. Compute magnitude and log-scale
    spectrogram = np.abs(stft)
    spectrogram = np.log1p(spectrogram)
    
    # 4. Transpose to (Time, Features) for RNN
    spectrogram = np.transpose(spectrogram)
    
    # 5. Normalize
    means = np.mean(spectrogram, axis=0)
    stds = np.std(spectrogram, axis=0)
    spectrogram = (spectrogram - means) / (stds + 1e-10)
    
    return spectrogram

def create_dataset(df, char_to_num):
    """Creates a tf.data.Dataset generator."""
    
    def generator():
        for _, row in df.iterrows():
            # Process Audio
            wav_path = os.path.join(WAVS_PATH, str(row['filename']))
            spectrogram = process_audio(wav_path)
            
            # Process Label
            label = row['transcript'].lower()
            label = char_to_num(list(label))
            
            yield spectrogram, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, FFT_LENGTH // 2 + 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )
    
    # Padding is crucial for variable length audio and text
    dataset = dataset.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [None, FFT_LENGTH // 2 + 1], 
            [None]
        ),
        padding_values=(0.0, tf.constant(0, dtype=tf.int64)) # 0 for padding labels
    )
    
    return dataset

def CTCLoss(y_true, y_pred):
    """
    Compute the training-time loss value using CTC.
    y_true: (batch_size, max_label_len) - Integer labels
    y_pred: (batch_size, time_steps, num_classes) - Logits
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    
    # Calculate label length by counting non-zero values (since 0 is padding)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int64), axis=1)
    
    input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")

    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=y_pred,
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits_time_major=False,
        blank_index=-1
    )
    return tf.reduce_mean(loss)

def build_model(input_dim, output_dim):
    """Builds the Deep Learning Model (CNN + RNN + CTC)."""
    
    input_spectrogram = layers.Input((None, input_dim), name="input")
    
    # Expand dims for 2D convolution if needed, or use 1D
    # Using 1D Conv for simplicity on spectrogram columns
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(input_spectrogram)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduce time dimension
    
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Recurrent Layers (The "Memory" of the network)
    # Optimized for speed/performance balance (i5/4060m friendly)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    # output_dim + 1 for the CTC "blank" symbol
    # We output LOGITS (no activation) for numerical stability with tf.nn.ctc_loss
    x = layers.Dense(output_dim + 1, name="output")(x)

    model = keras.models.Model(inputs=input_spectrogram, outputs=x, name="Audio_CTC_Model")
    return model

def main():
    print("--- Starting Audio CTC Training ---")
    
    # 1. Load Data
    df = load_data()
    if df is None:
        return

    print(f"Found {len(df)} audio samples.")
    
    # 2. Create Vocabulary
    # Get all unique characters from transcripts
    all_text = "".join(df['transcript'].tolist()).lower()
    unique_chars = sorted(list(set(all_text)))
    print(f"Vocabulary: {unique_chars}")
    
    # Mapping characters to integers
    # We use mask_token='' to reserve index 0 for padding
    char_to_num = layers.StringLookup(vocabulary=unique_chars, mask_token='')
    num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token='', invert=True)
    
    # 3. Prepare Dataset
    dataset = create_dataset(df, char_to_num)
    
    # 4. Build Model
    # Input dim is FFT_LENGTH // 2 + 1 (frequency bins)
    input_dim = FFT_LENGTH // 2 + 1
    vocab_size = char_to_num.vocabulary_size()
    
    model = build_model(input_dim, vocab_size)
    model.summary()
    
    # 5. Compile
    opt = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=opt, loss=CTCLoss)
    
    # 6. Train
    print("Starting training (Full Dataset - High Capacity)...")
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train on the full dataset for 200 epochs
    history = model.fit(dataset, epochs=200, callbacks=callbacks)
    
    # 7. Save (Final save just in case)
    print(f"Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    
    # Save vocabulary for inference
    vocab_path = MODEL_SAVE_PATH.replace(".keras", "_vocab.txt")
    with open(vocab_path, "w") as f:
        f.write("".join(unique_chars))
    print(f"Vocabulary saved to {vocab_path}")

if __name__ == "__main__":
    main()
