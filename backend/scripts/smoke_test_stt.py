import os
import numpy as np
import tensorflow as tf
import keras
import librosa

def smoke_test_stt():
    print("--- Starting STT Model Smoke Test ---")
    
    # Paths
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "../models/audio_ctc_model.keras")
    vocab_path = os.path.join(base_dir, "../models/audio_ctc_model_vocab.txt")
    wav_path = os.path.join(base_dir, "../datasets/audio/wavs/audio_001.wav") # Assuming this exists
    
    if not os.path.exists(model_path):
        print(f"FAIL: Model not found at {model_path}")
        return
    
    if not os.path.exists(vocab_path):
        print(f"FAIL: Vocab not found at {vocab_path}")
        return

    if not os.path.exists(wav_path):
        # Try to find any wav file
        wav_dir = os.path.join(base_dir, "../datasets/audio/wavs")
        wavs = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        if not wavs:
            print(f"FAIL: No wav files found in {wav_dir}")
            return
        wav_path = os.path.join(wav_dir, wavs[0])
        print(f"Using audio file: {wav_path}")

    try:
        # Load Model
        print("Loading model...")
        model = keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        # Load Vocab
        with open(vocab_path, 'r') as f:
            vocab_str = f.read()
        vocab = list(vocab_str)
        print(f"Vocabulary: {vocab}")
        
        # Preprocessing
        print("Preprocessing audio...")
        SAMPLE_RATE = 22050
        FFT_LENGTH = 384
        FRAME_LENGTH = 256
        FRAME_STEP = 160
        
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        stft = librosa.stft(audio, n_fft=FFT_LENGTH, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
        spectrogram = np.abs(stft)
        spectrogram = np.log1p(spectrogram)
        spectrogram = np.transpose(spectrogram)
        
        means = np.mean(spectrogram, axis=0)
        stds = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - means) / (stds + 1e-10)
        
        # Add batch dimension
        X = np.expand_dims(spectrogram, axis=0)
        
        # Predict
        print("Running prediction...")
        preds = model.predict(X, verbose=0)
        print(f"Prediction shape: {preds.shape}")
        
        # Decode
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        preds_t = tf.transpose(preds, perm=[1, 0, 2])
        
        decoded, _ = tf.nn.ctc_greedy_decoder(preds_t, sequence_length=tf.cast(input_len, tf.int32), merge_repeated=True)
        results = tf.sparse.to_dense(decoded[0], default_value=-1)
        
        num_to_char = keras.layers.StringLookup(vocabulary=vocab, mask_token='', invert=True)
        
        res = results[0]
        res = res[res != -1]
        pred_str = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        
        print(f"Predicted Text: '{pred_str}'")
        
        if len(pred_str) > 0:
            print("PASS: Model produced output.")
        else:
            print("WARN: Model produced empty output (might be expected if audio is silence, but unlikely).")
            
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    smoke_test_stt()
