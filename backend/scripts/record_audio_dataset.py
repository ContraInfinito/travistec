import os
import sounddevice as sd
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import time

# --- Configuration ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/audio")
WAVS_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")

# Audio Settings
SAMPLE_RATE = 22050  # Must match the training script
DURATION = 3.0       # Seconds per clip (adjust if you speak longer phrases)


def record_clip(filename, duration=DURATION):
    print(f"  [REC] Recording for {duration} seconds...", end="", flush=True)

    # Record audio
    recording = sd.rec(int(duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished

    print(" Done.")

    # Save as WAV
    filepath = os.path.join(WAVS_PATH, filename)
    # Convert to 16-bit PCM for compatibility
    data = (recording * 32767).astype(np.int16)
    wav.write(filepath, SAMPLE_RATE, data)
    print(f"  [SAVE] Saved to {filepath}")


def update_metadata(filename, transcript):
    # Load or create dataframe
    if os.path.exists(METADATA_PATH):
        try:
            df = pd.read_csv(METADATA_PATH)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['filename', 'transcript'])
    else:
        df = pd.DataFrame(columns=['filename', 'transcript'])

    # Add new row
    new_row = pd.DataFrame(
        {'filename': [filename], 'transcript': [transcript]})
    df = pd.concat([df, new_row], ignore_index=True)

    # Save
    df.to_csv(METADATA_PATH, index=False)
    print(f"  [META] Updated metadata.csv with: '{transcript}'")


def main():
    # Ensure directories exist
    if not os.path.exists(WAVS_PATH):
        os.makedirs(WAVS_PATH)

    print("==========================================")
    print("   AUDIO DATASET RECORDER TOOL")
    print("==========================================")
    print(f"Saving to: {WAVS_PATH}")
    print(f"Metadata:  {METADATA_PATH}")
    print("------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Type the text you want to say.")
    print("2. Press ENTER.")
    print(f"3. Speak clearly for {DURATION} seconds.")
    print("4. Type 'q' to quit.")
    print("------------------------------------------")

    # Determine starting index
    count = 1
    if os.path.exists(METADATA_PATH):
        try:
            df = pd.read_csv(METADATA_PATH)
            if not df.empty:
                # Try to extract number from last filename
                last_file = df.iloc[-1]['filename']
                try:
                    # Assumes format audio_XXX.wav
                    count = int(last_file.split('_')[1].split('.')[0]) + 1
                except:
                    count = len(df) + 1
        except:
            pass

    while True:
        print(f"\n--- Sample #{count} ---")
        transcript = input("Enter transcript (or 'q' to quit): ").strip()

        if transcript.lower() == 'q':
            print("Exiting...")
            break

        if not transcript:
            print("Transcript cannot be empty.")
            continue

        filename = f"audio_{count:03d}.wav"

        input(f"Press ENTER to start recording '{transcript}'...")

        try:
            record_clip(filename)
            update_metadata(filename, transcript)
            count += 1
        except Exception as e:
            print(f"Error recording: {e}")
            print("Make sure you have a microphone connected and configured.")


if __name__ == "__main__":
    main()
