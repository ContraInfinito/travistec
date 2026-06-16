from services.stt_service import STTService
import os
import sys
import pandas as pd

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_model_accuracy():
    print("Initializing STT Service...")
    stt = STTService()

    dataset_dir = os.path.join(os.path.dirname(__file__), '../datasets/audio')
    wavs_dir = os.path.join(dataset_dir, 'wavs')
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')

    if not os.path.exists(metadata_path):
        print("Metadata file not found.")
        return

    df = pd.read_csv(metadata_path)

    print(f"\nTesting {len(df)} files from dataset...")
    print("-" * 60)
    print(f"{'Filename':<15} | {'Expected':<15} | {'Predicted':<15} | {'Match'}")
    print("-" * 60)

    correct = 0
    total = 0

    for index, row in df.iterrows():
        filename = row['filename']
        expected = str(row['transcript']).lower().strip()

        file_path = os.path.join(wavs_dir, filename)
        if not os.path.exists(file_path):
            continue

        try:
            # Use the local_ctc method directly to avoid async/service type check overhead for this test
            predicted = stt.transcribe_local_ctc(file_path).lower().strip()

            is_match = expected == predicted
            if is_match:
                correct += 1
            total += 1

            match_icon = "✅" if is_match else "❌"
            print(f"{filename:<15} | {expected:<15} | {predicted:<15} | {match_icon}")

            # Limit to first 20 for brevity if needed, or run all
            if total >= 20:
                break

        except Exception as e:
            print(f"{filename:<15} | {expected:<15} | ERROR: {e}")

    print("-" * 60)
    if total > 0:
        print(
            f"Accuracy (on first {total}): {correct}/{total} ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    test_model_accuracy()
