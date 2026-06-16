"""
Expand the CTC voice command dataset using edge-tts neural TTS.

Strategy: 9 voices × 6 variants (original + 5 augmentations) = 54 samples/command.
Covers all 32 commands. Spanish commands get ES-priority voices; English get EN-priority.

Run:  pip install edge-tts soundfile
      python backend/scripts/generate_tts_dataset.py
"""

import asyncio
import os
import sys
import time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

try:
    import edge_tts
except ImportError:
    sys.exit("edge-tts not installed. Run: pip install edge-tts")

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "audio"
WAVS_PATH    = DATASET_PATH / "wavs"
METADATA_PATH = DATASET_PATH / "metadata.csv"
SAMPLE_RATE  = 22050

# ── Voices ───────────────────────────────────────────────────────────────────
EN_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    # en-US-DavisNeural excluded — rejects single-word inputs
]

ES_VOICES = [
    "es-ES-ElviraNeural",
    "es-ES-AlvaroNeural",
    "es-MX-DaliaNeural",
    "es-MX-JorgeNeural",
]

# Commands better pronounced by Spanish-native voices
SPANISH_COMMANDS = {"aguacate", "carro", "masa", "corporal"}

def _voices_for(command: str) -> list:
    if command in SPANISH_COMMANDS:
        return ES_VOICES + EN_VOICES[:5]
    return EN_VOICES + ES_VOICES[:4]


# ── Vocabulary corrections ────────────────────────────────────────────────────
# Map corrupt/typo labels in existing metadata to their correct form.
LABEL_FIXES = {
    "recomned": "recommend",
}


# ── TTS ───────────────────────────────────────────────────────────────────────
async def _tts_to_wav(text: str, voice: str, wav_path: Path) -> np.ndarray:
    """Generate speech with edge-tts and return audio array at SAMPLE_RATE."""
    tmp_mp3 = wav_path.with_suffix(".tmp.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(tmp_mp3))
    audio, _ = librosa.load(str(tmp_mp3), sr=SAMPLE_RATE)
    tmp_mp3.unlink(missing_ok=True)
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return audio


# ── Augmentation ──────────────────────────────────────────────────────────────
def _augment(audio: np.ndarray, sr: int):
    """Yield (label, audio) pairs for the 5 augmented variants."""
    yield "pitch_m2", librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    yield "pitch_p2", librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    yield "slow09",   librosa.effects.time_stretch(audio, rate=0.9)
    yield "fast11",   librosa.effects.time_stretch(audio, rate=1.1)
    noise = np.random.normal(0, 0.003, len(audio)).astype(np.float32)
    yield "noise",    np.clip(audio + noise, -1.0, 1.0)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    WAVS_PATH.mkdir(parents=True, exist_ok=True)

    # Load and clean existing metadata
    df = pd.read_csv(METADATA_PATH)
    df["transcript"] = df["transcript"].replace(LABEL_FIXES)
    print(f"Existing samples: {len(df)}")

    # Canonical command list (fixed vocabulary)
    commands = sorted(df["transcript"].unique())
    print(f"Commands ({len(commands)}): {commands}\n")

    # Counter for new filenames — start after existing files to avoid collisions
    existing_nums = [
        int(f.stem.split("_")[1])
        for f in WAVS_PATH.glob("tts_*.wav")
        if f.stem.split("_")[1].isdigit()
    ] if any(WAVS_PATH.glob("tts_*.wav")) else []
    counter = max(existing_nums, default=0) + 1

    new_rows = []
    failed  = []

    for cmd_idx, command in enumerate(commands, 1):
        voices = _voices_for(command)
        current_count = (df["transcript"] == command).sum()
        print(f"[{cmd_idx}/{len(commands)}] '{command}' — {current_count} existing -> generating {len(voices)} voices x 6 variants")

        for voice in voices:
            short = voice.split("-")[-1].replace("Neural", "").lower()
            base_name  = f"tts_{counter:05d}_{command}_{short}"
            wav_path   = WAVS_PATH / f"{base_name}.wav"

            try:
                audio = await _tts_to_wav(command, voice, wav_path)
                new_rows.append({"filename": wav_path.name, "transcript": command})
                counter += 1

                for aug_label, aug_audio in _augment(audio, SAMPLE_RATE):
                    aug_name = f"tts_{counter:05d}_{command}_{short}_{aug_label}"
                    aug_path = WAVS_PATH / f"{aug_name}.wav"
                    sf.write(str(aug_path), aug_audio, SAMPLE_RATE)
                    new_rows.append({"filename": aug_path.name, "transcript": command})
                    counter += 1

                await asyncio.sleep(0.25)  # polite rate limiting

            except Exception as e:
                print(f"  FAILED {voice}: {e}")
                failed.append((command, voice, str(e)))

    # Persist updated metadata
    combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(METADATA_PATH, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"New samples generated : {len(new_rows)}")
    print(f"Failed                : {len(failed)}")
    print(f"Total in metadata     : {len(combined)}")
    print("\nSamples per command:")
    print(combined["transcript"].value_counts().to_string())

    if failed:
        print("\nFailed calls:")
        for cmd, voice, err in failed:
            print(f"  {cmd} / {voice}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
