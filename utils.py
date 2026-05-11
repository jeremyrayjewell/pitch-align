from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOG_PATH = OUTPUT_DIR / "pitch-align.log"
SUPPORTED_INPUT_EXTENSIONS = (".wav", ".mp3", ".m4a")

NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

SCALE_INTERVALS = {
    "major": np.array([0, 2, 4, 5, 7, 9, 11], dtype=int),
    "minor": np.array([0, 2, 3, 5, 7, 8, 10], dtype=int),
    "harmonic minor": np.array([0, 2, 3, 5, 7, 8, 11], dtype=int),
    "melodic minor": np.array([0, 2, 3, 5, 7, 9, 11], dtype=int),
    "major pentatonic": np.array([0, 2, 4, 7, 9], dtype=int),
    "minor pentatonic": np.array([0, 3, 5, 7, 10], dtype=int),
    "blues": np.array([0, 3, 5, 6, 7, 10], dtype=int),
    "dorian": np.array([0, 2, 3, 5, 7, 9, 10], dtype=int),
    "phrygian": np.array([0, 1, 3, 5, 7, 8, 10], dtype=int),
    "lydian": np.array([0, 2, 4, 6, 7, 9, 11], dtype=int),
    "mixolydian": np.array([0, 2, 4, 5, 7, 9, 10], dtype=int),
    "locrian": np.array([0, 1, 3, 5, 6, 8, 10], dtype=int),
    "whole tone": np.array([0, 2, 4, 6, 8, 10], dtype=int),
    "chromatic": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int),
}


def ensure_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def append_log(message):
    ensure_directories()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def load_audio(path):
    suffix = Path(path).suffix.lower()

    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        try:
            audio, sr = librosa.load(path, sr=None, mono=False)
        except Exception as fallback_exc:
            if suffix == ".m4a":
                raise RuntimeError(
                    "Could not decode the .m4a file. Install an AAC-capable backend such as FFmpeg and try again."
                ) from fallback_exc
            raise fallback_exc from exc

        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        else:
            audio = audio.T
        return audio.astype(np.float32), sr

    return audio.astype(np.float32), sr


def make_output_path(input_path, output_path=None):
    ensure_directories()
    input_name = Path(input_path).stem
    default_path = OUTPUT_DIR / f"{input_name}_aligned.wav"

    if not output_path:
        return default_path

    candidate = Path(output_path)
    if candidate.suffix.lower() != ".wav":
        candidate = candidate.with_suffix(".wav")

    if candidate.resolve().parent != OUTPUT_DIR.resolve():
        candidate = OUTPUT_DIR / candidate.name

    if candidate.resolve() == Path(input_path).resolve():
        candidate = default_path

    return candidate


def save_audio(path, audio, sr):
    ensure_directories()
    safe_audio = normalize_audio(audio)
    sf.write(path, safe_audio, sr)


def normalize_audio(audio, peak=0.98):
    if audio.size == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val <= peak or max_val == 0:
        return audio.astype(np.float32)
    return (audio * (peak / max_val)).astype(np.float32)


def to_mono(audio):
    if audio.ndim == 1 or audio.shape[1] == 1:
        return audio.reshape(-1)
    return np.mean(audio, axis=1)


def ensure_stereo(audio):
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio
