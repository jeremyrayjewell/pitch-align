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
    "aeolian major": np.array([0, 2, 4, 5, 7, 8, 10], dtype=int),
    "aeolian minor": np.array([0, 2, 3, 5, 7, 8, 10], dtype=int),
    "harmonic minor": np.array([0, 2, 3, 5, 7, 8, 11], dtype=int),
    "melodic minor": np.array([0, 2, 3, 5, 7, 9, 11], dtype=int),
    "major pentatonic": np.array([0, 2, 4, 7, 9], dtype=int),
    "minor pentatonic": np.array([0, 3, 5, 7, 10], dtype=int),
    "blues": np.array([0, 3, 5, 6, 7, 10], dtype=int),
    "dorian": np.array([0, 2, 3, 5, 7, 9, 10], dtype=int),
    "dorian major": np.array([0, 2, 4, 5, 7, 9, 10], dtype=int),
    "dorian minor": np.array([0, 2, 3, 5, 7, 9, 10], dtype=int),
    "phrygian": np.array([0, 1, 3, 5, 7, 8, 10], dtype=int),
    "lydian": np.array([0, 2, 4, 6, 7, 9, 11], dtype=int),
    "mixolydian": np.array([0, 2, 4, 5, 7, 9, 10], dtype=int),
    "locrian": np.array([0, 1, 3, 5, 6, 8, 10], dtype=int),
    "whole tone": np.array([0, 2, 4, 6, 8, 10], dtype=int),
    "chromatic": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int),
}

AVAILABLE_SCALES = list(SCALE_INTERVALS.keys())


DEFAULT_PROCESSING_SETTINGS = {
    "pitch_align_enabled": True,
    "skip_long_files": False,
    "key": "D",
    "scale": "major",
    "pitch_strength": 0.9,
    "pitch_mix": 1.0,
    "hard_tune": False,
    "range_explorer": False,
    "range_explorer_amount": 0.35,
    "range_explorer_seed": None,
    "dc_remove_enabled": True,
    "highpass_enabled": True,
    "highpass_cutoff": 80.0,
    "low_shelf_enabled": False,
    "low_shelf_gain": 0.0,
    "noise_gate_enabled": False,
    "noise_gate_threshold": -45.0,
    "noise_gate_release": 120.0,
    "mid_boost_enabled": True,
    "mid_boost_gain": 3.0,
    "presence_boost_enabled": False,
    "presence_boost_gain": 0.0,
    "de_esser_enabled": False,
    "de_esser_intensity": 0.25,
    "notch_enabled": False,
    "notch_frequency": 60.0,
    "notch_q": 20.0,
    "high_cut_enabled": False,
    "high_cut_freq": 9000.0,
    "high_cut_mix": 0.35,
    "high_shelf_enabled": False,
    "high_shelf_gain": 0.0,
    "compression_enabled": False,
    "compression_intensity": 0.35,
    "compression_threshold": -18.0,
    "compression_attack": 12.0,
    "compression_release": 90.0,
    "compression_makeup": 0.0,
    "saturation_enabled": False,
    "saturation_amount": 0.2,
    "stereo_width_enabled": False,
    "stereo_width_amount": 0.3,
    "stereo_balance_enabled": False,
    "stereo_balance": 0.0,
    "reverb_enabled": False,
    "reverb_mix": 0.12,
    "reverb_decay": 0.22,
    "reverb_predelay": 20.0,
    "limiter_enabled": True,
    "limiter_ceiling": 0.98,
}


def _preset(**overrides):
    values = DEFAULT_PROCESSING_SETTINGS.copy()
    values.update(overrides)
    return values


PRESETS = {
    "Natural Vocal Align": _preset(),
    "Hard Tune": _preset(
        pitch_strength=1.0,
        pitch_mix=1.0,
        hard_tune=True,
        compression_enabled=True,
        compression_intensity=0.45,
        compression_threshold=-20.0,
        presence_boost_enabled=True,
        presence_boost_gain=2.5,
        de_esser_enabled=True,
        de_esser_intensity=0.35,
    ),
    "Subtle Cleanup": _preset(
        pitch_strength=0.45,
        pitch_mix=0.7,
        mid_boost_enabled=False,
        noise_gate_enabled=True,
        noise_gate_threshold=-52.0,
        noise_gate_release=150.0,
        high_cut_enabled=True,
        high_cut_freq=10000.0,
        high_cut_mix=0.2,
        compression_enabled=True,
        compression_intensity=0.2,
        compression_threshold=-20.0,
        limiter_ceiling=0.96,
    ),
    "Experimental Range Explorer": _preset(
        pitch_strength=0.95,
        pitch_mix=1.0,
        range_explorer=True,
        range_explorer_amount=0.75,
        stereo_width_enabled=True,
        stereo_width_amount=0.2,
        reverb_enabled=True,
        reverb_mix=0.16,
        reverb_decay=0.35,
    ),
    "Bright Vocal Polish": _preset(
        pitch_strength=0.8,
        presence_boost_enabled=True,
        presence_boost_gain=4.0,
        de_esser_enabled=True,
        de_esser_intensity=0.35,
        compression_enabled=True,
        compression_intensity=0.38,
        compression_threshold=-20.0,
        high_shelf_enabled=True,
        high_shelf_gain=3.5,
        reverb_enabled=True,
        reverb_mix=0.1,
        reverb_decay=0.2,
    ),
    "Dark Soft Vocal": _preset(
        pitch_strength=0.75,
        mid_boost_gain=1.5,
        high_cut_enabled=True,
        high_cut_freq=7000.0,
        high_cut_mix=0.55,
        high_shelf_enabled=True,
        high_shelf_gain=-2.5,
        compression_enabled=True,
        compression_intensity=0.28,
        compression_threshold=-19.0,
        saturation_enabled=True,
        saturation_amount=0.1,
        reverb_enabled=True,
        reverb_mix=0.14,
        reverb_decay=0.3,
    ),
}

AVAILABLE_PRESETS = list(PRESETS.keys())
PROCESSING_MODE_PITCH_DSP = "Pitch + DSP"
PROCESSING_MODE_PITCH_ONLY = "Pitch only"
PROCESSING_MODE_DSP_ONLY = "DSP only"
AVAILABLE_PROCESSING_MODES = [
    PROCESSING_MODE_PITCH_DSP,
    PROCESSING_MODE_PITCH_ONLY,
    PROCESSING_MODE_DSP_ONLY,
]


def get_preset_settings(name):
    if name not in PRESETS:
        raise ValueError(f"Unsupported preset: {name}")
    return PRESETS[name].copy()


def validate_processing_mode(mode):
    if mode not in AVAILABLE_PROCESSING_MODES:
        raise ValueError(f"Unsupported processing mode: {mode}")
    return mode


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

    if candidate.resolve() == Path(input_path).resolve():
        return default_path

    if candidate.resolve().parent != OUTPUT_DIR.resolve():
        candidate = OUTPUT_DIR / candidate.name

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
