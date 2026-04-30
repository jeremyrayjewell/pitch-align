import librosa
import numpy as np
from scipy import signal

from utils import NOTE_TO_PC, SCALE_INTERVALS, ensure_stereo, normalize_audio, to_mono


def _apply_sos(audio, sos):
    processed = np.zeros_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        processed[:, channel] = signal.sosfiltfilt(sos, audio[:, channel]).astype(np.float32)
    return processed


def dc_offset_removal(audio):
    return (audio - np.mean(audio, axis=0, keepdims=True)).astype(np.float32)


def highpass_filter(audio, sr, cutoff=80.0):
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return _apply_sos(audio, sos)


def low_shelf_boost(audio, sr, gain_db=0.0, cutoff=180.0):
    if abs(gain_db) < 0.01:
        return audio
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    low_band = _apply_sos(audio, sos)
    gain = 10 ** (gain_db / 20.0)
    return audio + low_band * (gain - 1.0)


def noise_gate(audio, sr, threshold_db=-45.0, release_ms=120.0):
    mono = to_mono(audio)
    envelope = np.abs(mono)
    release_samples = max(1, int(sr * release_ms / 1000.0))
    kernel = np.ones(release_samples, dtype=np.float32) / release_samples
    smooth_env = np.convolve(envelope, kernel, mode="same")
    threshold = 10 ** (threshold_db / 20.0)
    gate = np.clip(smooth_env / max(threshold, 1e-6), 0.0, 1.0) ** 1.5
    return audio * gate[:, np.newaxis]


def midrange_boost(audio, sr, gain_db=3.0, low_hz=200.0, high_hz=900.0):
    sos = signal.butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    band = _apply_sos(audio, sos)
    gain = 10 ** (gain_db / 20.0)
    return audio + band * (gain - 1.0)


def presence_boost(audio, sr, gain_db=0.0, low_hz=2000.0, high_hz=5000.0):
    if abs(gain_db) < 0.01:
        return audio
    sos = signal.butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    band = _apply_sos(audio, sos)
    gain = 10 ** (gain_db / 20.0)
    return audio + band * (gain - 1.0)


def de_esser(audio, sr, intensity=0.25, low_hz=4500.0, high_hz=9500.0):
    if intensity <= 0:
        return audio

    sos = signal.butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    sibilance_band = _apply_sos(audio, sos)
    envelope = np.abs(to_mono(sibilance_band))
    window = max(1, int(sr * 0.010))
    kernel = np.ones(window, dtype=np.float32) / window
    envelope = np.convolve(envelope, kernel, mode="same")
    threshold = np.percentile(envelope, 75) + 1e-6
    reduction = np.clip((envelope - threshold) / threshold, 0.0, 1.0) * intensity
    return audio - sibilance_band * reduction[:, np.newaxis]


def notch_filter(audio, sr, center_hz=60.0, q=20.0):
    b, a = signal.iirnotch(center_hz, q, fs=sr)
    filtered = np.zeros_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        filtered[:, channel] = signal.filtfilt(b, a, audio[:, channel]).astype(np.float32)
    return filtered


def high_frequency_smoothing(audio, sr, cutoff=9000.0, wet=0.35):
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    smoothed = _apply_sos(audio, sos)
    return audio * (1.0 - wet) + smoothed * wet


def high_shelf_boost(audio, sr, gain_db=0.0, cutoff=6500.0):
    if abs(gain_db) < 0.01:
        return audio
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    high_band = _apply_sos(audio, sos)
    gain = 10 ** (gain_db / 20.0)
    return audio + high_band * (gain - 1.0)


def compress_audio(
    audio,
    sr,
    intensity=0.4,
    threshold_db=-18.0,
    attack_ms=12.0,
    release_ms=90.0,
    makeup_db=0.0,
):
    if intensity <= 0:
        if abs(makeup_db) < 0.01:
            return audio
        return audio * (10 ** (makeup_db / 20.0))

    mono = to_mono(audio)
    power = mono**2
    attack_coeff = np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    env = np.zeros_like(power, dtype=np.float32)

    for idx, sample in enumerate(power):
        previous = env[idx - 1] if idx else sample
        coeff = attack_coeff if sample > previous else release_coeff
        env[idx] = coeff * previous + (1.0 - coeff) * sample

    rms = np.sqrt(env + 1e-8)
    threshold = 10 ** (threshold_db / 20.0)
    ratio = 1.0 + intensity * 5.0
    over = np.maximum(rms / threshold, 1.0)
    gain = np.where(over > 1.0, over ** (-(ratio - 1.0) / ratio), 1.0)
    makeup = 10 ** (makeup_db / 20.0)
    return audio * gain[:, np.newaxis] * makeup


def soft_saturation(audio, amount=0.0):
    if amount <= 0:
        return audio
    drive = 1.0 + amount * 4.0
    return np.tanh(audio * drive) / np.tanh(drive)


def stereo_widen(audio, sr, amount=0.3, max_delay_ms=18.0):
    stereo = ensure_stereo(audio)
    if amount <= 0:
        return stereo

    delay_samples = max(1, int(sr * max_delay_ms * amount / 1000.0))
    delayed_right = np.roll(stereo[:, 1], delay_samples)
    delayed_right[:delay_samples] = stereo[:delay_samples, 1]

    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - delayed_right)
    widened_left = mid + side * (1.0 + amount)
    widened_right = mid - side * (1.0 + amount)
    return np.column_stack([widened_left, widened_right])


def stereo_balance(audio, amount=0.0):
    stereo = ensure_stereo(audio)
    amount = float(np.clip(amount, -1.0, 1.0))
    left_gain = 1.0 - max(0.0, amount)
    right_gain = 1.0 + min(0.0, amount)
    return np.column_stack([stereo[:, 0] * left_gain, stereo[:, 1] * right_gain])


def add_reverb(audio, sr, wet=0.15, decay=0.22, predelay_ms=20.0):
    stereo = ensure_stereo(audio)
    if wet <= 0:
        return stereo

    predelay_samples = max(0, int(sr * predelay_ms / 1000.0))
    ir_length = max(32, int(sr * decay))
    impulse = np.zeros(predelay_samples + ir_length, dtype=np.float32)
    t = np.linspace(0.0, decay, ir_length, endpoint=False)
    impulse[predelay_samples:] = np.exp(-6.0 * t / max(decay, 1e-4)).astype(np.float32)
    if predelay_samples < len(impulse):
        impulse[predelay_samples] = 1.0

    wet_audio = np.zeros_like(stereo, dtype=np.float32)
    for channel in range(stereo.shape[1]):
        wet_audio[:, channel] = signal.fftconvolve(stereo[:, channel], impulse, mode="full")[: len(stereo)]

    return stereo * (1.0 - wet) + wet_audio * wet


def limiter(audio, ceiling=0.98):
    peak = np.max(np.abs(audio))
    if peak <= ceiling or peak == 0:
        return audio.astype(np.float32)
    return (audio * (ceiling / peak)).astype(np.float32)


def _scale_pitch_classes(key, scale):
    if key not in NOTE_TO_PC:
        raise ValueError(f"Unsupported key: {key}")
    if scale not in SCALE_INTERVALS:
        raise ValueError(f"Unsupported scale: {scale}")
    root = NOTE_TO_PC[key]
    return (root + SCALE_INTERVALS[scale]) % 12


def _nearest_scale_midi(midi_value, scale_pcs):
    octave = int(np.floor(midi_value / 12.0))
    candidates = []
    for octave_offset in (-1, 0, 1):
        base = (octave + octave_offset) * 12
        candidates.extend((scale_pcs + base).tolist())
    candidates = np.array(candidates, dtype=np.float32)
    return candidates[np.argmin(np.abs(candidates - midi_value))]


def _smooth_pitch_track(values, window=7):
    if len(values) == 0:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _resample_to_length(samples, target_length, rate):
    if target_length <= 1 or len(samples) <= 1 or np.isclose(rate, 1.0, atol=1e-4):
        return samples[:target_length]

    shifted = librosa.resample(samples, orig_sr=1.0, target_sr=float(rate), res_type="soxr_hq")
    if len(shifted) >= target_length:
        start = max(0, (len(shifted) - target_length) // 2)
        return shifted[start : start + target_length]

    pad_total = target_length - len(shifted)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(shifted, (pad_before, pad_after), mode="edge")


def _pitch_shift_frame(frame, semitone_shift):
    if abs(semitone_shift) < 0.01:
        return frame.astype(np.float32)

    rate = 2 ** (semitone_shift / 12.0)
    shifted = np.zeros_like(frame, dtype=np.float32)
    for channel in range(frame.shape[1]):
        shifted[:, channel] = _resample_to_length(frame[:, channel], frame.shape[0], rate)
    return shifted


def pitch_align(audio, sr, key="D", scale="major", strength=0.9, mix=1.0):
    stereo_audio = ensure_stereo(audio).astype(np.float32)
    mono = to_mono(stereo_audio)
    if len(mono) < 4096:
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    frame_length = 2048
    hop_length = 256
    scale_pcs = _scale_pitch_classes(key, scale)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        mono,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if f0 is None or np.all(~np.isfinite(f0)):
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    midi = librosa.hz_to_midi(f0)
    voiced = np.isfinite(midi)
    if voiced_flag is not None:
        voiced &= voiced_flag.astype(bool)
    if not np.any(voiced):
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    target_midi = np.copy(midi)
    target_midi[voiced] = np.array(
        [_nearest_scale_midi(midi_value, scale_pcs) for midi_value in midi[voiced]],
        dtype=np.float32,
    )

    semitone_shift = np.zeros_like(target_midi, dtype=np.float32)
    semitone_shift[voiced] = target_midi[voiced] - midi[voiced]
    semitone_shift = np.clip(semitone_shift, -6.0, 6.0)
    semitone_shift[voiced] *= float(np.clip(strength, 0.0, 1.0))
    semitone_shift[voiced] = _smooth_pitch_track(semitone_shift[voiced], window=11)
    semitone_shift = _smooth_pitch_track(semitone_shift, window=5)
    mix = float(np.clip(mix, 0.0, 1.0))

    if voiced_prob is None:
        voiced_strength = voiced.astype(np.float32)
    else:
        voiced_strength = np.where(voiced, voiced_prob.astype(np.float32), 0.0)
        voiced_strength = np.clip(voiced_strength, 0.0, 1.0)

    padded = np.pad(stereo_audio, ((frame_length // 2, frame_length // 2), (0, 0)), mode="reflect")
    output = np.zeros((len(padded), stereo_audio.shape[1]), dtype=np.float32)
    weights = np.zeros(len(padded), dtype=np.float32)
    analysis_window = np.hanning(frame_length).astype(np.float32)

    for frame_idx, shift in enumerate(semitone_shift):
        center = frame_idx * hop_length + frame_length // 2
        start = center - frame_length // 2
        end = start + frame_length
        if end > len(padded):
            break

        frame = padded[start:end]
        voiced_amount = float(voiced_strength[frame_idx]) if frame_idx < len(voiced_strength) else 0.0
        corrected = _pitch_shift_frame(frame, float(shift))
        blend = voiced_amount * mix * analysis_window
        mixed = frame * (1.0 - blend[:, np.newaxis]) + corrected * blend[:, np.newaxis]

        output[start:end] += mixed * analysis_window[:, np.newaxis]
        weights[start:end] += analysis_window

    silent = weights < 1e-6
    weights[silent] = 1.0
    output /= weights[:, np.newaxis]
    output[silent] = padded[silent]
    output = output[frame_length // 2 : frame_length // 2 + len(stereo_audio)]

    if audio.ndim == 1:
        return to_mono(output)
    if audio.shape[1] == 1:
        return output[:, :1]
    return output


def process_chain(audio, sr, settings):
    processed = audio.astype(np.float32).copy()

    if settings.get("pitch_align_enabled", True):
        processed = pitch_align(
            processed,
            sr,
            key=settings.get("key", "D"),
            scale=settings.get("scale", "major"),
            strength=settings.get("pitch_strength", 0.9),
            mix=settings.get("pitch_mix", 1.0),
        )

    if settings.get("dc_remove_enabled"):
        processed = dc_offset_removal(processed)

    if settings.get("highpass_enabled"):
        processed = highpass_filter(processed, sr, cutoff=settings.get("highpass_cutoff", 80.0))

    if settings.get("low_shelf_enabled"):
        processed = low_shelf_boost(processed, sr, gain_db=settings.get("low_shelf_gain", 0.0))

    if settings.get("noise_gate_enabled"):
        processed = noise_gate(
            processed,
            sr,
            threshold_db=settings.get("noise_gate_threshold", -45.0),
            release_ms=settings.get("noise_gate_release", 120.0),
        )

    if settings.get("mid_boost_enabled"):
        processed = midrange_boost(processed, sr, gain_db=settings.get("mid_boost_gain", 3.0))

    if settings.get("presence_boost_enabled"):
        processed = presence_boost(processed, sr, gain_db=settings.get("presence_boost_gain", 0.0))

    if settings.get("de_esser_enabled"):
        processed = de_esser(processed, sr, intensity=settings.get("de_esser_intensity", 0.25))

    if settings.get("notch_enabled"):
        processed = notch_filter(
            processed,
            sr,
            center_hz=settings.get("notch_frequency", 60.0),
            q=settings.get("notch_q", 20.0),
        )

    if settings.get("high_cut_enabled"):
        processed = high_frequency_smoothing(
            processed,
            sr,
            cutoff=settings.get("high_cut_freq", 9000.0),
            wet=settings.get("high_cut_mix", 0.35),
        )

    if settings.get("high_shelf_enabled"):
        processed = high_shelf_boost(processed, sr, gain_db=settings.get("high_shelf_gain", 0.0))

    if settings.get("compression_enabled"):
        processed = compress_audio(
            processed,
            sr,
            intensity=settings.get("compression_intensity", 0.4),
            threshold_db=settings.get("compression_threshold", -18.0),
            attack_ms=settings.get("compression_attack", 12.0),
            release_ms=settings.get("compression_release", 90.0),
            makeup_db=settings.get("compression_makeup", 0.0),
        )

    if settings.get("saturation_enabled"):
        processed = soft_saturation(processed, amount=settings.get("saturation_amount", 0.0))

    if settings.get("stereo_width_enabled"):
        processed = stereo_widen(processed, sr, amount=settings.get("stereo_width_amount", 0.3))

    if settings.get("stereo_balance_enabled"):
        processed = stereo_balance(processed, amount=settings.get("stereo_balance", 0.0))

    if settings.get("reverb_enabled"):
        processed = add_reverb(
            processed,
            sr,
            wet=settings.get("reverb_mix", 0.15),
            decay=settings.get("reverb_decay", 0.22),
            predelay_ms=settings.get("reverb_predelay", 20.0),
        )

    if settings.get("limiter_enabled"):
        processed = limiter(processed, ceiling=settings.get("limiter_ceiling", 0.98))

    return normalize_audio(processed)
