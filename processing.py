import time

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from utils import NOTE_TO_PC, SCALE_INTERVALS, ensure_stereo, normalize_audio, to_mono


FMIN_HZ = 65.40639
FMAX_HZ = 2093.00452


class ProcessingCancelled(Exception):
    pass


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("Processing cancelled by user.")


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
    envelope = np.abs(mono).astype(np.float32)
    release_samples = max(1, int(sr * release_ms / 1000.0))
    smooth_env = uniform_filter1d(envelope, size=release_samples, mode="nearest")
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


def _hz_to_midi(frequencies):
    frequencies = np.asarray(frequencies, dtype=np.float32)
    midi = np.full(frequencies.shape, np.nan, dtype=np.float32)
    valid = frequencies > 0
    midi[valid] = 69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)
    return midi


def _nearest_scale_midi_values(midi_values, scale_pcs):
    if len(midi_values) == 0:
        return midi_values.astype(np.float32)

    octaves = np.floor(midi_values / 12.0).astype(np.int32)
    octave_bases = ((octaves[:, np.newaxis] + np.array([-1, 0, 1], dtype=np.int32)) * 12).astype(np.float32)
    candidates = octave_bases[:, :, np.newaxis] + scale_pcs[np.newaxis, np.newaxis, :].astype(np.float32)
    candidates = candidates.reshape(len(midi_values), -1)
    nearest = np.argmin(np.abs(candidates - midi_values[:, np.newaxis]), axis=1)
    return candidates[np.arange(len(midi_values)), nearest].astype(np.float32)


def _scale_candidate_matrix(midi_values, scale_pcs):
    octaves = np.floor(midi_values / 12.0).astype(np.int32)
    octave_bases = ((octaves[:, np.newaxis] + np.array([-1, 0, 1], dtype=np.int32)) * 12).astype(np.float32)
    candidates = octave_bases[:, :, np.newaxis] + scale_pcs[np.newaxis, np.newaxis, :].astype(np.float32)
    candidates = candidates.reshape(len(midi_values), -1)
    candidates.sort(axis=1)
    return candidates.astype(np.float32)


def _range_explorer_midi_values(midi_values, scale_pcs, exploration=0.35, seed=None):
    if len(midi_values) == 0:
        return midi_values.astype(np.float32)

    candidates = _scale_candidate_matrix(midi_values, scale_pcs)
    nearest_indices = np.argmin(np.abs(candidates - midi_values[:, np.newaxis]), axis=1)
    explored = candidates[np.arange(len(midi_values)), nearest_indices].astype(np.float32)

    exploration = float(np.clip(exploration, 0.0, 1.0))
    if exploration <= 1e-4:
        return explored

    rng = np.random.default_rng(seed)
    max_step_span = max(1, int(round(1.0 + exploration * 3.0)))
    segment_min = max(1, int(round(8 - exploration * 5.0)))
    segment_max = max(segment_min, int(round(18 - exploration * 10.0)))

    frame_idx = 0
    while frame_idx < len(midi_values):
        segment_length = int(rng.integers(segment_min, segment_max + 1))
        end_idx = min(len(midi_values), frame_idx + segment_length)

        local_candidates = candidates[frame_idx]
        local_nearest = int(nearest_indices[frame_idx])
        lower_bound = max(0, local_nearest - max_step_span)
        upper_bound = min(len(local_candidates) - 1, local_nearest + max_step_span)
        chosen_index = int(rng.integers(lower_bound, upper_bound + 1))
        explored[frame_idx:end_idx] = local_candidates[chosen_index]
        frame_idx = end_idx

    return explored


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

    shifted_length = max(1, int(round(len(samples) * rate)))
    source_positions = np.arange(len(samples), dtype=np.float32)
    target_positions = np.linspace(0.0, len(samples) - 1, num=shifted_length, dtype=np.float32)
    shifted = np.interp(target_positions, source_positions, samples).astype(np.float32)
    if len(shifted) >= target_length:
        start = max(0, (len(shifted) - target_length) // 2)
        return shifted[start : start + target_length]

    pad_total = target_length - len(shifted)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(shifted, (pad_before, pad_after), mode="edge")


def _resample_mono(samples, target_length):
    if target_length <= 1 or len(samples) <= 1 or target_length == len(samples):
        return samples[:target_length].astype(np.float32)

    source_positions = np.arange(len(samples), dtype=np.float32)
    target_positions = np.linspace(0.0, len(samples) - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _pitch_shift_frame(frame, semitone_shift):
    if abs(semitone_shift) < 0.01:
        return frame.astype(np.float32)

    rate = 2 ** (semitone_shift / 12.0)
    shifted = np.zeros_like(frame, dtype=np.float32)
    for channel in range(frame.shape[1]):
        shifted[:, channel] = _resample_to_length(frame[:, channel], frame.shape[0], rate)
    return shifted


def _estimate_pitch_track(mono, sr, frame_length, hop_length, fmin, fmax):
    if len(mono) < frame_length:
        return np.array([], dtype=np.float32), np.array([], dtype=bool), np.array([], dtype=np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(mono, frame_length)[::hop_length]
    if len(frames) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=bool), np.array([], dtype=np.float32)

    window = np.hanning(frame_length).astype(np.float32)
    spectra = np.fft.rfft(frames * window[np.newaxis, :], axis=1)
    magnitudes = np.abs(spectra).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr)

    valid_bins = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    if len(valid_bins) == 0:
        empty = np.zeros(len(frames), dtype=np.float32)
        return empty, np.zeros(len(frames), dtype=bool), empty

    band_magnitudes = magnitudes[:, valid_bins]
    peak_indices = np.argmax(band_magnitudes, axis=1)
    peak_bins = valid_bins[peak_indices]
    peak_values = band_magnitudes[np.arange(len(frames)), peak_indices]
    mean_values = np.mean(band_magnitudes, axis=1) + 1e-6
    voiced_strength = np.clip((peak_values / mean_values - 1.0) / 4.0, 0.0, 1.0).astype(np.float32)
    voiced_flag = voiced_strength > 0.08

    f0 = freqs[peak_bins].astype(np.float32)
    f0 = np.where(voiced_flag, f0, np.nan).astype(np.float32)
    return f0, voiced_flag, voiced_strength


def pitch_align(
    audio,
    sr,
    key="D",
    scale="major",
    strength=0.9,
    mix=1.0,
    skip_long_files=False,
    progress_callback=None,
    hard_tune=False,
    range_explorer=False,
    range_explorer_amount=0.35,
    range_explorer_seed=None,
    cancel_event=None,
):
    _check_cancel(cancel_event)
    stereo_audio = ensure_stereo(audio).astype(np.float32)
    mono = to_mono(stereo_audio)

    if len(mono) < 4096:
        if progress_callback:
            progress_callback("Skipping pitch alignment (file too short)")
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    duration = len(mono) / sr
    if skip_long_files and duration > 60:
        if progress_callback:
            progress_callback(f"Skipping pitch alignment (long file: {duration:.1f}s)")
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)
    if duration > 60 and progress_callback:
        progress_callback(f"Warning: Long file ({duration:.1f}s) - pitch detection may be slow")

    frame_length = 2048
    hop_length = 256
    if duration > 90.0:
        hop_length = 2048
    elif duration > 40.0:
        hop_length = 1024
    elif duration > 20.0:
        hop_length = 512

    scale_pcs = _scale_pitch_classes(key, scale)
    synthesis_total_frames = 1 + max(0, (len(mono) - frame_length) // hop_length)

    analysis_sr = min(sr, 16000)
    analysis_mono = mono
    analysis_frame_length = frame_length
    analysis_hop_length = hop_length
    if sr > analysis_sr:
        analysis_length = max(1, int(round(len(mono) * analysis_sr / sr)))
        analysis_mono = _resample_mono(mono, analysis_length)
        frame_scale = analysis_sr / sr
        analysis_frame_length = max(1024, int(round(frame_length * frame_scale)))
        analysis_hop_length = max(128, int(round(hop_length * frame_scale)))

    analysis_method = "fft"

    if progress_callback:
        expected_frames = 0
        if len(analysis_mono) >= analysis_frame_length:
            expected_frames = 1 + max(0, (len(analysis_mono) - analysis_frame_length) // analysis_hop_length)
        progress_callback(
            f"Starting pitch analysis ({analysis_method})... "
            f"duration={duration:.1f}s analysis_sr={analysis_sr} "
            f"frame_length={analysis_frame_length} hop_length={analysis_hop_length} frames~{expected_frames}"
        )

    pyin_start = time.time()
    _check_cancel(cancel_event)
    f0, voiced_flag, voiced_prob = _estimate_pitch_track(
        analysis_mono,
        analysis_sr,
        analysis_frame_length,
        analysis_hop_length,
        FMIN_HZ,
        FMAX_HZ,
    )
    pyin_time = time.time() - pyin_start

    if progress_callback:
        total_frames = 0 if f0 is None else len(f0)
        if voiced_flag is not None:
            voiced_frames = int(np.sum(voiced_flag))
        else:
            voiced_frames = 0 if f0 is None else int(np.sum(np.isfinite(f0)))
        voiced_pct = (100.0 * voiced_frames / total_frames) if total_frames else 0.0
        progress_callback(
            f"Pitch analysis complete ({pyin_time:.1f}s) - voiced {voiced_frames}/{total_frames} ({voiced_pct:.1f}%)"
        )

    if f0 is None or np.all(~np.isfinite(f0)):
        if progress_callback:
            progress_callback("No pitch detected - skipping alignment")
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    if progress_callback:
        progress_callback("Converting to MIDI...")

    _check_cancel(cancel_event)
    midi = _hz_to_midi(f0)
    voiced = np.isfinite(midi)
    if voiced_flag is not None:
        voiced &= voiced_flag.astype(bool)
    if not np.any(voiced):
        if progress_callback:
            progress_callback("No voiced frames found - skipping alignment")
        return stereo_audio if audio.ndim > 1 else audio.astype(np.float32)

    if progress_callback:
        progress_callback("Mapping to scale...")

    _check_cancel(cancel_event)
    target_midi = np.copy(midi)
    voiced_midi = midi[voiced].astype(np.float32)
    if range_explorer:
        if progress_callback:
            progress_callback("Exploring scale range...")
        target_midi[voiced] = _range_explorer_midi_values(
            voiced_midi,
            scale_pcs,
            exploration=range_explorer_amount,
            seed=range_explorer_seed,
        )
    else:
        target_midi[voiced] = _nearest_scale_midi_values(voiced_midi, scale_pcs)

    semitone_shift = np.zeros_like(target_midi, dtype=np.float32)
    semitone_shift[voiced] = target_midi[voiced] - midi[voiced]
    semitone_shift = np.clip(semitone_shift, -6.0, 6.0)

    if hard_tune:
        effective_strength = max(float(np.clip(strength, 0.0, 1.0)), 0.98)
        semitone_shift[voiced] *= effective_strength
        semitone_shift[voiced] = _smooth_pitch_track(semitone_shift[voiced], window=3)
        semitone_shift = _smooth_pitch_track(semitone_shift, window=3)
        mix = max(float(np.clip(mix, 0.0, 1.0)), 0.98)
        if progress_callback:
            progress_callback("Hard tune mode active")
    else:
        semitone_shift[voiced] *= float(np.clip(strength, 0.0, 1.0))
        semitone_shift[voiced] = _smooth_pitch_track(semitone_shift[voiced], window=11)
        semitone_shift = _smooth_pitch_track(semitone_shift, window=5)
        mix = float(np.clip(mix, 0.0, 1.0))

    if voiced_prob is None:
        voiced_strength = voiced.astype(np.float32)
    else:
        voiced_strength = np.where(voiced, voiced_prob.astype(np.float32), 0.0)
        voiced_strength = np.clip(voiced_strength, 0.0, 1.0)
    if hard_tune:
        voiced_strength = np.where(voiced, np.maximum(voiced_strength, 0.85), 0.0)

    if len(semitone_shift) != synthesis_total_frames and synthesis_total_frames > 0:
        if len(semitone_shift) == 1:
            semitone_shift = np.full(synthesis_total_frames, float(semitone_shift[0]), dtype=np.float32)
            voiced_strength = np.full(synthesis_total_frames, float(voiced_strength[0]), dtype=np.float32)
        else:
            analysis_centers = (
                np.arange(len(semitone_shift), dtype=np.float32) * analysis_hop_length + analysis_frame_length / 2.0
            ) / float(analysis_sr)
            synthesis_centers = (
                np.arange(synthesis_total_frames, dtype=np.float32) * hop_length + frame_length / 2.0
            ) / float(sr)
            semitone_shift = np.interp(
                synthesis_centers,
                analysis_centers,
                semitone_shift.astype(np.float32),
                left=float(semitone_shift[0]),
                right=float(semitone_shift[-1]),
            ).astype(np.float32)
            voiced_strength = np.interp(
                synthesis_centers,
                analysis_centers,
                voiced_strength.astype(np.float32),
                left=float(voiced_strength[0]),
                right=float(voiced_strength[-1]),
            ).astype(np.float32)

    if progress_callback:
        progress_callback("Preparing audio frames...")

    _check_cancel(cancel_event)
    padded = np.pad(stereo_audio, ((frame_length // 2, frame_length // 2), (0, 0)), mode="reflect")
    output = np.zeros((len(padded), stereo_audio.shape[1]), dtype=np.float32)
    weights = np.zeros(len(padded), dtype=np.float32)
    analysis_window = np.hanning(frame_length).astype(np.float32)

    total_frames = len(semitone_shift)
    frame_start_time = time.time()

    for frame_idx, shift in enumerate(semitone_shift):
        if frame_idx % 10 == 0:
            _check_cancel(cancel_event)
        if frame_idx % 50 == 0 and progress_callback:
            progress = int((frame_idx / total_frames) * 100)
            elapsed = time.time() - frame_start_time
            eta = (elapsed / (frame_idx + 1)) * (total_frames - frame_idx) if frame_idx > 0 else 0
            progress_callback(f"Processing frames... ({progress}%, ETA: {eta:.1f}s)")

        center = frame_idx * hop_length + frame_length // 2
        start = center - frame_length // 2
        end = start + frame_length
        if end > len(padded):
            break

        frame = padded[start:end]
        voiced_amount = float(voiced_strength[frame_idx]) if frame_idx < len(voiced_strength) else 0.0
        blend_amount = voiced_amount * mix
        if blend_amount <= 1e-4 or abs(float(shift)) < 0.01:
            mixed = frame
        else:
            corrected = _pitch_shift_frame(frame, float(shift))
            blend = blend_amount * analysis_window
            mixed = frame * (1.0 - blend[:, np.newaxis]) + corrected * blend[:, np.newaxis]

        output[start:end] += mixed * analysis_window[:, np.newaxis]
        weights[start:end] += analysis_window

    if progress_callback:
        progress_callback("Finalizing pitch correction...")

    _check_cancel(cancel_event)
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


def process_chain(audio, sr, settings, progress_callback=None, cancel_event=None):
    processed = audio.astype(np.float32).copy()

    def report_progress(stage):
        _check_cancel(cancel_event)
        if progress_callback:
            progress_callback(stage)

    report_progress("Pitch detection...")
    if settings.get("pitch_align_enabled", True):
        processed = pitch_align(
            processed,
            sr,
            key=settings.get("key", "D"),
            scale=settings.get("scale", "major"),
            strength=settings.get("pitch_strength", 0.9),
            mix=settings.get("pitch_mix", 1.0),
            skip_long_files=settings.get("skip_long_files", False),
            progress_callback=progress_callback,
            hard_tune=settings.get("hard_tune", False),
            range_explorer=settings.get("range_explorer", False),
            range_explorer_amount=settings.get("range_explorer_amount", 0.35),
            range_explorer_seed=settings.get("range_explorer_seed"),
            cancel_event=cancel_event,
        )

    report_progress("DC removal...")
    if settings.get("dc_remove_enabled"):
        _check_cancel(cancel_event)
        processed = dc_offset_removal(processed)

    report_progress("Highpass filtering...")
    if settings.get("highpass_enabled"):
        _check_cancel(cancel_event)
        processed = highpass_filter(processed, sr, cutoff=settings.get("highpass_cutoff", 80.0))

    report_progress("Low shelf...")
    if settings.get("low_shelf_enabled"):
        _check_cancel(cancel_event)
        processed = low_shelf_boost(processed, sr, gain_db=settings.get("low_shelf_gain", 0.0))

    report_progress("Noise gate...")
    if settings.get("noise_gate_enabled"):
        _check_cancel(cancel_event)
        processed = noise_gate(
            processed,
            sr,
            threshold_db=settings.get("noise_gate_threshold", -45.0),
            release_ms=settings.get("noise_gate_release", 120.0),
        )

    report_progress("Mid boost...")
    if settings.get("mid_boost_enabled"):
        _check_cancel(cancel_event)
        processed = midrange_boost(processed, sr, gain_db=settings.get("mid_boost_gain", 3.0))

    report_progress("Presence boost...")
    if settings.get("presence_boost_enabled"):
        _check_cancel(cancel_event)
        processed = presence_boost(processed, sr, gain_db=settings.get("presence_boost_gain", 0.0))

    report_progress("De-esser...")
    if settings.get("de_esser_enabled"):
        _check_cancel(cancel_event)
        processed = de_esser(processed, sr, intensity=settings.get("de_esser_intensity", 0.25))

    report_progress("Notch filter...")
    if settings.get("notch_enabled"):
        _check_cancel(cancel_event)
        processed = notch_filter(
            processed,
            sr,
            center_hz=settings.get("notch_frequency", 60.0),
            q=settings.get("notch_q", 20.0),
        )

    report_progress("High cut...")
    if settings.get("high_cut_enabled"):
        _check_cancel(cancel_event)
        processed = high_frequency_smoothing(
            processed,
            sr,
            cutoff=settings.get("high_cut_freq", 9000.0),
            wet=settings.get("high_cut_mix", 0.35),
        )

    report_progress("High shelf...")
    if settings.get("high_shelf_enabled"):
        _check_cancel(cancel_event)
        processed = high_shelf_boost(processed, sr, gain_db=settings.get("high_shelf_gain", 0.0))

    report_progress("Compression...")
    if settings.get("compression_enabled"):
        _check_cancel(cancel_event)
        processed = compress_audio(
            processed,
            sr,
            intensity=settings.get("compression_intensity", 0.4),
            threshold_db=settings.get("compression_threshold", -18.0),
            attack_ms=settings.get("compression_attack", 12.0),
            release_ms=settings.get("compression_release", 90.0),
            makeup_db=settings.get("compression_makeup", 0.0),
        )

    report_progress("Saturation...")
    if settings.get("saturation_enabled"):
        _check_cancel(cancel_event)
        processed = soft_saturation(processed, amount=settings.get("saturation_amount", 0.0))

    report_progress("Stereo width...")
    if settings.get("stereo_width_enabled"):
        _check_cancel(cancel_event)
        processed = stereo_widen(processed, sr, amount=settings.get("stereo_width_amount", 0.3))

    report_progress("Stereo balance...")
    if settings.get("stereo_balance_enabled"):
        _check_cancel(cancel_event)
        processed = stereo_balance(processed, amount=settings.get("stereo_balance", 0.0))

    report_progress("Reverb...")
    if settings.get("reverb_enabled"):
        _check_cancel(cancel_event)
        processed = add_reverb(
            processed,
            sr,
            wet=settings.get("reverb_mix", 0.15),
            decay=settings.get("reverb_decay", 0.22),
            predelay_ms=settings.get("reverb_predelay", 20.0),
        )

    report_progress("Limiter...")
    if settings.get("limiter_enabled"):
        _check_cancel(cancel_event)
        processed = limiter(processed, ceiling=settings.get("limiter_ceiling", 0.98))

    report_progress("Finalizing...")
    _check_cancel(cancel_event)
    return normalize_audio(processed)
