from pathlib import Path

import numpy as np

import utils


def test_available_scales_matches_scale_definitions():
    assert utils.AVAILABLE_SCALES == list(utils.SCALE_INTERVALS.keys())


def test_available_presets_include_expected_names():
    expected = {
        "Natural Vocal Align",
        "Hard Tune",
        "Subtle Cleanup",
        "Experimental Range Explorer",
        "Bright Vocal Polish",
        "Dark Soft Vocal",
    }
    assert expected.issubset(set(utils.AVAILABLE_PRESETS))


def test_available_processing_modes_include_expected_names():
    assert utils.AVAILABLE_PROCESSING_MODES == [
        "Pitch + DSP",
        "Pitch only",
        "DSP only",
    ]


def test_validate_processing_mode_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="Unsupported processing mode"):
        utils.validate_processing_mode("Pitch maybe")


def test_get_preset_settings_returns_copy_and_expected_values():
    preset = utils.get_preset_settings("Experimental Range Explorer")
    assert preset["range_explorer"] is True
    assert preset["range_explorer_amount"] > 0.35
    assert preset["range_explorer_seed"] is None

    preset["range_explorer"] = False
    fresh = utils.get_preset_settings("Experimental Range Explorer")
    assert fresh["range_explorer"] is True


def test_get_preset_settings_rejects_unknown_preset():
    import pytest

    with pytest.raises(ValueError, match="Unsupported preset"):
        utils.get_preset_settings("Imaginary Preset")


def test_normalize_audio_leaves_audio_below_peak_unchanged():
    audio = np.array([[0.2, -0.2], [0.5, -0.5]], dtype=np.float32)

    normalized = utils.normalize_audio(audio, peak=0.98)

    np.testing.assert_allclose(normalized, audio)
    assert normalized.dtype == np.float32


def test_normalize_audio_scales_down_to_target_peak():
    audio = np.array([1.5, -0.75, 0.25], dtype=np.float32)

    normalized = utils.normalize_audio(audio, peak=0.5)

    assert np.max(np.abs(normalized)) == np.float32(0.5)
    np.testing.assert_allclose(normalized, np.array([0.5, -0.25, 1.0 / 12.0], dtype=np.float32))


def test_make_output_path_uses_output_dir_and_wav_suffix(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(utils, "INPUT_DIR", input_dir)
    monkeypatch.setattr(utils, "OUTPUT_DIR", output_dir)

    result = utils.make_output_path(input_dir / "take01.mp3")

    assert result == output_dir / "take01_aligned.wav"
    assert result.suffix == ".wav"
    assert output_dir.exists()


def test_make_output_path_rewrites_external_output_to_repo_output_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    external_dir = tmp_path / "elsewhere"
    monkeypatch.setattr(utils, "INPUT_DIR", input_dir)
    monkeypatch.setattr(utils, "OUTPUT_DIR", output_dir)

    result = utils.make_output_path(
        input_dir / "voice.wav",
        external_dir / "custom-name.flac",
    )

    assert result == output_dir / "custom-name.wav"


def test_make_output_path_avoids_same_path_as_input(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(utils, "INPUT_DIR", input_dir)
    monkeypatch.setattr(utils, "OUTPUT_DIR", output_dir)

    input_path = input_dir / "lead.wav"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path.touch()

    result = utils.make_output_path(input_path, input_path)

    assert result == output_dir / "lead_aligned.wav"
