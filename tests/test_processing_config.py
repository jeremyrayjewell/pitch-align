import numpy as np
import pytest

from processing import _nearest_scale_midi_values, _scale_pitch_classes


def test_scale_pitch_classes_for_d_major():
    result = _scale_pitch_classes("D", "major")
    expected = np.array([2, 4, 6, 7, 9, 11, 1], dtype=int)
    assert np.array_equal(result, expected)


def test_scale_pitch_classes_rejects_invalid_key():
    with pytest.raises(ValueError, match="Unsupported key"):
        _scale_pitch_classes("H", "major")


def test_scale_pitch_classes_rejects_invalid_scale():
    with pytest.raises(ValueError, match="Unsupported scale"):
        _scale_pitch_classes("C", "super-locrian-ish")


def test_nearest_scale_midi_values_snaps_to_scale():
    midi_values = np.array([61.2, 63.8, 70.9], dtype=np.float32)
    c_major_pcs = _scale_pitch_classes("C", "major")

    snapped = _nearest_scale_midi_values(midi_values, c_major_pcs)

    expected = np.array([62.0, 64.0, 71.0], dtype=np.float32)
    np.testing.assert_allclose(snapped, expected)
