# pitch-align

`pitch-align` is a Python desktop application for scale-aware pitch correction and lightweight modular audio processing. It is designed to align vocals or instruments to a selected musical scale while preserving timing, sample rate, and as much of the original character as possible.

The app uses a Tkinter GUI and a modular DSP chain built with `librosa`, `numpy`, `scipy`, and `soundfile`.

## Features

- Load `.wav` and `.mp3` input files
- Save processed audio as `.wav` in the `output/` folder
- Detect pitch frame by frame with `librosa.pyin`
- Map each detected pitch to the nearest note in a selected key and scale
- Preserve original duration and timing
- Keep stereo intact and only widen when requested
- Apply optional DSP stages in a fixed, deterministic order

## Project Structure

```text
pitch-align/
|
|-- main.py
|-- gui.py
|-- processing.py
|-- utils.py
|-- input/
`-- output/
```

## Requirements

- Python 3.10 or newer recommended
- Desktop environment with Tkinter available

Install dependencies:

```powershell
pip install librosa numpy scipy soundfile
```

If `pip` points to the wrong Python version, use:

```powershell
py -m pip install librosa numpy scipy soundfile
```

## Running The App

From the project folder:

```powershell
python main.py
```

If needed:

```powershell
py main.py
```

## Basic Workflow

1. Launch the app with `python main.py`.
2. Choose an input `.wav` or `.mp3` file.
3. Choose or accept the suggested output `.wav` path.
4. Enable `Pitch Align` if you want scale correction.
5. Select the target key and scale.
6. Enable any DSP modules you want to use.
7. Click `Process Audio`.

The processed file is saved to `output/<originalname>_aligned.wav` unless you choose another output filename inside the `output/` folder.

## Pitch Alignment

Pitch correction is handled by `pitch_align(audio, sr, key="D", scale="major", strength=0.9, mix=1.0)` in `processing.py`.

The correction flow is:

1. Convert the signal to mono for pitch analysis only.
2. Detect `f0` over time using `librosa.pyin`.
3. Convert detected frequency values to MIDI note numbers.
4. Map each voiced frame to the nearest valid note in the selected scale.
5. Compute a per-frame semitone correction amount.
6. Smooth the correction curve.
7. Blend corrected frames back into the original audio without changing tempo.

This is not a global pitch shift. Each voiced frame is corrected independently toward the chosen scale.

## Available Scale Types

The app currently supports:

- `major`
- `minor`
- `harmonic minor`
- `melodic minor`
- `major pentatonic`
- `minor pentatonic`
- `blues`
- `dorian`
- `phrygian`
- `lydian`
- `mixolydian`
- `locrian`
- `whole tone`
- `chromatic`

## DSP Chain

Processing runs in a fixed order in `process_chain(...)` inside `processing.py`.

Current modules include:

- `DC remove`: removes DC offset from the signal
- `Highpass`: removes low-end rumble
- `Low shelf`: boosts or cuts lows gently
- `Noise gate`: reduces low-level background noise between phrases
- `Mid boost`: emphasizes the `200 Hz` to `900 Hz` area
- `Presence boost`: emphasizes upper mids for clarity
- `De-esser`: reduces harsh sibilance
- `Notch`: removes a narrow problem frequency
- `High cut`: smooths top-end content
- `High shelf`: boosts or cuts high frequencies
- `Compression`: simple envelope-based dynamic control
- `Saturation`: adds soft non-linear warmth
- `Stereo width`: Haas-style widening
- `Stereo balance`: adjusts left/right balance
- `Reverb`: light reverb with decay and predelay
- `Limiter`: final peak protection

## Pitch Controls

- `Enable pitch align`: turns scale correction on or off
- `Key`: root note for the target scale
- `Scale`: scale or mode used for note mapping
- `Pitch strength`: how strongly frames are pulled toward target notes
- `Pitch mix`: blend between dry and corrected signal

## DSP Controls Guide

- `Highpass cutoff`: higher values remove more low-end
- `Low shelf gain`: positive values add bass, negative values reduce it
- `Noise gate threshold`: higher threshold means more aggressive gating
- `Noise gate release`: controls how quickly the gate closes
- `Mid boost gain`: increases vocal or instrument body
- `Presence boost gain`: helps a source cut through the mix
- `De-esser intensity`: stronger reduction of harsh consonants
- `Notch frequency`: center frequency to remove
- `Notch Q`: higher values make the notch narrower
- `High cut frequency`: lower values darken the sound more
- `High cut mix`: blend between dry and smoothed high end
- `Compression intensity`: stronger overall compression
- `Compression threshold`: lower values compress more of the signal
- `Compression attack`: lower values react faster to peaks
- `Compression release`: higher values recover more slowly
- `Compression makeup`: restores loudness after compression
- `Saturation amount`: adds more harmonic drive
- `Stereo width amount`: increases perceived width
- `Stereo balance`: shifts energy left or right
- `Reverb wet/dry`: amount of reverb in the output
- `Reverb decay`: length of the reverb tail
- `Reverb predelay`: delay before the reverb starts
- `Limiter ceiling`: maximum output peak before final normalization

## Notes And Limitations

- Pitch detection works best on monophonic or clearly dominant melodic material.
- Dense chords, noisy recordings, and heavily percussive material may produce less stable correction.
- The current pitch-shifting approach is lightweight and deterministic, but it is not a replacement for a dedicated commercial vocal tuning engine.
- Output is always written as `.wav`.
- The app keeps the original sample rate and does not time-stretch the file.

## Main Files

- `main.py`: application entry point
- `gui.py`: Tkinter interface and user interaction
- `processing.py`: pitch alignment and DSP chain
- `utils.py`: file paths, audio loading/saving, note and scale definitions

## Development Check

A quick syntax check:

```powershell
python -m py_compile main.py gui.py processing.py utils.py
```

## License

No license file is currently included in this repository.
