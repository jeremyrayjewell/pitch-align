# pitch-align

`pitch-align` is a Python desktop application for scale-aware pitch correction and lightweight modular audio processing. It is designed to align vocals or instruments to a selected musical scale while preserving timing, sample rate, and as much of the original character as possible.

The app uses a Tkinter GUI and a modular DSP chain built with `librosa`, `numpy`, `scipy`, and `soundfile`.

The pitch-correction stage is experimental. It works best on clean, mostly monophonic material with a clear dominant pitch, such as a single vocal line or lead instrument.

## Features

- Load `.wav`, `.mp3`, and `.m4a` input files
- Save processed audio as `.wav` in the `output/` folder
- Detect pitch frame by frame with a lightweight FFT-based tracker
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

## Setup

Create and activate a virtual environment if you want an isolated install:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

If `pip` points to the wrong Python version, use:

```powershell
py -m pip install -r requirements.txt
```

On some systems, decoding `.mp3` or `.m4a` files may depend on the audio backends available in your Python environment. If `.m4a` loading fails, install an AAC-capable decoder such as FFmpeg.

## Running The App

From the project folder:

```powershell
python main.py
```

If needed:

```powershell
py main.py
```

The app opens a Tkinter window where you can choose an input file, select output settings, enable or disable pitch alignment, and run the processing chain.

## Basic Workflow

1. Launch the app with `python main.py`.
2. Choose an input `.wav`, `.mp3`, or `.m4a` file.
3. Choose or accept the suggested output `.wav` path.
4. Enable `Pitch Align` if you want scale correction.
5. Select the target key and scale.
6. Enable any DSP modules you want to use.
7. Click `Process Audio`.

The processed file is saved to `output/<originalname>_aligned.wav` unless you choose another output filename inside the `output/` folder.

## Development Check

A quick syntax check:

```powershell
python -m compileall main.py gui.py processing.py utils.py
```

## Pitch Alignment

Pitch correction is handled by `pitch_align(audio, sr, key="D", scale="major", strength=0.9, mix=1.0)` in `processing.py`.

The correction flow is:

1. Convert the signal to mono for pitch analysis only.
2. Detect `f0` over time using a lightweight FFT-based pitch tracker.
3. Convert detected frequency values to MIDI note numbers.
4. Map each voiced frame to the nearest valid note in the selected scale.
   If `Range explorer` is enabled, the target can instead wander to nearby in-scale notes for a less stable, more exploratory melodic result.
5. Compute a per-frame semitone correction amount.
6. Smooth the correction curve.
7. Blend corrected frames back into the original audio without changing tempo.

This is not a global pitch shift. Each voiced frame is corrected independently toward the chosen scale.

The current implementation is intentionally lightweight and experimental. It is useful for exploration and simple correction, but it is not a replacement for a dedicated commercial tuning engine.

## Available Scale Types

The app currently supports:

- `major`
- `minor`
- `aeolian major`
- `aeolian minor`
- `harmonic minor`
- `melodic minor`
- `major pentatonic`
- `minor pentatonic`
- `blues`
- `dorian`
- `dorian major`
- `dorian minor`
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
- `Hard tune mode`: makes correction more immediate and less smooth
- `Range explorer`: randomly jumps among nearby valid scale notes instead of always choosing the nearest one
- `Explorer amount`: controls how far and how often those random in-scale jumps wander

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

## Known Limitations

- Pitch correction is experimental and works best on clean monophonic or clearly dominant melodic material.
- Dense chords, noisy recordings, breathy vocals, and heavily percussive material may produce unstable or inaccurate correction.
- Pitch analysis is frame-based and lightweight, so artifacts may be audible on exposed material.
- The current pitch-shifting approach is lightweight and deterministic, but it is not a replacement for a dedicated commercial vocal tuning engine.
- Output is always written as `.wav`.
- The app keeps the original sample rate and does not time-stretch the file.
- The `Cancel` button only updates the UI state; it does not forcibly stop the background processing thread.
- Long files can take noticeably longer to process, especially when pitch alignment is enabled.

## Main Files

- `main.py`: application entry point
- `gui.py`: Tkinter interface and user interaction
- `processing.py`: pitch alignment and DSP chain
- `utils.py`: file paths, audio loading/saving, note and scale definitions

## License

This repository is available under the MIT License. See [LICENSE](/d:/git/pitch-align/LICENSE).
