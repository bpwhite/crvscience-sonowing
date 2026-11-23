# SonoWing v1.1.0 User Guide

SonoWing is a safe-Rust (`#![forbid(unsafe_code)]`) bioacoustic workstation for bat echolocation (`15–200 kHz`) and avian vocalizations (`0.3–12 kHz`), integrating STFT spectrograms, zero-phase Butterworth filtering, Shift-drag caliper measurements, harmonic stack tracking, MFCC/UMAP clustering, ultrasonic auditioning, and GUANO v1.0 / Raven Pro exports.

---

## 1. Startup Calibration Hub & Loading Audio

Launching `sonowing` opens the **Startup Calibration Hub** (pass `--skip-hub` to jump straight to the workstation):
- **9 Calibrated Field Presets**:
  - Bats: `B1` (*Myotis lucifugus* FM search-approach-buzz), `B2` (*Lasiurus cinereus* low-frequency FM), `B3` (*Corynorhinus townsendii* 2nd-harmonic whisper bat), `B4` (*Rhinolophus ferrumequinum* CF/FM-CF-FM horseshoe bat).
  - Birds: `A1` (*Catharus ustulatus* & *Setophaga coronata* sympatric NFCs), `A2` (*Zonotrichia leucophrys* dawn trill), `A3` (*Strix varia* low-frequency hoot), `A4` (*Ixoreus naevius* & *Megascops kennicottii* multi-harmonic stack).
  - Dual: `M1` (Riparian mixed `192 kHz` bat + avian nocturnal soundscape).
- **External Audio**: Click **Browse...** or drag-and-drop any 8/16/24/32-bit PCM or 32-bit IEEE float `.wav` (with optional `guan` GUANO v1.0 metadata) or Anabat `.zc` / `.zcf` file onto the window.

---

## 2. Spectrogram Navigation, STFT Presets & Visualization

### Three One-Click STFT Presets (`Left DSP Panel`)
Under **STFT Spectrogram Engine**, quickly configure FFT size, hop step, window taper, and detector band:
1. **Bat FM 512/64**: `512`-pt Blackman-Harris 4-Term, `64`-sample hop, `BatUltrasonicOnly` mode (maximizes temporal tracking of steep `2–5 ms` FM sweeps).
2. **Bat CF 2048/128**: `2048`-pt Blackman-Harris 4-Term, `128`-sample hop, `BatUltrasonicOnly` mode (maximizes frequency resolution on narrowband constant-frequency pulses).
3. **Avian NFC 1024/256**: `1024`-pt Hann window, `256`-sample hop, `BirdAudibleOnly` mode (balanced resolution for nocturnal flight calls & song).

### Max-Pooled Display, Log-kHz Axis & 256-Entry Color Maps
- **Max-Pooled Column/Row Aggregation**: On long recordings where multiple STFT frames collapse into one pixel column, SonoWing plots the maximum dBFS per column/row so brief FM pulses remain visible when zoomed out.
- **Log kHz Y-Axis Toggle**: Check **Log kHz** in the canvas toolbar (or **Logarithmic Frequency Axis** in the left dock) to view low-frequency bird harmonics and ultrasonic bat calls on a logarithmic frequency grid (`0.25 kHz` floor).
- **256-Entry Perceptually Uniform Palettes**: Select **Magma**, **Viridis**, **Inferno**, **Cividis** (colorblind-safe), or **Inverted Grayscale** (`View -> Spectrogram Colormap`), accompanied by a live **vertical dBFS colorbar legend** on the right margin.

### Zooming, Panning & Call Stepping
- **Scroll Wheel**: Cursor-anchored zoom (`Shift + Scroll` zooms frequency only; `Ctrl + Scroll` zooms time only).
- **Right / Middle-Drag**: Pan time and frequency windows.
- **Left / Right Arrow Keys**: Step backward or forward through detected call events and automatically center the view window on the selected call.
- **`0` / `Z` or Double-Click**: Reset full recording time and Nyquist frequency view.

---

## 3. Shift-Drag Spectrogram Caliper & Zero-Phase Bandpass

1. Hold **Shift** and **Primary-Drag** diagonally across any call or spectral feature in the spectrogram.
2. SonoWing draws a cyan bounding box with an amber diagonal slope vector and live readout:
   - **`dt (ms)`**: Duration between drag start and end.
   - **`f_lo - f_hi (kHz)`**: Frequency bounds.
   - **`slope (kHz/ms)`**: Signed frequency sweep slope (`(f_end - f_start) / dt_ms`).
   - **`pk (dBFS)`**: Peak spectral energy inside the selected time-frequency box.
3. Click **Apply Bandpass to Selection** in the canvas toolbar to engage a 4th-order zero-phase Butterworth bandpass (`filtfilt` with `-3 dB` pre-warped cutoffs and Gustafsson padding) locked to `[f_lo, f_hi]`, automatically re-running call extraction, harmonics, and clustering! Press `Esc` or click **Clear** to dismiss the caliper.

---

## 4. Ultrasonic Auditioning (`1`–`3`, `Space`), Exports & SQLite

- **Hotkeys `1` / `2` / `3`**: Switch active audition mode (`1`: Time Expansion `10x`, `2`: Time Expansion `20x`, `3`: Digital Heterodyne at `heterodyne_carrier_khz`), then press **`Space`** to play/pause via `cpal`.
- **Bottom Tabs**: Inspect **Detected Calls**, **MFCC/UMAP Clusters**, **Zero-Crossing Trace**, **Ecoacoustic Soundscape Indices** (`ACI`, `NDSI`, `BIO`, `ADI`, `AEI`, `H`), editable **GUANO v1.0** metadata, and export **Raven Pro `.selections.txt`**, **RFC 4180 CSV**, **Anabat `.zc`**, or **SHA-256 Signed `.sonoworkset.json`**.
- **SQLite WAL Sessions**: Press `F5` to save the current inspection, vetting, clusters, and indices to SQLite WAL (`sonowing_sessions.sqlite3`), or `F9` to browse and reload saved sessions.

---

## 5. License

Copyright (C) 2026 CRV Science Network. Licensed under **GPL-3.0-or-later** (`LICENSE`).
