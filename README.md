# SonoWing (`sonowing` v1.1.0)

![SonoWing — Ultrasonic Bat Echolocation & Avian Bioacoustic Spectrogram, Harmonic Tracker & MFCC/UMAP Workstation](validation/headless_screenshot.png)

Native safe-Rust (`#![forbid(unsafe_code)]`) desktop GUI workstation (`eframe` / `egui` `0.31`) and headless parallel CLI (`clap` `4.5` + `rayon` `1.10`) for **Chiropteran (bat) ultrasonic echolocation** and **Avian (bird) nocturnal flight call (`NFC`) & song** waveform, spectrogram, unsupervised `MFCC`/`UMAP` clustering, harmonic stack tracking, live microphone capture, and soundscape analysis.

- **User Guide**: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- **Scientific Methods & Equations**: [`docs/SCIENTIFIC_METHODS.md`](docs/SCIENTIFIC_METHODS.md)
- **License**: GNU General Public License v3.0 or later ([`LICENSE`](LICENSE))

## Scientific & Architectural Highlights

- **Exact Single-Sided STFT & Max-Pooled 256-LUT Spectrograms (`src/dsp.rs`, `src/gui.rs`)**:
  - Applies unitary `1 / win_sum` normalization to real-valued DC (`b = 0`) and Nyquist (`b = N/2`) DFT bins (eliminating the `+6.02 dB` inflation present in naive `2 / win_sum` scaling) while using `2 / win_sum` for interior complex-conjugate pairs.
  - Pre-warps 2nd-order Butterworth filters by `(sqrt(2) - 1)^(1/4)` for exact `-3.01 dB` zero-phase `filtfilt` cutoffs and pads edges with Gustafsson (1996) odd-symmetric reflection + steady-state initial filter states.
  - Uses overlap-discard (`65,536`-sample FFT with `16,384`-sample guard aprons) in `hilbert_envelope` for click-free envelopes across chunk boundaries.
  - Max-pools `(frame, bin)` energy per screen pixel column/row across 256-entry perceptually uniform LUTs (**Magma**, **Viridis**, **Inferno**, **Cividis**, **Inverted Grayscale**), preventing brief FM pulses from dropping out when zoomed out.
- **Standardized Ecoacoustic Indices & Anti-Aliased Auditioning (`src/soundscape_indices.rs`, `src/ultrasonic_audio.rs`)**:
  - Aligns `NDSI` anthrophony exactly to Kasten et al. (2012) `1,000–2,000 Hz`, sets `ADI`/`AEI` occupancy threshold to `-50.0 dBFS`, and uses half-open frequency-bin intervals `[kb0, kb1_excl)` so shared boundary bins are never double-counted.
  - Constrains Heterodyne, Frequency Division, and resampled decimation low-pass cutoffs safely below `0.45 * out_rate` to prevent ultrasonic foldover aliasing.
- **MFCC/UMAP Clustering, Subharmonic Harmonic Tracker & Live UAC2 Mic (`src/mfcc_cluster.rs`, `src/harmonic_tracker.rs`, `src/live_mic.rs`)**:
  - `24`-filter Mel/Ultrasonic Log-Mel + `DCT-II` (`13` static + `5` delta-MFCC + `4` contour features), fuzzy simplicial UMAP layout, `k`-Means++ with Silhouette `K*` selection, `f0..5f0` harmonic overtone tracking, and `192–384 kHz` live mic capture.

## Desktop Controls & Keyboard Shortcuts

All single-key shortcuts ignore input while typing in text fields (`!ctx.wants_keyboard_input()`).

| Input | Action |
|---|---|
| `Space` | Play / Pause audible ultrasonic auditioning |
| `Left` / `Right` | Step backward / forward through detected calls & auto-center view window |
| `0` or `Z` | Reset time & frequency spectrogram/waveform zoom |
| `1` / `2` / `3` | Switch audition mode (`1`: TE `10x`, `2`: TE `20x`, `3`: Digital Heterodyne) |
| `Shift + Primary-Drag` | Draw spectrogram caliper (`Δt ms`, `f_lo–f_hi kHz`, slope `kHz/ms`, peak dBFS) |
| `Primary-Click / Drag` | Move cursor playhead & select overlapping call (outside Shift-caliper) |
| `Secondary / Middle-Drag` | Pan spectrogram time and frequency viewport |
| `Scroll Wheel` | Cursor-anchored zoom (Shift: frequency only, Ctrl: time only) |
| `Double-Click` | Reset full time & frequency zoom |
| `F1` | Toggle 20-species acoustic reference key & shortcut modal |
| `Ctrl+K` | Open fuzzy command palette (`Up`/`Down`/`Enter`/`Esc`) for species presets, STFT/FFT modes, filters & exports |
| `F5` / `F9` | Quick save / open saved SQLite WAL sessions |
| `Esc` | Close active modal -> Clear spectrogram caliper |
| `Drag & Drop` | Drop `.wav`, `.zc`, `.zcf`, or `.sonoworkset.json` onto window |

## Quick Start & CLI Subcommands

```bash
# Run full unit & integration test suite (use --no-default-features on headless Linux without ALSA)
cargo test --all-targets

# Launch native desktop GUI workstation
cargo run --release

# Inspect a WAV or ZC recording headlessly with Raven Pro selection table & CSV summary
cargo run --release -- inspect samples/MYLU_Willamette_Demo_250kHz.wav \
  --raven-out mylu.Table.1.selections.txt \
  --csv-out mylu_summary.csv

# Parallel batch scan across a directory of field recordings
cargo run --release -- batch recordings_dir/ --out-dir sonowing_batch_out --snr-db 12.0

# Capture 4 seconds of live microphone audio (or --simulated 192 kHz nocturnal stream)
cargo run --release -- record-mic --seconds 4.0 --simulated --output live_capture.wav

# Transcode ultrasonic WAV to audible 48 kHz WAV (te10 | te20 | heterodyne | fdiv)
cargo run --release -- transcode samples/MYLU_Willamette_Demo_250kHz.wav \
  --mode heterodyne --carrier-khz 41.0 --output mylu_het.wav

# Generate synthetic calibrated preset WAV (B1..B4, A1..A4, M1)
cargo run --release -- synth-demo --preset B1 --output demo_b1.wav

# Export batch call-classifier SNR sweep & confusion-matrix diagnostic (JSON + CSV)
cargo run --release -- classifier-diag --snr-list 8.0,12.0,16.0,20.0 --out-dir exports
```

| CLI Flag / Subcommand | Default | Purpose |
|---|---|---|
| `--open <PATH>` | — | Open `.wav` or `.zc` file directly in GUI |
| `--skip-hub` | `false` | Bypass Startup Calibration Hub directly into workstation |
| `--db <PATH>` | `sonowing_sessions.sqlite3` | SQLite WAL session repository path |
| `--headless-screenshot <PATH>` | — | Render `1600x1000` offscreen RGBA PNG (`--screenshot-view 1\|2\|3`) |
| `inspect <FILE>` | `--snr-db 12.0` | Headless STFT, morphometrics, harmonics, MFCC/UMAP, Raven/CSV/SQLite |
| `batch <DIR>` | `--out-dir sonowing_batch_out` | Rayon parallel batch directory scan |
| `record-mic` | `--seconds 3.0 --ultrasonic` | Capture live hardware UAC2 or `--simulated` audio to GUANO WAV |
| `transcode <FILE>` | `--mode te10` | Export audible `48 kHz` WAV (`te10`, `te20`, `heterodyne`, `fdiv`) |
| `synth-demo` | `--preset B1` | Synthesize calibrated bat/bird GUANO WAV (`B1–B4`, `A1–A4`, `M1`) |
| `classifier-diag` | `--snr-list 8.0,12.0,16.0,20.0` | Export 9-preset SNR & confusion-matrix diagnostic summary (`JSON` + `CSV`) |

## Validation Suite & Empirical Metrics

Run `$env:CARGO_BUILD_JOBS='4'; cargo test --all-targets` (`33` unit tests in `src/lib.rs`, `2` in `src/main.rs`, `10` integration tests in `tests/bioacoustic_suite.rs`):

| Test | Verified Metric / Outcome |
|---|---|
| `dsp::tests::stft_dc_and_nyquist_bins_not_inflated_by_6db` | Pure `x = 0.5` DC signal yields `mag == 0.5` (`-6.02 ± 0.05 dBFS`) |
| `dsp::tests::hilbert_envelope_seamless_across_32768_chunk_boundary` | `40,000`-sample `10 kHz` unit sinusoid stays within `[0.98, 1.02]` across `n = 32,768` |
| `dsp::tests::zero_phase_butterworth_prewarps_minus_3db_and_avoids_edge_transient` | Zero-phase LP hits `-3.01 ± 0.20 dB` at `fc = 10 kHz` with flat DC step response |
| `soundscape_indices::tests::adi_aei_half_open_bins_do_not_double_count_shared_boundary_bins` | Boundary bin `1.2 kHz` belongs solely to band 1 (`ADI == 0.0`, `AEI == 0.9`) |
| `gui::tests::hotkeys_caliper_log_axis_and_256_lut_palettes` | `Left`/`Right` auto-center, `Z` zoom reset, `1-3` audition modes, Shift-caliper slope, Log-kHz, 256-LUT |
| `tests/bioacoustic_suite.rs` | Parseval FFT (`1.0`), `MYLU`/`RHFE`/`COTO`/`A1`/`A4` detection, MFCC/UMAP, Raven TSV, SQLite WAL roundtrip |

## Citation & CRV Science Apps Hub

- **CRV Science Apps Hub**: <https://crvscience.com/apps>
- **Companion Preprint (`APP-PREPRINT-17`)**: [SonoWing: Analyzing Bat Echolocation and Bird Calls with Spectrograms and Harmonic Tracking](https://crvscience.com/apps/preprints/sonowing-bioacoustic-echolocation-analyzer)
- **Zenodo DOI**: [10.5281/zenodo.24244091](https://doi.org/10.5281/zenodo.24244091)
- **Machine-Readable Metadata**: [`CITATION.cff`](CITATION.cff)
