# SonoWing v1.1.0 Scientific Methods & Numerical Formulation

This document specifies the spectral analysis, digital filtering, Hilbert envelope estimation, ecoacoustic indices, and resampling formulations implemented in SonoWing (`GPL-3.0-or-later`).

---

## 1. Single-Sided Windowed STFT Amplitude Calibration (`src/dsp.rs`)

Given window `w[0..N]` of length `N = fft_size` with coherent sum `W_1 = sum_{n=0}^{N-1} w[n]`, frame `t` centered at `offset = t * hop_size` computes the radix-2 DFT `X_t[b]`.

Because real-valued signals exhibit Hermitian symmetry `X[N - b] = conj(X[b])`, interior bins `1 <= b < N/2` carry half their amplitude in the negative-frequency conjugate bin, whereas real-valued DC (`b = 0`) and Nyquist (`b = N/2`) bins have no negative-frequency partner:

```text
M_t[b] = |X_t[b]| * (1 / W_1)    for b = 0 or b = N / 2
M_t[b] = |X_t[b]| * (2 / W_1)    for 1 <= b < N / 2
dBFS_t[b] = max(20 * log10(M_t[b]), min_dbfs)
```

For a constant DC signal `x[n] = 0.5`, `X[0] = 0.5 * W_1`, yielding exact magnitude `M_t[0] = 0.5` (`-6.0206 dBFS`) without `+6.02 dB` endpoint bias.

---

## 2. Zero-Phase Butterworth `filtfilt` & Gustafsson (1996) Padding (`src/dsp.rs`)

### 2.1 `-3 dB` Pre-Warped Bilinear Cutoff for Forward-Backward Cascade
A single pass of a 2nd-order Butterworth filter with analog/bilinear frequency `w_0` has magnitude squared:

```text
|H_1(w)|^2 = 1 / (1 + (tan(w/2) / tan(w_0/2))^4)       [Low-Pass]
|H_1(w)|^2 = 1 / (1 + (tan(w_0/2) / tan(w/2))^4)       [High-Pass]
```

Applying forward and reverse passes (`filtfilt`) squares the magnitude (`|H_zp(w)| = |H_1(w)|^2`). To preserve the exact `-3.01 dB` (`1 / sqrt(2)`) cutoff at desired frequency `f_c` (`w_c = 2 * pi * f_c / f_s`), SonoWing pre-warps using `c_warp = (sqrt(2) - 1)^(1/4) ≈ 0.80220366`:

```text
tan(w_0 / 2) = tan(w_c / 2) / c_warp    [Low-Pass]
tan(w_0 / 2) = tan(w_c / 2) * c_warp    [High-Pass]
```

### 2.2 Odd-Symmetric Reflection Padding & Steady-State Initial Condition
Before filtering `x[0..N]`, `apply_zero_phase` constructs `pad = min(12, N - 1)` odd-symmetric reflected boundary samples around `x[0]` and `x[N - 1]` (Gustafsson 1996):

```text
x_left[k]  = 2 * x[0]     - x[pad - k]      for k = 0..pad-1
x_right[k] = 2 * x[N - 1] - x[N - 2 - k]    for k = 0..pad-1
```

Direct Form II Transposed filter registers `(z1, z2)` are initialized to exact DC steady state for initial sample `x_0`:

```text
y_ss = ((b0 + b1 + b2) / (1 + a1 + a2)) * x_0
z1   = y_ss - b0 * x_0
z2   = b2 * x_0 - a2 * y_ss
```

eliminating start-up and turn-around edge transients across both low-pass and high-pass filtering.

---

## 3. Overlap-Discard Analytic Hilbert Envelope (`src/dsp.rs`)

For long bioacoustic recordings (`N > 16,384`), `hilbert_envelope` computes `|x[n] + j * H{x[n]}|` via 50% overlap-discard blocks of length `M = 65,536` (`keep = 32,768` interior samples, `apron = 16,384` samples on both sides). Within each block, frequency bins are weighted by the analytic multiplier (`1` at `k = 0, M/2`; `2` at `1 <= k < M/2`; `0` at `k > M/2`) before inverse FFT. Because interior output samples lie at least `16,384` samples from circular FFT boundaries (`1 / (pi * 16384) < 2e-5`), chunk transitions at `n = 32,768` remain continuous and click-free.

---

## 4. Ecoacoustic Soundscape Indices (`src/soundscape_indices.rs`) & Anti-Aliasing (`src/ultrasonic_audio.rs`)

- **NDSI (Kasten et al. 2012)**:
  ```text
  Anthrophony (A) = sum of mean bin PSD in [1000, 2000) Hz
  Biophony    (B) = sum of mean bin PSD in [2000, 11000] Hz (+ >=16 kHz when Nyquist > 25 kHz)
  NDSI = (B - A) / (B + A)
  ```
- **ADI & AEI (Villanueva-Rivera et al. 2011)**:
  Occupancy `p_k` across `10` equal frequency bands is computed over strictly **half-open** bin ranges `[kb0(k), kb1_excl(k))` (`kb1_excl(k) = kb0(k + 1)`) using threshold `-50.0 dBFS`, preventing shared boundary bins from inflating diversity across adjacent bands:
  ```text
  ADI = -sum_{k: p_k > 0} (p_k / P) * ln(p_k / P),    P = sum_k p_k
  AEI = Gini(p_0, ..., p_9) in [0, 1]
  ```
- **Anti-Aliased Audible Resampling**:
  Heterodyne, Frequency Division, and linear downsampling enforce zero-phase Butterworth anti-aliasing cutoffs `f_lp <= min(desired_hz, 0.45 * out_rate, 0.45 * in_rate)` prior to decimation.

---

## 5. License

Copyright (C) 2026 CRV Science Network. Released under **GPL-3.0-or-later** (`LICENSE`).
