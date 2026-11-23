// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Zero-dependency safe-Rust bioacoustic Digital Signal Processing (`DSP`) engine:
//! - Radix-2 Cooley-Tukey FFT & IFFT
//! - Selectable spectral windows (`Hann`, `Hamming`, `BlackmanHarris4`, `KaiserBessel`)
//! - Parallel (`rayon`) STFT spectrogram with median-per-bin background noise subtraction
//! - Zero-phase (`filtfilt`) 4th-order Butterworth Biquad filterbank (high-pass, low-pass, band-pass)
//! - Teager-Kaiser Energy Operator (`TKEO`, Kaiser 1990) & Hilbert analytic envelope
//! - Virtual Anabat Zero-Crossing (`ZCA`) trajectory extractor

use crate::wav_guano::ZeroCrossDot;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowFunction {
    Hann,
    Hamming,
    BlackmanHarris4,
    KaiserBessel,
}

impl WindowFunction {
    pub const ALL: [Self; 4] = [
        Self::Hann,
        Self::Hamming,
        Self::BlackmanHarris4,
        Self::KaiserBessel,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Hann => "Hann (Balanced)",
            Self::Hamming => "Hamming (-43 dB sidelobe)",
            Self::BlackmanHarris4 => "Blackman-Harris 4-Term (-92 dB ultrasonic)",
            Self::KaiserBessel => "Kaiser-Bessel (beta = 7.0)",
        }
    }

    pub fn generate(self, n: usize) -> Vec<f64> {
        if n <= 1 {
            return vec![1.0; n];
        }
        let denom = (n - 1) as f64;
        (0..n)
            .map(|i| {
                let x = i as f64 / denom;
                match self {
                    Self::Hann => 0.5 * (1.0 - (TAU * x).cos()),
                    Self::Hamming => 0.54 - 0.46 * (TAU * x).cos(),
                    Self::BlackmanHarris4 => {
                        0.35875 - 0.48829 * (TAU * x).cos() + 0.14128 * (2.0 * TAU * x).cos()
                            - 0.01168 * (3.0 * TAU * x).cos()
                    }
                    Self::KaiserBessel => {
                        let beta = 7.0;
                        let u = 2.0 * x - 1.0;
                        let arg = beta * (1.0 - u * u).max(0.0).sqrt();
                        bessel_i0(arg) / bessel_i0(beta)
                    }
                }
            })
            .collect()
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x_sq = 0.25 * x * x;
    for k in 1..28 {
        term *= half_x_sq / ((k * k) as f64);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// In-place Radix-2 Cooley-Tukey FFT (`inverse = false`) or IFFT (`inverse = true`).
/// `buf.len()` MUST be a power of two.
pub fn fft_radix2_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FFT size must be a power of two");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = sign * TAU / (len as f64);
        let w_step = Complex64::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half].mul(w);
                buf[start + k] = Complex64::new(u.re + v.re, u.im + v.im);
                buf[start + k + half] = Complex64::new(u.re - v.re, u.im - v.im);
                w = w.mul(w_step);
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for c in buf.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StftConfig {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowFunction,
    pub subtract_median_noise_floor: bool,
    pub min_dbfs: f32,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 256,
            window: WindowFunction::BlackmanHarris4,
            subtract_median_noise_floor: false,
            min_dbfs: -96.0,
        }
    }
}

/// Complete Short-Time Fourier Transform spectrogram and frame-by-frame ridge statistics.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub hop_size: usize,
    pub num_frames: usize,
    pub num_bins: usize,
    /// Row-major `[frame_idx * num_bins + bin_idx]` power values in dBFS.
    pub db_grid: Vec<f32>,
    /// Linear magnitude `[frame_idx * num_bins + bin_idx]` for ecoacoustic indices (`ACI`, `NDSI`, `H`).
    pub mag_grid: Vec<f32>,
    /// Dominant peak frequency per STFT frame (parabolic interpolated, in Hz).
    pub frame_peak_freq_hz: Vec<f64>,
    /// Peak bin level in dBFS per STFT frame.
    pub frame_peak_dbfs: Vec<f32>,
    /// Spectral centroid per STFT frame in Hz.
    pub frame_centroid_hz: Vec<f64>,
    /// Wiener spectral flatness (`0.0 = pure tone`, `1.0 = white noise`) per frame.
    pub frame_wiener_entropy: Vec<f64>,
    /// Estimated background noise floor in dBFS per frequency bin.
    pub bin_noise_floor_dbfs: Vec<f32>,
}

impl Spectrogram {
    #[inline]
    pub fn frame_time_s(&self, frame_idx: usize) -> f64 {
        (frame_idx * self.hop_size + self.fft_size / 2) as f64 / self.sample_rate_hz as f64
    }

    #[inline]
    pub fn bin_freq_hz(&self, bin_idx: usize) -> f64 {
        bin_idx as f64 * self.sample_rate_hz as f64 / self.fft_size as f64
    }

    #[inline]
    pub fn freq_to_bin(&self, freq_hz: f64) -> usize {
        let raw = (freq_hz * self.fft_size as f64 / self.sample_rate_hz as f64).round() as isize;
        raw.clamp(0, (self.num_bins.saturating_sub(1)) as isize) as usize
    }

    pub fn db_at(&self, frame_idx: usize, bin_idx: usize) -> f32 {
        if frame_idx >= self.num_frames || bin_idx >= self.num_bins {
            return -120.0;
        }
        self.db_grid[frame_idx * self.num_bins + bin_idx]
    }

    pub fn slice_spectrum_db(&self, frame_idx: usize) -> &[f32] {
        let f = frame_idx.min(self.num_frames.saturating_sub(1));
        let start = f * self.num_bins;
        &self.db_grid[start..start + self.num_bins]
    }
}

/// Compute a parallel STFT spectrogram using `rayon` across sliding windows.
pub fn compute_stft(samples: &[f32], sample_rate_hz: u32, cfg: StftConfig) -> Spectrogram {
    let fft_size = cfg.fft_size.next_power_of_two().clamp(128, 8192);
    let hop_size = cfg.hop_size.clamp(16, fft_size);
    let num_bins = fft_size / 2 + 1;
    let win = cfg.window.generate(fft_size);
    let win_sum: f64 = win.iter().sum::<f64>().max(1e-9);
    let scale_ac = 2.0 / win_sum;
    let scale_dc_nyq = 1.0 / win_sum;
    let nyq_bin = fft_size / 2;

    let num_frames = if samples.len() <= fft_size {
        1
    } else {
        (samples.len() - fft_size) / hop_size + 1
    };

    let per_frame: Vec<(Vec<f32>, Vec<f32>, f64, f32, f64, f64)> = (0..num_frames)
        .into_par_iter()
        .map(|f_idx| {
            let offset = f_idx * hop_size;
            let mut buf = vec![Complex64::default(); fft_size];
            for i in 0..fft_size {
                let s = samples.get(offset + i).copied().unwrap_or(0.0) as f64;
                buf[i] = Complex64::new(s * win[i], 0.0);
            }
            fft_radix2_in_place(&mut buf, false);

            let mut db_row = vec![cfg.min_dbfs; num_bins];
            let mut mag_row = vec![0.0_f32; num_bins];
            let mut max_bin = 1usize;
            let mut max_mag = 0.0_f64;
            let mut weighted_freq = 0.0_f64;
            let mut sum_power = 0.0_f64;
            let mut log_power_sum = 0.0_f64;
            let active_bins = (num_bins - 1).max(1);

            for b in 0..num_bins {
                let bin_scale = if b == 0 || b == nyq_bin {
                    scale_dc_nyq
                } else {
                    scale_ac
                };
                let mag = buf[b].norm() * bin_scale;
                mag_row[b] = mag as f32;
                let db = if mag <= 1e-10 {
                    cfg.min_dbfs
                } else {
                    (20.0 * mag.log10() as f32).max(cfg.min_dbfs)
                };
                db_row[b] = db;

                if b > 0 {
                    let p = mag * mag + 1e-20;
                    let freq = b as f64 * sample_rate_hz as f64 / fft_size as f64;
                    weighted_freq += freq * p;
                    sum_power += p;
                    log_power_sum += p.ln();
                    if mag > max_mag {
                        max_mag = mag;
                        max_bin = b;
                    }
                }
            }

            // Parabolic sub-bin peak interpolation in dB domain
            let interp_bin = if max_bin > 1 && max_bin + 1 < num_bins {
                let alpha = db_row[max_bin - 1] as f64;
                let beta = db_row[max_bin] as f64;
                let gamma = db_row[max_bin + 1] as f64;
                let denom = alpha - 2.0 * beta + gamma;
                if denom.abs() > 1e-9 {
                    max_bin as f64 + 0.5 * (alpha - gamma) / denom
                } else {
                    max_bin as f64
                }
            } else {
                max_bin as f64
            };
            let peak_freq_hz = interp_bin * sample_rate_hz as f64 / fft_size as f64;
            let peak_dbfs = db_row[max_bin];
            let centroid_hz = if sum_power > 1e-18 {
                weighted_freq / sum_power
            } else {
                0.0
            };
            let geom_mean = (log_power_sum / active_bins as f64).exp();
            let arith_mean = (sum_power / active_bins as f64).max(1e-20);
            let wiener = (geom_mean / arith_mean).clamp(0.0, 1.0);

            (db_row, mag_row, peak_freq_hz, peak_dbfs, centroid_hz, wiener)
        })
        .collect();

    let mut db_grid = Vec::with_capacity(num_frames * num_bins);
    let mut mag_grid = Vec::with_capacity(num_frames * num_bins);
    let mut frame_peak_freq_hz = Vec::with_capacity(num_frames);
    let mut frame_peak_dbfs = Vec::with_capacity(num_frames);
    let mut frame_centroid_hz = Vec::with_capacity(num_frames);
    let mut frame_wiener_entropy = Vec::with_capacity(num_frames);

    for (db_row, mag_row, pf, pdb, cent, w) in per_frame {
        db_grid.extend_from_slice(&db_row);
        mag_grid.extend_from_slice(&mag_row);
        frame_peak_freq_hz.push(pf);
        frame_peak_dbfs.push(pdb);
        frame_centroid_hz.push(cent);
        frame_wiener_entropy.push(w);
    }

    // Estimate per-bin median background noise floor
    let mut bin_noise_floor_dbfs = vec![cfg.min_dbfs; num_bins];
    let mut scratch = Vec::with_capacity(num_frames);
    for b in 0..num_bins {
        scratch.clear();
        for f in 0..num_frames {
            scratch.push(db_grid[f * num_bins + b]);
        }
        scratch.sort_by(|a, c| a.total_cmp(c));
        let med = scratch[scratch.len() / 2];
        bin_noise_floor_dbfs[b] = med;
    }

    if cfg.subtract_median_noise_floor {
        for f in 0..num_frames {
            for b in 0..num_bins {
                let idx = f * num_bins + b;
                // Re-anchor relative to -84 dBFS clean display baseline
                let snr_db = (db_grid[idx] - bin_noise_floor_dbfs[b]).max(0.0);
                db_grid[idx] = (-84.0 + snr_db).clamp(cfg.min_dbfs, 0.0);
            }
        }
    }

    Spectrogram {
        sample_rate_hz,
        fft_size,
        hop_size,
        num_frames,
        num_bins,
        db_grid,
        mag_grid,
        frame_peak_freq_hz,
        frame_peak_dbfs,
        frame_centroid_hz,
        frame_wiener_entropy,
        bin_noise_floor_dbfs,
    }
}

/// Second-order Biquad section (`Direct Form II Transposed`) used for zero-phase `filtfilt`.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Robert Bristow-Johnson Audio EQ Cookbook 2nd-order Butterworth Low-Pass (`Q = 1/sqrt(2)`),
    /// pre-warped by `1 / (sqrt(2) - 1)^(1/4)` so forward-backward `filtfilt` hits `-3.01 dB` at `cutoff_hz`.
    pub fn butterworth_lowpass(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        let fc = cutoff_hz.clamp(5.0, 0.495 * sample_rate_hz);
        let c_warp = (2.0_f64.sqrt() - 1.0).powf(0.25);
        let tan_half = (PI * fc / sample_rate_hz).tan() / c_warp;
        let w0 = (2.0 * tan_half.atan()).min(TAU * 0.495);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / 2.0_f64.sqrt();
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) * 0.5) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// 2nd-order Butterworth High-Pass (`Q = 1/sqrt(2)`), pre-warped by `(sqrt(2) - 1)^(1/4)`
    /// so forward-backward `filtfilt` hits `-3.01 dB` at `cutoff_hz`.
    pub fn butterworth_highpass(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        let fc = cutoff_hz.clamp(5.0, 0.495 * sample_rate_hz);
        let c_warp = (2.0_f64.sqrt() - 1.0).powf(0.25);
        let tan_half = (PI * fc / sample_rate_hz).tan() * c_warp;
        let w0 = (2.0 * tan_half.atan()).max(TAU * (5.0 / sample_rate_hz));
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / 2.0_f64.sqrt();
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) * 0.5) / a0,
            b1: -(1.0 + cos_w0) / a0,
            b2: ((1.0 + cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    #[inline]
    fn steady_state_zi(&self, x0: f64) -> (f64, f64) {
        let denom = 1.0 + self.a1 + self.a2;
        let y_ss = if denom.abs() > 1e-12 {
            ((self.b0 + self.b1 + self.b2) / denom) * x0
        } else {
            0.0
        };
        let z1 = y_ss - self.b0 * x0;
        let z2 = self.b2 * x0 - self.a2 * y_ss;
        (z1, z2)
    }

    fn apply_forward_f64_zi(&self, input: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(input.len());
        let Some(&x0) = input.first() else {
            return out;
        };
        let (mut z1, mut z2) = self.steady_state_zi(x0);
        for &x in input {
            let y = self.b0 * x + z1;
            z1 = self.b1 * x - self.a1 * y + z2;
            z2 = self.b2 * x - self.a2 * y;
            out.push(y);
        }
        out
    }

    pub fn apply_forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;
        for &x_f32 in input {
            let x = x_f32 as f64;
            let y = self.b0 * x + z1;
            z1 = self.b1 * x - self.a1 * y + z2;
            z2 = self.b2 * x - self.a2 * y;
            out.push(y as f32);
        }
        out
    }

    /// Forward-backward (`filtfilt`) zero-phase 4th-order response using Gustafsson (1996)
    /// odd-symmetric reflection padding and steady-state initial filter states.
    pub fn apply_zero_phase(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            let denom = 1.0 + self.a1 + self.a2;
            let dc_gain = if denom.abs() > 1e-12 {
                (self.b0 + self.b1 + self.b2) / denom
            } else {
                0.0
            };
            return vec![(input[0] as f64 * dc_gain * dc_gain) as f32];
        }
        let pad = 12usize.min(n - 1);
        let mut padded = Vec::with_capacity(n + 2 * pad);
        let x_start = input[0] as f64;
        let x_end = input[n - 1] as f64;
        for i in (1..=pad).rev() {
            padded.push(2.0 * x_start - (input[i] as f64));
        }
        for &v in input {
            padded.push(v as f64);
        }
        for i in 1..=pad {
            padded.push(2.0 * x_end - (input[n - 1 - i] as f64));
        }
        let mut fwd = self.apply_forward_f64_zi(&padded);
        fwd.reverse();
        let mut bwd = self.apply_forward_f64_zi(&fwd);
        bwd.reverse();
        bwd[pad..pad + n].iter().map(|&v| v as f32).collect()
    }
}

/// Band-pass filter using cascaded zero-phase Butterworth high-pass + low-pass (`filtfilt`).
pub fn bandpass_zero_phase(
    samples: &[f32],
    low_cut_hz: f64,
    high_cut_hz: f64,
    sample_rate_hz: u32,
) -> Vec<f32> {
    let sr = sample_rate_hz as f64;
    let nyq = 0.492 * sr;
    let mut stage = if low_cut_hz > 10.0 && low_cut_hz < nyq {
        BiquadCoeffs::butterworth_highpass(low_cut_hz, sr).apply_zero_phase(samples)
    } else {
        samples.to_vec()
    };
    if high_cut_hz > low_cut_hz && high_cut_hz < nyq {
        stage = BiquadCoeffs::butterworth_lowpass(high_cut_hz, sr).apply_zero_phase(&stage);
    }
    stage
}

/// Discrete Teager-Kaiser Energy Operator (`psi[n] = x[n]^2 - x[n-1]*x[n+1]`, Kaiser 1990).
/// Proportional to `A^2 * omega^2`, giving sharp transient amplification for bat pulses
/// and avian flight calls while suppressing low-frequency wind rumble.
pub fn compute_tkeo(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; n];
    for i in 1..n.saturating_sub(1) {
        let x0 = samples[i] as f64;
        let xm = samples[i - 1] as f64;
        let xp = samples[i + 1] as f64;
        let psi = (x0 * x0 - xm * xp).max(0.0);
        out[i] = psi as f32;
    }
    if n >= 2 {
        out[0] = out[1];
        out[n - 1] = out[n - 2];
    }
    out
}

/// Analytic signal envelope `|x(t) + j H{x(t)}|` via overlap-discard FFT Hilbert transform.
pub fn hilbert_envelope(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![samples[0].abs()];
    }
    let fft_len = if n <= 16_384 {
        (n * 2).next_power_of_two().clamp(64, 65_536)
    } else {
        65_536
    };
    let apron = fft_len / 4;
    let keep = fft_len / 2;
    let mut env = Vec::with_capacity(n);

    let reflect_sample = |idx: isize| -> f64 {
        let ni = n as isize;
        let mut p = idx;
        if p < 0 {
            p = -p;
        }
        if p >= ni {
            let period = 2 * (ni - 1);
            if period <= 0 {
                return samples[0] as f64;
            }
            p %= period;
            if p >= ni {
                p = period - p;
            }
        }
        samples[p as usize] as f64
    };

    let mut offset = 0usize;
    while offset < n {
        let take = (n - offset).min(keep);
        let mut buf = vec![Complex64::default(); fft_len];
        for (j, slot) in buf.iter_mut().enumerate() {
            let src_idx = (offset as isize) - (apron as isize) + (j as isize);
            *slot = Complex64::new(reflect_sample(src_idx), 0.0);
        }
        fft_radix2_in_place(&mut buf, false);
        // Analytic multiplier: 1 at DC and Nyquist, 2 on positive frequencies, 0 on negative
        for k in 1..(fft_len / 2) {
            buf[k].re *= 2.0;
            buf[k].im *= 2.0;
        }
        for b in buf.iter_mut().take(fft_len).skip(fft_len / 2 + 1) {
            *b = Complex64::default();
        }
        fft_radix2_in_place(&mut buf, true);
        for b in buf.iter().skip(apron).take(take) {
            env.push(b.norm() as f32);
        }
        offset += take;
    }
    env
}

/// Virtual Zero-Crossing Analyzer (`ZCA`) converting a full-spectrum waveform into Anabat-style
/// zero-crossing dots (`division_ratio` typically `8` or `16`, gated by `min_rms_gate`).
pub fn extract_zero_crossing_dots(
    samples: &[f32],
    sample_rate_hz: u32,
    division_ratio: u8,
    min_amp_gate: f32,
    min_freq_hz: f64,
    max_freq_hz: f64,
) -> Vec<ZeroCrossDot> {
    let div = division_ratio.max(1) as usize;
    let sr = sample_rate_hz as f64;
    let mut upward_crossings: Vec<(f64, f32)> = Vec::new();
    let mut peak_since_last = 0.0_f32;

    for i in 1..samples.len() {
        let s0 = samples[i - 1];
        let s1 = samples[i];
        peak_since_last = peak_since_last.max(s1.abs());
        if s0 <= 0.0 && s1 > 0.0 {
            let frac = if (s1 - s0).abs() > 1e-12 {
                (-s0 / (s1 - s0)) as f64
            } else {
                0.0
            };
            let t = ((i - 1) as f64 + frac) / sr;
            upward_crossings.push((t, peak_since_last));
            peak_since_last = 0.0;
        }
    }

    let mut dots = Vec::new();
    let mut idx = 0usize;
    while idx + div < upward_crossings.len() {
        let (t0, _) = upward_crossings[idx];
        let (t1, _) = upward_crossings[idx + div];
        let max_amp = upward_crossings[idx..=idx + div]
            .iter()
            .fold(0.0_f32, |a, &(_, p)| a.max(p));
        let dt = t1 - t0;
        if max_amp >= min_amp_gate && dt > 0.0 {
            let freq_hz = div as f64 / dt;
            if freq_hz >= min_freq_hz && freq_hz <= max_freq_hz {
                dots.push(ZeroCrossDot {
                    time_s: 0.5 * (t0 + t1),
                    freq_hz,
                    slope_khz_per_ms: 0.0,
                });
            }
        }
        idx += div;
    }

    // Compute local slope (kHz / ms) across sequential active dots
    for i in 1..dots.len() {
        let dt_ms = (dots[i].time_s - dots[i - 1].time_s) * 1000.0;
        if dt_ms > 0.0 && dt_ms < 3.0 {
            let df_khz = (dots[i].freq_hz - dots[i - 1].freq_hz) * 0.001;
            dots[i].slope_khz_per_ms = df_khz / dt_ms;
        }
    }
    if dots.len() >= 2 && dots[0].slope_khz_per_ms == 0.0 {
        dots[0].slope_khz_per_ms = dots[1].slope_khz_per_ms;
    }
    dots
}

/// Compute waveform min/max envelope buckets for crisp oscilloscope rendering at any zoom level.
pub fn downsample_waveform_minmax(
    samples: &[f32],
    start_idx: usize,
    end_idx: usize,
    columns: usize,
) -> Vec<(f32, f32)> {
    let cols = columns.max(1);
    let s = start_idx.min(samples.len());
    let e = end_idx.min(samples.len()).max(s);
    let span = e - s;
    if span == 0 {
        return vec![(0.0, 0.0); cols];
    }
    (0..cols)
        .map(|c| {
            let i0 = s + (c * span) / cols;
            let i1 = (s + ((c + 1) * span) / cols).max(i0 + 1).min(e);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in &samples[i0..i1] {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if !lo.is_finite() {
                (0.0, 0.0)
            } else {
                (lo, hi)
            }
        })
        .collect()
}

/// Check Parseval energy conservation ratio for self-testing FFT accuracy (`~1.0`).
pub fn parseval_ratio_check(signal: &[f64]) -> f64 {
    let m = signal.len().next_power_of_two();
    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    if time_energy <= 1e-18 {
        return 1.0;
    }
    let mut buf = vec![Complex64::default(); m];
    for (i, &x) in signal.iter().enumerate() {
        buf[i] = Complex64::new(x, 0.0);
    }
    fft_radix2_in_place(&mut buf, false);
    let freq_energy: f64 = buf.iter().map(|c| c.norm_sqr()).sum::<f64>() / (m as f64);
    let _ = PI; // keep math import clean
    freq_energy / time_energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_dc_and_nyquist_bins_not_inflated_by_6db() {
        let dc = vec![0.5_f32; 2048];
        let spec = compute_stft(
            &dc,
            48_000,
            StftConfig {
                fft_size: 512,
                hop_size: 128,
                window: WindowFunction::Hann,
                subtract_median_noise_floor: false,
                min_dbfs: -96.0,
            },
        );
        let mag0 = spec.mag_grid[0];
        let db0 = spec.db_at(0, 0);
        assert!(
            (mag0 - 0.5).abs() < 1e-4,
            "Expected DC bin magnitude == 0.5, got {mag0}"
        );
        assert!(
            (db0 - (-6.0206)).abs() < 0.05,
            "Expected DC bin -6.02 ± 0.05 dBFS, got {db0}"
        );
    }

    #[test]
    fn hilbert_envelope_seamless_across_32768_chunk_boundary() {
        let sr = 250_000.0_f64;
        let n = 40_000usize;
        let sig: Vec<f32> = (0..n)
            .map(|i| (TAU * 10_000.0 * (i as f64) / sr).sin() as f32)
            .collect();
        let env = hilbert_envelope(&sig);
        assert_eq!(env.len(), n);
        for (i, &val) in env.iter().enumerate().take(36_000).skip(28_000) {
            assert!(
                (0.98..=1.02).contains(&val),
                "Hilbert envelope at sample {i} deviated from unit amplitude: {val}"
            );
        }
    }

    #[test]
    fn zero_phase_butterworth_prewarps_minus_3db_and_avoids_edge_transient() {
        let sr = 100_000.0_f64;
        let fc = 10_000.0_f64;
        let n = 4096usize;
        let tone: Vec<f32> = (0..n)
            .map(|i| (TAU * fc * (i as f64) / sr).sin() as f32)
            .collect();
        let lp = BiquadCoeffs::butterworth_lowpass(fc, sr).apply_zero_phase(&tone);
        let rms_in: f64 = (tone[1024..3072].iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
            / 2048.0)
            .sqrt();
        let rms_out: f64 = (lp[1024..3072].iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
            / 2048.0)
            .sqrt();
        let gain_db = 20.0 * (rms_out / rms_in).log10();
        assert!(
            (gain_db - (-3.01)).abs() < 0.20,
            "Expected zero-phase LP -3.01 dB at fc={fc}, got {gain_db:.3} dB"
        );

        let dc_sig = vec![1.0_f32; 256];
        let dc_out = BiquadCoeffs::butterworth_lowpass(fc, sr).apply_zero_phase(&dc_sig);
        assert!(
            dc_out.iter().all(|&v| (v - 1.0).abs() < 1e-4),
            "Gustafsson steady-state padding should keep constant signal flat"
        );
    }
}
