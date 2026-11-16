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

//! Subharmonic Summation (`SHS`) Multi-Harmonic Stack Tracker (`f0`, `2f0`, `3f0`, `4f0`, `5f0`),
//! Harmonic-to-Noise Ratio (`HNR dB`), and Spectral Inharmonicity analyzer for bat multi-harmonic
//! passes (*Corynorhinus townsendii*, *Rhinolophus*) and avian harmonic stacks (*Ixoreus*, corvids,
//! woodpeckers, owls).

use crate::call_extractor::{DetectedCallEvent, TaxonDomain};
use crate::dsp::Spectrogram;
use serde::{Deserialize, Serialize};

pub const MAX_HARMONICS: usize = 5;

/// Detailed harmonic stack structure for a single detected bat pulse or bird vocalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicStackProfile {
    pub call_id: usize,
    /// Mean fundamental frequency `f0` (`kHz`) across active frames of the call.
    pub fundamental_f0_khz: f64,
    /// Relative harmonic amplitudes `[h1 (f0), h2 (2f0), h3 (3f0), h4 (4f0), h5 (5f0)]` in dB
    /// relative to the strongest harmonic (`0.0 dB` at dominant harmonic, `-40.0 dB` if above Nyquist/absent).
    pub relative_harmonic_db: [f32; MAX_HARMONICS],
    /// Number of harmonic rungs within `22 dB` of the dominant harmonic (`1..=5`).
    pub active_harmonic_count: usize,
    /// Harmonic order (`1..=5`) carrying peak acoustic power (`2` in whisper bats with dominant `2f0`).
    pub dominant_harmonic_order: usize,
    /// Harmonic-to-Noise Ratio (`HNR`) in dB comparing harmonic comb peaks against inter-harmonic troughs.
    pub hnr_db: f32,
    /// Mean fractional deviation `|f_obs(h) - h*f0| / (h*f0) * 100%` across active overtones (`h >= 2`).
    pub inharmonicity_pct: f64,
    /// Time-frequency overtone ridge contours `(harmonic_order, Vec<(time_s, freq_khz)>)` for `h = 2..=5`
    /// ready to overlay directly onto the STFT spectrogram.
    pub overtone_contours: Vec<(usize, Vec<(f64, f64)>)>,
}

/// Track fundamental `f0` and overtones `2f0..5f0` across all detected call events.
pub fn analyze_harmonic_stacks(
    calls: &[DetectedCallEvent],
    spec: &Spectrogram,
) -> Vec<HarmonicStackProfile> {
    if calls.is_empty() || spec.num_frames == 0 {
        return Vec::new();
    }
    calls
        .iter()
        .map(|ev| track_single_call_harmonics(ev, spec))
        .collect()
}

pub fn track_single_call_harmonics(
    ev: &DetectedCallEvent,
    spec: &Spectrogram,
) -> HarmonicStackProfile {
    let nyquist_khz = spec.sample_rate_hz as f64 * 0.0005;
    let bin_khz = (spec.sample_rate_hz as f64 / spec.fft_size as f64) * 0.001;
    let min_f0_khz = match ev.domain {
        TaxonDomain::BatChiroptera => 12.0,
        TaxonDomain::BirdAves => 0.45,
    };

    let mut f0_acc = 0.0_f64;
    let mut h_lin_acc = [0.0_f64; MAX_HARMONICS];
    let mut comb_power_sum = 0.0_f64;
    let mut trough_power_sum = 0.0_f64;
    let mut inharm_acc = 0.0_f64;
    let mut inharm_cnt = 0usize;
    let mut overtone_pts: [Vec<(f64, f64)>; MAX_HARMONICS] = Default::default();

    for &(t_s, ridge_f_khz) in &ev.ridge_trace {
        let fr = ((t_s * spec.sample_rate_hz as f64) / spec.hop_size as f64)
            .round()
            .clamp(0.0, (spec.num_frames - 1) as f64) as usize;
        let base = fr * spec.num_bins;

        // Subharmonic Summation (SHS) check: test whether `ridge_f_khz` is f0, 2f0, or 3f0
        let mut best_f0 = ridge_f_khz.max(min_f0_khz);
        let mut best_shs = -1.0_f64;

        for div in [1usize, 2, 3] {
            let cand_f0 = ridge_f_khz / (div as f64);
            if cand_f0 < min_f0_khz {
                continue;
            }
            let b_cand = spec.freq_to_bin(cand_f0 * 1000.0);
            let snr_cand = spec.db_at(fr, b_cand) - spec.bin_noise_floor_dbfs[b_cand];
            if div > 1 && snr_cand < 6.5 {
                continue;
            }
            let mut shs_score = 0.0_f64;
            for h in 1..=MAX_HARMONICS {
                let fh = cand_f0 * (h as f64);
                if fh >= nyquist_khz * 0.98 {
                    break;
                }
                let bh = spec.freq_to_bin(fh * 1000.0);
                let mag = spec.mag_grid[base + bh] as f64;
                // Weight lower harmonics via classic Hermes 0.84^(h-1) decay
                shs_score += mag * 0.84_f64.powi((h - 1) as i32);
            }
            if shs_score > best_shs {
                best_shs = shs_score;
                best_f0 = cand_f0;
            }
        }

        f0_acc += best_f0;

        // Sample harmonic peaks & mid-harmonic troughs (`(h + 0.5) * f0`)
        for h in 1..=MAX_HARMONICS {
            let target_khz = best_f0 * (h as f64);
            if target_khz >= nyquist_khz * 0.97 {
                continue;
            }
            let center_b = spec.freq_to_bin(target_khz * 1000.0);
            let search_rad = ((0.08 * best_f0 / bin_khz.max(1e-4)).ceil() as usize).clamp(1, 5);
            let b_lo = center_b.saturating_sub(search_rad).max(1);
            let b_hi = (center_b + search_rad).min(spec.num_bins - 1);

            let mut pk_b = center_b;
            let mut pk_mag = 0.0_f64;
            for b in b_lo..=b_hi {
                let m = spec.mag_grid[base + b] as f64;
                if m > pk_mag {
                    pk_mag = m;
                    pk_b = b;
                }
            }
            let p_harm = pk_mag * pk_mag;
            h_lin_acc[h - 1] += p_harm;
            comb_power_sum += p_harm;

            let obs_khz = pk_b as f64 * bin_khz;
            let snr_h = spec.db_at(fr, pk_b) - spec.bin_noise_floor_dbfs[pk_b];
            if h >= 2 && snr_h >= 7.0 {
                overtone_pts[h - 1].push((t_s, obs_khz));
                inharm_acc += ((obs_khz - target_khz).abs() / target_khz.max(0.1)) * 100.0;
                inharm_cnt += 1;
            }

            // Inter-harmonic trough at `(h + 0.5) * f0`
            let trough_khz = best_f0 * (h as f64 + 0.5);
            if trough_khz < nyquist_khz * 0.97 {
                let tb = spec.freq_to_bin(trough_khz * 1000.0);
                let tm = spec.mag_grid[base + tb] as f64;
                trough_power_sum += tm * tm;
            }
        }
    }

    let n_frames = ev.ridge_trace.len().max(1) as f64;
    let fundamental_f0_khz = if ev.ridge_trace.is_empty() {
        ev.f_char_khz
    } else {
        f0_acc / n_frames
    };

    let max_h_pow = h_lin_acc
        .iter()
        .copied()
        .fold(1e-20_f64, f64::max);

    let mut relative_harmonic_db = [-40.0_f32; MAX_HARMONICS];
    let mut dominant_harmonic_order = 1usize;
    let mut best_db = -120.0_f32;
    let mut active_harmonic_count = 0usize;

    for h in 0..MAX_HARMONICS {
        if h_lin_acc[h] > 1e-22 {
            let rel_db = (10.0 * (h_lin_acc[h] / max_h_pow).log10() as f32).max(-40.0);
            relative_harmonic_db[h] = rel_db;
            if rel_db > best_db {
                best_db = rel_db;
                dominant_harmonic_order = h + 1;
            }
            if rel_db >= -22.0 {
                active_harmonic_count += 1;
            }
        }
    }
    active_harmonic_count = active_harmonic_count.max(1);

    let hnr_db = (10.0 * ((comb_power_sum + 1e-18) / (trough_power_sum + 1e-18)).log10() as f32)
        .clamp(-10.0, 48.0);
    let inharmonicity_pct = if inharm_cnt > 0 {
        inharm_acc / inharm_cnt as f64
    } else {
        0.0
    };

    let mut overtone_contours = Vec::new();
    for h in 2..=MAX_HARMONICS {
        if overtone_pts[h - 1].len() >= 2 && relative_harmonic_db[h - 1] >= -24.0 {
            overtone_contours.push((h, overtone_pts[h - 1].clone()));
        }
    }

    HarmonicStackProfile {
        call_id: ev.id,
        fundamental_f0_khz,
        relative_harmonic_db,
        active_harmonic_count,
        dominant_harmonic_order,
        hnr_db,
        inharmonicity_pct,
        overtone_contours,
    }
}
