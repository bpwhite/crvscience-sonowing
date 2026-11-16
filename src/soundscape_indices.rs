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

//! Six standardized peer-reviewed Ecoacoustic Soundscape Indices:
//! 1. `ACI` — Acoustic Complexity Index (Pieretti et al. 2011)
//! 2. `NDSI` — Normalized Difference Soundscape Index (Kasten et al. 2012)
//! 3. `BIO` — Bioacoustic Index (Boelman et al. 2007)
//! 4. `ADI` — Acoustic Diversity Index (Villanueva-Rivera et al. 2011)
//! 5. `AEI` — Acoustic Evenness Index (Villanueva-Rivera et al. 2011, Gini coefficient)
//! 6. `H = Ht * Hf` — Total Acoustic Entropy (Sueur et al. 2008)

use crate::dsp::{hilbert_envelope, Spectrogram};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundscapeIndices {
    /// Acoustic Complexity Index (`ACI`, Pieretti et al. 2011).
    pub aci: f64,
    /// Normalized Difference Soundscape Index (`[-1.0, +1.0]`, Kasten et al. 2012: 2-11 kHz Biophony vs. 1-2 kHz Anthrophony).
    pub ndsi: f64,
    /// Bioacoustic Index (`BIO`, Boelman et al. 2007: dB*kHz area above floor across 2-8 kHz).
    pub bio_index: f64,
    /// Acoustic Diversity Index (`ADI`, Shannon entropy across 1 kHz occupancy bands above -50 dBFS).
    pub adi: f64,
    /// Acoustic Evenness Index (`AEI`, Gini coefficient across 1 kHz occupancy bands in `[0, 1]`).
    pub aei: f64,
    /// Temporal amplitude envelope Shannon entropy (`Ht` in `[0, 1]`, Sueur et al. 2008).
    pub temporal_entropy_ht: f64,
    /// Spectral mean power Shannon entropy (`Hf` in `[0, 1]`, Sueur et al. 2008).
    pub spectral_entropy_hf: f64,
    /// Total Acoustic Entropy (`H = Ht * Hf` in `[0, 1]`, Sueur et al. 2008).
    pub total_entropy_h: f64,
    /// Ultrasonic Chiropteran energy share (`> 16 kHz` power fraction in `[0, 1]`).
    pub ultrasonic_energy_fraction: f64,
    /// Per-band occupancy proportions (`(band_label, occupancy_fraction)`).
    pub band_occupancy: Vec<(String, f64)>,
}

pub fn compute_soundscape_indices(samples: &[f32], spec: &Spectrogram) -> SoundscapeIndices {
    if spec.num_frames < 2 || spec.num_bins < 4 || samples.is_empty() {
        return SoundscapeIndices {
            aci: 0.0,
            ndsi: 0.0,
            bio_index: 0.0,
            adi: 0.0,
            aei: 0.0,
            temporal_entropy_ht: 0.0,
            spectral_entropy_hf: 0.0,
            total_entropy_h: 0.0,
            ultrasonic_energy_fraction: 0.0,
            band_occupancy: Vec::new(),
        };
    }

    let nyquist_hz = spec.sample_rate_hz as f64 * 0.5;
    let bin_hz = spec.sample_rate_hz as f64 / spec.fft_size as f64;

    // 1. ACI (Pieretti et al. 2011): sum over frequency bins k of (sum_t |I_{t,k} - I_{t-1,k}| / sum_t I_{t,k})
    let mut aci = 0.0_f64;
    for b in 1..spec.num_bins {
        let mut diff_sum = 0.0_f64;
        let mut mag_sum = 0.0_f64;
        for f in 0..spec.num_frames {
            let m = spec.mag_grid[f * spec.num_bins + b] as f64;
            mag_sum += m;
            if f > 0 {
                let m_prev = spec.mag_grid[(f - 1) * spec.num_bins + b] as f64;
                diff_sum += (m - m_prev).abs();
            }
        }
        if mag_sum > 1e-12 {
            aci += diff_sum / mag_sum;
        }
    }

    // 2. NDSI (Kasten et al. 2012): Biophony (2-11 kHz, or 16-90 kHz if pure ultrasonic) vs Anthrophony (0.5-2 kHz)
    let mut anthro_psd = 0.0_f64;
    let mut biophony_psd = 0.0_f64;
    let mut ultra_psd = 0.0_f64;
    let mut total_psd = 0.0_f64;
    let mut mean_bin_db = vec![-96.0_f64; spec.num_bins];
    let mut mean_bin_power = vec![0.0_f64; spec.num_bins];

    for b in 1..spec.num_bins {
        let freq = b as f64 * bin_hz;
        let mut p_sum = 0.0_f64;
        for f in 0..spec.num_frames {
            let m = spec.mag_grid[f * spec.num_bins + b] as f64;
            p_sum += m * m;
        }
        let p_mean = p_sum / spec.num_frames as f64;
        mean_bin_power[b] = p_mean;
        mean_bin_db[b] = if p_mean > 1e-20 {
            10.0 * p_mean.log10()
        } else {
            -100.0
        };
        total_psd += p_mean;

        if (500.0..2_000.0).contains(&freq) {
            anthro_psd += p_mean;
        } else if (2_000.0..=11_000.0).contains(&freq) {
            biophony_psd += p_mean;
        }
        if freq >= 16_000.0 {
            ultra_psd += p_mean;
        }
    }

    // Include ultrasonic biophony when analyzing bat recordings (> 48 kHz sample rate)
    let effective_bio = if nyquist_hz > 25_000.0 {
        biophony_psd + ultra_psd
    } else {
        biophony_psd
    };
    let ndsi = if effective_bio + anthro_psd > 1e-20 {
        ((effective_bio - anthro_psd) / (effective_bio + anthro_psd)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let ultrasonic_energy_fraction = if total_psd > 1e-20 {
        (ultra_psd / total_psd).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // 3. BIO (Boelman et al. 2007): area under mean dB spectrum across 2-8 kHz (or 18-80 kHz for ultrasonic-only)
    let (bio_lo, bio_hi) = if ultrasonic_energy_fraction > 0.65 && nyquist_hz > 40_000.0 {
        (18_000.0, nyquist_hz.min(90_000.0))
    } else {
        (2_000.0, nyquist_hz.min(8_000.0))
    };
    let b_lo = spec.freq_to_bin(bio_lo).max(1);
    let b_hi = spec.freq_to_bin(bio_hi).min(spec.num_bins - 1);
    let bio_index = if b_hi > b_lo {
        let min_db = mean_bin_db[b_lo..=b_hi]
            .iter()
            .fold(f64::INFINITY, |a, &x| a.min(x));
        let df_khz = bin_hz * 0.001;
        mean_bin_db[b_lo..=b_hi]
            .iter()
            .map(|&db| (db - min_db).max(0.0) * df_khz)
            .sum::<f64>()
    } else {
        0.0
    };

    // 4 & 5. ADI & AEI (Villanueva-Rivera et al. 2011): 10 equal frequency bands up to active ceiling, -50 dBFS threshold
    let num_bands = 10usize;
    let max_band_hz = if nyquist_hz > 25_000.0 {
        nyquist_hz.min(120_000.0)
    } else {
        nyquist_hz.min(12_000.0)
    };
    let step_hz = max_band_hz / num_bands as f64;
    let thresh_dbfs = -52.0_f32;
    let mut props = Vec::with_capacity(num_bands);
    let mut band_occupancy = Vec::with_capacity(num_bands);

    for k in 0..num_bands {
        let f0 = k as f64 * step_hz;
        let f1 = (k + 1) as f64 * step_hz;
        let kb0 = spec.freq_to_bin(f0).max(1);
        let kb1 = spec.freq_to_bin(f1).min(spec.num_bins - 1).max(kb0);
        let total_cells = ((kb1 - kb0 + 1) * spec.num_frames).max(1);
        let mut active_cells = 0usize;
        for fr in 0..spec.num_frames {
            for b in kb0..=kb1 {
                if spec.db_at(fr, b) >= thresh_dbfs {
                    active_cells += 1;
                }
            }
        }
        let occ = active_cells as f64 / total_cells as f64;
        props.push(occ);
        band_occupancy.push((format!("{:.1}-{:.1} kHz", f0 * 0.001, f1 * 0.001), occ));
    }

    let prop_sum: f64 = props.iter().sum();
    let adi = if prop_sum > 1e-12 {
        props
            .iter()
            .filter(|&&v| v > 1e-12)
            .map(|&v| {
                let p = v / prop_sum;
                -p * p.ln()
            })
            .sum::<f64>()
    } else {
        0.0
    };

    let aei = gini_coefficient(&props);

    // 6. Total Acoustic Entropy H = Ht * Hf (Sueur et al. 2008)
    let env = hilbert_envelope(samples);
    let env_sum: f64 = env.iter().map(|&e| e.max(0.0) as f64).sum();
    let ht = if env_sum > 1e-12 && env.len() > 1 {
        let h_raw: f64 = env
            .iter()
            .filter(|&&e| e > 1e-12)
            .map(|&e| {
                let p = (e as f64) / env_sum;
                -p * p.ln()
            })
            .sum();
        (h_raw / (env.len() as f64).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let active_spec_bins = (spec.num_bins - 1).max(1);
    let hf = if total_psd > 1e-20 && active_spec_bins > 1 {
        let s_raw: f64 = mean_bin_power[1..]
            .iter()
            .filter(|&&p| p > 1e-24)
            .map(|&p| {
                let q = p / total_psd;
                -q * q.ln()
            })
            .sum();
        (s_raw / (active_spec_bins as f64).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    SoundscapeIndices {
        aci,
        ndsi,
        bio_index,
        adi,
        aei,
        temporal_entropy_ht: ht,
        spectral_entropy_hf: hf,
        total_entropy_h: ht * hf,
        ultrasonic_energy_fraction,
        band_occupancy,
    }
}

fn gini_coefficient(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().map(|&x| x.max(0.0)).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = sorted.iter().sum();
    if sum <= 1e-12 {
        return 0.0;
    }
    let mut weighted = 0.0_f64;
    for (i, &v) in sorted.iter().enumerate() {
        weighted += ((i + 1) as f64) * v;
    }
    let n_f = n as f64;
    ((2.0 * weighted) / (n_f * sum) - (n_f + 1.0) / n_f).clamp(0.0, 1.0)
}
