//! Ultrasonic-to-audible acoustic auditioning engine for human monitoring and playable WAV export:
//! 1. **Time Expansion (`10x` / `20x`)** — Exact sample-rate rescaling preserving harmonics and phase.
//! 2. **Digital Heterodyne (`f_tune = 15..130 kHz`)** — Ring modulation `x(t)*cos(2*pi*f_tune*t)`
//!    followed by a 4th-order `4.5 kHz` Butterworth low-pass filter and linear resampling to `48 kHz`.
//! 3. **Frequency Division (`1/8`, `1/10`, `1/16`) with Hilbert/RMS Envelope Restoration** —
//!    Zero-crossing frequency divider modulated by the instantaneous amplitude envelope.

use crate::dsp::{hilbert_envelope, BiquadCoeffs};
use crate::wav_guano::{BioAcousticRecording, GuanoMetadata, SampleEncoding};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AuditionMode {
    TimeExpansion10x,
    TimeExpansion20x,
    Heterodyne { carrier_khz: f64 },
    FrequencyDivision { ratio: u8 },
}

impl AuditionMode {
    pub fn label(self) -> String {
        match self {
            Self::TimeExpansion10x => "Time Expansion (10x, Full Harmonic Fidelity)".to_string(),
            Self::TimeExpansion20x => "Time Expansion (20x, Ultra-High >90 kHz)".to_string(),
            Self::Heterodyne { carrier_khz } => {
                format!("Digital Heterodyne Mixer ({carrier_khz:.1} kHz Carrier)")
            }
            Self::FrequencyDivision { ratio } => {
                format!("Frequency Division (1/{ratio} + Envelope Restoration)")
            }
        }
    }
}

/// Convert an ultrasonic (or wide-band) recording into a human-audible PCM waveform.
pub fn render_audible_recording(
    rec: &BioAcousticRecording,
    mode: AuditionMode,
    target_audible_rate_hz: u32,
) -> BioAcousticRecording {
    let out_rate = target_audible_rate_hz.clamp(16_000, 96_000);
    let mut new_guano = rec
        .guano
        .clone()
        .unwrap_or_else(GuanoMetadata::new_v1);

    match mode {
        AuditionMode::TimeExpansion10x | AuditionMode::TimeExpansion20x => {
            let factor = if matches!(mode, AuditionMode::TimeExpansion10x) {
                10u32
            } else {
                20u32
            };
            // Slow playback rate by `factor`, then resample cleanly to `out_rate`
            let slowed_virtual_rate = (rec.sample_rate_hz / factor).max(8_000);
            let resampled = linear_resample(&rec.samples, slowed_virtual_rate, out_rate);
            new_guano.time_expansion = Some(factor);
            new_guano.samplerate_hz = Some(rec.sample_rate_hz);
            new_guano
                .extra_fields
                .insert("SONOWING|AuditionMode".to_string(), format!("TE_{factor}x"));

            BioAcousticRecording {
                source_name: format!("{}_TE{}x.wav", strip_ext(&rec.source_name), factor),
                sample_rate_hz: out_rate,
                raw_header_rate_hz: out_rate,
                channels: 1,
                encoding: SampleEncoding::PcmI16,
                samples: normalize_peak(&resampled, 0.92),
                guano: Some(new_guano),
                riff_info: BTreeMap::new(),
            }
        }
        AuditionMode::Heterodyne { carrier_khz } => {
            let sr = rec.sample_rate_hz as f64;
            let f_lo = (carrier_khz * 1000.0).clamp(8_000.0, 0.48 * sr);
            let mut mixed = Vec::with_capacity(rec.samples.len());
            for (i, &x) in rec.samples.iter().enumerate() {
                let phase = TAU * f_lo * (i as f64) / sr;
                mixed.push((x as f64 * phase.cos() * 2.0) as f32);
            }
            // Low-pass filter difference-frequency audio band at 4.5 kHz before decimation
            let lp = BiquadCoeffs::butterworth_lowpass(4_500.0_f64.min(0.45 * sr), sr);
            let baseband = lp.apply_zero_phase(&mixed);
            let resampled = linear_resample(&baseband, rec.sample_rate_hz, out_rate);

            new_guano.time_expansion = Some(1);
            new_guano.extra_fields.insert(
                "SONOWING|AuditionMode".to_string(),
                format!("Heterodyne_{carrier_khz:.1}kHz"),
            );

            BioAcousticRecording {
                source_name: format!(
                    "{}_HET_{:.0}kHz.wav",
                    strip_ext(&rec.source_name),
                    carrier_khz
                ),
                sample_rate_hz: out_rate,
                raw_header_rate_hz: out_rate,
                channels: 1,
                encoding: SampleEncoding::PcmI16,
                samples: normalize_peak(&resampled, 0.92),
                guano: Some(new_guano),
                riff_info: BTreeMap::new(),
            }
        }
        AuditionMode::FrequencyDivision { ratio } => {
            let div = ratio.max(2) as usize;
            let env = hilbert_envelope(&rec.samples);
            let mut divided = vec![0.0_f32; rec.samples.len()];
            let mut zc_count = 0usize;
            let mut sign_out = 1.0_f32;

            for i in 1..rec.samples.len() {
                if (rec.samples[i - 1] <= 0.0 && rec.samples[i] > 0.0)
                    || (rec.samples[i - 1] >= 0.0 && rec.samples[i] < 0.0)
                {
                    zc_count += 1;
                    if zc_count >= div {
                        zc_count = 0;
                        sign_out = -sign_out;
                    }
                }
                divided[i] = sign_out * env[i];
            }

            // Smooth square-wave harmonics above 9 kHz so playback sounds crisp rather than harsh
            let sr = rec.sample_rate_hz as f64;
            let lp = BiquadCoeffs::butterworth_lowpass(9_000.0_f64.min(0.45 * sr), sr);
            let smoothed = lp.apply_zero_phase(&divided);
            let resampled = linear_resample(&smoothed, rec.sample_rate_hz, out_rate);

            new_guano.time_expansion = Some(1);
            new_guano.extra_fields.insert(
                "SONOWING|AuditionMode".to_string(),
                format!("FreqDiv_1_{div}"),
            );

            BioAcousticRecording {
                source_name: format!("{}_FDIV_{}.wav", strip_ext(&rec.source_name), div),
                sample_rate_hz: out_rate,
                raw_header_rate_hz: out_rate,
                channels: 1,
                encoding: SampleEncoding::PcmI16,
                samples: normalize_peak(&resampled, 0.92),
                guano: Some(new_guano),
                riff_info: BTreeMap::new(),
            }
        }
    }
}

fn linear_resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = input.get(idx).copied().unwrap_or(0.0);
        let s1 = input.get(idx + 1).copied().unwrap_or(s0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

fn normalize_peak(samples: &[f32], target_peak: f32) -> Vec<f32> {
    let peak = samples
        .iter()
        .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    if peak <= 1e-9 {
        return samples.to_vec();
    }
    let gain = target_peak / peak;
    samples.iter().map(|&x| (x * gain).clamp(-1.0, 1.0)).collect()
}

fn strip_ext(name: &str) -> &str {
    name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name)
}
