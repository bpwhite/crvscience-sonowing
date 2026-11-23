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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
            // Slow playback rate by `factor` without clamping below 8 kHz so 48 kHz
            // recordings achieve exact 10x / 20x audible duration ratios.
            let slowed_virtual_rate = (rec.sample_rate_hz.max(1) as f64) / (factor as f64);
            let resampled = linear_resample_f64(&rec.samples, slowed_virtual_rate, out_rate as f64);
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
            let sr = (rec.sample_rate_hz as f64).max(1_000.0);
            let safe_carrier_khz = if carrier_khz.is_finite() {
                carrier_khz
            } else {
                40.0
            };
            let hi_lo = (0.48 * sr).max(200.0);
            let min_lo = 8_000.0_f64.min(hi_lo * 0.5);
            let f_lo = (safe_carrier_khz * 1000.0).clamp(min_lo, hi_lo);
            let mut mixed = Vec::with_capacity(rec.samples.len());
            for (i, &x) in rec.samples.iter().enumerate() {
                let phase = TAU * f_lo * (i as f64) / sr;
                let x_clean = if x.is_finite() { x as f64 } else { 0.0 };
                mixed.push((x_clean * phase.cos() * 2.0) as f32);
            }
            // Low-pass filter difference-frequency audio band safely below 0.45 * out_rate before decimation
            let lp_cutoff = 4_500.0_f64.min(0.45 * (out_rate as f64)).min(0.45 * sr);
            let lp = BiquadCoeffs::butterworth_lowpass(lp_cutoff, sr);
            let baseband = lp.apply_zero_phase(&mixed);
            let resampled = linear_resample(&baseband, rec.sample_rate_hz.max(1_000), out_rate);

            new_guano.time_expansion = Some(1);
            new_guano.extra_fields.insert(
                "SONOWING|AuditionMode".to_string(),
                format!("Heterodyne_{safe_carrier_khz:.1}kHz"),
            );

            BioAcousticRecording {
                source_name: format!(
                    "{}_HET_{:.0}kHz.wav",
                    strip_ext(&rec.source_name),
                    safe_carrier_khz
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
            let clean_samples: Vec<f32> = rec
                .samples
                .iter()
                .map(|&s| if s.is_finite() { s } else { 0.0 })
                .collect();
            let env = hilbert_envelope(&clean_samples);
            let mut divided = vec![0.0_f32; clean_samples.len()];
            let mut zc_count = 0usize;
            let mut sign_out = 1.0_f32;

            for i in 1..clean_samples.len() {
                if (clean_samples[i - 1] <= 0.0 && clean_samples[i] > 0.0)
                    || (clean_samples[i - 1] >= 0.0 && clean_samples[i] < 0.0)
                {
                    zc_count += 1;
                    if zc_count >= div {
                        zc_count = 0;
                        sign_out = -sign_out;
                    }
                }
                divided[i] = sign_out * env[i];
            }

            // Smooth square-wave harmonics safely below 0.45 * out_rate before decimation
            let sr = (rec.sample_rate_hz as f64).max(1_000.0);
            let lp_cutoff = 9_000.0_f64.min(0.45 * (out_rate as f64)).min(0.45 * sr);
            let lp = BiquadCoeffs::butterworth_lowpass(lp_cutoff, sr);
            let smoothed = lp.apply_zero_phase(&divided);
            let resampled = linear_resample(&smoothed, rec.sample_rate_hz.max(1_000), out_rate);

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

pub fn linear_resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    linear_resample_f64(input, src_rate as f64, dst_rate as f64)
}

fn linear_resample_f64(input: &[f32], src_rate: f64, dst_rate: f64) -> Vec<f32> {
    if input.is_empty()
        || !src_rate.is_finite()
        || !dst_rate.is_finite()
        || src_rate <= 0.0
        || dst_rate <= 0.0
    {
        return Vec::new();
    }
    if (src_rate - dst_rate).abs() <= 1e-6 {
        return input
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .collect();
    }
    let filtered_buf;
    let src_samples: &[f32] = if dst_rate < src_rate && input.len() >= 16 {
        let anti_alias_hz = (0.45 * dst_rate).min(0.45 * src_rate);
        if anti_alias_hz > 5.0 && anti_alias_hz < 0.49 * src_rate {
            let clean: Vec<f32> = input
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect();
            filtered_buf = BiquadCoeffs::butterworth_lowpass(anti_alias_hz, src_rate)
                .apply_zero_phase(&clean);
            &filtered_buf
        } else {
            input
        }
    } else {
        input
    };
    let ratio = src_rate / dst_rate;
    let out_len = ((src_samples.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = src_samples
            .get(idx)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let s1 = src_samples
            .get(idx + 1)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(s0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

fn normalize_peak(samples: &[f32], target_peak: f32) -> Vec<f32> {
    let peak = samples
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    if peak <= 1e-9 {
        return samples
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .collect();
    }
    let gain = target_peak / peak;
    samples
        .iter()
        .map(|&x| {
            if x.is_finite() {
                (x * gain).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn strip_ext(name: &str) -> &str {
    name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name)
}

#[derive(Debug, Clone)]
pub struct AudiblePlaybackBuffer {
    pub samples: Arc<Vec<f32>>,
    pub sample_rate_hz: u32,
    pub source_duration_s: f64,
    pub mode: AuditionMode,
}

/// Render an ultrasonic/wideband recording into an audible mono buffer resampled to
/// `device_sample_rate_hz` for real-time `cpal` output.
pub fn prepare_playback_buffer(
    rec: &BioAcousticRecording,
    mode: AuditionMode,
    device_sample_rate_hz: u32,
) -> AudiblePlaybackBuffer {
    let target_dev_sr = device_sample_rate_hz.clamp(8_000, 192_000);
    let render_sr = target_dev_sr.clamp(16_000, 96_000);
    let audible = render_audible_recording(rec, mode, render_sr);
    let samples = if audible.sample_rate_hz == target_dev_sr {
        audible.samples
    } else {
        linear_resample(&audible.samples, audible.sample_rate_hz, target_dev_sr)
    };
    AudiblePlaybackBuffer {
        samples: Arc::new(samples),
        sample_rate_hz: target_dev_sr,
        source_duration_s: rec.duration_s().max(0.0),
        mode,
    }
}

/// Map the current playback sample cursor in the audible buffer back to the original
/// source recording timestamp in seconds `[0.0, source_duration_s]`.
pub fn cursor_to_source_time_s(
    cursor_sample: usize,
    total_samples: usize,
    source_duration_s: f64,
) -> f64 {
    if total_samples == 0 || !source_duration_s.is_finite() || source_duration_s <= 0.0 {
        return 0.0;
    }
    let frac = (cursor_sample.min(total_samples) as f64) / (total_samples as f64);
    (frac * source_duration_s).clamp(0.0, source_duration_s)
}

pub fn fill_output_slice_f32(
    out: &mut [f32],
    channels: usize,
    samples: &[f32],
    cursor: &AtomicUsize,
    is_paused: &AtomicBool,
    finished: &AtomicBool,
) {
    let ch = channels.max(1);
    if is_paused.load(Ordering::Relaxed) || samples.is_empty() {
        out.fill(0.0);
        return;
    }
    let mut pos = cursor.load(Ordering::Relaxed);
    for frame in out.chunks_mut(ch) {
        if pos < samples.len() {
            let raw = samples[pos];
            let s = if raw.is_finite() {
                raw.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            frame.fill(s);
            pos += 1;
        } else {
            frame.fill(0.0);
            finished.store(true, Ordering::Relaxed);
        }
    }
    cursor.store(pos, Ordering::Relaxed);
    if pos >= samples.len() {
        finished.store(true, Ordering::Relaxed);
    }
}

pub fn fill_output_slice_i16(
    out: &mut [i16],
    channels: usize,
    samples: &[f32],
    cursor: &AtomicUsize,
    is_paused: &AtomicBool,
    finished: &AtomicBool,
) {
    let ch = channels.max(1);
    if is_paused.load(Ordering::Relaxed) || samples.is_empty() {
        out.fill(0);
        return;
    }
    let mut pos = cursor.load(Ordering::Relaxed);
    for frame in out.chunks_mut(ch) {
        if pos < samples.len() {
            let raw = samples[pos];
            let clamped = if raw.is_finite() {
                raw.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let s = (clamped * 32767.0).round() as i16;
            frame.fill(s);
            pos += 1;
        } else {
            frame.fill(0);
            finished.store(true, Ordering::Relaxed);
        }
    }
    cursor.store(pos, Ordering::Relaxed);
    if pos >= samples.len() {
        finished.store(true, Ordering::Relaxed);
    }
}

pub fn fill_output_slice_u16(
    out: &mut [u16],
    channels: usize,
    samples: &[f32],
    cursor: &AtomicUsize,
    is_paused: &AtomicBool,
    finished: &AtomicBool,
) {
    let ch = channels.max(1);
    if is_paused.load(Ordering::Relaxed) || samples.is_empty() {
        out.fill(32768);
        return;
    }
    let mut pos = cursor.load(Ordering::Relaxed);
    for frame in out.chunks_mut(ch) {
        if pos < samples.len() {
            let raw = samples[pos];
            let clamped = if raw.is_finite() {
                raw.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let s = ((clamped * 32767.0).round() as i32 + 32768)
                .clamp(0, 65535) as u16;
            frame.fill(s);
            pos += 1;
        } else {
            frame.fill(32768);
            finished.store(true, Ordering::Relaxed);
        }
    }
    cursor.store(pos, Ordering::Relaxed);
    if pos >= samples.len() {
        finished.store(true, Ordering::Relaxed);
    }
}

struct PlaybackSharedState {
    cursor_sample: AtomicUsize,
    total_samples: AtomicUsize,
    source_duration_bits: AtomicU64,
    is_playing: AtomicBool,
    is_paused: AtomicBool,
    finished: AtomicBool,
    last_error: Mutex<Option<String>>,
}

impl PlaybackSharedState {
    fn new() -> Self {
        Self {
            cursor_sample: AtomicUsize::new(0),
            total_samples: AtomicUsize::new(0),
            source_duration_bits: AtomicU64::new(0.0_f64.to_bits()),
            is_playing: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }
    }
}

/// Thread-safe, zero-unsafe `cpal` audible output player with synchronized source playhead.
pub struct UltrasonicAudioPlayer {
    shared: Arc<PlaybackSharedState>,
    #[cfg(feature = "hardware-mic")]
    stream: Option<cpal::Stream>,
    active_mode: Option<AuditionMode>,
}

impl Default for UltrasonicAudioPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for UltrasonicAudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

impl UltrasonicAudioPlayer {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(PlaybackSharedState::new()),
            #[cfg(feature = "hardware-mic")]
            stream: None,
            active_mode: None,
        }
    }

    pub fn is_playing(&self) -> bool {
        self.shared.is_playing.load(Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.shared.is_paused.load(Ordering::Relaxed)
    }

    pub fn is_finished(&self) -> bool {
        self.shared.finished.load(Ordering::Relaxed)
    }

    pub fn active_mode(&self) -> Option<AuditionMode> {
        self.active_mode
    }

    pub fn current_source_time_s(&self) -> Option<f64> {
        if !self.is_playing() {
            return None;
        }
        let cursor = self.shared.cursor_sample.load(Ordering::Relaxed);
        let total = self.shared.total_samples.load(Ordering::Relaxed);
        let dur_s = f64::from_bits(self.shared.source_duration_bits.load(Ordering::Relaxed));
        Some(cursor_to_source_time_s(cursor, total, dur_s))
    }

    pub fn take_error(&self) -> Option<String> {
        self.shared
            .last_error
            .lock()
            .ok()
            .and_then(|mut g| g.take())
    }

    pub fn pause_or_resume(&mut self) {
        if !self.is_playing() {
            return;
        }
        let was_paused = self.shared.is_paused.load(Ordering::Relaxed);
        let next_paused = !was_paused;
        self.shared.is_paused.store(next_paused, Ordering::Relaxed);
        #[cfg(feature = "hardware-mic")]
        if let Some(ref stream) = self.stream {
            use cpal::traits::StreamTrait;
            let res = if next_paused {
                stream.pause().map_err(|e| e.to_string())
            } else {
                stream.play().map_err(|e| e.to_string())
            };
            if let Err(e) = res {
                if let Ok(mut g) = self.shared.last_error.lock() {
                    *g = Some(format!("Audio stream pause/resume error: {e}"));
                }
            }
        }
    }

    pub fn stop(&mut self) {
        #[cfg(feature = "hardware-mic")]
        {
            self.stream = None;
        }
        self.active_mode = None;
        self.shared.is_playing.store(false, Ordering::Relaxed);
        self.shared.is_paused.store(false, Ordering::Relaxed);
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.cursor_sample.store(0, Ordering::Relaxed);
    }

    /// Transcode and stream the recording to the default OS audio output device via `cpal`,
    /// starting at `start_fraction` in `[0.0, 0.99]`. Returns an informative `Err` (never panics)
    /// when running in headless/CI environments without an audio output device.
    pub fn play_recording(
        &mut self,
        rec: &BioAcousticRecording,
        mode: AuditionMode,
        start_fraction: f64,
    ) -> Result<String, String> {
        self.stop();
        if rec.samples.is_empty() {
            return Err("Cannot play empty recording".to_string());
        }

        #[cfg(feature = "hardware-mic")]
        {
            use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .ok_or_else(|| "No hardware audio output device detected".to_string())?;
            let dev_name = device
                .name()
                .unwrap_or_else(|_| "Default Output Device".to_string());
            let supported = device
                .default_output_config()
                .map_err(|e| format!("Audio output config error: {e}"))?;
            let sample_format = supported.sample_format();
            let cfg: cpal::StreamConfig = supported.into();
            let dev_sr = cfg.sample_rate.0.max(8_000);
            let channels = cfg.channels.max(1) as usize;

            let buf = prepare_playback_buffer(rec, mode, dev_sr);
            let total = buf.samples.len();
            if total == 0 {
                return Err("Audible playback buffer is empty".to_string());
            }

            let safe_frac = if start_fraction.is_finite() {
                start_fraction.clamp(0.0, 0.99)
            } else {
                0.0
            };
            let start_idx = ((safe_frac * total as f64).round() as usize)
                .min(total.saturating_sub(1));

            self.shared.cursor_sample.store(start_idx, Ordering::Relaxed);
            self.shared.total_samples.store(total, Ordering::Relaxed);
            self.shared
                .source_duration_bits
                .store(buf.source_duration_s.to_bits(), Ordering::Relaxed);
            self.shared.is_paused.store(false, Ordering::Relaxed);
            self.shared.finished.store(false, Ordering::Relaxed);
            if let Ok(mut g) = self.shared.last_error.lock() {
                *g = None;
            }

            let samples_cb = Arc::clone(&buf.samples);
            let shared_cb = Arc::clone(&self.shared);
            let shared_err = Arc::clone(&self.shared);
            let err_fn = move |err: cpal::StreamError| {
                if let Ok(mut g) = shared_err.last_error.lock() {
                    *g = Some(err.to_string());
                }
            };

            let stream = match sample_format {
                cpal::SampleFormat::F32 => device
                    .build_output_stream(
                        &cfg,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            fill_output_slice_f32(
                                data,
                                channels,
                                &samples_cb,
                                &shared_cb.cursor_sample,
                                &shared_cb.is_paused,
                                &shared_cb.finished,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building F32 output stream: {e}"))?,
                cpal::SampleFormat::I16 => device
                    .build_output_stream(
                        &cfg,
                        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            fill_output_slice_i16(
                                data,
                                channels,
                                &samples_cb,
                                &shared_cb.cursor_sample,
                                &shared_cb.is_paused,
                                &shared_cb.finished,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building I16 output stream: {e}"))?,
                cpal::SampleFormat::U16 => device
                    .build_output_stream(
                        &cfg,
                        move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                            fill_output_slice_u16(
                                data,
                                channels,
                                &samples_cb,
                                &shared_cb.cursor_sample,
                                &shared_cb.is_paused,
                                &shared_cb.finished,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building U16 output stream: {e}"))?,
                other => {
                    return Err(format!("Unsupported output sample format: {other:?}"));
                }
            };

            stream
                .play()
                .map_err(|e| format!("Failed starting audio playback stream: {e}"))?;

            self.shared.is_playing.store(true, Ordering::Relaxed);
            self.stream = Some(stream);
            self.active_mode = Some(mode);

            let audible_dur_s = total as f64 / dev_sr as f64;
            Ok(format!(
                "Playing {} via {} @ {} Hz ({:.2} s audible)",
                mode.label(),
                dev_name,
                dev_sr,
                audible_dur_s
            ))
        }
        #[cfg(not(feature = "hardware-mic"))]
        {
            let _ = (mode, start_fraction);
            Err("Compiled without 'hardware-mic' feature; audio output unavailable".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_presets::{synthesize_preset, BuiltinPreset};

    #[test]
    fn prepare_playback_buffer_resamples_and_maps_cursor() {
        let rec = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
        let src_dur = rec.duration_s();

        let buf_te10 = prepare_playback_buffer(&rec, AuditionMode::TimeExpansion10x, 44_100);
        assert_eq!(buf_te10.sample_rate_hz, 44_100);
        assert!((buf_te10.source_duration_s - src_dur).abs() < 1e-9);
        assert!(buf_te10.samples.len() > 10_000);

        let buf_het = prepare_playback_buffer(
            &rec,
            AuditionMode::Heterodyne { carrier_khz: 41.0 },
            48_000,
        );
        assert_eq!(buf_het.sample_rate_hz, 48_000);
        assert!(buf_het.samples.len() > 1_000);

        let mid = buf_te10.samples.len() / 2;
        let t_mid = cursor_to_source_time_s(mid, buf_te10.samples.len(), buf_te10.source_duration_s);
        assert!((t_mid - 0.5 * src_dur).abs() < 0.01);
        assert_eq!(cursor_to_source_time_s(0, buf_te10.samples.len(), src_dur), 0.0);
        assert!((cursor_to_source_time_s(buf_te10.samples.len() + 50, buf_te10.samples.len(), src_dur) - src_dur).abs() < 1e-9);
    }

    #[test]
    fn fill_output_slices_advance_cursor_respect_pause_and_finish() {
        let src = vec![0.25_f32, -0.5, 0.75, -1.0];
        let cursor = AtomicUsize::new(0);
        let paused = AtomicBool::new(true);
        let finished = AtomicBool::new(false);

        let mut out_f32 = [1.0_f32; 4];
        fill_output_slice_f32(&mut out_f32, 2, &src, &cursor, &paused, &finished);
        assert_eq!(out_f32, [0.0; 4]);
        assert_eq!(cursor.load(Ordering::Relaxed), 0);
        assert!(!finished.load(Ordering::Relaxed));

        paused.store(false, Ordering::Relaxed);
        let mut out_part = [0.0_f32; 4]; // 2 stereo frames
        fill_output_slice_f32(&mut out_part, 2, &src, &cursor, &paused, &finished);
        assert_eq!(out_part, [0.25, 0.25, -0.5, -0.5]);
        assert_eq!(cursor.load(Ordering::Relaxed), 2);
        assert!(!finished.load(Ordering::Relaxed));

        let mut out_end = [0.0_f32; 6]; // 3 stereo frames -> finishes after 2
        fill_output_slice_f32(&mut out_end, 2, &src, &cursor, &paused, &finished);
        assert_eq!(out_end, [0.75, 0.75, -1.0, -1.0, 0.0, 0.0]);
        assert_eq!(cursor.load(Ordering::Relaxed), 4);
        assert!(finished.load(Ordering::Relaxed));

        let c2 = AtomicUsize::new(0);
        let p2 = AtomicBool::new(false);
        let f2 = AtomicBool::new(false);
        let mut out_i16 = [0_i16; 4];
        fill_output_slice_i16(&mut out_i16, 2, &[0.5, -0.5], &c2, &p2, &f2);
        assert!(out_i16[0] > 16000 && out_i16[1] == out_i16[0]);
        assert!(out_i16[2] < -16000 && out_i16[3] == out_i16[2]);
        assert!(f2.load(Ordering::Relaxed));
    }

    #[test]
    fn audio_player_headless_graceful_no_panic() {
        let rec = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
        let mut player = UltrasonicAudioPlayer::new();
        assert!(!player.is_playing());
        match player.play_recording(&rec, AuditionMode::TimeExpansion10x, 0.0) {
            Ok(msg) => {
                assert!(!msg.is_empty());
                assert!(player.is_playing());
                player.pause_or_resume();
                assert!(player.is_paused());
                player.stop();
                assert!(!player.is_playing());
            }
            Err(err) => {
                assert!(!err.is_empty());
                assert!(!player.is_playing());
            }
        }
    }

    fn make_sine_rec(sr: u32, dur_s: f64, freq_hz: f64) -> BioAcousticRecording {
        let n = (sr as f64 * dur_s).round() as usize;
        let samples = (0..n)
            .map(|i| (TAU * freq_hz * (i as f64) / (sr as f64)).sin() as f32 * 0.5)
            .collect();
        BioAcousticRecording {
            source_name: format!("sine_{sr}Hz.wav"),
            sample_rate_hz: sr,
            raw_header_rate_hz: sr,
            channels: 1,
            encoding: SampleEncoding::PcmI16,
            samples,
            guano: None,
            riff_info: BTreeMap::new(),
        }
    }

    #[test]
    fn heterodyne_low_sample_rate_and_nan_carrier_no_panic_no_nan() {
        for (sr, carrier) in [(16_000_u32, 40.0_f64), (8_000_u32, 10.0_f64), (250_000_u32, f64::NAN)] {
            let rec = make_sine_rec(sr, 0.05, 1_500.0);
            let out = render_audible_recording(&rec, AuditionMode::Heterodyne { carrier_khz: carrier }, 48_000);
            assert!(!out.samples.is_empty());
            assert!(
                out.samples.iter().all(|s| s.is_finite()),
                "Heterodyne at sr={sr}, carrier={carrier} emitted non-finite samples"
            );
        }
    }

    #[test]
    fn cursor_nan_and_output_slice_nan_sanitized() {
        assert_eq!(cursor_to_source_time_s(10, 100, f64::NAN), 0.0);
        assert_eq!(cursor_to_source_time_s(10, 100, f64::INFINITY), 0.0);

        let src = [0.5_f32, -0.5, f32::NAN, 1.0];
        let cursor = AtomicUsize::new(0);
        let paused = AtomicBool::new(false);
        let finished = AtomicBool::new(false);
        let mut out = [99.0_f32; 8];
        fill_output_slice_f32(&mut out, 2, &src, &cursor, &paused, &finished);
        assert_eq!(out, [0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0]);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(finished.load(Ordering::Relaxed));
    }

    #[test]
    fn time_expansion_48khz_achieves_exact_10x_and_20x_ratios() {
        let rec_48 = make_sine_rec(48_000, 0.200, 3_000.0);
        let src_dur = rec_48.duration_s();
        for dev_sr in [44_100_u32, 48_000, 192_000] {
            let buf10 = prepare_playback_buffer(&rec_48, AuditionMode::TimeExpansion10x, dev_sr);
            let dur10 = buf10.samples.len() as f64 / buf10.sample_rate_hz as f64;
            assert!(
                (dur10 / src_dur - 10.0).abs() < 0.02,
                "48 kHz TE10x @ {dev_sr} Hz expected 10.0x, got {:.3}x",
                dur10 / src_dur
            );

            let buf20 = prepare_playback_buffer(&rec_48, AuditionMode::TimeExpansion20x, dev_sr);
            let dur20 = buf20.samples.len() as f64 / buf20.sample_rate_hz as f64;
            assert!(
                (dur20 / src_dur - 20.0).abs() < 0.02,
                "48 kHz TE20x @ {dev_sr} Hz expected 20.0x, got {:.3}x",
                dur20 / src_dur
            );
        }

        // Verify no regression at 250 kHz and 384 kHz
        for sr in [250_000_u32, 384_000] {
            let rec = make_sine_rec(sr, 0.100, 40_000.0);
            let buf10 = prepare_playback_buffer(&rec, AuditionMode::TimeExpansion10x, 48_000);
            let dur10 = buf10.samples.len() as f64 / buf10.sample_rate_hz as f64;
            assert!((dur10 / rec.duration_s() - 10.0).abs() < 0.02);
        }
    }

    #[test]
    fn frequency_division_low_sample_rate_and_nan_sanitized() {
        for sr in [0_u32, 8_u32, 500_u32] {
            let mut rec = make_sine_rec(sr.max(1_000), 0.02, 200.0);
            rec.sample_rate_hz = sr;
            rec.raw_header_rate_hz = sr;
            let out = render_audible_recording(
                &rec,
                AuditionMode::FrequencyDivision { ratio: 10 },
                48_000,
            );
            assert!(!out.samples.is_empty());
            assert!(
                out.samples.iter().all(|s| s.is_finite()),
                "FrequencyDivision at sr={sr} emitted non-finite samples"
            );
        }

        let mut rec_nan = make_sine_rec(250_000, 0.02, 40_000.0);
        rec_nan.samples[10] = f32::NAN;
        rec_nan.samples[100] = f32::INFINITY;
        rec_nan.samples[250] = f32::NEG_INFINITY;
        let out_nan = render_audible_recording(
            &rec_nan,
            AuditionMode::FrequencyDivision { ratio: 10 },
            48_000,
        );
        assert!(!out_nan.samples.is_empty());
        assert!(out_nan.samples.iter().all(|s| s.is_finite()));
        assert!(
            out_nan.samples.iter().any(|&s| s.abs() > 1e-4),
            "Single NaN/Inf sample must not zero out the entire FrequencyDivision chunk"
        );
    }
}
