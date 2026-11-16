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

//! Real-time Live Microphone & Ultrasonic UAC2 Stream Capture (`cpal`) + Simulated Live
//! Bioacoustic Stream Generator backed by a thread-safe rolling ring buffer.
//!
//! Supports both standard `44.1/48 kHz` avian field microphones and high-rate `192-384 kHz`
//! USB UAC2 ultrasonic detectors (e.g., AudioMoth USB Microphone firmware, Pettersson M500-384,
//! Dodotronic Ultramic), with 1-click **"Freeze & Promote Buffer"** into a full `BioAcousticRecording`.

use crate::wav_guano::{BioAcousticRecording, GuanoMetadata, SampleEncoding};
use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct RollingAudioBuffer {
    pub sample_rate_hz: u32,
    pub capacity_samples: usize,
    pub device_label: String,
    pub samples: VecDeque<f32>,
    pub total_pushed: u64,
    pub last_error: Option<String>,
}

impl RollingAudioBuffer {
    pub fn new(sample_rate_hz: u32, window_seconds: f64, device_label: String) -> Self {
        let cap = ((sample_rate_hz as f64 * window_seconds.clamp(0.5, 30.0)) as usize).max(4096);
        Self {
            sample_rate_hz,
            capacity_samples: cap,
            device_label,
            samples: VecDeque::with_capacity(cap),
            total_pushed: 0,
            last_error: None,
        }
    }

    pub fn push_mono_slice(&mut self, incoming: &[f32]) {
        for &s in incoming {
            if self.samples.len() >= self.capacity_samples {
                self.samples.pop_front();
            }
            self.samples.push_back(s.clamp(-1.0, 1.0));
        }
        self.total_pushed += incoming.len() as u64;
    }

    pub fn snapshot_vec(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }
}

pub enum ActiveMicBackend {
    #[cfg(feature = "hardware-mic")]
    HardwareCpal(cpal::Stream),
    SimulatedField {
        phase_cursor: f64,
        wideband_192k: bool,
    },
}

pub struct LiveMicEngine {
    pub ring: Arc<Mutex<RollingAudioBuffer>>,
    backend: Option<ActiveMicBackend>,
}

impl Default for LiveMicEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveMicEngine {
    pub fn new() -> Self {
        let ring = Arc::new(Mutex::new(RollingAudioBuffer::new(
            48_000,
            2.5,
            "Idle".to_string(),
        )));
        Self {
            ring,
            backend: None,
        }
    }

    pub fn is_capturing(&self) -> bool {
        self.backend.is_some()
    }

    /// List available hardware input microphone names when compiled with `hardware-mic`.
    pub fn list_input_devices() -> Vec<String> {
        #[cfg(feature = "hardware-mic")]
        {
            use cpal::traits::{DeviceTrait, HostTrait};
            let host = cpal::default_host();
            if let Ok(devs) = host.input_devices() {
                return devs
                    .filter_map(|d| d.name().ok())
                    .collect();
            }
            Vec::new()
        }
        #[cfg(not(feature = "hardware-mic"))]
        {
            Vec::new()
        }
    }

    /// Start hardware microphone capture via `cpal` (preferring highest supported sample rate
    /// when `prefer_ultrasonic` is enabled for USB UAC2 bat mics).
    pub fn start_hardware_capture(
        &mut self,
        window_seconds: f64,
        prefer_ultrasonic: bool,
    ) -> Result<String, String> {
        self.stop();
        #[cfg(feature = "hardware-mic")]
        {
            use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .ok_or_else(|| "No hardware audio input microphone detected".to_string())?;
            let dev_name = device
                .name()
                .unwrap_or_else(|_| "Default Input Microphone".to_string());

            let supported = if prefer_ultrasonic {
                pick_highest_rate_config(&device)
                    .or_else(|| device.default_input_config().ok())
                    .ok_or_else(|| "Unable to query microphone input configuration".to_string())?
            } else {
                device
                    .default_input_config()
                    .map_err(|e| format!("Microphone config error: {e}"))?
            };

            let sample_format = supported.sample_format();
            let cfg: cpal::StreamConfig = supported.into();
            let sr = cfg.sample_rate.0;
            let channels = cfg.channels.max(1) as usize;

            let ring = Arc::new(Mutex::new(RollingAudioBuffer::new(
                sr,
                window_seconds,
                dev_name.clone(),
            )));
            let ring_cb = Arc::clone(&ring);
            let ring_err = Arc::clone(&ring);

            let err_fn = move |err: cpal::StreamError| {
                if let Ok(mut guard) = ring_err.lock() {
                    guard.last_error = Some(err.to_string());
                }
            };

            let stream = match sample_format {
                cpal::SampleFormat::F32 => device
                    .build_input_stream(
                        &cfg,
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            let mut mono = Vec::with_capacity(data.len() / channels + 1);
                            for frame in data.chunks(channels) {
                                let avg = frame.iter().copied().sum::<f32>() / frame.len() as f32;
                                mono.push(avg);
                            }
                            if let Ok(mut guard) = ring_cb.lock() {
                                guard.push_mono_slice(&mono);
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building F32 input stream: {e}"))?,
                cpal::SampleFormat::I16 => device
                    .build_input_stream(
                        &cfg,
                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                            let mut mono = Vec::with_capacity(data.len() / channels + 1);
                            for frame in data.chunks(channels) {
                                let avg = frame
                                    .iter()
                                    .map(|&s| s as f32 / 32768.0)
                                    .sum::<f32>()
                                    / frame.len() as f32;
                                mono.push(avg);
                            }
                            if let Ok(mut guard) = ring_cb.lock() {
                                guard.push_mono_slice(&mono);
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building I16 input stream: {e}"))?,
                cpal::SampleFormat::U16 => device
                    .build_input_stream(
                        &cfg,
                        move |data: &[u16], _: &cpal::InputCallbackInfo| {
                            let mut mono = Vec::with_capacity(data.len() / channels + 1);
                            for frame in data.chunks(channels) {
                                let avg = frame
                                    .iter()
                                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                    .sum::<f32>()
                                    / frame.len() as f32;
                                mono.push(avg);
                            }
                            if let Ok(mut guard) = ring_cb.lock() {
                                guard.push_mono_slice(&mono);
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed building U16 input stream: {e}"))?,
                other => {
                    return Err(format!("Unsupported microphone sample format: {other:?}"));
                }
            };

            stream
                .play()
                .map_err(|e| format!("Failed starting microphone stream: {e}"))?;

            self.ring = ring;
            self.backend = Some(ActiveMicBackend::HardwareCpal(stream));
            Ok(format!("{dev_name} @ {sr} Hz ({channels} ch)"))
        }
        #[cfg(not(feature = "hardware-mic"))]
        {
            let _ = (window_seconds, prefer_ultrasonic);
            Err("Compiled without 'hardware-mic' feature; use simulated live stream or recompile with default features".to_string())
        }
    }

    /// Start a live synthetic nocturnal field stream (`192 kHz` dual bat+bird or `48 kHz` bird stream)
    /// for testing real-time rolling spectrogram updates when no external detector is plugged in.
    pub fn start_simulated_stream(&mut self, window_seconds: f64, wideband_192k: bool) -> String {
        self.stop();
        let sr = if wideband_192k { 192_000 } else { 48_000 };
        let label = if wideband_192k {
            "Simulated Live 192 kHz Riparian Bat+Bird Mic".to_string()
        } else {
            "Simulated Live 48 kHz Avian NFC Mic".to_string()
        };
        let mut buf = RollingAudioBuffer::new(sr, window_seconds, label.clone());
        // Pre-seed 0.4s so initial spectrogram frame is immediately non-empty
        let init_chunk = generate_live_stream_chunk(0.0, (sr as usize) * 4 / 10, sr, wideband_192k);
        buf.push_mono_slice(&init_chunk);
        self.ring = Arc::new(Mutex::new(buf));
        self.backend = Some(ActiveMicBackend::SimulatedField {
            phase_cursor: 0.4,
            wideband_192k,
        });
        label
    }

    /// Pump simulated samples when running in `SimulatedField` mode (`dt_s` elapsed wall time).
    pub fn pump_if_simulated(&mut self, dt_s: f64) {
        if let Some(ActiveMicBackend::SimulatedField {
            ref mut phase_cursor,
            wideband_192k,
        }) = self.backend
        {
            let step_s = dt_s.clamp(0.01, 0.12);
            let sr = if wideband_192k { 192_000 } else { 48_000 };
            let n_samp = (step_s * sr as f64) as usize;
            let chunk = generate_live_stream_chunk(*phase_cursor, n_samp, sr, wideband_192k);
            *phase_cursor += step_s;
            if let Ok(mut guard) = self.ring.lock() {
                guard.push_mono_slice(&chunk);
            }
        }
    }

    pub fn stop(&mut self) {
        self.backend = None;
    }

    /// Snapshot the current rolling buffer into a full `BioAcousticRecording` with GUANO v1.0 metadata.
    pub fn freeze_to_recording(&self) -> Option<BioAcousticRecording> {
        let guard = self.ring.lock().ok()?;
        let samples = guard.snapshot_vec();
        if samples.len() < 512 {
            return None;
        }
        let mut g = GuanoMetadata::new_v1();
        g.samplerate_hz = Some(guard.sample_rate_hz);
        g.length_s = Some(samples.len() as f64 / guard.sample_rate_hz as f64);
        g.make = Some("SonoWing Live Capture".to_string());
        g.model = Some(guard.device_label.clone());
        g.note = Some("Captured live rolling microphone spectrogram buffer".to_string());

        Some(BioAcousticRecording {
            source_name: format!(
                "LiveMic_{}kHz_{}ms.wav",
                guard.sample_rate_hz / 1000,
                (samples.len() * 1000) / guard.sample_rate_hz.max(1) as usize
            ),
            sample_rate_hz: guard.sample_rate_hz,
            raw_header_rate_hz: guard.sample_rate_hz,
            channels: 1,
            encoding: SampleEncoding::IeeeF32,
            samples,
            guano: Some(g),
            riff_info: BTreeMap::new(),
        })
    }
}

#[cfg(feature = "hardware-mic")]
fn pick_highest_rate_config(device: &cpal::Device) -> Option<cpal::SupportedStreamConfig> {
    use cpal::traits::DeviceTrait;
    let ranges = device.supported_input_configs().ok()?;
    let mut best: Option<cpal::SupportedStreamConfig> = None;
    let mut best_sr = 0u32;
    for r in ranges {
        let max_sr = r.max_sample_rate().0.min(384_000);
        if max_sr > best_sr {
            best_sr = max_sr;
            best = Some(r.with_sample_rate(cpal::SampleRate(max_sr)));
        }
    }
    best
}

fn generate_live_stream_chunk(
    t_start_s: f64,
    n_samp: usize,
    sr: u32,
    wideband_192k: bool,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_samp);
    let mut seed = ((t_start_s * 100_000.0) as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);

    for k in 0..n_samp {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let noise = (((seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0) * 0.010;
        let t = t_start_s + (k as f64) / (sr as f64);
        let mut sig = noise;

        // Periodic Swainson's Thrush upward flight call every 0.75s
        let bird_phase = t % 0.75;
        if bird_phase < 0.095 {
            let u = bird_phase / 0.095;
            let f_inst = 2_750.0 + 1_350.0 * u.powf(1.25);
            let env = (PI * u).sin().powf(0.5);
            sig += 0.52 * env * (TAU * f_inst * bird_phase).sin();
            sig += 0.14 * env * (2.0 * TAU * f_inst * bird_phase).sin();
        }

        // Periodic ultrasonic bat pulses every 0.14s when running at 192 kHz
        if wideband_192k {
            let bat_phase = (t + 0.04) % 0.14;
            if bat_phase < 0.0065 {
                let u = bat_phase / 0.0065;
                let f_khz = 40.8 + 31.0 * (-4.5 * u).exp();
                let env = (PI * u).sin().powf(0.35);
                sig += 0.65 * env * (TAU * (f_khz * 1000.0) * bat_phase).sin();
            }
        }
        out.push(sig.clamp(-0.98, 0.98) as f32);
    }
    out
}
