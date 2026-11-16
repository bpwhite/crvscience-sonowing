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

//! Pure-Rust RIFF/WAV (8/16/24/32-bit PCM & 32-bit IEEE float, 8 kHz to 500 kHz),
//! GUANO v1.0 (`guan` subchunk) metadata reader/writer, AudioMoth `LIST/INFO`
//! parser, and Titley Anabat Zero-Crossing (`.zc` / `.zcf`) sequence handler.
//!
//! Specification reference:
//! - GUANO v1.0: <https://github.com/riggsd/guano-spec/blob/master/guano_specification.md>

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleEncoding {
    PcmU8,
    PcmI16,
    PcmI24,
    PcmI32,
    IeeeF32,
    SyntheticZeroCross,
}

impl SampleEncoding {
    pub fn label(self) -> &'static str {
        match self {
            Self::PcmU8 => "8-bit Unsigned PCM",
            Self::PcmI16 => "16-bit Signed PCM",
            Self::PcmI24 => "24-bit Signed PCM",
            Self::PcmI32 => "32-bit Signed PCM",
            Self::IeeeF32 => "32-bit IEEE Float",
            Self::SyntheticZeroCross => "Zero-Crossing Reconstructed",
        }
    }
}

/// Extensible GUANO v1.0 metadata dictionary (`GUANO|Version: 1.0` first).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuanoMetadata {
    pub version: String,
    pub timestamp: Option<String>,
    pub samplerate_hz: Option<u32>,
    pub length_s: Option<f64>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub firmware_version: Option<String>,
    pub loc_position: Option<(f64, f64)>,
    pub loc_elevation_m: Option<f64>,
    pub temperature_ext_c: Option<f64>,
    pub humidity_pct: Option<f64>,
    pub time_expansion: Option<u32>,
    pub species_auto_id: Vec<String>,
    pub species_manual_id: Vec<String>,
    pub note: Option<String>,
    pub extra_fields: BTreeMap<String, String>,
}

impl GuanoMetadata {
    pub fn new_v1() -> Self {
        Self {
            version: "1.0".to_string(),
            ..Default::default()
        }
    }

    /// Parse UTF-8 GUANO text block per <https://github.com/riggsd/guano-spec>.
    pub fn parse_text(raw: &str) -> Self {
        let mut meta = Self {
            version: "1.0".to_string(),
            ..Default::default()
        };
        for raw_line in raw.lines() {
            let line = raw_line.trim_matches(|c: char| c.is_whitespace() || c == '\0');
            if line.is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once(':') else {
                continue;
            };
            let key = k.trim();
            let val = v.trim();
            if key.is_empty() {
                continue;
            }
            match key {
                "GUANO|Version" => meta.version = val.to_string(),
                "Timestamp" => meta.timestamp = Some(val.to_string()),
                "Samplerate" => meta.samplerate_hz = val.parse::<f64>().ok().map(|x| x as u32),
                "Length" => meta.length_s = val.parse::<f64>().ok(),
                "Make" => meta.make = Some(val.to_string()),
                "Model" => meta.model = Some(val.to_string()),
                "Firmware Version" => meta.firmware_version = Some(val.to_string()),
                "Loc Position" => {
                    let parts: Vec<&str> = val.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let (Ok(lat), Ok(lon)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                        {
                            meta.loc_position = Some((lat, lon));
                        }
                    }
                }
                "Loc Elevation" => meta.loc_elevation_m = val.parse::<f64>().ok(),
                "Temperature Ext" => meta.temperature_ext_c = val.parse::<f64>().ok(),
                "Humidity" => meta.humidity_pct = val.parse::<f64>().ok(),
                "TE" => meta.time_expansion = val.parse::<u32>().ok(),
                "Species Auto ID" => {
                    meta.species_auto_id = val
                        .split([',', ';'])
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "Species Manual ID" => {
                    meta.species_manual_id = val
                        .split([',', ';'])
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "Note" => meta.note = Some(val.to_string()),
                other => {
                    meta.extra_fields.insert(other.to_string(), val.to_string());
                }
            }
        }
        meta
    }

    /// Serialize GUANO metadata to a compliant UTF-8 text block (`GUANO|Version` first).
    pub fn to_guano_text(&self) -> String {
        let mut out = String::new();
        let ver = if self.version.is_empty() {
            "1.0"
        } else {
            &self.version
        };
        out.push_str(&format!("GUANO|Version: {ver}\n"));
        if let Some(ts) = &self.timestamp {
            out.push_str(&format!("Timestamp: {ts}\n"));
        }
        if let Some(sr) = self.samplerate_hz {
            out.push_str(&format!("Samplerate: {sr}\n"));
        }
        if let Some(len) = self.length_s {
            out.push_str(&format!("Length: {len:.4}\n"));
        }
        if let Some(make) = &self.make {
            out.push_str(&format!("Make: {make}\n"));
        }
        if let Some(model) = &self.model {
            out.push_str(&format!("Model: {model}\n"));
        }
        if let Some(fw) = &self.firmware_version {
            out.push_str(&format!("Firmware Version: {fw}\n"));
        }
        if let Some((lat, lon)) = self.loc_position {
            out.push_str(&format!("Loc Position: {lat:.6} {lon:.6}\n"));
        }
        if let Some(elev) = self.loc_elevation_m {
            out.push_str(&format!("Loc Elevation: {elev:.1}\n"));
        }
        if let Some(temp) = self.temperature_ext_c {
            out.push_str(&format!("Temperature Ext: {temp:.1}\n"));
        }
        if let Some(rh) = self.humidity_pct {
            out.push_str(&format!("Humidity: {rh:.1}\n"));
        }
        if let Some(te) = self.time_expansion {
            out.push_str(&format!("TE: {te}\n"));
        }
        if !self.species_auto_id.is_empty() {
            out.push_str(&format!(
                "Species Auto ID: {}\n",
                self.species_auto_id.join(", ")
            ));
        }
        if !self.species_manual_id.is_empty() {
            out.push_str(&format!(
                "Species Manual ID: {}\n",
                self.species_manual_id.join(", ")
            ));
        }
        if let Some(note) = &self.note {
            let clean = note.replace('\n', "\\n");
            out.push_str(&format!("Note: {clean}\n"));
        }
        for (k, v) in &self.extra_fields {
            if !k.contains(':') {
                out.push_str(&format!("{k}: {v}\n"));
            }
        }
        out
    }
}

/// Decoded full-spectrum bioacoustic recording ready for DSP analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioAcousticRecording {
    pub source_name: String,
    /// Effective real-world acoustic sample rate in Hz (adjusted by `TE` if time-expanded).
    pub sample_rate_hz: u32,
    /// Raw file sample rate stored in WAV header before `TE` scaling.
    pub raw_header_rate_hz: u32,
    pub channels: u16,
    pub encoding: SampleEncoding,
    /// Normalized mono waveform samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    pub guano: Option<GuanoMetadata>,
    pub riff_info: BTreeMap<String, String>,
}

impl BioAcousticRecording {
    pub fn duration_s(&self) -> f64 {
        if self.sample_rate_hz == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate_hz as f64
        }
    }

    pub fn nyquist_hz(&self) -> f64 {
        self.sample_rate_hz as f64 * 0.5
    }

    pub fn peak_dbfs(&self) -> f64 {
        let peak = self
            .samples
            .iter()
            .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        if peak <= 1e-12 {
            -120.0
        } else {
            20.0 * (peak as f64).log10()
        }
    }

    pub fn rms_dbfs(&self) -> f64 {
        if self.samples.is_empty() {
            return -120.0;
        }
        let sum_sq: f64 = self.samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let rms = (sum_sq / self.samples.len() as f64).sqrt();
        if rms <= 1e-12 {
            -120.0
        } else {
            20.0 * rms.log10()
        }
    }
}

/// Zero-crossing dot event (`time_s`, `freq_hz`, `period_us`) compatible with Anabat ZC displays.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZeroCrossDot {
    pub time_s: f64,
    pub freq_hz: f64,
    pub slope_khz_per_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCrossSequence {
    pub source_name: String,
    pub division_ratio: u8,
    pub guano: Option<GuanoMetadata>,
    pub dots: Vec<ZeroCrossDot>,
}

/// Read a RIFF/WAVE file from disk or byte slice, extracting PCM/IEEE-Float samples,
/// `guan` GUANO v1.0 metadata subchunks, and `LIST/INFO` AudioMoth comments.
pub fn load_wav_file(path: &Path) -> Result<BioAcousticRecording, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Failed to read WAV file '{}': {e}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("recording.wav")
        .to_string();
    parse_wav_bytes(&bytes, &name)
}

pub fn parse_wav_bytes(bytes: &[u8], source_name: &str) -> Result<BioAcousticRecording, String> {
    if bytes.len() < 44 {
        return Err("File is shorter than minimum 44-byte RIFF/WAVE header".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Missing RIFF/WAVE magic header bytes".to_string());
    }

    let mut pos = 12usize;
    let mut fmt_code: u16 = 0;
    let mut channels: u16 = 0;
    let mut raw_rate_hz: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_slice: Option<&[u8]> = None;
    let mut guano: Option<GuanoMetadata> = None;
    let mut riff_info: BTreeMap<String, String> = BTreeMap::new();

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start.saturating_add(chunk_size).min(bytes.len());
        let payload = &bytes[payload_start..payload_end];

        match chunk_id {
            b"fmt " => {
                if payload.len() < 16 {
                    return Err("Malformed WAV fmt chunk (< 16 bytes)".to_string());
                }
                fmt_code = u16::from_le_bytes([payload[0], payload[1]]);
                channels = u16::from_le_bytes([payload[2], payload[3]]);
                raw_rate_hz =
                    u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
                bits_per_sample = u16::from_le_bytes([payload[14], payload[15]]);
                // WAVE_FORMAT_EXTENSIBLE (0xFFFE): read subformat code at bytes 24..26
                if fmt_code == 0xFFFE && payload.len() >= 26 {
                    fmt_code = u16::from_le_bytes([payload[24], payload[25]]);
                }
            }
            b"data" => {
                data_slice = Some(payload);
            }
            b"guan" => {
                if let Ok(text) = std::str::from_utf8(payload) {
                    guano = Some(GuanoMetadata::parse_text(text));
                } else {
                    let lossy = String::from_utf8_lossy(payload);
                    guano = Some(GuanoMetadata::parse_text(&lossy));
                }
            }
            b"LIST" => {
                if payload.len() >= 4 && &payload[0..4] == b"INFO" {
                    parse_riff_info_subchunks(&payload[4..], &mut riff_info);
                }
            }
            _ => {}
        }

        // RIFF chunks are padded to even byte boundaries
        let padded = if chunk_size % 2 == 1 {
            chunk_size.saturating_add(1)
        } else {
            chunk_size
        };
        pos = payload_start.saturating_add(padded);
    }

    if channels == 0 || raw_rate_hz == 0 {
        return Err("WAV fmt chunk missing valid channel count or sample rate".to_string());
    }
    let raw_data = data_slice.ok_or_else(|| "WAV file contains no 'data' chunk".to_string())?;

    let (samples, encoding) =
        decode_pcm_to_mono(raw_data, fmt_code, channels as usize, bits_per_sample)?;

    // Synthesize AudioMoth metadata into GUANO if no explicit `guan` chunk was present
    if guano.is_none() && (riff_info.contains_key("IART") || riff_info.contains_key("ICMT")) {
        let mut am_guano = GuanoMetadata::new_v1();
        if let Some(artist) = riff_info.get("IART") {
            if artist.contains("AudioMoth") {
                am_guano.make = Some("Open Acoustic Devices".to_string());
                am_guano.model = Some("AudioMoth".to_string());
            }
        }
        if let Some(comment) = riff_info.get("ICMT") {
            am_guano.note = Some(comment.clone());
        }
        am_guano.samplerate_hz = Some(raw_rate_hz);
        guano = Some(am_guano);
    }

    // Apply Time-Expansion (TE) multiplier if file was saved as 10x slowed WAV
    let te_factor = guano
        .as_ref()
        .and_then(|g| g.time_expansion)
        .unwrap_or(1)
        .max(1);
    let effective_rate = if raw_rate_hz <= 50_000 && te_factor > 1 {
        raw_rate_hz.saturating_mul(te_factor)
    } else {
        guano
            .as_ref()
            .and_then(|g| g.samplerate_hz)
            .filter(|&sr| sr >= raw_rate_hz)
            .unwrap_or(raw_rate_hz)
    };

    Ok(BioAcousticRecording {
        source_name: source_name.to_string(),
        sample_rate_hz: effective_rate,
        raw_header_rate_hz: raw_rate_hz,
        channels,
        encoding,
        samples,
        guano,
        riff_info,
    })
}

fn parse_riff_info_subchunks(buf: &[u8], out: &mut BTreeMap<String, String>) {
    let mut idx = 0usize;
    while idx + 8 <= buf.len() {
        let tag = String::from_utf8_lossy(&buf[idx..idx + 4]).to_string();
        let sz =
            u32::from_le_bytes([buf[idx + 4], buf[idx + 5], buf[idx + 6], buf[idx + 7]]) as usize;
        let start = idx + 8;
        let end = start.saturating_add(sz).min(buf.len());
        let val = String::from_utf8_lossy(&buf[start..end])
            .trim_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_string();
        if !val.is_empty() {
            out.insert(tag, val);
        }
        let step = if sz % 2 == 1 { sz + 1 } else { sz };
        idx = start.saturating_add(step);
    }
}

fn decode_pcm_to_mono(
    data: &[u8],
    fmt_code: u16,
    channels: usize,
    bits_per_sample: u16,
) -> Result<(Vec<f32>, SampleEncoding), String> {
    let bytes_per_sample = (bits_per_sample as usize) / 8;
    if bytes_per_sample == 0 {
        return Err(format!("Unsupported bits_per_sample: {bits_per_sample}"));
    }
    let frame_bytes = bytes_per_sample * channels;
    let frame_count = data.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frame_count);

    match (fmt_code, bits_per_sample) {
        (1, 8) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0_f32;
                for ch in 0..channels {
                    let u = data[base + ch] as f32;
                    acc += (u - 128.0) / 128.0;
                }
                mono.push((acc / channels as f32).clamp(-1.0, 1.0));
            }
            Ok((mono, SampleEncoding::PcmU8))
        }
        (1, 16) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0_f32;
                for ch in 0..channels {
                    let o = base + ch * 2;
                    let s = i16::from_le_bytes([data[o], data[o + 1]]) as f32;
                    acc += s / 32768.0;
                }
                mono.push((acc / channels as f32).clamp(-1.0, 1.0));
            }
            Ok((mono, SampleEncoding::PcmI16))
        }
        (1, 24) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0_f32;
                for ch in 0..channels {
                    let o = base + ch * 3;
                    let sign_byte = if data[o + 2] & 0x80 != 0 { 0xFF } else { 0x00 };
                    let s = i32::from_le_bytes([data[o], data[o + 1], data[o + 2], sign_byte]);
                    acc += s as f32 / 8_388_608.0;
                }
                mono.push((acc / channels as f32).clamp(-1.0, 1.0));
            }
            Ok((mono, SampleEncoding::PcmI24))
        }
        (1, 32) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0_f32;
                for ch in 0..channels {
                    let o = base + ch * 4;
                    let s = i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    acc += (s as f64 / 2_147_483_648.0) as f32;
                }
                mono.push((acc / channels as f32).clamp(-1.0, 1.0));
            }
            Ok((mono, SampleEncoding::PcmI32))
        }
        (3, 32) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0_f32;
                for ch in 0..channels {
                    let o = base + ch * 4;
                    let s = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    if s.is_finite() {
                        acc += s;
                    }
                }
                mono.push((acc / channels as f32).clamp(-1.0, 1.0));
            }
            Ok((mono, SampleEncoding::IeeeF32))
        }
        _ => Err(format!(
            "Unsupported WAV format code {fmt_code} with {bits_per_sample} bits/sample"
        )),
    }
}

/// Write a 16-bit mono RIFF/WAVE file with an optional embedded `guan` GUANO v1.0 chunk.
pub fn write_wav_with_guano(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    guano: Option<&GuanoMetadata>,
) -> io::Result<()> {
    let bytes = encode_wav_with_guano_bytes(samples, sample_rate_hz, guano);
    fs::write(path, bytes)
}

pub fn encode_wav_with_guano_bytes(
    samples: &[f32],
    sample_rate_hz: u32,
    guano: Option<&GuanoMetadata>,
) -> Vec<u8> {
    let data_bytes_len = samples.len() * 2;
    let guano_bytes = guano.map(|g| g.to_guano_text().into_bytes());
    let guano_chunk_total = guano_bytes
        .as_ref()
        .map(|b| 8 + b.len() + (b.len() % 2))
        .unwrap_or(0);

    let riff_payload_len = 4 + (8 + 16) + (8 + data_bytes_len) + guano_chunk_total;
    let mut out = Vec::with_capacity(8 + riff_payload_len);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_payload_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk (16-bit mono PCM)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // 1 channel
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    let byte_rate = sample_rate_hz.saturating_mul(2);
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes_len as u32).to_le_bytes());
    for &x in samples {
        let clamped = x.clamp(-1.0, 1.0);
        let q = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }

    // optional `guan` subchunk
    if let Some(gb) = guano_bytes {
        out.extend_from_slice(b"guan");
        out.extend_from_slice(&(gb.len() as u32).to_le_bytes());
        out.extend_from_slice(&gb);
        if gb.len() % 2 == 1 {
            out.push(0);
        }
    }

    out
}

/// Export a clean text/TSV Zero-Crossing (`ZC`) dot sequence or parse a `.zc` dot table,
/// and also reconstruct a phase-continuous synthetic full-spectrum waveform from ZC dots
/// when opening legacy zero-crossing files.
pub fn write_zc_dot_table(path: &Path, seq: &ZeroCrossSequence) -> io::Result<()> {
    let mut text = String::new();
    text.push_str("# SonoWing Zero-Crossing Sequence v1.0\n");
    text.push_str(&format!("# Source: {}\n", seq.source_name));
    text.push_str(&format!("# DivisionRatio: {}\n", seq.division_ratio));
    if let Some(g) = &seq.guano {
        for line in g.to_guano_text().lines() {
            text.push_str(&format!("# GUANO {line}\n"));
        }
    }
    text.push_str("time_s\tfreq_hz\tslope_khz_per_ms\n");
    for d in &seq.dots {
        text.push_str(&format!(
            "{:.6}\t{:.1}\t{:.4}\n",
            d.time_s, d.freq_hz, d.slope_khz_per_ms
        ));
    }
    fs::write(path, text)
}

pub fn load_zc_dot_table(path: &Path) -> Result<ZeroCrossSequence, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("Unable to read zero-crossing file '{}': {e}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("sequence.zc")
        .to_string();
    let mut div_ratio = 8u8;
    let mut guano_lines = String::new();
    let mut dots = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# DivisionRatio:") {
            if let Ok(v) = rest.trim().parse::<u8>() {
                div_ratio = v.max(1);
            }
            continue;
        }
        if let Some(gline) = trimmed.strip_prefix("# GUANO ") {
            guano_lines.push_str(gline);
            guano_lines.push('\n');
            continue;
        }
        if trimmed.starts_with('#') || trimmed.starts_with("time_s") {
            continue;
        }
        let cols: Vec<&str> = trimmed.split(['\t', ',']).collect();
        if cols.len() >= 2 {
            if let (Ok(t), Ok(f)) = (cols[0].trim().parse::<f64>(), cols[1].trim().parse::<f64>()) {
                let slope = cols
                    .get(2)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .unwrap_or(0.0);
                dots.push(ZeroCrossDot {
                    time_s: t,
                    freq_hz: f,
                    slope_khz_per_ms: slope,
                });
            }
        }
    }

    let guano = if guano_lines.is_empty() {
        None
    } else {
        Some(GuanoMetadata::parse_text(&guano_lines))
    };

    Ok(ZeroCrossSequence {
        source_name: name,
        division_ratio: div_ratio,
        guano,
        dots,
    })
}

/// Reconstruct a `250 kHz` synthetic waveform from Zero-Crossing (`ZC`) dots so legacy ZC
/// sequences can be viewed and measured inside the unified waveform/spectrogram pipeline.
pub fn reconstruct_waveform_from_zc(seq: &ZeroCrossSequence, sample_rate_hz: u32) -> BioAcousticRecording {
    let max_t = seq
        .dots
        .last()
        .map(|d| d.time_s + 0.025)
        .unwrap_or(0.25)
        .clamp(0.05, 30.0);
    let total_samples = ((max_t * sample_rate_hz as f64) as usize).max(1024);
    let mut samples = vec![0.0_f32; total_samples];

    if seq.dots.len() >= 2 {
        let mut phase = 0.0_f64;
        for pair in seq.dots.windows(2) {
            let d0 = pair[0];
            let d1 = pair[1];
            let dt = d1.time_s - d0.time_s;
            if dt <= 0.0 || dt > 0.004 {
                phase = 0.0;
                continue;
            }
            let i0 = ((d0.time_s * sample_rate_hz as f64) as usize).min(total_samples);
            let i1 = ((d1.time_s * sample_rate_hz as f64) as usize).min(total_samples);
            let span = i1.saturating_sub(i0).max(1);
            for k in 0..span {
                let frac = k as f64 / span as f64;
                let f_inst = d0.freq_hz + frac * (d1.freq_hz - d0.freq_hz);
                phase += std::f64::consts::TAU * f_inst / sample_rate_hz as f64;
                samples[i0 + k] = (0.65 * phase.sin()) as f32;
            }
        }
    }

    BioAcousticRecording {
        source_name: seq.source_name.clone(),
        sample_rate_hz,
        raw_header_rate_hz: sample_rate_hz,
        channels: 1,
        encoding: SampleEncoding::SyntheticZeroCross,
        samples,
        guano: seq.guano.clone(),
        riff_info: BTreeMap::new(),
    }
}
