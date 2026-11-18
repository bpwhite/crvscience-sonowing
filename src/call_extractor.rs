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

//! Automated Bat Echolocation Pulse & Bird Vocalization / Nocturnal Flight Call (`NFC`)
//! detector and SonoBat / NABat / Raven Pro morphometric parameterizer (`v1.1.0` refined
//! band-maxima baseline + `-20 dB` ridge-gate contour tracking).

use crate::dsp::{compute_tkeo, Spectrogram};
use crate::taxa_library::{classify_call_event, TaxonMatch};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaxonDomain {
    BatChiroptera,
    BirdAves,
}

impl TaxonDomain {
    pub fn label(self) -> &'static str {
        match self {
            Self::BatChiroptera => "Chiroptera (Bat)",
            Self::BirdAves => "Aves (Bird)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMode {
    AutoWideband,
    BatUltrasonicOnly,
    BirdAudibleOnly,
}

impl DetectionMode {
    pub const ALL: [Self; 3] = [
        Self::AutoWideband,
        Self::BatUltrasonicOnly,
        Self::BirdAudibleOnly,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoWideband => "Auto Dual-Taxon (Bird 0.6-12 kHz + Bat 14-160 kHz)",
            Self::BatUltrasonicOnly => "Chiroptera Ultrasonic Only (14-160 kHz)",
            Self::BirdAudibleOnly => "Avian Audible & NFC Only (0.5-12 kHz)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallShapeTopology {
    SteepFm,
    FmQcf,
    FlatQcf,
    FmCfFm,
    AvianFlightCall,
    AvianSongWhistle,
    AvianHarmonicOrTrill,
}

impl CallShapeTopology {
    pub fn code(self) -> &'static str {
        match self {
            Self::SteepFm => "FM",
            Self::FmQcf => "FM-QCF",
            Self::FlatQcf => "QCF",
            Self::FmCfFm => "FM-CF-FM",
            Self::AvianFlightCall => "NFC-Sweep",
            Self::AvianSongWhistle => "Song-Whistle",
            Self::AvianHarmonicOrTrill => "Trill/Harmonic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BehavioralPhase {
    SearchPhase,
    ApproachPhase,
    TerminalFeedingBuzz,
    NocturnalFlightCall,
    TerritorialOrContactVocalization,
}

impl BehavioralPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::SearchPhase => "Search Phase",
            Self::ApproachPhase => "Approach Phase",
            Self::TerminalFeedingBuzz => "Terminal Feeding Buzz",
            Self::NocturnalFlightCall => "Nocturnal Flight Call (NFC)",
            Self::TerritorialOrContactVocalization => "Song / Territorial Call",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub mode: DetectionMode,
    pub snr_threshold_db: f32,
    pub min_duration_ms: f64,
    pub merge_gap_ms: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            mode: DetectionMode::AutoWideband,
            snr_threshold_db: 12.0,
            min_duration_ms: 1.4,
            merge_gap_ms: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedCallEvent {
    pub id: usize,
    pub domain: TaxonDomain,
    pub begin_time_s: f64,
    pub end_time_s: f64,
    pub duration_ms: f64,
    pub ipi_ms: Option<f64>,
    pub calls_per_sec: Option<f64>,
    pub f_max_khz: f64,
    pub f_min_khz: f64,
    pub bandwidth_khz: f64,
    pub f_peak_khz: f64,
    pub f_char_khz: f64,
    pub f_knee_khz: f64,
    pub slope_init_khz_per_ms: f64,
    pub slope_char_khz_per_ms: f64,
    pub slope_total_khz_per_ms: f64,
    pub wiener_entropy: f64,
    pub peak_dbfs: f32,
    pub snr_db: f32,
    pub mean_tkeo: f64,
    pub harmonic_2nd_rel_db: f32,
    pub topology: CallShapeTopology,
    pub phase: BehavioralPhase,
    pub ridge_trace: Vec<(f64, f64)>,
    pub top_matches: Vec<TaxonMatch>,
    pub ambiguity_couplet: Option<String>,
    pub manual_vetted_code: Option<String>,
}

impl DetectedCallEvent {
    pub fn primary_code(&self) -> &str {
        if let Some(m) = &self.manual_vetted_code {
            return m.as_str();
        }
        self.top_matches
            .first()
            .map(|t| t.code.as_str())
            .unwrap_or("UNID")
    }

    pub fn primary_confidence(&self) -> f64 {
        self.top_matches
            .first()
            .map(|t| t.probability)
            .unwrap_or(0.0)
    }
}

pub fn extract_calls(
    samples: &[f32],
    spec: &Spectrogram,
    cfg: DetectorConfig,
) -> Vec<DetectedCallEvent> {
    if spec.num_frames == 0 || samples.is_empty() {
        return Vec::new();
    }
    let tkeo = compute_tkeo(samples);
    let nyquist_hz = spec.sample_rate_hz as f64 * 0.5;
    let mut events = Vec::new();

    let run_bat = matches!(
        cfg.mode,
        DetectionMode::AutoWideband | DetectionMode::BatUltrasonicOnly
    ) && nyquist_hz >= 16_000.0;

    let run_bird = matches!(
        cfg.mode,
        DetectionMode::AutoWideband | DetectionMode::BirdAudibleOnly
    );

    if run_bat {
        let bat_lo_hz = 14_000.0;
        let bat_hi_hz = nyquist_hz.min(180_000.0);
        let mut bat_events = scan_band_events(
            samples,
            &tkeo,
            spec,
            TaxonDomain::BatChiroptera,
            bat_lo_hz,
            bat_hi_hz,
            cfg.snr_threshold_db,
            cfg.min_duration_ms.max(1.0),
            cfg.merge_gap_ms.min(1.5),
        );
        events.append(&mut bat_events);
    }

    if run_bird {
        let bird_lo_hz = 550.0;
        let bird_hi_hz = nyquist_hz.min(12_500.0);
        if bird_hi_hz > bird_lo_hz + 500.0 {
            let mut bird_events = scan_band_events(
                samples,
                &tkeo,
                spec,
                TaxonDomain::BirdAves,
                bird_lo_hz,
                bird_hi_hz,
                cfg.snr_threshold_db,
                cfg.min_duration_ms.max(18.0),
                cfg.merge_gap_ms.max(12.0),
            );
            events.append(&mut bird_events);
        }
    }

    events.sort_by(|a, b| a.begin_time_s.total_cmp(&b.begin_time_s));
    let mut last_bat_begin: Option<f64> = None;
    let mut last_bird_begin: Option<f64> = None;

    for (idx, ev) in events.iter_mut().enumerate() {
        ev.id = idx + 1;
        let prev = match ev.domain {
            TaxonDomain::BatChiroptera => last_bat_begin.replace(ev.begin_time_s),
            TaxonDomain::BirdAves => last_bird_begin.replace(ev.begin_time_s),
        };
        if let Some(t_prev) = prev {
            let ipi_ms = (ev.begin_time_s - t_prev) * 1000.0;
            if ipi_ms > 0.5 && ipi_ms < 5_000.0 {
                ev.ipi_ms = Some(ipi_ms);
                ev.calls_per_sec = Some(1000.0 / ipi_ms);
            }
        }

        if ev.domain == TaxonDomain::BatChiroptera {
            ev.phase = match ev.ipi_ms {
                Some(ipi) if ipi < 13.5 && ev.duration_ms < 4.5 => {
                    BehavioralPhase::TerminalFeedingBuzz
                }
                Some(ipi) if ipi < 38.0 && ev.duration_ms < 7.0 => BehavioralPhase::ApproachPhase,
                _ => BehavioralPhase::SearchPhase,
            };
        }

        let (matches, couplet) = classify_call_event(ev);
        ev.top_matches = matches;
        ev.ambiguity_couplet = couplet;
    }

    events
}

#[allow(clippy::too_many_arguments)]
fn scan_band_events(
    samples: &[f32],
    tkeo: &[f32],
    spec: &Spectrogram,
    domain: TaxonDomain,
    band_lo_hz: f64,
    band_hi_hz: f64,
    snr_thresh_db: f32,
    min_dur_ms: f64,
    merge_gap_ms: f64,
) -> Vec<DetectedCallEvent> {
    let b0 = spec.freq_to_bin(band_lo_hz).max(1);
    let b1 = spec.freq_to_bin(band_hi_hz).min(spec.num_bins.saturating_sub(1));
    if b1 <= b0 {
        return Vec::new();
    }

    let frame_dt_s = spec.hop_size as f64 / spec.sample_rate_hz as f64;
    let merge_frames = ((merge_gap_ms * 0.001) / frame_dt_s).round() as usize;

    let mut frame_dom_bin = vec![b0; spec.num_frames];
    let mut frame_dom_db = vec![-120.0_f32; spec.num_frames];

    for f in 0..spec.num_frames {
        let mut best_db = -120.0_f32;
        let mut best_bin = b0;
        for b in b0..=b1 {
            let db = spec.db_at(f, b);
            if db > best_db {
                best_db = db;
                best_bin = b;
            }
        }
        frame_dom_bin[f] = best_bin;
        frame_dom_db[f] = best_db;
    }

    // 35th-percentile of frame band-maxima provides a multiple-comparison-safe noise floor
    let mut sorted_max = frame_dom_db.clone();
    sorted_max.sort_by(|a, c| a.total_cmp(c));
    let band_baseline = sorted_max[(sorted_max.len() * 35 / 100).min(sorted_max.len() - 1)];
    let frame_snr: Vec<f32> = frame_dom_db.iter().map(|&d| d - band_baseline).collect();

    let sustain_db = snr_thresh_db * 0.58;
    let mut raw_spans: Vec<(usize, usize)> = Vec::new();
    let mut f = 0usize;
    while f < spec.num_frames {
        if frame_snr[f] >= snr_thresh_db {
            let start = f;
            let mut end = f;
            while end + 1 < spec.num_frames && frame_snr[end + 1] >= sustain_db {
                end += 1;
            }
            if let Some(last) = raw_spans.last_mut() {
                if start <= last.1 + merge_frames.max(1) {
                    last.1 = end;
                } else {
                    raw_spans.push((start, end));
                }
            } else {
                raw_spans.push((start, end));
            }
            f = end + 1;
        } else {
            f += 1;
        }
    }

    let total_dur_s = samples.len() as f64 / spec.sample_rate_hz as f64;
    let mut results = Vec::new();

    for (f_start, f_end) in raw_spans {
        let call_peak_db = frame_dom_db[f_start..=f_end]
            .iter()
            .copied()
            .fold(-120.0_f32, f32::max);
        let ridge_gate_db = call_peak_db - 20.0;

        let mut r_start = f_start;
        let mut r_end = f_end;
        while r_start < f_end && frame_dom_db[r_start] < ridge_gate_db {
            r_start += 1;
        }
        while r_end > r_start && frame_dom_db[r_end] < ridge_gate_db {
            r_end -= 1;
        }

        let t_begin = (r_start * spec.hop_size) as f64 / spec.sample_rate_hz as f64;
        let t_end = (((r_end + 1) * spec.hop_size) as f64 + spec.fft_size as f64 * 0.35)
            / spec.sample_rate_hz as f64;
        let t_end_clamped = t_end.min(total_dur_s);
        let dur_ms = (t_end_clamped - t_begin) * 1000.0;
        if dur_ms < min_dur_ms {
            continue;
        }

        struct FrameCandidate {
            fr: usize,
            t_s: f64,
            dom_b: usize,
            f_khz: f64,
            dom_db: f32,
            sub_khz: f64,
            sub_db: f32,
            is_local_peak: bool,
            has_valid_sub: bool,
        }

        let mut candidates: Vec<FrameCandidate> = Vec::with_capacity(r_end - r_start + 1);
        for fr in r_start..=r_end {
            if frame_dom_db[fr] < ridge_gate_db {
                continue;
            }
            let dom_b = frame_dom_bin[fr];
            let interp_b = if dom_b > b0 && dom_b < b1 {
                let a = spec.db_at(fr, dom_b - 1) as f64;
                let c = spec.db_at(fr, dom_b) as f64;
                let g = spec.db_at(fr, dom_b + 1) as f64;
                let denom = a - 2.0 * c + g;
                if denom.abs() > 1e-9 {
                    dom_b as f64 + 0.5 * (a - g) / denom
                } else {
                    dom_b as f64
                }
            } else {
                dom_b as f64
            };
            let f_khz = (interp_b * spec.sample_rate_hz as f64 / spec.fft_size as f64) * 0.001;
            let t_s = spec.frame_time_s(fr);
            let dom_db = frame_dom_db[fr];

            let mut sub_khz = f_khz * 0.5;
            let mut sub_db = -120.0_f32;
            let mut is_local_peak = false;
            let mut has_valid_sub = false;

            if domain == TaxonDomain::BatChiroptera && dom_b >= 2 * b0 {
                let target_half_b = (interp_b * 0.5).round() as usize;
                let sub_lo = target_half_b.saturating_sub(1).max(1);
                let sub_hi = (target_half_b + 1)
                    .min(dom_b.saturating_sub(2))
                    .min(spec.num_bins.saturating_sub(1));
                if sub_lo <= sub_hi {
                    let mut best_sub_b = sub_lo;
                    let mut best_sub_db = spec.db_at(fr, sub_lo);
                    for b in (sub_lo + 1)..=sub_hi {
                        let d = spec.db_at(fr, b);
                        if d > best_sub_db {
                            best_sub_db = d;
                            best_sub_b = b;
                        }
                    }
                    sub_db = best_sub_db;

                    let interp_sub_b = if best_sub_b > 1 && best_sub_b + 1 < spec.num_bins {
                        let a = spec.db_at(fr, best_sub_b - 1) as f64;
                        let c = spec.db_at(fr, best_sub_b) as f64;
                        let g = spec.db_at(fr, best_sub_b + 1) as f64;
                        let denom = a - 2.0 * c + g;
                        if denom.abs() > 1e-9 && denom < 0.0 {
                            (best_sub_b as f64 + 0.5 * (a - g) / denom)
                                .clamp(best_sub_b as f64 - 0.5, best_sub_b as f64 + 0.5)
                        } else {
                            best_sub_b as f64
                        }
                    } else {
                        best_sub_b as f64
                    };
                    sub_khz =
                        (interp_sub_b * spec.sample_rate_hz as f64 / spec.fft_size as f64) * 0.001;

                    let left_lo = best_sub_b.saturating_sub(2).max(1);
                    let right_hi = (best_sub_b + 2)
                        .min(dom_b.saturating_sub(1))
                        .min(spec.num_bins.saturating_sub(1));
                    let drop_left =
                        (left_lo..best_sub_b).any(|b| spec.db_at(fr, b) < sub_db - 1e-3);
                    let drop_right =
                        ((best_sub_b + 1)..=right_hi).any(|b| spec.db_at(fr, b) < sub_db - 1e-3);
                    let immediate_drop = (best_sub_b > 1
                        && spec.db_at(fr, best_sub_b - 1) < sub_db - 1e-3)
                        || (best_sub_b + 1 < dom_b
                            && best_sub_b + 1 < spec.num_bins
                            && spec.db_at(fr, best_sub_b + 1) < sub_db - 1e-3);
                    is_local_peak = left_lo <= right_hi
                        && (left_lo..=right_hi).all(|b| sub_db >= spec.db_at(fr, b) - 1e-4)
                        && drop_left
                        && drop_right
                        && immediate_drop;

                    let sub_snr = sub_db - spec.bin_noise_floor_dbfs[best_sub_b];
                    has_valid_sub = is_local_peak
                        && sub_db >= (dom_db - 18.0)
                        && sub_snr >= (snr_thresh_db * 0.72)
                        && sub_khz >= 18.0;
                }
            }

            candidates.push(FrameCandidate {
                fr,
                t_s,
                dom_b,
                f_khz,
                dom_db,
                sub_khz,
                sub_db,
                is_local_peak,
                has_valid_sub,
            });
        }

        let total_active = candidates.len();
        let sub_valid_count = candidates.iter().filter(|c| c.has_valid_sub).count();
        // Call-level quorum: require >= 40% of active call frames to exhibit a genuine
        // local subharmonic peak within -18 dB of the dominant harmonic before halving.
        let use_subharmonic_f0 = domain == TaxonDomain::BatChiroptera
            && total_active > 0
            && sub_valid_count * 5 >= total_active * 2;

        if candidates.is_empty() {
            continue;
        }

        let resolve_frame =
            |c: &FrameCandidate, ref_f0_khz: &mut Option<f64>| -> (f64, Option<f32>) {
                let consistent_with_2f0 = ref_f0_khz
                    .map(|r| (c.f_khz - 2.0 * r).abs() < (c.f_khz - r).abs())
                    .unwrap_or(true);
                if use_subharmonic_f0
                    && (c.has_valid_sub
                        || (consistent_with_2f0
                            && c.is_local_peak
                            && c.sub_khz >= 18.0
                            && c.sub_db >= (c.dom_db - 22.0)
                            && c.dom_b >= 2 * b0))
                {
                    *ref_f0_khz = Some(c.sub_khz);
                    (c.sub_khz, Some(c.dom_db - c.sub_db))
                } else if use_subharmonic_f0
                    && consistent_with_2f0
                    && c.dom_b >= 2 * b0
                    && (c.f_khz * 0.5) >= 18.0
                {
                    // Call-level >= 40% quorum already proved dominant frames were 2f0,
                    // and c.f_khz is closer to 2 * ref_f0 than ref_f0 itself; halve continuously.
                    let half = c.f_khz * 0.5;
                    *ref_f0_khz = Some(half);
                    (half, None)
                } else {
                    let h_rel =
                        if domain == TaxonDomain::BatChiroptera && c.dom_b * 2 < spec.num_bins {
                            let h2_lo = (c.dom_b * 2).saturating_sub(1);
                            let h2_hi = (c.dom_b * 2 + 1).min(spec.num_bins - 1);
                            let mut h2_db = spec.db_at(c.fr, c.dom_b * 2);
                            for b in h2_lo..=h2_hi {
                                h2_db = h2_db.max(spec.db_at(c.fr, b));
                            }
                            Some(h2_db - c.dom_db)
                        } else {
                            None
                        };
                    if use_subharmonic_f0 {
                        *ref_f0_khz = Some(c.f_khz);
                    }
                    (c.f_khz, h_rel)
                }
            };

        let mut resolved = vec![(0.0_f64, None::<f32>); total_active];
        if let Some(first_sub_idx) = use_subharmonic_f0
            .then(|| candidates.iter().position(|c| c.has_valid_sub))
            .flatten()
        {
            let seed = candidates[first_sub_idx].sub_khz;
            let mut ref_bwd = Some(seed);
            for idx in (0..first_sub_idx).rev() {
                resolved[idx] = resolve_frame(&candidates[idx], &mut ref_bwd);
            }
            let mut ref_fwd = Some(seed);
            for idx in first_sub_idx..total_active {
                resolved[idx] = resolve_frame(&candidates[idx], &mut ref_fwd);
            }
        } else {
            let mut ref_none = None;
            for idx in 0..total_active {
                resolved[idx] = resolve_frame(&candidates[idx], &mut ref_none);
            }
        }

        let mut ridge: Vec<(f64, f64)> = Vec::with_capacity(total_active);
        let mut peak_dbfs = -120.0_f32;
        let mut peak_snr = 0.0_f32;
        let mut peak_freq_khz = 0.0_f64;
        let mut wiener_acc = 0.0_f64;
        let mut harmonic_diff_acc = 0.0_f32;
        let mut harmonic_count = 0usize;

        for (c, (fund_khz, h_rel)) in candidates.iter().zip(resolved.into_iter()) {
            if let Some(rel) = h_rel {
                harmonic_diff_acc += rel;
                harmonic_count += 1;
            }
            ridge.push((c.t_s, fund_khz));
            wiener_acc += spec.frame_wiener_entropy[c.fr];
            if c.dom_db > peak_dbfs {
                peak_dbfs = c.dom_db;
                peak_freq_khz = fund_khz;
            }
            peak_snr = peak_snr.max(frame_snr[c.fr]);
        }

        if ridge.is_empty() {
            continue;
        }

        let f_max_khz = ridge.iter().fold(0.0_f64, |acc, &(_, fk)| acc.max(fk));
        let f_min_khz = ridge
            .iter()
            .fold(f64::INFINITY, |acc, &(_, fk)| acc.min(fk));
        let bandwidth_khz = (f_max_khz - f_min_khz).max(0.05);

        let (f_char_khz, f_knee_khz, slope_init, slope_char) =
            compute_call_contour_metrics(&ridge, f_max_khz, f_min_khz, peak_freq_khz);

        let slope_total = if dur_ms > 0.1 {
            (ridge.last().map(|x| x.1).unwrap_or(f_min_khz)
                - ridge.first().map(|x| x.1).unwrap_or(f_max_khz))
                / dur_ms
        } else {
            0.0
        };

        let s0 = ((t_begin * spec.sample_rate_hz as f64) as usize).min(samples.len());
        let s1 = ((t_end_clamped * spec.sample_rate_hz as f64) as usize)
            .min(samples.len())
            .max(s0 + 1);
        let mean_tkeo = tkeo[s0..s1.min(tkeo.len())]
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>()
            / ((s1 - s0).max(1) as f64);

        let wiener_mean = wiener_acc / ridge.len() as f64;
        let harmonic_2nd_rel_db = if harmonic_count > 0 {
            harmonic_diff_acc / harmonic_count as f32
        } else {
            -24.0
        };

        let (topology, phase) = classify_topology_and_initial_phase(
            domain,
            dur_ms,
            f_max_khz,
            f_min_khz,
            bandwidth_khz,
            slope_init,
            slope_char,
            wiener_mean,
            &ridge,
        );

        results.push(DetectedCallEvent {
            id: 0,
            domain,
            begin_time_s: t_begin,
            end_time_s: t_end_clamped,
            duration_ms: dur_ms,
            ipi_ms: None,
            calls_per_sec: None,
            f_max_khz,
            f_min_khz,
            bandwidth_khz,
            f_peak_khz: peak_freq_khz,
            f_char_khz,
            f_knee_khz,
            slope_init_khz_per_ms: slope_init,
            slope_char_khz_per_ms: slope_char,
            slope_total_khz_per_ms: slope_total,
            wiener_entropy: wiener_mean,
            peak_dbfs,
            snr_db: peak_snr,
            mean_tkeo,
            harmonic_2nd_rel_db,
            topology,
            phase,
            ridge_trace: ridge,
            top_matches: Vec::new(),
            ambiguity_couplet: None,
            manual_vetted_code: None,
        });
    }

    results
}

fn compute_call_contour_metrics(
    ridge: &[(f64, f64)],
    f_max_khz: f64,
    f_min_khz: f64,
    f_peak_khz: f64,
) -> (f64, f64, f64, f64) {
    let n = ridge.len();
    if n < 3 {
        return (f_min_khz.max(0.1), 0.5 * (f_max_khz + f_min_khz), 0.0, 0.0);
    }

    let mut local_slopes = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dt_ms = ((ridge[i + 1].0 - ridge[i].0) * 1000.0).max(0.05);
        let df = ridge[i + 1].1 - ridge[i].1;
        local_slopes.push(df / dt_ms);
    }

    let init_end = (local_slopes.len() / 3).max(1);
    let slope_init = local_slopes[..init_end].iter().sum::<f64>() / init_end as f64;

    let tail_start = (local_slopes.len() * 45 / 100).min(local_slopes.len() - 1);
    let mut min_abs_slope = f64::INFINITY;
    let mut fc_idx = tail_start;
    for (idx, &slp) in local_slopes.iter().enumerate().skip(tail_start) {
        if slp.abs() < min_abs_slope {
            min_abs_slope = slp.abs();
            fc_idx = idx;
        }
    }
    let f_char_khz = ridge[fc_idx + 1].1.clamp(f_min_khz, f_max_khz);
    let slope_char = local_slopes[fc_idx];

    let mut best_curv = -1.0_f64;
    let mut knee_idx = n / 3;
    for i in 1..local_slopes.len().min(fc_idx + 1) {
        let curv = (local_slopes[i] - local_slopes[i - 1]).abs();
        if curv > best_curv {
            best_curv = curv;
            knee_idx = i;
        }
    }
    let f_knee_khz = if best_curv > 0.05 {
        ridge[knee_idx].1
    } else {
        f_peak_khz
    };

    (f_char_khz, f_knee_khz, slope_init, slope_char)
}

#[allow(clippy::too_many_arguments)]
fn classify_topology_and_initial_phase(
    domain: TaxonDomain,
    dur_ms: f64,
    f_max_khz: f64,
    f_min_khz: f64,
    bw_khz: f64,
    slope_init: f64,
    slope_char: f64,
    wiener: f64,
    ridge: &[(f64, f64)],
) -> (CallShapeTopology, BehavioralPhase) {
    match domain {
        TaxonDomain::BatChiroptera => {
            let mid_f = ridge[ridge.len() / 2].1;
            let first_f = ridge[0].1;
            let last_f = ridge[ridge.len() - 1].1;
            let is_staple_cf = dur_ms >= 15.0
                && f_max_khz >= 65.0
                && slope_char.abs() < 0.35
                && (mid_f - first_f) > 2.0
                && (mid_f - last_f) > 2.0;

            let topo = if is_staple_cf {
                CallShapeTopology::FmCfFm
            } else if bw_khz < 6.5 || (slope_init.abs() < 1.2 && slope_char.abs() < 0.6) {
                CallShapeTopology::FlatQcf
            } else if slope_init.abs() > 2.8 && slope_char.abs() < 1.6 {
                CallShapeTopology::FmQcf
            } else {
                CallShapeTopology::SteepFm
            };
            let _ = f_min_khz;
            (topo, BehavioralPhase::SearchPhase)
        }
        TaxonDomain::BirdAves => {
            let is_nfc = dur_ms >= 35.0
                && dur_ms <= 210.0
                && f_min_khz >= 2.0
                && f_max_khz <= 9.8
                && bw_khz >= 0.45;
            if is_nfc && (slope_init > 0.002 || bw_khz > 0.9) {
                (
                    CallShapeTopology::AvianFlightCall,
                    BehavioralPhase::NocturnalFlightCall,
                )
            } else if wiener < 0.22 && bw_khz < 1.2 {
                (
                    CallShapeTopology::AvianSongWhistle,
                    BehavioralPhase::TerritorialOrContactVocalization,
                )
            } else {
                (
                    CallShapeTopology::AvianHarmonicOrTrill,
                    BehavioralPhase::TerritorialOrContactVocalization,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{compute_stft, StftConfig};
    use std::f64::consts::{PI, TAU};

    fn synth_chirp_250k(
        dur_ms: f64,
        f0_start_khz: f64,
        f0_end_khz: f64,
        harmonics: &[(f64, f64, f64)], // (harmonic_mult, rel_amp, frac_active)
    ) -> (Vec<f32>, Spectrogram) {
        let sr = 250_000_u32;
        let total_s = 0.08_f64;
        let n_samp = (total_s * sr as f64) as usize;
        let mut samples = vec![0.0_f32; n_samp];
        let start_s = 0.020;
        let dur_s = dur_ms * 0.001;
        let n_pulse = (dur_s * sr as f64) as usize;
        let i0 = (start_s * sr as f64) as usize;
        let mut phases = vec![0.0_f64; harmonics.len()];

        for k in 0..n_pulse {
            let u = k as f64 / n_pulse.max(1) as f64;
            // Exponential-like hockey-stick FM contour with exact phase integration
            let f_inst_hz = (f0_end_khz + (f0_start_khz - f0_end_khz) * (-4.2 * u).exp()) * 1000.0;
            let env = (PI * u).sin().powf(0.4);
            let mut val = 0.0_f64;
            for (h_idx, &(mult, amp, frac_active)) in harmonics.iter().enumerate() {
                phases[h_idx] += TAU * f_inst_hz * mult / sr as f64;
                if u > frac_active {
                    continue;
                }
                val += amp * phases[h_idx].sin();
            }
            samples[i0 + k] = (0.72 * env * val).clamp(-0.99, 0.99) as f32;
        }
        let spec = compute_stft(
            &samples,
            sr,
            StftConfig {
                fft_size: 512,
                hop_size: 64,
                ..Default::default()
            },
        );
        (samples, spec)
    }

    #[test]
    fn mylu_synthetic_weak_minus25db_and_sparse_subharmonic_noise_not_halved() {
        // Case A: MYLU with -25 dB continuous subharmonic noise across 100% of frames
        let (s_weak, spec_weak) = synth_chirp_250k(
            5.5,
            74.0,
            41.0,
            &[(1.0, 1.0, 1.0), (0.5, 0.0562, 1.0)],
        );
        let calls_weak = extract_calls(&s_weak, &spec_weak, DetectorConfig::default());
        assert!(!calls_weak.is_empty());
        assert!(
            calls_weak[0].f_char_khz > 36.0 && calls_weak[0].f_min_khz > 35.0,
            "Weak -25 dB subharmonic must not halve MYLU: Fc={}, Fmin={}",
            calls_weak[0].f_char_khz,
            calls_weak[0].f_min_khz
        );
        assert_eq!(calls_weak[0].primary_code(), "MYLU");

        // Case B: MYLU with strong -6 dB subharmonic present in < 40% (25%) of pulse frames
        let (s_sparse, spec_sparse) = synth_chirp_250k(
            6.0,
            74.0,
            41.0,
            &[(1.0, 1.0, 1.0), (0.5, 0.50, 0.25)],
        );
        let calls_sparse = extract_calls(&s_sparse, &spec_sparse, DetectorConfig::default());
        assert!(!calls_sparse.is_empty());
        assert!(
            calls_sparse[0].f_char_khz > 36.0 && calls_sparse[0].f_min_khz > 35.0,
            "Sparse (<40%) subharmonic must not halve MYLU: Fc={}, Fmin={}",
            calls_sparse[0].f_char_khz,
            calls_sparse[0].f_min_khz
        );
        assert_eq!(calls_sparse[0].primary_code(), "MYLU");
    }

    #[test]
    fn coto_20_5_khz_genuine_harmonic_call_classifies_as_coto() {
        let (samples, spec) = synth_chirp_250k(
            4.6,
            36.0,
            20.5,
            &[(1.0, 0.45, 1.0), (2.0, 1.0, 1.0), (3.0, 0.35, 1.0)],
        );
        let calls = extract_calls(&samples, &spec, DetectorConfig::default());
        assert!(!calls.is_empty());
        assert!(
            (19.0..=23.5).contains(&calls[0].f_char_khz),
            "Genuine COTO call should track f0 ~20.5 kHz, got Fc={}",
            calls[0].f_char_khz
        );
        assert!(
            calls[0].harmonic_2nd_rel_db > -9.0,
            "Expected dominant 2f0 relative dB > -9.0, got {}",
            calls[0].harmonic_2nd_rel_db
        );
        assert_eq!(calls[0].primary_code(), "COTO");
    }

    #[test]
    fn genuine_harmonic_call_within_minus18db_quorum_halves_to_f0() {
        // f0 at 0.18 (~ -14.9 dB relative to 2f0 = 1.0) across 65% of pulse (>= 40% quorum)
        let (samples, spec) = synth_chirp_250k(
            5.0,
            38.0,
            22.0,
            &[(2.0, 1.0, 1.0), (1.0, 0.18, 0.65)],
        );
        let calls = extract_calls(&samples, &spec, DetectorConfig::default());
        assert!(!calls.is_empty());
        assert!(
            (19.5..=26.5).contains(&calls[0].f_char_khz),
            "Genuine -14.9 dB f0 across >=40% frames should halve to f0 ~22 kHz, got Fc={}",
            calls[0].f_char_khz
        );
        assert!(calls[0].harmonic_2nd_rel_db > 0.0);
    }

    #[test]
    fn mid_call_2f0_to_f0_energy_shift_does_not_double_halve() {
        // f0 sweeps 58 -> 38 kHz over 6.0 ms:
        // First 50% has 2f0 dominant (1.0) and f0 at -10 dB (0.316) -> triggers subharmonic quorum.
        // Second 50% shifts dominant energy to f0 (1.0) with 2f0 at 0.20 (-14 dB).
        let sr = 250_000_u32;
        let total_s = 0.08_f64;
        let n_samp = (total_s * sr as f64) as usize;
        let mut samples = vec![0.0_f32; n_samp];
        let start_s = 0.020;
        let dur_s = 0.006;
        let n_pulse = (dur_s * sr as f64) as usize;
        let i0 = (start_s * sr as f64) as usize;
        let mut ph1 = 0.0_f64;
        let mut ph2 = 0.0_f64;

        for k in 0..n_pulse {
            let u = k as f64 / n_pulse.max(1) as f64;
            let f_inst_hz = (38.0 + (58.0 - 38.0) * (-4.2 * u).exp()) * 1000.0;
            ph1 += TAU * f_inst_hz / sr as f64;
            ph2 += TAU * (2.0 * f_inst_hz) / sr as f64;
            let env = (PI * u).sin().powf(0.4);
            let (a1, a2) = if u < 0.50 { (0.316, 1.0) } else { (1.0, 0.20) };
            let val = a1 * ph1.sin() + a2 * ph2.sin();
            samples[i0 + k] = (0.65 * env * val).clamp(-0.99, 0.99) as f32;
        }
        let spec = compute_stft(
            &samples,
            sr,
            StftConfig {
                fft_size: 512,
                hop_size: 64,
                ..Default::default()
            },
        );
        let calls = extract_calls(&samples, &spec, DetectorConfig::default());
        assert_eq!(calls.len(), 1);
        let ev = &calls[0];
        assert!(
            ev.f_min_khz >= 35.0 && ev.f_char_khz >= 36.0,
            "Mid-call 2f0->f0 shift must preserve f0 (~38-40 kHz), got Fc={}, Fmin={}",
            ev.f_char_khz,
            ev.f_min_khz
        );
        let half = ev.ridge_trace.len() / 2;
        assert!(
            ev.ridge_trace[half..].iter().all(|&(_, f)| f >= 35.0),
            "Second-half ridge_trace must remain near 40..38 kHz, got {:?}",
            &ev.ridge_trace[half..]
        );
    }

    #[test]
    fn flat_spectral_plateau_rejected_by_subharmonic_peak_check() {
        // Dominant 82 kHz tone with a flat spectral plateau injected around 41 kHz.
        let sr = 250_000_u32;
        let n_samp = (0.08 * sr as f64) as usize;
        let mut samples = vec![0.0_f32; n_samp];
        let i0 = (0.02 * sr as f64) as usize;
        let n_pulse = (0.005 * sr as f64) as usize;
        for k in 0..n_pulse {
            let u = k as f64 / n_pulse as f64;
            let env = (PI * u).sin().powf(0.4);
            let ph = TAU * 82_000.0 * (k as f64 / sr as f64);
            samples[i0 + k] = (0.7 * env * ph.sin()) as f32;
        }
        let mut spec = compute_stft(
            &samples,
            sr,
            StftConfig {
                fft_size: 512,
                hop_size: 64,
                ..Default::default()
            },
        );
        let dom_b = spec.freq_to_bin(82_000.0);
        let half_b = spec.freq_to_bin(41_000.0);
        for fr in 0..spec.num_frames {
            let dom_db = spec.db_at(fr, dom_b);
            if dom_db > -30.0 {
                let plateau_db = dom_db - 10.0;
                for b in half_b.saturating_sub(3)..=(half_b + 3).min(spec.num_bins - 1) {
                    spec.db_grid[fr * spec.num_bins + b] = plateau_db;
                }
            }
        }
        let calls = extract_calls(&samples, &spec, DetectorConfig::default());
        assert!(!calls.is_empty());
        assert!(
            calls[0].f_min_khz > 75.0 && calls[0].f_char_khz > 75.0,
            "Flat plateau at f/2 must be rejected, got Fc={}, Fmin={}",
            calls[0].f_char_khz,
            calls[0].f_min_khz
        );
    }

    #[test]
    fn exact_5_bin_flat_plateau_rejected_by_subharmonic_peak_check() {
        // Exact 5-bin [-20, -20, -20, -20, -20] plateau across half_b - 2 ..= half_b + 2
        // on an 82 kHz tone must not halve to ~40.5 kHz.
        let sr = 250_000_u32;
        let n_samp = (0.08 * sr as f64) as usize;
        let mut samples = vec![0.0_f32; n_samp];
        let i0 = (0.02 * sr as f64) as usize;
        let n_pulse = (0.005 * sr as f64) as usize;
        for k in 0..n_pulse {
            let u = k as f64 / n_pulse as f64;
            let env = (PI * u).sin().powf(0.4);
            let ph = TAU * 82_000.0 * (k as f64 / sr as f64);
            samples[i0 + k] = (0.7 * env * ph.sin()) as f32;
        }
        let mut spec = compute_stft(
            &samples,
            sr,
            StftConfig {
                fft_size: 512,
                hop_size: 64,
                ..Default::default()
            },
        );
        let dom_b = spec.freq_to_bin(82_000.0);
        let half_b = spec.freq_to_bin(41_000.0);
        for plateau_db in [-20.0_f32, -15.0_f32] {
            let mut test_spec = spec.clone();
            for fr in 0..test_spec.num_frames {
                let dom_db = test_spec.db_at(fr, dom_b);
                if dom_db > plateau_db + 3.0 {
                    for b in half_b.saturating_sub(2)..=(half_b + 2).min(test_spec.num_bins - 1) {
                        test_spec.db_grid[fr * test_spec.num_bins + b] = plateau_db;
                    }
                }
            }
            let calls = extract_calls(&samples, &test_spec, DetectorConfig::default());
            assert!(!calls.is_empty());
            assert!(
                calls[0].f_min_khz > 75.0 && calls[0].f_char_khz > 75.0,
                "5-bin plateau ({plateau_db} dB) at f/2 must be rejected, got Fc={}, Fmin={}",
                calls[0].f_char_khz,
                calls[0].f_min_khz
            );
        }
        let _ = &mut spec;
    }

    #[test]
    fn wide_reverse_f0_to_2f0_sweep_preserves_leading_f0_head_without_coto() {
        let sr = 250_000_u32;
        let total_s = 0.08_f64;
        let n_samp = (total_s * sr as f64) as usize;
        let start_s = 0.020;
        let dur_s = 0.006;
        let n_pulse = (dur_s * sr as f64) as usize;
        let i0 = (start_s * sr as f64) as usize;

        for (f0_start, f0_end) in [(60.0_f64, 28.0_f64), (80.0_f64, 38.0_f64)] {
            let mut samples = vec![0.0_f32; n_samp];
            let mut ph1 = 0.0_f64;
            let mut ph2 = 0.0_f64;

            for k in 0..n_pulse {
                let u = k as f64 / n_pulse.max(1) as f64;
                let f_inst_hz = (f0_end + (f0_start - f0_end) * (-4.2 * u).exp()) * 1000.0;
                ph1 += TAU * f_inst_hz / sr as f64;
                ph2 += TAU * (2.0 * f_inst_hz) / sr as f64;
                let env = (PI * u).sin().powf(0.4);
                let (a1, a2) = if u < 0.45 { (1.0, 0.0) } else { (0.316, 1.0) };
                let val = a1 * ph1.sin() + a2 * ph2.sin();
                samples[i0 + k] = (0.65 * env * val).clamp(-0.99, 0.99) as f32;
            }

            let spec = compute_stft(
                &samples,
                sr,
                StftConfig {
                    fft_size: 512,
                    hop_size: 64,
                    ..Default::default()
                },
            );
            let calls = extract_calls(&samples, &spec, DetectorConfig::default());
            assert_eq!(calls.len(), 1);
            let ev = &calls[0];
            assert!(
                ev.f_min_khz >= f0_end - 2.0 && ev.f_max_khz >= f0_start - 10.0,
                "Sweep {f0_start}->{f0_end} kHz leading head must not halve: Fmin={}, Fmax={}, Fc={}",
                ev.f_min_khz,
                ev.f_max_khz,
                ev.f_char_khz
            );
            assert_ne!(
                ev.primary_code(),
                "COTO",
                "Sweep {f0_start}->{f0_end} kHz must not misclassify as COTO (Fc={})",
                ev.f_char_khz
            );
            assert!(
                ev.ridge_trace.windows(2).all(|w| w[1].1 - w[0].1 <= 4.0),
                "Sweep {f0_start}->{f0_end} kHz ridge_trace had upward discontinuity: {:?}",
                ev.ridge_trace
            );
        }
    }
}
