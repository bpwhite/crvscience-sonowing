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

        let mut ridge: Vec<(f64, f64)> = Vec::with_capacity(r_end - r_start + 1);
        let mut peak_dbfs = -120.0_f32;
        let mut peak_snr = 0.0_f32;
        let mut peak_freq_khz = 0.0_f64;
        let mut wiener_acc = 0.0_f64;
        let mut harmonic_diff_acc = 0.0_f32;
        let mut harmonic_count = 0usize;

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

            let mut fund_khz = f_khz;
            if domain == TaxonDomain::BatChiroptera && dom_b >= 2 * b0 {
                let half_b = dom_b / 2;
                let sub_db = spec.db_at(fr, half_b);
                let sub_snr = sub_db - spec.bin_noise_floor_dbfs[half_b];
                if sub_snr >= snr_thresh_db * 0.72 && (f_khz * 0.5) >= 18.0 {
                    fund_khz = (half_b as f64 * spec.sample_rate_hz as f64 / spec.fft_size as f64)
                        * 0.001;
                    harmonic_diff_acc += frame_dom_db[fr] - sub_db;
                    harmonic_count += 1;
                } else if dom_b * 2 < spec.num_bins {
                    let h2_db = spec.db_at(fr, dom_b * 2);
                    harmonic_diff_acc += h2_db - frame_dom_db[fr];
                    harmonic_count += 1;
                }
            }

            ridge.push((t_s, fund_khz));
            wiener_acc += spec.frame_wiener_entropy[fr];
            if frame_dom_db[fr] > peak_dbfs {
                peak_dbfs = frame_dom_db[fr];
                peak_freq_khz = fund_khz;
            }
            peak_snr = peak_snr.max(frame_snr[fr]);
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
