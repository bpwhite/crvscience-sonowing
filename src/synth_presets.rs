//! Nine physically parameterized synthetic field recordings (`4` Chiroptera ultrasonic,
//! `4` Avian audible/NFC/harmonic-stack, and `1` Wideband `192 kHz` simultaneous Bat + Bird transect)
//! with realistic overtone harmonics and embedded GUANO v1.0 metadata.

use crate::wav_guano::{BioAcousticRecording, GuanoMetadata, SampleEncoding};
use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPreset {
    B1MyluFeedingBuzz,
    B2EpfuLanoSympatric,
    B3CotoWhisperHarmonic,
    B4RhfeHorseshoeCf,
    A1SwthYrwaNightFlight,
    A2BcchWcspDawnChorus,
    A3NswoFinleyTootTrain,
    A4VathHarmonicStack,
    M1RiparianBatAndBirdDual,
}

impl BuiltinPreset {
    pub const ALL: [Self; 9] = [
        Self::B1MyluFeedingBuzz,
        Self::B2EpfuLanoSympatric,
        Self::B3CotoWhisperHarmonic,
        Self::B4RhfeHorseshoeCf,
        Self::A1SwthYrwaNightFlight,
        Self::A2BcchWcspDawnChorus,
        Self::A3NswoFinleyTootTrain,
        Self::A4VathHarmonicStack,
        Self::M1RiparianBatAndBirdDual,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::B1MyluFeedingBuzz => "B1",
            Self::B2EpfuLanoSympatric => "B2",
            Self::B3CotoWhisperHarmonic => "B3",
            Self::B4RhfeHorseshoeCf => "B4",
            Self::A1SwthYrwaNightFlight => "A1",
            Self::A2BcchWcspDawnChorus => "A2",
            Self::A3NswoFinleyTootTrain => "A3",
            Self::A4VathHarmonicStack => "A4",
            Self::M1RiparianBatAndBirdDual => "M1",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        let up = code.trim().to_ascii_uppercase();
        Self::ALL.into_iter().find(|p| p.code() == up)
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::B1MyluFeedingBuzz => {
                "B1: Myotis lucifugus (MYLU) Search -> Approach -> Terminal Feeding Buzz (250 kHz)"
            }
            Self::B2EpfuLanoSympatric => {
                "B2: Sympatric Eptesicus fuscus (EPFU) & Lasionycteris noctivagans (LANO) Pass (250 kHz)"
            }
            Self::B3CotoWhisperHarmonic => {
                "B3: Corynorhinus townsendii (COTO) Whisper Bat 3-Harmonic Stack (250 kHz)"
            }
            Self::B4RhfeHorseshoeCf => {
                "B4: Rhinolophus ferrumequinum (RHFE) 83.6 kHz FM-CF-FM Staple Calls (384 kHz)"
            }
            Self::A1SwthYrwaNightFlight => {
                "A1: Swainson's Thrush (SWTH) & Yellow-rumped Warbler (YRWA) Nocturnal Flight Calls (48 kHz)"
            }
            Self::A2BcchWcspDawnChorus => {
                "A2: Black-capped Chickadee (BCCH) 'Fee-Bee' & White-crowned Sparrow Dawn Chorus (48 kHz)"
            }
            Self::A3NswoFinleyTootTrain => {
                "A3: Northern Saw-whet Owl (NSWO) 1.15 kHz Nocturnal Advertising Train (48 kHz)"
            }
            Self::A4VathHarmonicStack => {
                "A4: Varied Thrush (VATH) & Owl Multi-Harmonic Stack Benchmark (f0..5f0, 48 kHz)"
            }
            Self::M1RiparianBatAndBirdDual => {
                "M1: Dual-Taxon Riparian Transect — Hoary Bat (LACI 20 kHz) + Swainson's Thrush NFC (192 kHz)"
            }
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::B1MyluFeedingBuzz => {
                "41 kHz Fc hockey-stick pulses accelerating from 82 ms search IPI into an 8.5 ms terminal insect capture buzz"
            }
            Self::B2EpfuLanoSympatric => {
                "Separates sympatric EPFU and LANO into distinct MFCC/UMAP clusters via Fmax (>55 kHz) and body slope"
            }
            Self::B3CotoWhisperHarmonic => {
                "Oregon whisper bat exhibiting f0=26 kHz, dominant 2f0=52 kHz, and 3f0=78 kHz harmonic stack"
            }
            Self::B4RhfeHorseshoeCf => {
                "High-duty-cycle Palearctic horseshoe bat emitting 30 ms 83.6 kHz constant-frequency staple calls"
            }
            Self::A1SwthYrwaNightFlight => {
                "Separates upward 2.7->4.1 kHz Catharus ustulatus peeps from 6.7 kHz Setophaga coronata zeets in UMAP space"
            }
            Self::A2BcchWcspDawnChorus => {
                "Low-entropy Poecile atricapillus pure whistle step vs modulated Zonotrichia song syllables"
            }
            Self::A3NswoFinleyTootTrain => {
                "Rhythmic 1.15 kHz Aegolius acadicus whistles with 2.30 kHz & 3.45 kHz harmonic overtones"
            }
            Self::A4VathHarmonicStack => {
                "Sustained Ixoreus naevius harmonic whistle stack + Megascops kennicottii 5-harmonic trill ladder"
            }
            Self::M1RiparianBatAndBirdDual => {
                "Simultaneous wide-band 192 kHz capture clustering Lasiurus cinereus QCF pulses and avian NFCs"
            }
        }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }
}

pub fn synthesize_preset(preset: BuiltinPreset) -> BioAcousticRecording {
    match preset {
        BuiltinPreset::B1MyluFeedingBuzz => build_b1_mylu(),
        BuiltinPreset::B2EpfuLanoSympatric => build_b2_epfu_lano(),
        BuiltinPreset::B3CotoWhisperHarmonic => build_b3_coto(),
        BuiltinPreset::B4RhfeHorseshoeCf => build_b4_rhfe(),
        BuiltinPreset::A1SwthYrwaNightFlight => build_a1_nfc(),
        BuiltinPreset::A2BcchWcspDawnChorus => build_a2_dawn_chorus(),
        BuiltinPreset::A3NswoFinleyTootTrain => build_a3_nswo(),
        BuiltinPreset::A4VathHarmonicStack => build_a4_harmonic_stack(),
        BuiltinPreset::M1RiparianBatAndBirdDual => build_m1_dual(),
    }
}

#[allow(clippy::too_many_arguments)]
fn add_fm_qcf_bat_pulse(
    buf: &mut [f32],
    sr: u32,
    t0_s: f64,
    dur_ms: f64,
    f_start_khz: f64,
    f_char_khz: f64,
    amp: f64,
    harmonics: &[f64],
) {
    let n_samp = ((dur_ms * 0.001) * sr as f64) as usize;
    let i0 = (t0_s * sr as f64) as usize;
    if i0 >= buf.len() || n_samp < 4 {
        return;
    }
    let end = (i0 + n_samp).min(buf.len());
    let nyq_hz = sr as f64 * 0.48;
    let mut phase = 0.0_f64;
    for k in 0..(end - i0) {
        let u = k as f64 / n_samp as f64;
        let f_khz = f_char_khz + (f_start_khz - f_char_khz) * (-4.6 * u).exp() - 1.1 * u;
        let f_hz = f_khz * 1000.0;
        phase += TAU * f_hz / sr as f64;
        let env = if u < 0.12 {
            0.5 * (1.0 - (PI * u / 0.12).cos())
        } else if u > 0.82 {
            0.5 * (1.0 + (PI * (u - 0.82) / 0.18).cos())
        } else {
            1.0
        };
        let mut h_sum = 0.0_f64;
        for (idx, &w) in harmonics.iter().enumerate() {
            let order = (idx + 1) as f64;
            if f_hz * order < nyq_hz {
                h_sum += w * (order * phase).sin();
            }
        }
        buf[i0 + k] = (buf[i0 + k] as f64 + amp * env * h_sum).clamp(-0.98, 0.98) as f32;
    }
}

fn add_staple_cf_pulse(
    buf: &mut [f32],
    sr: u32,
    t0_s: f64,
    dur_ms: f64,
    f_cf_khz: f64,
    fm_drop_khz: f64,
    amp: f64,
) {
    let n_samp = ((dur_ms * 0.001) * sr as f64) as usize;
    let i0 = (t0_s * sr as f64) as usize;
    if i0 >= buf.len() || n_samp < 8 {
        return;
    }
    let end = (i0 + n_samp).min(buf.len());
    let nyq_hz = sr as f64 * 0.48;
    let mut phase = 0.0_f64;
    for k in 0..(end - i0) {
        let u = k as f64 / n_samp as f64;
        let f_khz = if u < 0.12 {
            let r = u / 0.12;
            (f_cf_khz - fm_drop_khz * 0.75) + (fm_drop_khz * 0.75) * r
        } else if u > 0.84 {
            let r = (u - 0.84) / 0.16;
            f_cf_khz - fm_drop_khz * r * r
        } else {
            f_cf_khz
        };
        let f_hz = f_khz * 1000.0;
        phase += TAU * f_hz / sr as f64;
        let env = (PI * u).sin().powf(0.32);
        let mut s = phase.sin();
        if 2.0 * f_hz < nyq_hz {
            s += 0.22 * (2.0 * phase).sin();
        }
        buf[i0 + k] = (buf[i0 + k] as f64 + amp * env * s).clamp(-0.98, 0.98) as f32;
    }
}

#[allow(clippy::too_many_arguments)]
fn add_bird_chirp_harmonics(
    buf: &mut [f32],
    sr: u32,
    t0_s: f64,
    dur_ms: f64,
    f0_hz: f64,
    f1_hz: f64,
    mod_hz: f64,
    mod_depth_hz: f64,
    amp: f64,
    harmonics: &[f64],
) {
    let n_samp = ((dur_ms * 0.001) * sr as f64) as usize;
    let i0 = (t0_s * sr as f64) as usize;
    if i0 >= buf.len() || n_samp < 8 {
        return;
    }
    let end = (i0 + n_samp).min(buf.len());
    let nyq_hz = sr as f64 * 0.48;
    let mut phase = 0.0_f64;
    for k in 0..(end - i0) {
        let u = k as f64 / n_samp as f64;
        let t_local = k as f64 / sr as f64;
        let f_inst =
            f0_hz + (f1_hz - f0_hz) * u.powf(1.25) + mod_depth_hz * (TAU * mod_hz * t_local).sin();
        phase += TAU * f_inst / sr as f64;
        let env = (PI * u).sin().powf(0.55);
        let mut h_sum = 0.0_f64;
        for (idx, &w) in harmonics.iter().enumerate() {
            let order = (idx + 1) as f64;
            if f_inst * order < nyq_hz {
                h_sum += w * (order * phase).sin();
            }
        }
        buf[i0 + k] = (buf[i0 + k] as f64 + amp * env * h_sum).clamp(-0.98, 0.98) as f32;
    }
}

fn ambient_bed(len: usize, noise_amp: f64, hum_hz: Option<(f64, f64, u32)>, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let mut out = vec![0.0_f32; len];
    for (i, sample) in out.iter_mut().enumerate() {
        let mut v = noise_amp * rng.next_f64();
        if let Some((f_hum, a_hum, sr)) = hum_hz {
            let t = i as f64 / sr as f64;
            v += a_hum * (TAU * f_hum * t).sin() + 0.5 * a_hum * (TAU * 1.9 * f_hum * t).sin();
        }
        *sample = v as f32;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn base_guano(
    sr: u32,
    len_s: f64,
    make: &str,
    model: &str,
    lat: f64,
    lon: f64,
    species: &[&str],
    note: &str,
) -> GuanoMetadata {
    let mut g = GuanoMetadata::new_v1();
    g.timestamp = Some("2026-08-19T22:14:08-07:00".to_string());
    g.samplerate_hz = Some(sr);
    g.length_s = Some(len_s);
    g.make = Some(make.to_string());
    g.model = Some(model.to_string());
    g.firmware_version = Some("1.10.0".to_string());
    g.loc_position = Some((lat, lon));
    g.loc_elevation_m = Some(74.0);
    g.temperature_ext_c = Some(17.4);
    g.humidity_pct = Some(62.0);
    g.time_expansion = Some(1);
    g.species_auto_id = species.iter().map(|s| (*s).to_string()).collect();
    g.note = Some(note.to_string());
    g
}

fn build_b1_mylu() -> BioAcousticRecording {
    let sr = 250_000u32;
    let dur_s = 0.62;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.010, None, 101);

    for (t0, f_start, fc, dur) in [
        (0.035, 76.0, 41.4, 5.6),
        (0.118, 75.0, 41.1, 5.4),
        (0.198, 77.5, 41.6, 5.2),
        (0.274, 74.0, 41.0, 5.0),
    ] {
        add_fm_qcf_bat_pulse(&mut samples, sr, t0, dur, f_start, fc, 0.72, &[1.0, 0.22]);
    }
    for (t0, f_start, fc, dur) in [
        (0.335, 72.0, 40.5, 3.8),
        (0.362, 69.0, 40.2, 3.3),
        (0.386, 66.0, 39.8, 2.9),
    ] {
        add_fm_qcf_bat_pulse(&mut samples, sr, t0, dur, f_start, fc, 0.64, &[1.0, 0.16]);
    }
    let mut t_buzz = 0.412;
    for k in 0..8 {
        let f_hi = 56.0 - (k as f64) * 1.4;
        let f_lo = 33.5 - (k as f64) * 0.6;
        add_fm_qcf_bat_pulse(&mut samples, sr, t_buzz, 2.0, f_hi, f_lo, 0.48, &[1.0, 0.10]);
        t_buzz += 0.0085;
    }
    add_fm_qcf_bat_pulse(&mut samples, sr, 0.545, 5.5, 75.5, 41.3, 0.70, &[1.0, 0.22]);

    BioAcousticRecording {
        source_name: "MYLU_Willamette_FeedingBuzz_250kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Wildlife Acoustics",
            "Song Meter SM4BAT-FS",
            44.5646,
            -123.2620,
            &["MYLU"],
            "Corvallis Willamette River riparian edge; search to terminal feeding buzz",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_b2_epfu_lano() -> BioAcousticRecording {
    let sr = 250_000u32;
    let dur_s = 0.72;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.011, None, 202);

    for t0 in [0.050, 0.165, 0.285] {
        add_fm_qcf_bat_pulse(&mut samples, sr, t0, 9.6, 58.0, 28.4, 0.68, &[1.0, 0.30, 0.10]);
    }
    for t0 in [0.415, 0.545, 0.665] {
        add_fm_qcf_bat_pulse(&mut samples, sr, t0, 12.8, 42.5, 26.1, 0.66, &[1.0, 0.18]);
    }

    BioAcousticRecording {
        source_name: "EPFU_LANO_BaldHill_Sympatric_250kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Open Acoustic Devices",
            "AudioMoth 1.2.0",
            44.5682,
            -123.3314,
            &["EPFU", "LANO"],
            "Bald Hill Natural Area oak-savanna clearing; sympatric 27 kHz guild comparison",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_b3_coto() -> BioAcousticRecording {
    let sr = 250_000u32;
    let dur_s = 0.50;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.009, None, 303);

    // Whisper bat with suppressed f0 (0.55), dominant 2f0 (0.95), and clear 3f0 (0.42)
    for t0 in [0.055, 0.145, 0.238, 0.330, 0.422] {
        add_fm_qcf_bat_pulse(
            &mut samples,
            sr,
            t0,
            4.5,
            42.0,
            25.8,
            0.46,
            &[0.58, 0.95, 0.42],
        );
    }

    BioAcousticRecording {
        source_name: "COTO_WhisperHarmonic_250kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Wildlife Acoustics",
            "Echo Meter Touch 2 Pro",
            44.4085,
            -123.3270,
            &["COTO"],
            "Townsend's big-eared bat 3-harmonic stack (f0=26 kHz, dominant 2f0=52 kHz, 3f0=78 kHz)",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_b4_rhfe() -> BioAcousticRecording {
    let sr = 384_000u32;
    let dur_s = 0.48;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.009, None, 404);

    for (t0, cf_khz, dur_ms) in [
        (0.035, 83.6, 29.0),
        (0.115, 83.8, 31.0),
        (0.205, 83.5, 28.0),
        (0.292, 83.9, 30.5),
        (0.382, 83.7, 29.5),
    ] {
        add_staple_cf_pulse(&mut samples, sr, t0, dur_ms, cf_khz, 15.5, 0.72);
    }

    BioAcousticRecording {
        source_name: "RHFE_HorseshoeBat_CF84kHz_384kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Open Acoustic Devices",
            "AudioMoth 1.2.0",
            42.6977,
            23.3219,
            &["RHFE"],
            "Rhinolophus ferrumequinum high-duty-cycle FM-CF-FM constant-frequency staple sequence",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_a1_nfc() -> BioAcousticRecording {
    let sr = 48_000u32;
    let dur_s = 1.60;
    let mut samples = ambient_bed(
        (sr as f64 * dur_s) as usize,
        0.012,
        Some((920.0, 0.022, sr)),
        505,
    );

    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.18,
        108.0,
        2_680.0,
        4_120.0,
        28.0,
        95.0,
        0.64,
        &[1.0, 0.28, 0.10],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.92,
        102.0,
        2_720.0,
        4_180.0,
        26.0,
        90.0,
        0.62,
        &[1.0, 0.26, 0.09],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.54,
        62.0,
        5_650.0,
        7_680.0,
        115.0,
        320.0,
        0.56,
        &[1.0, 0.22],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        1.30,
        64.0,
        5_600.0,
        7_620.0,
        110.0,
        310.0,
        0.58,
        &[1.0, 0.22],
    );

    BioAcousticRecording {
        source_name: "SWTH_YRWA_NocturnalFlightCalls_48kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Open Acoustic Devices",
            "AudioMoth NFC Rooftop",
            44.5646,
            -123.2620,
            &["SWTH", "YRWA"],
            "Pacific Flyway autumnal nocturnal flight call (NFC) passage over Willamette Valley",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_a2_dawn_chorus() -> BioAcousticRecording {
    let sr = 48_000u32;
    let dur_s = 1.85;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.010, None, 606);

    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.14,
        345.0,
        3_820.0,
        3_680.0,
        4.0,
        8.0,
        0.70,
        &[1.0, 0.15],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.56,
        360.0,
        3_240.0,
        3_190.0,
        3.0,
        6.0,
        0.68,
        &[1.0, 0.15],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        1.08,
        230.0,
        4_150.0,
        4_180.0,
        6.0,
        12.0,
        0.60,
        &[1.0, 0.32, 0.14],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        1.38,
        205.0,
        5_250.0,
        3_550.0,
        95.0,
        380.0,
        0.58,
        &[1.0, 0.28, 0.12],
    );

    BioAcousticRecording {
        source_name: "BCCH_WCSP_DawnChorus_48kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Wildlife Acoustics",
            "Song Meter Micro 2",
            44.4148,
            -123.3255,
            &["BCCH", "WCSP"],
            "Dawn chorus pure-whistle vs modulated-trill syllable benchmark",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_a3_nswo() -> BioAcousticRecording {
    let sr = 48_000u32;
    let dur_s = 1.90;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.009, None, 707);

    for t0 in [0.22, 0.86, 1.50] {
        add_bird_chirp_harmonics(
            &mut samples,
            sr,
            t0,
            136.0,
            1_140.0,
            1_165.0,
            5.0,
            6.0,
            0.66,
            &[1.0, 0.34, 0.16, 0.08],
        );
    }

    BioAcousticRecording {
        source_name: "NSWO_FinleyNWR_OwlTrain_48kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Wildlife Acoustics",
            "Song Meter Mini 2",
            44.4106,
            -123.3292,
            &["NSWO"],
            "William L. Finley NWR nocturnal Aegolius acadicus rhythmic 1.15 kHz advertising train",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_a4_harmonic_stack() -> BioAcousticRecording {
    let sr = 48_000u32;
    let dur_s = 1.95;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.009, None, 909);

    // Varied Thrush (VATH) rich harmonic sustained whistle (f0 ~ 2.95 kHz + 2f0, 3f0, 4f0)
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.14,
        620.0,
        2_950.0,
        2_980.0,
        28.0,
        22.0,
        0.48,
        &[1.0, 0.56, 0.34, 0.18],
    );
    // Western Screech-Owl (WSOW) 5-rung harmonic series (~0.80 kHz fundamental + overtones up to 4.0 kHz)
    for t0 in [0.98, 1.24, 1.48, 1.70] {
        add_bird_chirp_harmonics(
            &mut samples,
            sr,
            t0,
            96.0,
            780.0,
            810.0,
            6.0,
            5.0,
            0.46,
            &[1.0, 0.62, 0.40, 0.25, 0.14],
        );
    }

    BioAcousticRecording {
        source_name: "VATH_WSOW_HarmonicStack_48kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Wildlife Acoustics",
            "Song Meter SM4",
            44.5810,
            -123.3420,
            &["VATH", "WSOW"],
            "McDonald-Dunn Forest multi-harmonic avian stack benchmark (f0..5f0)",
        )),
        riff_info: BTreeMap::new(),
    }
}

fn build_m1_dual() -> BioAcousticRecording {
    let sr = 192_000u32;
    let dur_s = 0.78;
    let mut samples = ambient_bed((sr as f64 * dur_s) as usize, 0.010, None, 808);

    for t0 in [0.065, 0.310, 0.565] {
        add_fm_qcf_bat_pulse(&mut samples, sr, t0, 14.5, 25.8, 20.1, 0.66, &[1.0, 0.24, 0.09]);
    }
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.145,
        104.0,
        2_700.0,
        4_120.0,
        26.0,
        85.0,
        0.56,
        &[1.0, 0.26, 0.10],
    );
    add_bird_chirp_harmonics(
        &mut samples,
        sr,
        0.435,
        70.0,
        7_380.0,
        8_820.0,
        35.0,
        90.0,
        0.50,
        &[1.0, 0.20],
    );

    BioAcousticRecording {
        source_name: "LACI_SWTH_SAVS_WidebandRiparian_192kHz.wav".to_string(),
        sample_rate_hz: sr,
        raw_header_rate_hz: sr,
        channels: 1,
        encoding: SampleEncoding::PcmI16,
        samples,
        guano: Some(base_guano(
            sr,
            dur_s,
            "Open Acoustic Devices",
            "AudioMoth 192k Wideband",
            44.5646,
            -123.2620,
            &["LACI", "SWTH", "SAVS"],
            "Simultaneous 192 kHz nocturnal aeroecology capture of Hoary Bat QCF pulses and passerine flight calls",
        )),
        riff_info: BTreeMap::new(),
    }
}
