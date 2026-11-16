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

#![forbid(unsafe_code)]
//! Integration & analytical verification suite for SonoWing v1.1.0.

use sonowing::call_extractor::{
    extract_calls, BehavioralPhase, CallShapeTopology, DetectionMode, DetectorConfig, TaxonDomain,
};
use sonowing::dsp::{
    bandpass_zero_phase, compute_stft, compute_tkeo, extract_zero_crossing_dots,
    parseval_ratio_check, StftConfig,
};
use sonowing::exporters::{
    format_calls_and_summary_csv, format_raven_selection_table, SignedWorksetEnvelope,
    WorksetPayload,
};
use sonowing::harmonic_tracker::analyze_harmonic_stacks;
use sonowing::live_mic::LiveMicEngine;
use sonowing::mfcc_cluster::{
    analyze_repertoire_clusters, ClusterCountMode, ManifoldClusterConfig,
};
use sonowing::soundscape_indices::compute_soundscape_indices;
use sonowing::storage::SonoWingRepository;
use sonowing::synth_presets::{synthesize_preset, BuiltinPreset};
use sonowing::ultrasonic_audio::{render_audible_recording, AuditionMode};
use sonowing::wav_guano::{encode_wav_with_guano_bytes, parse_wav_bytes};
use std::f64::consts::TAU;

#[test]
fn fft_parseval_energy_conservation_and_tkeo_sinusoid() {
    let n = 1024usize;
    let sig: Vec<f64> = (0..n)
        .map(|i| 0.75 * (TAU * 42_000.0 * (i as f64) / 250_000.0).sin())
        .collect();
    let ratio = parseval_ratio_check(&sig);
    assert!(
        (ratio - 1.0).abs() < 1e-9,
        "Parseval energy ratio deviated: {ratio}"
    );

    let sig_f32: Vec<f32> = sig.iter().map(|&v| v as f32).collect();
    let psi = compute_tkeo(&sig_f32);
    let omega = TAU * 42_000.0 / 250_000.0;
    let expected = (0.75 * 0.75) * omega.sin().powi(2);
    let mid_psi = psi[512] as f64;
    assert!(
        (mid_psi - expected).abs() < 0.015,
        "TKEO mid-sample {mid_psi} vs expected {expected}"
    );
}

#[test]
fn wav_guano_v1_roundtrip_preserves_metadata_and_samples() {
    let preset = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
    let bytes = encode_wav_with_guano_bytes(
        &preset.samples,
        preset.sample_rate_hz,
        preset.guano.as_ref(),
    );
    let decoded = parse_wav_bytes(&bytes, "roundtrip.wav").expect("WAV decode should succeed");
    assert_eq!(decoded.sample_rate_hz, 250_000);
    assert_eq!(decoded.samples.len(), preset.samples.len());
    let g = decoded.guano.expect("GUANO chunk must be decoded");
    assert_eq!(g.version, "1.0");
    assert_eq!(g.species_auto_id, vec!["MYLU".to_string()]);
    assert!(g.loc_position.is_some());
}

#[test]
fn myotis_lucifugus_feeding_buzz_and_rhinolophus_cf_topology() {
    let rec = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
    let spec = compute_stft(
        &rec.samples,
        rec.sample_rate_hz,
        StftConfig {
            fft_size: 512,
            hop_size: 64,
            ..Default::default()
        },
    );
    let calls = extract_calls(&rec.samples, &spec, DetectorConfig::default());
    assert!(
        calls.len() >= 14,
        "Expected search + approach + feeding buzz pulses, got {}",
        calls.len()
    );
    assert_eq!(calls[0].primary_code(), "MYLU");
    assert!(
        calls
            .iter()
            .any(|c| c.phase == BehavioralPhase::TerminalFeedingBuzz),
        "Terminal feeding buzz phase must be detected"
    );

    let rhfe = synthesize_preset(BuiltinPreset::B4RhfeHorseshoeCf);
    let spec_rh = compute_stft(
        &rhfe.samples,
        rhfe.sample_rate_hz,
        StftConfig {
            fft_size: 512,
            hop_size: 64,
            ..Default::default()
        },
    );
    let calls_rh = extract_calls(&rhfe.samples, &spec_rh, DetectorConfig::default());
    assert!(!calls_rh.is_empty());
    assert_eq!(calls_rh[0].topology, CallShapeTopology::FmCfFm);
    assert_eq!(calls_rh[0].primary_code(), "RHFE");
}

#[test]
fn mfcc_umap_clustering_and_harmonic_stack_tracking_v11() {
    // 1. Test MFCC + UMAP clustering on sympatric NFC passage (SWTH + YRWA)
    let nfc_rec = synthesize_preset(BuiltinPreset::A1SwthYrwaNightFlight);
    let spec_a1 = compute_stft(&nfc_rec.samples, nfc_rec.sample_rate_hz, StftConfig::default());
    let calls_a1 = extract_calls(&nfc_rec.samples, &spec_a1, DetectorConfig::default());
    let report = analyze_repertoire_clusters(
        &calls_a1,
        &spec_a1,
        ManifoldClusterConfig {
            k_mode: ClusterCountMode::Fixed(2),
            umap_n_neighbors: 3,
            umap_min_dist: 0.12,
        },
    );
    assert_eq!(report.optimal_k, 2);
    assert_eq!(report.points.len(), calls_a1.len());
    assert!(
        report.global_silhouette > 0.10,
        "Expected clear SWTH vs YRWA MFCC cluster separation, got silhouette {}",
        report.global_silhouette
    );
    for pt in &report.points {
        assert!(pt.umap_x.abs() <= 1.0 && pt.umap_y.abs() <= 1.0);
    }

    // 2. Test Subharmonic Summation Harmonic Stack tracker on Whisper Bat (COTO) & Avian Stack (A4)
    let coto = synthesize_preset(BuiltinPreset::B3CotoWhisperHarmonic);
    let spec_b3 = compute_stft(
        &coto.samples,
        coto.sample_rate_hz,
        StftConfig {
            fft_size: 512,
            hop_size: 64,
            ..Default::default()
        },
    );
    let calls_b3 = extract_calls(&coto.samples, &spec_b3, DetectorConfig::default());
    let harm_b3 = analyze_harmonic_stacks(&calls_b3, &spec_b3);
    assert!(!harm_b3.is_empty());
    assert_eq!(harm_b3[0].dominant_harmonic_order, 2);
    assert!(harm_b3[0].active_harmonic_count >= 2);

    let a4 = synthesize_preset(BuiltinPreset::A4VathHarmonicStack);
    let spec_a4 = compute_stft(&a4.samples, a4.sample_rate_hz, StftConfig::default());
    let calls_a4 = extract_calls(&a4.samples, &spec_a4, DetectorConfig::default());
    let harm_a4 = analyze_harmonic_stacks(&calls_a4, &spec_a4);
    assert!(
        harm_a4.iter().any(|h| h.active_harmonic_count >= 3),
        "Expected multi-harmonic stack on A4"
    );

    // 3. Test LiveMicEngine simulated rolling buffer & freeze
    let mut mic = LiveMicEngine::new();
    mic.start_simulated_stream(1.5, true);
    mic.pump_if_simulated(0.10);
    let frozen = mic.freeze_to_recording().expect("freeze live buffer");
    assert_eq!(frozen.sample_rate_hz, 192_000);
    assert!(frozen.samples.len() > 10_000);
    mic.stop();
}

#[test]
fn exporters_v11_raven_csv_workset_and_sqlite_persistence() {
    let rec = synthesize_preset(BuiltinPreset::M1RiparianBatAndBirdDual);
    let filtered = bandpass_zero_phase(&rec.samples, 1_500.0, 60_000.0, rec.sample_rate_hz);
    let zc = extract_zero_crossing_dots(&filtered, rec.sample_rate_hz, 8, 0.05, 1_000.0, 60_000.0);
    assert!(!zc.is_empty());

    let spec = compute_stft(
        &filtered,
        rec.sample_rate_hz,
        StftConfig {
            fft_size: 512,
            hop_size: 64,
            ..Default::default()
        },
    );
    let calls = extract_calls(
        &filtered,
        &spec,
        DetectorConfig {
            mode: DetectionMode::AutoWideband,
            ..Default::default()
        },
    );
    let harmonics = analyze_harmonic_stacks(&calls, &spec);
    let clusters = analyze_repertoire_clusters(&calls, &spec, ManifoldClusterConfig::default());
    let indices = compute_soundscape_indices(&filtered, &spec);

    let raven_tsv = format_raven_selection_table(&rec.source_name, &calls, &harmonics, &clusters);
    assert!(raven_tsv.starts_with(
        "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)"
    ));
    assert!(raven_tsv.contains("MFCC Cluster"));

    let csv = format_calls_and_summary_csv(
        &rec.source_name,
        rec.sample_rate_hz,
        &calls,
        &harmonics,
        &clusters,
        &indices,
    );
    assert!(csv.contains("SUMMARY_CLUSTER"));

    let te10 = render_audible_recording(&rec, AuditionMode::TimeExpansion10x, 48_000);
    assert_eq!(te10.sample_rate_hz, 48_000);

    let payload = WorksetPayload {
        schema_version: "1.1.0".to_string(),
        source_name: rec.source_name.clone(),
        sample_rate_hz: rec.sample_rate_hz,
        duration_s: rec.duration_s(),
        guano: rec.guano.clone(),
        detected_calls: calls.clone(),
        harmonic_profiles: harmonics.clone(),
        cluster_report: clusters.clone(),
        soundscape_indices: indices.clone(),
    };
    let env = SignedWorksetEnvelope::build(payload).expect("envelope build");
    assert!(env.verify().expect("envelope verify"));

    let tmp_db = std::env::temp_dir().join("sonowing_v1_1_unit_test.sqlite3");
    let _ = std::fs::remove_file(&tmp_db);
    let repo = SonoWingRepository::open(&tmp_db).expect("open sqlite");
    let id = repo
        .save_session(
            "UnitTestSlotV11",
            &rec.source_name,
            rec.sample_rate_hz,
            rec.duration_s(),
            rec.guano.as_ref(),
            &calls,
            &harmonics,
            &clusters,
            &indices,
        )
        .expect("save session");
    let loaded = repo.load_session_payload(id).expect("load session");
    assert_eq!(loaded.detected_calls.len(), calls.len());
    assert_eq!(loaded.harmonic_profiles.len(), harmonics.len());
    let _ = std::fs::remove_file(&tmp_db);
}
