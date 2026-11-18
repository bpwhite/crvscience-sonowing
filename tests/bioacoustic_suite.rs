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
    extract_calls, BehavioralPhase, CallShapeTopology, DetectionMode, DetectorConfig,
};
use sonowing::dsp::{
    bandpass_zero_phase, compute_stft, compute_tkeo, extract_zero_crossing_dots,
    parseval_ratio_check, StftConfig,
};
use sonowing::exporters::{
    format_calls_and_summary_csv, format_raven_selection_table, SignedWorksetEnvelope,
    WorksetPayload,
};
use sonowing::gui::{AppStage, GuanoEditorDraft, SonoWingApp};
use sonowing::harmonic_tracker::analyze_harmonic_stacks;
use sonowing::live_mic::LiveMicEngine;
use sonowing::mfcc_cluster::{
    analyze_repertoire_clusters, ClusterCountMode, ManifoldClusterConfig,
};
use sonowing::soundscape_indices::compute_soundscape_indices;
use sonowing::storage::SonoWingRepository;
use sonowing::synth_presets::{synthesize_preset, BuiltinPreset};
use sonowing::ultrasonic_audio::{render_audible_recording, AuditionMode};
use sonowing::wav_guano::{encode_wav_with_guano_bytes, load_wav_file, parse_wav_bytes};
use std::f64::consts::TAU;
use std::path::PathBuf;

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
        source_path: None,
        preset_code: Some("M1".to_string()),
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

#[test]
fn demo_mylu_wav_classifies_as_mylu_without_subharmonic_halving() {
    let demo_path = PathBuf::from("samples/MYLU_Willamette_Demo_250kHz.wav");
    assert!(demo_path.exists(), "Demo WAV must exist at {}", demo_path.display());
    let rec = load_wav_file(&demo_path).expect("load demo WAV");
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
        calls.len() >= 5,
        "Expected >= 5 pulses in demo WAV, got {}",
        calls.len()
    );
    for (idx, ev) in calls.iter().take(3).enumerate() {
        assert!(
            (37.0..=45.0).contains(&ev.f_char_khz),
            "Call #{} Fc should be ~40 kHz, got {}",
            idx + 1,
            ev.f_char_khz
        );
        assert!(
            ev.f_min_khz >= 35.0,
            "Call #{} Fmin should remain >=35 kHz, got {}",
            idx + 1,
            ev.f_min_khz
        );
        assert_eq!(ev.primary_code(), "MYLU");
    }
    assert!(
        calls.iter().all(|c| c.primary_code() != "COTO"),
        "Zero calls in MYLU demo WAV should misclassify as COTO"
    );
}

#[test]
fn workset_json_roundtrip_and_tamper_rejection() {
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
    let harmonics = analyze_harmonic_stacks(&calls, &spec);
    let clusters = analyze_repertoire_clusters(&calls, &spec, ManifoldClusterConfig::default());
    let indices = compute_soundscape_indices(&rec.samples, &spec);

    let payload = WorksetPayload {
        schema_version: "1.1.0".to_string(),
        source_name: rec.source_name.clone(),
        source_path: None,
        preset_code: Some("B1".to_string()),
        sample_rate_hz: rec.sample_rate_hz,
        duration_s: rec.duration_s(),
        guano: rec.guano.clone(),
        detected_calls: calls.clone(),
        harmonic_profiles: harmonics.clone(),
        cluster_report: clusters.clone(),
        soundscape_indices: indices.clone(),
    };
    let env = SignedWorksetEnvelope::build(payload).expect("build envelope");
    let ws_path = std::env::temp_dir().join("sonowing_test_roundtrip.sonoworkset.json");
    let _ = std::fs::remove_file(&ws_path);
    env.save_to_file(&ws_path).expect("save workset");

    let loaded = SignedWorksetEnvelope::load_verified(&ws_path).expect("verify roundtrip");
    assert_eq!(loaded.payload.detected_calls.len(), calls.len());
    assert_eq!(loaded.payload.harmonic_profiles.len(), harmonics.len());
    assert_eq!(loaded.payload.cluster_report.optimal_k, clusters.optimal_k);

    // Mutate one field in the JSON without recomputing sha256_hex
    let raw = std::fs::read_to_string(&ws_path).expect("read json");
    let tampered = raw.replacen("MYLU", "COTO", 1);
    assert_ne!(raw, tampered);
    std::fs::write(&ws_path, tampered).expect("write tampered json");
    let err = SignedWorksetEnvelope::load_verified(&ws_path).unwrap_err();
    assert!(
        err.contains("SHA-256"),
        "Expected SHA-256 tamper error, got: {err}"
    );
    let _ = std::fs::remove_file(&ws_path);
}

#[test]
fn guano_editor_draft_validation_and_wav_roundtrip() {
    let rec = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
    let mut draft =
        GuanoEditorDraft::from_guano(rec.guano.as_ref(), rec.sample_rate_hz, rec.duration_s());
    draft.species_manual_id = "MYLU, MYEV".to_string();
    draft.loc_lat = "44.564566".to_string();
    draft.loc_lon = "-123.262044".to_string();
    draft.make = "Pettersson".to_string();
    draft.model = "M500-384".to_string();
    draft.note = "Riparian roost exit\n250 kHz full spectrum".to_string();

    let edited_guano = draft.to_guano(rec.duration_s()).expect("valid guano draft");
    let wav_bytes =
        encode_wav_with_guano_bytes(&rec.samples, rec.sample_rate_hz, Some(&edited_guano));
    let decoded = parse_wav_bytes(&wav_bytes, "edited.wav").expect("decode edited wav");
    let g = decoded.guano.expect("decoded guano");
    assert_eq!(g.species_manual_id, vec!["MYLU", "MYEV"]);
    assert_eq!(g.make.as_deref(), Some("Pettersson"));
    assert_eq!(g.model.as_deref(), Some("M500-384"));
    assert_eq!(
        g.note.as_deref(),
        Some("Riparian roost exit\n250 kHz full spectrum")
    );
    let (lat, lon) = g.loc_position.expect("loc position");
    assert!((lat - 44.564566).abs() < 1e-5 && (lon + 123.262044).abs() < 1e-5);

    // Invalid bounds must fail cleanly
    let mut bad_lat = draft.clone();
    bad_lat.loc_lat = "95.0".to_string();
    assert!(bad_lat.to_guano(rec.duration_s()).is_err());

    let mut bad_sr = draft;
    bad_sr.samplerate_hz = "not_a_number".to_string();
    assert!(bad_sr.to_guano(rec.duration_s()).is_err());
}

#[test]
fn sqlite_session_reload_and_workset_import_recomputes_spectrogram_and_preserves_vetting() {
    let demo_wav = PathBuf::from("samples/MYLU_Willamette_Demo_250kHz.wav");
    assert!(demo_wav.exists());
    let tmp_db = std::env::temp_dir().join(format!(
        "sonowing_reload_test_{}.sqlite3",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp_db);

    let ctx = egui::Context::default();
    let mut app = SonoWingApp::new_headless(&ctx, Some(demo_wav), true, tmp_db.clone());
    assert_eq!(app.raw_recording.sample_rate_hz, 250_000);
    assert!(!app.detected_calls.is_empty());
    app.detected_calls[0].manual_vetted_code = Some("MYLU".to_string());
    app.slot_name_input = "Willamette Reload Slot".to_string();
    app.quick_save_sqlite();

    // Also save a signed workset for import test
    let ws_path = std::env::temp_dir().join(format!(
        "sonowing_import_test_{}.sonoworkset.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&ws_path);
    let ws_payload = WorksetPayload {
        schema_version: "1.1.0".to_string(),
        source_name: app.raw_recording.source_name.clone(),
        source_path: app
            .current_source_path
            .as_ref()
            .map(|p| p.display().to_string()),
        preset_code: None,
        sample_rate_hz: app.raw_recording.sample_rate_hz,
        duration_s: app.raw_recording.duration_s(),
        guano: app.raw_recording.guano.clone(),
        detected_calls: app.detected_calls.clone(),
        harmonic_profiles: app.harmonic_profiles.clone(),
        cluster_report: app.cluster_report.clone(),
        soundscape_indices: app.soundscape_indices.clone(),
    };
    SignedWorksetEnvelope::build(ws_payload)
        .unwrap()
        .save_to_file(&ws_path)
        .unwrap();

    // Switch to an unrelated 48 kHz avian preset
    app.load_preset(BuiltinPreset::A1SwthYrwaNightFlight);
    assert_eq!(app.raw_recording.sample_rate_hz, 48_000);
    assert_eq!(app.spectrogram.sample_rate_hz, 48_000);

    // Reload SQLite session #1
    app.load_sqlite_session(1);
    assert_eq!(app.raw_recording.sample_rate_hz, 250_000);
    assert_eq!(app.raw_recording.samples.len(), 70_000);
    assert_eq!(app.spectrogram.sample_rate_hz, 250_000);
    assert!(app.spec_dirty, "spec_dirty must be set on session reload");
    assert!(!app.has_spec_texture(), "stale spec_texture must be cleared");
    assert!((app.view_end_s - 0.280).abs() < 1e-3);
    assert!((app.freq_max_khz - 125.0).abs() < 0.5);
    assert_eq!(app.stage, AppStage::Workstation);
    assert_eq!(
        app.detected_calls[0].manual_vetted_code.as_deref(),
        Some("MYLU")
    );

    // Switch to 384 kHz RHFE preset, then import .sonoworkset.json
    app.load_preset(BuiltinPreset::B4RhfeHorseshoeCf);
    assert_eq!(app.raw_recording.sample_rate_hz, 384_000);
    app.import_workset_from_path(&ws_path);
    assert_eq!(app.raw_recording.sample_rate_hz, 250_000);
    assert_eq!(app.spectrogram.sample_rate_hz, 250_000);
    assert!(app.spec_dirty);
    assert_eq!(
        app.detected_calls[0].manual_vetted_code.as_deref(),
        Some("MYLU")
    );

    let _ = std::fs::remove_file(&ws_path);
    let _ = std::fs::remove_file(&tmp_db);
}

#[test]
fn sqlite_session_reload_missing_wav_surfaces_error_gracefully() {
    let rec = synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz);
    let tmp_wav = std::env::temp_dir().join(format!(
        "sonowing_ephemeral_missing_{}.wav",
        std::process::id()
    ));
    let wav_bytes = encode_wav_with_guano_bytes(&rec.samples, rec.sample_rate_hz, rec.guano.as_ref());
    std::fs::write(&tmp_wav, wav_bytes).expect("write temp wav");

    let tmp_db = std::env::temp_dir().join(format!(
        "sonowing_missing_wav_{}.sqlite3",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp_db);

    let ctx = egui::Context::default();
    let mut app = SonoWingApp::new_headless(&ctx, Some(tmp_wav.clone()), true, tmp_db.clone());
    app.quick_save_sqlite();

    // Delete the temp WAV and switch to 48 kHz preset
    std::fs::remove_file(&tmp_wav).expect("remove temp wav");
    app.load_preset(BuiltinPreset::A1SwthYrwaNightFlight);

    // Reloading should not panic, should clear old preset spectrogram, and surface missing WAV in status_banner
    app.load_sqlite_session(1);
    assert!(
        app.status_banner.contains("could not be reloaded")
            || app.status_banner.contains("does not exist"),
        "Expected missing-WAV warning in status_banner, got: {}",
        app.status_banner
    );
    assert_eq!(app.spectrogram.sample_rate_hz, 250_000);
    assert!(app.spec_dirty);

    let _ = std::fs::remove_file(&tmp_db);
}
