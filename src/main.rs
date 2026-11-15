#![forbid(unsafe_code)]
//! # SonoWing (`sonowing` v1.1.0) Entry Point
//!
//! Launches the native `eframe`/`egui` 0.31 desktop bioacoustic workstation when invoked
//! without a subcommand, or executes headless inspection, batch scanning, ultrasonic
//! transcode, live microphone capture (`record-mic`), or preset synthesis via `clap`.

use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use sonowing::call_extractor::{extract_calls, DetectionMode, DetectorConfig, TaxonDomain};
use sonowing::dsp::{compute_stft, StftConfig};
use sonowing::exporters::{write_calls_csv, write_raven_selection_table};
use sonowing::gui::{load_any_bioacoustic_path, SonoWingApp};
use sonowing::harmonic_tracker::analyze_harmonic_stacks;
use sonowing::live_mic::LiveMicEngine;
use sonowing::mfcc_cluster::{analyze_repertoire_clusters, ManifoldClusterConfig};
use sonowing::soundscape_indices::compute_soundscape_indices;
use sonowing::storage::SonoWingRepository;
use sonowing::synth_presets::{synthesize_preset, BuiltinPreset};
use sonowing::ultrasonic_audio::{render_audible_recording, AuditionMode};
use sonowing::wav_guano::write_wav_with_guano;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "sonowing",
    version = "1.1.0",
    about = "Native safe-Rust bat echolocation and bird vocalization sonic/ultrasonic waveform, MFCC/UMAP, harmonic stack & live mic analyzer"
)]
struct Cli {
    /// Optional WAV or ZC file to open immediately in the GUI when no subcommand is given.
    #[arg(long)]
    open: Option<PathBuf>,

    /// Bypass the Startup Calibration Hub and enter the spectrogram workstation directly.
    #[arg(long)]
    skip_hub: bool,

    /// Path to the SQLite WAL session repository.
    #[arg(long, default_value = "sonowing_sessions.sqlite3")]
    db: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect a single .wav or .zc recording headlessly (STFT + call morphometrics + harmonics + MFCC/UMAP + ecoacoustics).
    Inspect {
        /// Input .wav (8..500 kHz) or .zc file path.
        file: PathBuf,
        /// Trigger threshold in dB above band noise baseline.
        #[arg(long, default_value_t = 12.0)]
        snr_db: f32,
        /// Optional output path for Cornell Lab Raven Pro 1.6 selection table (.selections.txt).
        #[arg(long)]
        raven_out: Option<PathBuf>,
        /// Optional output path for NABat/Avian RFC 4180 summary CSV.
        #[arg(long)]
        csv_out: Option<PathBuf>,
        /// Persist inspected session to SQLite WAL database.
        #[arg(long)]
        save_slot: Option<String>,
    },

    /// Parallel (`rayon`) batch scan of all .wav and .zc recordings in a directory.
    Batch {
        /// Input directory containing .wav or .zc field recordings.
        dir: PathBuf,
        /// Output directory for per-file Raven selection tables and combined batch summary CSV.
        #[arg(long, default_value = "sonowing_batch_out")]
        out_dir: PathBuf,
        /// Trigger SNR threshold in dB.
        #[arg(long, default_value_t = 12.0)]
        snr_db: f32,
    },

    /// Capture live microphone audio (hardware USB/UAC2 via `cpal` or simulated stream) directly to a GUANO WAV file.
    RecordMic {
        /// Duration to capture in seconds.
        #[arg(long, default_value_t = 3.0)]
        seconds: f64,
        /// Output .wav file path.
        #[arg(long, short)]
        output: PathBuf,
        /// Prefer highest USB UAC2 ultrasonic sample rate (up to 384 kHz).
        #[arg(long, default_value_t = true)]
        ultrasonic: bool,
        /// Use simulated 192 kHz live nocturnal stream instead of hardware microphone.
        #[arg(long)]
        simulated: bool,
    },

    /// Convert an ultrasonic bat recording into a human-audible 48 kHz WAV file with GUANO v1.0 metadata.
    Transcode {
        /// Source ultrasonic .wav file.
        input: PathBuf,
        /// Destination audible .wav file.
        #[arg(long, short)]
        output: PathBuf,
        /// Auditioning mode: `te10`, `te20`, `heterodyne`, or `fdiv`.
        #[arg(long, value_enum, default_value_t = CliAuditionMode::Te10)]
        mode: CliAuditionMode,
        /// Local oscillator carrier frequency in kHz (used when `--mode heterodyne`).
        #[arg(long, default_value_t = 41.0)]
        carrier_khz: f64,
        /// Zero-crossing division ratio (used when `--mode fdiv`).
        #[arg(long, default_value_t = 10)]
        div_ratio: u8,
    },

    /// Generate a calibrated synthetic bat or bird WAV file (`B1..B4`, `A1..A4`, `M1`) with embedded GUANO v1.0 metadata.
    SynthDemo {
        /// Preset code (`B1`, `B2`, `B3`, `B4`, `A1`, `A2`, `A3`, `A4`, or `M1`).
        #[arg(long, default_value = "B1")]
        preset: String,
        /// Output .wav path.
        #[arg(long, short)]
        output: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CliAuditionMode {
    Te10,
    Te20,
    Heterodyne,
    Fdiv,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        None => launch_gui(cli.open, cli.skip_hub, cli.db),
        Some(Commands::Inspect {
            file,
            snr_db,
            raven_out,
            csv_out,
            save_slot,
        }) => {
            run_inspect(
                &file,
                snr_db,
                raven_out.as_deref(),
                csv_out.as_deref(),
                save_slot.as_deref(),
                &cli.db,
            )?;
            Ok(())
        }
        Some(Commands::Batch {
            dir,
            out_dir,
            snr_db,
        }) => {
            run_batch(&dir, &out_dir, snr_db)?;
            Ok(())
        }
        Some(Commands::RecordMic {
            seconds,
            output,
            ultrasonic,
            simulated,
        }) => {
            let dur = seconds.clamp(0.5, 30.0);
            let mut engine = LiveMicEngine::new();
            if simulated {
                let lbl = engine.start_simulated_stream(dur + 0.5, ultrasonic);
                println!("[SonoWing Mic] Capturing simulated stream ({lbl}) for {dur:.1} s...");
                let steps = ((dur / 0.05).ceil() as usize).max(1);
                for _ in 0..steps {
                    engine.pump_if_simulated(0.05);
                }
            } else {
                let desc = engine.start_hardware_capture(dur + 0.5, ultrasonic)?;
                println!("[SonoWing Mic] Capturing hardware device ({desc}) for {dur:.1} s...");
                thread::sleep(Duration::from_secs_f64(dur));
            }
            let rec = engine
                .freeze_to_recording()
                .ok_or("Microphone capture buffer returned insufficient samples")?;
            engine.stop();
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            write_wav_with_guano(&output, &rec.samples, rec.sample_rate_hz, rec.guano.as_ref())?;
            println!(
                "[SonoWing Mic] Saved {:.2} s @ {} Hz -> {}",
                rec.duration_s(),
                rec.sample_rate_hz,
                output.display()
            );
            Ok(())
        }
        Some(Commands::Transcode {
            input,
            output,
            mode,
            carrier_khz,
            div_ratio,
        }) => {
            let rec = load_any_bioacoustic_path(&input)?;
            let aud_mode = match mode {
                CliAuditionMode::Te10 => AuditionMode::TimeExpansion10x,
                CliAuditionMode::Te20 => AuditionMode::TimeExpansion20x,
                CliAuditionMode::Heterodyne => AuditionMode::Heterodyne { carrier_khz },
                CliAuditionMode::Fdiv => AuditionMode::FrequencyDivision { ratio: div_ratio },
            };
            let audible = render_audible_recording(&rec, aud_mode, 48_000);
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            write_wav_with_guano(
                &output,
                &audible.samples,
                audible.sample_rate_hz,
                audible.guano.as_ref(),
            )?;
            println!(
                "[SonoWing] Wrote playable {} Hz WAV ({:.3} s, {}) -> {}",
                audible.sample_rate_hz,
                audible.duration_s(),
                aud_mode.label(),
                output.display()
            );
            Ok(())
        }
        Some(Commands::SynthDemo { preset, output }) => {
            let Some(p) = BuiltinPreset::from_code(&preset) else {
                return Err(format!(
                    "Unknown preset '{}'. Valid codes: B1, B2, B3, B4, A1, A2, A3, A4, M1",
                    preset
                )
                .into());
            };
            let rec = synthesize_preset(p);
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            write_wav_with_guano(&output, &rec.samples, rec.sample_rate_hz, rec.guano.as_ref())?;
            println!(
                "[SonoWing] Synthesized preset {} ('{}', {} Hz, {:.2} s) -> {}",
                p.code(),
                p.title(),
                rec.sample_rate_hz,
                rec.duration_s(),
                output.display()
            );
            Ok(())
        }
    }
}

fn launch_gui(
    initial_file: Option<PathBuf>,
    skip_hub: bool,
    db_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 920.0])
            .with_min_inner_size([1080.0, 700.0])
            .with_title(
                "SonoWing v1.1.0 — Bat & Bird Bioacoustic Waveform, MFCC/UMAP & Harmonic Analyzer",
            ),
        ..Default::default()
    };

    eframe::run_native(
        "SonoWing v1.1.0",
        native_options,
        Box::new(move |cc| Ok(Box::new(SonoWingApp::new(cc, initial_file, skip_hub, db_path)))),
    )
    .map_err(|e| format!("GUI error: {e}").into())
}

fn run_inspect(
    file: &Path,
    snr_db: f32,
    raven_out: Option<&Path>,
    csv_out: Option<&Path>,
    save_slot: Option<&str>,
    db_path: &Path,
) -> Result<(), String> {
    let rec = load_any_bioacoustic_path(file)?;
    let stft_cfg = StftConfig {
        fft_size: if rec.sample_rate_hz >= 160_000 { 512 } else { 1024 },
        hop_size: if rec.sample_rate_hz >= 160_000 { 64 } else { 256 },
        ..Default::default()
    };
    let det_cfg = DetectorConfig {
        mode: DetectionMode::AutoWideband,
        snr_threshold_db: snr_db,
        ..Default::default()
    };

    let spec = compute_stft(&rec.samples, rec.sample_rate_hz, stft_cfg);
    let calls = extract_calls(&rec.samples, &spec, det_cfg);
    let harmonics = analyze_harmonic_stacks(&calls, &spec);
    let clusters = analyze_repertoire_clusters(&calls, &spec, ManifoldClusterConfig::default());
    let idx = compute_soundscape_indices(&rec.samples, &spec);

    let bat_cnt = calls
        .iter()
        .filter(|c| c.domain == TaxonDomain::BatChiroptera)
        .count();
    let bird_cnt = calls
        .iter()
        .filter(|c| c.domain == TaxonDomain::BirdAves)
        .count();

    println!("========================================================================");
    println!("SonoWing v1.1.0 Bioacoustic Inspection Report");
    println!("Source      : {}", rec.source_name);
    println!(
        "Audio Format: {} | {} Hz (Nyquist {:.1} kHz) | {:.3} s | Peak {:.1} dBFS",
        rec.encoding.label(),
        rec.sample_rate_hz,
        rec.nyquist_hz() * 0.001,
        rec.duration_s(),
        rec.peak_dbfs()
    );
    println!(
        "Detections  : {} total ({} Bat pulses, {} Bird calls) | Optimal MFCC/UMAP K*={} (Sil={:.3})",
        calls.len(),
        bat_cnt,
        bird_cnt,
        clusters.optimal_k,
        clusters.global_silhouette
    );
    println!(
        "Soundscape  : ACI={:.2} | NDSI={:+.4} | BIO={:.2} | ADI={:.3} | AEI={:.3} | H={:.4}",
        idx.aci, idx.ndsi, idx.bio_index, idx.adi, idx.aei, idx.total_entropy_h
    );
    println!("------------------------------------------------------------------------");
    for (i, ev) in calls.iter().take(25).enumerate() {
        let cid = clusters
            .points
            .get(i)
            .map(|p| format!("C{}", p.cluster_id + 1))
            .unwrap_or_else(|| "C1".into());
        let (f0_k, h_cnt, hnr) = harmonics
            .get(i)
            .map(|h| (h.fundamental_f0_khz, h.active_harmonic_count, h.hnr_db))
            .unwrap_or((ev.f_char_khz, 1, 0.0));
        println!(
            "#{:<3} {:<3} {:<16} {:>5.3}-{:<5.3}s ({:>5.1} ms) Fc={:>6.2} f0={:>6.2}kHz ({}h, {:>+4.1}dB) -> {:<4} ({:>3.0}%)",
            ev.id,
            cid,
            ev.domain.label(),
            ev.begin_time_s,
            ev.end_time_s,
            ev.duration_ms,
            ev.f_char_khz,
            f0_k,
            h_cnt,
            hnr,
            ev.primary_code(),
            ev.primary_confidence() * 100.0,
        );
    }

    if let Some(rp) = raven_out {
        write_raven_selection_table(rp, &rec.source_name, &calls, &harmonics, &clusters)
            .map_err(|e| format!("Failed writing Raven table: {e}"))?;
        println!("Wrote Raven Pro selection table -> {}", rp.display());
    }
    if let Some(cp) = csv_out {
        write_calls_csv(
            cp,
            &rec.source_name,
            rec.sample_rate_hz,
            &calls,
            &harmonics,
            &clusters,
            &idx,
        )
        .map_err(|e| format!("Failed writing CSV: {e}"))?;
        println!("Wrote NABat/Avian summary CSV   -> {}", cp.display());
    }
    if let Some(slot) = save_slot {
        let repo = SonoWingRepository::open(db_path)?;
        let row_id = repo.save_session(
            slot,
            &rec.source_name,
            rec.sample_rate_hz,
            rec.duration_s(),
            rec.guano.as_ref(),
            &calls,
            &harmonics,
            &clusters,
            &idx,
        )?;
        println!(
            "Saved SQLite WAL v1.1.0 session #{row_id} ('{slot}') -> {}",
            db_path.display()
        );
    }
    Ok(())
}

fn run_batch(dir: &Path, out_dir: &Path, snr_db: f32) -> Result<(), String> {
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("Cannot create output folder '{}': {e}", out_dir.display()))?;

    let entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("Cannot read input folder '{}': {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "wav" | "zc" | "zcf"))
                .unwrap_or(false)
        })
        .collect();

    if entries.is_empty() {
        return Err(format!("No .wav or .zc files found inside '{}'", dir.display()));
    }

    let results: Vec<Result<(String, u32, usize, usize, usize, f64, f64), String>> = entries
        .par_iter()
        .map(|path| {
            let rec = load_any_bioacoustic_path(path)?;
            let stft_cfg = StftConfig {
                fft_size: if rec.sample_rate_hz >= 160_000 { 512 } else { 1024 },
                hop_size: if rec.sample_rate_hz >= 160_000 { 64 } else { 256 },
                ..Default::default()
            };
            let det_cfg = DetectorConfig {
                mode: DetectionMode::AutoWideband,
                snr_threshold_db: snr_db,
                ..Default::default()
            };
            let spec = compute_stft(&rec.samples, rec.sample_rate_hz, stft_cfg);
            let calls = extract_calls(&rec.samples, &spec, det_cfg);
            let harmonics = analyze_harmonic_stacks(&calls, &spec);
            let clusters =
                analyze_repertoire_clusters(&calls, &spec, ManifoldClusterConfig::default());
            let idx = compute_soundscape_indices(&rec.samples, &spec);

            let stem = rec
                .source_name
                .rsplit_once('.')
                .map(|(s, _)| s)
                .unwrap_or(&rec.source_name);
            let raven_p = out_dir.join(format!("{stem}.Table.1.selections.txt"));
            let csv_p = out_dir.join(format!("{stem}_calls.csv"));
            write_raven_selection_table(&raven_p, &rec.source_name, &calls, &harmonics, &clusters)
                .map_err(|e| e.to_string())?;
            write_calls_csv(
                &csv_p,
                &rec.source_name,
                rec.sample_rate_hz,
                &calls,
                &harmonics,
                &clusters,
                &idx,
            )
            .map_err(|e| e.to_string())?;

            let bat_n = calls
                .iter()
                .filter(|c| c.domain == TaxonDomain::BatChiroptera)
                .count();
            let bird_n = calls
                .iter()
                .filter(|c| c.domain == TaxonDomain::BirdAves)
                .count();
            Ok((
                rec.source_name,
                rec.sample_rate_hz,
                bat_n,
                bird_n,
                clusters.optimal_k,
                idx.aci,
                idx.ndsi,
            ))
        })
        .collect();

    let mut manifest = String::from(
        "file_name,sample_rate_hz,bat_pulses,bird_calls,total_events,optimal_k_clusters,aci,ndsi\r\n",
    );
    let mut total_files = 0usize;
    let mut total_bats = 0usize;
    let mut total_birds = 0usize;

    for item in results {
        let (name, sr, bn, an, opt_k, aci, ndsi) = item?;
        total_files += 1;
        total_bats += bn;
        total_birds += an;
        manifest.push_str(&format!(
            "{},{},{},{},{},{},{:.2},{:+.4}\r\n",
            name,
            sr,
            bn,
            an,
            bn + an,
            opt_k,
            aci,
            ndsi
        ));
    }
    manifest.push_str(&format!(
        "SUMMARY_BATCH,0,{},{},{},0,0.00,+0.0000\r\n",
        total_bats,
        total_birds,
        total_bats + total_birds
    ));

    let manifest_path = out_dir.join("batch_manifest_summary.csv");
    fs::write(&manifest_path, manifest).map_err(|e| e.to_string())?;
    println!(
        "[SonoWing Batch v1.1.0] Processed {total_files} files ({total_bats} bat pulses, {total_birds} bird calls) -> {}",
        manifest_path.display()
    );
    Ok(())
}
