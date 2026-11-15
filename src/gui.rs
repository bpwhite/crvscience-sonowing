//! Native `eframe` / `egui` `0.31` Bioacoustic Sonic & Ultrasonic Workstation (`v1.1.0`):
//! - Startup Calibration Hub (`AppStage::StartupHub`, 9 presets, bypassable via `--skip-hub`)
//! - Live Microphone Spectrogram Stream (`cpal` USB/Ultrasonic UAC2 + Simulated 192 kHz stream + Freeze Buffer)
//! - Multi-Harmonic Stack Overtone Contours (`2f0..5f0`) & Inspector Harmonic Ladder (`f0`, `HNR dB`, Inharmonicity)
//! - Unsupervised Syllable/Pulse `MFCC` + `UMAP`-Style 2D Manifold Scatter & `k`-Means++ Silhouette Dock Tab

use crate::call_extractor::{
    extract_calls, BehavioralPhase, DetectedCallEvent, DetectionMode, DetectorConfig, TaxonDomain,
};
use crate::dsp::{
    bandpass_zero_phase, compute_stft, compute_tkeo, downsample_waveform_minmax,
    extract_zero_crossing_dots, Spectrogram, StftConfig, WindowFunction,
};
use crate::exporters::{
    write_calls_csv, write_raven_selection_table, SignedWorksetEnvelope, WorksetPayload,
};
use crate::harmonic_tracker::{analyze_harmonic_stacks, HarmonicStackProfile};
use crate::live_mic::LiveMicEngine;
use crate::mfcc_cluster::{
    analyze_repertoire_clusters, ClusterCountMode, ManifoldClusterConfig, RepertoireClusterReport,
};
use crate::soundscape_indices::{compute_soundscape_indices, SoundscapeIndices};
use crate::storage::{SavedSessionSummary, SonoWingRepository};
use crate::synth_presets::{synthesize_preset, BuiltinPreset};
use crate::taxa_library::reference_species_catalog;
use crate::ultrasonic_audio::{render_audible_recording, AuditionMode};
use crate::wav_guano::{
    load_wav_file, load_zc_dot_table, reconstruct_waveform_from_zc, write_wav_with_guano,
    write_zc_dot_table, BioAcousticRecording, ZeroCrossDot, ZeroCrossSequence,
};
use eframe::egui::{
    self, Color32, ColorImage, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStage {
    StartupHub,
    Workstation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColormapPalette {
    MagmaBio,
    ViridisField,
    InfernoContrast,
    GrayscalePublication,
}

impl ColormapPalette {
    pub const ALL: [Self; 4] = [
        Self::MagmaBio,
        Self::ViridisField,
        Self::InfernoContrast,
        Self::GrayscalePublication,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::MagmaBio => "Magma (Dark Nocturnal)",
            Self::ViridisField => "Viridis (Perceptual Uniform)",
            Self::InfernoContrast => "Inferno (High Ultrasonic SNR)",
            Self::GrayscalePublication => "Inverted Grayscale (Print)",
        }
    }

    pub fn map_unit(self, t_raw: f32) -> Color32 {
        let t = t_raw.clamp(0.0, 1.0);
        let stops: [(f32, [u8; 3]); 5] = match self {
            Self::MagmaBio => [
                (0.00, [8, 7, 24]),
                (0.28, [66, 20, 104]),
                (0.55, [158, 47, 100]),
                (0.80, [238, 118, 52]),
                (1.00, [252, 250, 191]),
            ],
            Self::ViridisField => [
                (0.00, [18, 12, 46]),
                (0.28, [49, 82, 139]),
                (0.55, [33, 145, 140]),
                (0.80, [94, 201, 98]),
                (1.00, [253, 231, 37]),
            ],
            Self::InfernoContrast => [
                (0.00, [6, 5, 18]),
                (0.28, [87, 16, 110]),
                (0.55, [188, 55, 84]),
                (0.80, [249, 142, 9]),
                (1.00, [252, 255, 164]),
            ],
            Self::GrayscalePublication => [
                (0.00, [246, 247, 249]),
                (0.30, [195, 198, 204]),
                (0.60, [118, 124, 134]),
                (0.85, [48, 52, 60]),
                (1.00, [10, 12, 16]),
            ],
        };

        for pair in stops.windows(2) {
            let (t0, c0) = pair[0];
            let (t1, c1) = pair[1];
            if t <= t1 {
                let u = ((t - t0) / (t1 - t0).max(1e-6)).clamp(0.0, 1.0);
                let r = (c0[0] as f32 + u * (c1[0] as f32 - c0[0] as f32)).round() as u8;
                let g = (c0[1] as f32 + u * (c1[1] as f32 - c0[1] as f32)).round() as u8;
                let b = (c0[2] as f32 + u * (c1[2] as f32 - c0[2] as f32)).round() as u8;
                return Color32::from_rgb(r, g, b);
            }
        }
        Color32::WHITE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BottomDockTab {
    CallsTable,
    MfccUmapClusters,
    ZeroCrossingTrace,
    SoundscapeIndices,
    AuditionAndExport,
}

const CLUSTER_COLORS: [Color32; 6] = [
    Color32::from_rgb(82, 214, 203),
    Color32::from_rgb(247, 185, 85),
    Color32::from_rgb(255, 107, 147),
    Color32::from_rgb(138, 162, 255),
    Color32::from_rgb(118, 235, 132),
    Color32::from_rgb(232, 132, 255),
];

pub struct SonoWingApp {
    pub stage: AppStage,
    pub active_preset: BuiltinPreset,
    pub raw_recording: BioAcousticRecording,
    pub filtered_samples: Vec<f32>,
    pub tkeo_trace: Vec<f32>,
    pub zc_dots: Vec<ZeroCrossDot>,
    pub spectrogram: Spectrogram,
    pub detected_calls: Vec<DetectedCallEvent>,
    pub harmonic_profiles: Vec<HarmonicStackProfile>,
    pub cluster_report: RepertoireClusterReport,
    pub soundscape_indices: SoundscapeIndices,

    // Live Microphone Capture Engine (v1.1.0)
    pub live_mic: LiveMicEngine,
    pub mic_prefer_ultrasonic: bool,
    pub mic_window_seconds: f64,
    last_live_refresh: Instant,

    // DSP, Clustering & Filter Controls
    pub stft_cfg: StftConfig,
    pub detector_cfg: DetectorConfig,
    pub cluster_cfg: ManifoldClusterConfig,
    pub apply_bandpass: bool,
    pub bandpass_low_khz: f64,
    pub bandpass_high_khz: f64,
    pub zc_division_ratio: u8,

    // Viewport & Display Controls
    pub palette: ColormapPalette,
    pub dynamic_range_db: f32,
    pub gain_offset_db: f32,
    pub view_start_s: f64,
    pub view_end_s: f64,
    pub freq_min_khz: f64,
    pub freq_max_khz: f64,
    pub show_ridge_overlay: bool,
    pub show_harmonics_overlay: bool,
    pub show_zc_overlay: bool,
    pub show_call_boxes: bool,
    pub show_tkeo_overlay: bool,
    pub cursor_frame: usize,
    pub selected_call_idx: Option<usize>,

    // Ultrasonic Audition & Export Controls
    pub heterodyne_carrier_khz: f64,
    pub freq_div_ratio: u8,
    pub file_path_input: String,
    pub export_dir_input: String,
    pub slot_name_input: String,
    pub status_banner: String,

    // Modals & SQLite state
    pub show_reference_modal: bool,
    pub show_sessions_modal: bool,
    pub recent_sessions: Vec<SavedSessionSummary>,
    db_path: PathBuf,
    spec_texture: Option<TextureHandle>,
    spec_dirty: bool,
}

impl SonoWingApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial_file: Option<PathBuf>,
        skip_hub: bool,
        db_path: PathBuf,
    ) -> Self {
        configure_dark_scientific_theme(&cc.egui_ctx);

        let init_rec = if let Some(ref p) = initial_file {
            load_any_bioacoustic_path(p)
                .unwrap_or_else(|_| synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz))
        } else {
            synthesize_preset(BuiltinPreset::B1MyluFeedingBuzz)
        };

        let stft_cfg = StftConfig::default();
        let detector_cfg = DetectorConfig::default();
        let cluster_cfg = ManifoldClusterConfig::default();
        let dur_s = init_rec.duration_s().max(0.05);
        let nyq_khz = (init_rec.nyquist_hz() * 0.001).max(4.0);

        let mut app = Self {
            stage: if skip_hub || initial_file.is_some() {
                AppStage::Workstation
            } else {
                AppStage::StartupHub
            },
            active_preset: BuiltinPreset::B1MyluFeedingBuzz,
            raw_recording: init_rec,
            filtered_samples: Vec::new(),
            tkeo_trace: Vec::new(),
            zc_dots: Vec::new(),
            spectrogram: Spectrogram {
                sample_rate_hz: 48_000,
                fft_size: 512,
                hop_size: 64,
                num_frames: 1,
                num_bins: 257,
                db_grid: vec![-96.0; 257],
                mag_grid: vec![0.0; 257],
                frame_peak_freq_hz: vec![0.0],
                frame_peak_dbfs: vec![-96.0],
                frame_centroid_hz: vec![0.0],
                frame_wiener_entropy: vec![1.0],
                bin_noise_floor_dbfs: vec![-96.0; 257],
            },
            detected_calls: Vec::new(),
            harmonic_profiles: Vec::new(),
            cluster_report: RepertoireClusterReport::default(),
            soundscape_indices: SoundscapeIndices {
                aci: 0.0,
                ndsi: 0.0,
                bio_index: 0.0,
                adi: 0.0,
                aei: 0.0,
                temporal_entropy_ht: 0.0,
                spectral_entropy_hf: 0.0,
                total_entropy_h: 0.0,
                ultrasonic_energy_fraction: 0.0,
                band_occupancy: Vec::new(),
            },
            live_mic: LiveMicEngine::new(),
            mic_prefer_ultrasonic: true,
            mic_window_seconds: 2.0,
            last_live_refresh: Instant::now(),
            stft_cfg,
            detector_cfg,
            cluster_cfg,
            apply_bandpass: false,
            bandpass_low_khz: 1.0,
            bandpass_high_khz: nyq_khz * 0.95,
            zc_division_ratio: 8,
            palette: ColormapPalette::MagmaBio,
            dynamic_range_db: 72.0,
            gain_offset_db: 0.0,
            view_start_s: 0.0,
            view_end_s: dur_s,
            freq_min_khz: 0.0,
            freq_max_khz: nyq_khz,
            show_ridge_overlay: true,
            show_harmonics_overlay: true,
            show_zc_overlay: false,
            show_call_boxes: true,
            show_tkeo_overlay: true,
            cursor_frame: 0,
            selected_call_idx: None,
            heterodyne_carrier_khz: 41.0,
            freq_div_ratio: 10,
            file_path_input: initial_file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            export_dir_input: "exports".to_string(),
            slot_name_input: "Willamette Survey v1.1".to_string(),
            status_banner: "SonoWing v1.1.0 Ready — MFCC/UMAP clustering, harmonic stack tracking, and live microphone capture active.".to_string(),
            show_reference_modal: false,
            show_sessions_modal: false,
            recent_sessions: Vec::new(),
            db_path,
            spec_texture: None,
            spec_dirty: true,
        };
        app.recompute_full_pipeline(true);
        app.refresh_sessions_list();
        app
    }

    pub fn load_preset(&mut self, preset: BuiltinPreset) {
        self.live_mic.stop();
        self.active_preset = preset;
        self.raw_recording = synthesize_preset(preset);
        self.apply_bandpass = false;
        self.recompute_full_pipeline(true);
        self.status_banner = format!(
            "Loaded preset {} — {} ({:.2} s @ {:.1} kHz, {} calls, K*={} clusters)",
            preset.code(),
            self.raw_recording.source_name,
            self.raw_recording.duration_s(),
            self.raw_recording.sample_rate_hz as f64 * 0.001,
            self.detected_calls.len(),
            self.cluster_report.optimal_k
        );
    }

    pub fn load_external_path(&mut self, path: &Path) {
        self.live_mic.stop();
        match load_any_bioacoustic_path(path) {
            Ok(rec) => {
                self.raw_recording = rec;
                self.apply_bandpass = false;
                self.recompute_full_pipeline(true);
                self.stage = AppStage::Workstation;
                self.status_banner = format!(
                    "Loaded '{}' ({:.3} s @ {} Hz, {} calls, K*={} clusters)",
                    self.raw_recording.source_name,
                    self.raw_recording.duration_s(),
                    self.raw_recording.sample_rate_hz,
                    self.detected_calls.len(),
                    self.cluster_report.optimal_k
                );
            }
            Err(err) => {
                self.status_banner = format!("Load error: {err}");
            }
        }
    }

    pub fn recompute_full_pipeline(&mut self, reset_viewport: bool) {
        let sr = self.raw_recording.sample_rate_hz;
        let nyq_khz = (self.raw_recording.nyquist_hz() * 0.001).max(1.0);

        if reset_viewport {
            if sr >= 160_000 {
                self.stft_cfg.fft_size = 512;
                self.stft_cfg.hop_size = 64;
                self.heterodyne_carrier_khz = 41.0;
            } else {
                self.stft_cfg.fft_size = 1024;
                self.stft_cfg.hop_size = 256;
                self.heterodyne_carrier_khz = (nyq_khz * 0.5).clamp(2.0, 20.0);
            }
            self.bandpass_low_khz = if sr >= 160_000 { 15.0 } else { 1.2 };
            self.bandpass_high_khz = (nyq_khz * 0.94).max(self.bandpass_low_khz + 1.0);
        }

        self.filtered_samples = if self.apply_bandpass {
            bandpass_zero_phase(
                &self.raw_recording.samples,
                self.bandpass_low_khz * 1000.0,
                self.bandpass_high_khz * 1000.0,
                sr,
            )
        } else {
            self.raw_recording.samples.clone()
        };

        self.tkeo_trace = compute_tkeo(&self.filtered_samples);
        self.spectrogram = compute_stft(&self.filtered_samples, sr, self.stft_cfg);
        self.detected_calls =
            extract_calls(&self.filtered_samples, &self.spectrogram, self.detector_cfg);
        self.harmonic_profiles =
            analyze_harmonic_stacks(&self.detected_calls, &self.spectrogram);
        self.cluster_report = analyze_repertoire_clusters(
            &self.detected_calls,
            &self.spectrogram,
            self.cluster_cfg,
        );
        self.soundscape_indices =
            compute_soundscape_indices(&self.filtered_samples, &self.spectrogram);

        let min_zc_hz = if sr >= 120_000 { 14_000.0 } else { 600.0 };
        self.zc_dots = extract_zero_crossing_dots(
            &self.filtered_samples,
            sr,
            self.zc_division_ratio,
            0.055,
            min_zc_hz,
            self.raw_recording.nyquist_hz() * 0.96,
        );

        if reset_viewport {
            self.view_start_s = 0.0;
            self.view_end_s = self.raw_recording.duration_s().max(0.05);
            self.freq_min_khz = 0.0;
            self.freq_max_khz = nyq_khz;
            self.selected_call_idx = if self.detected_calls.is_empty() {
                None
            } else {
                Some(0)
            };
            if let Some(first) = self.detected_calls.first() {
                let mid_t = 0.5 * (first.begin_time_s + first.end_time_s);
                self.cursor_frame = self.time_to_frame(mid_t);
            } else {
                self.cursor_frame = self.spectrogram.num_frames / 2;
            }
        } else {
            self.cursor_frame = self
                .cursor_frame
                .min(self.spectrogram.num_frames.saturating_sub(1));
            if let Some(sel) = self.selected_call_idx {
                if sel >= self.detected_calls.len() {
                    self.selected_call_idx = self.detected_calls.len().checked_sub(1);
                }
            }
        }

        self.spec_dirty = true;
    }

    fn poll_live_microphone(&mut self, ctx: &egui::Context) {
        if !self.live_mic.is_capturing() {
            return;
        }
        let elapsed = self.last_live_refresh.elapsed();
        if elapsed >= Duration::from_millis(65) {
            self.live_mic.pump_if_simulated(elapsed.as_secs_f64());
            self.last_live_refresh = Instant::now();
            if let Some(live_rec) = self.live_mic.freeze_to_recording() {
                let prev_sr = self.raw_recording.sample_rate_hz;
                self.raw_recording = live_rec;
                let sr_changed = prev_sr != self.raw_recording.sample_rate_hz;
                self.recompute_full_pipeline(sr_changed);
                self.view_start_s = 0.0;
                self.view_end_s = self.raw_recording.duration_s().max(0.1);
                self.spec_dirty = true;
            }
        }
        ctx.request_repaint_after(Duration::from_millis(60));
    }

    fn time_to_frame(&self, t_s: f64) -> usize {
        if self.spectrogram.num_frames <= 1 {
            return 0;
        }
        let raw = (t_s * self.raw_recording.sample_rate_hz as f64)
            / (self.spectrogram.hop_size.max(1) as f64);
        (raw.round() as isize).clamp(0, (self.spectrogram.num_frames - 1) as isize) as usize
    }

    fn rebuild_spectrogram_texture(&mut self, ctx: &egui::Context) {
        let tex_w = self.spectrogram.num_frames.clamp(1, 2048);
        let tex_h = self.spectrogram.num_bins.clamp(1, 1024);

        let b_min = self.spectrogram.freq_to_bin(self.freq_min_khz * 1000.0);
        let b_max = self
            .spectrogram
            .freq_to_bin(self.freq_max_khz * 1000.0)
            .max(b_min + 1);

        let f_min = self.time_to_frame(self.view_start_s);
        let f_max = self
            .time_to_frame(self.view_end_s)
            .max(f_min + 1)
            .min(self.spectrogram.num_frames);

        let mut pixels = vec![Color32::BLACK; tex_w * tex_h];
        let ceil_db = -6.0 + self.gain_offset_db;
        let floor_db = ceil_db - self.dynamic_range_db.max(18.0);

        for row in 0..tex_h {
            let v_frac = 1.0 - (row as f64 / (tex_h.saturating_sub(1).max(1) as f64));
            let bin = b_min + ((v_frac * (b_max - b_min) as f64).round() as usize);
            let bin_clamped = bin.min(self.spectrogram.num_bins - 1);

            for col in 0..tex_w {
                let u_frac = col as f64 / (tex_w.saturating_sub(1).max(1) as f64);
                let fr = f_min + ((u_frac * (f_max - f_min) as f64).floor() as usize);
                let fr_clamped = fr.min(self.spectrogram.num_frames - 1);
                let db = self.spectrogram.db_at(fr_clamped, bin_clamped);
                let unit = (db - floor_db) / (ceil_db - floor_db).max(1.0);
                pixels[row * tex_w + col] = self.palette.map_unit(unit);
            }
        }

        let image = ColorImage {
            size: [tex_w, tex_h],
            pixels,
        };
        self.spec_texture =
            Some(ctx.load_texture("sonowing_spec_tex", image, TextureOptions::LINEAR));
        self.spec_dirty = false;
    }

    fn quick_save_sqlite(&mut self) {
        match SonoWingRepository::open(&self.db_path) {
            Ok(repo) => match repo.save_session(
                &self.slot_name_input,
                &self.raw_recording.source_name,
                self.raw_recording.sample_rate_hz,
                self.raw_recording.duration_s(),
                self.raw_recording.guano.as_ref(),
                &self.detected_calls,
                &self.harmonic_profiles,
                &self.cluster_report,
                &self.soundscape_indices,
            ) {
                Ok(id) => {
                    self.status_banner = format!(
                        "Saved v1.1.0 session #{id} ('{}', K*={}) to SQLite WAL '{}'",
                        self.slot_name_input,
                        self.cluster_report.optimal_k,
                        self.db_path.display()
                    );
                    self.refresh_sessions_list();
                }
                Err(e) => self.status_banner = format!("SQLite save error: {e}"),
            },
            Err(e) => self.status_banner = format!("SQLite open error: {e}"),
        }
    }

    fn refresh_sessions_list(&mut self) {
        if let Ok(repo) = SonoWingRepository::open(&self.db_path) {
            if let Ok(list) = repo.list_recent_sessions(25) {
                self.recent_sessions = list;
            }
        }
    }

    fn load_sqlite_session(&mut self, session_id: i64) {
        if let Ok(repo) = SonoWingRepository::open(&self.db_path) {
            match repo.load_session_payload(session_id) {
                Ok(payload) => {
                    self.detected_calls = payload.detected_calls;
                    self.harmonic_profiles = payload.harmonic_profiles;
                    self.cluster_report = payload.cluster_report;
                    self.soundscape_indices = payload.soundscape_indices;
                    if let Some(g) = payload.guano {
                        self.raw_recording.guano = Some(g);
                    }
                    self.status_banner = format!(
                        "Restored v1.1.0 session #{session_id} ('{}')",
                        payload.source_name
                    );
                }
                Err(e) => self.status_banner = format!("Session load failed: {e}"),
            }
        }
    }
}

impl eframe::App for SonoWingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_live_microphone(ctx);

        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(first_path) = dropped.first() {
            self.load_external_path(first_path);
        }

        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_reference_modal = !self.show_reference_modal;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.quick_save_sqlite();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F9)) {
            self.refresh_sessions_list();
            self.show_sessions_modal = true;
        }

        self.render_top_menu_bar(ctx);

        if self.stage == AppStage::StartupHub {
            self.render_startup_hub(ctx);
        } else {
            if self.spec_dirty || self.spec_texture.is_none() {
                self.rebuild_spectrogram_texture(ctx);
            }
            self.render_left_dsp_panel(ctx);
            self.render_right_inspector_panel(ctx);
            self.render_bottom_dock(ctx);
            self.render_central_waveform_and_spectrogram(ctx);
        }

        if self.show_reference_modal {
            self.render_reference_modal(ctx);
        }
        if self.show_sessions_modal {
            self.render_sessions_modal(ctx);
        }
    }
}

impl SonoWingApp {
    fn render_top_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("sonowing_top_menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.strong("SonoWing v1.1.0");
                ui.separator();

                ui.menu_button("File", |ui| {
                    if ui.button("Startup Main Menu Hub").clicked() {
                        self.stage = AppStage::StartupHub;
                        ui.close_menu();
                    }
                    if ui.button("Save Session to SQLite WAL (F5)").clicked() {
                        self.quick_save_sqlite();
                        ui.close_menu();
                    }
                    if ui.button("Load Saved SQLite Session (F9)").clicked() {
                        self.refresh_sessions_list();
                        self.show_sessions_modal = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Field Presets (9)", |ui| {
                    for preset in BuiltinPreset::ALL {
                        if ui.button(preset.title()).clicked() {
                            self.load_preset(preset);
                            self.stage = AppStage::Workstation;
                            ui.close_menu();
                        }
                    }
                });

                ui.menu_button("Live Mic Capture", |ui| {
                    ui.checkbox(
                        &mut self.mic_prefer_ultrasonic,
                        "Prefer Highest USB UAC2 Rate (up to 384 kHz)",
                    );
                    if ui.button("Start Hardware Microphone (cpal)").clicked() {
                        self.last_live_refresh = Instant::now();
                        match self
                            .live_mic
                            .start_hardware_capture(self.mic_window_seconds, self.mic_prefer_ultrasonic)
                        {
                            Ok(desc) => {
                                self.stage = AppStage::Workstation;
                                self.status_banner = format!("Live Mic Active: {desc}");
                            }
                            Err(e) => self.status_banner = format!("Mic unavailable ({e})"),
                        }
                        ui.close_menu();
                    }
                    if ui
                        .button("Start Simulated 192 kHz Bat+Bird Live Stream")
                        .clicked()
                    {
                        self.last_live_refresh = Instant::now();
                        let lbl = self
                            .live_mic
                            .start_simulated_stream(self.mic_window_seconds, true);
                        self.stage = AppStage::Workstation;
                        self.status_banner = format!("Streaming: {lbl}");
                        ui.close_menu();
                    }
                    if ui.button("Freeze & Analyze Captured Buffer").clicked() {
                        if let Some(rec) = self.live_mic.freeze_to_recording() {
                            self.live_mic.stop();
                            self.raw_recording = rec;
                            self.recompute_full_pipeline(true);
                            self.status_banner =
                                "Froze live microphone buffer into static analysis workspace."
                                    .to_string();
                        }
                        ui.close_menu();
                    }
                });

                ui.menu_button("View & Colormap", |ui| {
                    for pal in ColormapPalette::ALL {
                        if ui.radio(self.palette == pal, pal.label()).clicked() {
                            self.palette = pal;
                            self.spec_dirty = true;
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("Reset Full Time & Nyquist Zoom").clicked() {
                        self.view_start_s = 0.0;
                        self.view_end_s = self.raw_recording.duration_s().max(0.05);
                        self.freq_min_khz = 0.0;
                        self.freq_max_khz = (self.raw_recording.nyquist_hz() * 0.001).max(1.0);
                        self.spec_dirty = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help & Species Key", |ui| {
                    if ui
                        .button("20-Species Acoustic Reference Key & Shortcuts (F1)")
                        .clicked()
                    {
                        self.show_reference_modal = true;
                        ui.close_menu();
                    }
                });

                if self.live_mic.is_capturing() && ui.button("Freeze Live Mic").clicked() {
                    if let Some(rec) = self.live_mic.freeze_to_recording() {
                        self.live_mic.stop();
                        self.raw_recording = rec;
                        self.recompute_full_pipeline(true);
                        self.status_banner = "Froze live microphone spectrogram buffer.".to_string();
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let live_tag = if self.live_mic.is_capturing() {
                        " [LIVE MIC] "
                    } else {
                        ""
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "{}{} | {:.1} kHz | {} calls | K*={}",
                            live_tag,
                            self.raw_recording.source_name,
                            self.raw_recording.sample_rate_hz as f64 * 0.001,
                            self.detected_calls.len(),
                            self.cluster_report.optimal_k
                        ))
                        .color(if self.live_mic.is_capturing() {
                            Color32::from_rgb(255, 118, 148)
                        } else {
                            Color32::from_rgb(135, 214, 197)
                        }),
                    );
                });
            });
        });
    }

    fn render_startup_hub(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(10.0);
                ui.heading(
                    "SonoWing v1.1.0 — Bat & Bird Sonic Waveform, MFCC/UMAP & Harmonic Workstation",
                );
                ui.label(
                    "Full-spectrum & zero-crossing ultrasonic bat echolocation + avian flight-call/song analyzer with unsupervised MFCC/UMAP repertoire clustering, multi-harmonic stack tracking, and live microphone capture.",
                );
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    ui.label("Open WAV / ZC Path:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.file_path_input)
                            .desired_width(360.0)
                            .hint_text("/path/to/AudioMoth_384kHz.wav or sequence.zc"),
                    );
                    if ui.button("Load File").clicked() && !self.file_path_input.trim().is_empty() {
                        let p = PathBuf::from(self.file_path_input.trim());
                        self.load_external_path(&p);
                    }
                    if ui.button("Start Hardware Live Mic").clicked() {
                        self.last_live_refresh = Instant::now();
                        match self
                            .live_mic
                            .start_hardware_capture(self.mic_window_seconds, self.mic_prefer_ultrasonic)
                        {
                            Ok(desc) => {
                                self.stage = AppStage::Workstation;
                                self.status_banner = format!("Live Mic Active: {desc}");
                            }
                            Err(e) => {
                                let lbl = self
                                    .live_mic
                                    .start_simulated_stream(self.mic_window_seconds, true);
                                self.stage = AppStage::Workstation;
                                self.status_banner =
                                    format!("Hardware mic unavailable ({e}); launched {lbl}");
                            }
                        }
                    }
                    if ui.button("Enter Spectrogram Workstation ->").clicked() {
                        self.stage = AppStage::Workstation;
                    }
                });

                ui.add_space(12.0);
                ui.separator();
                ui.subheading_label(
                    "Calibrated Bioacoustic Benchmark Suites (9 Presets — Click Any Card to Launch)",
                );
                ui.add_space(6.0);

                egui::Grid::new("hub_presets_grid")
                    .num_columns(2)
                    .spacing([14.0, 12.0])
                    .show(ui, |ui| {
                        for (idx, preset) in BuiltinPreset::ALL.iter().enumerate() {
                            egui::Frame::group(ui.style())
                                .fill(Color32::from_rgb(20, 26, 36))
                                .inner_margin(10.0)
                                .show(ui, |ui| {
                                    ui.set_min_width(460.0);
                                    ui.strong(preset.title());
                                    ui.label(preset.subtitle());
                                    ui.add_space(4.0);
                                    if ui
                                        .button(format!("Launch {} in Spectrogram Lab", preset.code()))
                                        .clicked()
                                    {
                                        self.load_preset(*preset);
                                        self.stage = AppStage::Workstation;
                                    }
                                });
                            if idx % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });

                ui.add_space(14.0);
                ui.label(
                    egui::RichText::new(&self.status_banner)
                        .color(Color32::from_rgb(185, 205, 230)),
                );
            });
        });
    }

    fn render_left_dsp_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sonowing_left_dsp")
            .resizable(true)
            .default_width(295.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.strong("Live Mic & File Input");
                    ui.horizontal(|ui| {
                        if !self.live_mic.is_capturing() {
                            if ui.button("Live Mic").clicked() {
                                self.last_live_refresh = Instant::now();
                                if let Err(e) = self.live_mic.start_hardware_capture(
                                    self.mic_window_seconds,
                                    self.mic_prefer_ultrasonic,
                                ) {
                                    let lbl = self
                                        .live_mic
                                        .start_simulated_stream(self.mic_window_seconds, true);
                                    self.status_banner = format!("Using {lbl} ({e})");
                                } else {
                                    self.status_banner = "Hardware live microphone streaming...".to_string();
                                }
                            }
                            if ui.button("Sim Stream").clicked() {
                                self.last_live_refresh = Instant::now();
                                let lbl = self
                                    .live_mic
                                    .start_simulated_stream(self.mic_window_seconds, true);
                                self.status_banner = format!("Streaming {lbl}");
                            }
                        } else if ui.button("Freeze & Stop Mic").clicked() {
                            if let Some(rec) = self.live_mic.freeze_to_recording() {
                                self.live_mic.stop();
                                self.raw_recording = rec;
                                self.recompute_full_pipeline(true);
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.file_path_input)
                                .desired_width(190.0)
                                .hint_text("WAV or .zc file path"),
                        );
                        if ui.button("Load").clicked() && !self.file_path_input.trim().is_empty() {
                            let p = PathBuf::from(self.file_path_input.trim());
                            self.load_external_path(&p);
                        }
                    });
                    ui.small(format!(
                        "Encoding: {} | Peak: {:.1} dBFS | RMS: {:.1} dBFS",
                        self.raw_recording.encoding.label(),
                        self.raw_recording.peak_dbfs(),
                        self.raw_recording.rms_dbfs()
                    ));

                    ui.separator();
                    ui.strong("STFT Spectrogram Engine");
                    let mut dsp_changed = false;

                    egui::ComboBox::from_label("FFT Size")
                        .selected_text(format!("{} pts", self.stft_cfg.fft_size))
                        .show_ui(ui, |ui| {
                            for sz in [256usize, 512, 1024, 2048, 4096] {
                                if ui
                                    .selectable_value(
                                        &mut self.stft_cfg.fft_size,
                                        sz,
                                        format!("{sz} pts"),
                                    )
                                    .changed()
                                {
                                    dsp_changed = true;
                                }
                            }
                        });

                    egui::ComboBox::from_label("Hop Step")
                        .selected_text(format!("{} samp", self.stft_cfg.hop_size))
                        .show_ui(ui, |ui| {
                            for hop in [64usize, 128, 256, 512] {
                                if ui
                                    .selectable_value(
                                        &mut self.stft_cfg.hop_size,
                                        hop,
                                        format!("{hop} samp"),
                                    )
                                    .changed()
                                {
                                    dsp_changed = true;
                                }
                            }
                        });

                    egui::ComboBox::from_label("Window")
                        .selected_text(self.stft_cfg.window.label())
                        .show_ui(ui, |ui| {
                            for w in WindowFunction::ALL {
                                if ui
                                    .selectable_value(&mut self.stft_cfg.window, w, w.label())
                                    .changed()
                                {
                                    dsp_changed = true;
                                }
                            }
                        });

                    if ui
                        .checkbox(
                            &mut self.stft_cfg.subtract_median_noise_floor,
                            "Median Per-Bin Spectral Denoising",
                        )
                        .changed()
                    {
                        dsp_changed = true;
                    }

                    ui.separator();
                    ui.strong("Zero-Phase Butterworth Filter");
                    let nyq_khz = (self.raw_recording.nyquist_hz() * 0.001).max(1.0);
                    if ui
                        .checkbox(&mut self.apply_bandpass, "Enable 4th-Order Zero-Phase Bandpass")
                        .changed()
                    {
                        dsp_changed = true;
                    }
                    if self.apply_bandpass {
                        if ui
                            .add(
                                egui::Slider::new(&mut self.bandpass_low_khz, 0.2..=(nyq_khz * 0.9))
                                    .text("High-Pass (kHz)"),
                            )
                            .changed()
                        {
                            dsp_changed = true;
                        }
                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.bandpass_high_khz,
                                    (self.bandpass_low_khz + 0.5)..=(nyq_khz * 0.98),
                                )
                                .text("Low-Pass (kHz)"),
                            )
                            .changed()
                        {
                            dsp_changed = true;
                        }
                    }

                    ui.separator();
                    ui.strong("Detector & MFCC/UMAP Clustering");
                    egui::ComboBox::from_label("Taxon Band")
                        .selected_text(self.detector_cfg.mode.label())
                        .show_ui(ui, |ui| {
                            for m in DetectionMode::ALL {
                                if ui
                                    .selectable_value(&mut self.detector_cfg.mode, m, m.label())
                                    .changed()
                                {
                                    dsp_changed = true;
                                }
                            }
                        });
                    if ui
                        .add(
                            egui::Slider::new(&mut self.detector_cfg.snr_threshold_db, 6.0..=28.0)
                                .text("Trigger SNR (dB)"),
                        )
                        .changed()
                    {
                        dsp_changed = true;
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut self.detector_cfg.min_duration_ms, 0.8..=80.0)
                                .text("Min Duration (ms)"),
                        )
                        .changed()
                    {
                        dsp_changed = true;
                    }

                    if dsp_changed {
                        self.recompute_full_pipeline(false);
                    }

                    ui.separator();
                    ui.strong("Viewport & Overlays");
                    let dur_s = self.raw_recording.duration_s().max(0.05);
                    let mut view_changed = false;
                    view_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.view_start_s, 0.0..=(dur_s * 0.95))
                                .text("Start (s)"),
                        )
                        .changed();
                    view_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut self.view_end_s,
                                (self.view_start_s + 0.02)..=dur_s,
                            )
                            .text("End (s)"),
                        )
                        .changed();
                    view_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.freq_min_khz, 0.0..=(nyq_khz * 0.85))
                                .text("Min Freq (kHz)"),
                        )
                        .changed();
                    view_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut self.freq_max_khz,
                                (self.freq_min_khz + 1.0)..=nyq_khz,
                            )
                            .text("Max Freq (kHz)"),
                        )
                        .changed();
                    view_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.dynamic_range_db, 30.0..=96.0)
                                .text("Dyn Range (dB)"),
                        )
                        .changed();
                    view_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.gain_offset_db, -24.0..=24.0)
                                .text("Gain Offset (dB)"),
                        )
                        .changed();

                    if view_changed {
                        self.spec_dirty = true;
                    }

                    ui.checkbox(&mut self.show_call_boxes, "Show Bounding Boxes & Cluster/Taxon Tags");
                    ui.checkbox(&mut self.show_ridge_overlay, "Show Fundamental Peak Ridge");
                    ui.checkbox(
                        &mut self.show_harmonics_overlay,
                        "Show Harmonic Stack Overtones (2f0..5f0)",
                    );
                    ui.checkbox(&mut self.show_zc_overlay, "Overlay Anabat Zero-Crossing Dots");
                    ui.checkbox(&mut self.show_tkeo_overlay, "Show Teager-Kaiser Energy on Waveform");
                });
            });
    }

    fn render_right_inspector_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("sonowing_right_inspector")
            .resizable(true)
            .default_width(335.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.strong("Power Spectrum Slice @ Cursor");
                    let t_cur = self.spectrogram.frame_time_s(self.cursor_frame);
                    let f_pk = self
                        .spectrogram
                        .frame_peak_freq_hz
                        .get(self.cursor_frame)
                        .copied()
                        .unwrap_or(0.0);
                    ui.small(format!(
                        "Frame #{} ({:.4} s) | Peak {:.2} kHz ({:.1} dBFS)",
                        self.cursor_frame,
                        t_cur,
                        f_pk * 0.001,
                        self.spectrogram
                            .frame_peak_dbfs
                            .get(self.cursor_frame)
                            .copied()
                            .unwrap_or(-96.0)
                    ));

                    let (slice_rect, _) =
                        ui.allocate_exact_size(Vec2::new(ui.available_width(), 110.0), Sense::hover());
                    self.paint_spectrum_slice(ui.painter_at(slice_rect), slice_rect);

                    ui.separator();
                    ui.strong("Call Morphometrics, Harmonics & Cluster");

                    if let Some(sel_idx) = self.selected_call_idx {
                        if let Some(ev) = self.detected_calls.get(sel_idx).cloned() {
                            let cl_tag = self
                                .cluster_report
                                .points
                                .get(sel_idx)
                                .map(|p| format!("Cluster C{} (s={:.2})", p.cluster_id + 1, p.silhouette))
                                .unwrap_or_else(|| "Cluster C1".to_string());

                            ui.label(
                                egui::RichText::new(format!(
                                    "Call #{} — {} | {}",
                                    ev.id,
                                    ev.phase.label(),
                                    cl_tag
                                ))
                                .strong()
                                .color(Color32::from_rgb(245, 209, 110)),
                            );

                            egui::Grid::new("morpho_kv_grid")
                                .num_columns(2)
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Window / Duration:");
                                    ui.label(format!(
                                        "{:.3}-{:.3} s ({:.2} ms)",
                                        ev.begin_time_s, ev.end_time_s, ev.duration_ms
                                    ));
                                    ui.end_row();

                                    ui.label("Fmax / Fmin / BW:");
                                    ui.label(format!(
                                        "{:.2} / {:.2} / {:.2} kHz",
                                        ev.f_max_khz, ev.f_min_khz, ev.bandwidth_khz
                                    ));
                                    ui.end_row();

                                    ui.label("Fc (Char) / Fk (Knee):");
                                    ui.label(format!(
                                        "{:.2} kHz / {:.2} kHz",
                                        ev.f_char_khz, ev.f_knee_khz
                                    ));
                                    ui.end_row();

                                    ui.label("Slopes (Init / Sc):");
                                    ui.label(format!(
                                        "{:.2} / {:.2} kHz/ms",
                                        ev.slope_init_khz_per_ms, ev.slope_char_khz_per_ms
                                    ));
                                    ui.end_row();

                                    if let Some(hp) = self.harmonic_profiles.get(sel_idx) {
                                        ui.label("Tracked f0 / Dom h*:");
                                        ui.label(format!(
                                            "{:.2} kHz / h{} ({} active)",
                                            hp.fundamental_f0_khz,
                                            hp.dominant_harmonic_order,
                                            hp.active_harmonic_count
                                        ));
                                        ui.end_row();

                                        ui.label("Harmonic HNR / Inharm:");
                                        ui.label(format!(
                                            "{:+.1} dB / {:.2}%",
                                            hp.hnr_db, hp.inharmonicity_pct
                                        ));
                                        ui.end_row();
                                    }
                                });

                            if let Some(hp) = self.harmonic_profiles.get(sel_idx) {
                                ui.add_space(4.0);
                                ui.small("Harmonic Stack Profile (h1=f0 .. h5=5f0):");
                                for (h_idx, &rel_db) in hp.relative_harmonic_db.iter().enumerate() {
                                    let unit = ((rel_db + 36.0) / 36.0).clamp(0.0, 1.0);
                                    ui.horizontal(|ui| {
                                        ui.monospace(format!("h{}", h_idx + 1));
                                        ui.add(
                                            egui::ProgressBar::new(unit)
                                                .desired_width(180.0)
                                                .text(format!("{rel_db:+.1} dB")),
                                        );
                                    });
                                }
                            }

                            ui.add_space(5.0);
                            ui.strong("Top Taxonomic Likelihood Matches:");
                            for m in &ev.top_matches {
                                ui.horizontal(|ui| {
                                    let pct = (m.probability * 100.0) as f32;
                                    ui.monospace(format!("{:<4} {:>5.1}%", m.code, pct));
                                    ui.label(&m.common_name);
                                });
                            }

                            if let Some(ref couplet) = ev.ambiguity_couplet {
                                ui.colored_label(
                                    Color32::from_rgb(255, 178, 92),
                                    format!("Sympatric Couplet Advisory: {couplet}"),
                                );
                            }

                            let mut cur_manual = ev
                                .manual_vetted_code
                                .clone()
                                .unwrap_or_else(|| "AUTO".to_string());
                            egui::ComboBox::from_label("Manual Vetting")
                                .selected_text(&cur_manual)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_value(
                                            &mut cur_manual,
                                            "AUTO".to_string(),
                                            "AUTO (Classifier Top-1)",
                                        )
                                        .changed()
                                    {
                                        self.detected_calls[sel_idx].manual_vetted_code = None;
                                    }
                                    for sp in reference_species_catalog()
                                        .iter()
                                        .filter(|p| p.domain == ev.domain)
                                    {
                                        let lbl = format!("{} — {}", sp.code, sp.common_name);
                                        if ui
                                            .selectable_value(
                                                &mut cur_manual,
                                                sp.code.to_string(),
                                                lbl,
                                            )
                                            .changed()
                                        {
                                            self.detected_calls[sel_idx].manual_vetted_code =
                                                Some(sp.code.to_string());
                                        }
                                    }
                                });
                        }
                    } else {
                        ui.label("No call event selected or above SNR trigger threshold.");
                    }
                });
            });
    }

    fn render_bottom_dock(&mut self, ctx: &egui::Context) {
        let tab_id = egui::Id::new("sonowing_bottom_tab_v11");
        let mut active_tab = ctx
            .data_mut(|d| d.get_temp::<BottomDockTab>(tab_id))
            .unwrap_or(BottomDockTab::CallsTable);

        egui::TopBottomPanel::bottom("sonowing_bottom_dock")
            .resizable(true)
            .default_height(235.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut active_tab,
                        BottomDockTab::CallsTable,
                        format!("Detected Calls ({})", self.detected_calls.len()),
                    );
                    ui.selectable_value(
                        &mut active_tab,
                        BottomDockTab::MfccUmapClusters,
                        format!(
                            "MFCC & UMAP Clusters (K*={}, Sil={:.2})",
                            self.cluster_report.optimal_k, self.cluster_report.global_silhouette
                        ),
                    );
                    ui.selectable_value(
                        &mut active_tab,
                        BottomDockTab::ZeroCrossingTrace,
                        format!("Anabat ZC ({} dots)", self.zc_dots.len()),
                    );
                    ui.selectable_value(
                        &mut active_tab,
                        BottomDockTab::SoundscapeIndices,
                        "Soundscape Indices (6)",
                    );
                    ui.selectable_value(
                        &mut active_tab,
                        BottomDockTab::AuditionAndExport,
                        "Auditioning & Exports",
                    );
                    ui.separator();
                    ui.small(&self.status_banner);
                });
                ui.separator();

                match active_tab {
                    BottomDockTab::CallsTable => self.render_calls_table_tab(ui),
                    BottomDockTab::MfccUmapClusters => self.render_mfcc_umap_tab(ui),
                    BottomDockTab::ZeroCrossingTrace => self.render_zc_trace_tab(ui),
                    BottomDockTab::SoundscapeIndices => self.render_soundscape_tab(ui),
                    BottomDockTab::AuditionAndExport => self.render_audition_export_tab(ui),
                }
            });

        ctx.data_mut(|d| d.insert_temp(tab_id, active_tab));
    }

    fn render_calls_table_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("detected_calls_dock_grid")
                .striped(true)
                .num_columns(12)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    for h in [
                        "#",
                        "Cluster",
                        "Domain",
                        "Begin (s)",
                        "Dur (ms)",
                        "Fc (kHz)",
                        "f0 / Harm",
                        "HNR (dB)",
                        "IPI (ms)",
                        "Phase",
                        "Taxon ID",
                        "Conf",
                    ] {
                        ui.strong(h);
                    }
                    ui.end_row();

                    let mut clicked_idx: Option<usize> = None;
                    for (idx, ev) in self.detected_calls.iter().enumerate() {
                        let is_sel = self.selected_call_idx == Some(idx);
                        if ui
                            .selectable_label(is_sel, format!("#{}", ev.id))
                            .clicked()
                        {
                            clicked_idx = Some(idx);
                        }
                        let cid = self
                            .cluster_report
                            .points
                            .get(idx)
                            .map(|p| p.cluster_id)
                            .unwrap_or(0);
                        ui.colored_label(
                            CLUSTER_COLORS[cid % CLUSTER_COLORS.len()],
                            format!("C{}", cid + 1),
                        );
                        ui.label(match ev.domain {
                            TaxonDomain::BatChiroptera => "Bat",
                            TaxonDomain::BirdAves => "Bird",
                        });
                        ui.monospace(format!("{:.4}", ev.begin_time_s));
                        ui.monospace(format!("{:.2}", ev.duration_ms));
                        ui.monospace(format!("{:.2}", ev.f_char_khz));
                        if let Some(hp) = self.harmonic_profiles.get(idx) {
                            ui.monospace(format!(
                                "{:.1} (h{}*{})",
                                hp.fundamental_f0_khz,
                                hp.dominant_harmonic_order,
                                hp.active_harmonic_count
                            ));
                            ui.monospace(format!("{:+.1}", hp.hnr_db));
                        } else {
                            ui.label("-");
                            ui.label("-");
                        }
                        ui.monospace(
                            ev.ipi_ms
                                .map(|v| format!("{v:.1}"))
                                .unwrap_or_else(|| "-".to_string()),
                        );
                        ui.label(ev.phase.label());
                        ui.strong(ev.primary_code());
                        ui.monospace(format!("{:.0}%", ev.primary_confidence() * 100.0));
                        ui.end_row();
                    }

                    if let Some(ci) = clicked_idx {
                        self.selected_call_idx = Some(ci);
                        let mid = 0.5
                            * (self.detected_calls[ci].begin_time_s
                                + self.detected_calls[ci].end_time_s);
                        self.cursor_frame = self.time_to_frame(mid);
                    }
                });
        });
    }

    fn render_mfcc_umap_tab(&mut self, ui: &mut egui::Ui) {
        let mut recluster = false;
        ui.horizontal(|ui| {
            ui.label("Cluster Mode:");
            for mode in [
                ClusterCountMode::AutoSilhouette,
                ClusterCountMode::Fixed(2),
                ClusterCountMode::Fixed(3),
                ClusterCountMode::Fixed(4),
                ClusterCountMode::Fixed(5),
            ] {
                let lbl = match mode {
                    ClusterCountMode::AutoSilhouette => "Auto K*".to_string(),
                    ClusterCountMode::Fixed(k) => format!("K={k}"),
                };
                if ui
                    .selectable_value(&mut self.cluster_cfg.k_mode, mode, lbl)
                    .changed()
                {
                    recluster = true;
                }
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.cluster_cfg.umap_n_neighbors, 2..=12)
                        .text("k-NN"),
                )
                .changed()
            {
                recluster = true;
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.cluster_cfg.umap_min_dist, 0.04..=0.50)
                        .text("min_dist"),
                )
                .changed()
            {
                recluster = true;
            }
        });
        if recluster {
            self.cluster_report = analyze_repertoire_clusters(
                &self.detected_calls,
                &self.spectrogram,
                self.cluster_cfg,
            );
        }

        ui.columns(2, |cols| {
            let avail_h = cols[0].available_height().max(130.0);
            let (scatter_rect, resp) = cols[0].allocate_exact_size(
                Vec2::new(cols[0].available_width(), avail_h),
                Sense::click(),
            );
            let painter = cols[0].painter_at(scatter_rect);
            painter.rect_filled(
                scatter_rect,
                CornerRadius::same(4),
                Color32::from_rgb(10, 14, 22),
            );
            painter.rect_stroke(
                scatter_rect,
                CornerRadius::same(4),
                Stroke::new(1.0, Color32::from_rgb(46, 58, 78)),
                StrokeKind::Inside,
            );

            // Axes crosshairs
            painter.line_segment(
                [
                    Pos2::new(scatter_rect.left(), scatter_rect.center().y),
                    Pos2::new(scatter_rect.right(), scatter_rect.center().y),
                ],
                Stroke::new(0.8, Color32::from_rgb(30, 40, 56)),
            );
            painter.line_segment(
                [
                    Pos2::new(scatter_rect.center().x, scatter_rect.top()),
                    Pos2::new(scatter_rect.center().x, scatter_rect.bottom()),
                ],
                Stroke::new(0.8, Color32::from_rgb(30, 40, 56)),
            );

            let mut clicked_call: Option<usize> = None;
            for pt in &self.cluster_report.points {
                let px = scatter_rect.center().x + (pt.umap_x as f32) * scatter_rect.width() * 0.44;
                let py =
                    scatter_rect.center().y - (pt.umap_y as f32) * scatter_rect.height() * 0.42;
                let pos = Pos2::new(px, py);
                let col = CLUSTER_COLORS[pt.cluster_id % CLUSTER_COLORS.len()];
                let is_sel = self.selected_call_idx == Some(pt.call_index);
                painter.circle_filled(pos, if is_sel { 6.0 } else { 4.2 }, col);
                if is_sel {
                    painter.circle_stroke(pos, 8.0, Stroke::new(1.6, Color32::WHITE));
                }
                painter.text(
                    Pos2::new(pos.x + 6.0, pos.y - 6.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("#{} {}", pt.call_id, pt.species_code),
                    FontId::monospace(9.5),
                    col,
                );

                if resp.clicked() {
                    if let Some(cpos) = resp.interact_pointer_pos() {
                        if cpos.distance(pos) <= 12.0 {
                            clicked_call = Some(pt.call_index);
                        }
                    }
                }
            }
            if let Some(ci) = clicked_call {
                self.selected_call_idx = Some(ci);
                let mid = 0.5
                    * (self.detected_calls[ci].begin_time_s + self.detected_calls[ci].end_time_s);
                self.cursor_frame = self.time_to_frame(mid);
            }

            cols[1].strong(format!(
                "Unsupervised Repertoire Clusters (Optimal K*={}, Mean Silhouette={:.3})",
                self.cluster_report.optimal_k, self.cluster_report.global_silhouette
            ));
            egui::ScrollArea::vertical().show(&mut cols[1], |ui| {
                egui::Grid::new("umap_cluster_summary_grid")
                    .striped(true)
                    .num_columns(6)
                    .show(ui, |ui| {
                        for h in ["Cluster", "Calls", "Dom Taxon", "Mean Fc", "Mean Dur", "Silhouette"] {
                            ui.strong(h);
                        }
                        ui.end_row();
                        for cl in &self.cluster_report.clusters {
                            let col = CLUSTER_COLORS[cl.cluster_id % CLUSTER_COLORS.len()];
                            ui.colored_label(col, format!("C{}", cl.cluster_id + 1));
                            ui.monospace(format!("{}", cl.member_count));
                            ui.strong(&cl.dominant_species);
                            ui.monospace(format!("{:.2} kHz", cl.mean_fc_khz));
                            ui.monospace(format!("{:.1} ms", cl.mean_dur_ms));
                            ui.monospace(format!("{:+.3}", cl.mean_silhouette));
                            ui.end_row();
                        }
                    });
            });
        });
    }

    fn render_zc_trace_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Virtual Anabat Frequency Divider:");
            for div in [4u8, 8, 16] {
                if ui
                    .selectable_value(&mut self.zc_division_ratio, div, format!("1/{div}"))
                    .changed()
                {
                    self.recompute_full_pipeline(false);
                }
            }
            ui.small("Green dots = shallow QCF body | Cyan = FM sweep | Coral = steep terminal buzz");
        });
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), ui.available_height()),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(10, 14, 20));
        painter.rect_stroke(
            rect,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(48, 60, 78)),
            StrokeKind::Inside,
        );

        let t_span = (self.view_end_s - self.view_start_s).max(1e-4);
        let f_span = (self.freq_max_khz - self.freq_min_khz).max(0.5);
        for d in &self.zc_dots {
            if d.time_s < self.view_start_s || d.time_s > self.view_end_s {
                continue;
            }
            let fk = d.freq_hz * 0.001;
            if fk < self.freq_min_khz || fk > self.freq_max_khz {
                continue;
            }
            let x = rect.left() + ((d.time_s - self.view_start_s) / t_span) as f32 * rect.width();
            let y = rect.bottom() - ((fk - self.freq_min_khz) / f_span) as f32 * rect.height();
            let col = if d.slope_khz_per_ms.abs() < 1.2 {
                Color32::from_rgb(104, 232, 138)
            } else if d.slope_khz_per_ms.abs() < 5.0 {
                Color32::from_rgb(92, 204, 248)
            } else {
                Color32::from_rgb(255, 118, 136)
            };
            painter.circle_filled(Pos2::new(x, y), 2.1, col);
        }
    }

    fn render_soundscape_tab(&self, ui: &mut egui::Ui) {
        let idx = &self.soundscape_indices;
        ui.columns(2, |cols| {
            egui::Grid::new("eco_indices_grid")
                .striped(true)
                .num_columns(3)
                .show(&mut cols[0], |ui| {
                    ui.strong("Index");
                    ui.strong("Value");
                    ui.strong("Ecological Interpretation");
                    ui.end_row();

                    ui.label("ACI (Pieretti 2011)");
                    ui.monospace(format!("{:.2}", idx.aci));
                    ui.small("Temporal intensity modulation across STFT bins");
                    ui.end_row();

                    ui.label("NDSI (Kasten 2012)");
                    ui.monospace(format!("{:+.4}", idx.ndsi));
                    ui.small("-1.0 Anthrophony (0.5-2 kHz) .. +1.0 Biophony");
                    ui.end_row();

                    ui.label("BIO (Boelman 2007)");
                    ui.monospace(format!("{:.2}", idx.bio_index));
                    ui.small("Spectral dB*kHz area above quietest band floor");
                    ui.end_row();

                    ui.label("ADI / AEI (Villanueva-Rivera)");
                    ui.monospace(format!("{:.3} / {:.3}", idx.adi, idx.aei));
                    ui.small("Shannon band diversity / Gini spectral unevenness");
                    ui.end_row();

                    ui.label("Entropy H = Ht * Hf (Sueur)");
                    ui.monospace(format!(
                        "{:.4} ({:.3}*{:.3})",
                        idx.total_entropy_h, idx.temporal_entropy_ht, idx.spectral_entropy_hf
                    ));
                    ui.small("Combined temporal & spectral Shannon dispersion");
                    ui.end_row();

                    ui.label("Ultrasonic Share (>16 kHz)");
                    ui.monospace(format!("{:.1}%", idx.ultrasonic_energy_fraction * 100.0));
                    ui.small("Chiropteran high-frequency power proportion");
                    ui.end_row();
                });

            cols[1].strong("Spectral Band Occupancy (> -52 dBFS):");
            egui::ScrollArea::vertical().show(&mut cols[1], |ui| {
                for (lbl, occ) in &idx.band_occupancy {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{lbl:<15}"));
                        ui.add(
                            egui::ProgressBar::new((*occ as f32).clamp(0.0, 1.0))
                                .text(format!("{:.1}%", occ * 100.0)),
                        );
                    });
                }
            });
        });
    }

    fn render_audition_export_tab(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            cols[0].strong("Ultrasonic-to-Audible WAV Synthesis (48 kHz Playable)");
            cols[0].horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.heterodyne_carrier_khz, 12.0..=130.0)
                        .text("Heterodyne Carrier (kHz)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.freq_div_ratio, 4..=20).text("Freq Div Ratio"),
                );
            });
            cols[0].horizontal(|ui| {
                if ui.button("Export 10x Time-Expanded WAV").clicked() {
                    self.export_audible_wav(AuditionMode::TimeExpansion10x);
                }
                if ui.button("Export Heterodyne WAV").clicked() {
                    self.export_audible_wav(AuditionMode::Heterodyne {
                        carrier_khz: self.heterodyne_carrier_khz,
                    });
                }
                if ui.button("Export Freq-Div WAV").clicked() {
                    self.export_audible_wav(AuditionMode::FrequencyDivision {
                        ratio: self.freq_div_ratio,
                    });
                }
            });

            cols[1].strong("Raven Pro Selection Table, NABat CSV & SQLite WAL");
            cols[1].horizontal(|ui| {
                ui.label("Output Folder:");
                ui.add(egui::TextEdit::singleline(&mut self.export_dir_input).desired_width(150.0));
                ui.label("Slot:");
                ui.add(egui::TextEdit::singleline(&mut self.slot_name_input).desired_width(140.0));
            });
            cols[1].horizontal(|ui| {
                if ui.button("Export Raven .selections.txt").clicked() {
                    self.export_raven_and_csv(true, false);
                }
                if ui.button("Export NABat/Avian .csv").clicked() {
                    self.export_raven_and_csv(false, true);
                }
                if ui.button("Export .zc Dots").clicked() {
                    self.export_zc_file();
                }
                if ui.button("Export SHA-256 Workset").clicked() {
                    self.export_workset_envelope();
                }
                if ui.button("Save SQLite (F5)").clicked() {
                    self.quick_save_sqlite();
                }
            });
        });
    }

    fn export_audible_wav(&mut self, mode: AuditionMode) {
        let dir = PathBuf::from(self.export_dir_input.trim());
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.status_banner = format!("Cannot create export directory: {e}");
            return;
        }
        let audible = render_audible_recording(&self.raw_recording, mode, 48_000);
        let out_path = dir.join(&audible.source_name);
        match write_wav_with_guano(
            &out_path,
            &audible.samples,
            audible.sample_rate_hz,
            audible.guano.as_ref(),
        ) {
            Ok(()) => {
                self.status_banner = format!(
                    "Wrote playable 48 kHz WAV ({}) to '{}'",
                    mode.label(),
                    out_path.display()
                );
            }
            Err(e) => self.status_banner = format!("WAV export error: {e}"),
        }
    }

    fn export_raven_and_csv(&mut self, do_raven: bool, do_csv: bool) {
        let dir = PathBuf::from(self.export_dir_input.trim());
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.status_banner = format!("Cannot create export directory: {e}");
            return;
        }
        let stem = self
            .raw_recording
            .source_name
            .rsplit_once('.')
            .map(|(s, _)| s)
            .unwrap_or(&self.raw_recording.source_name);

        if do_raven {
            let p = dir.join(format!("{stem}.Table.1.selections.txt"));
            match write_raven_selection_table(
                &p,
                &self.raw_recording.source_name,
                &self.detected_calls,
                &self.harmonic_profiles,
                &self.cluster_report,
            ) {
                Ok(()) => {
                    self.status_banner =
                        format!("Exported Raven Pro 1.6 selection table '{}'", p.display())
                }
                Err(e) => self.status_banner = format!("Raven export failed: {e}"),
            }
        }
        if do_csv {
            let p = dir.join(format!("{stem}_sonowing_v1_1_summary.csv"));
            match write_calls_csv(
                &p,
                &self.raw_recording.source_name,
                self.raw_recording.sample_rate_hz,
                &self.detected_calls,
                &self.harmonic_profiles,
                &self.cluster_report,
                &self.soundscape_indices,
            ) {
                Ok(()) => self.status_banner = format!("Exported RFC 4180 CSV '{}'", p.display()),
                Err(e) => self.status_banner = format!("CSV export failed: {e}"),
            }
        }
    }

    fn export_zc_file(&mut self) {
        let dir = PathBuf::from(self.export_dir_input.trim());
        let _ = std::fs::create_dir_all(&dir);
        let stem = self
            .raw_recording
            .source_name
            .rsplit_once('.')
            .map(|(s, _)| s)
            .unwrap_or(&self.raw_recording.source_name);
        let p = dir.join(format!("{stem}.zc"));
        let seq = ZeroCrossSequence {
            source_name: self.raw_recording.source_name.clone(),
            division_ratio: self.zc_division_ratio,
            guano: self.raw_recording.guano.clone(),
            dots: self.zc_dots.clone(),
        };
        match write_zc_dot_table(&p, &seq) {
            Ok(()) => {
                self.status_banner = format!("Exported zero-crossing sequence '{}'", p.display())
            }
            Err(e) => self.status_banner = format!("ZC export failed: {e}"),
        }
    }

    fn export_workset_envelope(&mut self) {
        let dir = PathBuf::from(self.export_dir_input.trim());
        let _ = std::fs::create_dir_all(&dir);
        let stem = self
            .raw_recording
            .source_name
            .rsplit_once('.')
            .map(|(s, _)| s)
            .unwrap_or(&self.raw_recording.source_name);
        let p = dir.join(format!("{stem}.sonoworkset.json"));
        let payload = WorksetPayload {
            schema_version: "1.1.0".to_string(),
            source_name: self.raw_recording.source_name.clone(),
            sample_rate_hz: self.raw_recording.sample_rate_hz,
            duration_s: self.raw_recording.duration_s(),
            guano: self.raw_recording.guano.clone(),
            detected_calls: self.detected_calls.clone(),
            harmonic_profiles: self.harmonic_profiles.clone(),
            cluster_report: self.cluster_report.clone(),
            soundscape_indices: self.soundscape_indices.clone(),
        };
        match SignedWorksetEnvelope::build(payload).and_then(|env| env.save_to_file(&p)) {
            Ok(()) => {
                self.status_banner =
                    format!("Saved SHA-256 verified v1.1.0 workset '{}'", p.display());
            }
            Err(e) => self.status_banner = format!("Workset export error: {e}"),
        }
    }

    fn render_central_waveform_and_spectrogram(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let wave_h = (avail.y * 0.27).clamp(95.0, 210.0);
            let spec_h = (avail.y - wave_h - 10.0).max(150.0);

            let (wave_rect, wave_resp) =
                ui.allocate_exact_size(Vec2::new(avail.x, wave_h), Sense::click_and_drag());
            self.paint_oscillogram(ui.painter_at(wave_rect), wave_rect);

            ui.add_space(6.0);

            let (spec_rect, spec_resp) =
                ui.allocate_exact_size(Vec2::new(avail.x, spec_h), Sense::click_and_drag());
            self.paint_spectrogram_canvas(ui.painter_at(spec_rect), spec_rect, &spec_resp);

            for (rect, resp) in [(wave_rect, &wave_resp), (spec_rect, &spec_resp)] {
                if resp.clicked() || resp.dragged() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        let u = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                        let t_click =
                            self.view_start_s + (u as f64) * (self.view_end_s - self.view_start_s);
                        self.cursor_frame = self.time_to_frame(t_click);
                        if let Some((idx, _)) = self
                            .detected_calls
                            .iter()
                            .enumerate()
                            .find(|(_, ev)| t_click >= ev.begin_time_s && t_click <= ev.end_time_s)
                        {
                            self.selected_call_idx = Some(idx);
                        }
                    }
                }
            }
        });
    }

    fn paint_oscillogram(&self, painter: egui::Painter, rect: Rect) {
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(12, 16, 23));
        painter.rect_stroke(
            rect,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(44, 56, 74)),
            StrokeKind::Inside,
        );

        let mid_y = rect.center().y;
        painter.line_segment(
            [Pos2::new(rect.left(), mid_y), Pos2::new(rect.right(), mid_y)],
            Stroke::new(1.0, Color32::from_rgb(35, 45, 60)),
        );

        let sr = self.raw_recording.sample_rate_hz as f64;
        let s0 = ((self.view_start_s * sr) as usize).min(self.filtered_samples.len());
        let s1 = ((self.view_end_s * sr) as usize)
            .min(self.filtered_samples.len())
            .max(s0);
        let cols = (rect.width() as usize).clamp(64, 2048);
        let buckets = downsample_waveform_minmax(&self.filtered_samples, s0, s1, cols);

        let half_h = rect.height() * 0.46;
        for (c, &(lo, hi)) in buckets.iter().enumerate() {
            let x = rect.left() + (c as f32 / cols as f32) * rect.width();
            let y0 = mid_y - hi * half_h;
            let y1 = mid_y - lo * half_h;
            painter.line_segment(
                [Pos2::new(x, y0), Pos2::new(x, y1)],
                Stroke::new(1.0, Color32::from_rgb(90, 210, 198)),
            );
        }

        if self.show_tkeo_overlay && !self.tkeo_trace.is_empty() {
            let tkeo_buckets = downsample_waveform_minmax(&self.tkeo_trace, s0, s1, cols / 2);
            let max_tkeo = tkeo_buckets
                .iter()
                .fold(1e-6_f32, |a, &(_, hi)| a.max(hi));
            let mut prev: Option<Pos2> = None;
            let n_tk = tkeo_buckets.len().max(1);
            for (i, &(_, hi)) in tkeo_buckets.iter().enumerate() {
                let x = rect.left() + (i as f32 / n_tk as f32) * rect.width();
                let norm = (hi / max_tkeo).sqrt().clamp(0.0, 1.0);
                let y = rect.bottom() - 4.0 - norm * (rect.height() * 0.42);
                let pt = Pos2::new(x, y);
                if let Some(p0) = prev {
                    painter.line_segment([p0, pt], Stroke::new(1.2, Color32::from_rgb(248, 165, 75)));
                }
                prev = Some(pt);
            }
        }

        let t_span = (self.view_end_s - self.view_start_s).max(1e-5);
        if self.show_call_boxes {
            for (idx, ev) in self.detected_calls.iter().enumerate() {
                if ev.end_time_s < self.view_start_s || ev.begin_time_s > self.view_end_s {
                    continue;
                }
                let x0 = rect.left()
                    + (((ev.begin_time_s - self.view_start_s) / t_span) as f32 * rect.width())
                        .clamp(0.0, rect.width());
                let x1 = rect.left()
                    + (((ev.end_time_s - self.view_start_s) / t_span) as f32 * rect.width())
                        .clamp(0.0, rect.width());
                let fill = if Some(idx) == self.selected_call_idx {
                    Color32::from_rgba_unmultiplied(255, 215, 90, 38)
                } else {
                    Color32::from_rgba_unmultiplied(90, 185, 255, 20)
                };
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(x0, rect.top()),
                        Pos2::new(x1.max(x0 + 2.0), rect.bottom()),
                    ),
                    CornerRadius::ZERO,
                    fill,
                );
            }
        }

        let cur_t = self.spectrogram.frame_time_s(self.cursor_frame);
        if cur_t >= self.view_start_s && cur_t <= self.view_end_s {
            let cx = rect.left() + ((cur_t - self.view_start_s) / t_span) as f32 * rect.width();
            painter.line_segment(
                [Pos2::new(cx, rect.top()), Pos2::new(cx, rect.bottom())],
                Stroke::new(1.3, Color32::from_rgb(255, 232, 120)),
            );
        }
    }

    fn paint_spectrogram_canvas(
        &self,
        painter: egui::Painter,
        rect: Rect,
        resp: &egui::Response,
    ) {
        if let Some(ref tex) = self.spec_texture {
            painter.image(
                tex.id(),
                rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            painter.rect_filled(rect, CornerRadius::same(4), Color32::BLACK);
        }

        let t_span = (self.view_end_s - self.view_start_s).max(1e-5);
        let f_span = (self.freq_max_khz - self.freq_min_khz).max(0.2);

        for tick in 1..=5 {
            let frac = tick as f32 / 6.0;
            let y = rect.bottom() - frac * rect.height();
            let fk = self.freq_min_khz + (frac as f64) * f_span;
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(0.7, Color32::from_rgba_unmultiplied(200, 215, 235, 36)),
            );
            painter.text(
                Pos2::new(rect.left() + 6.0, y - 2.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{fk:.1} kHz"),
                FontId::monospace(10.5),
                Color32::from_rgb(215, 225, 240),
            );
        }

        if self.show_zc_overlay {
            for d in &self.zc_dots {
                if d.time_s < self.view_start_s || d.time_s > self.view_end_s {
                    continue;
                }
                let fk = d.freq_hz * 0.001;
                if fk < self.freq_min_khz || fk > self.freq_max_khz {
                    continue;
                }
                let x = rect.left() + ((d.time_s - self.view_start_s) / t_span) as f32 * rect.width();
                let y = rect.bottom() - ((fk - self.freq_min_khz) / f_span) as f32 * rect.height();
                painter.circle_filled(Pos2::new(x, y), 2.0, Color32::from_rgb(95, 250, 165));
            }
        }

        if self.show_call_boxes {
            for (idx, ev) in self.detected_calls.iter().enumerate() {
                if ev.end_time_s < self.view_start_s || ev.begin_time_s > self.view_end_s {
                    continue;
                }
                let x0 = rect.left()
                    + (((ev.begin_time_s - self.view_start_s) / t_span) as f32 * rect.width())
                        .clamp(0.0, rect.width());
                let x1 = rect.left()
                    + (((ev.end_time_s - self.view_start_s) / t_span) as f32 * rect.width())
                        .clamp(0.0, rect.width());
                let y_hi = rect.bottom()
                    - (((ev.f_max_khz - self.freq_min_khz) / f_span) as f32 * rect.height())
                        .clamp(0.0, rect.height());
                let y_lo = rect.bottom()
                    - (((ev.f_min_khz - self.freq_min_khz) / f_span) as f32 * rect.height())
                        .clamp(0.0, rect.height());

                let box_color = match ev.phase {
                    BehavioralPhase::TerminalFeedingBuzz => Color32::from_rgb(255, 102, 138),
                    BehavioralPhase::ApproachPhase => Color32::from_rgb(255, 190, 88),
                    BehavioralPhase::SearchPhase => Color32::from_rgb(88, 224, 255),
                    BehavioralPhase::NocturnalFlightCall => Color32::from_rgb(118, 245, 158),
                    BehavioralPhase::TerritorialOrContactVocalization => {
                        Color32::from_rgb(172, 232, 118)
                    }
                };
                let stroke_w = if Some(idx) == self.selected_call_idx {
                    2.2
                } else {
                    1.2
                };
                let call_rect = Rect::from_min_max(
                    Pos2::new(x0, y_hi.min(y_lo - 4.0)),
                    Pos2::new(x1.max(x0 + 4.0), y_lo),
                );
                painter.rect_stroke(
                    call_rect,
                    CornerRadius::same(2),
                    Stroke::new(stroke_w, box_color),
                    StrokeKind::Inside,
                );

                let cl_badge = self
                    .cluster_report
                    .points
                    .get(idx)
                    .map(|p| format!("C{}", p.cluster_id + 1))
                    .unwrap_or_default();
                painter.text(
                    Pos2::new(
                        call_rect.left() + 2.0,
                        (call_rect.top() - 13.0).max(rect.top() + 2.0),
                    ),
                    egui::Align2::LEFT_TOP,
                    format!("#{} {} {}", ev.id, ev.primary_code(), cl_badge),
                    FontId::monospace(10.5),
                    box_color,
                );

                if self.show_ridge_overlay && ev.ridge_trace.len() >= 2 {
                    for pair in ev.ridge_trace.windows(2) {
                        let (t0, f0) = pair[0];
                        let (t1, f1) = pair[1];
                        let px0 =
                            rect.left() + ((t0 - self.view_start_s) / t_span) as f32 * rect.width();
                        let py0 = rect.bottom()
                            - ((f0 - self.freq_min_khz) / f_span) as f32 * rect.height();
                        let px1 =
                            rect.left() + ((t1 - self.view_start_s) / t_span) as f32 * rect.width();
                        let py1 = rect.bottom()
                            - ((f1 - self.freq_min_khz) / f_span) as f32 * rect.height();
                        painter.line_segment(
                            [Pos2::new(px0, py0), Pos2::new(px1, py1)],
                            Stroke::new(1.8, Color32::from_rgb(120, 255, 235)),
                        );
                    }
                }

                // Multi-Harmonic Stack Overtone Contours (2f0..5f0)
                if self.show_harmonics_overlay {
                    if let Some(hp) = self.harmonic_profiles.get(idx) {
                        for (order, pts) in &hp.overtone_contours {
                            let h_col = match order {
                                2 => Color32::from_rgb(255, 214, 102),
                                3 => Color32::from_rgb(218, 152, 255),
                                _ => Color32::from_rgb(255, 148, 194),
                            };
                            for pair in pts.windows(2) {
                                let (t0, f0) = pair[0];
                                let (t1, f1) = pair[1];
                                if f0 > self.freq_max_khz && f1 > self.freq_max_khz {
                                    continue;
                                }
                                let px0 = rect.left()
                                    + ((t0 - self.view_start_s) / t_span) as f32 * rect.width();
                                let py0 = rect.bottom()
                                    - ((f0 - self.freq_min_khz) / f_span) as f32 * rect.height();
                                let px1 = rect.left()
                                    + ((t1 - self.view_start_s) / t_span) as f32 * rect.width();
                                let py1 = rect.bottom()
                                    - ((f1 - self.freq_min_khz) / f_span) as f32 * rect.height();
                                painter.line_segment(
                                    [Pos2::new(px0, py0), Pos2::new(px1, py1)],
                                    Stroke::new(1.3, h_col),
                                );
                            }
                        }
                    }
                }
            }
        }

        let cur_t = self.spectrogram.frame_time_s(self.cursor_frame);
        if cur_t >= self.view_start_s && cur_t <= self.view_end_s {
            let cx = rect.left() + ((cur_t - self.view_start_s) / t_span) as f32 * rect.width();
            painter.line_segment(
                [Pos2::new(cx, rect.top()), Pos2::new(cx, rect.bottom())],
                Stroke::new(1.2, Color32::from_rgb(255, 232, 120)),
            );
        }

        if let Some(hpos) = resp.hover_pos() {
            if rect.contains(hpos) {
                let u = ((hpos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                let v = 1.0 - ((hpos.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
                let ht_s = self.view_start_s + (u as f64) * t_span;
                let hf_khz = self.freq_min_khz + (v as f64) * f_span;
                let fr = self.time_to_frame(ht_s);
                let bin = self.spectrogram.freq_to_bin(hf_khz * 1000.0);
                let db = self.spectrogram.db_at(fr, bin);

                painter.text(
                    Pos2::new(rect.right() - 8.0, rect.top() + 6.0),
                    egui::Align2::RIGHT_TOP,
                    format!("Hover: {ht_s:.4} s | {hf_khz:.2} kHz | {db:.1} dBFS"),
                    FontId::monospace(11.0),
                    Color32::from_rgb(255, 244, 180),
                );
            }
        }

        painter.rect_stroke(
            rect,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(50, 62, 82)),
            StrokeKind::Inside,
        );
    }

    fn paint_spectrum_slice(&self, painter: egui::Painter, rect: Rect) {
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(12, 16, 24));
        painter.rect_stroke(
            rect,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(46, 58, 76)),
            StrokeKind::Inside,
        );

        let slice = self.spectrogram.slice_spectrum_db(self.cursor_frame);
        let b0 = self.spectrogram.freq_to_bin(self.freq_min_khz * 1000.0);
        let b1 = self
            .spectrogram
            .freq_to_bin(self.freq_max_khz * 1000.0)
            .max(b0 + 2)
            .min(slice.len().saturating_sub(1));

        let mut prev: Option<Pos2> = None;
        let span = (b1 - b0).max(1);
        for b in b0..=b1 {
            let x = rect.left() + ((b - b0) as f32 / span as f32) * rect.width();
            let norm = ((slice[b] + 96.0) / 96.0).clamp(0.0, 1.0);
            let y = rect.bottom() - 4.0 - norm * (rect.height() - 10.0);
            let pt = Pos2::new(x, y);
            if let Some(p0) = prev {
                painter.line_segment([p0, pt], Stroke::new(1.3, Color32::from_rgb(112, 224, 196)));
            }
            prev = Some(pt);
        }
    }

    fn render_reference_modal(&mut self, ctx: &egui::Context) {
        let mut open = self.show_reference_modal;
        egui::Window::new("20-Species Bat & Bird Acoustic Reference Key & Shortcuts (F1)")
            .open(&mut open)
            .default_width(780.0)
            .show(ctx, |ui| {
                ui.label("Shortcuts: F1 = Reference Key | F5 = Save SQLite WAL Session | F9 = Load SQLite Session | Drag & Drop .wav/.zc onto window");
                ui.separator();
                egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    egui::Grid::new("ref_species_modal_grid")
                        .striped(true)
                        .num_columns(6)
                        .show(ui, |ui| {
                            for h in ["Code", "Species", "Domain", "Fc Mean (kHz)", "Dur (ms)", "Shape / Couplet"] {
                                ui.strong(h);
                            }
                            ui.end_row();
                            for sp in reference_species_catalog() {
                                ui.monospace(sp.code);
                                ui.label(format!("{} ({})", sp.common_name, sp.scientific_name));
                                ui.label(sp.domain.label());
                                ui.monospace(format!("{:.2} +/- {:.2}", sp.fc_mean_khz, sp.fc_sd_khz));
                                ui.monospace(format!("{:.1}", sp.dur_mean_ms));
                                ui.label(format!(
                                    "{} {}",
                                    sp.preferred_topology.code(),
                                    sp.sympatric_couplet.unwrap_or("")
                                ));
                                ui.end_row();
                            }
                        });
                });
            });
        self.show_reference_modal = open;
    }

    fn render_sessions_modal(&mut self, ctx: &egui::Context) {
        let mut open = self.show_sessions_modal;
        let mut chosen_id: Option<i64> = None;
        egui::Window::new("Saved SQLite WAL Bioacoustic Sessions (F9)")
            .open(&mut open)
            .default_width(660.0)
            .show(ctx, |ui| {
                if self.recent_sessions.is_empty() {
                    ui.label("No saved sessions yet — press F5 in the workstation to save one.");
                } else {
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        egui::Grid::new("sqlite_sessions_grid")
                            .striped(true)
                            .num_columns(7)
                            .show(ui, |ui| {
                                for h in ["ID", "Slot", "Source", "Calls", "K*", "ACI / NDSI", "Action"] {
                                    ui.strong(h);
                                }
                                ui.end_row();
                                for s in &self.recent_sessions {
                                    ui.monospace(format!("#{}", s.id));
                                    ui.label(&s.slot_name);
                                    ui.label(&s.source_name);
                                    ui.monospace(format!("{}", s.call_count));
                                    ui.monospace(format!("{}", s.optimal_k));
                                    ui.monospace(format!("{:.1} / {:+.2}", s.aci, s.ndsi));
                                    if ui.button("Load").clicked() {
                                        chosen_id = Some(s.id);
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                }
            });
        if let Some(id) = chosen_id {
            self.load_sqlite_session(id);
            open = false;
        }
        self.show_sessions_modal = open;
    }
}

trait UiSubheadingExt {
    fn subheading_label(&mut self, text: &str);
}

impl UiSubheadingExt for egui::Ui {
    fn subheading_label(&mut self, text: &str) {
        self.label(
            egui::RichText::new(text)
                .size(15.5)
                .strong()
                .color(Color32::from_rgb(148, 214, 200)),
        );
    }
}

pub fn load_any_bioacoustic_path(path: &Path) -> Result<BioAcousticRecording, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "zc" || ext == "zcf" {
        let seq = load_zc_dot_table(path)?;
        Ok(reconstruct_waveform_from_zc(&seq, 250_000))
    } else {
        load_wav_file(path)
    }
}

fn configure_dark_scientific_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(15, 19, 27);
    visuals.window_fill = Color32::from_rgb(18, 23, 33);
    ctx.set_visuals(visuals);
}
