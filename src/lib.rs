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
//! # SonoWing (`sonowing` v1.1.0)
//!
//! Native safe-Rust bat echolocation and bird vocalization sonic/ultrasonic waveform,
//! STFT spectrogram, unsupervised MFCC/UMAP-style manifold clustering, multi-harmonic
//! stack tracking, live microphone capture, GUANO v1.0 metadata, and ecoacoustic
//! soundscape workstation.

pub mod call_extractor;
pub mod dsp;
pub mod exporters;
pub mod gui;
pub mod harmonic_tracker;
pub mod live_mic;
pub mod mfcc_cluster;
pub mod soundscape_indices;
pub mod storage;
pub mod synth_presets;
pub mod taxa_library;
pub mod ultrasonic_audio;
pub mod wav_guano;

pub use exporters::{
    evaluate_batch_classifier_diagnostic, export_classifier_diagnostic_files,
    format_classifier_diagnostic_csv, BatchClassifierDiagnosticReport, ConfusionMatrixCell,
    PresetSnrDiagnosticRow,
};
