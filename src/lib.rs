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
