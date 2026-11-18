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

//! Interoperable bioacoustic exporters (`v1.1.0`):
//! 1. Cornell Lab **Raven Pro 1.6 Selection Table (`.selections.txt` TSV)** preserving mandatory
//!    leftmost column order (`Selection`, `View`, `Channel`, `Begin Time (s)`, `End Time (s)`,
//!    `Low Freq (Hz)`, `High Freq (Hz)`), augmented with Harmonic `f0`/`HNR` and MFCC/UMAP cluster IDs.
//! 2. **NABat / SonoBat & Avian Pass/Pulse Summary RFC 4180 CSV** with call, harmonic, cluster, and soundscape summary rows.
//! 3. **SHA-256 Verified `.sonoworkset.json` Portable Session Envelope (`v1.1.0`, backward-compatible with `v1.0.0`)**.

use crate::call_extractor::{BehavioralPhase, DetectedCallEvent, TaxonDomain};
use crate::harmonic_tracker::HarmonicStackProfile;
use crate::mfcc_cluster::RepertoireClusterReport;
use crate::soundscape_indices::SoundscapeIndices;
use crate::wav_guano::GuanoMetadata;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

pub fn format_raven_selection_table(
    source_file: &str,
    events: &[DetectedCallEvent],
    harmonics: &[HarmonicStackProfile],
    clusters: &RepertoireClusterReport,
) -> String {
    let mut out = String::new();
    out.push_str(
        "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\t\
Delta Time (s)\tPeak Freq (Hz)\tFc (kHz)\tFk (kHz)\tSlope Char (kHz/ms)\tSNR (dB)\t\
Fundamental f0 (kHz)\tActive Harmonics\tDom Harmonic\tHNR (dB)\tMFCC Cluster\tUMAP X\tUMAP Y\t\
Domain\tTopology\tPhase\tSpecies Auto ID\tConfidence\tCouplet\tManual Vetted ID\tBegin File\n",
    );

    for (idx, ev) in events.iter().enumerate() {
        let sel = idx + 1;
        let begin_s = sanitize_finite(ev.begin_time_s);
        let end_s = sanitize_finite(ev.end_time_s.max(begin_s + 0.0001));
        let lo_hz = sanitize_finite(ev.f_min_khz * 1000.0);
        let hi_hz = sanitize_finite((ev.f_max_khz * 1000.0).max(lo_hz + 1.0));
        let dt_s = end_s - begin_s;
        let peak_hz = sanitize_finite(ev.f_peak_khz * 1000.0);
        let fc_khz = sanitize_finite(ev.f_char_khz);
        let fk_khz = sanitize_finite(ev.f_knee_khz);
        let sc = sanitize_finite(ev.slope_char_khz_per_ms);
        let snr = sanitize_finite(ev.snr_db as f64);

        let (f0_khz, h_cnt, dom_h, hnr_db) = harmonics
            .get(idx)
            .map(|h| {
                (
                    sanitize_finite(h.fundamental_f0_khz),
                    h.active_harmonic_count,
                    h.dominant_harmonic_order,
                    sanitize_finite(h.hnr_db as f64),
                )
            })
            .unwrap_or((fc_khz, 1, 1, 0.0));

        let (cid_str, ux, uy) = clusters
            .points
            .get(idx)
            .map(|pt| {
                (
                    format!("C{}", pt.cluster_id + 1),
                    sanitize_finite(pt.umap_x),
                    sanitize_finite(pt.umap_y),
                )
            })
            .unwrap_or_else(|| ("C1".to_string(), 0.0, 0.0));

        let auto_id = ev
            .top_matches
            .first()
            .map(|m| m.code.as_str())
            .unwrap_or("UNID");
        let conf = sanitize_finite(ev.primary_confidence());
        let couplet = ev.ambiguity_couplet.as_deref().unwrap_or("-");
        let manual = ev.manual_vetted_code.as_deref().unwrap_or("-");
        let clean_file = source_file.replace(['\t', '\r', '\n'], "_");

        out.push_str(&format!(
            "{sel}\tSpectrogram 1\t1\t{begin_s:.6}\t{end_s:.6}\t{lo_hz:.1}\t{hi_hz:.1}\t\
{dt_s:.6}\t{peak_hz:.1}\t{fc_khz:.3}\t{fk_khz:.3}\t{sc:.4}\t{snr:.2}\t\
{f0_khz:.3}\t{h_cnt}\t{dom_h}\t{hnr_db:.2}\t{cid_str}\t{ux:.4}\t{uy:.4}\t\
{}\t{}\t{}\t{auto_id}\t{conf:.4}\t{couplet}\t{manual}\t{clean_file}\n",
            ev.domain.label(),
            ev.topology.code(),
            ev.phase.label()
        ));
    }
    out
}

pub fn write_raven_selection_table(
    path: &Path,
    source_file: &str,
    events: &[DetectedCallEvent],
    harmonics: &[HarmonicStackProfile],
    clusters: &RepertoireClusterReport,
) -> io::Result<()> {
    fs::write(
        path,
        format_raven_selection_table(source_file, events, harmonics, clusters),
    )
}

pub fn format_calls_and_summary_csv(
    source_file: &str,
    sample_rate_hz: u32,
    events: &[DetectedCallEvent],
    harmonics: &[HarmonicStackProfile],
    clusters: &RepertoireClusterReport,
    indices: &SoundscapeIndices,
) -> String {
    let mut out = String::new();
    out.push_str(
        "record_type,source_file,sample_rate_hz,call_id,domain,begin_time_s,end_time_s,\
duration_ms,ipi_ms,calls_per_sec,f_max_khz,f_min_khz,f_peak_khz,f_char_khz,f_knee_khz,\
slope_init_khz_ms,slope_char_khz_ms,wiener_entropy,snr_db,f0_khz,active_harmonics,\
dominant_harmonic,hnr_db,mfcc_cluster,umap_x,umap_y,silhouette,topology,phase,\
auto_species_code,confidence,ambiguity_couplet,manual_vetted_code\r\n",
    );

    let mut bat_count = 0usize;
    let mut bird_count = 0usize;
    let mut buzz_count = 0usize;
    let mut species_counts: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, ev) in events.iter().enumerate() {
        match ev.domain {
            TaxonDomain::BatChiroptera => bat_count += 1,
            TaxonDomain::BirdAves => bird_count += 1,
        }
        if ev.phase == BehavioralPhase::TerminalFeedingBuzz {
            buzz_count += 1;
        }
        *species_counts
            .entry(ev.primary_code().to_string())
            .or_insert(0) += 1;

        let ipi_str = ev.ipi_ms.map(|v| format!("{v:.2}")).unwrap_or_default();
        let cps_str = ev
            .calls_per_sec
            .map(|v| format!("{v:.2}"))
            .unwrap_or_default();
        let auto_code = ev
            .top_matches
            .first()
            .map(|m| m.code.as_str())
            .unwrap_or("UNID");

        let (f0_s, hc_s, dh_s, hnr_s) = harmonics
            .get(idx)
            .map(|h| {
                (
                    format!("{:.3}", h.fundamental_f0_khz),
                    h.active_harmonic_count.to_string(),
                    h.dominant_harmonic_order.to_string(),
                    format!("{:.2}", h.hnr_db),
                )
            })
            .unwrap_or_else(|| (format!("{:.3}", ev.f_char_khz), "1".into(), "1".into(), "0.00".into()));

        let (cl_s, ux_s, uy_s, sil_s) = clusters
            .points
            .get(idx)
            .map(|pt| {
                (
                    format!("C{}", pt.cluster_id + 1),
                    format!("{:.4}", pt.umap_x),
                    format!("{:.4}", pt.umap_y),
                    format!("{:.4}", pt.silhouette),
                )
            })
            .unwrap_or_else(|| ("C1".into(), "0.0000".into(), "0.0000".into(), "0.0000".into()));

        let cols = [
            "CALL_EVENT".to_string(),
            csv_escape(source_file),
            sample_rate_hz.to_string(),
            ev.id.to_string(),
            csv_escape(ev.domain.label()),
            format!("{:.6}", ev.begin_time_s),
            format!("{:.6}", ev.end_time_s),
            format!("{:.3}", ev.duration_ms),
            ipi_str,
            cps_str,
            format!("{:.3}", ev.f_max_khz),
            format!("{:.3}", ev.f_min_khz),
            format!("{:.3}", ev.f_peak_khz),
            format!("{:.3}", ev.f_char_khz),
            format!("{:.3}", ev.f_knee_khz),
            format!("{:.4}", ev.slope_init_khz_per_ms),
            format!("{:.4}", ev.slope_char_khz_per_ms),
            format!("{:.4}", ev.wiener_entropy),
            format!("{:.2}", ev.snr_db),
            f0_s,
            hc_s,
            dh_s,
            hnr_s,
            cl_s,
            ux_s,
            uy_s,
            sil_s,
            csv_escape(ev.topology.code()),
            csv_escape(ev.phase.label()),
            csv_escape(auto_code),
            format!("{:.4}", ev.primary_confidence()),
            csv_escape(ev.ambiguity_couplet.as_deref().unwrap_or("")),
            csv_escape(ev.manual_vetted_code.as_deref().unwrap_or("")),
        ];
        out.push_str(&cols.join(","));
        out.push_str("\r\n");
    }

    out.push_str(&format!(
        "SUMMARY_TOTALS,{},{},{},All_Taxa,,,{:.3},,,,,,,,,,,,,,,,K*={},,,{:.4},FeedingBuzz={},{},ALL,1.0000,,\r\n",
        csv_escape(source_file),
        sample_rate_hz,
        events.len(),
        events.iter().map(|e| e.duration_ms).sum::<f64>(),
        clusters.optimal_k,
        clusters.global_silhouette,
        buzz_count,
        format!("Bat={bat_count};Bird={bird_count}")
    ));

    for cl in &clusters.clusters {
        out.push_str(&format!(
            "SUMMARY_CLUSTER,{},{},{},MFCC_UMAP_Cluster,,,,,,,,,{:.3},,,,,,,,,,C{},{:.4},{:.4},{:.4},,,{},1.0000,,\r\n",
            csv_escape(source_file),
            sample_rate_hz,
            cl.member_count,
            cl.mean_fc_khz,
            cl.cluster_id + 1,
            cl.centroid_umap.0,
            cl.centroid_umap.1,
            cl.mean_silhouette,
            csv_escape(&cl.dominant_species)
        ));
    }

    for (spp, cnt) in &species_counts {
        out.push_str(&format!(
            "SUMMARY_SPECIES,{},{},{},Species_Count,,,,,,,,,,,,,,,,,,,,,,,,,{},,,\r\n",
            csv_escape(source_file),
            sample_rate_hz,
            cnt,
            csv_escape(spp)
        ));
    }

    out.push_str(&format!(
        "SUMMARY_SOUNDSCAPE,{},{},0,Ecoacoustic_Indices,,,,,,,,,,,,,{:.4},,,,,,,,,,,{},ACI={:.2};NDSI={:.4};BIO={:.2};ADI={:.4};AEI={:.4},,,\r\n",
        csv_escape(source_file),
        sample_rate_hz,
        indices.total_entropy_h,
        format!("UltraShare={:.3}", indices.ultrasonic_energy_fraction),
        indices.aci,
        indices.ndsi,
        indices.bio_index,
        indices.adi,
        indices.aei
    ));

    out
}

pub fn write_calls_csv(
    path: &Path,
    source_file: &str,
    sample_rate_hz: u32,
    events: &[DetectedCallEvent],
    harmonics: &[HarmonicStackProfile],
    clusters: &RepertoireClusterReport,
    indices: &SoundscapeIndices,
) -> io::Result<()> {
    fs::write(
        path,
        format_calls_and_summary_csv(
            source_file,
            sample_rate_hz,
            events,
            harmonics,
            clusters,
            indices,
        ),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorksetPayload {
    pub schema_version: String,
    pub source_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_code: Option<String>,
    pub sample_rate_hz: u32,
    pub duration_s: f64,
    pub guano: Option<GuanoMetadata>,
    pub detected_calls: Vec<DetectedCallEvent>,
    #[serde(default)]
    pub harmonic_profiles: Vec<HarmonicStackProfile>,
    #[serde(default)]
    pub cluster_report: RepertoireClusterReport,
    pub soundscape_indices: SoundscapeIndices,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedWorksetEnvelope {
    pub payload: WorksetPayload,
    pub sha256_hex: String,
}

impl SignedWorksetEnvelope {
    pub fn build(payload: WorksetPayload) -> Result<Self, String> {
        let raw =
            serde_json::to_string(&payload).map_err(|e| format!("JSON serialize error: {e}"))?;
        let normalized: WorksetPayload =
            serde_json::from_str(&raw).map_err(|e| format!("JSON normalize error: {e}"))?;
        let canonical = Self::canonical_payload_json(&normalized)?;
        let sha256_hex = sha256_string(&canonical);
        Ok(Self {
            payload: normalized,
            sha256_hex,
        })
    }

    pub fn verify(&self) -> Result<bool, String> {
        let canonical = Self::canonical_payload_json(&self.payload)?;
        Ok(sha256_string(&canonical) == self.sha256_hex)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let value = serde_json::to_value(self)
            .map_err(|e| format!("JSON formatting error: {e}"))?;
        let pretty = serde_json::to_string_pretty(&value)
            .map_err(|e| format!("JSON formatting error: {e}"))?;
        fs::write(path, pretty)
            .map_err(|e| format!("Failed writing workset '{}': {e}", path.display()))
    }

    pub fn load_verified(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("Failed reading workset '{}': {e}", path.display()))?;
        let env: Self =
            serde_json::from_str(&text).map_err(|e| format!("Invalid workset JSON: {e}"))?;
        if !env.verify()? {
            return Err("Workset SHA-256 integrity verification failed".to_string());
        }
        Ok(env)
    }

    pub fn canonical_payload_json(payload: &WorksetPayload) -> Result<String, String> {
        let mut val =
            serde_json::to_value(payload).map_err(|e| format!("JSON serialize error: {e}"))?;
        canonicalize_json_value(&mut val);
        serde_json::to_string(&val).map_err(|e| format!("JSON serialize error: {e}"))
    }
}

fn canonicalize_json_value(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Number(num) => {
            if num.is_f64() {
                if let Some(f) = num.as_f64() {
                    if !f.is_finite() || f == 0.0 {
                        *num = serde_json::Number::from(0);
                    } else {
                        let rounded = format!("{f:.11e}");
                        if let Ok(r) = rounded.parse::<f64>() {
                            if let Some(rn) = serde_json::Number::from_f64(r) {
                                *num = rn;
                            }
                        }
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                canonicalize_json_value(item);
            }
        }
        serde_json::Value::Object(map) => {
            for val in map.values_mut() {
                canonicalize_json_value(val);
            }
        }
        _ => {}
    }
}

fn sha256_string(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn sanitize_finite(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call_extractor::{extract_calls, DetectorConfig};
    use crate::dsp::{compute_stft, StftConfig};
    use crate::harmonic_tracker::analyze_harmonic_stacks;
    use crate::mfcc_cluster::{analyze_repertoire_clusters, ManifoldClusterConfig};
    use crate::soundscape_indices::compute_soundscape_indices;
    use crate::synth_presets::{synthesize_preset, BuiltinPreset};

    #[test]
    fn workset_canonical_roundtrip_preserves_sha256() {
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
        let clusters =
            analyze_repertoire_clusters(&calls, &spec, ManifoldClusterConfig::default());
        let indices = compute_soundscape_indices(&rec.samples, &spec);

        let payload = WorksetPayload {
            schema_version: "1.1.0".to_string(),
            source_name: rec.source_name.clone(),
            source_path: None,
            preset_code: Some("B1".to_string()),
            sample_rate_hz: rec.sample_rate_hz,
            duration_s: rec.duration_s(),
            guano: rec.guano.clone(),
            detected_calls: calls,
            harmonic_profiles: harmonics,
            cluster_report: clusters,
            soundscape_indices: indices,
        };
        let env = SignedWorksetEnvelope::build(payload).unwrap();
        let before = SignedWorksetEnvelope::canonical_payload_json(&env.payload).unwrap();
        let ws_path = std::env::temp_dir().join("sonowing_exporters_roundtrip_check.sonoworkset.json");
        let _ = std::fs::remove_file(&ws_path);
        env.save_to_file(&ws_path).unwrap();
        let loaded = SignedWorksetEnvelope::load_verified(&ws_path).unwrap();
        let after = SignedWorksetEnvelope::canonical_payload_json(&loaded.payload).unwrap();
        let _ = std::fs::remove_file(&ws_path);
        assert_eq!(before, after);
    }
}
