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

//! Unsupervised Bioacoustic Syllable & Pulse Mel-Frequency Cepstral Coefficient (`MFCC`)
//! Feature Extraction, Fuzzy Simplicial `k`-NN Manifold Projection (`UMAP`-style 2D layout),
//! and `k`-Means++ Clustering with Automatic Rousseeuw Silhouette `K*` Selection.
//!
//! Primary references:
//! - MFCC filterbank & DCT-II: Davis & Mermelstein (1980), IEEE Trans. ASSP 28(4), 357-366
//!   <https://courses.physics.illinois.edu/ece417/fa2017/davis80.pdf>
//! - UMAP fuzzy simplicial manifold projection: McInnes, Healy & Melville (2018)
//!   <https://arxiv.org/abs/1802.03426>
//! - Unsupervised animal vocal repertoire UMAP clustering: Sainburg, Thielk & Gentner (2020),
//!   PLOS Computational Biology 16(10): e1008228
//!   <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008228>

use crate::call_extractor::{DetectedCallEvent, TaxonDomain};
use crate::dsp::Spectrogram;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::PI;

pub const NUM_MEL_FILTERS: usize = 24;
pub const NUM_MFCC_COEFFS: usize = 13;
pub const NUM_DELTA_MFCC: usize = 5;
pub const FEATURE_DIM: usize = NUM_MFCC_COEFFS + NUM_DELTA_MFCC + 4; // 22-D joint cepstral-morphometric vector

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterCountMode {
    AutoSilhouette,
    Fixed(usize),
}

impl ClusterCountMode {
    pub fn label(self) -> String {
        match self {
            Self::AutoSilhouette => "Auto Optimal K* (Max Silhouette)".to_string(),
            Self::Fixed(k) => format!("Fixed K = {k} Clusters"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ManifoldClusterConfig {
    pub k_mode: ClusterCountMode,
    pub umap_n_neighbors: usize,
    pub umap_min_dist: f64,
}

impl Default for ManifoldClusterConfig {
    fn default() -> Self {
        Self {
            k_mode: ClusterCountMode::AutoSilhouette,
            umap_n_neighbors: 6,
            umap_min_dist: 0.12,
        }
    }
}

/// Per-call MFCC vector, 2D manifold coordinates, cluster assignment, and silhouette score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEmbeddingPoint {
    pub call_id: usize,
    pub call_index: usize,
    pub domain: TaxonDomain,
    pub species_code: String,
    /// `13` static Mel-Frequency Cepstral Coefficients (`c0..c12`).
    pub mfcc: [f64; NUM_MFCC_COEFFS],
    /// `5` onset-to-tail temporal delta-MFCCs (`dc1..dc5`).
    pub delta_mfcc: [f64; NUM_DELTA_MFCC],
    /// 2D UMAP-style fuzzy-simplicial manifold coordinates normalized to `[-1.0, +1.0]`.
    pub umap_x: f64,
    pub umap_y: f64,
    /// Assigned `0`-indexed unsupervised cluster (`0..k`).
    pub cluster_id: usize,
    /// Rousseeuw (1987) point silhouette score `s(i) in [-1.0, +1.0]`.
    pub silhouette: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSummary {
    pub cluster_id: usize,
    pub member_count: usize,
    pub dominant_species: String,
    pub mean_fc_khz: f64,
    pub mean_dur_ms: f64,
    pub mean_silhouette: f64,
    pub centroid_umap: (f64, f64),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepertoireClusterReport {
    pub optimal_k: usize,
    pub global_silhouette: f64,
    pub umap_n_neighbors: usize,
    pub points: Vec<CallEmbeddingPoint>,
    pub clusters: Vec<ClusterSummary>,
}

#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz.max(0.0) / 700.0).log10()
}

#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a `NUM_MEL_FILTERS` triangular Mel filterbank over `[f_min_hz, f_max_hz]`.
/// Uses O'Shaughnessy Mel scaling, seamlessly covering both audible (`0.4-12 kHz`) avian bands
/// and ultrasonic (`12-140 kHz`) chiropteran bands.
fn build_mel_filterbank(
    num_bins: usize,
    sample_rate_hz: u32,
    fft_size: usize,
    f_min_hz: f64,
    f_max_hz: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mel_lo = hz_to_mel(f_min_hz);
    let mel_hi = hz_to_mel(f_max_hz.max(f_min_hz + 200.0));
    let n_pts = NUM_MEL_FILTERS + 2;
    let mel_pts: Vec<f64> = (0..n_pts)
        .map(|i| mel_lo + (i as f64 / (n_pts - 1) as f64) * (mel_hi - mel_lo))
        .collect();
    let bin_pts: Vec<f64> = mel_pts
        .iter()
        .map(|&m| mel_to_hz(m) * fft_size as f64 / sample_rate_hz as f64)
        .collect();

    let mut bank = Vec::with_capacity(NUM_MEL_FILTERS);
    for m in 0..NUM_MEL_FILTERS {
        let left = bin_pts[m];
        let center = bin_pts[m + 1];
        let right = bin_pts[m + 2];
        let mut weights = Vec::new();
        let b_start = (left.floor() as usize).clamp(1, num_bins.saturating_sub(1));
        let b_end = (right.ceil() as usize).clamp(b_start, num_bins.saturating_sub(1));
        for b in b_start..=b_end {
            let bf = b as f64;
            let w = if bf < center {
                (bf - left) / (center - left).max(1e-6)
            } else {
                (right - bf) / (right - center).max(1e-6)
            };
            if w > 0.0 {
                weights.push((b, w.clamp(0.0, 1.0)));
            }
        }
        if weights.is_empty() {
            let fallback_b = (center.round() as usize).clamp(1, num_bins.saturating_sub(1));
            weights.push((fallback_b, 1.0));
        }
        bank.push(weights);
    }
    bank
}

/// Orthonormal Type-II Discrete Cosine Transform (`DCT-II`) mapping `24` log-Mel filter energies
/// into `13` Mel-Frequency Cepstral Coefficients (`c0..c12`).
fn dct_ii_13(log_mel: &[f64; NUM_MEL_FILTERS]) -> [f64; NUM_MFCC_COEFFS] {
    let mut out = [0.0_f64; NUM_MFCC_COEFFS];
    let n = NUM_MEL_FILTERS as f64;
    let scale0 = (1.0 / n).sqrt();
    let scale_k = (2.0 / n).sqrt();
    for (k, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for (m, &val) in log_mel.iter().enumerate() {
            let angle = PI * (k as f64) * (2.0 * (m as f64) + 1.0) / (2.0 * n);
            sum += val * angle.cos();
        }
        *slot = sum * if k == 0 { scale0 } else { scale_k };
    }
    out
}

/// Compute MFCC + delta-MFCC + morphometric feature vectors, project onto a 2D UMAP-style
/// fuzzy simplicial manifold, and partition calls via `k`-Means++ with Silhouette `K*` selection.
pub fn analyze_repertoire_clusters(
    calls: &[DetectedCallEvent],
    spec: &Spectrogram,
    cfg: ManifoldClusterConfig,
) -> RepertoireClusterReport {
    let n = calls.len();
    if n == 0 || spec.num_frames == 0 {
        return RepertoireClusterReport::default();
    }

    let nyquist_hz = spec.sample_rate_hz as f64 * 0.5;
    let bird_bank = build_mel_filterbank(
        spec.num_bins,
        spec.sample_rate_hz,
        spec.fft_size,
        400.0,
        nyquist_hz.min(12_500.0),
    );
    let bat_bank = build_mel_filterbank(
        spec.num_bins,
        spec.sample_rate_hz,
        spec.fft_size,
        13_500.0_f64.min(nyquist_hz * 0.4),
        nyquist_hz.min(150_000.0),
    );

    let mut static_mfccs = Vec::with_capacity(n);
    let mut delta_mfccs = Vec::with_capacity(n);
    let mut raw_features = vec![[0.0_f64; FEATURE_DIM]; n];

    for (idx, ev) in calls.iter().enumerate() {
        let bank = match ev.domain {
            TaxonDomain::BatChiroptera if nyquist_hz >= 16_000.0 => &bat_bank,
            _ => &bird_bank,
        };
        let f_start = ((ev.begin_time_s * spec.sample_rate_hz as f64) / spec.hop_size as f64)
            .floor()
            .max(0.0) as usize;
        let f_end = ((ev.end_time_s * spec.sample_rate_hz as f64) / spec.hop_size as f64)
            .ceil()
            .max(f_start as f64) as usize;
        let f0 = f_start.min(spec.num_frames - 1);
        let f1 = f_end.min(spec.num_frames - 1).max(f0);
        let f_mid = (f0 + f1) / 2;

        let mean_log_mel = pool_log_mel_frames(spec, bank, f0, f1);
        let early_log_mel = pool_log_mel_frames(spec, bank, f0, f_mid);
        let late_log_mel = pool_log_mel_frames(spec, bank, f_mid, f1);

        let mfcc = dct_ii_13(&mean_log_mel);
        let mfcc_early = dct_ii_13(&early_log_mel);
        let mfcc_late = dct_ii_13(&late_log_mel);

        let mut d_mfcc = [0.0_f64; NUM_DELTA_MFCC];
        for k in 0..NUM_DELTA_MFCC {
            d_mfcc[k] = mfcc_late[k + 1] - mfcc_early[k + 1];
        }

        // Populate 22-D joint representation (c1..c12 + c0_norm + dc1..dc5 + log_Fc + log_BW + log_Dur + Sc)
        for k in 0..NUM_MFCC_COEFFS {
            raw_features[idx][k] = if k == 0 { mfcc[0] * 0.25 } else { mfcc[k] };
        }
        for k in 0..NUM_DELTA_MFCC {
            raw_features[idx][NUM_MFCC_COEFFS + k] = d_mfcc[k];
        }
        raw_features[idx][18] = (ev.f_char_khz.max(0.2)).ln();
        raw_features[idx][19] = (ev.bandwidth_khz.max(0.05)).ln();
        raw_features[idx][20] = (ev.duration_ms.max(0.5)).ln();
        raw_features[idx][21] = ev.slope_char_khz_per_ms.clamp(-25.0, 25.0);

        static_mfccs.push(mfcc);
        delta_mfccs.push(d_mfcc);
    }

    // Z-score standardize columns across the repertoire so no single scale dominates
    let z_features = z_score_columns(&raw_features);
    let dist_matrix = pairwise_euclidean(&z_features);

    // Project onto 2D UMAP-style fuzzy simplicial manifold
    let coords_2d = project_umap_2d(
        &z_features,
        &dist_matrix,
        cfg.umap_n_neighbors.clamp(2, 15),
        cfg.umap_min_dist.clamp(0.01, 0.8),
    );

    // Run k-Means++ & select optimal K via Silhouette Score
    let (chosen_k, assignments, sil_per_point, global_sil) =
        cluster_with_silhouette(&z_features, &dist_matrix, cfg.k_mode);

    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        points.push(CallEmbeddingPoint {
            call_id: calls[i].id,
            call_index: i,
            domain: calls[i].domain,
            species_code: calls[i].primary_code().to_string(),
            mfcc: static_mfccs[i],
            delta_mfcc: delta_mfccs[i],
            umap_x: coords_2d[i].0,
            umap_y: coords_2d[i].1,
            cluster_id: assignments[i],
            silhouette: sil_per_point[i],
        });
    }

    // Build per-cluster summaries
    let mut clusters = Vec::with_capacity(chosen_k);
    for cid in 0..chosen_k {
        let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == cid).collect();
        if members.is_empty() {
            continue;
        }
        let mut spp_hist: BTreeMap<String, usize> = BTreeMap::new();
        let mut fc_sum = 0.0_f64;
        let mut dur_sum = 0.0_f64;
        let mut sil_sum = 0.0_f64;
        let mut ux_sum = 0.0_f64;
        let mut uy_sum = 0.0_f64;

        for &idx in &members {
            *spp_hist
                .entry(calls[idx].primary_code().to_string())
                .or_insert(0) += 1;
            fc_sum += calls[idx].f_char_khz;
            dur_sum += calls[idx].duration_ms;
            sil_sum += sil_per_point[idx];
            ux_sum += coords_2d[idx].0;
            uy_sum += coords_2d[idx].1;
        }
        let dom_spp = spp_hist
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k)
            .unwrap_or_else(|| "UNID".to_string());
        let cnt = members.len() as f64;
        clusters.push(ClusterSummary {
            cluster_id: cid,
            member_count: members.len(),
            dominant_species: dom_spp,
            mean_fc_khz: fc_sum / cnt,
            mean_dur_ms: dur_sum / cnt,
            mean_silhouette: sil_sum / cnt,
            centroid_umap: (ux_sum / cnt, uy_sum / cnt),
        });
    }

    RepertoireClusterReport {
        optimal_k: chosen_k,
        global_silhouette: global_sil,
        umap_n_neighbors: cfg.umap_n_neighbors,
        points,
        clusters,
    }
}

fn pool_log_mel_frames(
    spec: &Spectrogram,
    bank: &[Vec<(usize, f64)>],
    f0: usize,
    f1: usize,
) -> [f64; NUM_MEL_FILTERS] {
    let mut out = [0.0_f64; NUM_MEL_FILTERS];
    let frame_count = (f1 - f0 + 1).max(1) as f64;
    for (m, filt) in bank.iter().enumerate() {
        let mut e_acc = 0.0_f64;
        for fr in f0..=f1 {
            let base = fr * spec.num_bins;
            let mut band_p = 0.0_f64;
            for &(b, w) in filt {
                let mag = spec.mag_grid[base + b] as f64;
                band_p += w * mag * mag;
            }
            e_acc += (band_p + 1e-16).ln();
        }
        out[m] = e_acc / frame_count;
    }
    out
}

fn z_score_columns(raw: &[[f64; FEATURE_DIM]]) -> Vec<[f64; FEATURE_DIM]> {
    let n = raw.len();
    if n <= 1 {
        return raw.to_vec();
    }
    let mut means = [0.0_f64; FEATURE_DIM];
    let mut sds = [1.0_f64; FEATURE_DIM];
    for row in raw {
        for d in 0..FEATURE_DIM {
            means[d] += row[d];
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    for d in 0..FEATURE_DIM {
        let var: f64 = raw
            .iter()
            .map(|r| {
                let diff = r[d] - means[d];
                diff * diff
            })
            .sum::<f64>()
            / (n as f64);
        sds[d] = var.sqrt().max(1e-6);
    }
    raw.iter()
        .map(|r| {
            let mut z = [0.0_f64; FEATURE_DIM];
            for d in 0..FEATURE_DIM {
                z[d] = (r[d] - means[d]) / sds[d];
            }
            z
        })
        .collect()
}

fn pairwise_euclidean(feats: &[[f64; FEATURE_DIM]]) -> Vec<Vec<f64>> {
    let n = feats.len();
    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = feats[i]
                .iter()
                .zip(feats[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

/// Deterministic PCA initialization + UMAP fuzzy simplicial set force relaxation in 2D.
fn project_umap_2d(
    feats: &[[f64; FEATURE_DIM]],
    dist: &[Vec<f64>],
    n_neighbors: usize,
    min_dist: f64,
) -> Vec<(f64, f64)> {
    let n = feats.len();
    if n == 1 {
        return vec![(0.0, 0.0)];
    }
    if n == 2 {
        return vec![(-0.6, 0.0), (0.6, 0.0)];
    }

    // Step 1: Deterministic 2-component PCA spectral seed via deflated power iteration
    let mut coords = init_pca_2d(feats);

    // Step 2: Fuzzy simplicial k-NN membership strengths p_{i|j} (McInnes et al. 2018 Eq. 3)
    let k_eff = n_neighbors.min(n - 1).max(1);
    let target_sum = (k_eff as f64 + 1.0).log2();
    let mut cond_p = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist[i][j]))
            .collect();
        neighbors.sort_by(|a, b| a.1.total_cmp(&b.1));
        let knn = &neighbors[..k_eff];
        let rho = knn[0].1;

        // Binary search smooth bandwidth sigma_i
        let mut lo = 1e-4_f64;
        let mut hi = 64.0_f64;
        let mut sigma = 1.0_f64;
        for _ in 0..24 {
            sigma = 0.5 * (lo + hi);
            let psum: f64 = knn
                .iter()
                .map(|&(_, d)| (-(d - rho).max(0.0) / sigma).exp())
                .sum();
            if psum > target_sum {
                hi = sigma;
            } else {
                lo = sigma;
            }
        }
        for &(j, d) in knn {
            cond_p[i][j] = (-(d - rho).max(0.0) / sigma).exp();
        }
    }

    // Fuzzy union symmetrization: w_ij = p_{i|j} + p_{j|i} - p_{i|j}*p_{j|i}
    let mut sym_w = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let pij = cond_p[i][j];
            let pji = cond_p[j][i];
            let w = pij + pji - pij * pji;
            sym_w[i][j] = w;
            sym_w[j][i] = w;
        }
    }

    // Step 3: UMAP attractive + repulsive gradient descent epochs
    let a_param = 1.0 / (min_dist.max(0.02) * 4.5);
    let b_param = 0.895_f64;
    let epochs = 150usize;

    for ep in 0..epochs {
        let lr = 0.28 * (1.0 - ep as f64 / epochs as f64).max(0.05);
        let mut grad = vec![(0.0_f64, 0.0_f64); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let d2 = (dx * dx + dy * dy).max(1e-4);
                let w = sym_w[i][j];

                if w > 0.02 {
                    // Attractive force along fuzzy 1-simplices
                    let f_attr = -2.0 * a_param * b_param * d2.powf(b_param - 1.0)
                        / (1.0 + a_param * d2.powf(b_param))
                        * w;
                    let gx = (f_attr * dx).clamp(-4.0, 4.0);
                    let gy = (f_attr * dy).clamp(-4.0, 4.0);
                    grad[i].0 += gx;
                    grad[i].1 += gy;
                    grad[j].0 -= gx;
                    grad[j].1 -= gy;
                } else {
                    // Repulsive force between non-neighbor simplices
                    let rep_w = 1.0 - w;
                    let f_rep = (2.0 * b_param * rep_w)
                        / ((0.01 + d2) * (1.0 + a_param * d2.powf(b_param)));
                    let gx = (f_rep * dx * 0.18).clamp(-2.5, 2.5);
                    let gy = (f_rep * dy * 0.18).clamp(-2.5, 2.5);
                    grad[i].0 += gx;
                    grad[i].1 += gy;
                    grad[j].0 -= gx;
                    grad[j].1 -= gy;
                }
            }
        }

        for i in 0..n {
            coords[i].0 += lr * grad[i].0;
            coords[i].1 += lr * grad[i].1;
        }
    }

    // Normalize coordinates cleanly into [-0.90, +0.90]
    normalize_2d_coords(&mut coords);
    coords
}

fn init_pca_2d(feats: &[[f64; FEATURE_DIM]]) -> Vec<(f64, f64)> {
    let mut cov = [[0.0_f64; FEATURE_DIM]; FEATURE_DIM];
    for row in feats {
        for a in 0..FEATURE_DIM {
            for b in 0..FEATURE_DIM {
                cov[a][b] += row[a] * row[b];
            }
        }
    }

    let v1 = power_iteration_sym(&cov, None);
    let v2 = power_iteration_sym(&cov, Some(&v1));

    let mut pts: Vec<(f64, f64)> = feats
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let mut x = 0.0_f64;
            let mut y = 0.0_f64;
            for d in 0..FEATURE_DIM {
                x += r[d] * v1[d];
                y += r[d] * v2[d];
            }
            // Tiny deterministic index tie-breaker so identical synthetic calls never sit at d=0
            let jitter = (idx as f64) * 1e-3;
            (x + jitter.cos() * 0.01, y + jitter.sin() * 0.01)
        })
        .collect();
    normalize_2d_coords(&mut pts);
    pts
}

fn power_iteration_sym(
    cov: &[[f64; FEATURE_DIM]; FEATURE_DIM],
    ortho: Option<&[f64; FEATURE_DIM]>,
) -> [f64; FEATURE_DIM] {
    let mut v = [0.0_f64; FEATURE_DIM];
    for (d, val) in v.iter_mut().enumerate() {
        *val = if ortho.is_some() {
            ((d + 3) as f64 * 1.37).sin()
        } else {
            ((d + 1) as f64 * 0.73).cos()
        };
    }
    for _ in 0..36 {
        let mut next = [0.0_f64; FEATURE_DIM];
        for a in 0..FEATURE_DIM {
            for b in 0..FEATURE_DIM {
                next[a] += cov[a][b] * v[b];
            }
        }
        if let Some(u) = ortho {
            let dot: f64 = next.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            for d in 0..FEATURE_DIM {
                next[d] -= dot * u[d];
            }
        }
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for d in 0..FEATURE_DIM {
            v[d] = next[d] / norm;
        }
    }
    v
}

fn normalize_2d_coords(coords: &mut [(f64, f64)]) {
    if coords.is_empty() {
        return;
    }
    let n = coords.len() as f64;
    let mx = coords.iter().map(|p| p.0).sum::<f64>() / n;
    let my = coords.iter().map(|p| p.1).sum::<f64>() / n;
    let mut max_r = 1e-6_f64;
    for p in coords.iter_mut() {
        p.0 -= mx;
        p.1 -= my;
        max_r = max_r.max(p.0.abs()).max(p.1.abs());
    }
    let scale = 0.88 / max_r;
    for p in coords.iter_mut() {
        p.0 *= scale;
        p.1 *= scale;
    }
}

fn cluster_with_silhouette(
    feats: &[[f64; FEATURE_DIM]],
    dist: &[Vec<f64>],
    mode: ClusterCountMode,
) -> (usize, Vec<usize>, Vec<f64>, f64) {
    let n = feats.len();
    if n <= 2 {
        return (1, vec![0; n], vec![0.0; n], 0.0);
    }

    let k_candidates: Vec<usize> = match mode {
        ClusterCountMode::Fixed(k) => vec![k.clamp(1, (n - 1).max(1))],
        ClusterCountMode::AutoSilhouette => {
            let k_max = 6.min(n - 1).max(2);
            (2..=k_max).collect()
        }
    };

    let mut best_k = k_candidates[0];
    let mut best_assign = vec![0usize; n];
    let mut best_sils = vec![0.0_f64; n];
    let mut best_global = -2.0_f64;

    for k in k_candidates {
        if k <= 1 {
            return (1, vec![0; n], vec![0.0; n], 0.0);
        }
        let assign = run_kmeans_plusplus(feats, dist, k);
        let (sils, mean_sil) = compute_silhouette_scores(dist, &assign, k);
        if mean_sil > best_global {
            best_global = mean_sil;
            best_k = k;
            best_assign = assign;
            best_sils = sils;
        }
    }

    (best_k, best_assign, best_sils, best_global.clamp(-1.0, 1.0))
}

/// Deterministic farthest-first / k-Means++ centroid seeding followed by Lloyd iterations.
fn run_kmeans_plusplus(
    feats: &[[f64; FEATURE_DIM]],
    dist: &[Vec<f64>],
    k: usize,
) -> Vec<usize> {
    let n = feats.len();
    let mut centroids = Vec::with_capacity(k);
    centroids.push(feats[0]);

    let mut chosen_idx = vec![0usize];
    while centroids.len() < k {
        let mut best_pt = 0usize;
        let mut best_min_d = -1.0_f64;
        for i in 0..n {
            if chosen_idx.contains(&i) {
                continue;
            }
            let min_d = chosen_idx
                .iter()
                .map(|&c| dist[i][c])
                .fold(f64::INFINITY, f64::min);
            if min_d > best_min_d {
                best_min_d = min_d;
                best_pt = i;
            }
        }
        chosen_idx.push(best_pt);
        centroids.push(feats[best_pt]);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..40 {
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let d2: f64 = feats[i]
                    .iter()
                    .zip(cent.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = c;
                }
            }
            if assign[i] != best_c {
                assign[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![[0.0_f64; FEATURE_DIM]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            for d in 0..FEATURE_DIM {
                sums[c][d] += feats[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..FEATURE_DIM {
                    centroids[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
        }
    }
    assign
}

fn compute_silhouette_scores(
    dist: &[Vec<f64>],
    assign: &[usize],
    k: usize,
) -> (Vec<f64>, f64) {
    let n = assign.len();
    let mut sils = vec![0.0_f64; n];
    for i in 0..n {
        let ci = assign[i];
        let mut own_sum = 0.0_f64;
        let mut own_cnt = 0usize;
        let mut other_sums = vec![0.0_f64; k];
        let mut other_cnts = vec![0usize; k];

        for j in 0..n {
            if i == j {
                continue;
            }
            let cj = assign[j];
            if cj == ci {
                own_sum += dist[i][j];
                own_cnt += 1;
            } else {
                other_sums[cj] += dist[i][j];
                other_cnts[cj] += 1;
            }
        }

        if own_cnt == 0 {
            sils[i] = 0.0;
            continue;
        }
        let a_i = own_sum / own_cnt as f64;
        let b_i = (0..k)
            .filter(|&c| c != ci && other_cnts[c] > 0)
            .map(|c| other_sums[c] / other_cnts[c] as f64)
            .fold(f64::INFINITY, f64::min);

        if b_i.is_finite() && a_i.max(b_i) > 1e-12 {
            sils[i] = (b_i - a_i) / a_i.max(b_i);
        }
    }
    let mean_s = sils.iter().sum::<f64>() / (n.max(1) as f64);
    (sils, mean_s)
}
