//! Bundled SQLite WAL (`rusqlite`) persistence repository (`sonowing_sessions.sqlite3`, `v1.1.0`)
//! storing analyzed recordings, extracted bat/bird call morphometrics, harmonic stack profiles,
//! MFCC/UMAP cluster assignments, manual vetting annotations, and ecoacoustic soundscape indices.

use crate::call_extractor::DetectedCallEvent;
use crate::exporters::WorksetPayload;
use crate::harmonic_tracker::HarmonicStackProfile;
use crate::mfcc_cluster::RepertoireClusterReport;
use crate::soundscape_indices::SoundscapeIndices;
use crate::wav_guano::GuanoMetadata;
use rusqlite::{params, Connection};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SavedSessionSummary {
    pub id: i64,
    pub slot_name: String,
    pub source_name: String,
    pub sample_rate_hz: u32,
    pub call_count: usize,
    pub optimal_k: usize,
    pub aci: f64,
    pub ndsi: f64,
    pub saved_at_utc: String,
}

pub struct SonoWingRepository {
    conn: Connection,
}

impl SonoWingRepository {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open SQLite DB '{}': {e}", db_path.display()))?;
        let _: String = conn
            .query_row("PRAGMA journal_mode=WAL;", [], |r| r.get(0))
            .unwrap_or_else(|_| "DELETE".to_string());
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_name TEXT NOT NULL,
                source_name TEXT NOT NULL,
                sample_rate_hz INTEGER NOT NULL,
                duration_s REAL NOT NULL,
                call_count INTEGER NOT NULL,
                optimal_k INTEGER NOT NULL DEFAULT 1,
                aci REAL NOT NULL,
                ndsi REAL NOT NULL,
                bio_index REAL NOT NULL,
                total_entropy_h REAL NOT NULL,
                payload_json TEXT NOT NULL,
                saved_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE TABLE IF NOT EXISTS vetted_calls_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                call_num INTEGER NOT NULL,
                domain TEXT NOT NULL,
                begin_time_s REAL NOT NULL,
                duration_ms REAL NOT NULL,
                f_char_khz REAL NOT NULL,
                f_max_khz REAL NOT NULL,
                species_code TEXT NOT NULL,
                confidence REAL NOT NULL,
                cluster_id INTEGER NOT NULL DEFAULT 0,
                hnr_db REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY(session_id) REFERENCES analysis_sessions(id) ON DELETE CASCADE
            );",
        )
        .map_err(|e| format!("Failed initializing SQLite schema: {e}"))?;

        Ok(Self { conn })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn save_session(
        &self,
        slot_name: &str,
        source_name: &str,
        sample_rate_hz: u32,
        duration_s: f64,
        guano: Option<&GuanoMetadata>,
        calls: &[DetectedCallEvent],
        harmonics: &[HarmonicStackProfile],
        clusters: &RepertoireClusterReport,
        indices: &SoundscapeIndices,
    ) -> Result<i64, String> {
        let payload = WorksetPayload {
            schema_version: "1.1.0".to_string(),
            source_name: source_name.to_string(),
            sample_rate_hz,
            duration_s,
            guano: guano.cloned(),
            detected_calls: calls.to_vec(),
            harmonic_profiles: harmonics.to_vec(),
            cluster_report: clusters.clone(),
            soundscape_indices: indices.clone(),
        };
        let payload_json =
            serde_json::to_string(&payload).map_err(|e| format!("Serialize session error: {e}"))?;

        self.conn
            .execute(
                "INSERT INTO analysis_sessions (
                    slot_name, source_name, sample_rate_hz, duration_s, call_count,
                    optimal_k, aci, ndsi, bio_index, total_entropy_h, payload_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    slot_name,
                    source_name,
                    sample_rate_hz,
                    duration_s,
                    calls.len() as i64,
                    clusters.optimal_k as i64,
                    indices.aci,
                    indices.ndsi,
                    indices.bio_index,
                    indices.total_entropy_h,
                    payload_json
                ],
            )
            .map_err(|e| format!("SQLite insert session failed: {e}"))?;

        let session_id = self.conn.last_insert_rowid();
        for (idx, ev) in calls.iter().enumerate() {
            let cid = clusters
                .points
                .get(idx)
                .map(|p| p.cluster_id as i64)
                .unwrap_or(0);
            let hnr = harmonics
                .get(idx)
                .map(|h| h.hnr_db as f64)
                .unwrap_or(0.0);
            let _ = self.conn.execute(
                "INSERT INTO vetted_calls_index (
                    session_id, call_num, domain, begin_time_s, duration_ms,
                    f_char_khz, f_max_khz, species_code, confidence, cluster_id, hnr_db
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    session_id,
                    ev.id as i64,
                    ev.domain.label(),
                    ev.begin_time_s,
                    ev.duration_ms,
                    ev.f_char_khz,
                    ev.f_max_khz,
                    ev.primary_code(),
                    ev.primary_confidence(),
                    cid,
                    hnr
                ],
            );
        }

        Ok(session_id)
    }

    pub fn list_recent_sessions(&self, limit: usize) -> Result<Vec<SavedSessionSummary>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, slot_name, source_name, sample_rate_hz, call_count, optimal_k, aci, ndsi, saved_at_utc
                 FROM analysis_sessions
                 ORDER BY id DESC
                 LIMIT ?1",
            )
            .map_err(|e| format!("SQLite prepare list failed: {e}"))?;

        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(SavedSessionSummary {
                    id: r.get(0)?,
                    slot_name: r.get(1)?,
                    source_name: r.get(2)?,
                    sample_rate_hz: r.get::<_, i64>(3)? as u32,
                    call_count: r.get::<_, i64>(4)? as usize,
                    optimal_k: r.get::<_, i64>(5).unwrap_or(1) as usize,
                    aci: r.get(6)?,
                    ndsi: r.get(7)?,
                    saved_at_utc: r.get(8)?,
                })
            })
            .map_err(|e| format!("SQLite query list failed: {e}"))?;

        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(|e| format!("Row decode error: {e}"))?);
        }
        Ok(out)
    }

    pub fn load_session_payload(&self, id: i64) -> Result<WorksetPayload, String> {
        let raw_json: String = self
            .conn
            .query_row(
                "SELECT payload_json FROM analysis_sessions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .map_err(|e| format!("Session #{id} not found: {e}"))?;

        serde_json::from_str(&raw_json).map_err(|e| format!("Corrupt session JSON: {e}"))
    }
}
