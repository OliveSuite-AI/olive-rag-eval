use crate::{
    clients::Timed,
    config::{Config, Model},
    data::{Chunk, RankedChunk},
    fingerprint,
};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};
use turbovec::IdMapIndex;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub fingerprint: String,
    pub model: Model,
    pub corpus_hash: String,
    pub index_bits: usize,
    pub normalization: String,
    pub count: usize,
    pub index_sha256: String,
    pub metadata_sha256: String,
    pub build_evidence: serde_json::Value,
}
pub fn snapshot_key(chunks: &[Chunk], model: &Model, bits: usize) -> String {
    fingerprint(&(1, chunks, model, bits, "NFC+L2"))
}
pub fn snapshot_dir(c: &Config, m: &Model, chunks: &[Chunk]) -> PathBuf {
    c.artifacts
        .join("indexes")
        .join(&m.name)
        .join(snapshot_key(chunks, m, c.index_bits))
}
pub fn file_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
pub fn build_snapshot(
    c: &Config,
    m: &Model,
    chunks: &[Chunk],
    vectors: &[Vec<f32>],
    evidence: serde_json::Value,
) -> Result<PathBuf> {
    ensure!(chunks.len() == vectors.len(), "vector count mismatch");
    let dir = snapshot_dir(c, m, chunks);
    ensure!(!dir.exists(), "snapshot already exists");
    let parent = dir.parent().unwrap();
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".building-{}-{}",
        std::process::id(),
        &snapshot_key(chunks, m, c.index_bits)[..16]
    ));
    ensure!(
        !staging.exists(),
        "stale snapshot staging directory: {}",
        staging.display()
    );
    std::fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        let mut flat = Vec::new();
        for vector in vectors {
            let mut v = vector.clone();
            crate::data::normalize(&mut v, m.dimensions)?;
            flat.extend(v);
        }
        let mut index = IdMapIndex::new(m.dimensions, c.index_bits)?;
        let ids: Vec<u64> = (1..=chunks.len() as u64).collect();
        index.add_with_ids(&flat, &ids)?;
        index.write(staging.join("index.tvim"))?;
        let mut db = Connection::open(staging.join("chunks.sqlite"))?;
        db.execute_batch("PRAGMA journal_mode=DELETE; CREATE TABLE chunks (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);")?;
        let tx = db.transaction()?;
        for (id, chunk) in ids.iter().zip(chunks) {
            tx.execute(
                "INSERT INTO chunks VALUES (?1, ?2)",
                params![id, serde_json::to_string(chunk)?],
            )?;
        }
        tx.commit()?;
        db.close().map_err(|(_, e)| e)?;
        let manifest = Manifest {
            version: 1,
            fingerprint: snapshot_key(chunks, m, c.index_bits),
            model: m.clone(),
            corpus_hash: fingerprint(&chunks),
            index_bits: c.index_bits,
            normalization: "NFC+L2".into(),
            count: chunks.len(),
            index_sha256: file_hash(&staging.join("index.tvim"))?,
            metadata_sha256: file_hash(&staging.join("chunks.sqlite"))?,
            build_evidence: evidence,
        };
        let manifest_path = staging.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        for path in ["index.tvim", "chunks.sqlite", "manifest.json"] {
            crate::persistence::sync_file(&staging.join(path))?;
        }
        crate::persistence::sync_directory(&staging)?;
        std::fs::rename(&staging, &dir)?;
        crate::persistence::sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result?;
    Ok(dir)
}

pub struct Snapshot {
    pub manifest: Manifest,
    index: IdMapIndex,
    chunks: BTreeMap<u64, Chunk>,
}
impl Snapshot {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
        ensure!(manifest.version == 1, "unsupported snapshot version");
        ensure!(
            file_hash(&dir.join("index.tvim"))? == manifest.index_sha256
                && file_hash(&dir.join("chunks.sqlite"))? == manifest.metadata_sha256,
            "corrupted snapshot"
        );
        let index = IdMapIndex::load(dir.join("index.tvim"))?;
        index.prepare();
        let db = Connection::open_with_flags(
            dir.join("chunks.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let mut stmt = db.prepare("SELECT id, payload FROM chunks ORDER BY id")?;
        let entries = stmt.query_map([], |r| Ok((r.get::<_, u64>(0)?, r.get::<_, String>(1)?)))?;
        let mut chunks = BTreeMap::new();
        for entry in entries {
            let (id, payload) = entry?;
            chunks.insert(id, serde_json::from_str(&payload)?);
        }
        ensure!(
            chunks.len() == manifest.count && index.len() == manifest.count,
            "snapshot count mismatch"
        );
        Ok(Self {
            manifest,
            index,
            chunks,
        })
    }
    pub fn search(
        &self,
        vector: &[f32],
        languages: &[String],
        k: usize,
    ) -> Result<Timed<Vec<RankedChunk>>> {
        let mut vector = vector.to_vec();
        crate::data::normalize(&mut vector, self.manifest.model.dimensions)?;
        let allowed: Vec<_> = self
            .chunks
            .iter()
            .filter(|(_, c)| languages.is_empty() || languages.contains(&c.language))
            .map(|(&id, _)| id)
            .collect();
        if allowed.is_empty() {
            return Ok(Timed {
                value: Vec::new(),
                seconds: 0.,
                warmup_seconds: 0.,
                raw: serde_json::json!({"allowed_count":0}),
            });
        }
        let start = Instant::now();
        let (scores, ids) = self
            .index
            .search_with_allowlist(&vector, k, Some(&allowed))?;
        let seconds = start.elapsed().as_secs_f64();
        let mut entries: Vec<_> = scores.into_iter().zip(ids).collect();
        ensure!(
            entries.iter().all(|(s, _)| s.is_finite()),
            "nonfinite TurboVec score"
        );
        entries.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let value = entries
            .into_iter()
            .enumerate()
            .map(|(i, (score, id))| {
                Ok(RankedChunk {
                    chunk: self
                        .chunks
                        .get(&id)
                        .context("index refers to unknown chunk")?
                        .clone(),
                    vector_rank: Some(i + 1),
                    vector_score: Some(score),
                    rank: i + 1,
                    rerank_score: None,
                    sparse_rank: None,
                    sparse_score: None,
                    fused_rank: None,
                    rrf_score: None,
                    raw: serde_json::Value::Null,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Timed {
            value,
            seconds,
            warmup_seconds: 0.,
            raw: serde_json::json!({"allowed_count":allowed.len()}),
        })
    }
}

pub struct StageStore {
    db: Connection,
}
impl StageStore {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS stages (key TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);")?;
        Ok(Self { db })
    }
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let payload: Option<String> = self
            .db
            .query_row(
                "SELECT payload FROM stages WHERE key=?1 AND status='completed'",
                [key],
                |r| r.get(0),
            )
            .optional()?;
        payload
            .map(|p| serde_json::from_str(&p).map_err(Into::into))
            .transpose()
    }
    pub fn put(
        &mut self,
        key: &str,
        kind: &str,
        status: &str,
        value: &impl Serialize,
    ) -> Result<()> {
        let tx = self.db.transaction()?;
        tx.execute("INSERT INTO stages VALUES (?1,?2,?3,?4) ON CONFLICT(key) DO UPDATE SET status=excluded.status,payload=excluded.payload", params![key,kind,status,serde_json::to_string(value)?])?;
        tx.commit()?;
        Ok(())
    }
}
