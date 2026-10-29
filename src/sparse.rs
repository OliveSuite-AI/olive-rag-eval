use crate::{
    clients::Timed,
    config::{Bm25Config, Config},
    data::{Chunk, RankedChunk, nfc},
    fingerprint,
    store::file_hash,
};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};
use unicode_segmentation::UnicodeSegmentation;

pub const TOKENIZER_VERSION: &str = "nfc-uax29-unicode-segmentation-1.13.3-lowercase-v1";
pub fn tokenize(text: &str) -> Vec<String> {
    nfc(text).unicode_words().map(str::to_lowercase).collect()
}
pub fn corpus_key(chunks: &[Chunk]) -> String {
    let content: Vec<_> = chunks
        .iter()
        .map(|c| {
            (
                &c.id,
                &c.text,
                &c.source,
                c.page,
                &c.language,
                &c.source_url,
            )
        })
        .collect();
    fingerprint(&(TOKENIZER_VERSION, content))
}
pub fn snapshot_dir(c: &Config, chunks: &[Chunk]) -> PathBuf {
    c.artifacts.join("sparse").join(corpus_key(chunks))
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    fingerprint: String,
    tokenizer: String,
    count: usize,
    database_sha256: String,
}

pub fn build(c: &Config, chunks: &[Chunk]) -> Result<PathBuf> {
    let dir = snapshot_dir(c, chunks);
    build_at(&dir, chunks)
}
pub fn build_at(dir: &Path, chunks: &[Chunk]) -> Result<PathBuf> {
    ensure!(!chunks.is_empty(), "corpus must not be empty");
    let mut ids = std::collections::HashSet::new();
    for c in chunks {
        ensure!(
            !c.id.is_empty()
                && ids.insert(&c.id)
                && !c.text.trim().is_empty()
                && c.page > 0
                && !c.source.is_empty()
                && !c.language.is_empty(),
            "invalid or duplicate sparse chunk"
        );
    }
    if dir.exists() {
        let existing = SparseIndex::load(dir)?;
        ensure!(
            existing.fingerprint == corpus_key(chunks),
            "existing sparse snapshot belongs to a different corpus"
        );
        return Ok(dir.to_path_buf());
    }
    let parent = dir.parent().unwrap();
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".building-{}-{}",
        std::process::id(),
        &corpus_key(chunks)[..16]
    ));
    std::fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        let mut db = Connection::open(staging.join("bm25.sqlite"))?;
        db.execute_batch("PRAGMA journal_mode=DELETE; CREATE TABLE documents(id TEXT PRIMARY KEY, payload TEXT NOT NULL, length INTEGER NOT NULL); CREATE TABLE postings(term TEXT NOT NULL, id TEXT NOT NULL, tf INTEGER NOT NULL, PRIMARY KEY(term,id));")?;
        let tx = db.transaction()?;
        for c in chunks {
            let terms = tokenize(&c.text);
            let mut frequencies = BTreeMap::<String, usize>::new();
            for term in &terms {
                *frequencies.entry(term.clone()).or_default() += 1;
            }
            let mut payload = c.clone();
            payload.embedding = None;
            payload.embedding_profile = None;
            tx.execute(
                "INSERT INTO documents VALUES (?1,?2,?3)",
                params![c.id, serde_json::to_string(&payload)?, terms.len()],
            )?;
            for (term, tf) in frequencies {
                tx.execute(
                    "INSERT INTO postings VALUES (?1,?2,?3)",
                    params![term, c.id, tf],
                )?;
            }
        }
        tx.commit()?;
        db.close().map_err(|(_, e)| e)?;
        let manifest = Manifest {
            version: 1,
            fingerprint: corpus_key(chunks),
            tokenizer: TOKENIZER_VERSION.into(),
            count: chunks.len(),
            database_sha256: file_hash(&staging.join("bm25.sqlite"))?,
        };
        std::fs::write(
            staging.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        for name in ["bm25.sqlite", "manifest.json"] {
            crate::persistence::sync_file(&staging.join(name))?;
        }
        crate::persistence::sync_directory(&staging)?;
        std::fs::rename(&staging, dir)?;
        crate::persistence::sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result?;
    Ok(dir.to_path_buf())
}

#[async_trait]
pub trait SparseRetriever {
    async fn retrieve(
        &self,
        query: &str,
        languages: &[String],
        parameters: &Bm25Config,
    ) -> Result<Timed<Vec<RankedChunk>>>;
}
pub struct SparseIndex {
    pub fingerprint: String,
    documents: BTreeMap<String, (Chunk, usize)>,
    postings: BTreeMap<String, Vec<(String, usize)>>,
}
impl SparseIndex {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
        ensure!(
            manifest.version == 1 && manifest.tokenizer == TOKENIZER_VERSION,
            "unsupported sparse snapshot/tokenizer version"
        );
        ensure!(
            file_hash(&dir.join("bm25.sqlite"))? == manifest.database_sha256,
            "corrupt sparse snapshot"
        );
        let db =
            Connection::open_with_flags(dir.join("bm25.sqlite"), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut stmt = db.prepare("SELECT id,payload,length FROM documents ORDER BY id")?;
        let mut documents = BTreeMap::new();
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, usize>(2)?,
            ))
        })? {
            let (id, payload, length) = row?;
            documents.insert(id, (serde_json::from_str(&payload)?, length));
        }
        ensure!(
            documents.len() == manifest.count,
            "sparse snapshot count mismatch"
        );
        let mut stmt = db.prepare("SELECT term,id,tf FROM postings ORDER BY term,id")?;
        let mut postings = BTreeMap::<String, Vec<(String, usize)>>::new();
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, usize>(2)?,
            ))
        })? {
            let (term, id, tf) = row?;
            ensure!(
                documents.contains_key(&id) && tf > 0,
                "invalid sparse posting"
            );
            postings.entry(term).or_default().push((id, tf));
        }
        Ok(Self {
            fingerprint: manifest.fingerprint,
            documents,
            postings,
        })
    }
    pub fn search(
        &self,
        query: &str,
        languages: &[String],
        p: &Bm25Config,
    ) -> Result<Timed<Vec<RankedChunk>>> {
        ensure!(
            p.k1.is_finite()
                && p.k1 > 0.
                && p.b.is_finite()
                && (0. ..=1.).contains(&p.b)
                && p.top_k > 0,
            "invalid BM25 parameters"
        );
        let start = Instant::now();
        let allowed: BTreeSet<_> = self
            .documents
            .iter()
            .filter(|(_, (c, _))| languages.is_empty() || languages.contains(&c.language))
            .map(|(id, _)| id.as_str())
            .collect();
        let terms: BTreeSet<_> = tokenize(query).into_iter().collect();
        let mut scores = BTreeMap::<String, f64>::new();
        let n = allowed.len() as f64;
        let avg = if n > 0. {
            allowed
                .iter()
                .map(|id| self.documents[*id].1 as f64)
                .sum::<f64>()
                / n
        } else {
            0.
        };
        if avg > 0. {
            for term in &terms {
                if let Some(postings) = self.postings.get(term) {
                    let scoped: Vec<_> = postings
                        .iter()
                        .filter(|(id, _)| allowed.contains(id.as_str()))
                        .collect();
                    let df = scoped.len() as f64;
                    let idf = (1. + (n - df + 0.5) / (df + 0.5)).ln();
                    for (id, tf) in scoped {
                        let f = *tf as f64;
                        let dl = self.documents[id].1 as f64;
                        *scores.entry(id.clone()).or_default() +=
                            idf * (f * (p.k1 + 1.)) / (f + p.k1 * (1. - p.b + p.b * dl / avg));
                    }
                }
            }
        }
        let mut sorted: Vec<_> = scores.into_iter().filter(|(_, s)| *s > 0.).collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted.truncate(p.top_k);
        let value = sorted
            .into_iter()
            .enumerate()
            .map(|(i, (id, score))| RankedChunk {
                chunk: self.documents[&id].0.clone(),
                vector_rank: None,
                vector_score: None,
                sparse_rank: Some(i + 1),
                sparse_score: Some(score),
                fused_rank: None,
                rrf_score: None,
                rank: i + 1,
                rerank_score: None,
                raw: Value::Null,
            })
            .collect();
        Ok(Timed {
            value,
            seconds: start.elapsed().as_secs_f64(),
            warmup_seconds: 0.,
            raw: json!({"allowed_count":allowed.len(),"unique_query_terms":terms.len(),"average_document_length":avg,"tokenizer":TOKENIZER_VERSION}),
        })
    }
}
#[async_trait]
impl SparseRetriever for SparseIndex {
    async fn retrieve(
        &self,
        query: &str,
        languages: &[String],
        parameters: &Bm25Config,
    ) -> Result<Timed<Vec<RankedChunk>>> {
        self.search(query, languages, parameters)
    }
}
