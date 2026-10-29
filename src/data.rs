use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub source: String,
    pub page: u32,
    pub language: String,
    pub source_url: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub embedding_profile: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelevantPage {
    pub source: String,
    pub page: u32,
    pub grade: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub semantic_case_id: String,
    pub query: String,
    pub query_language: String,
    pub query_script: String,
    pub answer_language: String,
    pub corpus_languages: Vec<String>,
    pub question_type: String,
    pub review_status: String,
    pub expected_answer: String,
    pub key_facts: Vec<String>,
    pub relevant_pages: Vec<RelevantPage>,
    pub unanswerable: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationExample {
    pub case: Case,
    pub context: Vec<Chunk>,
    pub answer: String,
    pub label_review_status: String,
    pub expected_judgment: Option<crate::metrics::Judgment>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedChunk {
    pub chunk: Chunk,
    pub vector_rank: Option<usize>,
    pub vector_score: Option<f32>,
    #[serde(default)]
    pub sparse_rank: Option<usize>,
    #[serde(default)]
    pub sparse_score: Option<f64>,
    #[serde(default)]
    pub fused_rank: Option<usize>,
    #[serde(default)]
    pub rrf_score: Option<f64>,
    pub rank: usize,
    pub rerank_score: Option<f64>,
    #[serde(default)]
    pub raw: serde_json::Value,
}

pub fn nfc(s: &str) -> String {
    s.nfc().collect()
}
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    std::fs::read_to_string(path)?
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))
        })
        .collect()
}
pub fn load(corpus: &Path, dataset: &Path) -> Result<(Vec<Chunk>, Vec<Case>)> {
    let chunks: Vec<Chunk> = read_jsonl(corpus)?;
    let cases: Vec<Case> = read_jsonl(dataset)?;
    ensure!(
        !chunks.is_empty() && !cases.is_empty(),
        "corpus and dataset must not be empty"
    );
    let mut ids = HashSet::new();
    for c in &chunks {
        ensure!(
            !c.id.is_empty() && ids.insert(&c.id),
            "duplicate/empty chunk id: {}",
            c.id
        );
        ensure!(
            !c.text.trim().is_empty()
                && !c.source.is_empty()
                && !c.language.is_empty()
                && c.page > 0,
            "invalid chunk {}",
            c.id
        );
    }
    ids.clear();
    for c in &cases {
        ensure!(
            !c.id.is_empty() && ids.insert(&c.id),
            "duplicate/empty case id"
        );
        ensure!(
            !c.query.trim().is_empty()
                && !c.semantic_case_id.is_empty()
                && !c.query_language.is_empty()
                && !c.answer_language.is_empty()
                && !c.query_script.is_empty(),
            "missing query metadata: {}",
            c.id
        );
        ensure!(
            !c.corpus_languages.is_empty()
                && c.corpus_languages.iter().collect::<HashSet<_>>().len()
                    == c.corpus_languages.len(),
            "invalid corpus language scope"
        );
        ensure!(
            matches!(c.review_status.as_str(), "draft" | "reviewed"),
            "review_status must be draft or reviewed"
        );
        ensure!(
            c.unanswerable == c.relevant_pages.is_empty(),
            "unanswerable must agree with relevance labels: {}",
            c.id
        );
        ensure!(
            !c.key_facts.is_empty() && !c.expected_answer.is_empty(),
            "reference facts/answer required"
        );
        let mut pages = HashSet::new();
        for p in &c.relevant_pages {
            ensure!(
                (1..=2).contains(&p.grade) && pages.insert((&p.source, p.page)),
                "duplicate/invalid relevance label"
            );
            ensure!(
                chunks.iter().any(|x| x.source == p.source
                    && x.page == p.page
                    && c.corpus_languages.contains(&x.language)),
                "relevance page absent from scoped corpus: {} {}:{}",
                c.id,
                p.source,
                p.page
            );
        }
    }
    Ok((chunks, cases))
}

pub fn normalize(v: &mut [f32], dim: usize) -> Result<()> {
    ensure!(
        v.len() == dim && v.iter().all(|x| x.is_finite()),
        "invalid embedding dimensions/values"
    );
    let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    ensure!(norm > 1e-10, "zero embedding");
    for x in v {
        *x = (*x as f64 / norm) as f32;
    }
    Ok(())
}
