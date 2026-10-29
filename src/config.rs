use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub name: String,
    pub model: String,
    pub revision: String,
    pub artifact_sha256: String,
    pub url: String,
    #[serde(default)]
    pub dimensions: usize,
    #[serde(default)]
    pub query_prefix: String,
    #[serde(default)]
    pub document_prefix: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub context_tokens: usize,
    #[serde(default = "default_output")]
    pub max_output_tokens: usize,
    #[serde(default)]
    pub api_key_env: Option<String>,
}
fn default_output() -> usize {
    1024
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    #[default]
    Dense,
    Bm25,
    HybridRrf,
}
impl RetrievalMode {
    pub fn uses_dense(self) -> bool {
        self != Self::Bm25
    }
    pub fn uses_sparse(self) -> bool {
        self != Self::Dense
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pipeline {
    pub name: String,
    pub retrieval: RetrievalMode,
    pub embedding: Option<String>,
    pub reranker: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bm25Config {
    pub k1: f64,
    pub b: f64,
    pub top_k: usize,
}
impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            top_k: 20,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RrfConfig {
    pub rank_constant: f64,
    pub top_k: usize,
}
impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            rank_constant: 60.,
            top_k: 20,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub corpus: PathBuf,
    pub dataset: PathBuf,
    #[serde(default)]
    pub calibration: Option<PathBuf>,
    pub artifacts: PathBuf,
    #[serde(default = "twenty")]
    pub candidate_k: usize,
    #[serde(default = "five")]
    pub cutoff: usize,
    #[serde(default = "four")]
    pub index_bits: usize,
    #[serde(default = "one")]
    pub repeats: usize,
    #[serde(default = "timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "batch")]
    pub embedding_batch_size: usize,
    #[serde(default)]
    pub allow_truncation: bool,
    #[serde(default)]
    pub fixture: bool,
    #[serde(default)]
    pub fixture_stdio: Option<PathBuf>,
    #[serde(default)]
    pub python_executable: Option<PathBuf>,
    pub embeddings: Vec<Model>,
    #[serde(default)]
    pub rerankers: Vec<Model>,
    pub answer: Option<Model>,
    pub judge: Option<Model>,
    #[serde(default)]
    pub llm_reranker: Option<Model>,
    #[serde(default)]
    pub pipelines: Vec<Pipeline>,
    #[serde(default)]
    pub bm25: Bm25Config,
    #[serde(default)]
    pub rrf: RrfConfig,
}
fn twenty() -> usize {
    20
}
fn five() -> usize {
    5
}
fn four() -> usize {
    4
}
fn one() -> usize {
    1
}
fn timeout() -> u64 {
    120
}
fn batch() -> usize {
    32
}

impl Config {
    pub fn resolved_pipelines(&self) -> Vec<Pipeline> {
        if !self.pipelines.is_empty() {
            return self.pipelines.clone();
        }
        self.embeddings
            .iter()
            .flat_map(|m| {
                std::iter::once(None)
                    .chain(self.rerankers.iter().map(Some))
                    .map(move |r| Pipeline {
                        name: format!("{}-{}", m.name, r.map_or("vector", |r| r.name.as_str())),
                        retrieval: RetrievalMode::Dense,
                        embedding: Some(m.name.clone()),
                        reranker: r.map(|r| r.name.clone()),
                    })
            })
            .collect()
    }
    pub fn active_embeddings(&self) -> Vec<&Model> {
        let names: HashSet<_> = self
            .resolved_pipelines()
            .into_iter()
            .filter_map(|p| p.embedding)
            .collect();
        self.embeddings
            .iter()
            .filter(|m| names.contains(&m.name))
            .collect()
    }
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = toml::from_str(&std::fs::read_to_string(path)?)
            .context("invalid TOML configuration")?;
        let base = path.parent().unwrap_or(Path::new("."));
        c.resolve_paths(base)
    }
    pub fn resolve_paths(mut self, base: &Path) -> Result<Self> {
        let c = &mut self;
        for p in [&mut c.corpus, &mut c.dataset, &mut c.artifacts] {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        if let Some(p) = &mut c.fixture_stdio
            && p.is_relative()
        {
            *p = base.join(&*p);
        }
        if let Some(p) = &mut c.calibration
            && p.is_relative()
        {
            *p = base.join(&*p);
        }
        if let Some(p) = &mut c.python_executable
            && p.is_relative()
            && p.components().count() > 1
        {
            *p = base.join(&*p);
        }
        c.validate()?;
        Ok(self)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.fixture || self.fixture_stdio.is_none(),
            "stdio fixture transport requires fixture=true"
        );
        ensure!(
            self.cutoff > 0
                && self.candidate_k > 0
                && (!self
                    .resolved_pipelines()
                    .iter()
                    .any(|p| p.retrieval.uses_dense())
                    || self.candidate_k >= self.cutoff),
            "require positive cutoff/candidate_k; dense candidate_k must cover cutoff"
        );
        ensure!(
            matches!(self.index_bits, 2 | 4),
            "index_bits must be 2 or 4"
        );
        ensure!(
            self.repeats > 0 && self.embedding_batch_size > 0 && self.timeout_seconds > 0,
            "repeats, batch size and timeout must be positive"
        );
        ensure!(
            !self.resolved_pipelines().is_empty(),
            "at least one pipeline or embedding profile is required"
        );
        let mut names = HashSet::new();
        for m in self
            .embeddings
            .iter()
            .chain(&self.rerankers)
            .chain(self.answer.iter())
            .chain(self.judge.iter())
            .chain(self.llm_reranker.iter())
        {
            ensure!(
                !m.name.is_empty() && m.name != "vector" && names.insert(&m.name),
                "model names must be unique, nonempty and not vector"
            );
            ensure!(
                m.name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "model names must be path-safe"
            );
            ensure!(
                !m.model.is_empty() && !m.revision.is_empty(),
                "model and revision required"
            );
            if !self.fixture {
                ensure!(
                    m.artifact_sha256.len() == 64
                        && m.artifact_sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "{} requires a real SHA256 artifact digest",
                    m.name
                );
                ensure!(
                    m.revision != "main" && m.revision != "latest",
                    "pin model revisions"
                );
            }
            let url = reqwest::Url::parse(&m.url)?;
            ensure!(
                matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1")),
                "local model endpoints only: {}",
                m.url
            );
            ensure!(
                matches!(url.scheme(), "http" | "https"),
                "HTTP endpoint required"
            );
        }
        for m in &self.embeddings {
            ensure!(
                m.dimensions > 0 && m.dimensions <= 16384 && m.dimensions.is_multiple_of(8),
                "unsupported TurboVec dimensions for {}",
                m.name
            );
        }
        for m in self
            .answer
            .iter()
            .chain(self.judge.iter())
            .chain(self.llm_reranker.iter())
        {
            ensure!(
                m.context_tokens > m.max_output_tokens && m.max_output_tokens > 0,
                "chat context must exceed positive output budget"
            );
        }
        ensure!(
            self.bm25.k1.is_finite()
                && self.bm25.k1 > 0.
                && self.bm25.b.is_finite()
                && (0. ..=1.).contains(&self.bm25.b)
                && self.bm25.top_k > 0
                && (!self
                    .resolved_pipelines()
                    .iter()
                    .any(|p| p.retrieval.uses_sparse())
                    || self.bm25.top_k >= self.cutoff),
            "invalid BM25 parameters or cutoff"
        );
        ensure!(
            self.rrf.rank_constant.is_finite()
                && self.rrf.rank_constant > 0.
                && self.rrf.top_k > 0
                && (!self
                    .resolved_pipelines()
                    .iter()
                    .any(|p| p.retrieval == RetrievalMode::HybridRrf)
                    || self.rrf.top_k >= self.cutoff),
            "invalid RRF parameters or cutoff"
        );
        let mut pipelines = HashSet::new();
        for p in self.resolved_pipelines() {
            ensure!(
                !p.name.is_empty()
                    && p.name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                    && pipelines.insert(p.name.clone()),
                "duplicate or invalid pipeline name"
            );
            if p.retrieval.uses_dense() {
                ensure!(
                    self.embeddings
                        .iter()
                        .any(|m| Some(&m.name) == p.embedding.as_ref()),
                    "pipeline {} must reference an embedding profile",
                    p.name
                );
            } else {
                ensure!(
                    p.embedding.is_none(),
                    "BM25-only pipeline must not declare an embedding"
                );
            }
            if let Some(r) = &p.reranker {
                ensure!(
                    self.rerankers
                        .iter()
                        .chain(self.llm_reranker.iter())
                        .any(|m| &m.name == r),
                    "unknown pipeline reranker {r}"
                );
            }
        }
        Ok(())
    }
}
