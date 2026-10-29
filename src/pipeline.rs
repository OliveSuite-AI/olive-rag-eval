use crate::{
    clients::{
        ANSWER_PROMPT, ChatModel, Embedder, JUDGE_PROMPT, LLM_RERANK_PROMPT, LlmReranker,
        LocalClient, Reranker, Timed,
    },
    config::{Config, Model, Pipeline, RetrievalMode},
    data::{Case, Chunk, RankedChunk},
    fingerprint,
    metrics::{self, Judgment, Scores},
    sparse::{self, SparseIndex, SparseRetriever},
    store::{self, Snapshot, StageStore},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, future::Future, path::PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub key: String,
    pub cached: bool,
    pub seconds: f64,
    pub warmup_seconds: f64,
    pub raw: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub case: Case,
    pub embedding: Option<String>,
    #[serde(default)]
    pub pipeline: String,
    #[serde(default)]
    pub retrieval_mode: RetrievalMode,
    #[serde(default)]
    pub stage_metrics: BTreeMap<String, Scores>,
    #[serde(default)]
    pub diagnostics: Scores,
    #[serde(default)]
    pub retrieval_lists: BTreeMap<String, Vec<RankedChunk>>,
    pub ranker: String,
    pub repeat: usize,
    pub candidates: Vec<RankedChunk>,
    pub ranked: Vec<RankedChunk>,
    pub retrieval: Option<Scores>,
    pub candidate_recall: Option<f64>,
    pub generation: Option<Scores>,
    pub answer: Option<String>,
    pub judgment: Option<Judgment>,
    pub stages: BTreeMap<String, Evidence>,
    pub status: String,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Run {
    pub version: u32,
    pub fingerprint: String,
    pub mode: String,
    pub fixture: bool,
    pub config: Config,
    pub dataset_hash: String,
    pub corpus_hash: String,
    pub answer_prompt: String,
    pub judge_prompt: String,
    #[serde(default)]
    pub llm_rerank_prompt: String,
    pub same_answer_judge_model: bool,
    pub outcomes: Vec<Outcome>,
    #[serde(default)]
    pub calibration: Vec<CalibrationOutcome>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationOutcome {
    pub example: crate::data::CalibrationExample,
    pub judgment: Option<Judgment>,
    pub scores: Option<Scores>,
    pub reviewed_label_mean_absolute_error: Option<f64>,
    pub stages: BTreeMap<String, Evidence>,
    pub error: Option<String>,
}

async fn stage<T: Serialize + serde::de::DeserializeOwned, F: Future<Output = Result<Timed<T>>>>(
    db: &mut StageStore,
    key: &str,
    kind: &str,
    future: F,
    evidence: &mut BTreeMap<String, Evidence>,
) -> Result<T> {
    let (r, cached) = if let Some(r) = db.get::<Timed<T>>(key)? {
        (r, true)
    } else {
        match future.await {
            Ok(r) => {
                db.put(key, kind, "completed", &r)?;
                (r, false)
            }
            Err(e) => {
                db.put(key, kind, "failed", &json!({"error":format!("{e:#}"),"raw":e.downcast_ref::<crate::clients::ObservedFailure>().map(|e|&e.evidence)}))?;
                return Err(e.context(kind.to_owned()));
            }
        }
    };
    evidence.insert(
        kind.into(),
        Evidence {
            key: key.into(),
            cached,
            seconds: r.seconds,
            warmup_seconds: r.warmup_seconds,
            raw: r.raw,
        },
    );
    Ok(r.value)
}

pub async fn index(c: &Config, chunks: &[Chunk], client: &LocalClient) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for m in c.active_embeddings() {
        let dir = store::snapshot_dir(c, m, chunks);
        if dir.exists() {
            Snapshot::load(&dir)?;
            paths.push(dir);
            continue;
        }
        let mut vectors = Vec::new();
        let mut evidence = Vec::new();
        let profile = fingerprint(m);
        for batch in chunks.chunks(c.embedding_batch_size) {
            // A batch may mix supplied and generated vectors, but supplied vectors must match the full profile fingerprint.
            let missing: Vec<_> = batch
                .iter()
                .enumerate()
                .filter(|(_, x)| x.embedding.is_none())
                .map(|(i, _)| i)
                .collect();
            let texts: Vec<_> = missing.iter().map(|&i| batch[i].text.clone()).collect();
            let generated = if texts.is_empty() {
                Vec::new()
            } else {
                let r = client.embed(m, &texts, false).await?;
                evidence.push(
                    json!({"seconds":r.seconds,"warmup_seconds":r.warmup_seconds,"raw":r.raw}),
                );
                r.value
            };
            let mut generated = generated.into_iter();
            for chunk in batch {
                if let Some(v) = &chunk.embedding {
                    ensure!(
                        chunk.embedding_profile.as_deref() == Some(&profile),
                        "imported vector {} must identify embedding profile fingerprint {}",
                        chunk.id,
                        profile
                    );
                    let mut v = v.clone();
                    crate::data::normalize(&mut v, m.dimensions)?;
                    vectors.push(v);
                } else {
                    vectors.push(generated.next().context("embedding count mismatch")?);
                }
            }
        }
        paths.push(store::build_snapshot(
            c,
            m,
            chunks,
            &vectors,
            json!(evidence),
        )?);
    }
    if c.resolved_pipelines()
        .iter()
        .any(|p| p.retrieval.uses_sparse())
    {
        paths.push(sparse::build(c, chunks)?);
    }
    Ok(paths)
}

struct RetrievalBundle {
    key: String,
    dense: Vec<RankedChunk>,
    sparse: Vec<RankedChunk>,
    candidates: Vec<RankedChunk>,
}
struct RetrievalContext<'a> {
    c: &'a Config,
    client: &'a LocalClient,
    dense: BTreeMap<String, Result<Snapshot>>,
    sparse: Option<Result<SparseIndex>>,
}
impl<'a> RetrievalContext<'a> {
    fn load(c: &'a Config, chunks: &[Chunk], client: &'a LocalClient) -> Self {
        let dense = c
            .active_embeddings()
            .into_iter()
            .map(|m| {
                let result = Snapshot::load(&store::snapshot_dir(c, m, chunks)).and_then(|s| {
                    ensure!(
                        s.manifest.fingerprint == store::snapshot_key(chunks, m, c.index_bits),
                        "dense snapshot profile mismatch"
                    );
                    Ok(s)
                });
                (m.name.clone(), result)
            })
            .collect();
        let sparse = if c
            .resolved_pipelines()
            .iter()
            .any(|p| p.retrieval.uses_sparse())
        {
            Some(
                SparseIndex::load(&sparse::snapshot_dir(c, chunks)).and_then(|s| {
                    ensure!(
                        s.fingerprint == sparse::corpus_key(chunks),
                        "sparse snapshot corpus mismatch"
                    );
                    Ok(s)
                }),
            )
        } else {
            None
        };
        Self {
            c,
            client,
            dense,
            sparse,
        }
    }
    async fn get(
        &self,
        p: &Pipeline,
        query: &str,
        languages: &[String],
        repeat: usize,
        db: &mut StageStore,
        evidence: &mut BTreeMap<String, Evidence>,
    ) -> Result<RetrievalBundle> {
        let mut dense = Vec::new();
        let mut sparse = Vec::new();
        let mut dense_key = String::new();
        let mut sparse_key = String::new();
        if p.retrieval.uses_dense() {
            let name = p
                .embedding
                .as_ref()
                .context("dense pipeline requires embedding")?;
            let m = self
                .c
                .embeddings
                .iter()
                .find(|m| &m.name == name)
                .context("unknown embedding")?;
            let snapshot = self
                .dense
                .get(name)
                .context("dense index not configured")?
                .as_ref()
                .map_err(|e| anyhow::anyhow!("dense index unavailable; run index: {e:#}"))?;
            let query_key = fingerprint(&("query-v1", m, query, repeat));
            dense_key = fingerprint(&(
                "search-v1",
                &query_key,
                &snapshot.manifest.fingerprint,
                languages,
                self.c.candidate_k,
            ));
            let vectors = stage(
                db,
                &query_key,
                "embedding",
                self.client.embed(m, &[query.to_owned()], true),
                evidence,
            )
            .await?;
            dense = stage(
                db,
                &dense_key,
                "search",
                async { snapshot.search(&vectors[0], languages, self.c.candidate_k) },
                evidence,
            )
            .await?;
        }
        if p.retrieval.uses_sparse() {
            let index = self
                .sparse
                .as_ref()
                .context("sparse index not configured")?
                .as_ref()
                .map_err(|e| anyhow::anyhow!("sparse index unavailable; run index: {e:#}"))?;
            sparse_key = fingerprint(&(
                "bm25-v1",
                &index.fingerprint,
                query,
                languages,
                &self.c.bm25,
                repeat,
            ));
            sparse = stage(
                db,
                &sparse_key,
                "bm25",
                index.retrieve(query, languages, &self.c.bm25),
                evidence,
            )
            .await?;
        }
        let (key, candidates) = match p.retrieval {
            RetrievalMode::Dense => (dense_key, dense.clone()),
            RetrievalMode::Bm25 => (sparse_key, sparse.clone()),
            RetrievalMode::HybridRrf => {
                let key = fingerprint(&("rrf-v1", dense_key, sparse_key, &self.c.rrf));
                let hits = stage(
                    db,
                    &key,
                    "rrf",
                    async { crate::fusion::fuse(&dense, &sparse, &self.c.rrf) },
                    evidence,
                )
                .await?;
                (key, hits)
            }
        };
        Ok(RetrievalBundle {
            key,
            dense,
            sparse,
            candidates,
        })
    }
}
pub async fn retrieve_named(
    c: &Config,
    chunks: &[Chunk],
    client: &LocalClient,
    p: &Pipeline,
    query: &str,
    languages: &[String],
) -> Result<Vec<RankedChunk>> {
    let context = RetrievalContext::load(c, chunks, client);
    let mut db = StageStore::open(&c.artifacts.join("stages.sqlite"))?;
    Ok(context
        .get(p, query, languages, 0, &mut db, &mut BTreeMap::new())
        .await?
        .candidates)
}

pub async fn retrieve(
    c: &Config,
    chunks: &[Chunk],
    client: &LocalClient,
    m: &Model,
    query: &str,
    languages: &[String],
) -> Result<Vec<RankedChunk>> {
    let snapshot = Snapshot::load(&store::snapshot_dir(c, m, chunks))
        .context("index missing or invalid; run index first")?;
    ensure!(
        snapshot.manifest.fingerprint == store::snapshot_key(chunks, m, c.index_bits),
        "index profile mismatch"
    );
    let r = client.embed(m, &[query.to_owned()], true).await?;
    Ok(snapshot
        .search(&r.value[0], languages, c.candidate_k)?
        .value)
}

pub async fn evaluate(
    c: &Config,
    chunks: &[Chunk],
    cases: &[Case],
    full: bool,
    client: &LocalClient,
) -> Result<(Run, PathBuf)> {
    evaluate_with_progress(c, chunks, cases, full, client, true).await
}
pub async fn evaluate_with_progress(
    c: &Config,
    chunks: &[Chunk],
    cases: &[Case],
    full: bool,
    client: &LocalClient,
    progress: bool,
) -> Result<(Run, PathBuf)> {
    if full {
        ensure!(
            c.answer.is_some() && c.judge.is_some(),
            "evaluate requires answer and judge profiles"
        );
    }
    let calibration: Vec<crate::data::CalibrationExample> = if full {
        c.calibration
            .as_ref()
            .map(|p| crate::data::read_jsonl(p))
            .transpose()?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut run = Run {
        version: 2,
        fingerprint: fingerprint(&(
            c,
            chunks,
            cases,
            &calibration,
            full,
            ANSWER_PROMPT,
            JUDGE_PROMPT,
        )),
        mode: if full { "full" } else { "retrieval" }.into(),
        fixture: c.fixture,
        config: c.clone(),
        dataset_hash: fingerprint(&cases),
        corpus_hash: fingerprint(&chunks),
        answer_prompt: ANSWER_PROMPT.into(),
        judge_prompt: JUDGE_PROMPT.into(),
        llm_rerank_prompt: LLM_RERANK_PROMPT.into(),
        same_answer_judge_model: match (&c.answer, &c.judge) {
            (Some(a), Some(j)) => a.model == j.model && a.artifact_sha256 == j.artifact_sha256,
            _ => false,
        },
        outcomes: Vec::new(),
        calibration: Vec::new(),
    };
    let run_dir = c.artifacts.join("runs").join(&run.fingerprint);
    std::fs::create_dir_all(&run_dir)?;
    let output = run_dir.join("run.json");
    let mut db = StageStore::open(&c.artifacts.join("stages.sqlite"))?;
    let context = RetrievalContext::load(c, chunks, client);
    let llm = LlmReranker { client };
    for p in c.resolved_pipelines() {
        for case in cases {
            for repeat in 0..c.repeats {
                let mut out = Outcome {
                    case: case.clone(),
                    embedding: p.embedding.clone(),
                    pipeline: if c.pipelines.is_empty() {
                        String::new()
                    } else {
                        p.name.clone()
                    },
                    retrieval_mode: p.retrieval,
                    stage_metrics: BTreeMap::new(),
                    diagnostics: Scores::new(),
                    retrieval_lists: BTreeMap::new(),
                    ranker: p
                        .reranker
                        .as_deref()
                        .unwrap_or(if c.pipelines.is_empty() {
                            "vector"
                        } else {
                            "none"
                        })
                        .into(),
                    repeat,
                    candidates: Vec::new(),
                    ranked: Vec::new(),
                    retrieval: None,
                    candidate_recall: None,
                    generation: None,
                    answer: None,
                    judgment: None,
                    stages: BTreeMap::new(),
                    status: "completed".into(),
                    error: None,
                };
                let result: Result<()> = async {
                    let bundle = context
                        .get(
                            &p,
                            &case.query,
                            &case.corpus_languages,
                            repeat,
                            &mut db,
                            &mut out.stages,
                        )
                        .await?;
                    out.candidates = bundle.candidates;
                    if p.retrieval.uses_dense() {
                        out.retrieval_lists
                            .insert("dense".into(), bundle.dense.clone());
                    }
                    if p.retrieval.uses_sparse() {
                        out.retrieval_lists
                            .insert("bm25".into(), bundle.sparse.clone());
                        out.diagnostics.insert(
                            "sparse_zero_hit".into(),
                            if bundle.sparse.is_empty() { 1. } else { 0. },
                        );
                    }
                    if p.retrieval == RetrievalMode::HybridRrf {
                        out.retrieval_lists
                            .insert("rrf".into(), out.candidates.clone());
                        out.diagnostics.extend(crate::fusion::diagnostics(
                            &bundle.dense,
                            &bundle.sparse,
                            &out.candidates,
                            &case.relevant_pages,
                        ));
                    }
                    for (name, list) in &out.retrieval_lists {
                        if let Some(mut scores) =
                            metrics::retrieval(list, &case.relevant_pages, c.cutoff)
                        {
                            let k = match name.as_str() {
                                "dense" => c.candidate_k,
                                "bm25" => c.bm25.top_k,
                                _ => c.rrf.top_k,
                            };
                            let recall = metrics::retrieval(list, &case.relevant_pages, k).unwrap()
                                [&format!("recall@{k}")];
                            scores.insert("candidate_recall".into(), recall);
                            out.stage_metrics.insert(name.clone(), scores);
                        }
                    }
                    let rank_key = fingerprint(&(
                        "rank-v2",
                        &bundle.key,
                        &p.reranker,
                        &out.candidates,
                        c.llm_reranker
                            .as_ref()
                            .filter(|m| Some(&m.name) == p.reranker.as_ref()),
                        LLM_RERANK_PROMPT,
                    ));
                    out.ranked = if let Some(name) = &p.reranker {
                        if let Some(m) = c.llm_reranker.as_ref().filter(|m| &m.name == name) {
                            stage(
                                &mut db,
                                &rank_key,
                                "llm_rerank",
                                llm.rerank(m, &case.query, &out.candidates),
                                &mut out.stages,
                            )
                            .await?
                        } else {
                            let m = c
                                .rerankers
                                .iter()
                                .find(|m| &m.name == name)
                                .context("unknown reranker")?;
                            let key = fingerprint(&(&rank_key, m));
                            stage(
                                &mut db,
                                &key,
                                "rerank",
                                client.rerank(m, &case.query, &out.candidates),
                                &mut out.stages,
                            )
                            .await?
                        }
                    } else {
                        out.candidates.clone()
                    };
                    out.retrieval = metrics::retrieval(&out.ranked, &case.relevant_pages, c.cutoff);
                    if let Some(scores) = &out.retrieval {
                        out.stage_metrics.insert("final".into(), scores.clone());
                    }
                    if let (Some(scores), Some(baseline)) = (
                        &mut out.retrieval,
                        metrics::retrieval(&out.candidates, &case.relevant_pages, c.cutoff),
                    ) {
                        for metric in [
                            format!("ndcg@{}", c.cutoff),
                            format!("recall@{}", c.cutoff),
                            "mrr".into(),
                        ] {
                            scores.insert(
                                format!("delta_{metric}"),
                                scores[&metric] - baseline[&metric],
                            );
                        }
                    }
                    let k = match p.retrieval {
                        RetrievalMode::Dense => c.candidate_k,
                        RetrievalMode::Bm25 => c.bm25.top_k,
                        RetrievalMode::HybridRrf => c.rrf.top_k,
                    };
                    out.candidate_recall =
                        metrics::retrieval(&out.candidates, &case.relevant_pages, k)
                            .map(|s| s[&format!("recall@{k}")]);
                    if !full {
                        return Ok(());
                    }
                    let top: Vec<_> = out.ranked.iter().take(c.cutoff).cloned().collect();
                    let a = c.answer.as_ref().unwrap();
                    let j = c.judge.as_ref().unwrap();
                    let answer_key = fingerprint(&(
                        ANSWER_PROMPT,
                        a,
                        &case.query,
                        &case.answer_language,
                        crate::clients::context(&top),
                        repeat,
                    ));
                    let answer = stage(
                        &mut db,
                        &answer_key,
                        "answer",
                        client.answer(a, &case.query, &case.answer_language, &top),
                        &mut out.stages,
                    )
                    .await?;
                    out.answer = Some(answer.clone());
                    let judge_key = fingerprint(&(
                        JUDGE_PROMPT,
                        j,
                        &case.query,
                        &case.expected_answer,
                        &case.key_facts,
                        &case.answer_language,
                        case.unanswerable,
                        &answer,
                        crate::clients::context(&top),
                        repeat,
                    ));
                    let judgment = stage(
                        &mut db,
                        &judge_key,
                        "judge",
                        client.judge(j, case, &top, &answer),
                        &mut out.stages,
                    )
                    .await?;
                    out.generation =
                        Some(judgment.scores(&top, case.key_facts.len(), case.unanswerable)?);
                    out.judgment = Some(judgment);
                    Ok(())
                }
                .await;
                if let Err(e) = result {
                    out.status = "failed".into();
                    out.error = Some(format!("{e:#}"));
                }
                if progress {
                    eprintln!(
                        "{} {} repeat {}: {}",
                        case.id,
                        p.name,
                        repeat + 1,
                        out.status
                    );
                }
                run.outcomes.push(out);
                if c.pipelines.is_empty() {
                    run.outcomes.sort_by_key(|o| {
                        (
                            c.embeddings
                                .iter()
                                .position(|m| Some(&m.name) == o.embedding.as_ref())
                                .unwrap_or(usize::MAX),
                            cases
                                .iter()
                                .position(|c| c.id == o.case.id)
                                .unwrap_or(usize::MAX),
                            o.repeat,
                            if o.ranker == "vector" {
                                0
                            } else {
                                1 + c
                                    .rerankers
                                    .iter()
                                    .position(|m| m.name == o.ranker)
                                    .unwrap_or(usize::MAX - 1)
                            },
                        )
                    });
                }
                save_run(&output, &run)?;
            }
        }
    }
    for example in calibration {
        let top: Vec<_> = example
            .context
            .iter()
            .enumerate()
            .map(|(i, chunk)| RankedChunk {
                chunk: chunk.clone(),
                vector_rank: Some(i + 1),
                vector_score: None,
                rank: i + 1,
                rerank_score: None,
                sparse_rank: None,
                sparse_score: None,
                fused_rank: None,
                rrf_score: None,
                raw: Value::Null,
            })
            .collect();
        let mut out = CalibrationOutcome {
            example: example.clone(),
            judgment: None,
            scores: None,
            reviewed_label_mean_absolute_error: None,
            stages: BTreeMap::new(),
            error: None,
        };
        let result: Result<()> = async {
            ensure!(
                matches!(example.label_review_status.as_str(), "draft" | "reviewed"),
                "invalid calibration review status"
            );
            let m = c.judge.as_ref().unwrap();
            let key = fingerprint(&(
                JUDGE_PROMPT,
                m,
                &example.case.query,
                &example.case.expected_answer,
                &example.case.key_facts,
                &example.case.answer_language,
                example.case.unanswerable,
                &example.answer,
                crate::clients::context(&top),
                "calibration",
            ));
            let judgment = stage(
                &mut db,
                &key,
                "judge_calibration",
                client.judge(m, &example.case, &top, &example.answer),
                &mut out.stages,
            )
            .await?;
            let scores = judgment.scores(
                &top,
                example.case.key_facts.len(),
                example.case.unanswerable,
            )?;
            if example.label_review_status == "reviewed" {
                let expected = example
                    .expected_judgment
                    .as_ref()
                    .context("reviewed calibration example requires human judgment")?;
                let target = expected.scores(
                    &top,
                    example.case.key_facts.len(),
                    example.case.unanswerable,
                )?;
                out.reviewed_label_mean_absolute_error = Some(
                    target
                        .iter()
                        .map(|(k, v)| (scores.get(k).copied().unwrap_or(0.) - v).abs())
                        .sum::<f64>()
                        / target.len() as f64,
                );
            }
            out.judgment = Some(judgment);
            out.scores = Some(scores);
            Ok(())
        }
        .await;
        if let Err(e) = result {
            out.error = Some(format!("{e:#}"));
        }
        run.calibration.push(out);
        save_run(&output, &run)?;
    }
    Ok((run, output))
}
fn save_run(path: &std::path::Path, run: &Run) -> Result<()> {
    crate::persistence::atomic_write(path, &serde_json::to_vec_pretty(run)?)?;
    Ok(())
}
