use rag_eval::{
    clients::{LlmReranker, LocalClient, Reranker},
    config::{Config, RetrievalMode},
    data, pipeline, report, sparse,
};
use std::path::Path;

fn config(artifacts: &Path) -> Config {
    let mut c =
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/hybrid-fixture.toml"))
            .unwrap();
    c.artifacts = artifacts.into();
    c.calibration = None;
    c
}
fn outcome<'a>(run: &'a pipeline::Run, name: &str, id: &str) -> &'a pipeline::Outcome {
    run.outcomes
        .iter()
        .find(|o| o.pipeline == name && o.case.id == id)
        .unwrap()
}

#[tokio::test]
async fn hybrid_matrix_resume_fusion_and_llm_invalidation() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config(temp.path());
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let cases = vec![
        cases[0].clone(),
        cases[1].clone(),
        cases[2].clone(),
        cases[3].clone(),
        cases[54].clone(),
    ];
    let client = LocalClient::new(&c).unwrap();
    let paths = pipeline::index(&c, &chunks, &client).await.unwrap();
    assert_eq!(paths.len(), 2);
    let (run, path) = pipeline::evaluate(&c, &chunks, &cases, true, &client)
        .await
        .unwrap();
    assert_eq!(run.version, 2);
    assert_eq!(run.outcomes.len(), 20);
    assert!(
        run.outcomes.iter().all(|o| o.status == "completed"),
        "{:?}",
        run.outcomes
            .iter()
            .filter_map(|o| o.error.as_ref())
            .collect::<Vec<_>>()
    );
    for case in &cases {
        let dense = outcome(&run, "dense", &case.id);
        let bm25 = outcome(&run, "bm25", &case.id);
        let fusion = outcome(&run, "hybrid-rrf", &case.id);
        let llm = outcome(&run, "hybrid-llm", &case.id);
        assert!(bm25.embedding.is_none());
        assert!(
            bm25.ranked
                .iter()
                .all(|h| h.vector_score.is_none() && h.sparse_rank.is_some())
        );
        assert_eq!(dense.stages["search"].key, fusion.stages["search"].key);
        assert_eq!(bm25.stages["bm25"].key, fusion.stages["bm25"].key);
        assert_eq!(fusion.stages["rrf"].key, llm.stages["rrf"].key);
        assert_eq!(
            fusion
                .candidates
                .iter()
                .map(|h| &h.chunk.id)
                .collect::<Vec<_>>(),
            llm.candidates
                .iter()
                .map(|h| &h.chunk.id)
                .collect::<Vec<_>>()
        );
        assert!(llm.ranked.iter().all(|h| h.rerank_score.is_none()));
        assert!(llm.answer.is_some());
        assert_eq!(llm.retrieval_mode, RetrievalMode::HybridRrf);
        assert!(
            llm.retrieval_lists.contains_key("dense")
                && llm.retrieval_lists.contains_key("bm25")
                && llm.retrieval_lists.contains_key("rrf")
        );
    }
    report::write(&run, &path.parent().unwrap().join("reports")).unwrap();
    let summary = report::summarize(&run);
    assert!(
        summary
            .groups
            .contains_key("hybrid-llm/llm-reranker-local/draft/all")
    );
    assert!(summary.latency.contains_key("bm25/bm25"));
    assert!(summary.latency.contains_key("rrf/rrf"));
    assert!(
        summary
            .latency
            .contains_key("llm-reranker-local/llm_rerank")
    );
    let before = client.fixture_stats().await.unwrap();
    let (resumed, _) = pipeline::evaluate(&c, &chunks, &cases, true, &client)
        .await
        .unwrap();
    assert_eq!(before, client.fixture_stats().await.unwrap());
    assert!(
        resumed
            .outcomes
            .iter()
            .flat_map(|o| o.stages.values())
            .all(|e| e.cached)
    );
    c.rrf.rank_constant = 42.;
    let (changed, _) = pipeline::evaluate(&c, &chunks, &cases, false, &client)
        .await
        .unwrap();
    assert!(
        changed
            .outcomes
            .iter()
            .flat_map(|o| o.stages.iter())
            .filter(|(kind, _)| matches!(kind.as_str(), "search" | "bm25" | "embedding"))
            .all(|(_, e)| e.cached)
    );
    assert!(
        changed
            .outcomes
            .iter()
            .any(|o| o.stages.get("rrf").is_some_and(|s| !s.cached))
    );
    c.llm_reranker.as_mut().unwrap().revision = "fixture-new-model".into();
    let (reranked, _) = pipeline::evaluate(&c, &chunks, &cases, false, &client)
        .await
        .unwrap();
    assert!(
        reranked
            .outcomes
            .iter()
            .flat_map(|o| o.stages.iter())
            .filter(|(kind, _)| matches!(kind.as_str(), "search" | "bm25" | "rrf"))
            .all(|(_, e)| e.cached)
    );
    assert!(
        reranked
            .outcomes
            .iter()
            .filter(|o| o.pipeline == "hybrid-llm")
            .all(|o| !o.stages["llm_rerank"].cached)
    );
}

#[tokio::test]
async fn sparse_only_needs_no_embeddings_and_is_not_duplicated() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config(temp.path());
    c.pipelines.retain(|p| p.retrieval == RetrievalMode::Bm25);
    c.embeddings.clear();
    c.llm_reranker = None;
    c.validate().unwrap();
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    assert_eq!(
        pipeline::index(&c, &chunks, &client).await.unwrap().len(),
        1
    );
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..4], false, &client)
        .await
        .unwrap();
    assert_eq!(run.outcomes.len(), 4);
    assert!(
        run.outcomes
            .iter()
            .all(|o| o.status == "completed" && o.embedding.is_none())
    );
    assert_eq!(client.fixture_stats().await.unwrap(), serde_json::json!({}));
}

#[tokio::test]
async fn missing_sparse_branch_fails_hybrid_without_silent_dense_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let c = config(temp.path());
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    std::fs::rename(
        sparse::snapshot_dir(&c, &chunks),
        temp.path().join("removed-sparse"),
    )
    .unwrap();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], false, &client)
        .await
        .unwrap();
    assert_eq!(outcome(&run, "dense", &cases[0].id).status, "completed");
    for name in ["bm25", "hybrid-rrf", "hybrid-llm"] {
        assert_eq!(outcome(&run, name, &cases[0].id).status, "failed");
    }
}

#[tokio::test]
async fn invalid_llm_ranking_repair_capacity_and_raw_failure_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config(temp.path());
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let p = c.pipelines.iter().find(|p| p.name == "hybrid-rrf").unwrap();
    let hits = pipeline::retrieve_named(
        &c,
        &chunks,
        &client,
        p,
        &cases[0].query,
        &cases[0].corpus_languages,
    )
    .await
    .unwrap();
    let llm = LlmReranker { client: &client };
    let mut model = c.llm_reranker.clone().unwrap();
    model.model = "fixture-repair-ranking".into();
    let repaired = llm.rerank(&model, &cases[0].query, &hits).await.unwrap();
    assert!(repaired.raw.get("initial").is_some() && repaired.raw.get("repair").is_some());
    let before = client.fixture_stats().await.unwrap();
    assert!(
        llm.rerank(&model, "query", &hits[..1])
            .await
            .unwrap()
            .raw
            .get("skipped")
            .is_some()
    );
    assert!(
        llm.rerank(&model, "query", &[])
            .await
            .unwrap()
            .value
            .is_empty()
    );
    assert_eq!(before, client.fixture_stats().await.unwrap());
    let mut small = model.clone();
    small.context_tokens = 64;
    small.max_output_tokens = 32;
    assert!(llm.rerank(&small, &cases[0].query, &hits).await.is_err());
    c.llm_reranker.as_mut().unwrap().model = "fixture-bad-ranking".into();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], true, &client)
        .await
        .unwrap();
    let failed = outcome(&run, "hybrid-llm", &cases[0].id);
    assert_eq!(failed.status, "failed");
    assert!(failed.answer.is_none());
    assert!(
        run.outcomes
            .iter()
            .filter(|o| o.pipeline != "hybrid-llm")
            .all(|o| o.status == "completed")
    );
    let db = rusqlite::Connection::open(temp.path().join("stages.sqlite")).unwrap();
    let payload: String = db
        .query_row(
            "SELECT payload FROM stages WHERE kind='llm_rerank' AND status='failed'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let raw: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert!(raw["raw"]["initial"].is_object() && raw["raw"]["repair"].is_object());
}
