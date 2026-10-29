use rag_eval::{
    clients::LocalClient,
    config::{Config, RetrievalMode},
    data, pipeline, report,
};
use std::path::Path;

#[test]
fn pipeline_configuration_references_and_unused_limits() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/hybrid-fixture.toml");
    let mut c = Config::load(&path).unwrap();
    c.pipelines[0].embedding = Some("missing".into());
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.pipelines[1].embedding = Some("embeddinggemma-q8".into());
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.pipelines[3].reranker = Some("missing".into());
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.pipelines[1].name = c.pipelines[0].name.clone();
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.rrf.rank_constant = f64::NAN;
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.bm25.b = 1.1;
    assert!(c.validate().is_err());
    let mut c = Config::load(&path).unwrap();
    c.pipelines.retain(|p| p.retrieval == RetrievalMode::Dense);
    c.cutoff = 25;
    c.candidate_k = 25;
    c.validate().unwrap();
}

#[tokio::test]
async fn version_one_run_reports_and_dense_snapshot_remain_readable() {
    let temp = tempfile::tempdir().unwrap();
    let mut c =
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/fixture.toml")).unwrap();
    c.artifacts = temp.path().into();
    c.embeddings.truncate(1);
    c.rerankers.clear();
    c.calibration = None;
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    let paths = pipeline::index(&c, &chunks, &client).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(paths[0].join("manifest.json")).unwrap()
        )
        .unwrap()["version"],
        1
    );
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], false, &client)
        .await
        .unwrap();
    let mut old = serde_json::to_value(&run).unwrap();
    old["version"] = 1.into();
    old.as_object_mut().unwrap().remove("llm_rerank_prompt");
    for field in ["bm25", "rrf", "pipelines", "llm_reranker"] {
        old["config"].as_object_mut().unwrap().remove(field);
    }
    for o in old["outcomes"].as_array_mut().unwrap() {
        for field in [
            "pipeline",
            "retrieval_mode",
            "stage_metrics",
            "diagnostics",
            "retrieval_lists",
        ] {
            o.as_object_mut().unwrap().remove(field);
        }
        for list in ["candidates", "ranked"] {
            for h in o[list].as_array_mut().unwrap() {
                for field in ["sparse_rank", "sparse_score", "fused_rank", "rrf_score"] {
                    h.as_object_mut().unwrap().remove(field);
                }
            }
        }
    }
    let decoded: pipeline::Run = serde_json::from_value(old).unwrap();
    assert_eq!(decoded.outcomes[0].retrieval_mode, RetrievalMode::Dense);
    assert!(
        decoded.outcomes[0]
            .ranked
            .iter()
            .all(|h| h.vector_rank.is_some() && h.sparse_rank.is_none())
    );
    report::write(&decoded, &temp.path().join("v1-reports")).unwrap();
    assert!(temp.path().join("v1-reports/report.md").exists());
    assert_eq!(pipeline::index(&c, &chunks, &client).await.unwrap(), paths);
}
