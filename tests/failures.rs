use rag_eval::{
    clients::{ChatModel, LocalClient},
    config::Config,
    data, pipeline,
};
use std::path::Path;

#[tokio::test]
async fn failed_judge_keeps_answer_retrieval_and_both_raw_responses() {
    let temp = tempfile::tempdir().unwrap();
    let mut c =
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/fixture.toml")).unwrap();
    c.artifacts = temp.path().into();
    c.embeddings.truncate(1);
    c.rerankers.clear();
    c.calibration = None;
    c.judge.as_mut().unwrap().model = "fixture-bad-judge".into();
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], true, &client)
        .await
        .unwrap();
    let o = &run.outcomes[0];
    assert_eq!(o.status, "failed");
    assert!(o.answer.is_some());
    assert!(o.retrieval.is_some());
    assert!(o.generation.is_none());
    let db = rusqlite::Connection::open(temp.path().join("stages.sqlite")).unwrap();
    let payload: String = db
        .query_row(
            "SELECT payload FROM stages WHERE kind='judge' AND status='failed'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert!(value["raw"]["initial"].is_object());
    assert!(value["raw"]["repair"].is_object());
}

#[tokio::test]
async fn fixed_context_calibration_requires_reviewed_labels_for_agreement() {
    let temp = tempfile::tempdir().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut c = Config::load(&root.join("configs/fixture.toml")).unwrap();
    c.artifacts = temp.path().into();
    c.embeddings.truncate(1);
    c.rerankers.clear();
    let client = LocalClient::new(&c).unwrap();
    let mut examples: Vec<data::CalibrationExample> =
        data::read_jsonl(c.calibration.as_ref().unwrap()).unwrap();
    // Synthetic label used only for protocol testing, not included in shipped human-reviewed data.
    let e = &mut examples[0];
    let ranked = e
        .context
        .iter()
        .enumerate()
        .map(|(i, c)| data::RankedChunk {
            chunk: c.clone(),
            vector_rank: Some(i + 1),
            vector_score: None,
            rank: i + 1,
            rerank_score: None,
            sparse_rank: None,
            sparse_score: None,
            fused_rank: None,
            rrf_score: None,
            raw: serde_json::Value::Null,
        })
        .collect::<Vec<_>>();
    let result = client
        .judge(c.judge.as_ref().unwrap(), &e.case, &ranked, &e.answer)
        .await
        .unwrap();
    e.label_review_status = "reviewed".into();
    e.expected_judgment = Some(result.value);
    let path = temp.path().join("calibration.jsonl");
    std::fs::write(
        &path,
        examples
            .iter()
            .map(|e| serde_json::to_string(e).unwrap() + "\n")
            .collect::<String>(),
    )
    .unwrap();
    c.calibration = Some(path);
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], true, &client)
        .await
        .unwrap();
    assert_eq!(
        run.calibration[0].reviewed_label_mean_absolute_error,
        Some(0.)
    );
    assert!(
        run.calibration[1]
            .reviewed_label_mean_absolute_error
            .is_none()
    );
    assert_eq!(
        rag_eval::report::summarize(&run).calibration["reviewed_scored_examples"],
        1
    );
}
