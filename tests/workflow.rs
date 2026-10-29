use rag_eval::{
    clients::{ChatModel, Embedder, LocalClient, Reranker},
    config::Config,
    data, pipeline, report,
};
use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, Command, Stdio},
};

struct Server {
    child: Child,
    url: String,
}
impl Server {
    fn start() -> Self {
        let mut child = Command::new("python3")
            .args([
                "-B",
                "services/runtime.py",
                "--kind",
                "fixture",
                "--port",
                "0",
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let ready: serde_json::Value =
            serde_json::from_str(&line).expect("fixture service must start");
        Self {
            child,
            url: format!("http://127.0.0.1:{}", ready["port"]),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn config(server: Option<&Server>, artifacts: &Path) -> Config {
    let mut c =
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/fixture.toml")).unwrap();
    c.artifacts = artifacts.into();
    c.embeddings.truncate(1);
    if let Some(server) = server {
        c.fixture_stdio = None;
        for m in c
            .embeddings
            .iter_mut()
            .chain(&mut c.rerankers)
            .chain(c.answer.iter_mut())
            .chain(c.judge.iter_mut())
        {
            m.url = format!("{}/v1", server.url);
        }
    }
    c
}

#[tokio::test]
async fn full_workflow_resume_and_judge_only_invalidation() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config(None, temp.path());
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let cases = vec![cases[0].clone(), cases[1].clone(), cases[54].clone()];
    let client = LocalClient::new(&c).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let (run, path) = pipeline::evaluate(&c, &chunks, &cases, true, &client)
        .await
        .unwrap();
    assert_eq!(run.outcomes.len(), 9);
    for o in &run.outcomes {
        assert_eq!(o.status, "completed", "{:?}", o.error);
        assert!(o.answer.is_some());
        assert!(o.generation.is_some());
        assert!(
            o.ranked
                .iter()
                .all(|r| o.case.corpus_languages.contains(&r.chunk.language))
        );
    }
    assert_eq!(
        run.outcomes[0]
            .candidates
            .iter()
            .map(|c| &c.chunk.id)
            .collect::<Vec<_>>(),
        run.outcomes[1]
            .candidates
            .iter()
            .map(|c| &c.chunk.id)
            .collect::<Vec<_>>()
    );
    assert!(run.outcomes[6].retrieval.is_none());
    report::write(&run, &path.parent().unwrap().join("reports")).unwrap();
    for file in ["report.md", "summary.json", "results.csv"] {
        assert!(path.parent().unwrap().join("reports").join(file).exists());
    }
    let before = client.fixture_stats().await.unwrap();
    let (resumed, _) = pipeline::evaluate(&c, &chunks, &cases, true, &client)
        .await
        .unwrap();
    assert_eq!(
        before,
        client.fixture_stats().await.unwrap(),
        "completed runs must not call inference again"
    );
    assert!(
        resumed
            .outcomes
            .iter()
            .flat_map(|o| o.stages.values())
            .all(|e| e.cached)
    );
    c.judge.as_mut().unwrap().revision = "fixture-v2".into();
    let (changed, _) = pipeline::evaluate(&c, &chunks, &cases, true, &client)
        .await
        .unwrap();
    let after = client.fixture_stats().await.unwrap();
    assert_eq!(before["/v1/embeddings"], after["/v1/embeddings"]);
    assert_eq!(before["/v1/rerank"], after["/v1/rerank"]);
    assert!(
        after["/v1/chat/completions"].as_u64().unwrap()
            > before["/v1/chat/completions"].as_u64().unwrap()
    );
    assert!(changed.outcomes.iter().all(|o| o.stages["answer"].cached));
    assert!(changed.outcomes.iter().any(|o| !o.stages["judge"].cached));
    for (old, new) in run.outcomes.iter().zip(&changed.outcomes) {
        assert_ne!(old.stages["judge"].key, new.stages["judge"].key);
    }
    let summary = report::summarize(&run);
    assert!(
        summary
            .groups
            .keys()
            .any(|k| k.contains("/draft/track/or->or->or"))
    );
    assert!(summary.fixture);
}

#[tokio::test]
async fn invalid_reranker_does_not_discard_other_results() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config(None, temp.path());
    c.rerankers[0].model = "fixture-bad-ids".into();
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    let client = LocalClient::new(&c).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], false, &client)
        .await
        .unwrap();
    assert_eq!(run.outcomes[0].status, "completed");
    assert_eq!(run.outcomes[1].status, "failed");
    assert_eq!(run.outcomes[2].status, "completed");
    let (retry, _) = pipeline::evaluate(&c, &chunks, &cases[..1], false, &client)
        .await
        .unwrap();
    assert!(retry.outcomes[0].stages.values().all(|s| s.cached));
    assert_eq!(retry.outcomes[1].status, "failed");
}

#[tokio::test]
async fn model_capacity_profile_mismatch_and_synthetic_server_guard() {
    let temp = tempfile::tempdir().unwrap();
    let c = config(None, temp.path());
    let client = LocalClient::new(&c).unwrap();
    let mut wrong = c.embeddings[0].clone();
    wrong.dimensions = 8;
    assert!(client.embed(&wrong, &["query".into()], true).await.is_err());
    let mut chat = c.answer.clone().unwrap();
    chat.context_tokens = 33;
    chat.max_output_tokens = 32;
    assert!(
        client
            .answer(
                &chat,
                "a long question that exceeds a tiny budget",
                "or",
                &[]
            )
            .await
            .is_err()
    );
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let candidates = pipeline::retrieve(
        &c,
        &chunks,
        &client,
        &c.embeddings[0],
        &cases[0].query,
        &["en".into()],
    )
    .await
    .unwrap();
    let mut bad = c.rerankers[0].clone();
    bad.model = "fixture-bad-ids".into();
    assert!(client.rerank(&bad, "query", &candidates).await.is_err());
    let mut real = c.clone();
    real.fixture = false;
    assert!(LocalClient::new(&real).is_err());
}

#[tokio::test]
#[ignore = "requires permission to bind loopback sockets; run cargo test --test workflow -- --ignored"]
async fn real_http_transport_contract() {
    let server = Server::start();
    let temp = tempfile::tempdir().unwrap();
    let c = config(Some(&server), temp.path());
    let client = LocalClient::new(&c).unwrap();
    assert!(client.health(&c.embeddings[0]).await.is_ok());
    let embedded = client
        .embed(&c.embeddings[0], &["ଓଡ଼ିଆ".into()], true)
        .await
        .unwrap();
    assert_eq!(embedded.value[0].len(), 768);
    let (chunks, cases) = data::load(&c.corpus, &c.dataset).unwrap();
    pipeline::index(&c, &chunks, &client).await.unwrap();
    let (run, _) = pipeline::evaluate(&c, &chunks, &cases[..1], true, &client)
        .await
        .unwrap();
    assert!(run.outcomes.iter().all(|o| o.status == "completed"));
    let mut real = c.clone();
    real.fixture = false;
    let guarded = LocalClient::new(&real).unwrap();
    assert!(guarded.health(&real.embeddings[0]).await.is_err());
}
