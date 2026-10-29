use rag_eval::{
    config::Config,
    data::{self, Chunk},
    store::{self, Snapshot, StageStore},
};
use std::path::Path;

fn config() -> Config {
    Config::load(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("configs/fixture.toml")
            .as_path(),
    )
    .unwrap()
}
#[test]
fn snapshot_roundtrip_filters_languages_and_detects_corruption() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config();
    c.artifacts = temp.path().into();
    c.embeddings[0].dimensions = 8;
    let chunks: Vec<Chunk> = [("a", "or"), ("b", "en")]
        .into_iter()
        .map(|(id, language)| Chunk {
            id: id.into(),
            text: "ଓଡ଼ିଆ".into(),
            source: id.into(),
            page: 1,
            language: language.into(),
            source_url: None,
            embedding: None,
            embedding_profile: None,
        })
        .collect();
    let vectors = vec![
        vec![1., 0., 0., 0., 0., 0., 0., 0.],
        vec![0., 1., 0., 0., 0., 0., 0., 0.],
    ];
    let dir = store::build_snapshot(
        &c,
        &c.embeddings[0],
        &chunks,
        &vectors,
        serde_json::json!({}),
    )
    .unwrap();
    let snapshot = Snapshot::load(&dir).unwrap();
    let r = snapshot.search(&vectors[0], &["en".into()], 20).unwrap();
    assert_eq!(r.value.len(), 1);
    assert_eq!(r.value[0].chunk.id, "b");
    assert_eq!(r.value[0].chunk.text, "ଓଡ଼ିଆ");
    assert!(
        snapshot
            .search(&vectors[0], &["hi".into()], 20)
            .unwrap()
            .value
            .is_empty()
    );
    assert!(
        store::build_snapshot(
            &c,
            &c.embeddings[0],
            &chunks,
            &vectors,
            serde_json::json!({})
        )
        .is_err()
    );
    std::fs::write(dir.join("index.tvim"), b"corrupt").unwrap();
    assert!(Snapshot::load(&dir).is_err());
}
#[test]
fn cache_only_reuses_completed_exact_keys() {
    let temp = tempfile::tempdir().unwrap();
    let mut s = StageStore::open(&temp.path().join("stages.sqlite")).unwrap();
    s.put("one", "judge", "failed", &42).unwrap();
    assert!(s.get::<i32>("one").unwrap().is_none());
    s.put("one", "judge", "completed", &42).unwrap();
    assert_eq!(s.get::<i32>("one").unwrap(), Some(42));
    assert!(s.get::<i32>("two").unwrap().is_none());
}
#[test]
fn embedding_validation_and_unicode_preserve_odia_marks() {
    let text = "ଓଡ଼ିଆ\u{200d} ୧୨୩";
    let n = data::nfc(text);
    assert!(n.contains('\u{200d}'));
    assert!(n.contains('଼'));
    assert!(n.contains('୧'));
    assert_eq!(data::nfc(&n), n);
    let mut v = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    data::normalize(&mut v, 8).unwrap();
    assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-5);
    assert!(data::normalize(&mut [0.; 8], 8).is_err());
    assert!(data::normalize(&mut [f32::NAN; 8], 8).is_err());
    assert!(data::normalize(&mut [1.; 7], 8).is_err());
}
#[test]
fn configuration_rejects_remote_servers_bad_dimensions_and_placeholder_artifacts() {
    let mut c = config();
    c.embeddings[0].url = "https://example.com/v1".into();
    assert!(c.validate().is_err());
    let mut c = config();
    c.embeddings[0].dimensions = 7;
    assert!(c.validate().is_err());
    let mut c = config();
    c.fixture = false;
    assert!(c.validate().is_err());
    let mut c = config();
    c.rerankers[0].name = c.embeddings[0].name.clone();
    assert!(c.validate().is_err());
}
#[test]
fn answer_prompt_excludes_references_and_enforces_requested_language() {
    let messages = rag_eval::clients::answer_messages("question", "or", &[]).to_string();
    assert!(messages.contains("Odia"));
    assert!(messages.contains("question"));
    assert!(!messages.contains("reference_answer"));
    assert!(!messages.contains("key_facts"));
}
