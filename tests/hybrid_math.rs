use rag_eval::{
    clients::validate_ordering,
    config::{Bm25Config, Config, RrfConfig},
    data::{Chunk, RankedChunk},
    fusion,
    sparse::{self, SparseIndex},
};
use std::path::Path;

fn config() -> Config {
    Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/hybrid-fixture.toml"))
        .unwrap()
}
fn chunk(id: &str, text: &str, language: &str) -> Chunk {
    Chunk {
        id: id.into(),
        text: text.into(),
        source: id.into(),
        page: 1,
        language: language.into(),
        source_url: None,
        embedding: None,
        embedding_profile: None,
    }
}
fn hit(id: &str, rank: usize, dense: bool) -> RankedChunk {
    RankedChunk {
        chunk: chunk(id, id, "en"),
        vector_rank: dense.then_some(rank),
        vector_score: dense.then_some(999.),
        sparse_rank: (!dense).then_some(rank),
        sparse_score: (!dense).then_some(0.01),
        fused_rank: None,
        rrf_score: None,
        rank,
        rerank_score: None,
        raw: serde_json::Value::Null,
    }
}

#[test]
fn bm25_scoped_arithmetic_and_repeated_query_terms() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config();
    c.artifacts = temp.path().into();
    let chunks = vec![
        chunk("a", "cat cat", "en"),
        chunk("b", "cat dog dog", "en"),
        chunk("c", "cat", "or"),
    ];
    let dir = sparse::build(&c, &chunks).unwrap();
    let index = SparseIndex::load(&dir).unwrap();
    let p = Bm25Config::default();
    let result = index.search("cat", &["en".into()], &p).unwrap();
    assert_eq!(result.value[0].chunk.id, "a");
    assert_eq!(result.value.len(), 2);
    let expected = 1.2_f64.ln() * (2. * 2.2) / (2. + 1.2 * (0.25 + 0.75 * 2. / 2.5));
    assert!((result.value[0].sparse_score.unwrap() - expected).abs() < 1e-12);
    let repeated = index.search("cat cat CAT", &["en".into()], &p).unwrap();
    assert_eq!(result.value[0].sparse_score, repeated.value[0].sparse_score);
    let odia = index.search("cat", &["or".into()], &p).unwrap();
    assert_eq!(odia.value.len(), 1);
    assert_eq!(odia.value[0].chunk.id, "c");
    assert!((odia.value[0].sparse_score.unwrap() - (4_f64 / 3.).ln()).abs() < 1e-12);
    assert!(
        result
            .value
            .iter()
            .all(|h| h.vector_rank.is_none() && h.vector_score.is_none())
    );
    assert!(
        index
            .search("unmatched", &["en".into()], &p)
            .unwrap()
            .value
            .is_empty()
    );
    assert!(
        index
            .search("cat", &["hi".into()], &p)
            .unwrap()
            .value
            .is_empty()
    );
    assert!(index.search("...", &[], &p).unwrap().value.is_empty());
    assert_eq!(dir, sparse::build(&c, &chunks).unwrap());
    let mut changed = chunks.clone();
    changed[0].embedding = Some(vec![1.; 8]);
    assert_eq!(sparse::corpus_key(&chunks), sparse::corpus_key(&changed));
    std::fs::write(dir.join("bm25.sqlite"), "broken").unwrap();
    assert!(SparseIndex::load(&dir).is_err());
}
#[test]
fn odia_tokens_preserve_marks_conjuncts_and_distinct_numerals() {
    let words = sparse::tokenize("ଓଡ଼ିଆ ପୃଥିବୀ କ୍ଷେତ୍ର ABC abc ୧୨ 12");
    assert_eq!(
        words,
        vec!["ଓଡ଼ିଆ", "ପୃଥିବୀ", "କ୍ଷେତ୍ର", "abc", "abc", "୧୨", "12"]
    );
    assert_eq!(sparse::tokenize("ODIA"), vec!["odia"]);
    assert_ne!(sparse::tokenize("Odia"), sparse::tokenize("ଓଡ଼ିଆ"));
}
#[test]
fn rrf_uses_ranks_merges_overlap_and_keeps_sparse_only_hits() {
    let result = fusion::fuse(
        &[hit("a", 1, true), hit("b", 2, true)],
        &[hit("b", 1, false), hit("c", 2, false)],
        &RrfConfig::default(),
    )
    .unwrap();
    assert_eq!(
        result
            .value
            .iter()
            .map(|h| h.chunk.id.as_str())
            .collect::<Vec<_>>(),
        vec!["b", "a", "c"]
    );
    assert!((result.value[0].rrf_score.unwrap() - (1. / 61. + 1. / 62.)).abs() < 1e-12);
    assert_eq!(result.value[0].vector_rank, Some(2));
    assert_eq!(result.value[0].sparse_rank, Some(1));
    assert_eq!(result.value[2].vector_rank, None);
    assert_eq!(result.value[2].vector_score, None);
    assert_eq!(result.value[2].fused_rank, Some(3));
    assert_eq!(result.raw["union_count"], 3);
}
#[test]
fn rrf_disjoint_ties_empty_branches_truncation_and_invalid_duplicates() {
    let p = RrfConfig {
        rank_constant: 60.,
        top_k: 2,
    };
    let dense = vec![hit("z", 1, true), hit("b", 2, true)];
    let sparse = vec![hit("a", 1, false), hit("c", 2, false)];
    let r = fusion::fuse(&dense, &sparse, &p).unwrap();
    assert_eq!(
        r.value
            .iter()
            .map(|h| h.chunk.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "z"]
    );
    let empty = fusion::fuse(&dense, &[], &p).unwrap();
    assert_eq!(empty.value[0].chunk.id, "z");
    assert_eq!(empty.value[0].sparse_rank, None);
    assert!(fusion::fuse(&[], &[], &p).unwrap().value.is_empty());
    assert!(fusion::fuse(&[hit("a", 1, true), hit("a", 2, true)], &[], &p).is_err());
    let mut conflict = hit("z", 1, false);
    conflict.chunk.text = "different".into();
    assert!(fusion::fuse(&dense, &[conflict], &p).is_err());
    let mut a = hit("a", 1, true);
    let mut b = hit("b", 2, true);
    a.chunk.source = "same".into();
    b.chunk.source = "same".into();
    assert_eq!(fusion::fuse(&[a, b], &[], &p).unwrap().value.len(), 2);
}
#[test]
fn listwise_validation_requires_exact_permutation_without_fake_scores() {
    let candidates = vec![hit("a", 1, true), hit("b", 2, false)];
    let r = validate_ordering(r#"{"ordered_ids":["b","a"]}"#, &candidates).unwrap();
    assert_eq!(r[0].chunk.id, "b");
    assert!(r.iter().all(|h| h.rerank_score.is_none()));
    assert_eq!(r[0].vector_rank, None);
    for bad in [
        r#"{"ordered_ids":["a","a"]}"#,
        r#"{"ordered_ids":["a"]}"#,
        r#"{"ordered_ids":["a","c"]}"#,
        r#"{"ordered_ids":["a","b","c"]}"#,
        r#"{"ordered_ids":["a","b"],"scores":[]}"#,
        "a,b",
    ] {
        assert!(validate_ordering(bad, &candidates).is_err(), "{bad}");
    }
    let prompt = rag_eval::clients::llm_rerank_messages("q", &candidates);
    let data: serde_json::Value =
        serde_json::from_str(prompt[1]["content"].as_str().unwrap()).unwrap();
    assert!(data.get("reference_answer").is_none());
    assert!(data.get("key_facts").is_none());
    assert_eq!(data["documents"][0].as_object().unwrap().len(), 2);
}

#[test]
fn fusion_diagnostics_count_relevant_pages_added_and_lost() {
    let dense = vec![hit("a", 1, true), hit("b", 2, true)];
    let sparse = vec![hit("b", 1, false), hit("c", 2, false)];
    let fused = vec![hit("b", 1, true), hit("c", 2, false)];
    let labels = vec![
        rag_eval::data::RelevantPage {
            source: "a".into(),
            page: 1,
            grade: 2,
        },
        rag_eval::data::RelevantPage {
            source: "c".into(),
            page: 1,
            grade: 1,
        },
    ];
    let s = fusion::diagnostics(&dense, &sparse, &fused, &labels);
    assert_eq!(s["relevant_pages_added_by_fusion"], 1.);
    assert_eq!(s["relevant_pages_lost_by_fusion"], 1.);
    assert_eq!(s["branch_overlap_count"], 1.);
    assert!((s["branch_jaccard"] - 1. / 3.).abs() < 1e-12);
    assert!(
        !fusion::diagnostics(&dense, &sparse, &fused, &[])
            .contains_key("relevant_pages_added_by_fusion")
    );
}
