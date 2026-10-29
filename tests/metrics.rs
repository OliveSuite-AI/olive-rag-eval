use rag_eval::{
    data::{Chunk, RankedChunk, RelevantPage},
    metrics::{self, Claim, Judgment},
};

fn ranked(pages: &[u32]) -> Vec<RankedChunk> {
    pages
        .iter()
        .enumerate()
        .map(|(i, &page)| RankedChunk {
            chunk: Chunk {
                id: format!("c{i}"),
                text: "evidence".into(),
                source: "source".into(),
                page,
                language: "or".into(),
                source_url: None,
                embedding: None,
                embedding_profile: None,
            },
            vector_rank: Some(i + 1),
            vector_score: Some(1.),
            rank: i + 1,
            rerank_score: None,
            sparse_rank: None,
            sparse_score: None,
            fused_rank: None,
            rrf_score: None,
            raw: serde_json::Value::Null,
        })
        .collect()
}
fn labels() -> Vec<RelevantPage> {
    [(7, 2), (76, 2), (77, 1)]
        .into_iter()
        .map(|(page, grade)| RelevantPage {
            source: "source".into(),
            page,
            grade,
        })
        .collect()
}

#[test]
fn reference_metric_example_credits_pages_once_without_collapsing_positions() {
    let s = metrics::retrieval(&ranked(&[76, 3, 7, 76, 9, 77]), &labels(), 5).unwrap();
    assert_eq!(s["hit_rate@5"], 1.);
    assert_eq!(s["precision@5"], 0.4);
    assert!((s["recall@5"] - 2. / 3.).abs() < 1e-10);
    assert_eq!(s["mrr"], 1.);
    assert!((s["map"] - (1. + 2. / 3. + 3. / 6.) / 3.).abs() < 1e-10);
    let ideal = 3. + 3. / 3f64.log2() + 1. / 4f64.log2();
    assert!((s["ndcg@5"] - 4.5 / ideal).abs() < 1e-10);
}
#[test]
fn missing_pages_short_rankings_and_unanswerable() {
    let s = metrics::retrieval(&ranked(&[7]), &labels(), 5).unwrap();
    assert_eq!(s["precision@5"], 0.2);
    assert_eq!(s["recall@5"], 1. / 3.);
    let s = metrics::retrieval(&[], &labels(), 5).unwrap();
    assert!(s.values().all(|&v| v == 0.));
    assert!(metrics::retrieval(&ranked(&[7]), &[], 5).is_none());
}
#[test]
fn distinct_sources_with_same_page_are_distinct() {
    let mut r = ranked(&[7, 7]);
    r[1].chunk.source = "translation".into();
    let mut ls = labels();
    ls.push(RelevantPage {
        source: "translation".into(),
        page: 7,
        grade: 2,
    });
    assert_eq!(metrics::retrieval(&r, &ls, 5).unwrap()["recall@5"], 0.5);
}
fn judgment() -> Judgment {
    Judgment {
        claims: vec![
            Claim {
                text: "grounded".into(),
                supported: true,
            },
            Claim {
                text: "invented".into(),
                supported: false,
            },
        ],
        unsupported_claims: vec!["invented".into()],
        useful_chunk_ids: vec!["c0".into()],
        covered_fact_ids: vec![1],
        answer_relevance: 0.5,
        abstained: false,
        answer_language_correct: true,
    }
}
#[test]
fn generation_scores_expose_unsupported_claims() {
    let s = judgment().scores(&ranked(&[7, 76]), 2, false).unwrap();
    assert_eq!(s["faithfulness"], 0.5);
    assert_eq!(s["hallucination_rate"], 1.);
    assert_eq!(s["context_relevance"], 0.5);
    assert_eq!(s["answer_correctness"], 0.5);
    assert_eq!(s["abstention_accuracy"], 1.);
}
#[test]
fn abstentions_cannot_hide_behind_perfect_faithfulness() {
    let mut j = judgment();
    j.claims.clear();
    j.unsupported_claims.clear();
    j.useful_chunk_ids.clear();
    j.abstained = true;
    assert_eq!(j.scores(&[], 2, false).unwrap()["faithfulness"], 1.);
    assert_eq!(j.scores(&[], 2, false).unwrap()["abstention_accuracy"], 0.);
    assert_eq!(j.scores(&[], 2, true).unwrap()["abstention_accuracy"], 1.);
    assert!(
        !j.scores(&[], 2, true)
            .unwrap()
            .contains_key("context_relevance")
    );
}
#[test]
fn malformed_judge_evidence_is_rejected() {
    let r = ranked(&[7]);
    let mut j = judgment();
    j.unsupported_claims.clear();
    assert!(j.scores(&r, 2, false).is_err());
    let mut j = judgment();
    j.useful_chunk_ids = vec!["missing".into()];
    assert!(j.scores(&r, 2, false).is_err());
    let mut j = judgment();
    j.covered_fact_ids = vec![1, 1];
    assert!(j.scores(&r, 2, false).is_err());
    let mut j = judgment();
    j.covered_fact_ids = vec![0];
    assert!(j.scores(&r, 2, false).is_err());
    let mut j = judgment();
    j.answer_relevance = 0.7;
    assert!(j.scores(&r, 2, false).is_err());
}
