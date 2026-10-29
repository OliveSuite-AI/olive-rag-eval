use crate::{clients::Timed, config::RrfConfig, data::RankedChunk};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    collections::{BTreeMap, HashSet},
    time::Instant,
};

pub fn diagnostics(
    dense: &[RankedChunk],
    sparse: &[RankedChunk],
    fused: &[RankedChunk],
    labels: &[crate::data::RelevantPage],
) -> crate::metrics::Scores {
    let dense_ids: HashSet<_> = dense.iter().map(|h| &h.chunk.id).collect();
    let sparse_ids: HashSet<_> = sparse.iter().map(|h| &h.chunk.id).collect();
    let overlap = dense_ids.intersection(&sparse_ids).count();
    let union = dense_ids.union(&sparse_ids).count();
    let mut scores = BTreeMap::from([
        ("branch_overlap_count".into(), overlap as f64),
        (
            "branch_jaccard".into(),
            if union == 0 {
                0.
            } else {
                overlap as f64 / union as f64
            },
        ),
    ]);
    if !labels.is_empty() {
        let relevant = |hits: &[RankedChunk]| -> HashSet<(String, u32)> {
            hits.iter()
                .filter(|h| {
                    labels
                        .iter()
                        .any(|l| l.source == h.chunk.source && l.page == h.chunk.page)
                })
                .map(|h| (h.chunk.source.clone(), h.chunk.page))
                .collect()
        };
        let before = relevant(dense);
        let after = relevant(fused);
        scores.insert(
            "relevant_pages_added_by_fusion".into(),
            after.difference(&before).count() as f64,
        );
        scores.insert(
            "relevant_pages_lost_by_fusion".into(),
            before.difference(&after).count() as f64,
        );
    }
    scores
}

pub fn fuse(
    dense: &[RankedChunk],
    sparse: &[RankedChunk],
    p: &RrfConfig,
) -> Result<Timed<Vec<RankedChunk>>> {
    ensure!(
        p.rank_constant.is_finite() && p.rank_constant > 0. && p.top_k > 0,
        "invalid RRF parameters"
    );
    let start = Instant::now();
    let mut hits = BTreeMap::<String, RankedChunk>::new();
    for (branch, list) in [("dense", dense), ("sparse", sparse)] {
        let mut seen = HashSet::new();
        for (i, hit) in list.iter().enumerate() {
            ensure!(
                seen.insert(&hit.chunk.id),
                "duplicate chunk ID within {branch} list"
            );
            let entry = hits.entry(hit.chunk.id.clone()).or_insert_with(|| {
                let mut h = hit.clone();
                h.vector_rank = None;
                h.vector_score = None;
                h.sparse_rank = None;
                h.sparse_score = None;
                h.fused_rank = None;
                h.rrf_score = Some(0.);
                h.rerank_score = None;
                h
            });
            ensure!(
                entry.chunk.text == hit.chunk.text
                    && entry.chunk.source == hit.chunk.source
                    && entry.chunk.page == hit.chunk.page
                    && entry.chunk.language == hit.chunk.language,
                "conflicting chunk identity across search branches"
            );
            if branch == "dense" {
                entry.vector_rank = Some(i + 1);
                entry.vector_score = hit.vector_score;
            } else {
                entry.sparse_rank = Some(i + 1);
                entry.sparse_score = hit.sparse_score;
            }
            entry.rrf_score =
                Some(entry.rrf_score.unwrap() + 1. / (p.rank_constant + (i + 1) as f64));
        }
    }
    let union_count = hits.len();
    let mut value: Vec<_> = hits.into_values().collect();
    value.sort_by(|a, b| {
        b.rrf_score
            .unwrap()
            .total_cmp(&a.rrf_score.unwrap())
            .then(a.chunk.id.cmp(&b.chunk.id))
    });
    value.truncate(p.top_k);
    for (i, h) in value.iter_mut().enumerate() {
        h.rank = i + 1;
        h.fused_rank = Some(i + 1);
    }
    Ok(Timed {
        value,
        seconds: start.elapsed().as_secs_f64(),
        warmup_seconds: 0.,
        raw: json!({"union_count":union_count,"rank_constant":p.rank_constant,"dense_count":dense.len(),"sparse_count":sparse.len()}),
    })
}
