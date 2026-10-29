use crate::data::{RankedChunk, RelevantPage};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub type Scores = BTreeMap<String, f64>;
pub fn retrieval(ranked: &[RankedChunk], labels: &[RelevantPage], k: usize) -> Option<Scores> {
    if labels.is_empty() {
        return None;
    }
    let mut seen = HashSet::new();
    let grades: Vec<u8> = ranked
        .iter()
        .map(|r| {
            let key = (&r.chunk.source, r.chunk.page);
            if !seen.insert(key) {
                return 0;
            }
            labels
                .iter()
                .find(|p| p.source == *key.0 && p.page == key.1)
                .map_or(0, |p| p.grade)
        })
        .collect();
    let hits = grades.iter().take(k).filter(|&&g| g > 0).count() as f64;
    let rr = grades
        .iter()
        .position(|&g| g > 0)
        .map_or(0., |p| 1. / (p + 1) as f64);
    let mut found = 0.;
    let ap = grades
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            if g > 0 {
                found += 1.;
                found / (i + 1) as f64
            } else {
                0.
            }
        })
        .sum::<f64>()
        / labels.len() as f64;
    let dcg = |gs: &[u8]| {
        gs.iter()
            .take(k)
            .enumerate()
            .map(|(i, g)| (2f64.powi(*g as i32) - 1.) / ((i + 2) as f64).log2())
            .sum::<f64>()
    };
    let mut ideal: Vec<_> = labels.iter().map(|p| p.grade).collect();
    ideal.sort_by(|a, b| b.cmp(a));
    Some(BTreeMap::from([
        (format!("hit_rate@{k}"), if hits > 0. { 1. } else { 0. }),
        (format!("precision@{k}"), hits / k as f64),
        (format!("recall@{k}"), hits / labels.len() as f64),
        ("mrr".into(), rr),
        ("map".into(), ap),
        (format!("ndcg@{k}"), dcg(&grades) / dcg(&ideal)),
    ]))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub text: String,
    pub supported: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Judgment {
    pub claims: Vec<Claim>,
    pub unsupported_claims: Vec<String>,
    pub useful_chunk_ids: Vec<String>,
    pub covered_fact_ids: Vec<usize>,
    pub answer_relevance: f64,
    pub abstained: bool,
    pub answer_language_correct: bool,
}
impl Judgment {
    pub fn scores(
        &self,
        chunks: &[RankedChunk],
        n_facts: usize,
        unanswerable: bool,
    ) -> Result<Scores> {
        ensure!(
            [0., 0.5, 1.].contains(&self.answer_relevance),
            "invalid answer relevance"
        );
        ensure!(
            self.claims.iter().all(|c| !c.text.trim().is_empty()),
            "empty claim"
        );
        let unsupported: Vec<_> = self
            .claims
            .iter()
            .filter(|c| !c.supported)
            .map(|c| c.text.clone())
            .collect();
        ensure!(
            unsupported == self.unsupported_claims,
            "unsupported claim evidence inconsistent"
        );
        let ids: HashSet<_> = chunks.iter().map(|c| &c.chunk.id).collect();
        ensure!(
            self.useful_chunk_ids.iter().all(|id| ids.contains(id))
                && self.useful_chunk_ids.iter().collect::<HashSet<_>>().len()
                    == self.useful_chunk_ids.len(),
            "invalid/duplicate useful chunk IDs"
        );
        ensure!(
            self.covered_fact_ids
                .iter()
                .all(|&id| id > 0 && id <= n_facts)
                && self.covered_fact_ids.iter().collect::<HashSet<_>>().len()
                    == self.covered_fact_ids.len(),
            "invalid/duplicate fact IDs"
        );
        let supported = self.claims.iter().filter(|c| c.supported).count();
        let mut scores = BTreeMap::from([
            (
                "faithfulness".into(),
                if self.claims.is_empty() {
                    1.
                } else {
                    supported as f64 / self.claims.len() as f64
                },
            ),
            (
                "hallucination_rate".into(),
                if unsupported.is_empty() { 0. } else { 1. },
            ),
            ("answer_relevance".into(), self.answer_relevance),
            (
                "abstention_accuracy".into(),
                if self.abstained == unanswerable {
                    1.
                } else {
                    0.
                },
            ),
            (
                "answer_language_compliance".into(),
                if self.answer_language_correct { 1. } else { 0. },
            ),
        ]);
        if !chunks.is_empty() {
            scores.insert(
                "context_relevance".into(),
                self.useful_chunk_ids.len() as f64 / chunks.len() as f64,
            );
        }
        if n_facts > 0 {
            scores.insert(
                "answer_correctness".into(),
                self.covered_fact_ids.len() as f64 / n_facts as f64,
            );
        }
        Ok(scores)
    }
}

pub fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(
        values[((values.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)],
    )
}
