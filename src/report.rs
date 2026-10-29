use crate::{
    metrics::{Scores, percentile},
    pipeline::{Outcome, Run},
};
use anyhow::Result;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

#[derive(Serialize)]
pub struct MetricSummary {
    pub mean: f64,
    pub count: usize,
    pub semantic_cases: usize,
    pub cluster_ci95: Option<[f64; 2]>,
}
#[derive(Serialize)]
pub struct GroupSummary {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub metrics: BTreeMap<String, MetricSummary>,
}
#[derive(Serialize)]
pub struct Summary {
    pub fixture: bool,
    pub same_answer_judge_model: bool,
    pub groups: BTreeMap<String, GroupSummary>,
    pub balanced_language_track_means: BTreeMap<String, Scores>,
    pub latency: BTreeMap<String, serde_json::Value>,
    pub provider_usage: BTreeMap<String, BTreeMap<String, u64>>,
    pub caveats: Vec<String>,
    pub calibration: serde_json::Value,
}
fn metrics(o: &Outcome) -> Scores {
    let mut s = o.retrieval.clone().unwrap_or_default();
    s.extend(o.generation.clone().unwrap_or_default());
    for (stage, scores) in &o.stage_metrics {
        for (key, value) in scores {
            s.insert(format!("stage/{stage}/{key}"), *value);
        }
    }
    for (key, value) in &o.diagnostics {
        s.insert(format!("diagnostic/{key}"), *value);
    }
    if let Some(r) = o.candidate_recall {
        s.insert("candidate_recall".into(), r);
    }
    s
}
fn track(o: &Outcome) -> String {
    format!(
        "{}->{}->{}",
        o.case.query_language,
        o.case.corpus_languages.join("+"),
        o.case.answer_language
    )
}
fn usage(raw: &serde_json::Value, totals: &mut BTreeMap<String, u64>) {
    if let Some(object) = raw.get("usage").and_then(serde_json::Value::as_object) {
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            if let Some(n) = object.get(key).and_then(serde_json::Value::as_u64) {
                *totals.entry(key.into()).or_default() += n;
            }
        }
    }
    for field in ["initial", "repair"] {
        if let Some(value) = raw.get(field) {
            usage(value, totals);
        }
    }
}
fn bootstrap(samples: &BTreeMap<String, Vec<f64>>) -> Option<[f64; 2]> {
    if samples.len() < 2 {
        return None;
    }
    let groups: Vec<_> = samples.values().collect();
    let mut seed: u64 = 0x0d1a2026;
    let mut means = Vec::new();
    for _ in 0..1000 {
        let mut sum = 0.;
        let mut n = 0;
        for _ in 0..groups.len() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let sample = groups[(seed as usize) % groups.len()];
            sum += sample.iter().sum::<f64>();
            n += sample.len();
        }
        means.push(sum / n as f64);
    }
    let low = percentile(&mut means, 0.025)?;
    let high = percentile(&mut means, 0.975)?;
    Some([low, high])
}
pub fn summarize(run: &Run) -> Summary {
    let mut buckets: BTreeMap<String, Vec<&Outcome>> = BTreeMap::new();
    for o in &run.outcomes {
        let profile = if o.pipeline.is_empty() {
            o.embedding.as_deref().unwrap_or("none")
        } else {
            &o.pipeline
        };
        let root = format!("{}/{}/{}", profile, o.ranker, o.case.review_status);
        for group in [
            format!("{root}/all"),
            format!("{root}/track/{}", track(o)),
            format!("{root}/script/{}", o.case.query_script),
            format!("{root}/type/{}", o.case.question_type),
        ] {
            buckets.entry(group).or_default().push(o);
        }
    }
    let mut groups = BTreeMap::new();
    for (name, outcomes) in buckets {
        let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut clusters: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
        for o in &outcomes {
            for (key, value) in metrics(o) {
                values.entry(key.clone()).or_default().push(value);
                clusters
                    .entry(key)
                    .or_default()
                    .entry(o.case.semantic_case_id.clone())
                    .or_default()
                    .push(value);
            }
        }
        let metric_summary = values
            .into_iter()
            .map(|(k, v)| {
                let summary = MetricSummary {
                    mean: v.iter().sum::<f64>() / v.len() as f64,
                    count: v.len(),
                    semantic_cases: clusters[&k].len(),
                    cluster_ci95: bootstrap(&clusters[&k]),
                };
                (k, summary)
            })
            .collect();
        let completed = outcomes.iter().filter(|o| o.status == "completed").count();
        groups.insert(
            name,
            GroupSummary {
                total: outcomes.len(),
                completed,
                failed: outcomes.len() - completed,
                metrics: metric_summary,
            },
        );
    }
    let mut balanced: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for (name, g) in &groups {
        if let Some((root, _)) = name.split_once("/track/") {
            for (k, v) in &g.metrics {
                balanced
                    .entry(root.into())
                    .or_default()
                    .entry(k.clone())
                    .or_default()
                    .push(v.mean);
            }
        }
    }
    let balanced_language_track_means = balanced
        .into_iter()
        .map(|(root, ms)| {
            (
                root,
                ms.into_iter()
                    .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
                    .collect(),
            )
        })
        .collect();
    let mut timings: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut warmups: BTreeMap<String, f64> = BTreeMap::new();
    let mut provider_usage: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for o in &run.outcomes {
        for (kind, e) in &o.stages {
            if seen.insert(e.key.clone()) {
                let profile = match kind.as_str() {
                    "rerank" | "llm_rerank" => &o.ranker,
                    "bm25" => "bm25",
                    "rrf" => "rrf",
                    "answer" => run
                        .config
                        .answer
                        .as_ref()
                        .map_or("none", |m| m.name.as_str()),
                    "judge" => run
                        .config
                        .judge
                        .as_ref()
                        .map_or("none", |m| m.name.as_str()),
                    _ => o.embedding.as_deref().unwrap_or("none"),
                };
                let key = format!("{profile}/{kind}");
                usage(&e.raw, provider_usage.entry(key.clone()).or_default());
                timings.entry(key.clone()).or_default().push(e.seconds);
                *warmups.entry(key).or_default() += e.warmup_seconds;
            }
        }
    }
    for calibration in &run.calibration {
        for (kind, e) in &calibration.stages {
            if seen.insert(e.key.clone()) {
                let profile = run
                    .config
                    .judge
                    .as_ref()
                    .map_or("none", |m| m.name.as_str());
                let key = format!("{profile}/{kind}");
                usage(&e.raw, provider_usage.entry(key.clone()).or_default());
                timings.entry(key.clone()).or_default().push(e.seconds);
                *warmups.entry(key).or_default() += e.warmup_seconds;
            }
        }
    }
    provider_usage.retain(|_, counts| !counts.is_empty());
    let latency = timings.into_iter().map(|(k,mut v)| {
        let p50=percentile(&mut v,0.5); let p95=percentile(&mut v,0.95);
        let val=serde_json::json!({"samples":v.len(),"p50_seconds":p50,"p95_seconds":p95,"warmup_seconds":warmups[&k],"includes_original_measurements_from_cached_stages":true}); (k,val)
    }).collect();
    let calibrated: Vec<_> = run
        .calibration
        .iter()
        .filter_map(|c| c.reviewed_label_mean_absolute_error)
        .collect();
    let calibration = serde_json::json!({"examples":run.calibration.len(),"reviewed_scored_examples":calibrated.len(),"failed_examples":run.calibration.iter().filter(|c|c.error.is_some()).count(),"reviewed_label_mean_absolute_error":if calibrated.is_empty() {None} else {Some(calibrated.iter().sum::<f64>()/calibrated.len() as f64)}});
    Summary {fixture:run.fixture,same_answer_judge_model:run.same_answer_judge_model,groups,balanced_language_track_means,latency,provider_usage,calibration,caveats:vec![
        "Draft translations and judgments are preliminary; reviewed status requires a fluent speaker.".into(),
        "Confidence intervals resample semantic cases, grouping translations, variants and repeats; small suites give uncertain estimates.".into(),
        "Absent or failed metrics are omitted from means; counts and failure totals must be read alongside scores.".into(),
        "Answer-language compliance is judge-assessed; Odia judge calibration is required before quality claims.".into(),
    ]}
}
fn escape(s: &str) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}
pub fn write(run: &Run, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let summary = summarize(run);
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let mut md = format!(
        "# Odia-first RAG evaluation\n\nMode: {}. Fixture: **{}**. Same answer/judge model: {}.\n\n",
        run.mode, run.fixture, run.same_answer_judge_model
    );
    for c in &summary.caveats {
        md.push_str(&format!("- {c}\n"));
    }
    md.push_str(&format!("\nJudge calibration: {} examples, {} reviewed/scored, {} failures. Reviewed-label MAE: {}.\n",summary.calibration["examples"],summary.calibration["reviewed_scored_examples"],summary.calibration["failed_examples"],summary.calibration["reviewed_label_mean_absolute_error"]));
    md.push_str("\n## Metrics\n\n| Group | Completed / total | Metric | Mean | Samples | Semantic cases | Cluster 95% interval |\n|---|---:|---|---:|---:|---:|---|\n");
    for (name, g) in &summary.groups {
        for (metric, m) in &g.metrics {
            md.push_str(&format!(
                "| {} | {} / {} | {} | {:.4} | {} | {} | {} |\n",
                escape(name),
                g.completed,
                g.total,
                metric,
                m.mean,
                m.count,
                m.semantic_cases,
                m.cluster_ci95
                    .map_or("—".into(), |[l, h]| format!("{l:.4}–{h:.4}"))
            ));
        }
    }
    md.push_str("\n## Equal-weight language-track means\n\n| Profile / review status | Metric | Mean |\n|---|---|---:|\n");
    for (p, ms) in &summary.balanced_language_track_means {
        for (k, v) in ms {
            md.push_str(&format!("| {} | {k} | {v:.4} |\n", escape(p)));
        }
    }
    md.push_str("\n## Latency\n\nOriginal measured stage timings; cached stages counted once per key.\n\n| Model / stage | Samples | p50 seconds | p95 seconds | Warm-up seconds |\n|---|---:|---:|---:|---:|\n");
    for (k, v) in &summary.latency {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            escape(k),
            v["samples"],
            v["p50_seconds"],
            v["p95_seconds"],
            v["warmup_seconds"]
        ));
    }
    md.push_str("\n## Provider-reported usage\n\nMeasured successful requests only, deduplicated by stage key; excludes warm-up and unavailable usage. No estimates.\n\n");
    md.push_str(&format!(
        "```json\n{}\n```\n",
        serde_json::to_string_pretty(&summary.provider_usage)?
    ));
    md.push_str("\n## Per-query results\n\n| Case | Pipeline | Retrieval | Embedding / ranker | Status | Answer | Error |\n|---|---|---|---|---|---|---|\n");
    let mut csv = csv::Writer::from_path(dir.join("results.csv"))?;
    csv.write_record([
        "case_id",
        "semantic_case_id",
        "query_language",
        "corpus_languages",
        "answer_language",
        "script",
        "review_status",
        "embedding",
        "pipeline",
        "retrieval_mode",
        "ranker",
        "repeat",
        "status",
        "metric",
        "value",
        "error",
    ])?;
    for o in &run.outcomes {
        md.push_str(&format!(
            "| {} | {} | {} | {} / {} | {} | {} | {} |\n",
            escape(&o.case.id),
            escape(&o.pipeline),
            serde_json::to_string(&o.retrieval_mode)?.trim_matches('"'),
            escape(o.embedding.as_deref().unwrap_or("—")),
            escape(&o.ranker),
            o.status,
            escape(o.answer.as_deref().unwrap_or("")),
            escape(o.error.as_deref().unwrap_or(""))
        ));
        let ms = metrics(o);
        let ms = if ms.is_empty() {
            vec![(String::new(), String::new())]
        } else {
            ms.into_iter().map(|(k, v)| (k, v.to_string())).collect()
        };
        for (k, v) in ms {
            csv.write_record([
                o.case.id.clone(),
                o.case.semantic_case_id.clone(),
                o.case.query_language.clone(),
                o.case.corpus_languages.join("+"),
                o.case.answer_language.clone(),
                o.case.query_script.clone(),
                o.case.review_status.clone(),
                o.embedding.clone().unwrap_or_default(),
                o.pipeline.clone(),
                serde_json::to_string(&o.retrieval_mode)?
                    .trim_matches('"')
                    .into(),
                o.ranker.clone(),
                o.repeat.to_string(),
                o.status.clone(),
                k,
                v,
                o.error.clone().unwrap_or_default(),
            ])?;
        }
    }
    csv.flush()?;
    std::fs::write(dir.join("report.md"), md)?;
    Ok(())
}
