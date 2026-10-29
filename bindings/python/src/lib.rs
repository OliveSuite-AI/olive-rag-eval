use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
};
use rag_eval::{
    clients::LocalClient,
    config::{Bm25Config, Config, RrfConfig},
    data::{self, Chunk, RankedChunk, RelevantPage},
    pipeline, report,
    sparse::{self, SparseIndex},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn failure(error: anyhow::Error) -> PyErr {
    if error.chain().any(|e| e.is::<std::io::Error>()) {
        PyOSError::new_err(format!("{error:#}"))
    } else {
        PyRuntimeError::new_err(format!("{error:#}"))
    }
}
fn input_failure(error: anyhow::Error) -> PyErr {
    if error.chain().any(|e| e.is::<std::io::Error>()) {
        failure(error)
    } else {
        PyValueError::new_err(format!("{error:#}"))
    }
}
fn parse<T: DeserializeOwned>(text: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(|e| PyValueError::new_err(e.to_string()))
}
fn encode(value: &impl Serialize) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?)
}

#[pyfunction]
fn config_from_file(py: Python<'_>, path: String, python: String) -> PyResult<String> {
    let c = py
        .detach(move || Config::load(Path::new(&path)))
        .map_err(input_failure)?;
    let mut c = c;
    c.python_executable = Some(python.into());
    encode(&c)
}
#[pyfunction]
fn config_from_dict(
    py: Python<'_>,
    text: String,
    base_dir: String,
    python: String,
) -> PyResult<String> {
    let c: Config = parse(&text)?;
    let mut c = py
        .detach(move || c.resolve_paths(Path::new(&base_dir)))
        .map_err(input_failure)?;
    c.python_executable = Some(python.into());
    encode(&c)
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Arguments {
    online: bool,
    query: Option<String>,
    pipeline: Option<String>,
    embedding: Option<String>,
    languages: Vec<String>,
    limit: Option<usize>,
    verbose: bool,
}

#[pyfunction]
fn execute(
    py: Python<'_>,
    config: String,
    operation: String,
    arguments: String,
) -> PyResult<String> {
    let c: Config = parse(&config)?;
    c.validate().map_err(input_failure)?;
    let a: Arguments = parse(&arguments)?;
    if a.limit == Some(0) {
        return Err(PyValueError::new_err("limit must be positive"));
    }
    if operation == "retrieve"
        && (a.query.as_deref().unwrap_or("").trim().is_empty()
            || (a.pipeline.is_some() && a.embedding.is_some()))
    {
        return Err(PyValueError::new_err(
            "provide a nonempty query and at most one of pipeline/embedding",
        ));
    }
    let value=py.detach(move ||->anyhow::Result<Value> {
        let (chunks,mut cases)=data::load(&c.corpus,&c.dataset)?;
        runtime()?.block_on(async move {
            if operation=="validate" {
                let mut health=serde_json::Map::new();
                if a.online {let client=LocalClient::new(&c)?;for m in c.embeddings.iter().chain(&c.rerankers).chain(c.answer.iter()).chain(c.judge.iter()).chain(c.llm_reranker.iter()) {health.insert(m.name.clone(),client.health(m).await?);}}
                return Ok(json!({"chunks":chunks.len(),"cases":cases.len(),"semantic_cases":cases.iter().map(|c|&c.semantic_case_id).collect::<std::collections::HashSet<_>>().len(),"draft_cases":cases.iter().filter(|c|c.review_status=="draft").count(),"fixture":c.fixture,"pipelines":c.resolved_pipelines(),"health":health}));
            }
            let client=LocalClient::new(&c)?;
            match operation.as_str() {
                "index"=>Ok(serde_json::to_value(pipeline::index(&c,&chunks,&client).await?)?),
                "retrieve"=>{
                    let query=a.query.unwrap();
                    let hits=if let Some(name)=a.pipeline {
                        let p=c.resolved_pipelines().into_iter().find(|p|p.name==name).ok_or_else(||anyhow::anyhow!("unknown pipeline {name}"))?;
                        pipeline::retrieve_named(&c,&chunks,&client,&p,&query,&a.languages).await?
                    } else {
                        let m=if let Some(name)=a.embedding {c.embeddings.iter().find(|m|m.name==name).ok_or_else(||anyhow::anyhow!("unknown embedding {name}"))?} else {c.embeddings.first().ok_or_else(||anyhow::anyhow!("select a pipeline for sparse-only retrieval"))?};
                        pipeline::retrieve(&c,&chunks,&client,m,&query,&a.languages).await?
                    };Ok(serde_json::to_value(hits)?)
                },
                "compare"|"evaluate"=>{
                    if let Some(limit)=a.limit {cases.truncate(limit);}
                    let (run,path)=pipeline::evaluate_with_progress(&c,&chunks,&cases,operation=="evaluate",&client,a.verbose).await?;
                    Ok(json!({"run":run,"artifact_path":path}))
                },
                _=>anyhow::bail!("unknown operation {operation}"),
            }
        })
    }).map_err(failure)?;
    encode(&value)
}
fn read_run(text: &str) -> PyResult<pipeline::Run> {
    let run: pipeline::Run = parse(text)?;
    if !matches!(run.version, 1 | 2) {
        return Err(PyValueError::new_err("unsupported run version"));
    }
    Ok(run)
}
#[pyfunction]
fn summarize(py: Python<'_>, text: String) -> PyResult<String> {
    let run = read_run(&text)?;
    let summary = py.detach(move || report::summarize(&run));
    encode(&summary)
}
#[pyfunction]
fn write_report(py: Python<'_>, text: String, directory: String) -> PyResult<()> {
    let run = read_run(&text)?;
    py.detach(move || report::write(&run, Path::new(&directory)))
        .map_err(failure)
}

#[pyclass(frozen, name = "_BM25Index")]
struct PyBm25Index {
    index: SparseIndex,
    path: PathBuf,
}
#[pymethods]
impl PyBm25Index {
    #[staticmethod]
    fn build(py: Python<'_>, text: String, path: String) -> PyResult<Self> {
        let chunks: Vec<Chunk> = parse(&text)?;
        py.detach(move || {
            let path = sparse::build_at(Path::new(&path), &chunks)?;
            Ok(Self {
                index: SparseIndex::load(&path)?,
                path,
            })
        })
        .map_err(input_failure)
    }
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<Self> {
        py.detach(move || -> anyhow::Result<Self> {
            Ok(Self {
                index: SparseIndex::load(Path::new(&path))?,
                path: path.into(),
            })
        })
        .map_err(input_failure)
    }
    #[getter]
    fn path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }
    fn search(
        &self,
        py: Python<'_>,
        query: String,
        languages: Vec<String>,
        k1: f64,
        b: f64,
        top_k: usize,
    ) -> PyResult<String> {
        let result = py
            .detach(|| {
                self.index
                    .search(&query, &languages, &Bm25Config { k1, b, top_k })
            })
            .map_err(input_failure)?;
        encode(&result.value)
    }
}
#[pyfunction]
fn tokenize(py: Python<'_>, text: String) -> Vec<String> {
    py.detach(|| sparse::tokenize(&text))
}
#[pyfunction]
fn rrf(
    py: Python<'_>,
    dense: String,
    sparse: String,
    rank_constant: f64,
    top_k: usize,
) -> PyResult<String> {
    let dense: Vec<RankedChunk> = parse(&dense)?;
    let sparse: Vec<RankedChunk> = parse(&sparse)?;
    let result = py
        .detach(|| {
            rag_eval::fusion::fuse(
                &dense,
                &sparse,
                &RrfConfig {
                    rank_constant,
                    top_k,
                },
            )
        })
        .map_err(input_failure)?;
    encode(&result.value)
}
#[pyfunction]
fn retrieval_metrics(py: Python<'_>, hits: String, labels: String, k: usize) -> PyResult<String> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be positive"));
    }
    let hits: Vec<RankedChunk> = parse(&hits)?;
    let labels: Vec<RelevantPage> = parse(&labels)?;
    let mut pages = std::collections::HashSet::new();
    if labels
        .iter()
        .any(|l| !(1..=2).contains(&l.grade) || !pages.insert((&l.source, l.page)) || l.page == 0)
    {
        return Err(PyValueError::new_err(
            "duplicate or invalid relevance labels",
        ));
    }
    let scores = py.detach(|| rag_eval::metrics::retrieval(&hits, &labels, k));
    encode(&scores)
}
#[pyfunction]
fn score_judgment(
    py: Python<'_>,
    text: String,
    hits: String,
    n_facts: usize,
    unanswerable: bool,
) -> PyResult<String> {
    let j: rag_eval::metrics::Judgment = parse(&text)?;
    let hits: Vec<RankedChunk> = parse(&hits)?;
    let scores = py
        .detach(|| j.scores(&hits, n_facts, unanswerable))
        .map_err(input_failure)?;
    encode(&scores)
}
#[pyfunction]
fn run_cli(py: Python<'_>, arguments: Vec<String>) -> i32 {
    py.detach(move || {
        let result = runtime().and_then(|r| r.block_on(rag_eval::cli::run(arguments)));
        match result {
            Ok(code) => code,
            Err(e) => {
                eprintln!("Error: {e:#}");
                1
            }
        }
    })
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(config_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(config_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    m.add_function(wrap_pyfunction!(write_report, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(rrf, m)?)?;
    m.add_function(wrap_pyfunction!(retrieval_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(score_judgment, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_class::<PyBm25Index>()?;
    Ok(())
}
