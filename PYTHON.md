# olive-rag-eval

An importable Python API and command-line tool backed by the existing Rust RAG
evaluation engine. It evaluates dense retrieval, BM25, RRF, local cross-encoder,
Laya and LLM reranking, and local answer generation/judging. Odia and English
examples are included; the same schemas support other Indic languages.

## Installation

The package is **not yet published to PyPI**. For a built wheel:

```sh
python -m pip install /path/to/olive_rag_eval-0.1.0-cp310-abi3-PLATFORM.whl
uv pip install /path/to/olive_rag_eval-0.1.0-cp310-abi3-PLATFORM.whl
```

After a release is published, the intended commands are:

```sh
pip install olive-rag-eval
uv add olive-rag-eval
uv tool install olive-rag-eval
uvx olive-rag-eval --help
```

Import as `olive_rag_eval`. Prebuilt CPython ABI3 wheels target Python 3.10+ on
macOS ARM64/x86-64 (macOS 11+), Linux x86-64/ARM64 (manylinux2014/glibc 2.17+),
and Windows x86-64. These are CI build targets; only platforms actually exercised
by the release CI should be described as validated. PyPy, musl Linux and
free-threaded CPython are outside the first wheel matrix.

Wheels need no Rust compiler, Cargo, NumPy, PyTorch or system SQLite. Source
builds require Rust 1.98+ and a C compiler. The base package has no Python runtime
dependencies and never downloads or starts a real model automatically. Connect
to your separately managed local inference servers.

## Run a complete synthetic example

```python
from olive_rag_eval import Evaluator, init_example

config_path = init_example("olive-demo", hybrid=True)
evaluator = Evaluator.from_file(config_path)
print(evaluator.validate())
evaluator.index()

hits = evaluator.retrieve(
    "ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ କିଏ?",
    pipeline="hybrid-rrf",
    languages=["or"],
)
result = evaluator.evaluate(limit=6)
print(result.ok, result.failed_count)
print(result.summary())
print(result.write_report("olive-demo/reports"))
```

The example is self-contained and runs without sockets, model weights or the
source checkout. It uses the packaged stdlib-only synthetic worker over
stdin/stdout. Its scores are **not evidence of real model accuracy or Odia
quality**. The Odia examples and calibration labels remain drafts. Re-running
reuses completed stages. `init_example` refuses to overwrite any existing example
file and never writes inside the installed package.

`hybrid=True` compares dense-only, BM25-only, RRF and RRF plus listwise LLM
reranking. `hybrid=False` creates the original dense/MiniLM/BGE/Laya fixture.
Retrieval returns pre-reranking hits; `compare()` also runs reranking and
`evaluate()` additionally generates and judges answers.

The same shared Rust CLI is available through:

```sh
olive-rag-eval --config olive-demo/fixture.toml validate
olive-rag-eval --config olive-demo/fixture.toml index
olive-rag-eval --config olive-demo/fixture.toml evaluate --limit 6
python -m olive_rag_eval --help
```

No subprocess wrapping of an external `rag-eval` binary is used. CLI parsing and
execution are shared with the original Rust executable. Python library calls
return exceptions and results, rather than exiting the interpreter.

## Configuration and results

Use existing TOML files with `Evaluator.from_file(path)`. Paths resolve relative
to that TOML file. Alternatively, use ordinary dictionaries:

```python
config = evaluator.config  # independent copy with resolved paths
config["artifacts"] = "another-run"
other = Evaluator.from_dict(config, base_dir="/path/to/project")
other.compare(limit=2)
```

Without `base_dir`, dictionary paths resolve against the working directory at
construction. Configuration validation stays in Rust. The Python API supplies
`sys.executable` for fixture execution; it does not change global environment
variables. Explicit paths support spaces and non-ASCII characters.

`RunResult` contains:

- `run`: full structured run dictionary, including raw decisions and evidence.
- `artifact_path`: saved `run.json` path.
- `outcomes`, `failed_count`, `ok`: convenient status accessors.
- `summary()`: native metrics, language/pipeline breakdowns and timing summaries.
- `write_report(directory=None)`: Markdown, JSON and CSV; defaults beside the run.
- `RunResult.from_file(path)`: load saved version-1 or version-2 runs.

Incomplete runs retain their artifacts and completed stages:

```python
from olive_rag_eval import EvaluationError

try:
    result = evaluator.evaluate()
except EvaluationError as error:
    print(error.result.artifact_path)
    print(error.result.outcomes)

# Or inspect failures without raising:
result = evaluator.evaluate(raise_on_failure=False)
```

Malformed configuration/input raises `ValueError`; filesystem failures raise
`OSError`; other native execution failures raise `RuntimeError`.
Calls release the GIL during native work. They are synchronous and quiet by
default; use `verbose=True` for evaluation progress. In an asynchronous app:

```python
import asyncio
result = await asyncio.to_thread(evaluator.evaluate, limit=6)
```

Each call owns its runtime/resources. Concurrent independent runs should use
separate artifact directories; concurrent writes to the same snapshot/run are
not an advertised parallel-execution feature.

## BM25, RRF and scoring without a full pipeline

```python
from olive_rag_eval import BM25Index, retrieval_metrics, rrf, tokenize

chunks = [
    {"id": "a", "text": "ପୃଥିବୀ ଏକ ଗ୍ରହ।", "source": "odia", "page": 1, "language": "or"},
    {"id": "b", "text": "The Moon is a satellite.", "source": "english", "page": 1, "language": "en"},
]
index = BM25Index.build(chunks, "my-sparse-index")
hits = index.search("ପୃଥିବୀ", languages=["or"], top_k=5)
reloaded = BM25Index.load(index.path)
print(tokenize("ଓଡ଼ିଆ ପୃଥିବୀ"))

# Supply your own ordered dense hits, or use engine hits from retrieve().
fused = rrf(chunks, hits, rank_constant=60.0, top_k=5)
scores = retrieval_metrics(
    fused, [{"source": "odia", "page": 1, "grade": 2}], k=5
)
```

Chunk dictionaries use the existing schema: nonempty `id`, `text`, `source`,
one-based `page` and `language`; `source_url` and imported embeddings are optional.
Search/fusion return hit dictionaries with nested `chunk`, final `rank`, optional
dense/sparse/fused provenance and no invented dense scores. RRF accepts these
hits or plain chunks in rank order. Relevant pages have `source`, `page`, `grade`
(1 or 2). Unanswerable retrieval metrics return `None`.

`score_judgment(judgment, chunks, key_facts, unanswerable=False)` exposes native
generation scoring. Judgment fields are `claims` (text/supported),
`unsupported_claims`, `useful_chunk_ids`, `covered_fact_ids`,
`answer_relevance`, `abstained`, and `answer_language_correct`. Invalid evidence
is rejected; no LLM calls occur in arithmetic scoring helpers.

## Real model servers

Use the configuration schemas and endpoint contracts in [FRAMEWORK.md](FRAMEWORK.md).
EmbeddingGemma Q8_0 and chat LLMs run through local llama.cpp servers. The Python
embedding/cross-encoder and Laya server entry point is packaged as:

```sh
python -m olive_rag_eval.model_server --help
```

Install model dependencies separately, preferably in separate embedding/
cross-encoder and Laya environments. Importing this package never imports these
heavy dependencies. The package does not claim a validated Odia answer/judge
model; run real-model smoke checks and fluent-speaker calibration before making
quality claims.

## Build and release

```sh
uv venv
uv pip install 'maturin>=1.9,<2'
maturin build --release --locked --out dist
maturin sdist --out dist
uv pip install dist/olive_rag_eval-*.whl
```

The binding crate is a separate workspace member. `cargo test --locked` continues
to test the Rust core by default. Build source distributions from the repository
root; they include the core crate, binding crate, Python sources, packaged demo
assets and lockfile. Generated runs, caches, model weights and large source
snapshots are excluded.

The wheel workflow builds all five architecture targets and tests installations
outside the checkout with both pip and uv. Publishing is a separate, manually
requested action using a configured PyPI Trusted Publisher and a protected
`pypi` environment. No upload occurs during normal builds or this implementation.
