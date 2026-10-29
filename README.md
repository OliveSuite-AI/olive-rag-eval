# olive-rag-eval

**Evaluate local RAG pipelines from Python, with a Rust engine and Odia-first multilingual benchmarks.**

Compare dense retrieval, BM25, reciprocal rank fusion (RRF), cross-encoder,
Laya decision-model and LLM reranking. Then measure generated answers using a
separately configured LLM judge. Embedded **TurboVec and SQLite** provide local
indexing and persistence without a vector database service.

```text
Dense search ─┐
              ├─ RRF ─ Reranking ─ Answer generation ─ LLM judgment ─ Reports
BM25 search ─┘
```

Distribution: `olive-rag-eval` · Import: `olive_rag_eval` · CLI: `olive-rag-eval`

**Release status:** version 0.1.0 is prepared locally and has not been published
to PyPI. macOS ARM64 wheels have been tested with CPython 3.10 and 3.14. The
remaining platform builds are configured in CI and await verification. Synthetic
tests validate pipeline behavior; real-model Odia quality has not been validated.

## Why use it?

- **Compare complete pipelines.** Run dense-only, BM25-only, dense + BM25 + RRF,
  and RRF + LLM reranking against the same cases and context budget.
- **Evaluate local rankers.** Use cross-encoder scores, Laya relevance
  probabilities or listwise LLM ordering, with retained decision evidence.
- **Keep inference separate.** Connect to user-managed loopback model servers.
  Importing the package does not load PyTorch or download weights.
- **Inspect multilingual results.** Track query language, script, answer language
  and corpus language scope. Draft examples include Odia, English, Romanized Odia
  and code-switched queries.
- **Resume and audit runs.** Reuse completed stages with fingerprinted caches;
  retain raw responses, failures, timings and model identities.
- **Use native Python objects.** Receive dictionaries, lists and `Path` values.
  Native operations release the GIL; the API does not shell out to a Rust binary.

## Installation

Requires **CPython 3.10+**. A matching wheel needs no Rust compiler, system SQLite
or Python ML dependencies. Install the local wheel from the `rag-eval-rs` directory:

```sh
python -m pip install dist/olive_rag_eval-0.1.0-cp310-abi3-macosx_11_0_arm64.whl
```

With uv:

```sh
uv venv
uv pip install dist/olive_rag_eval-0.1.0-cp310-abi3-macosx_11_0_arm64.whl
```

Activate your environment before running the examples. The wheel above is for
Apple Silicon macOS. Use a matching wheel for other systems once built.

After publication, the intended registry installation commands are
`python -m pip install olive-rag-eval` and `uv add olive-rag-eval`. They are not
the current installation path for this unreleased package.

| Platform | Wheel target | Verification |
|---|---|---|
| macOS 11+ | ARM64 | Locally tested on CPython 3.10 and 3.14 |
| macOS 11+ | x86_64 | CI configured; not yet verified |
| Linux, glibc 2.17+ | x86_64, ARM64 | manylinux2014 CI configured; not yet verified |
| Windows | x86_64 | CI configured; not yet verified |

The first wheel matrix does not cover PyPy, musl Linux or free-threaded CPython.
Source builds require Rust 1.98+ and a C compiler; see [Building from source](#building-from-source).

## Quick start

This complete example uses the bundled synthetic worker. It requires no model
downloads, API keys, network access or listening sockets after installation.
Run it from a writable directory:

```python
from olive_rag_eval import Evaluator, init_example

config_path = init_example("olive-demo", hybrid=True)
evaluator = Evaluator.from_file(config_path)
evaluator.validate()
evaluator.index()

hits = evaluator.retrieve(
    "ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ କିଏ?",
    pipeline="hybrid-rrf",
    languages=["or"],
)
print([hit["chunk"]["id"] for hit in hits])

result = evaluator.evaluate(limit=6)
print("Successful:", result.ok)
print("Failed outcomes:", result.failed_count)
print("Saved run:", result.artifact_path)
print("Reports:", result.write_report())
```

`init_example()` copies fixtures into your directory and refuses to overwrite
existing files. To resume, load `olive-demo/fixture.toml` directly instead of
calling `init_example()` again. To run all 60 draft query instances across the
four hybrid pipelines, omit `limit`.

The example uses synthetic embeddings, rankings, answers and judgments. Its
scores measure test behavior, **not real model accuracy, fluency or latency**.
Runs and reports carry `fixture=true`. Use `hybrid=False` for the dense fixture
with synthetic cross-encoder and Laya comparisons.

## Choose the evaluation stage

| Python call | Work performed |
|---|---|
| `validate()` | Validate configuration, corpus and evaluation cases |
| `validate(online=True)` | Also check configured model endpoints |
| `index()` | Build or reuse required dense and sparse indexes |
| `retrieve(query, pipeline=..., languages=...)` | Return retrieval/fusion candidates before reranking |
| `compare(limit=...)` | Evaluate retrieval and configured reranking |
| `evaluate(limit=...)` | Also generate and judge answers |

`limit` counts query instances, not total pipeline outcomes. `retrieve()` does
not invoke the reranker; inspect `compare()` or `evaluate()` results for the
final ordering. Calls are synchronous and quiet by default; pass `verbose=True`
to `compare()` or `evaluate()` for progress.

### Command-line usage

After creating the example above:

```sh
olive-rag-eval --config olive-demo/fixture.toml validate
olive-rag-eval --config olive-demo/fixture.toml index
olive-rag-eval --config olive-demo/fixture.toml compare --limit 6
olive-rag-eval --config olive-demo/fixture.toml evaluate --limit 6
python -m olive_rag_eval --help
```

Regenerate reports from an existing run without model calls:

```sh
olive-rag-eval report /path/to/run.json --output ./reports
```

## Connect real local models

Use the repository's `configs/local.toml` or `configs/hybrid-local.toml` as a
starting point. Replace all model revision/checksum placeholders, set your corpus
and dataset paths, and start the required services. Placeholders deliberately
fail validation. The detailed [model and framework guide](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/FRAMEWORK.md#configure-real-local-models)
contains setup commands, pinned dependency files and endpoint contracts.

| Role | Supported integration |
|---|---|
| Dense embeddings | EmbeddingGemma-300M **Q8_0** through llama.cpp; Python MiniLM baseline |
| Cross-encoder reranking | Local Python service, including BGE reranker profiles |
| Laya decisions | Local Laya service; ranking uses the `relevant` probability |
| LLM reranking | Local chat server; validated permutation of candidate IDs |
| Answer generation | Separately configured local chat model |
| Answer evaluation | Separately configured local LLM judge |

Model dependencies and weights are installed separately. The optional packaged
Python server entry point is `python -m olive_rag_eval.model_server --help`.
EmbeddingGemma Q8_0 quantizes model weights; TurboVec's `index_bits` separately
controls vector compression.

Chat stages require `/v1/chat/completions`, `/apply-template` and `/tokenize`;
chat compatibility alone is insufficient for context-budget checks. The judge
and LLM reranker also require JSON-mode output. Endpoints are restricted to
loopback. Pin model artifacts and runtime identities to keep cached results
meaningful. Use distinct answer and judge models when your evaluation design
requires independence, and calibrate the judge against reviewed human labels.

Load your completed configuration and check the endpoints before indexing:

```python
from olive_rag_eval import Evaluator

evaluator = Evaluator.from_file("configs/my-local.toml")
evaluator.validate(online=True)
evaluator.index()
result = evaluator.evaluate(limit=6)
result.write_report()
```

TOML paths resolve relative to the configuration file. For in-memory settings,
use `Evaluator.from_dict(config, base_dir=...)`; `evaluator.config` returns an
independent copy. Dictionary paths default to the working directory at construction.

## Datasets and multilingual evaluation

Inputs are UTF-8 JSONL. Each corpus chunk has a stable ID, text, source,
one-based page number and language:

```json
{"id":"earth-or-1","text":"ପୃଥିବୀ ସୂର୍ଯ୍ୟଠାରୁ ତୃତୀୟ ଗ୍ରହ।","source":"earth-or","page":1,"language":"or"}
```

Cases define the query, reference answer, key facts, relevant source/page labels,
language scope, script and review status. The full [dataset contracts](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/FRAMEWORK.md#corpus-and-benchmark-contracts)
include complete case examples and calibration requirements.

Use language tags such as `or`, `en`, `hi` and `bn`, and script tags such as
`Orya`, `Latn` or `mixed`. The schema accepts other languages; bundled examples
currently focus on Odia and English. Draft translations and calibration examples
need fluent-speaker review before substantive comparisons.

BM25 uses NFC normalization and Unicode word boundaries, preserving Odia marks
and conjuncts. It applies no implicit translation, transliteration, stemming or
digit folding. Cross-language lexical search may return no matches. Language
scope is applied before scoring and Top-K selection, including BM25 corpus statistics.

## Metrics, artifacts and failures

Retrieval reports include Recall, MRR and nDCG, with dense/BM25/RRF stage
diagnostics. Answer scoring covers faithfulness, hallucination indicators,
relevance, key-fact correctness, abstention and answer-language compliance.
The judge supplies structured evidence; Rust validates it and computes scores.
Invalid judgments receive no score after one failed repair attempt.

Each run saves `run.json`; `write_report()` produces `report.md`, `summary.json`
and `results.csv`. Reports group results by pipeline and language/script/review
status. Saved evidence supports inspection of individual failures and decisions.
Re-running the same configuration reuses completed stages.

```python
from olive_rag_eval import EvaluationError, RunResult

try:
    result = evaluator.evaluate()
except EvaluationError as error:
    # Successful work and failed outcomes remain available on disk.
    result = error.result
    print(result.failed_count, result.artifact_path)

saved = RunResult.from_file(result.artifact_path)
print(saved.summary())
saved.write_report("reports")
```

Use `raise_on_failure=False` to inspect incomplete runs without raising.
Malformed configuration raises `ValueError`, filesystem errors raise `OSError`,
and other native execution errors raise `RuntimeError`.

Native calls release the GIL. In async applications use
`await asyncio.to_thread(evaluator.evaluate, limit=6)`. Give concurrent independent
runs separate artifact directories; concurrent writes to the same run are not
an advertised execution mode. Raw artifacts contain corpus text and model responses;
store them according to your data handling requirements.

## Use BM25 and RRF independently

```python
from olive_rag_eval import BM25Index, retrieval_metrics, rrf

chunks = [
    {"id": "or-1", "text": "ପୃଥିବୀ ଏକ ଗ୍ରହ।", "source": "earth-or", "page": 1, "language": "or"},
    {"id": "en-1", "text": "The Moon is a satellite.", "source": "moon-en", "page": 1, "language": "en"},
]
sparse = BM25Index.build(chunks, "sparse-demo")
sparse_hits = sparse.search("ପୃଥିବୀ", languages=["or"], top_k=5)

# Replace this illustrative ordered list with your dense retriever's results.
dense_hits = [chunks[0]]
fused = rrf(dense_hits, sparse_hits, rank_constant=60.0, top_k=5)
print(retrieval_metrics(fused, [{"source": "earth-or", "page": 1, "grade": 2}], k=5))
```

Indexes are immutable; reload with `BM25Index.load(sparse.path)`. Use another
directory when the corpus changes. RRF accepts engine hits or plain chunks in
rank order and merges by chunk ID. Sparse-only hits retain null dense provenance.
`tokenize()` and `score_judgment()` are also public native helpers; see the
[Python API guide](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/PYTHON.md#bm25-rrf-and-scoring-without-a-full-pipeline).

## Troubleshooting

| Symptom | Action |
|---|---|
| Wheel is incompatible | Match OS, architecture and CPython version; use a normal GIL-enabled build |
| `init_example` raises `FileExistsError` | Load the existing `fixture.toml` or choose a new directory |
| Configuration rejects a model profile | Replace revision/checksum placeholders and verify endpoint contracts |
| LLM context budget is exceeded | Adjust candidate count or context budget explicitly; candidates are not silently dropped |
| `EvaluationError` is raised | Inspect `error.result.outcomes` and saved evidence, fix the cause, then rerun |
| Odia BM25 returns no hits | Check corpus language scope and exact script/word overlap; translation is not automatic |

## Building from source

From the `rag-eval-rs` directory with Rust 1.98+ and a C compiler installed:

```sh
uv venv
uv pip install 'maturin>=1.9,<2'
uv run maturin build --release --locked --out dist
uv run maturin sdist --out dist
```

Install the wheel matching your platform from `dist/`. Wheels bundle the native
extension and SQLite, type information and small demo assets. They exclude
model weights, generated runs, caches and large source snapshots.

The release workflow builds five platform/architecture targets and tests clean
pip/uv installations outside the checkout. Publishing requires a separate manual
action with a protected PyPI environment; ordinary builds do not upload packages.
See [Python build and release details](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/PYTHON.md#build-and-release).

## Validation and license

Local verification includes 30 Rust tests, four runtime tests, six installed SDK
tests, a source-archive rebuild, and 240 successful synthetic hybrid outcomes
through the installed Python API. See [validation evidence](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/VALIDATION.md) for
scope and limitations. Real inference and fluent-speaker judge calibration are
required before reporting model quality.

The framework is MIT licensed; see [LICENSE](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/LICENSE). Bundled draft factual
examples and model-license considerations are described in [NOTICE](https://github.com/OliveSuite-AI/olive-rag-eval/blob/main/NOTICE).
External model weights retain their publishers' licenses and access terms.
