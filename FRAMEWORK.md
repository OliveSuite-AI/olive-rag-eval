# Odia-first local RAG evaluation

Python package: **`olive-rag-eval`**, imported as `olive_rag_eval`. See
[Python installation and API](PYTHON.md) for wheel installation, pip/uv usage,
the packaged hybrid demo and the native API. The package is prepared for release;
it has not yet been uploaded to PyPI.

A Rust evaluation framework for Odia/English retrieval, local cross-encoder and
Laya or LLM reranking, hybrid dense/BM25 retrieval, and local LLM answer evaluation. Other Indic languages use the
same JSONL contracts and language tags.

The Python reference folder remains unchanged. This project replaces Qdrant with
embedded TurboVec and SQLite; it does not require a vector database service.

## Run the offline example

Requirements: a recent Rust toolchain (tested with 1.98.1) and Python 3.10+.
No ML packages, model downloads, API credentials or listening sockets are needed
for the synthetic example. Cargo needs network access for its initial dependency
download; subsequent commands can use `--offline`.

```sh
cd rag-eval-rs
cargo build --locked
target/debug/rag-eval --config configs/fixture.toml validate
target/debug/rag-eval --config configs/fixture.toml index
target/debug/rag-eval --config configs/fixture.toml compare
target/debug/rag-eval --config configs/fixture.toml evaluate
```

`fixture.toml` runs a persistent Python fixture over stdin/stdout. Its hash-based
embeddings, word-overlap reranking and canned answer/judge logic exercise the
actual Rust pipeline, persistence and reports. **Its scores say nothing about
real model accuracy, Odia fluency or model latency.** Every run/report is marked
`fixture=true`, and fixture transports are rejected in real-model mode.

`evaluate` prints the path to `artifacts/fixture/runs/<fingerprint>/run.json`.
The adjacent `reports/` directory contains `report.md`, `summary.json` and
`results.csv`. Re-running reuses completed stages. Regenerate reports without
model calls:

```sh
target/debug/rag-eval report artifacts/fixture/runs/RUN_FINGERPRINT/run.json
```

Other commands:

```sh
target/debug/rag-eval --config configs/fixture.toml retrieve \
  'ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ କିଏ?' --languages or
target/debug/rag-eval --config configs/fixture.toml evaluate --limit 6
target/debug/rag-eval --help
```

Paths inside TOML resolve relative to the TOML file, not the shell directory.

## Hybrid retrieval: BM25 + dense → RRF → LLM reranking

The hybrid preset compares four pipelines against the same corpus, answer LLM,
judge, query language scopes and Top-5 context budget:

| Pipeline | Retrieval and ranking |
|---|---|
| `dense` | TurboVec Top 20 |
| `bm25` | Native Rust BM25 Top 20 |
| `hybrid-rrf` | Dense Top 20 + BM25 Top 20 → RRF Top 20 |
| `hybrid-llm` | Dense + BM25 → RRF Top 20 → local LLM ordering |

All pipelines then generate and judge an answer in `evaluate` mode. `compare`
stops after retrieval/reranking metrics. The existing dense configurations retain
their current behavior unless an explicit `[[pipelines]]` list is supplied.

Run the socket-free synthetic preset:

```sh
target/debug/rag-eval --config configs/hybrid-fixture.toml validate
target/debug/rag-eval --config configs/hybrid-fixture.toml index
target/debug/rag-eval --config configs/hybrid-fixture.toml evaluate
target/debug/rag-eval --config configs/hybrid-fixture.toml retrieve \
  'ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ କିଏ?' \
  --pipeline hybrid-rrf --languages or
```

`retrieve --pipeline` shows pre-reranking retrieval/fusion candidates. It does
not invoke the configured reranker. Use `compare` or `evaluate` to inspect the
final LLM ranking in `run.json`. `--pipeline` and `--embedding` are mutually
exclusive on the retrieval command.

For real models, copy `configs/hybrid-local.toml`, fill pinned artifacts, and
start the embedding, answer and judge servers as described below. Start another
local chat server on port **18087** for `[llm_reranker]`, or point that role at an
existing answer server. The profile has its own name, model, revision, artifact
checksum, endpoint and context/output budgets. Sharing a server/model is allowed;
reranking, generation and judging remain separate stages. The new role requires
the same `/apply-template`, `/tokenize` and JSON-mode chat routes as the judge.

Explicit pipeline example:

```toml
[bm25]
k1 = 1.2
b = 0.75
top_k = 20

[rrf]
rank_constant = 60.0
top_k = 20

[[pipelines]]
name = "hybrid-llm"
retrieval = "hybrid_rrf"
embedding = "embeddinggemma-q8"
reranker = "llm-reranker-local"
```

Names reference declared model profiles. Retrieval modes are `dense`, `bm25` and
`hybrid_rrf`. Omitting `reranker` preserves retrieval order. Existing HTTP
cross-encoder and Laya profiles can also be referenced. A BM25-only pipeline has
no `embedding` field and runs once per query, regardless of how many embedding
profiles are declared. A sparse-only configuration may use `embeddings = []`.
`index` builds only the dense profiles actually used by named pipelines.

BM25 uses a separate immutable SQLite inverted-index snapshot shared across
embedding profiles. It stores term frequencies and document lengths, and uses
`ln(1 + (N-df+0.5)/(df+0.5))` IDF with standard BM25 saturation. Corpus-language
filtering happens before scoring: `N`, document frequencies and average length
all refer to the allowed language scope. Query terms are deduplicated.

Tokenization is NFC plus Unicode UAX #29 word boundaries and lowercase, with the
`unicode-segmentation` version pinned in both Cargo and the sparse fingerprint.
It preserves Odia combining marks and conjuncts, applies no stop-word removal or
stemming, and does not equate Odia digits with ASCII digits or Romanized Odia
with Odia script. Lexical search may return no hits for cross-language queries;
the report exposes that instead of translating the query implicitly.

RRF combines ranks, not incomparable dense/BM25 scores:
`score(d) = sum(1 / (rank_constant + one_based_rank(d)))`. Each branch has equal
weight. Hits are merged by stable chunk ID; different chunks on the same page
remain distinct. Equal fused scores sort by chunk ID. An empty BM25 result is
valid and leaves dense hits eligible. A missing/corrupt index or failed search
fails the hybrid stage; no fallback pipeline is silently substituted.

The reranking LLM sees only the query and ID-tagged passage texts in fused order,
without retrieval scores or reference answers. It must return an exact ID
permutation as `{"ordered_ids":[...]}`. Unknown, missing, extra or duplicate IDs
trigger one repair attempt; failed outputs are retained as evidence and receive
no answer score. Zero/one-candidate lists skip inference. Context overflow fails
explicitly; this stage never silently shortens passages or drops candidates.
The final ranking is ordinal and has no invented relevance logits.

Version-2 `run.json` includes pipeline identity, retrieval mode, dense/BM25/RRF
lists, nullable dense/sparse scores and ranks, fused scores, per-stage retrieval
metrics, branch overlap, sparse zero-hit indicators and relevant pages added/lost
by fusion relative to the dense candidate list. Overlap uses the two branch lists
before fusion truncation. A BM25-only hit has `vector_rank = null` and `vector_score = null`.
Reports group by named pipeline and retain the existing language/script/review
breakdowns. Both the original version-1 run format and dense snapshots remain
readable. Existing Rust consumers should handle optional dense provenance.

Search, BM25, RRF and LLM reranking have independent cache keys and timings.
Changing RRF parameters reuses both searches; changing the LLM reranker reuses
fusion; changing BM25 scoring parameters reuses the inverted index and dense
search. The default runner executes stages sequentially, so latency evidence
does not imply concurrent dense/sparse execution.

## Configure real local models

Copy `configs/local.toml` and fill the model revisions and artifact SHA256 values.
Placeholders deliberately fail validation. Use immutable Hugging Face commits,
GGUF checksums and pinned local server builds. Never reuse a profile after
changing its weights, tokenizer, prompt/pooling policy or inference runtime:
update its revision to incorporate the runtime/build identity and re-index when
the embedding profile changes. Full model settings are part of cache keys.

The default matrix is:

| Role | Model/runtime | Port |
|---|---|---:|
| Multilingual embeddings | EmbeddingGemma-300M **Q8_0**, llama.cpp | 18081 |
| English embedding baseline | all-MiniLM-L6-v2, Python | 18082 |
| Multilingual cross-encoder | BAAI/bge-reranker-v2-m3, Python | 18083 |
| Typed relevance decisions | convaiinnovations/laya-multilingual, Python | 18084 |
| Answer LLM | Your Odia-capable local model | 18085 |
| Judge LLM | Your independently validated local model | 18086 |

The model servers are user-managed. No large models are downloaded or started by
the Rust CLI. Use explicit setup to download pinned snapshots, then serve offline.
Model licenses and access terms come from their publishers.
For retrieval-only use, remove the optional `[answer]` and `[judge]` sections and
`calibration` setting from your copied configuration.

### EmbeddingGemma Q8_0

Download a pinned revision of
[`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF)
and serve the Q8_0 GGUF using a pinned llama.cpp build:

```sh
llama-server --model /path/to/embeddinggemma-Q8_0.gguf \
  --alias embeddinggemma-300m-q8 --embedding \
  --host 127.0.0.1 --port 18081 --ctx-size 2048
python3 scripts/model_manifest.py /path/to/embeddinggemma-Q8_0.gguf \
  --model embeddinggemma-300m-q8 --revision YOUR_PINNED_ARTIFACT_AND_RUNTIME_REVISION
```

The Rust adapter supplies `task: search result | query: ` for queries and
`title: none | text: ` for untitled documents, once each. Input corpus text must
not already contain these prefixes. It uses full 768-dimensional output and L2
normalization. Do not point it at an endpoint that also adds task prefixes.
Q8_0 is **model-weight** quantization. TurboVec's separate `index_bits=4` controls
vector compression. MiniLM and EmbeddingGemma never share an index.

### Python embedding and cross-encoder services

Use a separate environment from Laya, with Python 3.10–3.12 supported by the
pinned PyTorch version:

```sh
python3.12 -m venv .venv-models
.venv-models/bin/python -m pip install -r services/requirements-models.txt
.venv-models/bin/python -m pip freeze > artifacts/models-environment.lock.txt
python3 scripts/model_manifest.py /path/to/minilm-snapshot \
  --model sentence-transformers/all-MiniLM-L6-v2 --revision HF_COMMIT_AND_RUNTIME_REVISION
.venv-models/bin/python services/runtime.py --kind embedding --port 18082 \
  --model-dir /path/to/minilm-snapshot \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --revision HF_COMMIT_AND_RUNTIME_REVISION --artifact-sha256 MODEL_TREE_SHA256
.venv-models/bin/python services/runtime.py --kind cross --port 18083 \
  --model-dir /path/to/bge-snapshot --model BAAI/bge-reranker-v2-m3 \
  --revision HF_COMMIT_AND_RUNTIME_REVISION --artifact-sha256 MODEL_TREE_SHA256 \
  --max-tokens 512 --batch-size 16
```

Run each server in its own terminal. The model-directory digest hashes sorted
relative filenames and bytes; it is not the checksum of only the weights file.
The service verifies it at startup. Use `--device mps` or `--device cuda` where
supported; CPU is the default. Context length, device, versions and batching are
returned by `/health` and retained in stage evidence.

The same cross-encoder server supports the English reference model
`cross-encoder/ms-marco-MiniLM-L-6-v2` as an additional TOML reranker profile.

### Laya relevance server

```sh
python3.12 -m venv .venv-laya
.venv-laya/bin/python -m pip install -r services/requirements-laya.txt
.venv-laya/bin/python -m pip freeze > artifacts/laya-environment.lock.txt
.venv-laya/bin/python services/runtime.py --kind laya --port 18084 \
  --model-dir /path/to/laya-multilingual-snapshot \
  --model convaiinnovations/laya-multilingual \
  --revision HF_COMMIT_AND_RUNTIME_REVISION --artifact-sha256 MODEL_TREE_SHA256 \
  --max-tokens 1024
```

This uses the upstream Python `laya` runtime, not a Core ML export. It explicitly
loads the selected checkpoint without language routing. Each query/passage pair
gets a binary typed choice; ranking uses the `relevant` option probability,
**not** the winning-option confidence. Raw decisions are retained. Laya's
capacity check includes its typed question/options, not just the passage. The
adapter targets the pinned Laya API, including its question budget helpers.

### Answer and judge LLMs

Start local llama.cpp chat servers with `--alias` matching each TOML `model`,
on ports 18085/18086, with context sizes matching the configured budgets. Choose
models based on measured Odia competence; no particular LLM is assumed.

The chat interface uses `/v1/chat/completions`. It also requires llama.cpp's
`/apply-template` and `/tokenize` routes to count the rendered chat template
before inference. Other OpenAI-compatible servers need equivalent routes or an
adapter; chat compatibility alone is insufficient for the capacity check.
The judge must accept `response_format={"type":"json_object"}`. Reserve enough
output tokens for claim-level JSON. Temperature defaults to zero.

```sh
target/debug/rag-eval --config configs/my-local.toml validate --online
target/debug/rag-eval --config configs/my-local.toml index
target/debug/rag-eval --config configs/my-local.toml evaluate --limit 6
python3 scripts/smoke_real.py --config configs/my-local.toml
```

Endpoints are restricted to loopback. An optional `api_key_env` names a local
server authentication environment variable; secrets are never copied into run
manifests. Python server health checks verify model/revision/checksum. llama.cpp
health may not expose artifact identity, so its operator must keep the configured
GGUF checksum and alias aligned with the running server.

## Corpus and benchmark contracts

Input is UTF-8 JSONL. Chunk example:

```json
{"id":"earth-or-1","text":"ପୃଥିବୀ ସୂର୍ଯ୍ୟଠାରୁ ତୃତୀୟ ଗ୍ରହ।","source":"earth-or","page":1,"language":"or","source_url":"https://science.nasa.gov/earth/facts/"}
```

`page` is one-based. Starter pages are authored corpus sections, not HTML/PDF
page numbers. Document/source identities must differ across translated versions.
Text is preserved; NFC working copies are used for inference. No transliteration,
translation, digit folding or removal of Odia marks/joiners happens implicitly.

Optional chunk `embedding` arrays must include `embedding_profile`, the full
profile fingerprint printed by `validate`. Imported vectors must be finite,
nonzero and dimension-compatible; the framework normalizes them.

Evaluation case example:

```json
{"id":"q1","semantic_case_id":"fact1","query":"ପୃଥିବୀ ସୂର୍ଯ୍ୟଠାରୁ କେତେ ନମ୍ବର ଗ୍ରହ?","query_language":"or","query_script":"Orya","answer_language":"or","corpus_languages":["en"],"question_type":"direct","review_status":"draft","expected_answer":"ପୃଥିବୀ ତୃତୀୟ ଗ୍ରହ।","key_facts":["ପୃଥିବୀ ତୃତୀୟ ଗ୍ରହ।"],"relevant_pages":[{"source":"earth-en","page":1,"grade":2}],"unanswerable":false}
```

Use BCP-47 language tags (`or`, `en`, later `hi`, `bn`, etc.) and script tags
(`Orya`, `Latn`, or `mixed`). `corpus_languages` filters **inside TurboVec** before
Top-K selection. A missing relevant page in that scope is a validation error.
Grade 2 means answer-bearing; grade 1 means partially useful. Repeated relevant
pages receive credit once while keeping their original positions. Unanswerable
cases require empty relevance labels and are excluded from retrieval means.

The starter contains 10 semantic cases and 60 query instances:

- 40 instances across Odia→Odia, English→English, Odia→English and English→Odia.
- 10 Odia questions against a mixed Odia/English corpus.
- 10 Romanized Odia or code-switched variants.

All are **draft**, including the Odia translations. NASA source HTML/text
snapshots and SHA256 provenance are in `data/sources/` and
`data/starter_manifest.json`. The small astronomy suite is a wiring/initial
quality check, not evidence of broad Indic capability. Number/script handling
also has deterministic Unicode tests. Extend the corpus with harder distractors,
native-authored questions and spelling/numeral variants for substantive use.

`data/reference_cases.jsonl` preserves the reference's 43 English cases. Its
payments corpus is absent from the original project and must be supplied with
matching source/page identities. There is no runtime dependency on the reference
Python code. `scripts/build_starter.py` regenerates the draft starter and converts
the reference when that folder is available.

## Answer scoring, review and calibration

The answer model sees only the query and selected chunks, with requested output
language and citation instructions. The judge sees that exact context, answer,
reference answer and key facts. Cross-language meaning is judged directly;
automatic English translation is not used as a surrogate evaluation.

The judge returns claims with supported flags, unsupported claim text, useful
chunk IDs, covered 1-based fact IDs, relevance (0/0.5/1), abstention and language
compliance. Rust rejects invalid IDs, duplicates and inconsistent evidence.
One repair attempt is allowed; failed judgments receive no scores. Raw successful
responses and both failed repair responses are saved in SQLite stage evidence.

Scores are arithmetic: faithfulness, per-answer hallucination indicator, answer
relevance, context relevance, key-fact correctness, abstention accuracy and
language compliance. A refusal has faithfulness 1 by convention but can still
fail correctness/abstention. Language compliance is judge-assessed, not a claim
that every Odia-script string is fluent Odia. Citations are prompted and retained;
citation entailment is not a separate automated metric in v1.

`calibration` optionally names a JSONL file of **fixed context/answer examples**.
This avoids incorrectly attaching a human judgment to a different generated
answer or reranker's context. `data/calibration_draft.jsonl` includes grounded,
invented and incorrect-refusal Odia examples. No human scores are fabricated.

A calibration row contains `case` (the case schema), `context` (ordered chunks),
`answer`, `label_review_status` and `expected_judgment` (the judge JSON schema).
A fluent reviewer must populate the expected judgment and set label status to
`reviewed`. Only reviewed labels contribute to judge-vs-human mean absolute
error; absent labels produce a null aggregate, with counts shown. Review case
wording, source support, cross-language equivalence and key facts before changing
case `review_status` to `reviewed`. Keep human review provenance with your dataset.

## Reproducibility and reporting

- One candidate list per query/embedding/repeat is reused across rerankers.
- Default: Top 20 candidates, Top 5 context, sequential inference, one repeat.
- Embedding, search, rerank, answer and judge stages are saved transactionally.
- Changing the judge reuses answers; changing the answer model reuses retrieval.
  Identical answer/context inputs can share a judgment across configurations.
- Failed stages remain failed and retry on the next run. Independent comparisons
  continue; incomplete runs return a nonzero exit status.
- Model responses, raw decisions, original/reranked positions, prompts' version
  IDs, parameters, provenance and available provider usage are retained in
  `run.json` and `stages.sqlite`. Token counts are never estimated as provider usage.
- Reports separate language directions, scripts, question types and review status;
  show missing/failure counts, per-query means and equal-weight track means.
- Bootstrap intervals resample semantic cases, keeping translations, robustness
  variants and repeated runs together. Small sample intervals remain uncertain.
- Timings separate warm-up and measured requests. Cached stage measurements are
  deduplicated by key. TurboVec prepares its layout before search timing; corpus
  index-build evidence remains in the snapshot manifest.

## Tests

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python3 -m unittest discover -s services -p 'test_*.py' -v
```

The default integration tests use the stdio fixture and exercise the Rust client
contracts, real TurboVec/SQLite, full answer evaluation and resume invalidation.
Run the socket-based HTTP contract test on a machine that permits loopback binds:

```sh
cargo test --locked --test workflow -- --ignored
```

Real-model validation is explicitly opt-in through `scripts/smoke_real.py`.
HTTP transport and real model quality must be verified before claiming a
production-ready or human-validated Odia benchmark.
