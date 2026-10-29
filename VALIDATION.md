# Implementation validation — 2026-10-30

Verified with Rust 1.98.1 on Apple Silicon macOS.

| Check | Result |
|---|---|
| `cargo fmt --check` | Passed |
| `cargo clippy --locked --all-targets -- -D warnings` | Passed |
| `cargo test --locked` | 30 passed; 1 explicitly ignored HTTP test |
| `python3 -m unittest discover -s services -p 'test_*.py' -v` | 4 passed |
| Full synthetic CLI evaluation | 360/360 comparisons completed |
| Full hybrid CLI evaluation | 240/240 comparisons completed |
| Version-1 saved-run report regeneration | Passed with the version-2 executable |
| Fixed-context calibration protocol | 3/3 examples completed; no reviewed human labels |
| Original reference repository | No working-tree changes |

Python packaging validation:

- Built the `olive-rag-eval` 0.1.0 release wheel with PyO3's CPython 3.10+
  stable ABI and Maturin 1.16.0, plus a source distribution.
- Six installed-package tests passed on CPython 3.10.18 and 3.14.4 on macOS
  ARM64. Fresh `pip` and `uv` environments were tested outside the checkout,
  including CLI execution without Cargo on `PATH` and Unicode filesystem paths.
- Rebuilt and tested a wheel from the extracted source distribution with a
  fresh Cargo target directory. Checked archive contents, metadata and bundled
  fixture asset consistency. Installed the console command with `uv tool`.
- The installed Python API completed all 240 synthetic hybrid comparisons with
  zero failed outcomes and generated reports. Evidence is in
  [the packaged-run summary](artifacts/python-package-smoke.json).
- The workflow prepares macOS ARM64/x86_64, Linux ARM64/x86_64 and Windows
  x86_64 wheels. Only macOS ARM64 was executed locally; other targets require CI.
- Nothing was published to PyPI. See [PYTHON.md](PYTHON.md) for local wheel
  installation, API usage, optional model serving and release instructions.

Cargo checks used `--offline` and an isolated dependency cache after downloading
the lockfile's dependencies. The executable fixture uses stdin/stdout because
this execution sandbox refuses loopback socket binds (`Operation not permitted`).
The HTTP integration test is retained as an explicit opt-in test, not reported
as passed. Real model inference was not validated in this session.

The full run used 60 draft query instances × 2 embedding fixtures × 3 ranking
configurations, plus 3 fixed-answer judge examples. It exercises real Rust
orchestration, TurboVec indexing/search, SQLite persistence, scoring and reports.
Synthetic inference outputs are not evidence of EmbeddingGemma, BGE, Laya or an
LLM's Odia quality. No provider token usage was available for this fixture.

The hybrid run used the same 60 query instances across dense-only, BM25-only,
RRF and RRF-plus-LLM pipelines, plus 3 fixed-answer judge examples. It verified
nullable dense provenance on sparse-only hits, exact listwise ranking IDs,
separate BM25/RRF/LLM timing and per-stage retrieval metrics. The test suite also
covers scoped BM25 arithmetic, Unicode/Odia tokenization, fusion ties and
gain/loss diagnostics, independent cache invalidation, malformed ranking repair,
branch failure without fallback, and sparse-only operation without embeddings.

Local session artifacts:

- [Full fixture report](artifacts/fixture/runs/63c6f39d806d2f81d4b7f03232c2e06719498e2541ee56478c3ca5b3a2812784/reports/report.md)
- [Machine-readable summary](artifacts/fixture/runs/63c6f39d806d2f81d4b7f03232c2e06719498e2541ee56478c3ca5b3a2812784/reports/summary.json)
- [Raw run and model-response evidence](artifacts/fixture/runs/63c6f39d806d2f81d4b7f03232c2e06719498e2541ee56478c3ca5b3a2812784/run.json)
- [Hybrid fixture report](artifacts/hybrid-fixture/runs/f929686e197e879387b2d10a855dbde3a47bce99f8517fff4b136f417758c1b9/reports/report.md)
- [Hybrid raw run](artifacts/hybrid-fixture/runs/f929686e197e879387b2d10a855dbde3a47bce99f8517fff4b136f417758c1b9/run.json)

Before substantive model comparisons, fill `configs/local.toml`, start the local
services, run `scripts/smoke_real.py`, review the Odia corpus/questions, and add
fluent-speaker labels to the fixed-context calibration examples. Setup and
contracts are in [README.md](README.md).
