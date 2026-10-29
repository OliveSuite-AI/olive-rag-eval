"""Python-facing API. All scoring, retrieval and evaluation run in Rust."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from . import _native

JSON = dict[str, Any]
PathLike = str | os.PathLike[str]


def _json(value: Any) -> str:
    def path(value: Any) -> str:
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=path)


def _positive(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _hits(records: Sequence[Mapping[str, Any]]) -> list[JSON]:
    """Accept engine hits or plain chunk dictionaries, without fake dense scores."""
    hits = []
    for i, record in enumerate(records, 1):
        hit = dict(record) if "chunk" in record else {"chunk": dict(record)}
        hit.setdefault("rank", i)
        for key in ("vector_rank", "vector_score", "sparse_rank", "sparse_score", "fused_rank", "rrf_score", "rerank_score"):
            hit.setdefault(key, None)
        hits.append(hit)
    return hits


class EvaluationError(RuntimeError):
    """An incomplete run; ``result`` retains successful stages and artifacts."""
    def __init__(self, message: str, result: RunResult):
        super().__init__(message)
        self.result = result


@dataclass
class RunResult:
    run: JSON
    artifact_path: Path

    @property
    def outcomes(self) -> list[JSON]:
        return self.run["outcomes"]

    @property
    def failed_count(self) -> int:
        return sum(o["status"] != "completed" for o in self.outcomes) + sum(c.get("error") is not None for c in self.run.get("calibration", []))

    @property
    def ok(self) -> bool:
        return self.failed_count == 0

    def summary(self) -> JSON:
        return json.loads(_native.summarize(_json(self.run)))

    def write_report(self, directory: PathLike | None = None) -> Path:
        path = Path(directory).resolve() if directory is not None else self.artifact_path.parent / "reports"
        _native.write_report(_json(self.run), str(path))
        return path

    @classmethod
    def from_file(cls, path: PathLike) -> RunResult:
        path = Path(path).resolve()
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("version") not in (1, 2):
            raise ValueError("Unsupported run artifact version")
        return cls(run, path)


class Evaluator:
    """Immutable configuration; each operation owns its Rust runtime/resources.

    Calls release the GIL. Use ``asyncio.to_thread`` from asynchronous applications.
    Use separate artifact directories for concurrent independent evaluations.
    """
    def __init__(self, *, _config_json: str):
        self._config_json = _config_json

    @classmethod
    def from_file(cls, path: PathLike) -> Evaluator:
        config = _native.config_from_file(str(Path(path).resolve()), sys.executable)
        return cls(_config_json=config)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], base_dir: PathLike | None = None) -> Evaluator:
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
        text = _native.config_from_dict(_json(dict(config)), str(base), sys.executable)
        return cls(_config_json=text)

    @property
    def config(self) -> JSON:
        return json.loads(self._config_json)

    def _execute(self, operation: str, **kwargs: Any) -> Any:
        return json.loads(_native.execute(self._config_json, operation, _json(kwargs)))

    def validate(self, online: bool = False) -> JSON:
        return self._execute("validate", online=online)

    def index(self) -> list[Path]:
        return [Path(p) for p in self._execute("index")]

    def retrieve(self, query: str, *, pipeline: str | None = None, embedding: str | None = None, languages: Sequence[str] | None = None) -> list[JSON]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if pipeline is not None and embedding is not None:
            raise ValueError("Specify pipeline or embedding, not both")
        config = self.config
        if pipeline is not None and pipeline not in {p["name"] for p in config["pipelines"]}:
            # Implicit legacy names are supplied by the Rust validation view.
            if pipeline not in {p["name"] for p in self.validate()["pipelines"]}:
                raise ValueError(f"Unknown pipeline: {pipeline}")
        if embedding is not None and embedding not in {m["name"] for m in config["embeddings"]}:
            raise ValueError(f"Unknown embedding: {embedding}")
        return self._execute("retrieve", query=query, pipeline=pipeline, embedding=embedding, languages=list(languages or []))

    def _run(self, operation: str, limit: int | None, raise_on_failure: bool, verbose: bool) -> RunResult:
        if limit is not None:
            _positive(limit, "limit")
        raw = self._execute(operation, limit=limit, verbose=verbose)
        result = RunResult(raw["run"], Path(raw["artifact_path"]))
        if raise_on_failure and not result.ok:
            raise EvaluationError(f"{result.failed_count} stages/comparisons failed; results saved to {result.artifact_path}", result)
        return result

    def compare(self, limit: int | None = None, raise_on_failure: bool = True, *, verbose: bool = False) -> RunResult:
        return self._run("compare", limit, raise_on_failure, verbose)

    def evaluate(self, limit: int | None = None, raise_on_failure: bool = True, *, verbose: bool = False) -> RunResult:
        return self._run("evaluate", limit, raise_on_failure, verbose)


class BM25Index:
    """Native immutable sparse index. Returned hits match the evaluator schema."""
    def __init__(self, *, _index: Any):
        self._index = _index

    @classmethod
    def build(cls, chunks: Sequence[Mapping[str, Any]], directory: PathLike) -> BM25Index:
        return cls(_index=_native._BM25Index.build(_json(list(chunks)), str(Path(directory).resolve())))

    @classmethod
    def load(cls, path: PathLike) -> BM25Index:
        return cls(_index=_native._BM25Index.load(str(Path(path).resolve())))

    @property
    def path(self) -> Path:
        return Path(self._index.path)

    def search(self, query: str, languages: Sequence[str] | None = None, *, k1: float = 1.2, b: float = 0.75, top_k: int = 20) -> list[JSON]:
        _positive(top_k, "top_k")
        return json.loads(self._index.search(query, list(languages or []), k1, b, top_k))


def tokenize(text: str) -> list[str]:
    return _native.tokenize(text)


def rrf(dense_hits: Sequence[Mapping[str, Any]], sparse_hits: Sequence[Mapping[str, Any]], *, rank_constant: float = 60.0, top_k: int = 20) -> list[JSON]:
    _positive(top_k, "top_k")
    return json.loads(_native.rrf(_json(_hits(dense_hits)), _json(_hits(sparse_hits)), rank_constant, top_k))


def retrieval_metrics(hits: Sequence[Mapping[str, Any]], relevant_pages: Sequence[Mapping[str, Any]], k: int = 5) -> dict[str, float] | None:
    _positive(k, "k")
    return json.loads(_native.retrieval_metrics(_json(_hits(hits)), _json(list(relevant_pages)), k))


def score_judgment(judgment: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]], key_facts: Sequence[str], *, unanswerable: bool = False) -> dict[str, float]:
    return json.loads(_native.score_judgment(_json(dict(judgment)), _json(_hits(chunks)), len(key_facts), unanswerable))
