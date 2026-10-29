"""Rust-powered local RAG evaluation, with Odia-first multilingual examples."""
from ._native import __version__
from .api import BM25Index, EvaluationError, Evaluator, RunResult, retrieval_metrics, rrf, score_judgment, tokenize
from .examples import init_example

__all__ = ["BM25Index", "EvaluationError", "Evaluator", "RunResult", "init_example", "retrieval_metrics", "rrf", "score_judgment", "tokenize", "__version__"]
