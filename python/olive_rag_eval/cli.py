"""Console entry point; delegates directly to the shared Rust CLI."""
import sys
from . import _native


def main() -> None:
    raise SystemExit(_native.run_cli(["olive-rag-eval", *sys.argv[1:]]))
