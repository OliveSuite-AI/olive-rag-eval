"""Refresh or check the small packaged fixture resources from repository data."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expected():
    result = {}
    for source, target in (("starter_corpus.jsonl", "corpus.jsonl"), ("starter_cases.jsonl", "cases.jsonl"), ("calibration_draft.jsonl", "calibration.jsonl")):
        result[target] = (ROOT / "data" / source).read_bytes()
    provenance = json.loads((ROOT / "data/starter_manifest.json").read_text())
    provenance["source_snapshots_bundled"] = False
    result["provenance.json"] = (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode()
    for source, target in (("fixture.toml", "dense.toml"), ("hybrid-fixture.toml", "hybrid.toml")):
        config = (ROOT / "configs" / source).read_text()
        for old, new in (("../data/starter_corpus.jsonl", "corpus.jsonl"), ("../data/starter_cases.jsonl", "cases.jsonl"), ("../data/calibration_draft.jsonl", "calibration.jsonl"), ("../artifacts/hybrid-fixture", "artifacts"), ("../artifacts/fixture", "artifacts")):
            config = config.replace(old, new)
        config = config.replace('fixture_stdio = "../services/runtime.py"', 'fixture_stdio = __RUNTIME__\npython_executable = __PYTHON__')
        result[target] = config.encode()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    directory = ROOT / "python/olive_rag_eval/assets"
    for name, data in expected().items():
        path = directory / name
        if check:
            assert path.read_bytes() == data, f"Stale packaged resource: {name}; run scripts/sync_python_assets.py"
        else:
            path.write_bytes(data)
