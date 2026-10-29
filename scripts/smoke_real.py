"""Opt-in real-model smoke run; requires already running local inference servers."""
import argparse
import json
from pathlib import Path
import subprocess

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,default=Path("target/debug/rag-eval"))
    p.add_argument("--config",type=Path,default=Path("configs/local.toml"))
    args=p.parse_args()
    prefix=[str(args.binary.resolve()),"--config",str(args.config.resolve())]
    result=subprocess.run(prefix+["validate","--online"],text=True,capture_output=True,check=True)
    if "Fixture: false" not in result.stdout:
        raise SystemExit("Refusing to label a fixture run as real-model evidence")
    subprocess.run(prefix+["index"],check=True)
    run=subprocess.run(prefix+["evaluate","--limit","6"],text=True,capture_output=True,check=True)
    path=Path(run.stdout.strip().splitlines()[-1])
    data=json.loads(path.read_text())
    assert not data["fixture"] and all(o["status"]=="completed" for o in data["outcomes"])
    print(f"Real-model smoke passed: {len(data['outcomes'])} comparisons. Evidence: {path}")
