"""Create a self-contained synthetic example outside the installed package."""
from importlib.resources import files
import json
from pathlib import Path
import sys
from .api import PathLike


def init_example(directory: PathLike, hybrid: bool = True) -> Path:
    destination = Path(directory).resolve()
    names = ["corpus.jsonl", "cases.jsonl", "calibration.jsonl", "provenance.json", "fixture.toml"]
    for name in names:
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite {destination / name}")
    assets = files("olive_rag_eval").joinpath("assets")
    config = assets.joinpath("hybrid.toml" if hybrid else "dense.toml").read_text(encoding="utf-8")
    runtime = Path(__file__).with_name("_runtime.py").resolve()
    config = config.replace("__RUNTIME__", json.dumps(str(runtime), ensure_ascii=False)).replace("__PYTHON__", json.dumps(sys.executable, ensure_ascii=False))
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        data = config.encode() if name == "fixture.toml" else assets.joinpath(name).read_bytes()
        with (destination / name).open("xb") as stream:
            stream.write(data)
    return destination / "fixture.toml"
