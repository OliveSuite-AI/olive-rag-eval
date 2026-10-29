"""Validate distribution contents and metadata without third-party dependencies."""
import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import re
from zipfile import ZipFile


def check(path):
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)), "Duplicate wheel members"
            metadata = BytesParser().parsebytes(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
            assert metadata["Name"] == "olive-rag-eval"
            cargo = (Path(__file__).resolve().parents[1] / "Cargo.toml").read_text()
            expected = re.search(r'\[workspace.package\]\s*version\s*=\s*"([^"]+)"', cargo).group(1)
            assert metadata["Version"] == expected
            assert metadata["Requires-Python"] == ">=3.10"
            assert not metadata.get_all("Requires-Dist"), "Base wheel must not pull ML dependencies"
            assert any(n.endswith((".abi3.so", ".pyd")) for n in names), "Native extension missing"
            for name in ("py.typed", "_native.pyi", "_runtime.py", "assets/hybrid.toml", "assets/corpus.jsonl"):
                assert "olive_rag_eval/" + name in names, name
            assert "abi3" in archive.read(next(n for n in names if n.endswith(".dist-info/WHEEL"))).decode()
            assert "olive-rag-eval" in archive.read(next(n for n in names if n.endswith(".dist-info/entry_points.txt"))).decode()
            assert not any(n.startswith(("target/", "artifacts/", "python/")) or "__pycache__" in n for n in names)
    else:
        with tarfile.open(path) as archive:
            names = [n.split("/", 1)[1] for n in archive.getnames() if "/" in n]
            for name in ("Cargo.toml", "Cargo.lock", "src/lib.rs", "bindings/python/Cargo.toml", "bindings/python/src/lib.rs", "python/olive_rag_eval/api.py"):
                assert name in names, name
            assert not any(n.startswith(("target/", "artifacts/", "dist/", "data/sources/", ".venv")) for n in names)
    print(f"Validated {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    for p in parser.parse_args().paths:
        check(p)
