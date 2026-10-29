"""Rebuild a source archive outside the repository to check source completeness."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--maturin", default="maturin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="olive source build ") as temp:
        with tarfile.open(args.archive.resolve()) as archive:
            archive.extractall(temp, filter="data")
        roots = [p for p in Path(temp).iterdir() if p.is_dir()]
        assert len(roots) == 1
        metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=roots[0], text=True))
        for package in metadata["packages"]:
            assert Path(package["manifest_path"]).resolve().is_relative_to(roots[0].resolve()), "Source archive references a manifest outside the extracted tree"
            for dependency in package["dependencies"]:
                if dependency.get("path"):
                    assert Path(dependency["path"]).resolve().is_relative_to(roots[0].resolve()), "External path dependency in source archive"
        environment = os.environ.copy()
        # Normalized archive mtimes can make a shared Cargo target appear fresh
        # despite different source contents. This is deliberately a clean build.
        environment["CARGO_TARGET_DIR"] = str(Path(temp) / "target")
        subprocess.run([str(Path(args.maturin).resolve()) if "/" in args.maturin or "\\" in args.maturin else args.maturin,
                        "build", "--locked", "--out", str(output)], cwd=roots[0], env=environment, check=True)
