"""Install a wheel into a fresh environment outside the checkout and test it."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--installer", choices=["pip", "uv"], required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    tests = Path(__file__).resolve().parents[1] / "python" / "tests" / "test_installed.py"
    with tempfile.TemporaryDirectory(prefix="olive fresh wheel ") as directory:
        root = Path(directory)
        env = root / "environment"
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if args.installer == "pip":
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(env)
            install = [str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--only-binary=:all:", str(wheel)]
        else:
            subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True)
            install = ["uv", "pip", "install", "--offline", "--no-deps", "--python", str(python), str(wheel)]
        subprocess.run(install, check=True, cwd=root)
        test_env = os.environ.copy()
        test_env.pop("PYTHONPATH", None)
        test_env["PYTHONUTF8"] = "1"
        test_env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run([str(python), "-I", str(tests)], cwd=root, env=test_env, check=True)
        print(f"Fresh {args.installer} install and tests passed: {wheel.name}")
