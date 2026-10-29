"""Test an installed wheel, not an editable/source import. Stdlib only."""
import contextlib
import io
import json
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import olive_rag_eval as olive


class InstalledTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="olive wheel ")
        self.addCleanup(self.temp.cleanup)
        self.root = (Path(self.temp.name) / "ଓଡ଼ିଆ space").resolve()

    def evaluator(self):
        return olive.Evaluator.from_file(olive.init_example(self.root))

    def test_native_import_resources_and_no_heavy_dependencies(self):
        self.assertIn("site-packages", str(Path(olive.__file__)))
        self.assertEqual(olive.__version__, version("olive-rag-eval"))
        self.assertEqual(olive.tokenize("ଓଡ଼ିଆ ପୃଥିବୀ ABC ୧୨ 12"), ["ଓଡ଼ିଆ", "ପୃଥିବୀ", "abc", "୧୨", "12"])
        for module in ("torch", "numpy", "transformers", "laya"):
            self.assertNotIn(module, sys.modules)
        self.assertTrue(Path(olive.__file__).with_name("py.typed").exists())
        path = olive.init_example(self.root)
        self.assertTrue(path.exists())
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            olive.init_example(self.root)
        self.assertEqual(before, path.read_bytes())

    def test_full_api_run_resume_reports_and_path_resolution(self):
        e = self.evaluator()
        self.assertEqual(e.config["python_executable"], sys.executable)
        self.assertEqual(e.validate()["cases"], 60)
        self.assertEqual(len(e.validate(online=True)["health"]), 4)
        self.assertEqual(len(e.index()), 2)
        hits = e.retrieve("ପୃଥିବୀ", pipeline="hybrid-rrf", languages=["or"])
        self.assertTrue(hits)
        self.assertTrue(all(h["chunk"]["language"] == "or" for h in hits))
        result = e.evaluate(limit=2)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.outcomes), 8)
        self.assertTrue(result.artifact_path.exists())
        self.assertEqual(result.failed_count, 0)
        self.assertTrue(result.summary()["fixture"])
        report = result.write_report()
        self.assertTrue(all((report / p).exists() for p in ("report.md", "summary.json", "results.csv")))
        loaded = olive.RunResult.from_file(result.artifact_path)
        self.assertEqual(loaded.summary(), result.summary())
        resumed = e.evaluate(limit=2)
        self.assertTrue(all(s["cached"] for o in resumed.outcomes for s in o["stages"].values()))
        self.assertEqual(resumed.artifact_path, result.artifact_path)
        config = e.config
        config["artifacts"] = "relative-artifacts"
        second = olive.Evaluator.from_dict(config, base_dir=self.root)
        self.assertEqual(Path(second.config["artifacts"]), self.root / "relative-artifacts")
        self.assertNotEqual(e.config["artifacts"], second.config["artifacts"])
        copy = e.config
        copy["cutoff"] = -1
        self.assertEqual(e.config["cutoff"], 5)

    def test_partial_results_and_invalid_configuration(self):
        e = self.evaluator()
        config = e.config
        config["llm_reranker"]["model"] = "fixture-bad-ranking"
        e = olive.Evaluator.from_dict(config)
        e.index()
        with self.assertRaises(olive.EvaluationError) as caught:
            e.evaluate(limit=1)
        result = caught.exception.result
        self.assertEqual(result.failed_count, 1)
        self.assertTrue(result.artifact_path.exists())
        self.assertEqual(sum(o["status"] == "completed" for o in result.outcomes), 3)
        self.assertFalse(e.evaluate(limit=1, raise_on_failure=False).ok)
        with self.assertRaises(ValueError):
            e.evaluate(limit=0)
        with self.assertRaises(TypeError):
            e.compare(limit=True)
        with self.assertRaises(ValueError):
            e.retrieve("q", pipeline="missing")
        config["embeddings"][0]["url"] = "https://example.com/v1"
        with self.assertRaises(ValueError):
            olive.Evaluator.from_dict(config)
        with self.assertRaises(OSError):
            olive.Evaluator.from_file(self.root / "absent.toml")
        path = self.root / "invalid.toml"
        path.write_text("broken = [", encoding="utf-8")
        with self.assertRaises(ValueError):
            olive.Evaluator.from_file(path)

    def test_low_level_bm25_rrf_and_metrics(self):
        self.root.mkdir(parents=True)
        chunks = [
            {"id": "a", "text": "cat cat", "source": "a", "page": 1, "language": "en"},
            {"id": "b", "text": "cat dog dog", "source": "b", "page": 1, "language": "en"},
            {"id": "c", "text": "ଓଡ଼ିଆ", "source": "c", "page": 1, "language": "or"},
        ]
        index = olive.BM25Index.build(chunks, self.root / "sparse index")
        hits = index.search("cat", ["en"], top_k=2)
        self.assertEqual([h["chunk"]["id"] for h in hits], ["a", "b"])
        self.assertAlmostEqual(hits[0]["sparse_score"], __import__("math").log(1.2) * 4.4 / (2 + 1.2 * (.25 + .75 * 2 / 2.5)))
        self.assertEqual(hits, olive.BM25Index.load(index.path).search("cat", ["en"], top_k=2))
        fused = olive.rrf(list(reversed(hits)), hits, top_k=2)
        self.assertEqual([h["chunk"]["id"] for h in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertEqual(olive.retrieval_metrics(fused, [{"source": "a", "page": 1, "grade": 2}], 5)["precision@5"], .2)
        self.assertIsNone(olive.retrieval_metrics(fused, [], 5))
        judgment = {"claims": [{"text": "x", "supported": False}], "unsupported_claims": ["x"], "useful_chunk_ids": ["a"], "covered_fact_ids": [1], "answer_relevance": 1., "abstained": False, "answer_language_correct": True}
        scores = olive.score_judgment(judgment, chunks, ["fact"])
        self.assertEqual(scores["faithfulness"], 0.)
        self.assertEqual(scores["hallucination_rate"], 1.)
        with self.assertRaises(ValueError):
            olive.BM25Index.build(chunks[:1], index.path)
        with self.assertRaises(ValueError):
            olive.rrf(hits + hits, [])
        with self.assertRaises(ValueError):
            index.search("cat", b=2.)
        with self.assertRaises(ValueError):
            olive.retrieval_metrics(hits, [], 0)

    def test_cli_help_and_installed_example_without_cargo_on_path(self):
        config = olive.init_example(self.root)
        environment = os.environ.copy()
        binary_dir = str(Path(sys.executable).parent)
        system_dir = str(Path(os.environ.get("SystemRoot", "/")) / "System32") if os.name == "nt" else os.defpath
        environment["PATH"] = binary_dir + os.pathsep + system_dir
        environment.pop("PYTHONPATH", None)
        self.assertIsNone(shutil.which("cargo", path=environment["PATH"]))
        cli = shutil.which("olive-rag-eval", path=environment["PATH"])
        self.assertIsNotNone(cli)
        def run(*args):
            return subprocess.run([cli, *args], cwd=self.root, env=environment, text=True, encoding="utf-8", capture_output=True, timeout=120)
        self.assertEqual(run("--help").returncode, 0)
        self.assertEqual(run("--version").returncode, 0)
        self.assertIn("olive-rag-eval", run("--version").stdout)
        self.assertEqual(run("bad-command").returncode, 2)
        for command in ("validate", "index", "evaluate"):
            args = ["--config", str(config), command]
            if command == "evaluate":
                args += ["--limit", "1"]
            result = run(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
        module = subprocess.run([sys.executable, "-I", "-m", "olive_rag_eval", "--help"], cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(module.returncode, 0, module.stderr)
        server_help = subprocess.run([sys.executable, "-I", "-m", "olive_rag_eval.model_server", "--help"], cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(server_help.returncode, 0, server_help.stderr)

    def test_native_work_releases_gil(self):
        gate = threading.Event()
        stop = threading.Event()
        ticks = [0]
        def worker():
            gate.wait()
            while not stop.is_set():
                ticks[0] += 1
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        old = sys.getswitchinterval()
        sys.setswitchinterval(.5)
        try:
            text = "ଓଡ଼ିଆ " * 20000
            gate.set()
            olive.tokenize(text)
            stop.set()
            self.assertGreater(ticks[0], 0)
        finally:
            stop.set()
            sys.setswitchinterval(old)
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
