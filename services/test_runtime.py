import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from runtime import FixtureRuntime, LayaRuntime, digest_path, require_capacity, validate_request


class RuntimeTests(unittest.TestCase):
    def test_validation_and_capacity(self):
        validate_request({"model":"m","input":["ଓଡ଼ିଆ"]},"m")
        for payload in ({"model":"wrong","input":["x"]},{"model":"m","input":[]},{"model":"m","input":[3]}):
            with self.assertRaises(ValueError):
                validate_request(payload,"m")
        with self.assertRaises(ValueError):
            validate_request({"model":"m","query":"x","documents":[{"id":"a","text":"x"}]*2},"m",True)
        with self.assertRaises(ValueError):
            require_capacity(513,512,False)
        self.assertTrue(require_capacity(513,512,True))
        self.assertFalse(require_capacity(512,512,False))

    def test_digest_changes_when_name_or_bytes_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"weights";p.write_bytes(b"abc");a=digest_path(d)
            p.write_bytes(b"abd");self.assertNotEqual(a,digest_path(d))
            p.write_bytes(b"abc");p.rename(Path(d)/"renamed");self.assertNotEqual(a,digest_path(d))

    def test_fixture_preserves_odia_and_is_marked_synthetic(self):
        f=FixtureRuntime();r=f.route("/v1/embeddings",{"model":"fixture-gemma","input":["ଓଡ଼ିଆ"]})
        self.assertEqual(len(r["data"][0]["embedding"]),768)
        self.assertTrue(r["fixture"])
        request={"model":"fixture-chat","messages":[{"role":"user","content":json.dumps({"question":"q","context":[{"id":"a","source":"s","page":1,"text":"ଓଡ଼ିଆ"}]})}]}
        self.assertIn("ଓଡ଼ିଆ",f.route("/v1/chat/completions",request)["choices"][0]["message"]["content"])

    def test_fixture_judge_detects_unsupported_claim(self):
        f=FixtureRuntime()
        data={"question":"q","context":[],"answer":"invented","reference_answer":"ref","key_facts":["fact"],"expected_answer_language":"en"}
        r=f.route("/v1/chat/completions",{"model":"m","messages":[{"role":"user","content":json.dumps(data)}]})
        j=json.loads(r["choices"][0]["message"]["content"])
        self.assertEqual(j["unsupported_claims"],["invented"])


if __name__=="__main__":
    unittest.main()
