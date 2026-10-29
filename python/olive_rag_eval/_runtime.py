"""Loopback-only inference service. Heavy model imports occur only in real runtimes."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import unicodedata

MAX_BODY = 4_000_000
QUESTION = {
    "relevance": {
        "type": "choice",
        "instructions": "Does the passage contain information that helps answer the query?",
        "criteria": {
            "relevant": "Contains evidence answering at least part of the query",
            "irrelevant": "Does not help answer the query",
        },
    }
}


def digest_path(path):
    """Digest regular files as relative-name NUL bytes NUL, in sorted path order."""
    path = Path(path)
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty model artifact")
    for file in files:
        name = file.name if path.is_file() else file.relative_to(path).as_posix()
        h.update(name.encode())
        h.update(b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def validate_request(payload, model, rerank=False):
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise ValueError("Unknown model")
    if not isinstance(payload.get("truncate", False), bool):
        raise ValueError("truncate must be boolean")
    if rerank:
        if not isinstance(payload.get("query"), str) or not payload["query"].strip():
            raise ValueError("Nonempty query required")
        docs = payload.get("documents")
        if not isinstance(docs, list) or not 1 <= len(docs) <= 512:
            raise ValueError("Expected 1..512 documents")
        ids = []
        for d in docs:
            if not isinstance(d, dict) or not isinstance(d.get("id"), str) or not d["id"] or not isinstance(d.get("text"), str) or not d["text"].strip():
                raise ValueError("Invalid document")
            ids.append(d["id"])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document IDs")
    else:
        texts = payload.get("input")
        if not isinstance(texts, list) or not 1 <= len(texts) <= 512 or any(not isinstance(s, str) or not s.strip() for s in texts):
            raise ValueError("Expected 1..512 nonempty input strings")


def require_capacity(length, maximum, truncate):
    if length > maximum and not truncate:
        raise ValueError(f"Input has {length} tokens; model limit is {maximum}. Truncation disabled.")
    return length > maximum


class TorchRuntime:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.args = args
        self.torch = torch
        self.batch_size = args.batch_size
        if args.kind == "embedding":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(args.model_dir), device=args.device, local_files_only=True)
            self.tok = self.model.tokenizer
            self.limit = min(args.max_tokens, self.model.max_seq_length)
            self.model.max_seq_length = self.limit
        else:
            self.tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
            self.limit = min(args.max_tokens, self.model.config.max_position_embeddings, self.tok.model_max_length)

    def embed(self, p):
        validate_request(p, self.args.model)
        lengths = [len(self.tok(s, truncation=False)["input_ids"]) for s in p["input"]]
        truncated = [require_capacity(n, self.limit, p.get("truncate", False)) for n in lengths]
        # Prefixes are applied by Rust. encode(), rather than encode_query/document(), avoids applying them twice.
        vectors = self.model.encode(p["input"], batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True).tolist()
        return {"model": self.args.model, "truncated": any(truncated), "input_tokens": lengths,
                "data": [{"index": i, "embedding": v, "truncated": truncated[i]} for i, v in enumerate(vectors)]}

    def rerank(self, p):
        validate_request(p, self.args.model, rerank=True)
        query = p["query"]
        lengths = [len(self.tok(query, d["text"], truncation=False)["input_ids"]) for d in p["documents"]]
        truncated = [require_capacity(n, self.limit, p.get("truncate", False)) for n in lengths]
        scores = []
        for start in range(0, len(lengths), self.batch_size):
            docs = p["documents"][start:start + self.batch_size]
            inputs = self.tok([query] * len(docs), [d["text"] for d in docs], padding=True, truncation=True, max_length=self.limit, return_tensors="pt").to(self.args.device)
            with self.torch.inference_mode():
                values = self.model(**inputs).logits
            if values.ndim != 2 or values.shape[1] != 1:
                raise ValueError("Expected one cross-encoder logit per document")
            scores.extend(values[:, 0].float().cpu().tolist())
        return {"model": self.args.model, "results": [
            {"id": d["id"], "score": scores[i], "input_tokens": lengths[i], "truncated": truncated[i]}
            for i, d in enumerate(p["documents"])]}


class LayaRuntime:
    def __init__(self, args):
        import laya
        self.args = args
        self.agent = laya.load(str(args.model_dir), device=args.device)
        self.limit = min(args.max_tokens, 8192)

    def rerank(self, p):
        from laya.common import encode_text, state_room
        validate_request(p, self.args.model, rerank=True)
        internal = self.agent._to_internal(QUESTION["relevance"])
        room = state_room(self.agent.tok, internal, self.limit, 192)
        rows = []
        for doc in p["documents"]:
            state = json.dumps({"query": p["query"], "passage": doc["text"]}, ensure_ascii=False)
            ids = encode_text(self.agent.tok, state.replace(self.agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
            cut = require_capacity(len(ids), room, p.get("truncate", False))
            result = self.agent.predict(state, QUESTION, max_len=self.limit, head_max_len=192)
            answer = result["answers"]["relevance"]
            score = answer["probabilities"]["relevant"]
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid Laya relevance probability")
            actual_cut = bool(result.get("usage", {}).get("truncated", False))
            if actual_cut and not p.get("truncate", False):
                raise ValueError("Laya reported unexpected truncation")
            rows.append({"id": doc["id"], "score": score, "truncated": cut or actual_cut, "input_tokens": len(ids), "decision": result})
        return {"model": self.args.model, "results": rows}


class FixtureRuntime:
    """Synthetic protocol fixture, never evidence of real model or Odia quality."""
    def __init__(self, dimensions=768):
        self.dimensions = dimensions
        self.calls = {}

    def route(self, path, p):
        self.calls[path] = self.calls.get(path, 0) + 1
        if path == "/apply-template":
            return {"prompt": json.dumps(p["messages"], ensure_ascii=False)}
        if path == "/tokenize":
            return {"tokens": list(range(len(p["content"].encode()) // 3 + 1))}
        if path.endswith("/embeddings"):
            validate_request(p, p["model"])
            dimensions = 384 if "minilm" in p["model"] else self.dimensions
            vectors = []
            for i, text in enumerate(p["input"]):
                v = [0.0] * dimensions
                for word in re.findall(r"\w+", unicodedata.normalize("NFC", text).lower()):
                    index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little") % dimensions
                    v[index] += 1.0
                if not any(v):
                    v[0] = 1.0
                vectors.append({"index": i, "embedding": v, "truncated": False})
            return {"model": p["model"], "data": vectors, "truncated": False, "fixture": True}
        if path.endswith("/rerank"):
            validate_request(p, p["model"], rerank=True)
            if p["model"] == "fixture-bad-ids":
                return {"model": p["model"], "results": [{"id": "wrong", "score": 1., "truncated": False} for _ in p["documents"]]}
            terms = set(p["query"].lower().split())
            return {"model": p["model"], "results": [
                {"id": d["id"], "score": float(len(terms & set(d["text"].lower().split()))), "truncated": False, "fixture": True}
                for d in p["documents"]]}
        if path.endswith("/chat/completions"):
            if p["model"] == "fixture-bad-judge":
                return {"choices": [{"message": {"content": '{"broken":true}'}, "finish_reason": "stop"}]}
            user = next(m for m in p["messages"] if m["role"] == "user")
            data = json.loads(user["content"])
            if "documents" in data:
                terms=set(data["query"].lower().split())
                docs=sorted(data["documents"],key=lambda d:-len(terms & set(d["text"].lower().split())))
                ids=[d["id"] for d in docs]
                if p["model"]=="fixture-bad-ranking":
                    ids=["unknown"]*len(ids)
                elif p["model"]=="fixture-repair-ranking" and not any(m["role"]=="assistant" for m in p["messages"]):
                    ids=ids[:-1]
                text=json.dumps({"ordered_ids":ids},ensure_ascii=False)
            elif "reference_answer" in data:
                answer = data["answer"]
                abstained = answer.startswith(("The provided", "ଦିଆଯାଇଥିବା"))
                supported = any(c["text"] in answer for c in data["context"])
                claims = [] if abstained else [{"text": answer, "supported": supported}]
                value = {"claims": claims, "unsupported_claims": [answer] if claims and not supported else [],
                         "useful_chunk_ids": [c["id"] for c in data["context"] if c["text"] in answer],
                         "covered_fact_ids": [i + 1 for i, f in enumerate(data["key_facts"]) if f in answer],
                         "answer_relevance": 0.5, "abstained": abstained,
                         "answer_language_correct": (any("\u0b00" <= ch <= "\u0b7f" for ch in answer) if data["expected_answer_language"] == "or" else answer.isascii())}
                text = json.dumps(value, ensure_ascii=False)
            else:
                if not data["context"] or "unknown" in data["question"].lower() or "ଅଜଣା" in data["question"]:
                    text = "The provided documents don't answer this."
                else:
                    c = data["context"][0]
                    text = f"{c['text']} [{c['source']} p.{c['page']}]"
            return {"model": p["model"], "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": None, "fixture": True}
        raise ValueError("Unknown endpoint")


def handler(runtime, metadata):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, metadata)
            elif self.path == "/stats" and isinstance(runtime, FixtureRuntime):
                self.reply(200, runtime.calls)
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Invalid body size")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete body")
                def reject(_):
                    raise ValueError("Nonfinite JSON")
                p = json.loads(raw, parse_constant=reject)
                if isinstance(runtime, FixtureRuntime):
                    value = runtime.route(self.path, p)
                elif self.path == "/v1/embeddings" and hasattr(runtime, "embed"):
                    value = runtime.embed(p)
                elif self.path == "/v1/rerank" and hasattr(runtime, "rerank"):
                    value = runtime.rerank(p)
                else:
                    self.reply(404, {"error": "not_found"})
                    return
                self.reply(200, value)
            except (ValueError, TypeError, KeyError) as e:
                self.reply(422, {"error": str(e)})
            except Exception as e:
                self.reply(500, {"error": type(e).__name__, "message": str(e)[:300]})
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["embedding", "cross", "laya", "fixture"], required=True)
    p.add_argument("--stdio", action="store_true", help="Synthetic fixture only; JSON-line transport without sockets")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--artifact-sha256")
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.stdio:
        if args.kind != "fixture":
            p.error("stdio is reserved for synthetic fixtures")
        runtime = FixtureRuntime()
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["path"] == "/health":
                    value = {"status": "ok", "fixture": True}
                elif request["path"] == "/stats":
                    value = runtime.calls
                else:
                    value = runtime.route(request["path"], request["body"])
                reply = {"status": 200, "body": value}
            except Exception as e:
                reply = {"status": 422, "body": {"error": str(e)}}
            print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)
        return
    if args.kind == "fixture":
        runtime = FixtureRuntime()
        metadata = {"status": "ok", "fixture": True}
    else:
        if not args.model_dir or not args.model_dir.is_dir() or not args.model or not args.revision or not args.artifact_sha256:
            p.error("Real services require local --model-dir, --model, --revision and --artifact-sha256")
        actual = digest_path(args.model_dir)
        if actual != args.artifact_sha256:
            p.error("Model directory SHA256 does not match")
        if args.batch_size < 1 or args.max_tokens < 1:
            p.error("Batch size and max tokens must be positive")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        runtime = LayaRuntime(args) if args.kind == "laya" else TorchRuntime(args)
        versions = {}
        for name in ("torch", "transformers", "sentence-transformers", "laya"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {"status": "ok", "model": args.model, "revision": args.revision,
                    "artifact_sha256": actual, "runtime": args.kind, "device": args.device,
                    "batch_size": args.batch_size, "max_tokens": runtime.limit, "versions": versions}
    server = HTTPServer(("127.0.0.1", args.port), handler(runtime, metadata))
    server.timeout = 30
    print(json.dumps({"port": server.server_port, **metadata}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
