"""Print artifact metadata for filling a model profile. Does not download models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"services"))
from runtime import digest_path

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",type=Path)
    parser.add_argument("--model",required=True)
    parser.add_argument("--revision",required=True)
    args=parser.parse_args()
    if not args.path.exists():
        parser.error("Model path does not exist")
    # llama.cpp profiles use a plain GGUF file checksum. Python services use
    # the documented model-directory digest, including relative filenames.
    digest=hashlib.sha256(args.path.read_bytes()).hexdigest() if args.path.is_file() else digest_path(args.path)
    print(f"model = {json.dumps(args.model)}")
    print(f"revision = {json.dumps(args.revision)}")
    print(f"artifact_sha256 = {json.dumps(digest)}")
