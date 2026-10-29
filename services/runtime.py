"""Repository compatibility entry point for the packaged, stdlib-only runtime."""
import importlib.util
from pathlib import Path

_runtime_path=Path(__file__).resolve().parents[1]/"python"/"olive_rag_eval"/"_runtime.py"
_spec=importlib.util.spec_from_file_location("olive_rag_runtime_standalone",_runtime_path)
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name:value for name,value in vars(_module).items() if not name.startswith("_")})

if __name__=="__main__":
    main()
