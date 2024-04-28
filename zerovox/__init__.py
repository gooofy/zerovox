import os
from pathlib import Path

import torch

def get_cache_path() -> Path:
    return Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox"))
                
def _get_model_name(model:str, lang:str, version:str) -> str:
    return f"{model}-{lang}-{version}"

def _get_source_url(modelname:str, relpath:str) -> str:
    return f"https://huggingface.co/goooofy/{modelname}/resolve/main/{relpath}?download=true"

# def get_target_path(relpath, run_dir: str | Path | None = None):
#     if run_dir is None:
#         run_dir = Path(__file__).parent.parent / "model_repo" / RUN_NAME
#     return Path(run_dir) / relpath

def download_model_file(model:str, lang:str, version:str, relpath:str) -> Path:

    modelname   = _get_model_name(model, lang, version)
    target_dir  = get_cache_path() / "model_repo" / modelname
    target_path = target_dir / relpath

    if target_path.exists():
        return target_path

    os.makedirs (target_dir, exist_ok=True)

    url = _get_source_url(modelname, relpath)

    torch.hub.download_url_to_file(url, str(target_path))

    return target_path
