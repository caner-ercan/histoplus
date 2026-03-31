"""Loading utilities."""

import os
import pickle
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


ENV_HISTOWMICS_HOME = "HISTOWMICS_HOME"
DEFAULT_CACHE_DIR = Path("~/.histoplus").expanduser()

# Subfolder under HISTOWMICS_HOME
HF_SUBCACHE = "hf_cache"


class HistoPLUSNotFoundError(FileNotFoundError):
    """Raised when the requested file does not exist in the local container path."""

    def __init__(self, repo_id: str, filepath: str, revision: Optional[str]):
        super().__init__(
            f"File not found locally at '{filepath}'. (Original repo: '{repo_id}')"
        )


def _get_cache_dir() -> Path:
    """Resolve the base cache directory (under HISTOWMICS_HOME/hf_cache)."""
    base = Path(os.getenv(ENV_HISTOWMICS_HOME, DEFAULT_CACHE_DIR)).expanduser()
    cache_dir = base / HF_SUBCACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_weights_from_hub(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    *,
    map_location: Optional[torch.device] = None,
    pickle_module=pickle,
    local_files_only: bool = False,
    **pickle_load_args,
):
   """
    Load model weights with torch.load directly from the local container filesystem.
    Bypasses the Hugging Face Hub entirely.

    Parameters
    ----------
    repo_id : str
        Kept for backward compatibility with existing function calls.
    filename : str
        Path inside the repo, e.g. 'weights/model.pt'.
    revision : Optional[str]
        Kept for backward compatibility.
    map_location : Optional[torch.device]
        Where to map the loaded tensors (e.g., 'cpu').
    local_files_only : bool
        Ignored in this version, as it always loads locally.

    Returns
    -------
    Any
        The object returned by `torch.load(...)`.
    """
    # Fetch the directory defined during the Docker build, defaulting to /opt/histoplus_models
    base_model_dir = Path(os.getenv("HISTOPLUS_MODEL_DIR", "/opt/histoplus_models"))
    
    # Construct the full local path
    local_path = base_model_dir / filename

    if not local_path.exists():
        raise HistoPLUSNotFoundError(repo_id, str(local_path), revision)

    return torch.load(
        local_path,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
