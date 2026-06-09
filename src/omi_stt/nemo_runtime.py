from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .hf_auth import hf_token


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "text"):
        return str(output.text)
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(output)


@lru_cache(maxsize=2)
def load_nemo_model(repo_id: str, filename: str = "omimedstt-v1.nemo", device: str = "cuda") -> Any:
    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("NeMo runtime is not installed. Run: pip install 'omi-med-stt[nemo]'") from exc
    checkpoint = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token())
    return ASRModel.restore_from(checkpoint, map_location=device)


def transcribe_nemo(audio_paths: list[str | Path], repo_id: str, device: str = "cuda") -> list[str]:
    model = load_nemo_model(repo_id=repo_id, device=device)
    outputs = model.transcribe([str(p) for p in audio_paths])
    return [_extract_text(o).strip() for o in outputs]
