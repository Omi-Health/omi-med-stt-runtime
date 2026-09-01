from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
from huggingface_hub import hf_hub_download

from .hf_auth import hf_token


# This is the measured, bounded GPU recipe for the public Omi Med STT v1
# checkpoint. Keep these defaults together so the CLI cannot silently drift back
# to checkpoint defaults when NeMo changes its transcribe() implementation.
DEFAULT_NEMO_REVISION = "dc854cd350873636bf38bdf94b226f5ad7eaa2bd"
NEMO_ATTENTION_CONTEXT = (256, 256)
NEMO_BATCH_SIZE = 8
NEMO_MAX_BATCH_AUDIO_SECONDS = 900.0
NEMO_MAX_SYMBOLS = 10


def _torch_module() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised without the GPU extra
        raise RuntimeError(
            "The NeMo GPU runtime is not installed. Run: pip install 'omi-med-stt[nemo]'"
        ) from exc
    return torch


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "text"):
        return str(output.text)
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(output)


def _require_bf16_cuda(torch: Any, device: str) -> None:
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "Omi Med STT v1's qualified NeMo recipe requires an NVIDIA CUDA GPU; "
            "use the MLX runtime on Apple Silicon."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The NeMo runtime was selected, but CUDA is not available.")
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and not is_bf16_supported():
        raise RuntimeError(
            "This NVIDIA GPU does not support the qualified BF16 runtime. "
            "A BF16-capable Ampere-or-newer GPU is required."
        )


def configure_nemo_model(model: Any, *, device: str, torch: Any) -> Any:
    """Apply the receipt-backed GPU recipe to a restored NeMo model."""
    from omegaconf import open_dict

    _require_bf16_cuda(torch, device)
    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=list(NEMO_ATTENTION_CONTEXT),
    )

    decoding = model.cfg.decoding.copy()
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"
        decoding.greedy.max_symbols = NEMO_MAX_SYMBOLS
        decoding.greedy.use_cuda_graph_decoder = False
        if getattr(decoding, "beam", None) is not None:
            decoding.beam.return_best_hypothesis = True
            decoding.beam.preserve_alignments = False
    model.change_decoding_strategy(decoding, verbose=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


@lru_cache(maxsize=2)
def load_nemo_model(
    repo_id: str,
    filename: str = "omimedstt-v1.nemo",
    revision: str = DEFAULT_NEMO_REVISION,
    device: str = "cuda",
) -> Any:
    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as exc:  # pragma: no cover - exercised without the GPU extra
        raise RuntimeError(
            "The NeMo GPU runtime is not installed. Run: pip install 'omi-med-stt[nemo]'"
        ) from exc

    torch = _torch_module()
    _require_bf16_cuda(torch, device)
    checkpoint = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=hf_token(),
    )
    model = ASRModel.restore_from(checkpoint, map_location="cpu")
    return configure_nemo_model(model, device=device, torch=torch)


def _audio_duration(path: Path) -> float:
    return float(sf.info(str(path)).duration)


def _duration_bounded_batches(
    values: list[tuple[int, Path, float]],
    *,
    size: int = NEMO_BATCH_SIZE,
    max_audio_seconds: float = NEMO_MAX_BATCH_AUDIO_SECONDS,
) -> Iterable[list[tuple[int, Path, float]]]:
    batch: list[tuple[int, Path, float]] = []
    batch_audio_seconds = 0.0
    for value in values:
        duration = value[2]
        if batch and (len(batch) >= size or batch_audio_seconds + duration > max_audio_seconds):
            yield batch
            batch = []
            batch_audio_seconds = 0.0
        batch.append(value)
        batch_audio_seconds += duration
    if batch:
        yield batch


def transcribe_nemo(
    audio_paths: list[str | Path],
    repo_id: str,
    device: str = "cuda",
) -> list[str]:
    """Transcribe with the selected GPU recipe and restore caller input order."""
    torch = _torch_module()
    _require_bf16_cuda(torch, device)
    model = load_nemo_model(repo_id=repo_id, device=device)

    indexed = [
        (index, Path(path), _audio_duration(Path(path)))
        for index, path in enumerate(audio_paths)
    ]
    indexed.sort(key=lambda item: (item[2], item[0]))
    transcripts: list[str | None] = [None] * len(indexed)

    for batch in _duration_bounded_batches(indexed):
        paths = [str(item[1]) for item in batch]
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            outputs = model.transcribe(
                paths,
                batch_size=len(paths),
                timestamps=False,
                verbose=False,
            )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"NeMo returned {len(outputs)} transcripts for a batch of {len(batch)} inputs"
            )
        for (original_index, _path, _duration), output in zip(batch, outputs):
            transcripts[original_index] = _extract_text(output).strip()

    if any(text is None for text in transcripts):  # pragma: no cover - defensive invariant
        raise RuntimeError("NeMo did not return a transcript for every input")
    return [str(text) for text in transcripts]
