from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .hf_auth import hf_token


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False


def _download_or_resolve_model(model_id_or_path: str) -> tuple[Path, Path]:
    local = Path(model_id_or_path)
    if _path_exists(local):
        return local / "config.json", local / "model.safetensors"

    from huggingface_hub import hf_hub_download

    token = hf_token()
    config_path = Path(hf_hub_download(model_id_or_path, "config.json", token=token))
    weights_path = Path(hf_hub_download(model_id_or_path, "model.safetensors", token=token))
    return config_path, weights_path


def _install_omi_adapter_runtime(model, rank: int) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from parakeet_mlx.conformer import ConformerBlock

    class LinearAdapter(nn.Module):
        def __init__(self, d_model: int, dim: int):
            super().__init__()
            self.module = [
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim, bias=False),
                nn.SiLU(),
                nn.Linear(dim, d_model, bias=False),
            ]

        def __call__(self, x: mx.array) -> mx.array:
            y = x
            for layer in self.module:
                y = layer(y)
            return x + y

    class MedicalAdapterLayer(nn.Module):
        def __init__(self, d_model: int, dim: int):
            super().__init__()
            self.medical_v1d_rank128 = LinearAdapter(d_model, dim)

    def call_with_adapter(self, x, pos_emb=None, mask=None, cache=None):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))

        x_norm = self.norm_self_att(x)
        x = x + self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x = x + self.conv(self.norm_conv(x), cache=cache)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        x = self.norm_out(x)

        if hasattr(self, "adapter_layer"):
            x = self.adapter_layer.medical_v1d_rank128(x)
        return x

    d_model = model.encoder_config.d_model
    for layer in model.encoder.layers:
        layer.adapter_layer = MedicalAdapterLayer(d_model, rank)
    ConformerBlock.__call__ = call_with_adapter


@lru_cache(maxsize=2)
def _load_model(model_id_or_path: str):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from parakeet_mlx.utils import from_config

    config_path, weights_path = _download_or_resolve_model(model_id_or_path)
    config = json.loads(config_path.read_text())
    quantization = config.pop("quantization", None)

    adapter_rank = config.get("encoder", {}).pop("medical_adapter_rank", None)
    model = from_config(config)
    if adapter_rank is not None:
        _install_omi_adapter_runtime(model, int(adapter_rank))

    if quantization:
        nn.quantize(
            model,
            bits=int(quantization.get("bits", 8)),
            group_size=int(quantization.get("group_size", 64)),
        )

    model.load_weights(str(weights_path))

    if not quantization:
        curr_weights = dict(tree_flatten(model.parameters()))
        curr_weights = [(k, v.astype(mx.bfloat16)) for k, v in curr_weights.items()]
        model.update(tree_unflatten(curr_weights))
    model.eval()
    return model


def _result_to_text(result) -> str:
    if hasattr(result, "text"):
        text = result.text
        if text:
            return str(text).strip()
    if hasattr(result, "sentences"):
        parts = []
        for sentence in result.sentences:
            if hasattr(sentence, "text") and sentence.text:
                parts.append(str(sentence.text))
            elif hasattr(sentence, "tokens"):
                parts.extend(str(token.text) for token in sentence.tokens if getattr(token, "text", None))
        return " ".join(parts).strip()
    if hasattr(result, "tokens"):
        return " ".join(str(token.text) for token in result.tokens if getattr(token, "text", None)).strip()
    return str(result).strip()


def transcribe_mlx(audio_paths: list[str | Path], repo_id: str) -> list[str]:
    model = _load_model(str(repo_id))
    texts: list[str] = []
    for path in audio_paths:
        result = model.transcribe(Path(path))
        text = _result_to_text(result)
        if not text:
            raise RuntimeError(f"MLX runtime produced an empty transcript for {path}")
        texts.append(text)
    return texts
