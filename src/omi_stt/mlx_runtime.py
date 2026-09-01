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
    if quantization:
        # Bounded attention keeps the default q8 runtime below the memory
        # budget of ordinary Apple Silicon machines.  Clearing Metal's cache
        # between files (below) is required: without it, sequential long-file
        # draws can retain stale state and collapse into unknown tokens.
        config["encoder"]["self_attention_model"] = "rel_pos_local_attn"
        config["encoder"]["att_context_size"] = [256, 256]

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
    text = ""
    if hasattr(result, "text"):
        text = result.text
        if text:
            text = str(text)
    if not text and hasattr(result, "sentences"):
        parts = []
        for sentence in result.sentences:
            if hasattr(sentence, "text") and sentence.text:
                parts.append(str(sentence.text))
            elif hasattr(sentence, "tokens"):
                parts.extend(
                    str(token.text)
                    for token in sentence.tokens
                    if getattr(token, "text", None)
                )
        text = " ".join(parts)
    if not text and hasattr(result, "tokens"):
        text = " ".join(
            str(token.text)
            for token in result.tokens
            if getattr(token, "text", None)
        )
    if not text:
        text = str(result)

    # SentencePiece's unknown token is rendered as U+2047 by the reference
    # NeMo runtime.  Keeping the literal string ``<unk>`` changes both the
    # visible transcript and word-error scoring by inventing the word "unk".
    return text.replace("<unk>", "⁇").strip()


def _centered_logmel(audio, preprocessor):
    """Build the MLX log-mel input with NeMo's centred analysis window.

    parakeet-mlx historically right-padded a short Hann window to ``n_fft``.
    NeMo centres it.  The weights were trained with the centred placement, and
    the mismatch is measurable on the frozen board even though both paths have
    the same tensor shape.
    """
    import mlx.core as mx
    from parakeet_mlx.audio import hanning

    original_dtype = audio.dtype
    x = audio
    if preprocessor.pad_to > 0 and x.shape[-1] < preprocessor.pad_to:
        x = mx.pad(
            x,
            ((0, preprocessor.pad_to - x.shape[-1]),),
            constant_values=preprocessor.pad_value,
        )
    if preprocessor.preemph is not None:
        x = mx.concat([x[:1], x[1:] - preprocessor.preemph * x[:-1]], axis=0)

    window = hanning(preprocessor.win_length).astype(x.dtype)
    window_padding = preprocessor.n_fft - preprocessor.win_length
    left = window_padding // 2
    window = mx.pad(window, ((left, window_padding - left),))

    reflection = preprocessor.n_fft // 2
    x = mx.concatenate(
        [
            x[1 : reflection + 1][::-1],
            x,
            x[-(reflection + 1) : -1][::-1],
        ]
    )
    frame_count = (
        x.size - preprocessor.win_length + preprocessor.hop_length
    ) // preprocessor.hop_length
    frames = mx.as_strided(
        x,
        shape=(frame_count, preprocessor.n_fft),
        strides=(preprocessor.hop_length, 1),
    )
    spectrum = mx.fft.rfft(frames * window)

    # Preserve the published MLX checkpoint's established magnitude and
    # normalization convention; isolated arms showed window placement was the
    # useful parity fix, while changing all frontend conventions together was
    # neutral-to-worse.
    components = mx.abs(mx.view(spectrum, original_dtype))
    magnitude = components[..., ::2] + components[..., 1::2]
    if preprocessor.mag_power != 1.0:
        magnitude = mx.power(magnitude, preprocessor.mag_power)
    mel = mx.matmul(preprocessor._filterbanks.astype(magnitude.dtype), magnitude.T)
    mel = mx.log(mel + 1e-5)
    if preprocessor.normalize == "per_feature":
        mean = mx.mean(mel, axis=1, keepdims=True)
        std = mx.std(mel, axis=1, keepdims=True)
    else:
        mean = mx.mean(mel)
        std = mx.std(mel)
    mel = ((mel - mean) / (std + 1e-5)).T
    return mx.expand_dims(mel, axis=0).astype(original_dtype)


def _generate_centered(model, audio_path: Path):
    import mlx.core as mx
    from parakeet_mlx.audio import load_audio

    audio = load_audio(
        audio_path,
        model.preprocessor_config.sample_rate,
        mx.float32,
    )
    mel = _centered_logmel(audio, model.preprocessor_config)
    return model.generate(mel)[0]


def _clear_mlx_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()


def transcribe_mlx(audio_paths: list[str | Path], repo_id: str) -> list[str]:
    model = _load_model(str(repo_id))
    texts: list[str] = []
    for path in audio_paths:
        try:
            result = _generate_centered(model, Path(path))
            text = _result_to_text(result)
        finally:
            _clear_mlx_cache()
        if not text:
            raise RuntimeError(f"MLX runtime produced an empty transcript for {path}")
        texts.append(text)
    return texts
