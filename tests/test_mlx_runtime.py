from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType


def test_result_to_text_matches_nemo_unknown_token_rendering() -> None:
    from omi_stt.mlx_runtime import _result_to_text

    class Result:
        text = "I<unk>m testing <unk> tokens"

    assert _result_to_text(Result()) == "I⁇m testing ⁇ tokens"


def test_transcribe_mlx_uses_centered_frontend(monkeypatch, tmp_path) -> None:
    from omi_stt import mlx_runtime

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not-read-by-this-unit-test")
    model = object()
    calls = []

    class Result:
        text = "centered result"

    monkeypatch.setattr(mlx_runtime, "_load_model", lambda _repo: model)

    def fake_generate_centered(actual_model, path):
        calls.append((actual_model, path))
        return Result()

    monkeypatch.setattr(mlx_runtime, "_generate_centered", fake_generate_centered)
    cache_clears = []
    monkeypatch.setattr(mlx_runtime, "_clear_mlx_cache", lambda: cache_clears.append(True))

    assert mlx_runtime.transcribe_mlx([audio], "local-model") == ["centered result"]
    assert calls == [(model, audio)]
    assert cache_clears == [True]


def test_quantized_mlx_config_quantizes_before_loading(monkeypatch, tmp_path) -> None:
    from omi_stt import mlx_runtime

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "target": "dummy-target",
                "encoder": {"d_model": 1024},
                "quantization": {"bits": 8, "group_size": 64},
            }
        )
    )

    calls: list[tuple[str, object]] = []

    class FakeArray:
        def astype(self, dtype):
            calls.append(("astype", dtype))
            return self

    class FakeModel:
        def __init__(self):
            self.quantized = False
            self.loaded_after_quantize = None

        def load_weights(self, path):
            self.loaded_after_quantize = self.quantized
            calls.append(("load_weights", path))

        def parameters(self):
            return {"weight": FakeArray()}

        def update(self, _params):
            calls.append(("update", True))

        def eval(self):
            calls.append(("eval", True))

    fake_model = FakeModel()

    core = ModuleType("mlx.core")
    core.bfloat16 = "bfloat16"
    nn = ModuleType("mlx.nn")

    def fake_quantize(model, *, bits, group_size):
        model.quantized = True
        calls.append(("quantize", (bits, group_size)))

    nn.quantize = fake_quantize
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda params: list(params.items())
    utils.tree_unflatten = lambda items: dict(items)

    mlx = ModuleType("mlx")
    mlx.core = core
    mlx.nn = nn

    parakeet_mlx = ModuleType("parakeet_mlx")
    parakeet_utils = ModuleType("parakeet_mlx.utils")

    def fake_from_config(config):
        assert "quantization" not in config
        assert config["encoder"]["self_attention_model"] == "rel_pos_local_attn"
        assert config["encoder"]["att_context_size"] == [256, 256]
        calls.append(("from_config", config))
        return fake_model

    parakeet_utils.from_config = fake_from_config

    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", parakeet_mlx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.utils", parakeet_utils)

    mlx_runtime._load_model.cache_clear()
    loaded = mlx_runtime._load_model(str(model_dir))
    mlx_runtime._load_model.cache_clear()

    assert loaded is fake_model
    assert ("quantize", (8, 64)) in calls
    assert fake_model.loaded_after_quantize is True
    assert not any(call[0] == "astype" for call in calls)
    assert not any(call[0] == "update" for call in calls)


def test_resolve_model_survives_permission_error(monkeypatch) -> None:
    from omi_stt import mlx_runtime

    def boom(self) -> bool:
        raise PermissionError("CWD is not traversable")

    monkeypatch.setattr(mlx_runtime.Path, "exists", boom)

    downloaded = []

    def fake_hf_hub_download(repo_id, filename, token=None):
        downloaded.append(filename)
        return f"/cache/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    config_path, weights_path = mlx_runtime._download_or_resolve_model("omi-health/omi-med-stt-v1-mlx-q8")

    assert config_path == Path("/cache/config.json")
    assert weights_path == Path("/cache/model.safetensors")
    assert downloaded == ["config.json", "model.safetensors"]
