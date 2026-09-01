from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class _Torch:
    cuda = _Cuda()
    bfloat16 = "bf16"

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def autocast(*, device_type, dtype):
        assert device_type == "cuda"
        assert dtype == "bf16"
        return contextlib.nullcontext()


class _Decoding(SimpleNamespace):
    def copy(self):
        return copy.deepcopy(self)


def _decoding() -> _Decoding:
    return _Decoding(
        strategy="checkpoint",
        greedy=SimpleNamespace(max_symbols=1, use_cuda_graph_decoder=True),
        beam=SimpleNamespace(return_best_hypothesis=False, preserve_alignments=True),
    )


def test_configure_nemo_model_applies_qualified_recipe(monkeypatch) -> None:
    from omi_stt import nemo_runtime

    omegaconf = ModuleType("omegaconf")
    omegaconf.open_dict = lambda _value: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    class Model:
        def __init__(self):
            self.cfg = SimpleNamespace(decoding=_decoding())
            self.attention = None
            self.decoding = None
            self.to_args = None
            self.eval_called = False

        def change_attention_model(self, **kwargs):
            self.attention = kwargs

        def change_decoding_strategy(self, decoding, *, verbose):
            self.decoding = decoding
            assert verbose is False

        def to(self, **kwargs):
            self.to_args = kwargs
            return self

        def eval(self):
            self.eval_called = True

    model = Model()
    configured = nemo_runtime.configure_nemo_model(model, device="cuda:0", torch=_Torch)

    assert configured is model
    assert model.attention == {
        "self_attention_model": "rel_pos_local_attn",
        "att_context_size": [256, 256],
    }
    assert model.decoding.strategy == "greedy_batch"
    assert model.decoding.greedy.max_symbols == 10
    assert model.decoding.greedy.use_cuda_graph_decoder is False
    assert model.decoding.beam.return_best_hypothesis is True
    assert model.decoding.beam.preserve_alignments is False
    assert model.to_args == {"device": "cuda:0", "dtype": "bf16"}
    assert model.eval_called is True


def test_duration_bounded_batches_honor_width_and_audio_limit() -> None:
    from omi_stt.nemo_runtime import _duration_bounded_batches

    rows = [(index, Path(f"{index}.wav"), duration) for index, duration in enumerate([20, 30, 40, 50, 60])]
    batches = list(_duration_bounded_batches(rows, size=3, max_audio_seconds=100))

    assert [[row[2] for row in batch] for batch in batches] == [
        [20, 30, 40],
        [50],
        [60],
    ]


def test_transcribe_nemo_sorts_for_batching_and_restores_input_order(monkeypatch) -> None:
    from omi_stt import nemo_runtime

    paths = [Path("long.wav"), Path("short.wav"), Path("medium.wav")]
    durations = {"long.wav": 30.0, "short.wav": 10.0, "medium.wav": 20.0}
    calls = []

    class Model:
        def transcribe(self, actual_paths, **kwargs):
            calls.append((actual_paths, kwargs))
            return [SimpleNamespace(text=f"text-{Path(path).stem}") for path in actual_paths]

    monkeypatch.setattr(nemo_runtime, "_torch_module", lambda: _Torch)
    monkeypatch.setattr(nemo_runtime, "load_nemo_model", lambda **_kwargs: Model())
    monkeypatch.setattr(nemo_runtime, "_audio_duration", lambda path: durations[path.name])

    result = nemo_runtime.transcribe_nemo(paths, "omi-health/omi-med-stt-v1", device="cuda:0")

    assert result == ["text-long", "text-short", "text-medium"]
    assert calls == [
        (
            ["short.wav", "medium.wav", "long.wav"],
            {"batch_size": 3, "timestamps": False, "verbose": False},
        )
    ]


def test_nemo_runtime_rejects_non_cuda_device() -> None:
    from omi_stt.nemo_runtime import _require_bf16_cuda

    try:
        _require_bf16_cuda(_Torch, "cpu")
    except RuntimeError as exc:
        assert "requires an NVIDIA CUDA GPU" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("CPU device unexpectedly accepted by the NeMo GPU runtime")
