from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import numpy as np
import pytest
import soundfile as sf

import omi_stt
from omi_stt import cli


def test_package_version_matches_project_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    assert omi_stt.__version__ == project["version"]


def _wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    audio = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    sf.write(path, audio, sample_rate)


def test_default_runtime_mac_arm_prefers_mlx(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")

    assert cli._default_runtime() == "mlx"


def test_default_mlx_model_is_q8() -> None:
    args = cli.build_parser().parse_args(["transcribe", "audio.wav", "--runtime", "mlx"])

    runtime, model = cli._resolve_runtime_and_model(args)

    assert runtime == "mlx"
    assert model == "omi-health/omi-med-stt-v1-mlx-q8"


def test_default_runtime_nvidia_prefers_nemo(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cli, "_has_nvidia_gpu", lambda: True)

    assert cli._default_runtime() == "nemo"


def test_default_runtime_cpu_prefers_cpp(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cli.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(cli, "_has_nvidia_gpu", lambda: False)

    assert cli._default_runtime() == "cpp"


def test_windows_cpp_default_max_seconds_is_safer(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")

    assert cli._effective_max_seconds("cpp", 240.0) == 120.0
    assert cli._effective_chunk_seconds("cpp", 240.0, 300.0) == 120.0
    assert cli._effective_max_seconds("cpp", 60.0) == 60.0
    assert cli._effective_chunk_seconds("cpp", 60.0, 300.0) == 60.0
    assert cli._effective_max_seconds("nemo", 240.0) == 240.0


def test_non_windows_cpp_long_files_use_180s_chunks(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")

    assert cli._effective_max_seconds("cpp", 240.0) == 240.0
    assert cli._effective_chunk_seconds("cpp", 240.0, 300.0) == 180.0
    assert cli._effective_chunk_seconds("cpp", 240.0, 200.0) == 240.0
    assert cli._effective_chunk_seconds("cpp", 120.0, 300.0) == 120.0
    assert cli._effective_chunk_seconds("mlx", 240.0, 300.0) == 240.0


def test_cpp_long_audio_uses_pcm_chunks(monkeypatch, tmp_path, capsys) -> None:
    wav = tmp_path / "long.wav"
    _wav(wav, seconds=3)

    monkeypatch.setattr(cli, "_resolve_runtime_and_model", lambda args: ("cpp", "repo"))

    def fake_pcm_chunks(audio, model, **kwargs):
        assert Path(audio) == wav
        assert model == "repo"
        assert kwargs["chunk_seconds"] == 1.0
        return "merged transcript", [{"index": 0, "start": 0.0, "duration": 1.0, "transcript": "merged"}]

    import omi_stt.cpp_runtime as cpp_runtime

    monkeypatch.setattr(cpp_runtime, "transcribe_cpp_pcm_chunks", fake_pcm_chunks)

    args = cli.build_parser().parse_args([
        "transcribe",
        str(wav),
        "--runtime",
        "cpp",
        "--max-seconds",
        "1",
        "--json",
    ])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["auto_chunked"] is True
    assert payload["transcript"] == "merged transcript"
    assert payload["chunks"][0]["start"] == 0.0


def test_version_flag_prints_version_and_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "omi-med-stt" in out
    assert "default runtime:" in out


def test_short_version_flag_works(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["-V"])

    assert excinfo.value.code == 0
    assert "omi-med-stt" in capsys.readouterr().out


def test_hf_access_error_is_actionable(monkeypatch, tmp_path, capsys) -> None:
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    wav = tmp_path / "audio.wav"
    _wav(wav, seconds=1)

    monkeypatch.setattr(cli, "_resolve_runtime_and_model", lambda args: ("mlx", "omi-health/private"))

    response = httpx.Response(status_code=404, request=httpx.Request("GET", "https://hf.co/x"))

    def fake_runtime(*_args, **_kwargs):
        raise RepositoryNotFoundError("404 Client Error: repo not found", response=response)

    monkeypatch.setattr(cli, "_runtime_transcribe", fake_runtime)

    args = cli.build_parser().parse_args(["transcribe", str(wav), "--runtime", "mlx"])
    assert args.func(args) == 2

    err = capsys.readouterr().err
    assert "Cannot access omi-health/private" in err
    assert "huggingface-cli login" in err
    assert "HF_TOKEN" in err
    assert "repo not found" in err


def test_nemo_driver_too_old_error_is_actionable(monkeypatch, tmp_path, capsys) -> None:
    wav = tmp_path / "audio.wav"
    _wav(wav, seconds=1)

    monkeypatch.setattr(cli, "_resolve_runtime_and_model", lambda args: ("nemo", "omi-health/omi-med-stt-v1"))

    def fake_runtime(*_args, **_kwargs):
        raise RuntimeError(
            "CUDA error: the provided PTX was compiled with an unsupported toolchain; "
            "the NVIDIA driver on your system is too old"
        )

    monkeypatch.setattr(cli, "_runtime_transcribe", fake_runtime)

    args = cli.build_parser().parse_args(["transcribe", str(wav), "--runtime", "nemo"])
    assert args.func(args) == 2

    err = capsys.readouterr().err
    assert "NVIDIA driver is too old" in err
    assert "download.pytorch.org/whl/cu128" in err


def test_unrelated_runtime_error_propagates(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    _wav(wav, seconds=1)

    monkeypatch.setattr(cli, "_resolve_runtime_and_model", lambda args: ("mlx", "model"))

    def fake_runtime(*_args, **_kwargs):
        raise ValueError("something else broke")

    monkeypatch.setattr(cli, "_runtime_transcribe", fake_runtime)

    args = cli.build_parser().parse_args(["transcribe", str(wav), "--runtime", "mlx"])
    with pytest.raises(ValueError, match="something else broke"):
        args.func(args)
