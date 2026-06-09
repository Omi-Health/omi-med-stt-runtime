from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from omi_stt import cpp_runtime


def test_cmake_flags_use_native_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMI_MED_STT_GGML_NATIVE", raising=False)

    assert "-DGGML_NATIVE=OFF" not in cpp_runtime._cmake_backend_flags("cpu")


def test_cmake_flags_allow_portable_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMI_MED_STT_GGML_NATIVE", "OFF")

    assert "-DGGML_NATIVE=OFF" in cpp_runtime._cmake_backend_flags("cpu")


def test_backend_flags() -> None:
    assert "-DPARAKEET_GGML_METAL=ON" in cpp_runtime._cmake_backend_flags("metal")
    assert "-DPARAKEET_GGML_CUDA=ON" in cpp_runtime._cmake_backend_flags("cuda")
    assert "-DPARAKEET_GGML_VULKAN=ON" in cpp_runtime._cmake_backend_flags("vulkan")
    assert "-DPARAKEET_GGML_HIP=ON" in cpp_runtime._cmake_backend_flags("hip")


def test_backend_flags_reject_unknown() -> None:
    with pytest.raises(RuntimeError, match="Unsupported parakeet.cpp backend"):
        cpp_runtime._cmake_backend_flags("npu")


def test_cached_library_name_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "linux")
    assert cpp_runtime._cached_parakeet_lib().name == "libparakeet.so"

    monkeypatch.setattr(cpp_runtime.sys, "platform", "darwin")
    assert cpp_runtime._cached_parakeet_lib().name == "libparakeet.dylib"

    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    assert cpp_runtime._cached_parakeet_lib().name == "parakeet.dll"


def test_windows_cached_paths_include_multiconfig_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")

    cli_names = [str(path).replace("\\", "/") for path in cpp_runtime._cached_parakeet_cli_candidates("cpu")]
    lib_names = [str(path).replace("\\", "/") for path in cpp_runtime._cached_parakeet_lib_candidates("cpu")]

    assert any(name.endswith("examples/cli/Release/parakeet-cli.exe") for name in cli_names)
    assert any(name.endswith("bin/Release/parakeet-cli.exe") for name in cli_names)
    assert any(name.endswith("Release/parakeet.dll") for name in lib_names)
    assert any(name.endswith("bin/Release/parakeet.dll") for name in lib_names)
    assert any(name.endswith("Release/libparakeet.dll") for name in lib_names)


def test_native_library_directories_include_library_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    build = tmp_path / "build"
    lib_parent = build / "bin" / "Release"
    ggml_src = build / "third_party" / "ggml" / "src"
    lib_parent.mkdir(parents=True)
    ggml_src.mkdir(parents=True)

    monkeypatch.setattr(cpp_runtime, "_shared_build_dir", lambda _backend="auto": build)

    dirs = cpp_runtime._native_library_directories(lib_parent / "parakeet.dll")

    assert lib_parent in dirs
    assert ggml_src in dirs


def test_unix_preloads_bundled_ggml_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "linux")
    lib_parent = tmp_path / "build"
    ggml_src = lib_parent / "third_party" / "ggml" / "src"
    ggml_src.mkdir(parents=True)
    lib_path = lib_parent / "libparakeet.so"
    lib_path.write_bytes(b"")
    dep = ggml_src / "libggml.so.0"
    dep.write_bytes(b"")

    loaded = []

    class FakeCDLL:
        def __init__(self, path, mode=None):
            loaded.append((Path(path), mode))

    monkeypatch.setattr(cpp_runtime.ctypes, "CDLL", FakeCDLL)
    monkeypatch.setattr(cpp_runtime.ctypes, "RTLD_GLOBAL", 256, raising=False)
    monkeypatch.setattr(cpp_runtime, "_shared_build_dir", lambda _backend="auto": lib_parent)

    handles = cpp_runtime._preload_unix_shared_dependencies(lib_path)

    assert handles
    assert any(path == dep and mode == 256 for path, mode in loaded)


def test_backend_specific_install_dirs_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "darwin")

    cpu = cpp_runtime._cpp_install_dir("cpu")
    auto = cpp_runtime._cpp_install_dir("auto")

    assert auto == cpu
    assert cpu.name == "cpu"


def test_prebuilt_asset_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    assert cpp_runtime._prebuilt_bundle_asset("cpu") == "omi-med-stt-parakeet-cpp-windows-x86_64-cpu.zip"

    monkeypatch.setattr(cpp_runtime.sys, "platform", "linux")
    monkeypatch.setattr(cpp_runtime.os, "uname", lambda: SimpleNamespace(machine="x86_64"), raising=False)
    monkeypatch.delenv("PROCESSOR_ARCHITECTURE", raising=False)
    assert cpp_runtime._prebuilt_bundle_asset("cpu") == "omi-med-stt-parakeet-cpp-linux-x86_64-cpu.zip"

    assert cpp_runtime._prebuilt_bundle_asset("cuda") is None


def test_prebuilt_download_extracts_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.setattr(cpp_runtime, "_cpp_install_dir", lambda _backend="auto": tmp_path / "install")

    source_zip = tmp_path / "bundle.zip"
    with zipfile.ZipFile(source_zip, "w") as zf:
        zf.writestr("build/Release/parakeet.dll", b"dll")

    def fake_download_url(_url, destination, **_kwargs):
        Path(destination).write_bytes(source_zip.read_bytes())

    monkeypatch.setattr(cpp_runtime, "_download_url", fake_download_url)

    path = cpp_runtime._download_prebuilt_parakeet_cpp(backend="cpu")

    assert path == tmp_path / "install" / "build" / "Release" / "parakeet.dll"


def test_install_can_require_prebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "linux")
    monkeypatch.setattr(cpp_runtime.os, "uname", lambda: SimpleNamespace(machine="x86_64"), raising=False)
    monkeypatch.setattr(cpp_runtime, "_cpp_install_dir", lambda _backend="auto": tmp_path / "install")
    monkeypatch.setattr(cpp_runtime, "_download_prebuilt_parakeet_cpp", lambda **_kwargs: None)
    monkeypatch.setenv("OMI_MED_STT_CPP_REQUIRE_PREBUILT", "1")

    with pytest.raises(RuntimeError, match="Required prebuilt"):
        cpp_runtime.install_parakeet_cpp(backend="cpu")


def test_windows_build_uses_shared_library_without_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")

    assert not cpp_runtime._builds_parakeet_cli()
    assert cpp_runtime._cmake_windows_export_flags() == ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"]
    assert cpp_runtime._cmake_build_command(tmp_path) == [
        "cmake",
        "--build",
        str(tmp_path),
        "--config",
        "Release",
        "--parallel",
    ]


def test_windows_cached_dll_skips_rebuild_without_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cpp_runtime, "_cpp_install_dir", lambda _backend="auto": tmp_path)
    lib = tmp_path / "build" / "Release" / "parakeet.dll"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"dll")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("cached Windows DLL should not rebuild")

    monkeypatch.setattr(cpp_runtime, "_run", fail_run)

    assert cpp_runtime.install_parakeet_cpp(backend="cpu") == lib


def test_windows_install_rescans_multiconfig_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cpp_runtime, "_cpp_install_dir", lambda _backend="auto": tmp_path)
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: f"C:/bin/{name}.exe")
    monkeypatch.setattr(cpp_runtime, "_adapter_patch_path", lambda: tmp_path / "patch.diff")

    src = tmp_path / "src"
    build = tmp_path / "build"
    release_lib = build / "Release" / "parakeet.dll"

    def fake_run(cmd, *, cwd=None):
        if cmd[0] == "git" and cmd[1] == "clone":
            (src / "src").mkdir(parents=True)
        if cmd[0] == "cmake" and "--build" in cmd:
            release_lib.parent.mkdir(parents=True)
            release_lib.write_bytes(b"dll")

    monkeypatch.setattr(cpp_runtime, "_run", fake_run)
    monkeypatch.setattr(cpp_runtime, "_apply_windows_compat_patches", lambda _src: None)

    assert cpp_runtime.install_parakeet_cpp(backend="cpu", force=True) == release_lib


def test_non_windows_builds_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "linux")

    assert cpp_runtime._builds_parakeet_cli()
    assert cpp_runtime._cmake_windows_export_flags() == []
    assert cpp_runtime._cmake_build_command(tmp_path) == ["cmake", "--build", str(tmp_path), "-j"]


def test_windows_cpu_default_threads_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpp_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cpp_runtime.os, "cpu_count", lambda: 16)
    monkeypatch.delenv("OMI_MED_STT_CPP_THREADS", raising=False)

    assert cpp_runtime._default_cpp_threads("cpu") == 8

    monkeypatch.setattr(cpp_runtime.os, "cpu_count", lambda: 4)
    assert cpp_runtime._default_cpp_threads("cpu") == 4

    monkeypatch.setenv("OMI_MED_STT_CPP_THREADS", "3")
    assert cpp_runtime._default_cpp_threads("cpu") == 3


def test_windows_compat_patch_defines_m_pi(tmp_path: Path) -> None:
    src = tmp_path / "src" / "src"
    src.mkdir(parents=True)
    mel_gpu = src / "mel_gpu.cpp"
    fft = src / "fft.cpp"
    for cpp in (mel_gpu, fft):
        cpp.write_text(
            "#include <cmath>\n\nstatic const float two_pi = 2.0f * M_PI;\n",
            encoding="utf-8",
        )

    cpp_runtime._apply_windows_compat_patches(tmp_path / "src")

    for cpp in (mel_gpu, fft):
        patched = cpp.read_text(encoding="utf-8")
        assert "#define M_PI 3.14159265358979323846" in patched
        assert patched.count("#define M_PI") == 1


def test_transcribe_cpp_uses_capi_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)
    monkeypatch.setattr(cpp_runtime, "_transcribe_cpp_capi", lambda paths, *_args, **_kwargs: [f"capi:{Path(paths[0]).name}"])

    out = cpp_runtime.transcribe_cpp([audio], model_id_or_path=str(model))

    assert out == ["capi:audio.wav"]


def test_transcribe_cpp_capi_uses_pcm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    lib = tmp_path / "libparakeet.so"
    lib.write_bytes(b"fake")
    calls = []

    class FakeCAPI:
        def __init__(self, *_args):
            calls.append(("init", _args[-1]))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def transcribe_pcm(self, pcm):
            calls.append(pcm)
            return "pcm transcript"

    monkeypatch.setattr(cpp_runtime, "_find_parakeet_lib", lambda **_kwargs: lib)
    monkeypatch.setattr(cpp_runtime, "_ParakeetCAPI", FakeCAPI)
    monkeypatch.setattr(cpp_runtime, "np", cpp_runtime.np)

    from omi_stt import audio as audio_mod

    monkeypatch.setattr(audio_mod, "read_audio", lambda _path: (cpp_runtime.np.array([0.1, 0.2], dtype=cpp_runtime.np.float32), 16000))

    out = cpp_runtime._transcribe_cpp_capi([audio], model, auto_install=False, cpp_backend="cpu", decoder="tdt", threads=4)

    assert out == ["pcm transcript"]
    assert len(calls) == 2
    assert calls[0] == ("init", 4)
    assert calls[1].dtype == cpp_runtime.np.float32


def test_transcribe_cpp_pcm_chunks_uses_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    lib = tmp_path / "libparakeet.so"
    lib.write_bytes(b"fake")
    calls = []

    class FakeCAPI:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def transcribe_pcm_batch(self, batch):
            calls.append(("batch", len(batch)))
            return [f"chunk {i}" for i in range(len(batch))]

        def transcribe_pcm(self, _pcm):
            calls.append(("single", 1))
            return "single"

    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)
    monkeypatch.setattr(cpp_runtime, "_find_parakeet_lib", lambda **_kwargs: lib)
    monkeypatch.setattr(cpp_runtime, "_ParakeetCAPI", FakeCAPI)

    from omi_stt import audio as audio_mod

    fake_audio = cpp_runtime.np.zeros(16000 * 5, dtype=cpp_runtime.np.float32)
    monkeypatch.setattr(audio_mod, "read_audio", lambda _path: (fake_audio, 16000))
    monkeypatch.setenv("OMI_MED_STT_CPP_BATCH_CHUNKS", "1")

    transcript, chunks = cpp_runtime.transcribe_cpp_pcm_chunks(
        audio,
        model_id_or_path=str(model),
        chunk_seconds=2.0,
        overlap=0.0,
        auto_install=False,
        cpp_backend="cpu",
    )

    assert calls == [("batch", 3)]
    assert transcript
    assert len(chunks) == 3
    assert chunks[0]["transcript"] == "chunk 0"


def test_transcribe_cpp_pcm_chunks_uses_sequential_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    lib = tmp_path / "libparakeet.so"
    lib.write_bytes(b"fake")
    calls = []

    class FakeCAPI:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def transcribe_pcm_batch(self, _batch):
            calls.append(("batch", 1))
            return ["batch"]

        def transcribe_pcm(self, _pcm):
            calls.append(("single", 1))
            return f"single {len(calls)}"

    monkeypatch.delenv("OMI_MED_STT_CPP_BATCH_CHUNKS", raising=False)
    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)
    monkeypatch.setattr(cpp_runtime, "_find_parakeet_lib", lambda **_kwargs: lib)
    monkeypatch.setattr(cpp_runtime, "_ParakeetCAPI", FakeCAPI)

    from omi_stt import audio as audio_mod

    fake_audio = cpp_runtime.np.zeros(16000 * 5, dtype=cpp_runtime.np.float32)
    monkeypatch.setattr(audio_mod, "read_audio", lambda _path: (fake_audio, 16000))

    _transcript, chunks = cpp_runtime.transcribe_cpp_pcm_chunks(
        audio,
        model_id_or_path=str(model),
        chunk_seconds=2.0,
        overlap=0.0,
        auto_install=False,
        cpp_backend="cpu",
    )

    assert calls == [("single", 1), ("single", 1), ("single", 1)]
    assert len(chunks) == 3


def test_transcribe_cpp_falls_back_to_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    cli = tmp_path / cpp_runtime._exe_name("parakeet-cli")
    cli.write_bytes(b"fake executable")

    monkeypatch.delenv("OMI_MED_STT_CPP_REQUIRE_CAPI", raising=False)
    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)
    monkeypatch.setattr(cpp_runtime, "_find_parakeet_cli", lambda *args, **kwargs: str(cli))

    def fail_capi(*_args, **_kwargs):
        raise RuntimeError("no capi")

    monkeypatch.setattr(cpp_runtime, "_transcribe_cpp_capi", fail_capi)

    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="fallback transcript\n")

    monkeypatch.setattr(cpp_runtime.subprocess, "run", fake_run)

    out = cpp_runtime.transcribe_cpp([audio], model_id_or_path=str(model))

    assert out == ["fallback transcript"]
    assert commands[0][0] == str(cli)
    assert commands[0][1:3] == ["transcribe", "--model"]


def test_transcribe_cpp_warns_on_capi_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    cli = tmp_path / cpp_runtime._exe_name("parakeet-cli")
    cli.write_bytes(b"fake executable")

    monkeypatch.delenv("OMI_MED_STT_CPP_REQUIRE_CAPI", raising=False)
    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)
    monkeypatch.setattr(cpp_runtime, "_find_parakeet_cli", lambda *args, **kwargs: str(cli))

    def fail_capi(*_args, **_kwargs):
        raise OSError("WinError 193 not a valid Win32 application")

    monkeypatch.setattr(cpp_runtime, "_transcribe_cpp_capi", fail_capi)
    monkeypatch.setattr(
        cpp_runtime.subprocess,
        "run",
        lambda cmd, **_kwargs: SimpleNamespace(stdout="ok\n"),
    )

    out = cpp_runtime.transcribe_cpp([audio], model_id_or_path=str(model))

    assert out == ["ok"]
    err = capsys.readouterr().err
    assert "C API load failed" in err
    assert "OSError" in err
    assert "falling back to parakeet-cli subprocess" in err


def test_transcribe_cpp_combines_errors_when_both_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    monkeypatch.delenv("OMI_MED_STT_CPP_REQUIRE_CAPI", raising=False)
    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)

    def fail_capi(*_args, **_kwargs):
        raise OSError("WinError 193 not a valid Win32 application")

    def fail_cli(*_args, **_kwargs):
        raise RuntimeError("parakeet-cli not found")

    monkeypatch.setattr(cpp_runtime, "_transcribe_cpp_capi", fail_capi)
    monkeypatch.setattr(cpp_runtime, "_find_parakeet_cli", fail_cli)

    with pytest.raises(RuntimeError) as excinfo:
        cpp_runtime.transcribe_cpp([audio], model_id_or_path=str(model))

    message = str(excinfo.value)
    assert "parakeet-cli not found" in message
    assert "WinError 193" in message
    assert "original C API load error" in message


def test_transcribe_cpp_require_capi_reraises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    monkeypatch.setenv("OMI_MED_STT_CPP_REQUIRE_CAPI", "1")
    monkeypatch.setattr(cpp_runtime, "_download_or_resolve_gguf", lambda *args, **kwargs: model)

    def fail_capi(*_args, **_kwargs):
        raise RuntimeError("capi specific failure")

    monkeypatch.setattr(cpp_runtime, "_transcribe_cpp_capi", fail_capi)

    with pytest.raises(RuntimeError, match="capi specific failure"):
        cpp_runtime.transcribe_cpp([audio], model_id_or_path=str(model))


def test_download_or_resolve_gguf_survives_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cpp_runtime._download_or_resolve_gguf.cache_clear()

    def boom(self) -> bool:
        raise PermissionError("CWD is not traversable")

    monkeypatch.setattr(cpp_runtime.Path, "exists", boom)

    downloaded = {}

    def fake_hf_hub_download(**kwargs):
        downloaded.update(kwargs)
        return str(tmp_path / "model.gguf")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(cpp_runtime, "_verify_known_gguf", lambda *args, **kwargs: None)

    result = cpp_runtime._download_or_resolve_gguf(
        "omi-health/omi-med-stt-v1-gguf",
        "model.gguf",
        "rev",
        False,
    )

    assert result == tmp_path / "model.gguf"
    assert downloaded["repo_id"] == "omi-health/omi-med-stt-v1-gguf"
    cpp_runtime._download_or_resolve_gguf.cache_clear()
