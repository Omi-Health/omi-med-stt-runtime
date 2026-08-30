from __future__ import annotations

import ctypes
import json
import os
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .hf_auth import hf_token


DEFAULT_GGUF_REPO = "omi-health/omi-med-stt-v1-gguf"
DEFAULT_GGUF_FILE = "omi-med-stt-v1-q8_0.gguf"
DEFAULT_GGUF_REVISION = "c78333664b14e8dd8e47ed063be9692ed9054390"
PARAKEET_CPP_REPO = "https://github.com/mudler/parakeet.cpp"
PARAKEET_CPP_COMMIT = "b11fe5bca78ad8b342dd559a43d76df3984bb447"
PARAKEET_CPP_PATCH_VERSION = "omi-med-adapter-v2"
PARAKEET_CPP_BUNDLE_RELEASE = f"parakeet-cpp-{PARAKEET_CPP_COMMIT[:8]}-{PARAKEET_CPP_PATCH_VERSION}"
PARAKEET_CPP_BUNDLE_BASE_URL = (
    "https://github.com/Omi-Health/omi-med-stt-runtime/releases/download/"
    f"{PARAKEET_CPP_BUNDLE_RELEASE}"
)
KNOWN_GGUF_SHA256 = {
    "omi-med-stt-v1-f16.gguf": "813fb59fdaae8784203685a7607d29f6f12202d7606ef49f5932f72a0ee04f86",
    "omi-med-stt-v1-q8_0.gguf": "c4f364a730df7aa9bb0714cda1b1ad5e3104331db9919bb0e2a379d0fb64dbab",
}


def _cache_root() -> Path:
    explicit = os.environ.get("OMI_MED_STT_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "omi-med-stt"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "omi-med-stt"


def _exe_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def _resolve_backend(backend: str) -> str:
    return "cpu" if backend == "auto" else backend


def _cpp_install_dir(backend: str = "auto") -> Path:
    return _cache_root() / "parakeet.cpp" / PARAKEET_CPP_COMMIT / PARAKEET_CPP_PATCH_VERSION / _resolve_backend(backend)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _platform_tag() -> str | None:
    if sys.platform == "win32":
        machine = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()
    elif hasattr(os, "uname"):
        machine = os.uname().machine.lower()
    else:
        machine = ""
    if machine in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        arch = "arm64"
    else:
        return None
    if sys.platform == "win32":
        return f"windows-{arch}"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    return None


def _prebuilt_bundle_asset(backend: str) -> str | None:
    backend = _resolve_backend(backend)
    if backend != "cpu":
        return None
    tag = _platform_tag()
    if tag not in {"windows-x86_64", "linux-x86_64"}:
        return None
    return f"omi-med-stt-parakeet-cpp-{tag}-cpu.zip"


def _github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _download_url(url: str, destination: Path, token: str | None = None, *, accept: str | None = None) -> None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_github_release_asset(asset: str, destination: Path) -> bool:
    token = _github_token()
    if not token:
        return False
    api_url = (
        "https://api.github.com/repos/Omi-Health/omi-med-stt-runtime/releases/tags/"
        f"{PARAKEET_CPP_BUNDLE_RELEASE}"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="omi_med_stt_release_") as tmp:
            meta_path = Path(tmp) / "release.json"
            _download_url(api_url, meta_path, token=token, accept="application/vnd.github+json")
            release = json.loads(meta_path.read_text(encoding="utf-8"))
        for item in release.get("assets", []):
            if item.get("name") == asset and item.get("url"):
                _download_url(
                    item["url"],
                    destination,
                    token=token,
                    accept="application/octet-stream",
                )
                return True
    except Exception:
        return False
    return False


def _download_prebuilt_parakeet_cpp(*, backend: str = "auto") -> Path | None:
    if os.environ.get("OMI_MED_STT_CPP_NO_PREBUILT", "").upper() in {"1", "TRUE", "ON", "YES"}:
        return None
    backend = _resolve_backend(backend)
    asset = _prebuilt_bundle_asset(backend)
    if not asset:
        return None

    install_dir = _cpp_install_dir(backend)
    install_dir.mkdir(parents=True, exist_ok=True)
    url = f"{PARAKEET_CPP_BUNDLE_BASE_URL}/{asset}"
    with tempfile.TemporaryDirectory(prefix="omi_med_stt_cpp_bundle_") as tmp:
        archive = Path(tmp) / asset
        try:
            _download_url(url, archive)
        except Exception:
            if not _download_github_release_asset(asset, archive):
                return None
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(install_dir)
    if sys.platform != "win32":
        for candidate in _cached_parakeet_cli_candidates(backend):
            if candidate.exists():
                candidate.chmod(candidate.stat().st_mode | 0o111)

    lib = _cached_parakeet_lib(backend)
    cli = _cached_parakeet_cli(backend)
    cli_ready = cli.exists() and os.access(cli, os.X_OK) if _builds_parakeet_cli() else True
    if cli_ready and lib.exists():
        return cli if _builds_parakeet_cli() else lib
    return None


def _requires_prebuilt_parakeet_cpp() -> bool:
    return os.environ.get("OMI_MED_STT_CPP_REQUIRE_PREBUILT", "").upper() in {"1", "TRUE", "ON", "YES"}


def _cached_parakeet_cli_candidates(backend: str = "auto") -> list[Path]:
    build = _cpp_install_dir(backend) / "build"
    exe = _exe_name("parakeet-cli")
    candidates = [
        build / "examples" / "cli" / exe,
        build / "examples" / "cli" / "Release" / exe,
        build / "examples" / "cli" / "RelWithDebInfo" / exe,
        build / "bin" / exe,
        build / "bin" / "Release" / exe,
        build / "Release" / exe,
    ]
    return candidates


def _cached_parakeet_cli(backend: str = "auto") -> Path:
    return _first_existing(_cached_parakeet_cli_candidates(backend))


def _shared_build_dir(backend: str = "auto") -> Path:
    return _cpp_install_dir(backend) / "build"


def _cached_parakeet_lib_candidates(backend: str = "auto") -> list[Path]:
    build = _shared_build_dir(backend)
    if sys.platform == "darwin":
        names = ["libparakeet.dylib"]
    elif sys.platform == "win32":
        names = ["parakeet.dll", "libparakeet.dll"]
    else:
        names = ["libparakeet.so"]

    dirs = [
        build,
        build / "Release",
        build / "RelWithDebInfo",
        build / "bin",
        build / "bin" / "Release",
        build / "bin" / "RelWithDebInfo",
        build / "src",
        build / "src" / "Release",
        build / "src" / "RelWithDebInfo",
    ]
    return [directory / name for directory in dirs for name in names]


def _cached_parakeet_lib(backend: str = "auto") -> Path:
    return _first_existing(_cached_parakeet_lib_candidates(backend))


def _adapter_patch_path() -> Path:
    return Path(str(resources.files("omi_stt.resources").joinpath("parakeet-cpp-omi-adapter.patch")))


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _apply_windows_compat_patches(src_dir: Path) -> None:
    for cpp_file in (src_dir / "src").glob("*.cpp"):
        text = cpp_file.read_text(encoding="utf-8")
        if "M_PI" not in text or "#define M_PI" in text:
            continue

        lines = text.splitlines()
        insert_at = 0
        while insert_at < len(lines) and (
            lines[insert_at].startswith("#include") or not lines[insert_at].strip()
        ):
            insert_at += 1
        lines.insert(
            insert_at,
            "\n#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif",
        )
        cpp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmake_backend_flags(backend: str) -> list[str]:
    backend = _resolve_backend(backend)
    flags = ["-DPARAKEET_BUILD_TESTS=OFF"]
    if os.environ.get("OMI_MED_STT_GGML_NATIVE", "ON").upper() in {"0", "FALSE", "OFF", "NO"}:
        flags.append("-DGGML_NATIVE=OFF")
    if backend == "metal":
        flags.append("-DPARAKEET_GGML_METAL=ON")
    elif backend == "cuda":
        flags.append("-DPARAKEET_GGML_CUDA=ON")
    elif backend == "vulkan":
        flags.append("-DPARAKEET_GGML_VULKAN=ON")
    elif backend == "hip":
        flags.append("-DPARAKEET_GGML_HIP=ON")
    elif backend != "cpu":
        raise RuntimeError(f"Unsupported parakeet.cpp backend: {backend}")
    return flags


def _cmake_rpath_flags(build_dir: Path) -> list[str]:
    if sys.platform == "win32":
        return []
    rpaths = [
        build_dir,
        build_dir / "third_party" / "ggml" / "src",
        build_dir / "third_party" / "ggml" / "src" / "ggml-blas",
    ]
    return [f"-DCMAKE_BUILD_RPATH={';'.join(str(p) for p in rpaths)}"]


def _cmake_windows_export_flags() -> list[str]:
    if sys.platform != "win32":
        return []
    return ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"]


def _builds_parakeet_cli() -> bool:
    return sys.platform != "win32"


def _default_cpp_threads(backend: str = "auto") -> int:
    explicit = os.environ.get("OMI_MED_STT_CPP_THREADS")
    if explicit:
        try:
            return max(0, int(explicit))
        except ValueError as exc:
            raise RuntimeError("OMI_MED_STT_CPP_THREADS must be an integer") from exc
    backend = _resolve_backend(backend)
    if sys.platform == "win32" and backend == "cpu":
        return max(1, min(os.cpu_count() or 4, 8))
    return 0


def _cmake_build_command(build_dir: Path) -> list[str]:
    if sys.platform == "win32":
        return ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"]
    return ["cmake", "--build", str(build_dir), "-j"]


def _native_library_directories(lib_path: Path, backend: str = "auto") -> list[Path]:
    build = _shared_build_dir(backend)
    dirs = [
        lib_path.parent,
        build,
        build / "Release",
        build / "RelWithDebInfo",
        build / "bin",
        build / "bin" / "Release",
        build / "bin" / "RelWithDebInfo",
        build / "third_party" / "ggml" / "src",
        build / "third_party" / "ggml" / "src" / "Release",
        build / "third_party" / "ggml" / "src" / "RelWithDebInfo",
    ]
    return [path for path in dirs if path.exists()]


def _preload_unix_shared_dependencies(lib_path: Path, backend: str = "auto") -> list[Any]:
    if sys.platform == "win32":
        return []

    handles: list[Any] = []
    for directory in _native_library_directories(lib_path, backend):
        for pattern in ("libggml-base.so*", "libggml-cpu.so*", "libggml.so*", "libggml*.dylib"):
            for candidate in sorted(directory.glob(pattern)):
                if candidate == lib_path:
                    continue
                try:
                    handles.append(ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL))
                except OSError:
                    continue
    return handles


def install_parakeet_cpp(*, force: bool = False, backend: str = "auto") -> Path:
    backend = _resolve_backend(backend)
    cli = _cached_parakeet_cli(backend)
    lib = _cached_parakeet_lib(backend)
    cli_ready = cli.exists() and os.access(cli, os.X_OK) if _builds_parakeet_cli() else True
    if cli_ready and lib.exists() and not force:
        return cli if _builds_parakeet_cli() else lib
    if not force:
        prebuilt = _download_prebuilt_parakeet_cpp(backend=backend)
        if prebuilt is not None:
            return prebuilt
        if _requires_prebuilt_parakeet_cpp():
            asset = _prebuilt_bundle_asset(backend) or f"unsupported backend/platform: {backend}"
            raise RuntimeError(f"Required prebuilt parakeet.cpp runtime bundle was not available: {asset}")
    if not shutil.which("git"):
        raise RuntimeError("Installing parakeet.cpp requires git on PATH.")
    if not shutil.which("cmake"):
        raise RuntimeError("Installing parakeet.cpp requires cmake on PATH.")

    install_dir = _cpp_install_dir(backend)
    src_dir = install_dir / "src"
    build_dir = install_dir / "build"
    install_dir.mkdir(parents=True, exist_ok=True)
    if force and install_dir.exists():
        shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        _run(["git", "clone", "--recursive", PARAKEET_CPP_REPO, str(src_dir)])
    _run(["git", "fetch", "--all", "--tags"], cwd=src_dir)
    _run(["git", "reset", "--hard"], cwd=src_dir)
    _run(["git", "checkout", PARAKEET_CPP_COMMIT], cwd=src_dir)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=src_dir)
    _run(["git", "apply", "--check", str(_adapter_patch_path())], cwd=src_dir)
    _run(["git", "apply", str(_adapter_patch_path())], cwd=src_dir)
    if sys.platform == "win32":
        _apply_windows_compat_patches(src_dir)

    configure = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        "-DPARAKEET_SHARED=ON",
        f"-DPARAKEET_BUILD_CLI={'ON' if _builds_parakeet_cli() else 'OFF'}",
        *_cmake_backend_flags(backend),
        *_cmake_rpath_flags(build_dir),
        *_cmake_windows_export_flags(),
    ]
    _run(configure)
    _run(_cmake_build_command(build_dir))
    cli = _cached_parakeet_cli(backend)
    lib = _cached_parakeet_lib(backend)
    if _builds_parakeet_cli() and not cli.exists():
        raise RuntimeError(f"parakeet.cpp build finished but {cli} was not found.")
    if not lib.exists():
        raise RuntimeError(f"parakeet.cpp build finished but {lib} was not found.")
    return cli if cli.exists() else lib


def _find_parakeet_cli(
    explicit_path: str | None = None,
    *,
    auto_install: bool = True,
    backend: str = "auto",
) -> str:
    candidates = [
        explicit_path,
        os.environ.get("OMI_MED_STT_PARAKEET_CLI"),
        str(_cached_parakeet_cli(backend)),
        shutil.which("parakeet-cli"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
    if auto_install:
        return str(install_parakeet_cpp(backend=backend))
    raise RuntimeError(
        "parakeet.cpp runtime requires a parakeet-cli binary. Run "
        "`omi-med-stt install-cpp`, pass --parakeet-cli, or set OMI_MED_STT_PARAKEET_CLI."
    )


def _find_parakeet_lib(
    *,
    auto_install: bool = True,
    backend: str = "auto",
) -> Path:
    lib = _cached_parakeet_lib(backend)
    if lib.exists():
        return lib
    if auto_install:
        install_parakeet_cpp(backend=backend)
        lib = _cached_parakeet_lib(backend)
        if lib.exists():
            return lib
    raise RuntimeError(
        "parakeet.cpp C API requires libparakeet. Run `omi-med-stt install-cpp` "
        "or disable C API with OMI_MED_STT_CPP_DISABLE_CAPI=1."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_known_gguf(path: Path, gguf_file: str, verify_checksum: bool) -> None:
    if not verify_checksum:
        return
    expected = KNOWN_GGUF_SHA256.get(gguf_file)
    if not expected:
        return
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {gguf_file}: expected {expected}, got {actual}. "
            "Use --no-verify-checksum only if you intentionally selected a different artifact."
        )


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (PermissionError, OSError):
        return False


def _safe_repo_cache_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace("\\", "--")


def _download_gguf_direct(
    repo_id: str,
    gguf_file: str,
    revision: str,
    verify_checksum: bool,
) -> Path:
    destination = (
        _cache_root()
        / "gguf"
        / _safe_repo_cache_name(repo_id)
        / revision
        / gguf_file
    )
    if destination.exists():
        _verify_known_gguf(destination, gguf_file, verify_checksum)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    token = hf_token()
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{gguf_file}"
    try:
        _download_url(url, tmp, token=token)
        tmp.replace(destination)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    _verify_known_gguf(destination, gguf_file, verify_checksum)
    return destination


@lru_cache(maxsize=4)
def _download_or_resolve_gguf(
    model_id_or_path: str,
    gguf_file: str,
    revision: str,
    verify_checksum: bool,
) -> Path:
    local = Path(model_id_or_path).expanduser()
    if _path_exists(local):
        if _path_is_file(local):
            _verify_known_gguf(local, local.name, verify_checksum)
            return local
        candidate = local / gguf_file
        if _path_exists(candidate):
            _verify_known_gguf(candidate, gguf_file, verify_checksum)
            return candidate
        ggufs = sorted(local.glob("*.gguf"))
        if len(ggufs) == 1:
            _verify_known_gguf(ggufs[0], ggufs[0].name, verify_checksum)
            return ggufs[0]
        raise RuntimeError(f"No GGUF file found in {local}; pass --gguf-file explicitly.")

    from huggingface_hub import hf_hub_download

    def _fetch(rev: str) -> Path:
        try:
            return Path(
                hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=gguf_file,
                    repo_type="model",
                    revision=rev,
                    token=hf_token(),
                )
            )
        except Exception:
            return _download_gguf_direct(
                model_id_or_path,
                gguf_file,
                rev,
                verify_checksum,
            )

    try:
        path = _fetch(revision)
    except Exception:
        if revision == "main":
            raise
        # The pinned commit can be orphaned if the upstream HF repo's
        # history gets rewritten (as happened for the default model repo);
        # fall back to the live branch rather than hard-failing.
        path = _fetch("main")
    _verify_known_gguf(path, gguf_file, verify_checksum)
    return path


def _extract_transcript(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    # parakeet-cli currently prints only the transcript for non-JSON transcribe.
    return lines[-1]


def _transcribe_cpp_batch(
    cli: str,
    model_path: Path,
    audio_paths: list[str | Path],
    decoder: str,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="omi_med_stt_cpp_") as tmp:
        manifest = Path(tmp) / "manifest.txt"
        manifest.write_text(
            "".join(f"{Path(path)}\n" for path in audio_paths),
            encoding="utf-8",
        )
        cmd = [
            cli,
            "bench",
            "--model",
            str(model_path),
            "--manifest",
            str(manifest),
            "--decoder",
            decoder,
        ]
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = json.loads(proc.stdout)
    files = payload.get("files", [])
    if len(files) != len(audio_paths):
        raise RuntimeError(
            f"parakeet.cpp returned {len(files)} transcripts for {len(audio_paths)} inputs"
        )
    texts = [str(item.get("text", "")).strip() for item in files]
    if any(not text for text in texts):
        raise RuntimeError("parakeet.cpp produced an empty transcript in batch mode")
    return texts


def _decoder_id(decoder: str) -> int:
    if decoder == "ctc":
        return 1
    if decoder == "tdt":
        return 2
    return 0


class _ParakeetCAPI:
    def __init__(
        self,
        lib_path: Path,
        model_path: Path,
        decoder: str,
        backend: str = "auto",
        threads: int = 0,
    ) -> None:
        self._dll_directory_handles: list[Any] = []
        self._dependency_handles: list[Any] = []
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            self._dll_directory_handles = [
                os.add_dll_directory(str(path)) for path in _native_library_directories(lib_path, backend)
            ]
        else:
            self._dependency_handles = _preload_unix_shared_dependencies(lib_path, backend)
        self._lib = ctypes.CDLL(str(lib_path))
        self._decoder = _decoder_id(decoder)

        self._lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        self._lib.parakeet_capi_load.restype = ctypes.c_void_p
        self._lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        self._lib.parakeet_capi_transcribe_path.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.parakeet_capi_transcribe_path.restype = ctypes.c_void_p
        self._lib.parakeet_capi_transcribe_pcm.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.parakeet_capi_transcribe_pcm.restype = ctypes.c_void_p
        self._lib.parakeet_capi_transcribe_pcm_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.parakeet_capi_transcribe_pcm_batch.restype = ctypes.c_int
        self._lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        self._lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        self._lib.parakeet_capi_last_error.restype = ctypes.c_char_p

        self._threads = threads
        if threads > 0:
            setter = getattr(self._lib, "parakeet_capi_set_num_threads", None)
            if setter is not None:
                setter.argtypes = [ctypes.c_int]
                setter.restype = None
                setter(threads)
            elif os.environ.get("OMI_MED_STT_CPP_REQUIRE_THREAD_CONTROL", "").upper() in {"1", "TRUE", "ON", "YES"}:
                raise RuntimeError(
                    "Installed parakeet.cpp C API does not expose thread control. "
                    "Run `omi-med-stt install-cpp --force` to rebuild the patched runtime."
                )

        self._ctx = self._lib.parakeet_capi_load(str(model_path).encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"Failed to load parakeet.cpp model: {model_path}")

    def close(self) -> None:
        if self._ctx:
            self._lib.parakeet_capi_free(self._ctx)
            self._ctx = None

    def __enter__(self) -> "_ParakeetCAPI":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _last_error(self) -> str:
        raw = self._lib.parakeet_capi_last_error(self._ctx)
        return (raw or b"").decode("utf-8", errors="replace")

    def _take_string(self, ptr: int) -> str:
        try:
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            self._lib.parakeet_capi_free_string(ptr)

    def transcribe_path(self, path: str | Path) -> str:
        ptr = self._lib.parakeet_capi_transcribe_path(
            self._ctx,
            str(path).encode("utf-8"),
            self._decoder,
        )
        if not ptr:
            raise RuntimeError(f"parakeet.cpp C API failed on {path}: {self._last_error()}")
        text = self._take_string(ptr).strip()
        if not text:
            raise RuntimeError(f"parakeet.cpp C API produced an empty transcript for {path}")
        return text

    def transcribe_pcm(self, pcm16k: np.ndarray) -> str:
        arr = np.ascontiguousarray(pcm16k, dtype=np.float32)
        ptr = self._lib.parakeet_capi_transcribe_pcm(
            self._ctx,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(arr),
            16000,
            self._decoder,
        )
        if not ptr:
            raise RuntimeError(f"parakeet.cpp C API failed on PCM chunk: {self._last_error()}")
        text = self._take_string(ptr).strip()
        if not text:
            raise RuntimeError("parakeet.cpp C API produced an empty transcript for PCM chunk")
        return text

    def transcribe_pcm_batch(self, pcm16k_batch: list[np.ndarray]) -> list[str]:
        arrays = [np.ascontiguousarray(pcm, dtype=np.float32) for pcm in pcm16k_batch]
        n_clips = len(arrays)
        if n_clips == 0:
            return []

        sample_ptrs = (ctypes.POINTER(ctypes.c_float) * n_clips)(
            *(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for arr in arrays)
        )
        lengths = (ctypes.c_int * n_clips)(*(len(arr) for arr in arrays))
        out = (ctypes.c_void_p * n_clips)()
        rc = self._lib.parakeet_capi_transcribe_pcm_batch(
            self._ctx,
            sample_ptrs,
            lengths,
            n_clips,
            16000,
            self._decoder,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"parakeet.cpp C API batch failed: {self._last_error()}")

        texts = []
        for ptr in out:
            if not ptr:
                raise RuntimeError("parakeet.cpp C API batch produced a null transcript")
            text = self._take_string(ptr).strip()
            if not text:
                raise RuntimeError("parakeet.cpp C API batch produced an empty transcript")
            texts.append(text)
        return texts


def _capi_enabled() -> bool:
    return os.environ.get("OMI_MED_STT_CPP_DISABLE_CAPI", "").upper() not in {"1", "TRUE", "ON", "YES"}


def _batch_chunks_enabled() -> bool:
    return os.environ.get("OMI_MED_STT_CPP_BATCH_CHUNKS", "").upper() in {"1", "TRUE", "ON", "YES"}


def _transcribe_cpp_capi(
    audio_paths: list[str | Path],
    model_path: Path,
    *,
    auto_install: bool,
    cpp_backend: str,
    decoder: str,
    threads: int = 0,
) -> list[str]:
    from .audio import linear_resample, read_audio

    lib = _find_parakeet_lib(auto_install=auto_install, backend=cpp_backend)
    with _ParakeetCAPI(lib, model_path, decoder, cpp_backend, threads) as capi:
        texts = []
        for path in audio_paths:
            audio, sample_rate = read_audio(path)
            audio16k = np.ascontiguousarray(linear_resample(audio, sample_rate, 16000), dtype=np.float32)
            texts.append(capi.transcribe_pcm(audio16k))
        return texts


def transcribe_cpp_pcm_chunks(
    audio_path: str | Path,
    model_id_or_path: str = DEFAULT_GGUF_REPO,
    *,
    chunk_seconds: float,
    overlap: float,
    gguf_file: str = DEFAULT_GGUF_FILE,
    revision: str = DEFAULT_GGUF_REVISION,
    verify_checksum: bool = True,
    auto_install: bool = True,
    cpp_backend: str = "auto",
    decoder: str = "tdt",
    threads: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if overlap >= chunk_seconds:
        raise ValueError("--overlap must be smaller than --chunk-seconds")

    from .audio import linear_resample, read_audio
    from .merge import merge_transcripts

    model_path = _download_or_resolve_gguf(
        str(model_id_or_path),
        gguf_file,
        revision,
        verify_checksum,
    )
    lib = _find_parakeet_lib(auto_install=auto_install, backend=cpp_backend)

    audio, sample_rate = read_audio(audio_path)
    audio16k = np.ascontiguousarray(linear_resample(audio, sample_rate, 16000), dtype=np.float32)
    step = int(round((chunk_seconds - overlap) * 16000))
    chunk_len = int(round(chunk_seconds * 16000))

    chunks: list[dict[str, Any]] = []
    chunk_audio_batch: list[np.ndarray] = []
    start = 0
    index = 0
    while start < len(audio16k):
        end = min(len(audio16k), start + chunk_len)
        chunk_audio = audio16k[start:end]
        if len(chunk_audio) == 0:
            break
        chunk_audio_batch.append(np.ascontiguousarray(chunk_audio, dtype=np.float32))
        chunks.append({
            "index": index,
            "start": start / 16000.0,
            "duration": len(chunk_audio) / 16000.0,
        })
        if end >= len(audio16k):
            break
        start += step
        index += 1

    resolved_threads = _default_cpp_threads(cpp_backend) if threads is None else threads
    with _ParakeetCAPI(lib, model_path, decoder, cpp_backend, resolved_threads) as capi:
        if _batch_chunks_enabled():
            try:
                texts = capi.transcribe_pcm_batch(chunk_audio_batch)
            except Exception:
                if os.environ.get("OMI_MED_STT_CPP_REQUIRE_BATCH", "").upper() in {"1", "TRUE", "ON", "YES"}:
                    raise
                texts = [capi.transcribe_pcm(chunk_audio) for chunk_audio in chunk_audio_batch]
        else:
            texts = [capi.transcribe_pcm(chunk_audio) for chunk_audio in chunk_audio_batch]

    for chunk, text in zip(chunks, texts):
        chunk["transcript"] = text
    return merge_transcripts(texts), chunks


def transcribe_cpp(
    audio_paths: list[str | Path],
    model_id_or_path: str = DEFAULT_GGUF_REPO,
    *,
    gguf_file: str = DEFAULT_GGUF_FILE,
    revision: str = DEFAULT_GGUF_REVISION,
    verify_checksum: bool = True,
    parakeet_cli: str | None = None,
    auto_install: bool = True,
    cpp_backend: str = "auto",
    decoder: str = "tdt",
    threads: int | None = None,
) -> list[str]:
    model_path = _download_or_resolve_gguf(
        str(model_id_or_path),
        gguf_file,
        revision,
        verify_checksum,
    )
    resolved_threads = _default_cpp_threads(cpp_backend) if threads is None else threads
    capi_error: Exception | None = None
    if _capi_enabled() and parakeet_cli is None:
        try:
            return _transcribe_cpp_capi(
                audio_paths,
                model_path,
                auto_install=auto_install,
                cpp_backend=cpp_backend,
                decoder=decoder,
                threads=resolved_threads,
            )
        except Exception as exc:
            if os.environ.get("OMI_MED_STT_CPP_REQUIRE_CAPI", "").upper() in {"1", "TRUE", "ON", "YES"}:
                raise
            capi_error = exc
            print(
                f"omi-med-stt: C API load failed ({type(exc).__name__}: {str(exc)[:200]}); "
                "falling back to parakeet-cli subprocess.",
                file=sys.stderr,
            )

    try:
        return _transcribe_cpp_subprocess(
            audio_paths,
            model_path,
            parakeet_cli=parakeet_cli,
            auto_install=auto_install,
            cpp_backend=cpp_backend,
            decoder=decoder,
            resolved_threads=resolved_threads,
        )
    except Exception as exc:
        if capi_error is not None:
            raise RuntimeError(
                f"parakeet-cli subprocess fallback failed ({type(exc).__name__}: {exc}); "
                f"original C API load error ({type(capi_error).__name__}: {capi_error})"
            ) from exc
        raise


def _transcribe_cpp_subprocess(
    audio_paths: list[str | Path],
    model_path: Path,
    *,
    parakeet_cli: str | None,
    auto_install: bool,
    cpp_backend: str,
    decoder: str,
    resolved_threads: int,
) -> list[str]:
    cli = _find_parakeet_cli(parakeet_cli, auto_install=auto_install, backend=cpp_backend)
    if len(audio_paths) > 1 and os.environ.get("OMI_MED_STT_CPP_USE_BENCH", "").upper() in {"1", "TRUE", "ON", "YES"}:
        return _transcribe_cpp_batch(cli, model_path, audio_paths, decoder)

    texts: list[str] = []
    for path in audio_paths:
        cmd = [
            cli,
            "transcribe",
            "--model",
            str(model_path),
            "--input",
            str(path),
            "--decoder",
            decoder,
        ]
        if resolved_threads > 0:
            cmd.extend(["--threads", str(resolved_threads)])
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        text = _extract_transcript(proc.stdout)
        if not text:
            raise RuntimeError(f"parakeet.cpp produced an empty transcript for {path}")
        texts.append(text)
    return texts
