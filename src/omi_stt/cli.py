from __future__ import annotations

import argparse
import contextlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .audio import (
    AudioDecodeError,
    find_ffmpeg,
    make_chunks,
    normalize_to_16k_mono,
    normalize_with_ffmpeg_to_16k_pcm16,
)
from .merge import merge_transcripts
from .cpp_runtime import DEFAULT_GGUF_REVISION, install_parakeet_cpp


DEFAULT_NEMO_MODEL = "omi-health/omi-med-stt-v1"
DEFAULT_MLX_MODEL = "omi-health/omi-med-stt-v1-mlx-q8"
DEFAULT_CPP_MODEL = "omi-health/omi-med-stt-v1-gguf"
DEFAULT_GGUF_FILE = "omi-med-stt-v1-q8_0.gguf"
COMMANDS = {"transcribe", "transcribe-long", "install-cpp", "check", "-h", "--help"}
VERSION_FLAGS = {"--version", "-V"}


def _package_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("omi-med-stt")
    except Exception:
        return "unknown"


def _version_string() -> str:
    return f"omi-med-stt {_package_version()} (default runtime: {_default_runtime()})"


def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        proc = subprocess.run(
            [nvidia_smi, "-L"],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


def _default_runtime() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx"
    if _has_nvidia_gpu():
        return "nemo"
    return "cpp"



def _resolve_runtime_and_model(args: argparse.Namespace) -> tuple[str, str]:
    runtime = _default_runtime() if args.runtime == "auto" else args.runtime
    if args.model:
        model = args.model
    elif runtime == "mlx":
        model = DEFAULT_MLX_MODEL
    elif runtime == "cpp":
        model = DEFAULT_CPP_MODEL
    else:
        model = DEFAULT_NEMO_MODEL
    return runtime, model



def _runtime_transcribe(paths: list[Path], runtime: str, model: str, args: argparse.Namespace) -> list[str]:
    if runtime == "mlx":
        from .mlx_runtime import transcribe_mlx

        return transcribe_mlx(paths, model)
    if runtime == "cpp":
        from .cpp_runtime import transcribe_cpp

        return transcribe_cpp(
            paths,
            model,
            gguf_file=args.gguf_file,
            revision=args.revision,
            verify_checksum=not args.no_verify_checksum,
            parakeet_cli=args.parakeet_cli,
            auto_install=not args.no_auto_install,
            cpp_backend=args.cpp_backend,
            decoder=args.decoder,
            threads=args.cpp_threads,
        )
    from .nemo_runtime import transcribe_nemo

    return transcribe_nemo(paths, model, device=args.device)


def _quiet_runtime_transcribe(paths: list[Path], runtime: str, model: str, args: argparse.Namespace) -> list[str]:
    with contextlib.redirect_stdout(sys.stderr):
        return _runtime_transcribe(paths, runtime, model, args)



def _is_hf_access_error(exc: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except Exception:
        return False
    if isinstance(exc, (RepositoryNotFoundError, GatedRepoError)):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 401


def _is_driver_too_old_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return "driver" in text and "too old" in text


def _handle_runtime_error(exc: Exception, model: str) -> int:
    if isinstance(exc, AudioDecodeError):
        print(str(exc), file=sys.stderr)
        return 2
    if _is_hf_access_error(exc):
        print(
            f"Cannot access {model}: the repository is private/gated or your token is "
            "missing. Run `huggingface-cli login` or set HF_TOKEN, then retry.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2
    if _is_driver_too_old_error(exc):
        print(
            "Your NVIDIA driver is too old for the installed torch build. Install a "
            "matching torch first, e.g.: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128 — then reinstall: "
            "pip install -U 'omi-med-stt[nemo]'.",
            file=sys.stderr,
        )
        return 2
    if isinstance(exc, RuntimeError) and "requires" in str(exc):
        print(str(exc), file=sys.stderr)
        return 2
    raise exc


def _effective_max_seconds(runtime: str, requested: float) -> float:
    if runtime == "cpp" and platform.system() == "Windows" and requested == 240.0:
        return 120.0
    return requested


def _effective_chunk_seconds(runtime: str, requested: float, duration: float) -> float:
    if runtime != "cpp" or requested != 240.0:
        return requested
    if platform.system() == "Windows":
        return 120.0
    if duration > 240.0:
        return 180.0
    return requested


def cmd_transcribe(args: argparse.Namespace) -> int:
    runtime, model = _resolve_runtime_and_model(args)
    try:
        return _cmd_transcribe(args, runtime, model)
    except Exception as exc:
        return _handle_runtime_error(exc, model)


def _cmd_transcribe(args: argparse.Namespace, runtime: str, model: str) -> int:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="omi_stt_") as tmp:
        normalize = (
            normalize_with_ffmpeg_to_16k_pcm16 if runtime == "nemo" else normalize_to_16k_mono
        )
        info = normalize(args.audio, tmp)
        max_seconds = _effective_max_seconds(runtime, args.max_seconds)
        chunk_seconds = _effective_chunk_seconds(runtime, args.max_seconds, info.duration)
        # The selected NeMo recipe uses bounded local attention and accepts the
        # complete recording. Legacy automatic chunk/merge changes hypotheses and
        # was not part of the qualified GPU result, so only the C++ safety path
        # auto-chunks here. `transcribe-long` remains available when explicitly
        # requested by a caller.
        auto_chunked = runtime == "cpp" and info.duration > max_seconds
        if auto_chunked and runtime == "cpp":
            from .cpp_runtime import transcribe_cpp_pcm_chunks

            transcript, chunks = transcribe_cpp_pcm_chunks(
                args.audio,
                model,
                chunk_seconds=chunk_seconds,
                overlap=args.overlap,
                gguf_file=args.gguf_file,
                revision=args.revision,
                verify_checksum=not args.no_verify_checksum,
                auto_install=not args.no_auto_install,
                cpp_backend=args.cpp_backend,
                decoder=args.decoder,
                threads=args.cpp_threads,
            )
        elif auto_chunked:
            chunks = make_chunks(args.audio, args.max_seconds, args.overlap, tmp)
            texts = _quiet_runtime_transcribe([c.path for c in chunks], runtime, model, args)
            transcript = merge_transcripts(texts)
            chunks = [
                {"index": i, "path": str(c.path), "duration": c.duration, "transcript": t}
                for i, (c, t) in enumerate(zip(chunks, texts))
            ]
        else:
            chunks = []
            texts = _quiet_runtime_transcribe([info.path], runtime, model, args)
            transcript = texts[0]
        result = {
            "model": model,
            "runtime": runtime,
            "gguf_file": args.gguf_file if runtime == "cpp" else None,
            "revision": args.revision if runtime == "cpp" else None,
            "checksum_verified": (not args.no_verify_checksum) if runtime == "cpp" else None,
            "max_seconds": max_seconds,
            "chunk_seconds": chunk_seconds if auto_chunked else None,
            "audio": str(args.audio),
            "duration": info.duration,
            "auto_chunked": auto_chunked,
            "chunks": chunks if auto_chunked else None,
            "wall_time": time.time() - started,
            "transcript": transcript,
        }
    print(json.dumps(result, indent=2) if args.json else result["transcript"])
    return 0


def cmd_transcribe_long(args: argparse.Namespace) -> int:
    runtime, model = _resolve_runtime_and_model(args)
    try:
        return _cmd_transcribe_long(args, runtime, model)
    except Exception as exc:
        return _handle_runtime_error(exc, model)


def _cmd_transcribe_long(args: argparse.Namespace, runtime: str, model: str) -> int:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="omi_stt_long_") as tmp:
        if runtime == "cpp":
            from .cpp_runtime import transcribe_cpp_pcm_chunks

            transcript, chunks = transcribe_cpp_pcm_chunks(
                args.audio,
                model,
                chunk_seconds=args.chunk_seconds,
                overlap=args.overlap,
                gguf_file=args.gguf_file,
                revision=args.revision,
                verify_checksum=not args.no_verify_checksum,
                auto_install=not args.no_auto_install,
                cpp_backend=args.cpp_backend,
                decoder=args.decoder,
                threads=args.cpp_threads,
            )
        else:
            chunk_infos = make_chunks(args.audio, args.chunk_seconds, args.overlap, tmp)
            texts = _quiet_runtime_transcribe([c.path for c in chunk_infos], runtime, model, args)
            transcript = merge_transcripts(texts)
            chunks = [
                {"index": i, "path": str(c.path), "duration": c.duration, "transcript": t}
                for i, (c, t) in enumerate(zip(chunk_infos, texts))
            ]
        result = {
            "model": model,
            "runtime": runtime,
            "gguf_file": args.gguf_file if runtime == "cpp" else None,
            "revision": args.revision if runtime == "cpp" else None,
            "checksum_verified": (not args.no_verify_checksum) if runtime == "cpp" else None,
            "audio": str(args.audio),
            "chunks": chunks,
            "wall_time": time.time() - started,
            "transcript": transcript,
        }
    print(json.dumps(result, indent=2) if args.json else result["transcript"])
    return 0



def cmd_doctor(_: argparse.Namespace) -> int:
    ffmpeg_path = find_ffmpeg()
    checks = {
        "omi_med_stt_version": __version__,
        "python": sys.version.split()[0],
        "ffmpeg": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path,
        "audio_input": "wav/flac/ogg native; m4a/mp3/aac/mp4/mov/… via ffmpeg",
        "nvidia_gpu": _has_nvidia_gpu(),
        "parakeet_cli": shutil.which("parakeet-cli"),
        "default_runtime": _default_runtime(),
    }
    try:
        import parakeet_mlx  # noqa: F401

        checks["parakeet_mlx_python"] = True
    except Exception:
        checks["parakeet_mlx_python"] = False
    try:
        import nemo  # noqa: F401

        checks["nemo"] = True
    except Exception:
        checks["nemo"] = False
    try:
        from .cpp_runtime import DEFAULT_GGUF_FILE as cpp_default_gguf, _cached_parakeet_lib, _default_cpp_threads

        checks["parakeet_cpp_default_gguf"] = cpp_default_gguf
        checks["libparakeet"] = str(_cached_parakeet_lib())
        checks["libparakeet_exists"] = _cached_parakeet_lib().exists()
        checks["parakeet_cpp_default_threads"] = _default_cpp_threads()
    except Exception:
        checks["parakeet_cpp_default_gguf"] = None
        checks["libparakeet"] = None
        checks["libparakeet_exists"] = False
        checks["parakeet_cpp_default_threads"] = None
    try:
        import soundfile  # noqa: F401

        checks["soundfile"] = True
    except Exception:
        checks["soundfile"] = False
    print(json.dumps(checks, indent=2))
    return 0


def cmd_install_cpp(args: argparse.Namespace) -> int:
    started = time.time()
    cli = install_parakeet_cpp(force=args.force, backend=args.cpp_backend)
    from .cpp_runtime import _cached_parakeet_lib, _default_cpp_threads

    result = {
        "runtime": "cpp",
        "parakeet_cli": str(cli),
        "libparakeet": str(_cached_parakeet_lib(args.cpp_backend)),
        "backend": args.cpp_backend,
        "default_threads": _default_cpp_threads(args.cpp_backend),
        "wall_time": time.time() - started,
    }
    print(json.dumps(result, indent=2) if args.json else str(cli))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omi-med-stt",
        description="Transcribe audio with Omi Med STT v1. Use `omi-med-stt audio.wav` for the default path.",
    )
    parser.add_argument("--version", "-V", action="version", version=_version_string())
    sub = parser.add_subparsers(required=True)

    one = sub.add_parser("transcribe")
    one.add_argument("audio", type=Path)
    one.add_argument("--model")
    one.add_argument("--runtime", choices=["auto", "nemo", "mlx", "cpp"], default="auto")
    one.add_argument("--device", default="cuda")
    one.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    one.add_argument("--revision", default=DEFAULT_GGUF_REVISION)
    one.add_argument("--no-verify-checksum", action="store_true")
    one.add_argument("--parakeet-cli")
    one.add_argument("--no-auto-install", action="store_true")
    one.add_argument("--cpp-backend", choices=["auto", "cpu"], default="auto")
    one.add_argument("--cpp-threads", type=int, default=None, help="CPU threads for parakeet.cpp; default auto-tunes Windows CPU.")
    one.add_argument("--decoder", choices=["tdt", "ctc"], default="tdt")
    one.add_argument("--max-seconds", type=float, default=240.0)
    one.add_argument("--overlap", type=float, default=5.0)
    one.add_argument("--json", action="store_true")
    one.set_defaults(func=cmd_transcribe)

    long = sub.add_parser("transcribe-long")
    long.add_argument("audio", type=Path)
    long.add_argument("--model")
    long.add_argument("--runtime", choices=["auto", "nemo", "mlx", "cpp"], default="auto")
    long.add_argument("--device", default="cuda")
    long.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    long.add_argument("--revision", default=DEFAULT_GGUF_REVISION)
    long.add_argument("--no-verify-checksum", action="store_true")
    long.add_argument("--parakeet-cli")
    long.add_argument("--no-auto-install", action="store_true")
    long.add_argument("--cpp-backend", choices=["auto", "cpu"], default="auto")
    long.add_argument("--cpp-threads", type=int, default=None, help="CPU threads for parakeet.cpp; default auto-tunes Windows CPU.")
    long.add_argument("--decoder", choices=["tdt", "ctc"], default="tdt")
    long.add_argument("--chunk-seconds", type=float, default=25.0)
    long.add_argument("--overlap", type=float, default=3.0)
    long.add_argument("--json", action="store_true")
    long.set_defaults(func=cmd_transcribe_long)

    install_cpp = sub.add_parser("install-cpp", help="install or refresh the bundled parakeet.cpp runtime")
    install_cpp.add_argument("--force", action="store_true")
    install_cpp.add_argument("--cpp-backend", choices=["auto", "cpu"], default="auto")
    install_cpp.add_argument("--json", action="store_true")
    install_cpp.set_defaults(func=cmd_install_cpp)

    check = sub.add_parser("check", help="show local runtime/dependency diagnostics")
    check.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in VERSION_FLAGS:
        print(_version_string())
        raise SystemExit(0)
    if args_list and args_list[0] not in COMMANDS:
        args_list.insert(0, "transcribe")
    parser = build_parser()
    args = parser.parse_args(args_list)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
