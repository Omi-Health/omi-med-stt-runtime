#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> dict[str, Any]:
    started = time.time()
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + merged_env.get("PYTHONPATH", "")
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
    )
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_time": time.time() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if check and proc.returncode != 0:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(proc.returncode)
    return result


def has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    proc = subprocess.run([nvidia_smi, "-L"], text=True, capture_output=True)
    return proc.returncode == 0 and "GPU" in proc.stdout


def runtime_matrix() -> list[dict[str, Any]]:
    system = platform.system()
    machine = platform.machine().lower()
    return [
        {
            "platform": "macOS",
            "accelerator": "Apple GPU / MLX",
            "runtime": "mlx",
            "available_here": system == "Darwin" and machine in {"arm64", "aarch64"},
            "command": "omi-med-stt sample.wav --runtime mlx",
        },
        {
            "platform": "Linux/Windows",
            "accelerator": "CPU",
            "runtime": "cpp",
            "available_here": True,
            "command": "omi-med-stt sample.wav --runtime cpp --cpp-backend cpu",
        },
        {
            "platform": "Linux",
            "accelerator": "NVIDIA CUDA GPU / NeMo",
            "runtime": "nemo",
            "available_here": system == "Linux" and has_nvidia_gpu(),
            "command": "omi-med-stt sample.wav --runtime nemo",
        },
        {
            "platform": "Windows",
            "accelerator": "CPU",
            "runtime": "cpp",
            "available_here": system == "Windows",
            "command": "omi-med-stt sample.wav --runtime cpp --cpp-backend cpu",
        },
        {
            "platform": "Windows",
            "accelerator": "NPU",
            "runtime": None,
            "available_here": False,
            "status": "unsupported",
            "note": "No Omi Med STT NPU backend exists yet. Use cpp CPU on Windows.",
        },
    ]


def parse_runtime(item: str) -> tuple[str, str | None]:
    if ":" in item:
        runtime, backend = item.split(":", 1)
        return runtime, backend
    return item, None


def smoke_command(audio: Path, runtime: str, backend: str | None, *, long: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "omi_stt.cli",
        "transcribe",
        str(audio),
        "--runtime",
        runtime,
        "--json",
    ]
    if runtime == "cpp" and backend:
        cmd.extend(["--cpp-backend", backend])
    if long:
        cmd.extend(["--max-seconds", "180", "--overlap", "5"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local pre-publish checks for omi-med-stt.")
    parser.add_argument("--audio", type=Path, default=os.environ.get("OMI_MED_STT_SMOKE_AUDIO"))
    parser.add_argument("--runtime", action="append", default=[], help="Runtime smoke target, e.g. cpp:cpu, mlx, nemo.")
    parser.add_argument("--long", action="store_true", help="Force long-audio chunking parameters during smoke.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("local_smoke/prepublish_last.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "matrix": runtime_matrix(),
        "checks": [],
    }

    for name, cmd in [
        ("compileall", [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("runtime_check", [sys.executable, "-m", "omi_stt.cli", "check"]),
    ]:
        result = run(cmd)
        report["checks"].append({"name": name, "returncode": result["returncode"], "wall_time": result["wall_time"]})

    if not args.skip_build:
        result = run([sys.executable, "-m", "build"])
        report["checks"].append({"name": "build", "returncode": result["returncode"], "wall_time": result["wall_time"]})

    smoke_results = []
    if args.audio:
        audio = Path(args.audio).expanduser()
        if not audio.exists():
            raise SystemExit(f"Smoke audio not found: {audio}")
        targets = args.runtime or ["cpp:cpu"]
        for target in targets:
            runtime, backend = parse_runtime(target)
            result = run(smoke_command(audio, runtime, backend, long=args.long), check=False)
            parsed = None
            if result["stdout"].strip().startswith("{"):
                try:
                    parsed = json.loads(result["stdout"])
                except json.JSONDecodeError:
                    parsed = None
            smoke_results.append({
                "target": target,
                "returncode": result["returncode"],
                "wall_time": result["wall_time"],
                "parsed": {
                    "runtime": parsed.get("runtime") if parsed else None,
                    "duration": parsed.get("duration") if parsed else None,
                    "auto_chunked": parsed.get("auto_chunked") if parsed else None,
                    "chunks": len(parsed.get("chunks") or []) if parsed else None,
                    "transcript_chars": len(parsed.get("transcript") or "") if parsed else None,
                },
                "stderr_tail": result["stderr"][-1000:],
            })
            if result["returncode"] != 0:
                raise SystemExit(result["returncode"])
    report["smoke"] = smoke_results

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
