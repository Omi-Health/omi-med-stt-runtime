from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_TARGET_SR = 16000
# soundfile>=0.12 raises LibsndfileError; fall back to RuntimeError on older builds.
_SOUNDFILE_ERROR = getattr(sf, "LibsndfileError", RuntimeError)


class AudioDecodeError(RuntimeError):
    """An audio file could not be decoded (unsupported/corrupt, or ffmpeg unavailable)."""


@dataclass
class AudioInfo:
    path: Path
    sample_rate: int
    duration: float


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary: system PATH first, then bundled imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_with_ffmpeg(path: str | Path, target_sr: int = _TARGET_SR) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable input (m4a, mp3, aac, mp4, mov, wma, opus, webm, …)
    to mono float32 at ``target_sr``.

    Mirrors the mlx-whisper / parakeet-mlx approach: pipe signed 16-bit PCM out of
    ffmpeg's stdout — no temp files and no heavy audio dependency. ffmpeg performs the
    downmix-to-mono and resample in one pass.
    """
    exe = find_ffmpeg()
    if exe is None:
        raise AudioDecodeError(
            f"Cannot decode {Path(path).name!r}: this audio format needs FFmpeg, "
            "which was not found.\nThis should ship with omi-med-stt by default; "
            "if your install is broken, reinstall with:\n"
            "  pip install -U --force-reinstall omi-med-stt\n"
            "or install a system FFmpeg with one of:\n"
            "  brew install ffmpeg                 (macOS)\n"
            "  sudo apt install ffmpeg             (Debian/Ubuntu)\n"
            "  conda install -c conda-forge ffmpeg"
        )
    cmd = [
        exe, "-nostdin", "-threads", "0", "-i", str(path),
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as exc:  # ffmpeg path vanished between probe and run
        raise AudioDecodeError("FFmpeg was found but could not be executed.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(detail[-3:]) if detail else ""
        raise AudioDecodeError(
            f"FFmpeg could not decode {Path(path).name!r} — the file may be corrupt "
            f"or not an audio file.\n{tail}"
        ) from exc
    if not out:
        raise AudioDecodeError(f"No audio decoded from {Path(path).name!r} (empty stream).")
    audio = np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0
    return audio, target_sr


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file to ``(mono float32, sample_rate)``.

    Fast path: libsndfile via soundfile (wav/flac/ogg/aiff/…), returned at the file's
    native sample rate. For anything libsndfile can't open (m4a/aac/mp3/mp4/mov/…),
    fall back to ffmpeg, which returns audio already resampled to 16 kHz mono.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except _SOUNDFILE_ERROR:
        return _decode_with_ffmpeg(path)
    mono = audio.mean(axis=1)
    return mono, sr


def linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype("float32", copy=False)
    if len(audio) == 0:
        return audio.astype("float32", copy=False)
    duration = len(audio) / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype("float32")


def normalize_to_16k_mono(path: str | Path, tmp_dir: str | Path | None = None) -> AudioInfo:
    audio, sr = read_audio(path)
    audio = linear_resample(audio, sr, 16000)
    out_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="omi_stt_audio_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(path).stem}.16k.wav"
    sf.write(str(out_path), audio, 16000)
    return AudioInfo(path=out_path, sample_rate=16000, duration=len(audio) / 16000.0)


def make_chunks(path: str | Path, chunk_seconds: float, overlap: float, tmp_dir: str | Path | None = None) -> list[AudioInfo]:
    if overlap >= chunk_seconds:
        raise ValueError("--overlap must be smaller than --chunk-seconds")
    info = normalize_to_16k_mono(path, tmp_dir)
    audio, _ = read_audio(info.path)
    out_dir = info.path.parent
    step = int(round((chunk_seconds - overlap) * 16000))
    chunk_len = int(round(chunk_seconds * 16000))
    chunks: list[AudioInfo] = []
    start = 0
    idx = 0
    while start < len(audio):
        end = min(len(audio), start + chunk_len)
        chunk_audio = audio[start:end]
        if len(chunk_audio) == 0:
            break
        out = out_dir / f"{Path(path).stem}.chunk{idx:04d}.wav"
        sf.write(str(out), chunk_audio, 16000)
        chunks.append(AudioInfo(path=out, sample_rate=16000, duration=len(chunk_audio) / 16000.0))
        if end >= len(audio):
            break
        start += step
        idx += 1
    return chunks
