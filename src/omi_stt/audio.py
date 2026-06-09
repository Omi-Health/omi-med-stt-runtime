from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioInfo:
    path: Path
    sample_rate: int
    duration: float


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
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
